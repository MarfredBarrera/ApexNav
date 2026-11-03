"""VerificationGate: the Python half of the perception gate (no ROS imports).

Per episode:
    gate.start_episode(run_index, scene, episode_id, label, goal_object_ids)
    reopen = gate.observe(step, rgb, masks, labels, scores, centroids, semantic_instances,
                          pose, camera_pitch, fusion)        # after every sim step
    state = gate.handle(request, fusion, step)               # on /object/verification_request
    approach = gate.pending_approach(cluster_id)             # when state == 3 (approach)
    counts = gate.finish_episode()

``observe`` stores every target-class (label 0) detection as evidence, per
0.75 m cell of its world centroid (``evidence.EvidenceStore``: the most readable
views of each place plus its newest). ``handle`` verifies the best view of the
requested cluster with ``perception.Verifier`` over HTTP and returns the state
for the planner: 1 accepted, -1 rejected, 2 deferred. A request with ``cluster_id`` -1 is
the stop backstop asking about a position that no cluster covers (e.g. an
over-depth target); it is verified from the evidence nearest that position.

Verdicts are cached by evidence view (step, index): a cluster whose selected
view was already judged (a re-spawn of a judged cluster) reuses that verdict
without a server call. A distinct nearby object is judged on its own views: a
view closer to another cluster (as clusters stood at that step) is not
evidence for this one, and ties prefer the nearest view. ``at_stop`` requests
match evidence to the stop position the planner sends.

Deferral (``on_view_reject: defer``): when the raw frame accepts but the SAM3D view
says no, the cluster is deferred instead of rejected. The planner may approach it but
not stop at it: the stop backstop verifies it again there, preferring a new view. At
the stop a view "no" rejects (there is no closer view to wait for), and so does a
deferred verdict reused from the cache.

Approach (``approach``, server selector ``d1``): the server answers three-way
(``verdict.status``): correct -> 1 accepted, wrong -> -1 rejected, insufficient -> 3
approach: the planner drives to a viewpoint nearer the object point (the evidence
centroid) along the line of sight object -> camera of the judged view (jitter 0, +-20,
+-40 deg; only the side bearings when that camera was already within 1.5 m), faces it
and re-requests the cluster with ``approach_id`` and ``viewpoint_status`` 1 (arrived),
2 (no reachable viewpoint), 3 (aborted) or 4 (early). Early: while an approach is
pending, ``observe`` returns the cluster's id (with the reopens) on the first detection
of the object that is ``approach_min_closer_m`` closer than, or
``approach_min_angle_deg`` away from, every view already judged for the cluster; the
planner ends the approach and asks about that view. On arrival the newest such view is
judged. Approach re-requests bypass the re-check cooldown, ``max_checks`` and the
verdict cache. Budgets: ``approach_max_per_cluster`` approaches per cluster,
``approach_max_steps`` actions each, ``approach_episode_steps`` per episode. Failure
(unreachable, aborted, not re-detected, no budget, insufficient after the last
approach) rejects; a rejected cluster then follows the re-check rule below. At the stop
(``at_stop``) insufficient never accepts: one side-view approach (+-40 deg) if the
budget allows, else reject. Position requests (cluster -1) never approach.

Failure policy (fail open by default): a server error or timeout applies
``on_error``, a cluster without a stored view ``on_no_evidence``; both are
logged. ``max_consecutive_errors`` server errors in a row raise, which aborts
the run, so a run never goes on silently ungated.

Re-check: a rejected cluster is reopened (``observe`` returns its id; the caller
publishes state 0, so the planner's next selection of it requests a new check)
once all hold: ``recheck_cooldown_steps`` steps since its last check, a new view
of it this step whose camera is ``recheck_min_distance_m`` from every view used
before or looks at it from a direction ``recheck_min_angle_deg`` apart, and
fewer than ``max_checks`` checks so far.

Logs: ``<log_dir>/decisions.jsonl`` (one line per decision, with the server
settings, camera intrinsics and the semantic-instance ground truth) and, with
``save_images``, ``<log_dir>/epi<run>_<label>/c<id>_s<step>/{input.jpg, mask.png,
view.png, decision.json}``.
"""
import json
import math
import os
import time

import cv2
import numpy as np

from basic_utils.eval.habitat_utils import semantic_goal_overlap, slugify_label
from perception_gate import client
from perception_gate.evidence import MATCH_DISTANCE_M, EvidenceStore, mask_touches_border, view_quality

DEFAULTS = {
    "enabled": False,
    "url": "http://3dawareverification-dev-1:8770",
    "timeout_s": 120.0,
    "ping_timeout_s": 60.0,          # long enough to wait out one in-flight verification
    "on_error": "accept",
    "on_no_evidence": "accept",
    "on_view_reject": "reject",      # raw frame accepts, SAM3D view says no: reject or defer
    "max_consecutive_errors": 3,
    "max_checks": 3,
    "recheck_cooldown_steps": 25,
    "recheck_min_distance_m": 1.0,
    "recheck_min_angle_deg": 45.0,
    "match_distance_m": MATCH_DISTANCE_M,
    "log_dir": "{video_output_path}/verification",
    "save_images": True,
    "evidence_per_cell": 8,
    "max_evidence": 256,
    "ground_truth_semantic_overlap": 0.5,
    "approach": False,               # d1: insufficient -> approach (state 3); needs selector d1 and the planner's verification_approach
    "approach_max_per_cluster": 2,
    "approach_max_steps": 40,        # actions per approach
    "approach_episode_steps": 120,   # approach actions per episode
    "approach_min_closer_m": 0.5,    # a new view: this much closer than every judged view of the cluster ...
    "approach_min_angle_deg": 30.0,  # ... or this far around the object from each
}
POLICIES = ("accept", "reject")
ACCEPTED, UNVERIFIED, REJECTED, DEFERRED, APPROACH = 1, 0, -1, 2, 3
NAMES = {ACCEPTED: "ACCEPTED", REJECTED: "REJECTED", DEFERRED: "DEFERRED", APPROACH: "APPROACH"}
# VerificationRequest.viewpoint_status of an approach re-request
ARRIVED, UNREACHABLE, ABORTED, EARLY = 1, 2, 3, 4
STATUS_NAMES = {ARRIVED: "arrived", UNREACHABLE: "unreachable", ABORTED: "aborted", EARLY: "early"}
TRI_STATES = ("correct", "wrong", "insufficient")
APPROACH_COUNTS = ("insufficient", "moves", "unreachable", "aborted", "not_redetected", "fallback",
                   "approach_arrived", "approach_early", "approach_accepted", "approach_rejected",
                   "approach_insufficient", "approach_exhausted", "approach_steps", "sam3_decisions",
                   "qwen_decisions")


def target_name(label):
    """The Verifier category for an ApexNav label ("potted_plant", "table | desk")."""
    return str(label).split("|")[0].strip().replace("_", " ")


def _distance(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _bearing(camera, point):
    return math.atan2(point[1] - camera[1], point[0] - camera[0])


def _angle_between(a, b):
    return abs((a - b + math.pi) % (2 * math.pi) - math.pi)


class VerificationGate:
    """``camera``: intrinsics logged with every decision, e.g. {hfov, width, height}."""

    def __init__(self, cfg=None, video_output_path=".", verify=None, ping=None, camera=None):
        unknown = set(cfg or {}) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"unknown verification settings: {sorted(unknown)}")
        self.cfg = {**DEFAULTS, **{k: v for k, v in dict(cfg or {}).items() if v is not None}}
        for key in ("on_error", "on_no_evidence"):
            if self.cfg[key] not in POLICIES:
                raise ValueError(f"verification.{key} must be one of {POLICIES}")
        if self.cfg["on_view_reject"] not in ("reject", "defer"):
            raise ValueError("verification.on_view_reject must be reject or defer")
        self.approach = bool(self.cfg["approach"])
        for key in ("approach_max_per_cluster", "approach_max_steps", "approach_episode_steps"):
            if int(self.cfg[key]) < 0:
                raise ValueError(f"verification.{key} must be >= 0")
        self.enabled = bool(self.cfg["enabled"])
        self.log_dir = str(self.cfg["log_dir"]).format(video_output_path=video_output_path)
        self.camera = camera
        self._verify = verify or client.verify
        self.server = None
        self.consecutive_errors = 0
        self.start_episode(-1, "", "", "")
        if self.enabled:
            # A wrong URL, protocol or dead worker is a setup error, not a per-request failure.
            self.server = (ping or client.ping)(self.cfg["url"], timeout=float(self.cfg["ping_timeout_s"]))
            selector = (self.server.get("settings") or {}).get("selector")
            # Three-way verdicts need the approach on both sides, and the approach needs them
            if self.approach != (selector == "d1"):
                raise ValueError(f"verification.approach={self.approach} does not match the server's "
                                 f"selector {selector!r} (approach needs --selector d1, and d1 needs approach)")
            os.makedirs(self.log_dir, exist_ok=True)
            print(f"[verification] gate ON: {self.cfg['url']} {json.dumps(self.server.get('settings'))}")

    # ------------------------------------------------------------------
    # Episode lifecycle
    # ------------------------------------------------------------------

    def start_episode(self, run_index, scene, episode_id, label, goal_object_ids=()):
        self.episode = {"run_index": int(run_index), "scene": str(scene), "episode_id": str(episode_id),
                        "label": str(label), "target": target_name(label),
                        "goal_object_ids": [str(i) for i in goal_object_ids if i is not None]}
        self.evidence = EvidenceStore(self.cfg["match_distance_m"], self.cfg["evidence_per_cell"],
                                      self.cfg["max_evidence"])
        self.clusters, self.verdicts, self.events = {}, {}, []
        self.counts = {"requests": 0, "accepted": 0, "rejected": 0, "deferred": 0, "errors": 0, "no_evidence": 0,
                       "cached": 0, "reopened": 0}
        if self.approach:  # only with the approach, so other runs keep their counts
            self.counts.update({key: 0 for key in APPROACH_COUNTS})
        self.approach_steps_used = 0

    def finish_episode(self):
        """Per-episode counts (None when the gate is off)."""
        return dict(self.counts) if self.enabled else None

    def pop_events(self):
        """Decisions and reopens since the last call, for the step record."""
        events, self.events = self.events, []
        return events or None

    # ------------------------------------------------------------------
    # Evidence
    # ------------------------------------------------------------------

    def observe(self, step, rgb, masks, labels, scores, centroids, semantic_instances, pose,
                camera_pitch, fusion=None):
        """Store this step's target detections; returns rejected cluster ids to reopen."""
        if not self.enabled:
            return []
        snapshot = {int(c["id"]): tuple(c["centroid"]) for c in (fusion or {}).get("clusters", [])}
        new = []
        for index, label in enumerate(labels or []):
            if int(label) != 0 or index >= len(masks) or centroids is None or index >= len(centroids):
                continue
            if centroids[index] is None:
                continue
            mask = (np.asarray(masks[index]).squeeze() > 0).astype(np.uint8)
            area = int(mask.sum())
            if not area:
                continue
            score = float(scores[index]) if scores is not None and index < len(scores) else None
            item = {"step": int(step), "index": index, "rgb": rgb, "mask": mask, "score": score,
                    "area_fraction": area / float(mask.size),
                    "quality": view_quality(score, area / float(mask.size), mask_touches_border(mask)),
                    "centroid": (float(centroids[index][0]), float(centroids[index][1])),
                    "camera": (float(pose[0]), float(pose[1])), "pitch": float(camera_pitch),
                    "semantic_instances": (semantic_instances[index] if semantic_instances is not None
                                           and index < len(semantic_instances) else None),
                    "clusters": snapshot}
            new.append(item)
            self.evidence.add(item)
        return self._reopen(step, new, snapshot) + self._approach_views(step, new)

    def _novel(self, item, used):
        """True when ``item`` sees its object from far enough, or a different enough direction,
        compared with every view used before."""
        bearing = _bearing(item["camera"], item["centroid"])
        return all(_distance(item["camera"], u["camera"]) >= self.cfg["recheck_min_distance_m"]
                   or math.degrees(_angle_between(bearing, u["bearing"])) >= self.cfg["recheck_min_angle_deg"]
                   for u in used)

    def _reference(self, item, cluster_id, centroid, by_position=False):
        """Where ``item`` should be for ``cluster_id``: the cluster's centroid at the item's step
        (clusters drift as they grow), else ``centroid``. Position requests (-1, at_stop) use
        ``centroid``."""
        if cluster_id < 0 or by_position:
            return centroid
        return item["clusters"].get(cluster_id, centroid)

    def _closer_to_other(self, item, cluster_id, distance):
        """True when another cluster (as of the item's step) is nearer the item than ``distance``."""
        return any(other != cluster_id and _distance(item["centroid"], c) < distance
                   for other, c in item["clusters"].items())

    def _reopen(self, step, new, snapshot):
        reopened = []
        for cluster_id, state in self.clusters.items():
            if (state["state"] != REJECTED or state["checks"] >= self.cfg["max_checks"]
                    or step - state["last_step"] < self.cfg["recheck_cooldown_steps"]):
                continue
            centroid = snapshot.get(cluster_id, state["centroid"])
            if any(_distance(item["centroid"], self._reference(item, cluster_id, centroid))
                   <= self.cfg["match_distance_m"] and self._novel(item, state["used"]) for item in new):
                state["state"] = UNVERIFIED
                reopened.append(cluster_id)
                self.counts["reopened"] += 1
                self.events.append({"event": "reopen", "cluster_id": cluster_id, "step": int(step)})
                print(f"[verification] cluster {cluster_id} reopened at step {step} (new view)")
        return reopened

    def select_evidence(self, cluster_id, centroid, by_position=False):
        """(evidence, match distance) for a cluster (or a position: ``cluster_id`` -1 or
        ``by_position``): the most readable view within ``match_distance_m`` (ties: nearest),
        else the nearest within twice that, else (None, None). Views nearer another cluster do
        not count for a cluster. A re-check prefers views that are new by the re-check rule."""
        used = self.clusters.get(cluster_id, {}).get("used", [])
        pool = list(self.evidence)
        if used:
            pool = [e for e in pool if self._novel(e, used)] or pool

        def distance(item):
            return _distance(item["centroid"], self._reference(item, cluster_id, centroid, by_position))

        if cluster_id >= 0 and not by_position:
            pool = [e for e in pool if not self._closer_to_other(e, cluster_id, distance(e))]
        matched = [e for e in pool if distance(e) <= self.cfg["match_distance_m"]]
        if matched:
            best = max(matched, key=lambda e: (e["quality"], -distance(e)))
            return best, distance(best)
        near = [e for e in pool if distance(e) <= 2 * self.cfg["match_distance_m"]]
        if near:
            best = min(near, key=distance)
            return best, distance(best)
        return None, None

    # ------------------------------------------------------------------
    # Verification
    # ------------------------------------------------------------------

    def handle(self, request, fusion=None, step=0):
        """Verify the requested cluster; returns 1 (accept), -1 (reject), 2 (defer) or 3 (approach;
        ``pending_approach`` has it) for the planner.

        ``request``: dict with seq, cluster_id, centroid_x/y, confidence, observation_num,
        expl_result, at_stop, and for an approach re-request approach_id, viewpoint_status,
        viewpoint_x/y and approach_steps (the VerificationRequest message)."""
        cluster_id = int(request["cluster_id"])
        centroid = (float(request["centroid_x"]), float(request["centroid_y"]))
        approach_id = int(request.get("approach_id", 0) or 0)
        self.counts["requests"] += 1
        record = {**{k: self.episode.get(k) for k in ("run_index", "scene", "episode_id", "label", "target")},
                  "step": int(step), "seq": int(request.get("seq", 0)),
                  "cluster": {"id": cluster_id, "centroid": list(centroid),
                              "confidence": float(request.get("confidence", 0.0)),
                              "observation_num": int(request.get("observation_num", 0))},
                  "expl_result": int(request.get("expl_result", -1)),
                  "at_stop": bool(request.get("at_stop", False)),
                  "server": None if self.server is None else self.server.get("settings"),
                  "camera": self.camera, "evidence": None, "verdict": None, "latency_s": None,
                  "error": None, "policy": None, "source": "server", "ground_truth": None}
        if self.approach:
            record.update(status=None, model=None, approach=None)

        # Position requests (stop backstop, no cluster) carry no per-cluster state.
        state = None
        at_stop = bool(request.get("at_stop", False))
        if cluster_id >= 0:
            state = self.clusters.get(cluster_id)
            if state is None:
                state = self.clusters[cluster_id] = {"state": UNVERIFIED, "checks": 0, "last_step": int(step),
                                                     "used": [], "centroid": centroid, "approaches": 0,
                                                     "approach": None}
            # approach re-requests are part of the check that issued the approach
            state.update(checks=state["checks"] + (0 if approach_id else 1), last_step=int(step),
                         centroid=centroid)
            record["cluster"]["check"] = state["checks"]

        if approach_id and self.approach:
            return self._after_approach(record, request, state, step)
        if state is not None and state.get("approach") is not None:
            print(f"[verification] cluster {cluster_id}: stale approach {state['approach']['id']} dropped")
            state["approach"] = None

        evidence, match = self.select_evidence(cluster_id, centroid, by_position=at_stop)
        if evidence is None:
            self.counts["no_evidence"] += 1
            record["policy"] = f"on_no_evidence={self.cfg['on_no_evidence']}"
            return self._decide(record, ACCEPTED if self.cfg["on_no_evidence"] == "accept" else REJECTED,
                                None, None, state)
        return self._judge(record, evidence, match, state, step, at_stop)

    def _judge(self, record, evidence, match, state, step, at_stop, approach=None):
        """Verify ``evidence`` and decide. ``approach``: the approach this view comes from
        (its re-request), which never reuses a cached verdict."""
        cluster_id = record["cluster"]["id"]
        elevation = -math.degrees(evidence["pitch"])  # Habitat pitch is negative looking down
        record["evidence"] = {"step": evidence["step"], "index": evidence["index"], "score": evidence["score"],
                              "area_fraction": round(evidence["area_fraction"], 5),
                              "match_distance_m": round(match, 3), "camera": list(evidence["camera"]),
                              "centroid": list(evidence["centroid"]), "pitch": evidence["pitch"],
                              "input_elevation": elevation}
        record["ground_truth"] = semantic_goal_overlap(
            evidence["semantic_instances"], self.episode.get("goal_object_ids", []),
            self.cfg["ground_truth_semantic_overlap"])
        if state is not None:
            used = {"camera": evidence["camera"], "bearing": _bearing(evidence["camera"], evidence["centroid"])}
            if self.approach:
                used["range"] = _distance(evidence["camera"], evidence["centroid"])
            state["used"].append(used)
        key = (evidence["step"], evidence["index"])
        if key in self.verdicts and approach is None:  # this exact view was already judged (for another cluster)
            self.counts["cached"] += 1
            record.update(source="cached", verdict=self.verdicts[key]["verdict"])
            if self._tri_state(record["verdict"]) is not None:
                return self._decide_three_way(record, record["verdict"], evidence, None, state, at_stop, approach)
            cached = self.verdicts[key]["state"]
            return self._decide(record, REJECTED if cached == DEFERRED and at_stop else cached, evidence, None, state)

        start = time.time()
        try:
            reply = self._verify(self.cfg["url"], evidence["rgb"], evidence["mask"], self.episode["target"],
                                 input_elevation=elevation,
                                 request_id=f"epi{self.episode.get('run_index')}_c{cluster_id}_s{step}",
                                 timeout=float(self.cfg["timeout_s"]))
            verdict = reply["verdict"]
        except Exception as error:  # fail open (or closed) per on_error, always logged
            self.consecutive_errors += 1
            self.counts["errors"] += 1
            record.update(error=f"{type(error).__name__}: {error}",
                          policy=(f"{record['policy']}," if record["policy"] else "")
                          + f"on_error={self.cfg['on_error']}",
                          latency_s=round(time.time() - start, 3))
            decided = self._decide(record, ACCEPTED if self.cfg["on_error"] == "accept" else REJECTED,
                                   evidence, None, state)
            if self.consecutive_errors >= self.cfg["max_consecutive_errors"]:
                raise RuntimeError(f"perception server failed {self.consecutive_errors} times in a row "
                                   f"(last: {record['error']}); aborting rather than running ungated")
            return decided
        self.consecutive_errors = 0
        record.update(verdict=verdict, latency_s=round(time.time() - start, 3))
        if self._tri_state(verdict) is not None:
            self.verdicts[key] = {"state": None, "verdict": verdict}
            return self._decide_three_way(record, verdict, evidence, reply, state, at_stop, approach)
        decided = ACCEPTED if verdict["accepted"] else REJECTED
        if decided == REJECTED and self.defers(verdict) and not at_stop:
            decided = DEFERRED
        self.verdicts[key] = {"state": decided, "verdict": verdict}
        return self._decide(record, decided, evidence, reply, state)

    # ------------------------------------------------------------------
    # Approach (d1)
    # ------------------------------------------------------------------

    def _tri_state(self, verdict):
        """The three-way status of ``verdict`` when the approach is on, else None (binary)."""
        if not self.approach:
            return None
        status = (verdict or {}).get("status")
        if status not in TRI_STATES:
            raise RuntimeError(f"d1 verdict without a three-way status: {status!r}")
        return status

    def _decide_three_way(self, record, verdict, evidence, reply, state, at_stop, approach):
        """correct -> accept, wrong -> reject, insufficient -> approach (budget permitting) or reject."""
        status = self._tri_state(verdict)
        model = verdict.get("source")
        record.update(status=status, model=model)
        if model in ("sam3", "qwen"):
            self.counts[f"{model}_decisions"] += 1
        if approach is not None:
            self.counts[f"approach_{'accepted' if status == 'correct' else 'rejected' if status == 'wrong' else 'insufficient'}"] += 1
        if status == "correct":
            return self._decide(record, ACCEPTED, evidence, reply, state)
        if status == "wrong":
            return self._decide(record, REJECTED, evidence, reply, state)
        self.counts["insufficient"] += 1
        blocked = self._approach_blocked(record, state, at_stop, approach)
        if blocked is not None:
            self.counts["approach_exhausted"] += 1
            record["policy"] = f"insufficient,{blocked}"
            return self._decide(record, REJECTED, evidence, reply, state)
        self._issue_approach(record, state, evidence, at_stop)
        return self._decide(record, APPROACH, evidence, reply, state)

    def _approach_remaining(self):
        return int(self.cfg["approach_episode_steps"]) - self.approach_steps_used

    def _approach_blocked(self, record, state, at_stop, approach):
        """Why an insufficient view cannot be approached (None: it can)."""
        if record["cluster"]["id"] < 0 or state is None:
            return "position"
        if approach is not None and approach["at_stop"]:
            return "stop_side_view_used"  # one side-view approach at the stop
        if state["approaches"] >= int(self.cfg["approach_max_per_cluster"]):
            return "cluster_budget"
        if min(int(self.cfg["approach_max_steps"]), self._approach_remaining()) <= 0:
            return "episode_budget"
        return None

    def _issue_approach(self, record, state, evidence, at_stop):
        state["approaches"] += 1
        self.counts["moves"] += 1
        pending = {"id": state["approaches"], "issued_step": int(record["step"]),
                   "object": tuple(evidence["centroid"]), "camera": tuple(evidence["camera"]),
                   "max_steps": min(int(self.cfg["approach_max_steps"]), self._approach_remaining()),
                   "side_only": bool(at_stop), "at_stop": bool(at_stop), "early": None}
        state["approach"] = pending
        record["approach"] = {"id": pending["id"], "issued": True, "object": list(pending["object"]),
                              "camera": list(pending["camera"]), "max_steps": pending["max_steps"],
                              "side_only": pending["side_only"]}
        self.events.append({"event": "approach", "cluster_id": record["cluster"]["id"], "approach": pending["id"],
                            "object": list(pending["object"]), "camera": list(pending["camera"]),
                            "max_steps": pending["max_steps"], "side_only": pending["side_only"]})

    def pending_approach(self, cluster_id):
        """The approach a state-3 decision issued for ``cluster_id``: {approach_id, object, camera,
        max_steps, side_only}; None if there is none."""
        pending = self.clusters.get(int(cluster_id), {}).get("approach")
        if pending is None:
            return None
        return {"approach_id": pending["id"], "object": pending["object"], "camera": pending["camera"],
                "max_steps": pending["max_steps"], "side_only": pending["side_only"]}

    def _approach_view(self, item, cluster_id, pending, used):
        """Match distance when ``item`` is a new view of the approached object: taken since the
        approach was issued, of the object point (not nearer another cluster), and closer by
        ``approach_min_closer_m`` than, or ``approach_min_angle_deg`` around the object from,
        every view judged for the cluster. Else None."""
        if item["step"] < pending["issued_step"]:
            return None
        distance = _distance(item["centroid"], pending["object"])
        if distance > self.cfg["match_distance_m"] or self._closer_to_other(item, cluster_id, distance):
            return None
        rng = _distance(item["camera"], item["centroid"])
        bearing = _bearing(item["camera"], item["centroid"])
        closer, angle = float(self.cfg["approach_min_closer_m"]), math.radians(self.cfg["approach_min_angle_deg"])
        if all(rng <= u.get("range", _distance(u["camera"], pending["object"])) - closer
               or _angle_between(bearing, u["bearing"]) >= angle for u in used):
            return distance
        return None

    def _approach_views(self, step, new):
        """Clusters under approach with a new view this step (once per approach): the planner
        ends the approach early (viewpoint_status 4)."""
        if not self.approach:
            return []
        ready = []
        for cluster_id, state in self.clusters.items():
            pending = state.get("approach")
            if pending is None or pending["early"] is not None:
                continue
            views = [(item, d) for item in new
                     for d in [self._approach_view(item, cluster_id, pending, state["used"])] if d is not None]
            if views:
                pending["early"] = max(views, key=lambda v: (v[0]["quality"], -v[1]))
                ready.append(cluster_id)
                self.events.append({"event": "approach_view", "cluster_id": cluster_id, "step": int(step),
                                    "approach": pending["id"]})
                print(f"[verification] cluster {cluster_id}: new view at step {step} during approach {pending['id']}")
        return ready

    def _after_approach(self, record, request, state, step):
        """An approach re-request (viewpoint_status 1 arrived, 2 unreachable, 3 aborted, 4 early)."""
        approach_id = int(request.get("approach_id", 0))
        status = int(request.get("viewpoint_status", 0))
        steps = max(int(request.get("approach_steps", 0)), 0)
        self.approach_steps_used += steps
        self.counts["approach_steps"] += steps
        pending = state.get("approach") if state is not None else None
        record["approach"] = {"id": approach_id, "status": STATUS_NAMES.get(status, str(status)), "steps": steps,
                              "viewpoint": [float(request.get("viewpoint_x", 0.0)),
                                            float(request.get("viewpoint_y", 0.0))],
                              "episode_steps_used": self.approach_steps_used}
        if state is not None:
            state["approach"] = None
        if pending is None or pending["id"] != approach_id:
            record["error"] = (f"approach {approach_id} re-request, pending "
                               f"{None if pending is None else pending['id']}")
            print(f"[verification] cluster {record['cluster']['id']}: {record['error']}")
            return self._approach_failed(record, state, "approach_mismatch")
        record["at_stop"] = pending["at_stop"]
        record["approach"].update(object=list(pending["object"]), issued_step=pending["issued_step"])
        if status == UNREACHABLE:
            return self._approach_failed(record, state, "unreachable")
        if status == ABORTED:
            return self._approach_failed(record, state, "aborted")
        if status not in (ARRIVED, EARLY):
            record["error"] = f"unknown viewpoint_status {status}"
            return self._approach_failed(record, state, "approach_mismatch")
        self.counts["approach_arrived" if status == ARRIVED else "approach_early"] += 1
        view = None
        if status == EARLY and pending["early"] is not None:
            view = pending["early"]
        else:
            views = [(item, d) for item in self.evidence
                     for d in [self._approach_view(item, record["cluster"]["id"], pending, state["used"])]
                     if d is not None]
            if views:  # the newest: on arrival, the one facing the object
                view = max(views, key=lambda v: (v[0]["step"], v[0]["quality"]))
        if view is None:
            return self._approach_failed(record, state, "not_redetected")
        return self._judge(record, view[0], view[1], state, step, pending["at_stop"], approach=pending)

    def _approach_failed(self, record, state, policy):
        """Unreachable, aborted, not re-detected (or a protocol mismatch): reject."""
        if policy in self.counts:
            self.counts[policy] += 1
        self.counts["fallback"] += 1
        record["policy"] = policy
        return self._decide(record, REJECTED, None, None, state)

    def defers(self, verdict):
        """True when ``verdict`` is a SAM3D view rejecting what the raw frame accepted, and
        such rejections defer."""
        return (self.cfg["on_view_reject"] == "defer" and verdict.get("judged") == "view"
                and verdict.get("input_scores") is not None)

    def _decide(self, record, decided, evidence, reply, state):
        if state is not None:
            state["state"] = decided
        record["state"] = decided
        if decided != APPROACH:  # an approach is counted as a move when issued
            self.counts[NAMES[decided].lower()] += 1
        self._log(record, evidence, reply)
        label = (record["verdict"] or {}).get("label")
        self.events.append({"event": "decision", "cluster_id": record["cluster"]["id"], "state": decided,
                            "label": label, "source": record["source"], "error": record["error"]})
        print(f"[verification] {'position' if record['cluster']['id'] < 0 else 'cluster'} "
              f"{record['cluster']['id']} {NAMES[decided]}: label {label!r} "
              f"for {self.episode.get('target')!r} ({record['policy'] or record['source']}, "
              f"{record['latency_s'] or 0.0:.1f} s)")
        return decided

    def _log(self, record, evidence, reply):
        if not self.log_dir:
            return
        os.makedirs(self.log_dir, exist_ok=True)
        with open(os.path.join(self.log_dir, "decisions.jsonl"), "a", encoding="utf-8") as stream:
            stream.write(json.dumps(record) + "\n")
        if not self.cfg["save_images"] or evidence is None:
            return
        folder = os.path.join(self.log_dir, f"epi{record['run_index']}_{slugify_label(record['label'])}",
                              f"c{record['cluster']['id']}_s{record['step']}")
        os.makedirs(folder, exist_ok=True)
        cv2.imwrite(os.path.join(folder, "input.jpg"), cv2.cvtColor(evidence["rgb"], cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(folder, "mask.png"), evidence["mask"] * 255)
        if reply and reply.get("view_image"):
            cv2.imwrite(os.path.join(folder, "view.png"),
                        cv2.cvtColor(client.decode_png(reply["view_image"]), cv2.COLOR_RGB2BGR))
        with open(os.path.join(folder, "decision.json"), "w", encoding="utf-8") as stream:
            json.dump(record, stream, indent=1)
