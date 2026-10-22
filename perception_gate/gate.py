"""VerificationGate: the Python half of the perception gate (no ROS imports).

Per episode:
    gate.start_episode(run_index, scene, episode_id, label, goal_object_ids)
    reopen = gate.observe(step, rgb, masks, labels, scores, centroids, semantic_instances,
                          pose, camera_pitch, fusion)        # after every sim step
    state = gate.handle(request, fusion, step)               # on /object/verification_request
    counts = gate.finish_episode()

``observe`` stores every target-class (label 0) detection as evidence, per
0.75 m cell of its world centroid (``evidence.EvidenceStore``: the most readable
views of each place plus its newest). ``handle`` verifies the best view of the
requested cluster with ``perception.Verifier`` over HTTP and returns the state
for the planner: 1 accepted, -1 rejected. A request with ``cluster_id`` -1 is
the stop backstop asking about a position that no cluster covers (e.g. an
over-depth target); it is verified from the evidence nearest that position.

Verdicts are cached by evidence view (step, index): a cluster whose selected
view was already judged (a re-spawn of a judged cluster) reuses that verdict
without a server call. A distinct nearby object is judged on its own views: a
view closer to another cluster (as clusters stood at that step) is not
evidence for this one, and ties prefer the nearest view. ``at_stop`` requests
match evidence to the stop position the planner sends.

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
}
POLICIES = ("accept", "reject")
ACCEPTED, UNVERIFIED, REJECTED = 1, 0, -1


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
        self.counts = {"requests": 0, "accepted": 0, "rejected": 0, "errors": 0, "no_evidence": 0,
                       "cached": 0, "reopened": 0}

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
        return self._reopen(step, new, snapshot)

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
        """Verify the requested cluster; returns 1 (accept) or -1 (reject) for the planner.

        ``request``: dict with seq, cluster_id, centroid_x/y, confidence, observation_num,
        expl_result, at_stop (the VerificationRequest message)."""
        cluster_id = int(request["cluster_id"])
        centroid = (float(request["centroid_x"]), float(request["centroid_y"]))
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

        # Position requests (stop backstop, no cluster) carry no per-cluster state.
        state = None
        at_stop = bool(request.get("at_stop", False))
        if cluster_id >= 0:
            state = self.clusters.get(cluster_id)
            if state is None:
                state = self.clusters[cluster_id] = {"state": UNVERIFIED, "checks": 0, "last_step": int(step),
                                                     "used": [], "centroid": centroid}
            state.update(checks=state["checks"] + 1, last_step=int(step), centroid=centroid)
            record["cluster"]["check"] = state["checks"]

        evidence, match = self.select_evidence(cluster_id, centroid, by_position=at_stop)
        if evidence is None:
            self.counts["no_evidence"] += 1
            record["policy"] = f"on_no_evidence={self.cfg['on_no_evidence']}"
            return self._decide(record, ACCEPTED if self.cfg["on_no_evidence"] == "accept" else REJECTED,
                                None, None, state)

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
            state["used"].append({"camera": evidence["camera"],
                                  "bearing": _bearing(evidence["camera"], evidence["centroid"])})
        key = (evidence["step"], evidence["index"])
        if key in self.verdicts:  # this exact view was already judged (for another cluster)
            self.counts["cached"] += 1
            record.update(source="cached", verdict=self.verdicts[key]["verdict"])
            return self._decide(record, self.verdicts[key]["state"], evidence, None, state)

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
            record.update(error=f"{type(error).__name__}: {error}", policy=f"on_error={self.cfg['on_error']}",
                          latency_s=round(time.time() - start, 3))
            decided = self._decide(record, ACCEPTED if self.cfg["on_error"] == "accept" else REJECTED,
                                   evidence, None, state)
            if self.consecutive_errors >= self.cfg["max_consecutive_errors"]:
                raise RuntimeError(f"perception server failed {self.consecutive_errors} times in a row "
                                   f"(last: {record['error']}); aborting rather than running ungated")
            return decided
        self.consecutive_errors = 0
        record.update(verdict=verdict, latency_s=round(time.time() - start, 3))
        decided = ACCEPTED if verdict["accepted"] else REJECTED
        self.verdicts[key] = {"state": decided, "verdict": verdict}
        return self._decide(record, decided, evidence, reply, state)

    def _decide(self, record, decided, evidence, reply, state):
        if state is not None:
            state["state"] = decided
        record["state"] = decided
        self.counts["accepted" if decided == ACCEPTED else "rejected"] += 1
        self._log(record, evidence, reply)
        label = (record["verdict"] or {}).get("label")
        self.events.append({"event": "decision", "cluster_id": record["cluster"]["id"], "state": decided,
                            "label": label, "source": record["source"], "error": record["error"]})
        print(f"[verification] {'position' if record['cluster']['id'] < 0 else 'cluster'} "
              f"{record['cluster']['id']} {'ACCEPTED' if decided == ACCEPTED else 'REJECTED'}: label {label!r} "
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
