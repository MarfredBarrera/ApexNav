"""VerificationGate: the Python half of the perception gate (no ROS imports).

Per episode:
    gate.start_episode(run_index, scene, episode_id, label, goal_object_ids, max_episode_steps)
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
approach: the cluster becomes the planner's target and ApexNav's own object navigation
drives toward it. While an approach is pending, ``observe`` returns the cluster's id
(with the reopens) on the first detection of the object that is
``approach_min_closer_m`` closer than, or ``approach_min_angle_deg`` away from, every view
already judged for the cluster; the planner asks about it at once (``viewpoint_status`` 4):
accept, reject, or insufficient -> keep approaching (state 3 again, same approach). On
arrival (the object reached, the agent facing it; status 1) a view from that very step is
judged without those thresholds; insufficient there is a forced decision by SAM3's raw label on
that view, and not re-detected, by SAM3's raw label on the cluster's last judged view. The planner arrives within 1.5 m of the object
(standoff) or on reaching it; a view of the object matches the issued object point or the
cluster's centroid at that step. Status 2 (no
object path, target lost) and 3 (step budget used up) reject. Approach re-requests bypass
the re-check cooldown, ``max_checks`` and the verdict cache. Budgets:
``approach_max_per_cluster`` approaches per cluster, ``approach_max_steps`` actions each,
``approach_episode_steps`` per episode, never past the episode's step limit; no budget ->
reject. At the stop backstop (``at_stop``) and for positions (cluster -1) insufficient
rejects. A rejected cluster then follows the re-check rule below. Approach needs
``on_error: reject``.

Multi-view close range (``multiview``, needs ``approach`` and a server with
``settings.multiview``): while an approach is pending, every view of the object goes into the
cluster's view bank (``state["bank"]``, seeded from the evidence when the approach is issued,
at most ``multiview_bank`` views; fragments of the object, clusters within
``multiview_sibling_m`` of it, do not exclude views). Early re-checks, arrivals and (with
``multiview_failure``) failed approaches with ``multiview_min_views`` distinct-step views are
decided over ``select_views`` of the bank in one /verify_views request: Qwen correct/wrong
decides; insufficient keeps approaching (early) or falls back to SAM3 over the views
(``select.aggregate_views``): accept iff margin M >= ``multiview_arrival_margin`` (arrival) or
``multiview_failure_margin`` (failure) and the target vote V >= ``multiview_vote``, else reject.
``multiview_grid``: Qwen also sees the SAM3D grid of the best close view (slot 1). Rule C (not
re-detected) and rule D (SAM3 on the arrival view) are not used; an empty bank rejects.

JEV + SAM3 (server selector ``jev_sam3``, the d1mv approach and view bank, no grid): JEV-27B-VL only judges the uncertainty
(unsure -> approach; unsure views on the way keep approaching); SAM3 makes every decision. The first
request: SAM3's raw label (model ``sam3``). /verify_views on arrivals, failed approaches, and views on
the way JEV is sure about: SAM3 over the views decides with the fused (evidential) belief, accept iff
the target is its top (server ``select.fuse_views``; 2026-11-07, it replaced the margin-and-vote rule; the
``multiview_*_margin`` / ``multiview_vote`` settings apply to the d1 fallback only)
(model ``sam3``, policy ``multiview,<mode>,sam3_<accept|reject>``).

Failure policy (fail open by default): a server error or timeout applies
``on_error``, a cluster without a stored view ``on_no_evidence``; both are
logged. ``on_no_evidence`` may also be ``defer``: the planner carries on to the cluster and the stop
re-verifies it (by position, with the views gathered on the way); a request at the stop without a view
is rejected, as the planner treats a deferral at the stop. ``max_consecutive_errors`` server errors in a row raise, which aborts
the run, so a run never goes on silently ungated.

Re-check: a rejected (or deferred) cluster is reopened (``observe`` returns its id; the caller
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
    "approach": False,               # d1/jev_sam3: insufficient -> approach (state 3); needs selector d1 or jev_sam3 and the planner's verification_approach
    "approach_max_per_cluster": 2,
    "approach_max_steps": 40,        # actions per approach
    "approach_episode_steps": 120,   # approach actions per episode
    "approach_min_closer_m": 0.5,    # a new view: this much closer than every judged view of the cluster ...
    "approach_min_angle_deg": 30.0,  # ... or this far around the object from each
    "multiview": False,              # d1 close range: decide over the bank of the object's views (/verify_views)
    "multiview_max_views": 6,        # views per request (server limit 8)
    "multiview_bank": 24,            # views kept per cluster
    "multiview_min_views": 2,        # distinct-step views a failed approach needs for a decision
    "multiview_sibling_m": 0.5,      # a nearer cluster this close to the object is a fragment of it
    "multiview_dup_distance_m": 0.3, # a view this close to a chosen one ...
    "multiview_dup_angle_deg": 15.0, # ... and looking from this similar a direction is a duplicate
    "multiview_arrival_margin": 0.0, # d1 SAM3 fallback on arrival: accept iff M >= this and V >= vote
    "multiview_failure_margin": 0.15,  # ... after a failed approach
    "multiview_vote": 0.5,
    "multiview_failure": True,       # decide failed approaches over the bank: true, false or "aborted" (budget only)
    "multiview_grid": False,         # Qwen also sees the SAM3D grid of the best close view (server --multiview-grid)
    "multiview_sam3": "fallback",    # SAM3 per view: fallback (Qwen insufficient), always (ablation) or never
    "multiview_sam3_shortcut": False,  # ablation: a sufficient best close view lets SAM3 decide
    "multiview_variant": None,       # ablation: crop_small or forced2
    "multiview_forced2_min": 0.5,    # forced2: correct iff p(A) / (p(A) + p(B)) >= this
    "multiview_grid_scores": False,  # diagnostic: SAM3 top-1 of each grid render (logged only)
    "approach_planner": None,        # None: ApexNav's object navigation approaches (state 3); mppi / closer: the eval
                                     # loop drives the approach itself (perception_gate.approach, sync_step only)
    "last_mile": True,               # approach_planner: after an accept the gate's planner drives the last mile
                                     # (habitat_evaluation.run_gate_last_mile / run_gate_approach); false: ApexNav's
                                     # own navigation goes on to the accepted (unforced) cluster
    "close_in": False,               # terminal close-in at an accepted object when the planner stops / ends stucking
                                     # (perception_gate.close_in; habitat_evaluation.run_close_in)
    "close_in_budget": 12,           # actions before its final STOP
    "takeover": False,               # at the FIRST accept the bench's last-mile navigator (lastmile.live, F_full_cur) drives
                                     # to the end of the episode, never handing back (habitat_evaluation.run_takeover)
}
POLICIES = ("accept", "reject")
NO_EVIDENCE_POLICIES = POLICIES + ("defer",)
ACCEPTED, UNVERIFIED, REJECTED, DEFERRED, APPROACH = 1, 0, -1, 2, 3
NAMES = {ACCEPTED: "ACCEPTED", REJECTED: "REJECTED", DEFERRED: "DEFERRED", APPROACH: "APPROACH"}
# VerificationRequest.viewpoint_status of an approach re-request
ARRIVED, UNREACHABLE, ABORTED, EARLY = 1, 2, 3, 4
# approach_planner hand-off (approach_close): the gate's planner takes an approach once the object is this close
# and detected (target mask of >= HANDOFF_PX matching the object point within HANDOFF_MATCH_M)
HANDOFF_M, HANDOFF_MATCH_M, HANDOFF_PX = 3.0, 1.0, 200
HANDOFF_SEEN_M = 1.0   # ... or the evidence view was taken within this of the agent (seen from here: turning finds it)
STATUS_NAMES = {ARRIVED: "arrived", UNREACHABLE: "unreachable", ABORTED: "aborted", EARLY: "early"}
TRI_STATES = ("correct", "wrong", "insufficient")
APPROACH_COUNTS = ("insufficient", "moves", "unreachable", "aborted", "not_redetected", "fallback",
                   "approach_arrived", "approach_early", "approach_accepted", "approach_rejected", "approach_undecided",
                   "approach_insufficient", "approach_exhausted", "approach_steps", "sam3_decisions",
                   "qwen_decisions", "jev_decisions")
MULTIVIEW_COUNTS = ("multiview_requests", "multiview_qwen", "multiview_sam3", "multiview_failure_decided",
                    "multiview_views", "multiview_grid_errors", "multiview_jev")
MULTIVIEW_VARIANTS = (None, "crop_small", "forced2")
MULTIVIEW_SAM3 = ("fallback", "always", "never")
THREE_WAY_SELECTORS = ("d1", "jev_sam3")  # servers answering correct / wrong / insufficient (the approach)


def target_name(label):
    """The Verifier category for an ApexNav label ("potted_plant", "table | desk")."""
    return str(label).split("|")[0].strip().replace("_", " ")


def _distance(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _bearing(camera, point):
    return math.atan2(point[1] - camera[1], point[0] - camera[0])


def _angle_between(a, b):
    return abs((a - b + math.pi) % (2 * math.pi) - math.pi)


def _range(item):
    return _distance(item["camera"], item["centroid"])


def select_views(bank, k=6, dup_distance_m=0.3, dup_angle_deg=15.0):
    """(views, best): up to ``k`` views of one object from ``bank`` (evidence items), nearest
    first. ``best`` (slot 1, always included; the SAM3D grid view) is the most readable view
    among those at most the median range away. The rest are added by quality (ties: nearer,
    newer), skipping duplicates: the same (step, index), or a camera within ``dup_distance_m`` of
    a chosen view's looking from within ``dup_angle_deg`` of its direction."""
    items = list({(i["step"], i["index"]): i for i in bank}.values())
    if not items or k < 1:
        return [], None
    ranges = sorted(_range(i) for i in items)
    median = (ranges[(len(ranges) - 1) // 2] + ranges[len(ranges) // 2]) / 2
    order = sorted(items, key=lambda i: (-i["quality"], _range(i), -i["step"], i["index"]))
    best = next(i for i in order if _range(i) <= median)
    angle = math.radians(dup_angle_deg)

    def duplicate(a, b):
        return ((a["step"], a["index"]) == (b["step"], b["index"])
                or (_distance(a["camera"], b["camera"]) <= dup_distance_m
                    and _angle_between(_bearing(a["camera"], a["centroid"]),
                                       _bearing(b["camera"], b["centroid"])) <= angle))

    chosen = [best]
    for item in order:
        if len(chosen) >= k:
            break
        if not any(duplicate(item, c) for c in chosen):
            chosen.append(item)
    return sorted(chosen, key=lambda i: (_range(i), -i["step"])), best


class VerificationGate:
    """``camera``: intrinsics logged with every decision, e.g. {hfov, width, height}."""

    def __init__(self, cfg=None, video_output_path=".", verify=None, ping=None, camera=None, verify_views=None):
        unknown = set(cfg or {}) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"unknown verification settings: {sorted(unknown)}")
        self.cfg = {**DEFAULTS, **{k: v for k, v in dict(cfg or {}).items() if v is not None}}
        for key, allowed in (("on_error", POLICIES), ("on_no_evidence", NO_EVIDENCE_POLICIES)):
            if self.cfg[key] not in allowed:
                raise ValueError(f"verification.{key} must be one of {allowed}")
        if self.cfg["on_view_reject"] not in ("reject", "defer"):
            raise ValueError("verification.on_view_reject must be reject or defer")
        self.approach = bool(self.cfg["approach"])
        if self.approach and self.cfg["on_error"] != "reject":
            raise ValueError("verification.approach needs on_error=reject (every approach failure rejects)")
        for key in ("approach_max_per_cluster", "approach_max_steps", "approach_episode_steps"):
            if int(self.cfg[key]) < 0:
                raise ValueError(f"verification.{key} must be >= 0")
        self.approach_planner = self.cfg["approach_planner"]
        if self.approach_planner is not None:
            from perception_gate.approach import PLANNERS
            if self.approach_planner not in PLANNERS or not self.approach:
                raise ValueError(f"verification.approach_planner must be one of {PLANNERS} and needs approach")
        self.last_mile = bool(self.cfg["last_mile"]) and self.approach_planner is not None
        if not self.cfg["last_mile"] and self.approach_planner is None:
            raise ValueError("verification.last_mile=false needs approach_planner (no last mile runs without it)")
        self.close_in = bool(self.cfg["close_in"])
        self.takeover = bool(self.cfg["takeover"])
        self._frame = None
        self.multiview = bool(self.cfg["multiview"])
        if self.multiview:
            if not self.approach:
                raise ValueError("verification.multiview needs verification.approach")
            if self.cfg["multiview_variant"] not in MULTIVIEW_VARIANTS:
                raise ValueError(f"verification.multiview_variant must be one of {MULTIVIEW_VARIANTS}")
            if self.cfg["multiview_sam3"] not in MULTIVIEW_SAM3:
                raise ValueError(f"verification.multiview_sam3 must be one of {MULTIVIEW_SAM3}")
            if self.cfg["multiview_failure"] not in (True, False, "aborted"):
                raise ValueError("verification.multiview_failure must be true, false or aborted")
            if not 1 <= int(self.cfg["multiview_max_views"]) <= 8:
                raise ValueError("verification.multiview_max_views must be 1..8 (server limit)")
            if int(self.cfg["multiview_bank"]) < int(self.cfg["multiview_max_views"]):
                raise ValueError("verification.multiview_bank must be >= multiview_max_views")
        self.enabled = bool(self.cfg["enabled"])
        self.log_dir = str(self.cfg["log_dir"]).format(video_output_path=video_output_path)
        self.camera = camera
        self._verify = verify or client.verify
        self._verify_views = verify_views or client.verify_views
        self.server = None
        self.selector = None
        self.consecutive_errors = 0
        self.start_episode(-1, "", "", "")
        if self.enabled:
            # A wrong URL, protocol or dead worker is a setup error, not a per-request failure.
            self.server = (ping or client.ping)(self.cfg["url"], timeout=float(self.cfg["ping_timeout_s"]))
            selector = self.selector = (self.server.get("settings") or {}).get("selector")
            # Three-way verdicts need the approach on both sides, and the approach needs them
            if self.approach != (selector in THREE_WAY_SELECTORS):
                raise ValueError(f"verification.approach={self.approach} does not match the server's "
                                 f"selector {selector!r} (approach needs --selector d1 or jev_sam3, and they need approach)")
            settings = self.server.get("settings") or {}
            if self.multiview and not settings.get("multiview"):
                raise ValueError("verification.multiview needs a server with /verify_views (settings.multiview)")
            if self.multiview and self.cfg["multiview_grid"] and not settings.get("multiview_grid"):
                raise ValueError("verification.multiview_grid needs a server started with --multiview-grid")
            os.makedirs(self.log_dir, exist_ok=True)
            print(f"[verification] gate ON: {self.cfg['url']} {json.dumps(self.server.get('settings'))}")

    # ------------------------------------------------------------------
    # Episode lifecycle
    # ------------------------------------------------------------------

    def start_episode(self, run_index, scene, episode_id, label, goal_object_ids=(), max_episode_steps=None):
        """``max_episode_steps``: the episode's step limit (an approach never plans past it)."""
        self.max_episode_steps = None if max_episode_steps is None else int(max_episode_steps)
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
        if self.multiview:
            self.counts.update({key: 0 for key in MULTIVIEW_COUNTS})
        if self.close_in:
            self.counts.update({"close_in": 0, "close_in_actions": 0, "close_in_end": None})
        if self.takeover:
            self.counts.update({"takeover": 0, "takeover_step": None, "takeover_actions": 0, "takeover_end": None,
                                "takeover_reason": None, "takeover_object_dist": None})
        self.accepted_cloud = None   # the most recent accepted object's points (close_in)
        self.approach_steps_used = 0
        self.decided_objects = []   # (points, state) of objects the gate's approaches decided (same_as_decided)
        self._sam3_cache = {}  # multiview: SAM3 scores per view (step, index), sent back with later requests

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
                camera_pitch, fusion=None, external_approach=False):
        """Store this step's target detections; returns rejected cluster ids to reopen.
        ``external_approach``: a step of an approach the gate drives itself (approach_planner): evidence only,
        no reopens and no approach views (the planner waits for the verdict)."""
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
        if self.multiview:
            self._bank_views(new)
        if external_approach:
            return []
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
            if (state["state"] not in (REJECTED, DEFERRED) or state["checks"] >= self.cfg["max_checks"]
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

    def remember_object(self, points, state):
        """Keep a decided object's points (sim world frame; LiveApproach.object_points) with its verdict, so a later
        request for the same object, under another cluster id, inherits it (same_as_decided)."""
        if state in (ACCEPTED, REJECTED) and points is not None and len(points) >= 3:
            self.decided_objects.append((np.asarray(points, np.float32), state))
            if state == ACCEPTED:
                self.accepted_cloud = np.asarray(points, np.float32)

    def log_takeover(self, step, end, entry):
        """Record the takeover (habitat_evaluation.run_takeover) in the counts (episodes.jsonl "verification"), the step
        events and takeover.jsonl: accept step, actions, end (stop / cap / episode_over), the navigator's reason
        (radius / snapped / stagnation / policy) and its last object distance, cloud source and size, start pose."""
        entry = {"run_index": self.episode.get("run_index"), "episode_id": self.episode.get("episode_id"),
                 "accept_step": int(step), "end": end, **entry}
        self.counts.update(takeover=1, takeover_step=int(step), takeover_actions=entry["actions"], takeover_end=end,
                           takeover_reason=entry.get("reason"),
                           takeover_object_dist=None if entry.get("object_dist") is None else round(float(entry["object_dist"]), 3))
        self.events.append({"event": "takeover", **entry})
        print(f"[verification] takeover at step {step}: {end} ({entry.get('reason')}) after {entry['actions']} actions, "
              f"object {entry.get('object_dist')} m")
        if self.log_dir:
            os.makedirs(self.log_dir, exist_ok=True)
            with open(os.path.join(self.log_dir, "takeover.jsonl"), "a", encoding="utf-8") as stream:
                stream.write(json.dumps(entry, default=float) + "\n")

    def log_close_in(self, reason, entry):
        """Record the terminal close-in (habitat_evaluation.run_close_in) in the counts, the step events and
        close_in.jsonl: reason (stop / stucking), start_d, end_d, actions, blocks, end."""
        entry = {"run_index": self.episode.get("run_index"), "episode_id": self.episode.get("episode_id"),
                 "reason": reason, **entry}
        self.counts.update(close_in=1, close_in_actions=entry["actions"], close_in_end=entry["end"])
        self.events.append({"event": "close_in", **entry})
        print(f"[verification] close-in ({reason}): {entry['start_d']} -> {entry['end_d']} m, {entry['actions']} "
              f"actions, {entry['end']}")
        if self.log_dir:
            os.makedirs(self.log_dir, exist_ok=True)
            with open(os.path.join(self.log_dir, "close_in.jsonl"), "a", encoding="utf-8") as stream:
                stream.write(json.dumps(entry) + "\n")

    def same_as_decided(self, centroid, frame):
        """The identity rule of the approach (approach.associate) against the decided objects: each one's points
        projected into the request frame with the depth test (local_nav.track_mask) is its outline T; the request is
        that object when one of the frame's target masks near the requested point (HANDOFF_MATCH_M) has IoU >=
        ASSOC_MIN with T. ``frame``: {depth (m), masks, centroids (planner frame), cam (position, rotation)}.
        -> (state, iou) of the most recent such object, or None."""
        if not self.decided_objects or frame is None:
            return None
        from active_approach.local_nav import track_mask
        from perception_gate.approach import associate
        c = np.asarray(centroid, float)
        masks = [np.asarray(m).squeeze() > 0 for m, p in zip(frame["masks"], frame["centroids"])
                 if p is not None and np.hypot(*(np.asarray(p[:2], float) - c)) <= HANDOFF_MATCH_M]
        if not masks:
            return None
        for points, state in reversed(self.decided_objects):
            outline = track_mask(points, frame["depth"], *frame["cam"])
            if not outline.any():
                continue
            best, scores = associate(masks, outline)
            if best is not None:
                return state, scores[best][1]
        return None

    def handle(self, request, fusion=None, step=0, frame=None):
        """``_handle`` with the request frame kept for the accepted object's points (close_in)."""
        self._frame = frame
        try:
            return self._handle(request, fusion, step, frame)
        finally:
            self._frame = None

    def _handle(self, request, fusion, step, frame):
        """Verify the requested cluster; returns 1 (accept), -1 (reject), 2 (defer) or 3 (approach;
        ``pending_approach`` has it) for the planner.

        ``request``: dict with seq, cluster_id, centroid_x/y, confidence, observation_num,
        expl_result, at_stop, and for an approach re-request approach_id, viewpoint_status,
        viewpoint_x/y and approach_steps (the VerificationRequest message). ``frame`` (the request frame, see
        same_as_decided): a request for an object a gate approach already decided inherits that verdict."""
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

        same = self.same_as_decided(centroid, frame)
        if same is not None:
            decided, iou = same
            self.counts["same_object"] = self.counts.get("same_object", 0) + 1
            record["policy"] = "same_as_accepted" if decided == ACCEPTED else "same_as_rejected"
            record["same_iou"] = round(float(iou), 3)
            return self._decide(record, decided, None, None, state)
        evidence, match = self.select_evidence(cluster_id, centroid, by_position=at_stop)
        if evidence is None:
            self.counts["no_evidence"] += 1
            record["policy"] = f"on_no_evidence={self.cfg['on_no_evidence']}"
            decided = {"accept": ACCEPTED, "reject": REJECTED, "defer": DEFERRED}[self.cfg["on_no_evidence"]]
            # at the stop there is nothing left to defer to (the planner rejects a deferral there too)
            return self._decide(record, REJECTED if decided == DEFERRED and at_stop else decided, None, None, state)
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
        if state is not None:
            state["last_verdict"] = verdict
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
        """correct -> accept, wrong -> reject, insufficient -> approach (budget permitting), keep
        approaching (a new view on the way), SAM3's raw label (on arrival), or reject (at the stop, no budget)."""
        status = self._tri_state(verdict)
        model = verdict.get("source")
        record.update(status=status, model=model)
        if model in ("sam3", "qwen", "jev"):
            self.counts[f"{model}_decisions"] += 1
        if approach is not None:
            self.counts[f"approach_{'accepted' if status == 'correct' else 'rejected' if status == 'wrong' else 'insufficient'}"] += 1
        if status == "correct":
            return self._decide(record, ACCEPTED, evidence, reply, state)
        if status == "wrong":
            return self._decide(record, REJECTED, evidence, reply, state)
        self.counts["insufficient"] += 1
        if approach is not None and approach["status"] == EARLY:
            # A new view on the way that is still insufficient: keep approaching
            approach["early"] = None
            state["approach"] = approach
            record["policy"] = "insufficient,keep_approaching"
            return self._decide(record, APPROACH, evidence, reply, state)
        if approach is not None:
            # Arrival (close range) and still insufficient: forced decision, SAM3's raw label on
            # this view (accept iff it is the target)
            accepted = verdict.get("label") == self.episode["target"]
            record["policy"] = f"insufficient,arrival_sam3_{'accept' if accepted else 'reject'}"
            return self._decide(record, ACCEPTED if accepted else REJECTED, evidence, reply, state)
        blocked = self._approach_blocked(record, state, at_stop, approach)
        if blocked is not None:
            self.counts["approach_exhausted"] += 1
            record["policy"] = f"insufficient,{blocked}"
            return self._decide(record, REJECTED, evidence, reply, state)
        self._issue_approach(record, state, evidence)
        return self._decide(record, APPROACH, evidence, reply, state)

    def _approach_remaining(self):
        return int(self.cfg["approach_episode_steps"]) - self.approach_steps_used

    def _approach_steps_allowed(self, step):
        """Actions the next approach may take: per approach, per episode budget, and the steps
        left before the episode's step limit (the last step is the forced stop)."""
        allowed = min(int(self.cfg["approach_max_steps"]), self._approach_remaining())
        if self.max_episode_steps is not None:
            allowed = min(allowed, self.max_episode_steps - int(step) - 1)
        return allowed

    def _approach_blocked(self, record, state, at_stop, approach):
        """Why an insufficient view cannot be approached (None: it can)."""
        if record["cluster"]["id"] < 0 or state is None:
            return "position"
        if at_stop:
            return "at_stop"
        if state["approaches"] >= int(self.cfg["approach_max_per_cluster"]):
            return "cluster_budget"
        if min(int(self.cfg["approach_max_steps"]), self._approach_remaining()) <= 0:
            return "episode_budget"
        if self._approach_steps_allowed(record["step"]) <= 0:
            return "episode_end"
        return None

    def _issue_approach(self, record, state, evidence):
        state["approaches"] += 1
        self.counts["moves"] += 1
        pending = {"id": state["approaches"], "issued_step": int(record["step"]),
                   "object": tuple(evidence["centroid"]), "camera": tuple(evidence["camera"]),
                   "max_steps": self._approach_steps_allowed(record["step"]), "steps": 0,
                   "early": None, "status": None}
        state["approach"] = pending
        if self.multiview:  # seed the view bank with every stored view of the object
            for item in sorted(self.evidence, key=lambda e: e["step"]):
                if self._bank_match(item, record["cluster"]["id"], pending) is not None:
                    self._bank_add(state, item)
        record["approach"] = {"id": pending["id"], "issued": True, "object": list(pending["object"]),
                              "camera": list(pending["camera"]), "max_steps": pending["max_steps"]}
        self.events.append({"event": "approach", "cluster_id": record["cluster"]["id"], "approach": pending["id"],
                            "object": list(pending["object"]), "camera": list(pending["camera"]),
                            "max_steps": pending["max_steps"]})

    def pending_approach(self, cluster_id):
        """The approach a state-3 decision issued (or keeps) for ``cluster_id``: {approach_id,
        object, camera, max_steps}; None if there is none."""
        pending = self.clusters.get(int(cluster_id), {}).get("approach")
        if pending is None:
            return None
        return {"approach_id": pending["id"], "object": pending["object"], "camera": pending["camera"],
                "max_steps": pending["max_steps"]}

    def approach_close(self, request, last):
        """approach_planner: True when the gate's planner can take the pending approach of the requested cluster
        from the last executed step ``last`` (habitat_evaluation's ctx["last_step"]): the object point is within
        HANDOFF_M of the agent and a target detection of >= HANDOFF_PX matches it (HANDOFF_MATCH_M), or its evidence
        view was taken within HANDOFF_SEEN_M of the agent (out of frame now, in view after a turn). Farther or
        out of view, ApexNav's object navigation (state 3) approaches first: the local planner only knows what
        the agent has seen since it took over."""
        pending = self.clusters.get(int(request["cluster_id"]), {}).get("approach")
        if pending is None or last is None:
            return False
        obj = np.asarray(pending["object"], float)
        here = np.asarray(last["pose"][:2], float)
        if np.hypot(*(obj - here)) > HANDOFF_M:
            return False
        if np.hypot(*(np.asarray(pending["camera"], float) - here)) <= HANDOFF_SEEN_M:
            return True   # seen from where the agent stands (e.g. during a spin): the planner turns to it
        return any(c is not None and np.hypot(*(np.asarray(c[:2], float) - obj)) <= HANDOFF_MATCH_M
                   and int((np.asarray(m).squeeze() > 0).sum()) >= HANDOFF_PX
                   for m, c in zip(last["masks"], last["centroids"]))

    def approach_handoff(self, request, last):
        """approach_planner, for a re-request of the pending approach ApexNav drives: "planner" (the gate's
        planner takes over: the object is close in view, or ApexNav arrived / gave up), "native" (keep
        approaching, nothing judged), or None (not such a re-request: ``handle`` it)."""
        approach_id = int(request.get("approach_id", 0) or 0)
        pending = self.clusters.get(int(request["cluster_id"]), {}).get("approach")
        if not approach_id or pending is None or pending["id"] != approach_id:
            return None
        status = int(request.get("viewpoint_status", 0))
        if status == EARLY and not self.approach_close(request, last):
            pending["early"] = None   # as "insufficient,keep_approaching": the next new view re-requests
            return "native"
        steps = int(request.get("approach_steps", 0) or 0)
        self.approach_steps_used += steps
        self.counts["approach_steps"] += steps
        self.counts["approach_arrived" if status == ARRIVED else "approach_early"] += 1
        print(f"[verification] cluster {request['cluster_id']}: native approach {approach_id} "
              f"{STATUS_NAMES.get(status, status)} after {steps} steps -> {self.approach_planner}")
        return "planner"

    def approach_bank(self, cluster_id):
        """The view bank of the pending approach of ``cluster_id`` (every stored detection of the object, also from
        before the request; ``multiview``): the items the gate's planner seeds its decision flow with."""
        return list(self.clusters.get(int(cluster_id), {}).get("bank", []))

    def finish_external_approach(self, request, step, result, approach_log=None):
        """The verdict of an approach the gate drove itself (perception_gate.approach): ACCEPTED, REJECTED, or
        DEFERRED when it ended without evidence (status ``undecided``: the object not found, no counted frame) or
        stuck (end ``stuck``, whatever the forced verification said: the agent is wedged, not the object wrong),
        logged like any decision (policy ``approach_planner,<by>,<accept|reject|defer>``). ApexNav drops a
        cluster that is not accepted after an external approach and keeps exploring; the re-check rule reopens a
        rejected or deferred one, its cooldown counted from the approach's end (``step``)."""
        cluster_id = int(request["cluster_id"])
        state = self.clusters.get(cluster_id)
        pending = state.get("approach") if state is not None else None
        steps = int(result.get("actions", 0))
        self.approach_steps_used += steps
        self.counts["approach_steps"] += steps
        accept = bool(result.get("accept"))
        # an approach that ended stuck brought no evidence it could act on: DEFERRED, never REJECTED
        decided = ACCEPTED if accept else DEFERRED if result.get("status") == "undecided" or \
            result.get("end") == "stuck" else REJECTED
        verb = {ACCEPTED: "accept", REJECTED: "reject", DEFERRED: "defer"}[decided]
        self.counts[{ACCEPTED: "approach_accepted", REJECTED: "approach_rejected",
                     DEFERRED: "approach_undecided"}[decided]] += 1
        self.counts["sam3_decisions"] += result.get("by") in ("belief", "forced")
        if state is not None:
            state.update(approach=None, last_step=int(step))
        record = {**{k: self.episode.get(k) for k in ("run_index", "scene", "episode_id", "label", "target")},
                  "step": int(step), "seq": int(request.get("seq", 0)),
                  "cluster": {"id": cluster_id, "centroid": [float(request["centroid_x"]), float(request["centroid_y"])],
                              "confidence": float(request.get("confidence", 0.0)),
                              "observation_num": int(request.get("observation_num", 0))},
                  "expl_result": int(request.get("expl_result", -1)), "at_stop": bool(request.get("at_stop", False)),
                  "server": None if self.server is None else self.server.get("settings"), "camera": self.camera,
                  "evidence": None, "verdict": {"status": result.get("status"), "label": None},
                  "latency_s": result.get("seconds"), "error": None, "source": "server", "ground_truth": None,
                  "status": result.get("status"), "model": "sam3",
                  "policy": f"approach_planner,{result.get('by')},{verb}",
                  "approach": {"id": None if pending is None else pending["id"], "planner": self.approach_planner,
                               "status": result.get("end") or result.get("by"), "steps": steps,
                               "views": result.get("views"), "counted": result.get("counted"),
                               "object": None if pending is None else list(pending["object"]),
                               "issued_step": None if pending is None else pending["issued_step"],
                               "log": approach_log, "episode_steps_used": self.approach_steps_used}}
        return self._decide(record, decided, None, None, state)

    def _approach_view(self, item, cluster_id, pending, used, arrival_step=None):
        """Match distance when ``item`` is a new view of the approached object: taken since the
        approach was issued, of the object point (not nearer another cluster), and closer by
        ``approach_min_closer_m`` than, or ``approach_min_angle_deg`` around the object from,
        every view judged for the cluster. Else None. ``arrival_step`` (arrival: the agent
        stands at the object, facing it): a view taken at that step, without the thresholds."""
        novel = arrival_step is None
        if item["step"] < pending["issued_step"] or (not novel and (item["step"] != arrival_step or
                                                                    item["step"] == pending["issued_step"])):
            return None
        # The issued object point, or the cluster's centroid at the view's step (a large object's
        # centroid drifts as the approach sees more of it)
        distance = _distance(item["centroid"], pending["object"])
        if cluster_id in item["clusters"]:
            distance = min(distance, _distance(item["centroid"], item["clusters"][cluster_id]))
        if distance > self.cfg["match_distance_m"] or self._closer_to_other(item, cluster_id, distance):
            return None
        if not novel:
            return distance
        rng = _distance(item["camera"], item["centroid"])
        bearing = _bearing(item["camera"], item["centroid"])
        closer, angle = float(self.cfg["approach_min_closer_m"]), math.radians(self.cfg["approach_min_angle_deg"])
        if all(rng <= u.get("range", _distance(u["camera"], pending["object"])) - closer
               or _angle_between(bearing, u["bearing"]) >= angle for u in used):
            return distance
        return None

    def _approach_views(self, step, new):
        """Clusters under approach with a new view this step (once until it is judged): the
        planner asks about it at once (viewpoint_status 4)."""
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
        """An approach re-request (viewpoint_status 1 arrived, 2 no object path / target lost,
        3 aborted, 4 new view on the way)."""
        approach_id = int(request.get("approach_id", 0))
        status = int(request.get("viewpoint_status", 0))
        reported = max(int(request.get("approach_steps", 0)), 0)  # actions so far in this approach
        pending = state.get("approach") if state is not None else None
        steps = max(reported - (pending["steps"] if pending is not None else 0), 0)
        self.approach_steps_used += steps
        self.counts["approach_steps"] += steps
        record["approach"] = {"id": approach_id, "status": STATUS_NAMES.get(status, str(status)),
                              "steps": reported, "agent": [float(request.get("viewpoint_x", 0.0)),
                                                           float(request.get("viewpoint_y", 0.0))],
                              "episode_steps_used": self.approach_steps_used}
        if state is not None:
            state["approach"] = None
        if pending is None or pending["id"] != approach_id:
            record["error"] = (f"approach {approach_id} re-request, pending "
                               f"{None if pending is None else pending['id']}")
            print(f"[verification] cluster {record['cluster']['id']}: {record['error']}")
            return self._approach_failed(record, state, "approach_mismatch")
        pending.update(steps=reported, status=status)
        record["approach"].update(object=list(pending["object"]), issued_step=pending["issued_step"])
        if self.multiview:
            return self._after_approach_multiview(record, state, step, status, pending)
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
            # Arrival: a view from this very step (the agent at the object, facing it), without
            # the new-view thresholds; none -> not re-detected
            views = [(item, d) for item in self.evidence
                     for d in [self._approach_view(item, record["cluster"]["id"], pending, state["used"],
                                                   arrival_step=int(step) if status == ARRIVED else None)]
                     if d is not None]
            if views:
                view = max(views, key=lambda v: (v[0]["step"], v[0]["quality"]))
        if view is None:
            return self._not_redetected(record, state)
        return self._judge(record, view[0], view[1], state, step, status == ARRIVED, approach=pending)

    def _not_redetected(self, record, state):
        """Arrived but the object is not re-detected (often too close, out of frame): SAM3's raw
        label on the cluster's last judged view decides (accept iff it is the target)."""
        verdict = (state or {}).get("last_verdict")
        if verdict is None:
            return self._approach_failed(record, state, "not_redetected")
        self.counts["not_redetected"] += 1
        accepted = verdict.get("label") == self.episode["target"]
        record.update(verdict=verdict, source="cached", model="sam3",
                      policy=f"not_redetected,sam3_{'accept' if accepted else 'reject'}")
        return self._decide(record, ACCEPTED if accepted else REJECTED, None, None, state)

    def _approach_failed(self, record, state, policy):
        """Unreachable, aborted (or a protocol mismatch, or not re-detected without a verdict): reject."""
        if policy in self.counts:
            self.counts[policy] += 1
        self.counts["fallback"] += 1
        record["policy"] = policy
        return self._decide(record, REJECTED, None, None, state)

    # ------------------------------------------------------------------
    # Multi-view close range (multiview)
    # ------------------------------------------------------------------

    def _bank_match(self, item, cluster_id, pending):
        """Match distance when ``item`` is a view of the approached object for the bank, else None:
        within ``match_distance_m`` of the issued object point or the cluster's centroid at the
        item's step, and no other cluster nearer it, unless that cluster lies within
        ``multiview_sibling_m`` of this one (a fragment of the same object)."""
        reference = item["clusters"].get(cluster_id)
        distance = _distance(item["centroid"], pending["object"])
        if reference is not None:
            distance = min(distance, _distance(item["centroid"], reference))
        if distance > self.cfg["match_distance_m"]:
            return None
        reference = pending["object"] if reference is None else reference
        sibling = float(self.cfg["multiview_sibling_m"])
        if any(other != cluster_id and _distance(item["centroid"], c) < distance and _distance(c, reference) > sibling
               for other, c in item["clusters"].items()):
            return None
        return distance

    def _bank_add(self, state, item):
        """Add ``item`` to the cluster's bank once; past ``multiview_bank`` views drop the least
        readable one, never the newest."""
        bank = state.setdefault("bank", [])
        if any((b["step"], b["index"]) == (item["step"], item["index"]) for b in bank):
            return
        bank.append(item)
        while len(bank) > int(self.cfg["multiview_bank"]):
            newest = max(bank, key=lambda b: (b["step"], b["index"]))
            bank.remove(min((b for b in bank if b is not newest), key=lambda b: (b["quality"], b["step"])))

    def _bank_views(self, new):
        """Bank this step's views of every object under approach (no novelty thresholds)."""
        for cluster_id, state in self.clusters.items():
            pending = state.get("approach")
            if pending is None:
                continue
            for item in new:
                if item["step"] >= pending["issued_step"] and self._bank_match(item, cluster_id, pending) is not None:
                    self._bank_add(state, item)

    def _after_approach_multiview(self, record, state, step, status, pending):
        """An approach re-request with ``multiview``: decided over the cluster's view bank."""
        bank = state.get("bank", [])
        if status in (UNREACHABLE, ABORTED):
            name = STATUS_NAMES[status]
            failure = self.cfg["multiview_failure"]
            decide = failure is True or (failure == "aborted" and status == ABORTED)
            if decide and len({b["step"] for b in bank}) >= int(self.cfg["multiview_min_views"]):
                self.counts[name] += 1
                return self._judge_views(record, state, step, "failure", pending)
            return self._approach_failed(record, state, name)
        if status not in (ARRIVED, EARLY):
            record["error"] = f"unknown viewpoint_status {status}"
            return self._approach_failed(record, state, "approach_mismatch")
        self.counts["approach_arrived" if status == ARRIVED else "approach_early"] += 1
        if status == EARLY and pending["early"] is not None:
            self._bank_add(state, pending["early"][0])
            bank = state["bank"]
        if not bank:
            return self._approach_failed(record, state, "not_redetected")
        return self._judge_views(record, state, step, "early" if status == EARLY else "arrival", pending)

    def _judge_views(self, record, state, step, mode, pending):
        """One /verify_views decision over ``select_views`` of the cluster's bank; ``mode`` is
        "early", "arrival" or "failure" (the table in the module docstring)."""
        cluster_id = record["cluster"]["id"]
        views, best = select_views(state["bank"], int(self.cfg["multiview_max_views"]),
                                   float(self.cfg["multiview_dup_distance_m"]), float(self.cfg["multiview_dup_angle_deg"]))
        goals, overlap = self.episode.get("goal_object_ids", []), self.cfg["ground_truth_semantic_overlap"]
        record["evidence"] = {"step": best["step"], "index": best["index"], "score": best["score"],
                              "area_fraction": round(best["area_fraction"], 5), "camera": list(best["camera"]),
                              "centroid": list(best["centroid"]), "pitch": best["pitch"],
                              "input_elevation": -math.degrees(best["pitch"]), "range_m": round(_range(best), 3)}
        record["ground_truth"] = semantic_goal_overlap(best["semantic_instances"], goals, overlap)
        record["views"] = [{"step": v["step"], "index": v["index"], "range_m": round(_range(v), 3),
                            "area": round(v["area_fraction"], 5), "clipped": mask_touches_border(v["mask"]),
                            "quality": round(v["quality"], 4), "score": v["score"], "camera": list(v["camera"]),
                            "centroid": list(v["centroid"]), "semantic_instances": v["semantic_instances"],
                            "ground_truth": semantic_goal_overlap(v["semantic_instances"], goals, overlap)}
                           for v in views]
        record["multiview"] = {"mode": mode, "bank": len(state["bank"]), "grid_view": views.index(best)}
        for v in views:
            state["used"].append({"camera": v["camera"], "bearing": _bearing(v["camera"], v["centroid"]),
                                  "range": _range(v)})
        sam3 = self.cfg["multiview_sam3"]
        if mode == "early" and sam3 == "fallback":
            sam3 = "never"  # an insufficient early view keeps approaching; SAM3 is not needed yet
        variant = self.cfg["multiview_variant"]
        forced = {}
        if self.selector == "jev_sam3":  # forced at close range (arrival, failure); SAM3 decides (fused belief)
            forced = {"forced": mode != "early"}
        payload = [{"rgb": v["rgb"], "mask": v["mask"], "input_elevation": -math.degrees(v["pitch"]),
                    "step": v["step"], "index": v["index"], "range_m": _range(v),
                    "scores": self._sam3_cache.get((v["step"], v["index"]))} for v in views]
        self.counts["multiview_requests"] += 1
        self.counts["multiview_views"] += len(views)
        start = time.time()
        try:
            reply = self._verify_views(
                self.cfg["url"], payload, self.episode["target"], sam3=sam3,
                request_id=f"epi{self.episode.get('run_index')}_c{cluster_id}_s{step}_mv",
                timeout=float(self.cfg["timeout_s"]), grid=bool(self.cfg["multiview_grid"]),
                grid_view=views.index(best), crop_small=variant == "crop_small", forced2=variant == "forced2",
                forced2_min=float(self.cfg["multiview_forced2_min"]),
                sam3_shortcut=bool(self.cfg["multiview_sam3_shortcut"]),
                grid_scores=bool(self.cfg["multiview_grid_scores"]), **forced)
            verdict = reply["verdict"]
            status = verdict["status"]
            if status not in TRI_STATES:
                raise RuntimeError(f"multiview verdict without a three-way status: {status!r}")
        except Exception as error:  # as _judge: on_error (reject with the approach), always logged
            self.consecutive_errors += 1
            self.counts["errors"] += 1
            record.update(error=f"{type(error).__name__}: {error}", policy=f"multiview,{mode},on_error={self.cfg['on_error']}",
                          latency_s=round(time.time() - start, 3))
            decided = self._decide(record, ACCEPTED if self.cfg["on_error"] == "accept" else REJECTED, best, None,
                                   state, views)
            if self.consecutive_errors >= self.cfg["max_consecutive_errors"]:
                raise RuntimeError(f"perception server failed {self.consecutive_errors} times in a row "
                                   f"(last: {record['error']}); aborting rather than running ungated")
            return decided
        self.consecutive_errors = 0
        record.update(verdict=verdict, latency_s=round(time.time() - start, 3), status=status)
        for view in verdict.get("per_view") or []:
            if view.get("scores") is not None and view.get("step") is not None:
                self._sam3_cache[(int(view["step"]), int(view["index"]))] = view["scores"]
        if (verdict.get("qwen") or {}).get("reconstruct_error"):
            self.counts["multiview_grid_errors"] += 1
        self.counts[f"approach_{'accepted' if status == 'correct' else 'rejected' if status == 'wrong' else 'insufficient'}"] += 1
        vlm = "jev" if verdict.get("source") == "jev" else "qwen"  # who answered the views
        if status in ("correct", "wrong"):
            shortcut = verdict.get("source") == "sam3"
            record["model"] = "sam3" if shortcut else f"{vlm}_multi"
            self.counts["sam3_decisions" if shortcut else f"{vlm}_decisions"] += 1
            self.counts["multiview_sam3" if shortcut else f"multiview_{vlm}"] += 1
            if mode == "failure":
                self.counts["multiview_failure_decided"] += 1
            decider = ("sam3" if self.selector == "jev_sam3" else "sam3_shortcut") if shortcut else vlm
            record["policy"] = (f"multiview,{mode},{decider}_"
                                f"{'accept' if status == 'correct' else 'reject'}")
            return self._decide(record, ACCEPTED if status == "correct" else REJECTED, best, reply, state, views)
        self.counts["insufficient"] += 1
        if mode == "early":  # still insufficient on the way: keep approaching
            pending["early"] = None
            state["approach"] = pending
            record.update(model=f"{vlm}_multi", policy="multiview,early,insufficient,keep_approaching")
            return self._decide(record, APPROACH, best, reply, state, views)
        aggregate = verdict.get("aggregate") or {}
        margin, vote, used = aggregate.get("margin"), aggregate.get("vote"), aggregate.get("n_used", 0)
        threshold = float(self.cfg["multiview_arrival_margin" if mode == "arrival" else "multiview_failure_margin"])
        accepted = margin is not None and margin >= threshold and vote >= float(self.cfg["multiview_vote"])
        record["model"] = "sam3_multi"
        self.counts["sam3_decisions"] += 1
        self.counts["multiview_sam3"] += 1
        if mode == "failure":
            self.counts["multiview_failure_decided"] += 1
        summary = "no_sam3_match" if margin is None else f"M={margin:.2f},V={vote:.2f},n={used}"
        record["policy"] = f"multiview,{mode},insufficient,sam3_{'accept' if accepted else 'reject'}({summary})"
        return self._decide(record, ACCEPTED if accepted else REJECTED, best, reply, state, views)

    def defers(self, verdict):
        """True when ``verdict`` is a SAM3D view rejecting what the raw frame accepted, and
        such rejections defer."""
        return (self.cfg["on_view_reject"] == "defer" and verdict.get("judged") == "view"
                and verdict.get("input_scores") is not None)

    def _decide(self, record, decided, evidence, reply, state, views=None):
        if state is not None:
            state["state"] = decided
        record["state"] = decided
        if decided != APPROACH:  # an approach is counted as a move when issued
            self.counts[NAMES[decided].lower()] += 1
        if decided == ACCEPTED and self.close_in and self._frame is not None and record["policy"] != "same_as_accepted":
            from perception_gate import close_in
            points = close_in.object_points(self._frame, record["cluster"]["centroid"], HANDOFF_MATCH_M)
            if points is not None:   # (a gate approach's own points replace these: remember_object)
                self.accepted_cloud = points
        self._log(record, evidence, reply, views)
        label = (record["verdict"] or {}).get("label")
        self.events.append({"event": "decision", "cluster_id": record["cluster"]["id"], "state": decided,
                            "label": label, "source": record["source"], "error": record["error"]})
        print(f"[verification] {'position' if record['cluster']['id'] < 0 else 'cluster'} "
              f"{record['cluster']['id']} {NAMES[decided]}: label {label!r} "
              f"for {self.episode.get('target')!r} ({record['policy'] or record['source']}, "
              f"{record['latency_s'] or 0.0:.1f} s)")
        return decided

    def _log(self, record, evidence, reply, views=None):
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
        if reply and reply.get("grid_image"):  # multiview: the SAM3D grid Qwen saw
            cv2.imwrite(os.path.join(folder, "grid.jpg"),
                        cv2.cvtColor(client.decode_png(reply["grid_image"]), cv2.COLOR_RGB2BGR))
        if views:  # multiview: every view sent, nearest first
            os.makedirs(os.path.join(folder, "views"), exist_ok=True)
            for k, item in enumerate(views, start=1):
                name = os.path.join(folder, "views", f"v{k}_s{item['step']}_{item['index']}")
                cv2.imwrite(name + ".jpg", cv2.cvtColor(item["rgb"], cv2.COLOR_RGB2BGR))
                cv2.imwrite(name + "_mask.png", item["mask"] * 255)
        with open(os.path.join(folder, "decision.json"), "w", encoding="utf-8") as stream:
            json.dump(record, stream, indent=1)
