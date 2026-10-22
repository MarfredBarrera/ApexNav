"""VerificationGate: the Python half of the perception gate (no ROS imports).

Per episode:
    gate.start_episode(run_index, scene, episode_id, label, goal_object_ids)
    reopen = gate.observe(step, rgb, masks, labels, scores, centroids, semantic_instances,
                          pose, camera_pitch, fusion)        # after every sim step
    state = gate.handle(request, fusion, step)               # on /object/verification_request
    counts = gate.finish_episode()

``observe`` buffers every target-class (label 0) detection as evidence and keeps
the ``max_evidence`` most readable views (``_view_quality``). ``handle`` verifies
the best view of the requested cluster with ``perception.Verifier`` over HTTP
and returns the cluster state for the planner: 1 accepted, -1 rejected.

Failure policy (fail open by default): a server error or timeout applies
``on_error``, a cluster without a stored view ``on_no_evidence``; both are logged.

Re-check: a rejected cluster is reopened (``observe`` returns its id; the caller
publishes state 0, so the planner's next selection of it requests a new check)
once all hold: ``recheck_cooldown_steps`` steps since its last check, a new view
of it this step whose camera is ``recheck_min_distance_m`` from every view used
before or looks at it from a direction ``recheck_min_angle_deg`` apart, and
fewer than ``max_checks`` checks so far.

Logs: ``<log_dir>/decisions.jsonl`` (one line per decision, with the server
settings and the semantic-instance ground truth) and, with ``save_images``,
``<log_dir>/epi<run>_<label>/c<id>_s<step>/{input.jpg, mask.png, view.png, decision.json}``.
"""
import json
import math
import os
import time

import cv2
import numpy as np

from basic_utils.eval.habitat_utils import semantic_goal_overlap, slugify_label
from basic_utils.record_episode.episode_visualization import (
    COMMIT_MATCH_DISTANCE_M,
    _mask_touches_border,
    _view_quality,
)
from perception_gate import client

DEFAULTS = {
    "enabled": False,
    "url": "http://3dawareverification-dev-1:8770",
    "timeout_s": 120.0,
    "on_error": "accept",
    "on_no_evidence": "accept",
    "max_checks": 3,
    "recheck_cooldown_steps": 25,
    "recheck_min_distance_m": 1.0,
    "recheck_min_angle_deg": 45.0,
    "match_distance_m": COMMIT_MATCH_DISTANCE_M,
    "log_dir": "{video_output_path}/verification",
    "save_images": True,
    "max_evidence": 64,
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
    def __init__(self, cfg=None, video_output_path=".", verify=None, ping=None):
        unknown = set(cfg or {}) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"unknown verification settings: {sorted(unknown)}")
        self.cfg = {**DEFAULTS, **{k: v for k, v in dict(cfg or {}).items() if v is not None}}
        for key in ("on_error", "on_no_evidence"):
            if self.cfg[key] not in POLICIES:
                raise ValueError(f"verification.{key} must be one of {POLICIES}")
        self.enabled = bool(self.cfg["enabled"])
        self.log_dir = str(self.cfg["log_dir"]).format(video_output_path=video_output_path)
        self._verify = verify or client.verify
        self.server = None
        self.episode = {}
        self.evidence, self.clusters, self.events, self.counts = [], {}, [], {}
        if self.enabled:
            # A wrong URL or protocol is a setup error, not a per-request failure.
            self.server = (ping or client.ping)(self.cfg["url"])
            os.makedirs(self.log_dir, exist_ok=True)
            print(f"[verification] gate ON: {self.cfg['url']} {json.dumps(self.server.get('settings'))}")

    # ------------------------------------------------------------------
    # Episode lifecycle
    # ------------------------------------------------------------------

    def start_episode(self, run_index, scene, episode_id, label, goal_object_ids=()):
        self.episode = {"run_index": int(run_index), "scene": str(scene), "episode_id": str(episode_id),
                        "label": str(label), "target": target_name(label),
                        "goal_object_ids": [str(i) for i in goal_object_ids if i is not None]}
        self.evidence, self.clusters, self.events = [], {}, []
        self.counts = {"requests": 0, "accepted": 0, "rejected": 0, "errors": 0, "no_evidence": 0,
                       "reopened": 0}

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
        """Buffer this step's target detections; returns rejected cluster ids to reopen."""
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
                    "quality": _view_quality(score, area / float(mask.size), _mask_touches_border(mask)),
                    "centroid": (float(centroids[index][0]), float(centroids[index][1])),
                    "camera": (float(pose[0]), float(pose[1])), "pitch": float(camera_pitch),
                    "semantic_instances": (semantic_instances[index] if semantic_instances is not None
                                           and index < len(semantic_instances) else None),
                    "clusters": snapshot}
            new.append(item)
        self.evidence.extend(new)
        if len(self.evidence) > self.cfg["max_evidence"]:
            self.evidence.sort(key=lambda e: -e["quality"])
            del self.evidence[self.cfg["max_evidence"]:]
        return self._reopen(step, new, snapshot)

    def _novel(self, item, used):
        """True when ``item`` sees its object from far enough, or a different enough direction,
        compared with every view used before."""
        bearing = _bearing(item["camera"], item["centroid"])
        return all(_distance(item["camera"], u["camera"]) >= self.cfg["recheck_min_distance_m"]
                   or math.degrees(_angle_between(bearing, u["bearing"])) >= self.cfg["recheck_min_angle_deg"]
                   for u in used)

    def _matches(self, item, cluster_id, centroid):
        reference = item["clusters"].get(cluster_id, centroid)
        return reference is not None and _distance(item["centroid"], reference) <= self.cfg["match_distance_m"]

    def _reopen(self, step, new, snapshot):
        reopened = []
        for cluster_id, state in self.clusters.items():
            if (state["state"] != REJECTED or state["checks"] >= self.cfg["max_checks"]
                    or step - state["last_step"] < self.cfg["recheck_cooldown_steps"]):
                continue
            centroid = snapshot.get(cluster_id, state["centroid"])
            if any(self._matches(item, cluster_id, centroid) and self._novel(item, state["used"]) for item in new):
                state["state"] = UNVERIFIED
                reopened.append(cluster_id)
                self.counts["reopened"] += 1
                self.events.append({"event": "reopen", "cluster_id": cluster_id, "step": int(step)})
                print(f"[verification] cluster {cluster_id} reopened at step {step} (new view)")
        return reopened

    def select_evidence(self, cluster_id, centroid):
        """(evidence, match distance) for a cluster: the most readable view within
        ``match_distance_m``, else the nearest within twice that, else (None, None).
        A re-check prefers views that are new by the re-check rule."""
        used = self.clusters.get(cluster_id, {}).get("used", [])
        pool = self.evidence
        if used:
            pool = [e for e in self.evidence if self._novel(e, used)] or self.evidence

        def distance(item):
            reference = item["clusters"].get(cluster_id, centroid)
            return _distance(item["centroid"], reference)

        matched = [e for e in pool if distance(e) <= self.cfg["match_distance_m"]]
        if matched:
            best = max(matched, key=lambda e: e["quality"])
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
        expl_result (the VerificationRequest message)."""
        cluster_id = int(request["cluster_id"])
        centroid = (float(request["centroid_x"]), float(request["centroid_y"]))
        state = self.clusters.setdefault(cluster_id, {"state": UNVERIFIED, "checks": 0, "last_step": step,
                                                      "used": [], "centroid": centroid})
        state.update(checks=state["checks"] + 1, last_step=int(step), centroid=centroid)
        self.counts["requests"] += 1
        evidence, match = self.select_evidence(cluster_id, centroid)
        record = {**{k: self.episode.get(k) for k in ("run_index", "scene", "episode_id", "label", "target")},
                  "step": int(step), "seq": int(request.get("seq", 0)),
                  "cluster": {"id": cluster_id, "centroid": list(centroid),
                              "confidence": float(request.get("confidence", 0.0)),
                              "observation_num": int(request.get("observation_num", 0)),
                              "check": state["checks"]},
                  "expl_result": int(request.get("expl_result", -1)),
                  "server": None if self.server is None else self.server.get("settings"),
                  "evidence": None, "verdict": None, "latency_s": None, "error": None, "policy": None,
                  "ground_truth": None}
        reply, start = None, time.time()
        if evidence is None:
            accepted = self.cfg["on_no_evidence"] == "accept"
            record["policy"] = f"on_no_evidence={self.cfg['on_no_evidence']}"
            self.counts["no_evidence"] += 1
        else:
            elevation = -math.degrees(evidence["pitch"])  # Habitat pitch is negative looking down
            record["evidence"] = {"step": evidence["step"], "index": evidence["index"], "score": evidence["score"],
                                  "area_fraction": round(evidence["area_fraction"], 5),
                                  "match_distance_m": round(match, 3), "camera": list(evidence["camera"]),
                                  "centroid": list(evidence["centroid"]), "pitch": evidence["pitch"],
                                  "input_elevation": elevation}
            record["ground_truth"] = semantic_goal_overlap(
                evidence["semantic_instances"], self.episode.get("goal_object_ids", []),
                self.cfg["ground_truth_semantic_overlap"])
            try:
                reply = self._verify(self.cfg["url"], evidence["rgb"], evidence["mask"], self.episode["target"],
                                     input_elevation=elevation,
                                     request_id=f"epi{self.episode.get('run_index')}_c{cluster_id}_s{step}",
                                     timeout=float(self.cfg["timeout_s"]))
                record["verdict"] = reply["verdict"]
                accepted = bool(reply["verdict"]["accepted"])
            except Exception as error:  # fail open (or closed) per on_error, always logged
                accepted = self.cfg["on_error"] == "accept"
                record["error"] = f"{type(error).__name__}: {error}"
                record["policy"] = f"on_error={self.cfg['on_error']}"
                self.counts["errors"] += 1
            state["used"].append({"camera": evidence["camera"],
                                  "bearing": _bearing(evidence["camera"], evidence["centroid"])})
        record["latency_s"] = round(time.time() - start, 3)
        state["state"] = ACCEPTED if accepted else REJECTED
        record["state"] = state["state"]
        self.counts["accepted" if accepted else "rejected"] += 1
        self._log(record, evidence, reply)
        self.events.append({"event": "decision", "cluster_id": cluster_id, "state": state["state"],
                            "label": (record["verdict"] or {}).get("label"), "error": record["error"]})
        print(f"[verification] cluster {cluster_id} {'ACCEPTED' if accepted else 'REJECTED'}: "
              f"label {(record['verdict'] or {}).get('label')!r} for {self.episode.get('target')!r} "
              f"({record['policy'] or 'verdict'}, {record['latency_s']:.1f} s)")
        return state["state"]

    def _log(self, record, evidence, reply):
        if not self.log_dir:
            return
        os.makedirs(self.log_dir, exist_ok=True)
        with open(os.path.join(self.log_dir, "decisions.jsonl"), "a", encoding="utf-8") as stream:
            stream.write(json.dumps(record) + "\n")
        if not self.cfg["save_images"]:
            return
        folder = os.path.join(self.log_dir, f"epi{record['run_index']}_{slugify_label(record['label'])}",
                              f"c{record['cluster']['id']}_s{record['step']}")
        os.makedirs(folder, exist_ok=True)
        if evidence is not None:
            cv2.imwrite(os.path.join(folder, "input.jpg"), cv2.cvtColor(evidence["rgb"], cv2.COLOR_RGB2BGR))
            cv2.imwrite(os.path.join(folder, "mask.png"), evidence["mask"] * 255)
        if reply and reply.get("view_image"):
            cv2.imwrite(os.path.join(folder, "view.png"),
                        cv2.cvtColor(client.decode_png(reply["view_image"]), cv2.COLOR_RGB2BGR))
        with open(os.path.join(folder, "decision.json"), "w", encoding="utf-8") as stream:
            json.dump(record, stream, indent=1)
