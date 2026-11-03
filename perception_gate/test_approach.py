"""VerificationGate approach logic (server selector d1) with a fake server (no ROS, no GPU):

    python -m unittest perception_gate.test_approach
"""
import json
import math
import os
import tempfile
import unittest

import numpy as np

from perception_gate.gate import (ACCEPTED, APPROACH, REJECTED, UNVERIFIED, VerificationGate)

RGB = np.full((48, 64, 3), 120, dtype=np.uint8)
OBJECT = (2.0, 0.0)


def mask(size):
    m = np.zeros((48, 64), dtype=np.uint8)
    m[24 - size // 2:24 + size // 2, 32 - size // 2:32 + size // 2] = 1
    return m


def request(cluster_id=3, centroid=OBJECT, seq=1, at_stop=False, approach_id=0, status=0, steps=0):
    return {"seq": seq, "cluster_id": cluster_id, "centroid_x": centroid[0], "centroid_y": centroid[1],
            "confidence": 0.6, "observation_num": 3, "expl_result": 1, "at_stop": at_stop,
            "approach_id": approach_id, "viewpoint_status": status, "viewpoint_x": 0.0, "viewpoint_y": 0.0,
            "approach_steps": steps}


def decisions(folder):
    with open(os.path.join(folder, "v", "decisions.jsonl"), encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def d1_verdict(status, source="qwen"):
    return {"accepted": status == "correct", "label": "potted plant", "status": status, "source": source,
            "gate": {"score": 0.6, "threshold": 0.546, "insufficient": source == "qwen"}, "qwen": None}


class ApproachTests(unittest.TestCase):
    def setUp(self):
        self.calls, self.statuses = [], []
        self.folder = tempfile.mkdtemp()

        def verify(url, rgb, m, target, input_elevation=0.0, request_id=None, timeout=None):
            self.calls.append(request_id)
            return {"ok": True, "view_image": None, "verdict": d1_verdict(self.statuses.pop(0))}

        self.gate = VerificationGate({"enabled": True, "approach": True, "on_error": "reject",
                                      "log_dir": self.folder + "/{video_output_path}"}, "v",
                                     verify=verify, ping=lambda url, timeout: {"ok": True,
                                                                              "settings": {"selector": "d1"}})
        self.gate.start_episode(7, "scene", "ep", "potted_plant", goal_object_ids=["42"], max_episode_steps=500)

    def observe(self, step, camera, centroid=OBJECT, size=16, fusion=None):
        return self.gate.observe(step, RGB, [mask(size)], [0], [0.5], [centroid],
                                 [[{"id": "42", "pixels": 10}]], (*camera, 0.0), -math.radians(30), fusion)

    def insufficient_then(self, *statuses):
        self.statuses += ["insufficient", *statuses]

    # ------------------------------------------------------------------

    def test_verdict_mapping(self):
        for cluster_id, status, expected in ((3, "correct", ACCEPTED), (4, "wrong", REJECTED),
                                             (5, "insufficient", APPROACH)):
            centroid = (2.0 + 3 * cluster_id, 0.0)
            self.observe(cluster_id, (centroid[0] - 2.0, 0.0), centroid)
            self.statuses.append(status)
            self.assertEqual(self.gate.handle(request(cluster_id, centroid, seq=cluster_id), step=cluster_id),
                             expected)
        approach = self.gate.pending_approach(5)
        self.assertEqual(approach, {"approach_id": 1, "object": (17.0, 0.0), "camera": (15.0, 0.0),
                                    "min_range": 2.0, "max_steps": 40, "side_only": False})
        self.assertIsNone(self.gate.pending_approach(3))
        counts = self.gate.finish_episode()
        self.assertEqual((counts["accepted"], counts["rejected"], counts["moves"], counts["insufficient"],
                          counts["qwen_decisions"]), (1, 1, 1, 1, 3))
        line = decisions(self.folder)[-1]
        self.assertEqual((line["state"], line["status"], line["model"], line["approach"]["id"]),
                         (APPROACH, "insufficient", "qwen", 1))

    def test_early_view_and_accept(self):
        self.observe(1, (0.0, 0.0))
        self.insufficient_then("correct")
        self.assertEqual(self.gate.handle(request(), step=1), APPROACH)
        self.assertEqual(self.observe(2, (0.3, 0.0)), [])        # 0.3 m closer, same bearing: not new
        self.assertEqual(self.observe(3, (1.0, 0.0)), [3])       # 1.0 m closer: new -> early
        self.assertEqual(self.observe(4, (1.2, 0.0)), [])        # once per approach
        # Approach re-request: no cooldown, no max_checks, no cache; judged on the early view
        self.assertEqual(self.gate.handle(request(seq=2, approach_id=1, status=4, steps=2), step=3), ACCEPTED)
        self.assertEqual(self.gate.clusters[3]["used"][-1]["camera"], (1.0, 0.0))
        self.assertEqual(self.gate.clusters[3]["checks"], 1)
        counts = self.gate.finish_episode()
        self.assertEqual((counts["approach_early"], counts["approach_accepted"], counts["approach_steps"]),
                         (1, 1, 2))

    def test_side_view_counts_as_new(self):
        self.observe(1, (0.0, 0.0))
        self.insufficient_then("wrong")
        self.gate.handle(request(), step=1)
        self.assertEqual(self.observe(2, (0.5, 1.2)), [3])       # ~39 deg around the object
        self.assertEqual(self.gate.handle(request(seq=2, approach_id=1, status=4, steps=1), step=2), REJECTED)

    def test_arrival_uses_newest_new_view(self):
        self.observe(1, (0.0, 0.0))
        self.insufficient_then("correct")
        self.gate.handle(request(), step=1)
        self.gate.clusters[3]["approach"]["early"] = "ignored"  # e.g. the planner arrived the same step
        self.observe(5, (1.0, 0.0), size=10)
        self.observe(6, (1.1, 0.0), size=8)
        self.assertEqual(self.gate.handle(request(seq=2, approach_id=1, status=1, steps=6), step=6), ACCEPTED)
        self.assertEqual(self.gate.clusters[3]["used"][-1]["camera"], (1.1, 0.0))

    def test_arrival_has_no_new_view_threshold(self):
        # The viewpoint counts as reached up to 0.45 m off: an arrival is judged on the newest
        # view since the approach began, even one that is not 0.5 m closer
        self.observe(1, (0.0, 0.0))
        self.insufficient_then("correct")
        self.gate.handle(request(), step=1)
        self.assertEqual(self.observe(4, (0.3, 0.0)), [])  # not new: no early end
        self.assertEqual(self.gate.handle(request(seq=2, approach_id=1, status=1, steps=3), step=4), ACCEPTED)
        self.assertEqual(self.gate.clusters[3]["used"][-1]["camera"], (0.3, 0.0))
        # ... but an early end still needs a new view
        self.gate.clusters[3]["approach"] = {"id": 9, "issued_step": 4, "object": OBJECT, "camera": (0.3, 0.0),
                                             "min_range": 1.7, "max_steps": 40, "side_only": False,
                                             "at_stop": False, "early": None}
        self.assertEqual(self.observe(5, (0.5, 0.0)), [])
        self.assertEqual(self.gate.handle(request(seq=3, approach_id=9, status=4, steps=1), step=5), REJECTED)
        self.assertEqual(self.gate.finish_episode()["not_redetected"], 1)

    def test_min_range_over_all_judged_views(self):
        self.gate.clusters[3] = {"state": 0, "checks": 1, "last_step": 0, "centroid": OBJECT, "approaches": 0,
                                 "approach": None, "used": [{"camera": (1.0, 0.0), "bearing": 0.0, "range": 1.0}]}
        self.observe(2, (-2.0, 0.0), size=30)  # a new view from 4 m (by the re-check rule)
        self.insufficient_then()
        self.assertEqual(self.gate.handle(request(), step=2), APPROACH)
        approach = self.gate.pending_approach(3)
        self.assertEqual((approach["camera"], approach["min_range"]), ((-2.0, 0.0), 1.0))

    def test_max_steps_capped_by_episode_end(self):
        self.observe(480, (0.0, 0.0))
        self.insufficient_then()
        self.gate.handle(request(), step=480)
        self.assertEqual(self.gate.pending_approach(3)["max_steps"], 19)  # steps 481..499; 500 is the stop
        self.observe(499, (-1.0, 1.0), centroid=(5.0, 0.0))
        self.insufficient_then()
        self.assertEqual(self.gate.handle(request(4, (5.0, 0.0), seq=2), step=499), REJECTED)
        self.assertEqual(decisions(self.folder)[-1]["policy"], "insufficient,episode_end")

    def test_cache_bypass(self):
        # A view judged for a stop position is cached; the approach must not reuse that verdict
        self.observe(1, (0.0, 0.0))
        self.insufficient_then()
        self.gate.handle(request(), step=1)
        self.observe(2, (1.0, 0.0), size=20)
        self.statuses.append("wrong")
        self.gate.handle(request(cluster_id=-1, centroid=OBJECT, seq=2, at_stop=True), step=2)
        self.statuses.append("correct")
        self.assertEqual(self.gate.handle(request(seq=3, approach_id=1, status=1, steps=1), step=2), ACCEPTED)
        self.assertEqual(len(self.calls), 3)
        self.assertEqual(self.gate.finish_episode()["cached"], 0)

    def test_failures_reject(self):
        for cluster_id, status in ((3, 2), (4, 3), (5, 1)):  # unreachable, aborted, arrived without a new view
            centroid = (2.0 + 5 * cluster_id, 0.0)
            self.observe(10 * cluster_id, (centroid[0] - 2.0, 0.0), centroid)
            self.insufficient_then()
            self.assertEqual(self.gate.handle(request(cluster_id, centroid, seq=cluster_id), step=10 * cluster_id),
                             APPROACH)
            if status != 1:  # (an arrival with no view since the approach began is not re-detected)
                self.observe(10 * cluster_id + 1, (centroid[0] - 1.9, 0.0), centroid)  # not new
            self.assertEqual(self.gate.handle(request(cluster_id, centroid, seq=20 + cluster_id, approach_id=1,
                                                      status=status, steps=1), step=10 * cluster_id + 1), REJECTED)
            self.assertIsNone(self.gate.pending_approach(cluster_id))
        # A re-request for an approach that was never issued rejects too
        self.assertEqual(self.gate.handle(request(9, (40.0, 0.0), seq=40, approach_id=2, status=1), step=60),
                         REJECTED)
        counts = self.gate.finish_episode()
        self.assertEqual((counts["unreachable"], counts["aborted"], counts["not_redetected"], counts["fallback"],
                          counts["rejected"]), (1, 1, 1, 4, 4))
        self.assertEqual(len(self.calls), 3)

    def test_rejected_after_approach_follows_reopen_rule(self):
        self.observe(1, (0.0, 0.0))
        self.insufficient_then()
        self.gate.handle(request(), step=1)
        self.assertEqual(self.gate.handle(request(seq=2, approach_id=1, status=2), step=1), REJECTED)
        self.assertEqual(self.observe(10, (0.0, 3.0)), [])       # cooldown
        self.assertEqual(self.observe(30, (1.0, 1.0)), [3])      # cooldown over, new view
        self.assertEqual(self.gate.clusters[3]["state"], UNVERIFIED)

    def test_cluster_budget(self):
        self.observe(1, (0.0, 0.0))
        self.statuses += ["insufficient"] * 3
        self.assertEqual(self.gate.handle(request(), step=1), APPROACH)
        self.observe(2, (1.0, 0.0))
        self.assertEqual(self.gate.handle(request(seq=2, approach_id=1, status=1, steps=3), step=2), APPROACH)
        self.assertEqual(self.gate.pending_approach(3)["approach_id"], 2)
        self.observe(3, (1.6, 0.5))
        self.assertEqual(self.gate.handle(request(seq=3, approach_id=2, status=1, steps=3), step=3), REJECTED)
        counts = self.gate.finish_episode()
        self.assertEqual((counts["moves"], counts["approach_exhausted"], counts["approach_insufficient"]), (2, 1, 2))

    def test_episode_budget(self):
        self.gate.approach_steps_used = 100
        self.observe(1, (0.0, 0.0))
        self.insufficient_then()
        self.gate.handle(request(), step=1)
        self.assertEqual(self.gate.pending_approach(3)["max_steps"], 20)
        self.assertEqual(self.gate.handle(request(seq=2, approach_id=1, status=3, steps=20), step=21), REJECTED)
        self.observe(30, (6.0, 0.0), centroid=(8.0, 0.0))
        self.insufficient_then()
        self.assertEqual(self.gate.handle(request(4, (8.0, 0.0), seq=3), step=30), REJECTED)
        line = decisions(self.folder)[-1]
        self.assertEqual(line["policy"], "insufficient,episode_budget")

    def test_stop_never_stops_on_insufficient(self):
        # At the stop: one side-view approach, then insufficient rejects
        self.observe(1, (0.0, 0.0))
        self.statuses += ["insufficient", "insufficient"]
        self.assertEqual(self.gate.handle(request(at_stop=True), step=1), APPROACH)
        self.assertTrue(self.gate.pending_approach(3)["side_only"])
        self.observe(2, (0.5, 1.2))
        self.assertEqual(self.gate.handle(request(seq=2, approach_id=1, status=1, steps=4, at_stop=False), step=2),
                         REJECTED)
        # A stop position without a cluster never approaches
        self.observe(3, (4.0, 4.0), centroid=(6.0, 4.0))
        self.statuses.append("insufficient")
        self.assertEqual(self.gate.handle(request(-1, (6.0, 4.0), seq=3, at_stop=True), step=3), REJECTED)
        # Nor does a cluster whose budget is spent
        self.gate.clusters[3]["approaches"] = 2
        self.observe(4, (1.0, -1.0))
        self.statuses.append("insufficient")
        self.assertEqual(self.gate.handle(request(seq=4, at_stop=True), step=4), REJECTED)

    def test_stale_approach_dropped_by_ordinary_request(self):
        self.observe(1, (0.0, 0.0))
        self.statuses += ["insufficient", "correct"]
        self.gate.handle(request(), step=1)
        self.observe(2, (1.0, 0.0))
        self.assertEqual(self.gate.handle(request(seq=2), step=2), ACCEPTED)
        self.assertIsNone(self.gate.pending_approach(3))

    def test_selector_must_match(self):
        with self.assertRaises(ValueError):
            VerificationGate({"enabled": True, "approach": True, "on_error": "reject", "log_dir": self.folder},
                             "v", ping=lambda url, timeout: {"ok": True, "settings": {"selector": "input"}})
        with self.assertRaises(ValueError):  # every approach failure rejects, server errors too
            VerificationGate({"enabled": True, "approach": True, "log_dir": self.folder}, "v",
                             ping=lambda url, timeout: {"ok": True, "settings": {"selector": "d1"}})
        with self.assertRaises(ValueError):
            VerificationGate({"enabled": True, "log_dir": self.folder}, "v",
                             ping=lambda url, timeout: {"ok": True, "settings": {"selector": "d1"}})

    def test_binary_runs_keep_their_counts(self):
        gate = VerificationGate({"enabled": True, "log_dir": self.folder}, "v",
                                ping=lambda url, timeout: {"ok": True, "settings": {"selector": "input"}})
        self.assertEqual(set(gate.finish_episode()), {"requests", "accepted", "rejected", "deferred", "errors",
                                                      "no_evidence", "cached", "reopened"})
        # ... and a binary verdict never yields an approach (no status field)
        gate._verify = lambda *a, **k: {"ok": True, "verdict": {"accepted": False, "label": "lamp",
                                                                "status": "insufficient"}}
        gate.observe(1, RGB, [mask(16)], [0], [0.5], [OBJECT], None, (0.0, 0.0, 0.0), 0.0)
        self.assertEqual(gate.handle(request(), step=1), REJECTED)
        self.assertEqual(gate.observe(2, RGB, [mask(16)], [0], [0.5], [OBJECT], None, (1.0, 0.0, 0.0), 0.0), [])


if __name__ == "__main__":
    unittest.main()
