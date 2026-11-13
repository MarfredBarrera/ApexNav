"""VerificationGate logic with a fake server (no ROS, no GPU):

    python -m unittest perception_gate.test_gate
"""
import json
import math
import os
import tempfile
import unittest

import numpy as np

from perception_gate.gate import DEFERRED, REJECTED, UNVERIFIED, VerificationGate, target_name

RGB = np.full((48, 64, 3), 120, dtype=np.uint8)


def mask(size):
    """A centred square target mask of side ``size`` px."""
    m = np.zeros((48, 64), dtype=np.uint8)
    m[24 - size // 2:24 + size // 2, 32 - size // 2:32 + size // 2] = 1
    return m


def request(cluster_id=3, centroid=(2.0, 0.0), seq=1, at_stop=False):
    return {"seq": seq, "cluster_id": cluster_id, "centroid_x": centroid[0], "centroid_y": centroid[1],
            "confidence": 0.6, "observation_num": 3, "expl_result": 1, "at_stop": at_stop}


class GateTests(unittest.TestCase):
    def setUp(self):
        self.calls, self.accept, self.server_down = [], False, False
        self.folder = tempfile.mkdtemp()

        def verify(url, rgb, m, target, input_elevation=0.0, request_id=None, timeout=None):
            self.calls.append({"target": target, "area": int(m.sum()), "elevation": input_elevation,
                               "request_id": request_id})
            if self.server_down:
                raise TimeoutError("timed out")
            return {"ok": True, "view_image": None,
                    "verdict": {"accepted": self.accept, "label": target if self.accept else "picture"}}

        self.gate = VerificationGate({"enabled": True, "log_dir": self.folder + "/{video_output_path}"}, "v",
                                     verify=verify, camera={"hfov": 79, "width": 64, "height": 48},
                                     ping=lambda url, timeout: {"ok": True, "settings": {"selector": "vqf"}})
        self.gate.start_episode(7, "scene", "ep", "potted_plant", goal_object_ids=["42"])

    def observe(self, step, size, centroid, camera=(0.0, 0.0), instances=None, fusion=None):
        return self.gate.observe(step, RGB, [mask(size)], [0], [0.5], [centroid],
                                 [instances or [{"id": "42", "pixels": 10}]], (*camera, 0.0),
                                 -math.radians(30), fusion)

    def test_target_name(self):
        self.assertEqual(target_name("potted_plant"), "potted plant")
        self.assertEqual(target_name("table | dining table | desk"), "table")

    def test_evidence_matching(self):
        self.observe(1, 4, (2.1, 0.0))   # matches, small
        self.observe(2, 16, (2.2, 0.1))  # matches, most readable
        self.observe(3, 30, (5.0, 0.0))  # another object
        evidence, distance = self.gate.select_evidence(3, (2.0, 0.0))
        self.assertEqual(evidence["step"], 2)
        self.assertAlmostEqual(distance, math.hypot(0.2, 0.1), places=6)
        evidence, distance = self.gate.select_evidence(3, (4.0, 0.0))  # only the 1.5 m fallback
        self.assertEqual(evidence["step"], 3)
        self.assertEqual(self.gate.select_evidence(3, (9.0, 9.0)), (None, None))
        # A cluster centroid at the evidence step takes precedence over the current one
        self.observe(4, 20, (7.0, 0.0), fusion={"clusters": [{"id": 5, "centroid": [7.1, 0.0]}]})
        self.assertEqual(self.gate.select_evidence(5, (9.0, 0.0))[0]["step"], 4)

    def test_handle_and_logs(self):
        self.observe(1, 16, (2.0, 0.0))
        self.assertEqual(self.gate.handle(request(), step=1), REJECTED)
        self.assertEqual(self.calls[-1]["target"], "potted plant")
        self.assertAlmostEqual(self.calls[-1]["elevation"], 30.0)
        self.accept = True
        self.observe(2, 16, (5.0, 0.0))
        self.assertEqual(self.gate.handle(request(cluster_id=4, centroid=(5.0, 0.0), seq=2), step=2), 1)
        lines = [json.loads(line) for line in open(os.path.join(self.folder, "v", "decisions.jsonl"))]
        self.assertEqual([line["state"] for line in lines], [-1, 1])
        self.assertTrue(lines[0]["ground_truth"]["is_correct_instance"])
        self.assertEqual(lines[0]["server"], {"selector": "vqf"})
        self.assertEqual(lines[0]["camera"], {"hfov": 79, "width": 64, "height": 48})
        self.assertTrue(os.path.isfile(os.path.join(self.folder, "v", "epi7_potted_plant", "c3_s1", "input.jpg")))
        self.assertEqual(self.gate.finish_episode()["rejected"], 1)

    def test_fail_open(self):
        self.assertEqual(self.gate.handle(request(), step=1), 1)  # no evidence -> accept
        self.observe(2, 16, (2.0, 0.0))
        self.server_down = True
        self.assertEqual(self.gate.handle(request(cluster_id=4, seq=2), step=2), 1)  # error -> accept
        counts = self.gate.finish_episode()
        self.assertEqual((counts["no_evidence"], counts["errors"], counts["accepted"]), (1, 1, 2))

    def test_defer(self):
        """A view "no" after a raw "yes" defers; at the stop it rejects, also from the cache."""
        self.gate.cfg["on_view_reject"] = "defer"
        verdicts = []
        self.gate._verify = lambda *a, **k: {"ok": True, "view_image": None, "verdict": verdicts.pop(0)}
        view_no = {"accepted": False, "label": "lamp", "judged": "view", "input_scores": {"potted plant": 0.6}}
        raw_no = {"accepted": False, "label": "lamp", "judged": "input", "input_scores": {"lamp": 0.6}}
        self.observe(1, 16, (2.0, 0.0))
        verdicts += [view_no]
        self.assertEqual(self.gate.handle(request(), step=1), DEFERRED)
        self.assertEqual(self.gate.handle(request(seq=2, at_stop=True), step=2), REJECTED)  # cached view
        self.observe(3, 16, (6.0, 0.0))
        verdicts += [raw_no]
        self.assertEqual(self.gate.handle(request(cluster_id=5, centroid=(6.0, 0.0), seq=3), step=3), REJECTED)
        self.observe(4, 16, (9.0, 0.0))
        verdicts += [view_no]
        self.assertEqual(self.gate.handle(request(cluster_id=6, centroid=(9.0, 0.0), seq=4, at_stop=True),
                                         step=4), REJECTED)
        self.assertEqual(self.gate.finish_episode()["deferred"], 1)
        with self.assertRaises(ValueError):
            VerificationGate({"on_view_reject": "maybe"}, "v")

    def test_no_evidence_defer(self):
        """on_no_evidence=defer: no view defers; the stop re-verifies by position (with a view gathered on the
        way it is judged; still without one it rejects). on_error has no defer."""
        self.gate.cfg["on_no_evidence"] = "defer"
        self.assertEqual(self.gate.handle(request(), step=1), DEFERRED)
        self.assertEqual(self.calls, [])
        self.observe(2, 16, (2.1, 0.0))   # seen on the way to the cluster
        self.accept = True
        self.assertEqual(self.gate.handle(request(seq=2, centroid=(2.1, 0.0), at_stop=True), step=3), 1)
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(self.gate.handle(request(cluster_id=8, centroid=(9.0, 9.0), seq=3, at_stop=True),
                                          step=4), REJECTED)
        lines = [json.loads(line) for line in open(os.path.join(self.folder, "v", "decisions.jsonl"))]
        self.assertEqual([(line["policy"], line["state"]) for line in lines],
                         [("on_no_evidence=defer", DEFERRED), (None, 1), ("on_no_evidence=defer", REJECTED)])
        counts = self.gate.finish_episode()
        self.assertEqual((counts["no_evidence"], counts["deferred"], counts["rejected"]), (2, 1, 1))
        VerificationGate({"on_no_evidence": "defer"}, "v")
        with self.assertRaises(ValueError):
            VerificationGate({"on_error": "defer"}, "v")
        with self.assertRaises(ValueError):
            VerificationGate({"on_no_evidence": "maybe"}, "v")

    def test_recheck_cooldown(self):
        self.observe(1, 16, (2.0, 0.0), camera=(0.0, 0.0))
        self.assertEqual(self.gate.handle(request(), step=1), REJECTED)
        # Too soon, even from a new place
        self.assertEqual(self.observe(10, 16, (2.0, 0.0), camera=(0.0, 3.0)), [])
        # Cooldown over, but the same viewpoint
        self.assertEqual(self.observe(30, 16, (2.0, 0.0), camera=(0.2, 0.0)), [])
        # Cooldown over and a new viewpoint (1.4 m away)
        self.assertEqual(self.observe(31, 20, (2.0, 0.0), camera=(1.0, 1.0)), [3])
        self.assertEqual(self.gate.clusters[3]["state"], UNVERIFIED)
        # The re-check uses the most readable new view, not the rejected one
        self.gate.handle(request(seq=2), step=32)
        self.assertEqual(self.gate.clusters[3]["used"][-1]["camera"], (1.0, 1.0))
        # A different direction from nearby also counts as new
        self.gate.cfg["max_checks"] = 5
        self.assertEqual(self.observe(60, 16, (2.0, 0.0), camera=(1.0, 1.5)), [])  # 0.5 m, ~11 deg
        self.assertEqual(self.observe(61, 16, (2.0, 0.0), camera=(1.4, -0.3)), [3])  # 1.3 m from first
        # max_checks bounds re-checks
        self.gate.handle(request(seq=3), step=62)
        self.gate.cfg["max_checks"] = 3
        self.assertEqual(self.observe(100, 16, (2.0, 0.0), camera=(3.0, -2.0)), [])

    def test_eviction_keeps_other_objects(self):
        # The reviewer's case: 69 readable views of A, then one small view of B.
        for step in range(1, 70):
            self.observe(step, 16, (2.0 + 0.01 * (step % 5), 0.0))
        self.observe(70, 2, (6.0, 1.0))
        self.assertLessEqual(len(self.gate.evidence), 8 + 1 + 1)
        self.gate.handle(request(cluster_id=9, centroid=(6.0, 1.0)), step=71)
        self.assertEqual((len(self.calls), self.calls[-1]["area"]), (1, 4))  # B's own view
        self.assertEqual(self.gate.finish_episode()["no_evidence"], 0)
        # The newest view of a cell survives even when it is the least readable
        self.observe(72, 2, (2.0, 0.0))
        self.assertIn(72, [e["step"] for e in self.gate.evidence])

    def test_evidence_bound(self):
        self.gate.evidence.max_total = 20
        for step in range(1, 200):
            self.observe(step, 16, (0.8 * (step % 40), 0.0))
        self.assertLessEqual(len(self.gate.evidence), 40)  # one per cell at worst, never zero
        self.assertEqual(len(self.gate.evidence.cells), 40)

    def test_nearby_object_gets_its_own_check(self):
        # Reviewer repro: a real chair B 0.6 m from a rejected false positive A
        self.observe(1, 16, (2.0, 0.0))
        self.assertEqual(self.gate.handle(request(cluster_id=3), step=1), REJECTED)
        self.observe(5, 16, (2.6, 0.0))
        self.accept = True
        self.assertEqual(self.gate.handle(request(cluster_id=4, centroid=(2.6, 0.0), seq=2), step=5), 1)
        self.assertEqual(len(self.calls), 2)
        self.assertEqual(self.gate.clusters[4]["used"][0]["camera"], (0.0, 0.0))
        self.assertEqual(self.gate.clusters[4]["checks"], 1)

    def test_view_nearer_another_cluster_is_not_evidence(self):
        fusion = {"clusters": [{"id": 3, "centroid": [2.0, 0.0]}]}
        self.observe(1, 30, (2.0, 0.0), fusion=fusion)  # most readable, but A's
        self.observe(2, 8, (2.6, 0.0), fusion=fusion)
        evidence, _ = self.gate.select_evidence(4, (2.6, 0.0))
        self.assertEqual(evidence["step"], 2)

    def test_cached_verdict_for_respawn(self):
        self.observe(1, 16, (2.0, 0.0))
        self.assertEqual(self.gate.handle(request(cluster_id=3), step=1), REJECTED)
        # A re-spawned cluster at the same place selects the same view: cached, no call
        self.accept = True
        self.assertEqual(self.gate.handle(request(cluster_id=8, centroid=(2.1, 0.0), seq=2), step=3), REJECTED)
        self.assertEqual((len(self.calls), self.gate.finish_episode()["cached"]), (1, 1))

    def test_at_stop_matches_the_stop_position(self):
        # The cluster's centroid (as of the view) is far from the stop; the stop position decides
        self.observe(1, 16, (6.0, 0.0), fusion={"clusters": [{"id": 5, "centroid": [3.0, 0.0]}]})
        self.accept = True
        self.assertEqual(self.gate.handle(request(cluster_id=5, centroid=(6.1, 0.0), at_stop=True), step=2), 1)
        self.assertEqual((len(self.calls), self.gate.finish_episode()["no_evidence"]), (1, 0))

    def test_position_request(self):
        self.observe(1, 16, (4.0, 4.0))
        self.accept = True
        self.assertEqual(self.gate.handle(request(cluster_id=-1, centroid=(4.2, 4.0), at_stop=True), step=2), 1)
        self.assertEqual(len(self.calls), 1)
        self.assertNotIn(-1, self.gate.clusters)
        line = json.loads(open(os.path.join(self.folder, "v", "decisions.jsonl")).readline())
        self.assertTrue(line["at_stop"])

    def test_consecutive_errors_abort(self):
        self.server_down = True
        for step in range(1, 3):
            self.observe(step, 16, (2.0 * step, 0.0))
            self.assertEqual(self.gate.handle(request(cluster_id=step, centroid=(2.0 * step, 0.0)), step=step), 1)
        self.observe(3, 16, (6.0, 0.0))
        with self.assertRaises(RuntimeError):
            self.gate.handle(request(cluster_id=3, centroid=(6.0, 0.0)), step=3)
        # A success in between resets the count
        self.gate.consecutive_errors = 0
        self.server_down = False
        self.observe(4, 16, (8.0, 0.0))
        self.gate.handle(request(cluster_id=4, centroid=(8.0, 0.0)), step=4)
        self.assertEqual(self.gate.consecutive_errors, 0)

    def test_off(self):
        gate = VerificationGate({"enabled": False}, "v")
        self.assertEqual(gate.observe(1, RGB, [mask(8)], [0], [0.5], [(1.0, 1.0)], None, (0, 0, 0), 0.0), [])
        self.assertIsNone(gate.finish_episode())
        with self.assertRaises(ValueError):
            VerificationGate({"on_error": "maybe"}, "v")


if __name__ == "__main__":
    unittest.main()
