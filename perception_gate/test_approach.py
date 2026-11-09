"""VerificationGate approach logic (server selector d1) with a fake server (no ROS, no GPU):

    python -m unittest perception_gate.test_approach
"""
import json
import math
import os
import tempfile
import unittest

import numpy as np

from perception_gate.gate import (ACCEPTED, APPROACH, REJECTED, UNVERIFIED, VerificationGate, select_views)

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
        self.assertEqual(self.gate.pending_approach(5), {"approach_id": 1, "object": (17.0, 0.0),
                                                         "camera": (15.0, 0.0), "max_steps": 40})
        self.assertIsNone(self.gate.pending_approach(3))
        counts = self.gate.finish_episode()
        self.assertEqual((counts["accepted"], counts["rejected"], counts["moves"], counts["insufficient"],
                          counts["qwen_decisions"]), (1, 1, 1, 1, 3))
        line = decisions(self.folder)[-1]
        self.assertEqual((line["state"], line["status"], line["model"], line["approach"]["id"]),
                         (APPROACH, "insufficient", "qwen", 1))

    def test_new_view_on_the_way(self):
        """The first new view of the cluster is judged at once (no cooldown, no max_checks, no
        cache); insufficient keeps approaching, a later new view is judged again."""
        self.observe(1, (0.0, 0.0))
        self.statuses += ["insufficient", "insufficient", "correct"]
        self.assertEqual(self.gate.handle(request(), step=1), APPROACH)
        self.assertEqual(self.observe(2, (0.3, 0.0)), [])        # 0.3 m closer, same bearing: not new
        self.assertEqual(self.observe(3, (1.0, 0.0)), [3])       # 1.0 m closer: new
        self.assertEqual(self.observe(3, (1.05, 0.0)), [])       # once until judged
        self.assertEqual(self.gate.handle(request(seq=2, approach_id=1, status=4, steps=2), step=3), APPROACH)
        self.assertEqual(self.gate.pending_approach(3)["approach_id"], 1)  # the same approach goes on
        self.assertEqual(self.observe(4, (1.2, 0.0)), [])        # not new against the judged 1.0 m view
        self.assertEqual(self.observe(5, (1.6, 0.0)), [3])       # 0.6 m closer than it
        self.assertEqual(self.gate.handle(request(seq=3, approach_id=1, status=4, steps=4), step=5), ACCEPTED)
        self.assertEqual(self.gate.clusters[3]["used"][-1]["camera"], (1.6, 0.0))
        self.assertEqual(self.gate.clusters[3]["checks"], 1)
        counts = self.gate.finish_episode()
        self.assertEqual((counts["approach_early"], counts["approach_accepted"], counts["approach_steps"],
                          counts["moves"], len(self.calls)), (2, 1, 4, 1, 3))

    def test_side_view_counts_as_new(self):
        self.observe(1, (0.0, 0.0))
        self.insufficient_then("wrong")
        self.gate.handle(request(), step=1)
        self.assertEqual(self.observe(2, (0.5, 1.2)), [3])       # ~39 deg around the object
        self.assertEqual(self.gate.handle(request(seq=2, approach_id=1, status=4, steps=1), step=2), REJECTED)
        self.assertIsNone(self.gate.pending_approach(3))

    def test_arrival_judged_without_thresholds_and_never_insufficient(self):
        self.observe(1, (0.0, 0.0))
        self.insufficient_then("correct")
        self.gate.handle(request(), step=1)
        self.observe(5, (0.2, 0.0), size=30)                     # not new by the thresholds
        self.observe(6, (0.3, 0.0), size=8)
        self.assertEqual(self.gate.handle(request(3, OBJECT, seq=2, at_stop=True, approach_id=1,
                                                  status=1, steps=6), step=6), ACCEPTED)
        self.assertEqual(self.gate.clusters[3]["used"][-1]["camera"], (0.3, 0.0))  # this step's view
        # Insufficient on arrival: SAM3's raw label decides (d1_verdict's label is the target)
        self.observe(10, (4.0, 0.0), centroid=(6.0, 0.0))
        self.statuses += ["insufficient", "insufficient"]
        self.gate.handle(request(4, (6.0, 0.0), seq=3), step=10)
        self.observe(12, (4.5, 0.0), centroid=(6.0, 0.0))
        self.assertEqual(self.gate.handle(request(4, (6.0, 0.0), seq=4, at_stop=True, approach_id=1, status=1,
                                                  steps=2), step=12), ACCEPTED)
        self.assertEqual(decisions(self.folder)[-1]["policy"], "insufficient,arrival_sam3_accept")

    def test_arrival_needs_a_view_from_the_arrival_step(self):
        # Not re-detected on arrival: SAM3's raw label on the last judged view decides
        self.observe(1, (0.0, 0.0))
        self.insufficient_then()
        self.gate.handle(request(), step=1)
        self.observe(5, (1.5, 0.0))                              # seen on the way, not at arrival
        self.assertEqual(self.gate.handle(request(3, OBJECT, seq=2, at_stop=True, approach_id=1, status=1,
                                                  steps=7), step=8), ACCEPTED)
        self.assertEqual(decisions(self.folder)[-1]["policy"], "not_redetected,sam3_accept")
        counts = self.gate.finish_episode()
        self.assertEqual((counts["not_redetected"], counts["accepted"]), (1, 1))
        self.assertEqual(len(self.calls), 1)

    def test_not_redetected_rejects_when_sam3_label_is_wrong(self):
        self.observe(1, (0.0, 0.0))
        self.statuses.append("insufficient")
        verify = self.gate._verify
        self.gate._verify = lambda *a, **k: {**verify(*a, **k), "verdict": {**d1_verdict("insufficient"),
                                                                          "label": "vase"}}
        self.gate.handle(request(), step=1)
        self.assertEqual(self.gate.handle(request(3, OBJECT, seq=2, at_stop=True, approach_id=1, status=1,
                                                  steps=7), step=8), REJECTED)
        self.assertEqual(decisions(self.folder)[-1]["policy"], "not_redetected,sam3_reject")

    def test_arrival_view_matches_drifted_cluster_centroid(self):
        # A large object: the arrival view is 1.6 m from the issued point but at the cluster's centroid
        self.observe(1, (0.0, 0.0))
        self.insufficient_then("correct")
        self.gate.handle(request(), step=1)
        drifted = (OBJECT[0] + 1.6, 0.0)
        self.observe(8, (2.5, -1.5), centroid=drifted, fusion={"clusters": [{"id": 3, "centroid": drifted}]})
        self.assertEqual(self.gate.handle(request(3, drifted, seq=2, at_stop=True, approach_id=1, status=1,
                                                  steps=7), step=8), ACCEPTED)
        self.assertEqual(len(self.calls), 2)

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
        for cluster_id, status in ((3, 2), (4, 3)):  # no path / lost, budget
            centroid = (2.0 + 5 * cluster_id, 0.0)
            self.observe(10 * cluster_id, (centroid[0] - 2.0, 0.0), centroid)
            self.insufficient_then()
            self.assertEqual(self.gate.handle(request(cluster_id, centroid, seq=cluster_id), step=10 * cluster_id),
                             APPROACH)
            self.assertEqual(self.gate.handle(request(cluster_id, centroid, seq=20 + cluster_id, approach_id=1,
                                                      status=status, steps=1), step=10 * cluster_id), REJECTED)
            self.assertIsNone(self.gate.pending_approach(cluster_id))
        self.assertEqual(self.gate.handle(request(9, (40.0, 0.0), seq=40, approach_id=2, status=1), step=60),
                         REJECTED)
        counts = self.gate.finish_episode()
        self.assertEqual((counts["unreachable"], counts["aborted"], counts["fallback"], counts["rejected"]),
                         (1, 1, 3, 3))
        self.assertEqual(len(self.calls), 2)

    def test_rejected_after_approach_follows_reopen_rule(self):
        self.observe(1, (0.0, 0.0))
        self.insufficient_then()
        self.gate.handle(request(), step=1)
        self.assertEqual(self.gate.handle(request(seq=2, approach_id=1, status=2), step=1), REJECTED)
        self.assertEqual(self.observe(10, (0.0, 3.0)), [])       # cooldown
        self.assertEqual(self.observe(30, (1.0, 1.0)), [3])      # cooldown over, new view
        self.assertEqual(self.gate.clusters[3]["state"], UNVERIFIED)

    def test_cluster_budget(self):
        self.gate.cfg["max_checks"] = 5
        for n, step in ((1, 1), (2, 40)):
            self.observe(step, (0.0, 3.0 * n))
            self.insufficient_then()
            self.assertEqual(self.gate.handle(request(seq=n), step=step), APPROACH)
            self.assertEqual(self.gate.pending_approach(3)["approach_id"], n)
            self.gate.handle(request(seq=10 + n, approach_id=n, status=3, steps=5), step=step)
        self.observe(80, (-3.0, 0.0))
        self.insufficient_then()
        self.assertEqual(self.gate.handle(request(seq=3), step=80), REJECTED)
        self.assertEqual(decisions(self.folder)[-1]["policy"], "insufficient,cluster_budget")
        self.assertEqual(self.gate.finish_episode()["approach_steps"], 10)

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
        self.assertEqual(decisions(self.folder)[-1]["policy"], "insufficient,episode_budget")

    def test_max_steps_capped_by_episode_end(self):
        self.observe(480, (0.0, 0.0))
        self.insufficient_then()
        self.gate.handle(request(), step=480)
        self.assertEqual(self.gate.pending_approach(3)["max_steps"], 19)  # steps 481..499; 500 is the stop
        self.observe(499, (-1.0, 1.0), centroid=(5.0, 0.0))
        self.insufficient_then()
        self.assertEqual(self.gate.handle(request(4, (5.0, 0.0), seq=2), step=499), REJECTED)
        self.assertEqual(decisions(self.folder)[-1]["policy"], "insufficient,episode_end")

    def test_stop_never_stops_on_insufficient(self):
        # The stop backstop: insufficient rejects (no approach from the stop)
        self.observe(1, (0.0, 0.0))
        self.insufficient_then()
        self.assertEqual(self.gate.handle(request(at_stop=True), step=1), REJECTED)
        self.assertEqual(decisions(self.folder)[-1]["policy"], "insufficient,at_stop")
        # A stop position without a cluster never approaches
        self.observe(3, (4.0, 4.0), centroid=(6.0, 4.0))
        self.insufficient_then()
        self.assertEqual(self.gate.handle(request(-1, (6.0, 4.0), seq=3, at_stop=True), step=3), REJECTED)

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


def views_verdict(status, margin=None, vote=None, per_view=(), source="qwen"):
    return {"status": status, "source": source, "accepted": status == "correct", "qwen": {"status": status},
            "per_view": list(per_view), "n_views": len(per_view),
            "aggregate": None if margin is None else {"margin": margin, "vote": vote, "n_used": 2,
                                                      "label": "potted plant"}}


class MultiviewTests(unittest.TestCase):
    """verification.multiview: the view bank and /verify_views decisions (fake server)."""

    def setUp(self):
        self.calls, self.statuses = [], []      # single-view /verify
        self.mv_calls, self.mv_replies = [], []  # /verify_views
        self.folder = tempfile.mkdtemp()

        def verify(url, rgb, m, target, input_elevation=0.0, request_id=None, timeout=None):
            self.calls.append(request_id)
            return {"ok": True, "view_image": None, "verdict": d1_verdict(self.statuses.pop(0))}

        def verify_views(url, views, target, sam3="fallback", request_id=None, timeout=None, **variants):
            self.mv_calls.append({"views": views, "sam3": sam3, **variants})
            verdict = self.mv_replies.pop(0)
            if not verdict["per_view"]:
                verdict["per_view"] = [{"step": v["step"], "index": v["index"],
                                        "scores": {"potted plant": 0.6, "vase": 0.1} if sam3 != "never" else None}
                                       for v in views]
            return {"ok": True, "verdict": verdict, "grid_image": None}

        self.verify, self.verify_views = verify, verify_views
        self.gate = self.make_gate()

    def make_gate(self, settings=None, **cfg):
        gate = VerificationGate({"enabled": True, "approach": True, "on_error": "reject", "multiview": True,
                                 "log_dir": self.folder + "/{video_output_path}", **cfg}, "v",
                                verify=self.verify, verify_views=self.verify_views,
                                ping=lambda url, timeout: {"ok": True, "settings": settings or {
                                    "selector": "d1", "multiview": True, "multiview_grid": True}})
        gate.start_episode(7, "scene", "ep", "potted_plant", goal_object_ids=["42"], max_episode_steps=500)
        return gate

    def observe(self, step, camera, centroid=OBJECT, size=16, fusion=None, gate=None):
        return (gate or self.gate).observe(step, RGB, [mask(size)], [0], [0.5], [centroid],
                                           [[{"id": "42", "pixels": 10}]], (*camera, 0.0), -math.radians(30), fusion)

    def approach(self, cluster_id=3, centroid=OBJECT, step=1, camera=(0.0, 0.0)):
        self.observe(step, camera, centroid)
        self.statuses.append("insufficient")
        self.assertEqual(self.gate.handle(request(cluster_id, centroid), step=step), APPROACH)

    def bank(self, cluster_id=3):
        return [(b["step"], b["index"]) for b in self.gate.clusters[cluster_id].get("bank", [])]

    # ------------------------------------------------------------------

    def test_sibling_fragments_banked_competitors_excluded(self):
        self.approach()
        self.assertEqual(self.bank(), [(1, 0)])  # seeded from the evidence
        # 720-like: cluster 9 is a fragment 0.27 m from cluster 3; the view is nearer 9 than 3
        siblings = {"clusters": [{"id": 3, "centroid": OBJECT}, {"id": 9, "centroid": (2.1, 0.25)}]}
        self.observe(2, (0.5, 0.0), centroid=(2.1, 0.3), fusion=siblings)
        self.assertEqual(self.bank(), [(1, 0), (2, 0)])
        # a different object 1 m away still excludes a view nearer it
        other = {"clusters": [{"id": 3, "centroid": OBJECT}, {"id": 10, "centroid": (3.0, 0.3)}]}
        self.observe(3, (0.6, 0.0), centroid=(2.6, 0.3), fusion=other)
        self.assertEqual(self.bank(), [(1, 0), (2, 0)])
        # the single-view early trigger keeps its strict rule: the sibling view was not "new"
        self.assertIsNone(self.gate.clusters[3]["approach"]["early"])

    def test_arrival_uses_bank_without_rule_c(self):
        self.approach()
        self.observe(5, (1.0, 0.0), size=24)            # seen on the way, not at arrival
        self.mv_replies.append(views_verdict("wrong"))
        self.assertEqual(self.gate.handle(request(seq=2, at_stop=True, approach_id=1, status=1, steps=7), step=8),
                         REJECTED)
        line = decisions(self.folder)[-1]
        self.assertEqual((line["policy"], line["model"], line["multiview"]["mode"]),
                         ("multiview,arrival,qwen_reject", "qwen_multi", "arrival"))
        self.assertEqual([(v["step"], round(v["range_m"], 2)) for v in self.mv_calls[0]["views"]],
                         [(5, 1.0), (1, 2.0)])                # nearest first
        self.assertEqual(self.mv_calls[0]["grid_view"], 0)   # slot 1: best view at <= median range
        self.assertEqual(len(self.calls), 1)                 # only the initial single-view request
        counts = self.gate.finish_episode()
        self.assertEqual((counts["not_redetected"], counts["multiview_qwen"], counts["multiview_views"]), (0, 1, 2))
        folder = os.path.join(self.folder, "v", "epi7_potted_plant", "c3_s8", "views")
        self.assertEqual(sorted(os.listdir(folder)), ["v1_s5_0.jpg", "v1_s5_0_mask.png", "v2_s1_0.jpg",
                                                      "v2_s1_0_mask.png"])

    def test_arrival_sam3_fallback_thresholds(self):
        for n, (margin, vote, expected) in enumerate(((0.0, 0.5, ACCEPTED), (-0.01, 1.0, REJECTED),
                                                      (0.3, 0.4, REJECTED), (None, None, REJECTED))):
            cluster_id, centroid = 3 + n, (2.0 + 5 * n, 0.0)
            self.approach(cluster_id, centroid, step=10 * n + 1, camera=(centroid[0] - 2.0, 0.0))
            self.mv_replies.append(views_verdict("insufficient", margin, vote))
            self.assertEqual(self.gate.handle(request(cluster_id, centroid, seq=10 * n + 2, at_stop=True,
                                                      approach_id=1, status=1, steps=3), step=10 * n + 4), expected)
            self.assertEqual(decisions(self.folder)[-1]["model"], "sam3_multi")
        self.assertIn("sam3_accept(M=0.00,V=0.50,n=2)", decisions(self.folder)[1]["policy"])
        self.assertIn("no_sam3_match", decisions(self.folder)[-1]["policy"])
        self.assertEqual(self.mv_calls[0]["sam3"], "fallback")

    def test_failed_approach(self):
        # aborted with two views: Qwen decides
        self.approach()
        self.observe(4, (1.0, 0.0))
        self.mv_replies.append(views_verdict("correct"))
        self.assertEqual(self.gate.handle(request(seq=2, approach_id=1, status=3, steps=40), step=41), ACCEPTED)
        self.assertEqual(decisions(self.folder)[-1]["policy"], "multiview,failure,qwen_accept")
        # unreachable, Qwen insufficient, |M| < 0.15: reject
        self.approach(4, (10.0, 0.0), step=50, camera=(8.0, 0.0))
        self.observe(52, (8.5, 0.0), centroid=(10.0, 0.0))
        self.mv_replies.append(views_verdict("insufficient", 0.1, 1.0))
        self.assertEqual(self.gate.handle(request(4, (10.0, 0.0), seq=3, approach_id=1, status=2, steps=2),
                                          step=52), REJECTED)
        self.assertIn("multiview,failure,insufficient,sam3_reject", decisions(self.folder)[-1]["policy"])
        # ... and M >= 0.15 with the vote accepts
        self.approach(5, (20.0, 0.0), step=60, camera=(18.0, 0.0))
        self.observe(62, (18.5, 0.0), centroid=(20.0, 0.0))
        self.mv_replies.append(views_verdict("insufficient", 0.2, 1.0))
        self.assertEqual(self.gate.handle(request(5, (20.0, 0.0), seq=4, approach_id=1, status=3, steps=2),
                                          step=62), ACCEPTED)
        # one view: rejected without a server call
        self.approach(6, (30.0, 0.0), step=70, camera=(28.0, 0.0))
        self.assertEqual(self.gate.handle(request(6, (30.0, 0.0), seq=5, approach_id=1, status=3, steps=40),
                                          step=110), REJECTED)
        self.assertEqual(decisions(self.folder)[-1]["policy"], "aborted")
        counts = self.gate.finish_episode()
        self.assertEqual((counts["multiview_requests"], counts["multiview_failure_decided"], counts["aborted"],
                          counts["unreachable"]), (3, 3, 3, 1))

    def test_failure_off_or_aborted_only(self):
        gate = self.gate = self.make_gate(multiview_failure="aborted")
        self.approach()
        self.observe(4, (1.0, 0.0))
        self.assertEqual(gate.handle(request(seq=2, approach_id=1, status=2, steps=4), step=5), REJECTED)
        self.assertEqual((decisions(self.folder)[-1]["policy"], self.mv_calls), ("unreachable", []))

    def test_early(self):
        self.approach()
        self.assertEqual(self.observe(3, (1.0, 0.0)), [3])   # new view on the way
        self.mv_replies.append(views_verdict("insufficient"))
        self.assertEqual(self.gate.handle(request(seq=2, approach_id=1, status=4, steps=2), step=3), APPROACH)
        self.assertEqual(self.gate.pending_approach(3)["approach_id"], 1)  # the same approach goes on
        self.assertEqual(self.mv_calls[-1]["sam3"], "never")  # no SAM3 fallback on the way
        self.assertEqual(decisions(self.folder)[-1]["policy"], "multiview,early,insufficient,keep_approaching")
        self.assertEqual(self.observe(5, (1.6, 0.0)), [3])
        self.mv_replies.append(views_verdict("correct"))
        self.assertEqual(self.gate.handle(request(seq=3, approach_id=1, status=4, steps=4), step=5), ACCEPTED)
        self.assertEqual([v["step"] for v in self.mv_calls[-1]["views"]], [5, 3, 1])

    def test_sam3_scores_cached_and_sent_back(self):
        self.approach()
        self.observe(4, (1.0, 0.0))
        self.mv_replies.append(views_verdict("insufficient", 0.1, 1.0))
        self.gate.handle(request(seq=2, approach_id=1, status=3, steps=40), step=41)   # failure, rejected
        self.assertTrue(all(v["scores"] is None for v in self.mv_calls[0]["views"]))
        self.gate.clusters[3]["approaches"] = 0
        self.observe(60, (0.5, 1.5))
        self.statuses.append("insufficient")
        self.gate.handle(request(seq=3), step=60)
        self.mv_replies.append(views_verdict("wrong"))
        self.gate.handle(request(seq=4, approach_id=1, status=1, steps=1), step=61)
        sent = {(v["step"], v["index"]): v["scores"] for v in self.mv_calls[1]["views"]}
        self.assertEqual(sent[(4, 0)], {"potted plant": 0.6, "vase": 0.1})
        self.assertIsNone(sent[(60, 0)])

    def test_select_views(self):
        def item(step, camera, quality, centroid=OBJECT, index=0):
            return {"step": step, "index": index, "camera": camera, "centroid": centroid, "quality": quality}

        bank = [item(1, (0.0, 0.0), 0.9),      # 2.0 m, the best overall but farther than the median
                item(2, (0.0, 0.05), 0.8),     # duplicate of step 1
                item(3, (1.0, 0.0), 0.3),      # 1.0 m
                item(4, (1.5, 0.0), 0.5),      # 0.5 m: best at <= median range
                item(5, (0.5, 1.0), 0.4),
                item(4, (1.5, 0.0), 0.5)]      # the same view again
        views, best = select_views(bank, k=3)
        self.assertEqual((best["step"], [v["step"] for v in views]), (4, [4, 5, 1]))
        views, best = select_views(bank, k=6)
        self.assertEqual([v["step"] for v in views], [4, 3, 5, 1])
        self.assertEqual(select_views([], 6), ([], None))

    def test_bank_cap_keeps_newest(self):
        self.gate = self.make_gate(multiview_bank=6)
        self.approach()
        for step in range(2, 10):  # less and less readable
            self.observe(step, (0.1 * step, 0.0), size=22 - 2 * step)
        self.observe(10, (1.2, 0.0), size=2)   # the least readable, but the newest
        self.assertEqual(self.bank(), [(1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (10, 0)])

    def test_off_keeps_single_view_behaviour(self):
        gate = VerificationGate({"enabled": True, "approach": True, "on_error": "reject", "log_dir": self.folder},
                                "v", verify=self.verify, verify_views=self.verify_views,
                                ping=lambda url, timeout: {"ok": True, "settings": {"selector": "d1"}})
        self.assertNotIn("multiview_requests", gate.finish_episode())
        gate.start_episode(7, "scene", "ep", "potted_plant", goal_object_ids=["42"], max_episode_steps=500)
        self.observe(1, (0.0, 0.0), gate=gate)
        self.statuses += ["insufficient"]
        gate.handle(request(), step=1)
        self.observe(5, (1.0, 0.0), gate=gate)
        self.assertNotIn("bank", gate.clusters[3])
        self.assertEqual(gate.handle(request(seq=2, at_stop=True, approach_id=1, status=1, steps=7), step=8),
                         ACCEPTED)                           # rule C, as before
        self.assertEqual(self.mv_calls, [])

    def test_config_validation(self):
        with self.assertRaisesRegex(ValueError, "needs verification.approach"):
            VerificationGate({"enabled": True, "multiview": True, "log_dir": self.folder}, "v",
                             ping=lambda url, timeout: {"ok": True, "settings": {"selector": "input"}})
        with self.assertRaisesRegex(ValueError, "settings.multiview"):
            self.make_gate(settings={"selector": "d1"})
        with self.assertRaisesRegex(ValueError, "--multiview-grid"):
            self.make_gate(settings={"selector": "d1", "multiview": True}, multiview_grid=True)
        self.make_gate(settings={"selector": "d1", "multiview": True})  # images only is fine
        for bad in ({"multiview_variant": "nope"}, {"multiview_sam3": "nope"}, {"multiview_failure": "nope"},
                    {"multiview_max_views": 9}):
            with self.assertRaises(ValueError):
                self.make_gate(**bad)

    def test_variants_reach_the_server(self):
        self.gate = self.make_gate(multiview_grid=True, multiview_variant="forced2", multiview_sam3="always")
        self.approach()
        self.observe(5, (1.0, 0.0))
        self.mv_replies.append(views_verdict("correct"))
        self.gate.handle(request(seq=2, at_stop=True, approach_id=1, status=1, steps=7), step=8)
        call = self.mv_calls[-1]
        self.assertEqual((call["grid"], call["forced2"], call["crop_small"], call["sam3"]),
                         (True, True, False, "always"))

    def test_server_error_rejects(self):
        self.approach()
        self.observe(5, (1.0, 0.0))

        def broken(*args, **kwargs):
            raise RuntimeError("perception server: 500")

        self.gate._verify_views = broken
        self.assertEqual(self.gate.handle(request(seq=2, at_stop=True, approach_id=1, status=1, steps=7), step=8),
                         REJECTED)
        self.assertEqual(decisions(self.folder)[-1]["policy"], "multiview,arrival,on_error=reject")


class JevSam3Tests(unittest.TestCase):
    """Server selector jev_sam3: JEV judges the uncertainty, SAM3 decides; the gate passes the d1mv
    close-range thresholds per mode (fake server, run_strategy.sh jev_sam3 settings)."""

    def setUp(self):
        self.statuses, self.mv_calls, self.mv_replies = [], [], []
        self.folder = tempfile.mkdtemp()

        def verify(url, rgb, m, target, input_elevation=0.0, request_id=None, timeout=None):
            status = self.statuses.pop(0)
            return {"ok": True, "view_image": None,
                    "verdict": {**jev_verdict(status), "source": "jev" if status == "insufficient" else "sam3"}}

        def verify_views(url, views, target, sam3="fallback", request_id=None, timeout=None, **variants):
            self.mv_calls.append({"views": views, **variants})
            status = self.mv_replies.pop(0)
            if variants.get("forced") and status == "insufficient":
                raise AssertionError("a forced decision is never insufficient")
            decided = status != "insufficient"
            return {"ok": True, "grid_image": None, "verdict": {
                "status": status, "source": "sam3" if decided else "jev", "accepted": status == "correct",
                "qwen": None, "aggregate": {"margin": 0.3, "vote": 1.0, "n_used": len(views)} if decided else None,
                "per_view": [{"step": v["step"], "index": v["index"],
                              "scores": {"potted plant": 0.8} if decided else None} for v in views],
                "n_views": len(views), "jev": {"forced": variants.get("forced")}}}

        self.gate = VerificationGate({"enabled": True, "approach": True, "on_error": "reject", "multiview": True,
                                      "multiview_grid": False, "multiview_sam3": "never", "multiview_min_views": 1,
                                      "log_dir": self.folder + "/{video_output_path}"}, "v",
                                     verify=verify, verify_views=verify_views,
                                     ping=lambda url, timeout: {"ok": True, "settings": {
                                         "selector": "jev_sam3", "multiview": True, "multiview_grid": False}})
        self.gate.start_episode(7, "scene", "ep", "potted_plant", goal_object_ids=["42"], max_episode_steps=500)

    def observe(self, step, camera, size=16):
        return self.gate.observe(step, RGB, [mask(size)], [0], [0.5], [OBJECT], [[{"id": "42", "pixels": 10}]],
                                 (*camera, 0.0), -math.radians(30), None)

    def test_first_request_sam3_decides(self):
        self.observe(1, (0.0, 0.0))
        self.statuses.append("correct")
        self.assertEqual(self.gate.handle(request(), step=1), ACCEPTED)
        self.assertEqual(decisions(self.folder)[-1]["model"], "sam3")
        self.assertEqual(self.gate.finish_episode()["sam3_decisions"], 1)

    def test_thresholds_per_mode(self):
        self.observe(1, (0.0, 0.0))
        self.statuses.append("insufficient")
        self.assertEqual(self.gate.handle(request(), step=1), APPROACH)
        self.assertEqual(decisions(self.folder)[-1]["model"], "jev")
        # on the way, JEV unsure: keep approaching (unforced; jev_sam3 sends no margin or vote)
        self.observe(3, (1.0, 0.0))
        self.mv_replies.append("insufficient")
        self.assertEqual(self.gate.handle(request(seq=2, approach_id=1, status=4, steps=2), step=3), APPROACH)
        self.assertEqual(self.mv_calls[-1]["forced"], False)
        self.assertNotIn("sam3_margin", self.mv_calls[-1])
        self.assertEqual(decisions(self.folder)[-1]["model"], "jev_multi")
        # arrival: forced, SAM3 decides (the server's fused belief)
        self.observe(6, (1.5, 0.0), size=24)
        self.mv_replies.append("correct")
        self.assertEqual(self.gate.handle(request(seq=3, at_stop=True, approach_id=1, status=1, steps=6), step=7),
                         ACCEPTED)
        self.assertEqual(self.mv_calls[-1]["forced"], True)
        line = decisions(self.folder)[-1]
        self.assertEqual((line["policy"], line["model"]), ("multiview,arrival,sam3_accept", "sam3"))
        self.assertEqual(self.gate._sam3_cache[(6, 0)], {"potted plant": 0.8})   # SAM3 scores cached per view

    def test_external_approach(self):
        """approach_planner: the eval loop drives the approach; the gate logs its one verdict and no approach
        views or reopens come back while it runs."""
        self.gate.approach_planner = "mppi"
        self.observe(1, (0.0, 0.0))
        self.statuses.append("insufficient")
        self.assertEqual(self.gate.handle(request(), step=1), APPROACH)
        self.assertEqual(self.gate.pending_approach(3)["approach_id"], 1)
        self.assertEqual(self.gate.observe(3, RGB, [mask(24)], [0], [0.5], [OBJECT], None, (1.0, 0.0, 0.0), 0.0,
                                           None, external_approach=True), [])
        result = {"accept": True, "status": "correct", "by": "belief", "end": None, "actions": 9, "counted": 3,
                  "views": [0, 4, 9], "seconds": 30.0}
        self.assertEqual(self.gate.finish_external_approach(request(), 10, result, "log"), ACCEPTED)
        line = decisions(self.folder)[-1]
        self.assertEqual(line["policy"], "approach_planner,belief,accept")
        self.assertEqual((line["approach"]["planner"], line["approach"]["steps"]), ("mppi", 9))
        self.assertIsNone(self.gate.pending_approach(3))
        counts = self.gate.finish_episode()
        self.assertEqual((counts["approach_accepted"], counts["approach_steps"], counts["accepted"]), (1, 9, 1))

    def test_approach_planner_needs_approach(self):
        with self.assertRaises(ValueError):
            VerificationGate({"enabled": False, "approach_planner": "mppi"}, "v")
        with self.assertRaises(ValueError):
            VerificationGate({"enabled": False, "approach": True, "on_error": "reject", "approach_planner": "x"}, "v")

    def test_failure_forced(self):
        self.observe(1, (0.0, 0.0))
        self.statuses.append("insufficient")
        self.gate.handle(request(), step=1)
        self.mv_replies.append("wrong")
        self.assertEqual(self.gate.handle(request(seq=2, approach_id=1, status=3, steps=40), step=41), REJECTED)
        self.assertEqual(self.mv_calls[-1]["forced"], True)
        self.assertEqual(decisions(self.folder)[-1]["policy"], "multiview,failure,sam3_reject")


def jev_verdict(status):
    return {"accepted": status == "correct", "label": "potted plant" if status != "wrong" else "vase",
            "status": status, "source": "jev", "judged": "jev", "qwen": None,
            "gate": {"score": 0.3 if status != "insufficient" else 0.7, "threshold": 0.5076,
                     "insufficient": status == "insufficient", "entropy": 0.1, "contention": 0.01},
            "jev": {"probabilities": {"potted plant": 0.9}}}


if __name__ == "__main__":
    unittest.main()
