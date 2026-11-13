"""VerificationGate approach logic (server selector d1) with a fake server (no ROS, no GPU):

    python -m unittest perception_gate.test_approach
"""
import json
import math
import os
import tempfile
import unittest

import numpy as np

from perception_gate.gate import (ACCEPTED, APPROACH, DEFERRED, REJECTED, UNVERIFIED, VerificationGate,
                                  select_views)

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

    def test_approach_handoff(self):
        """approach_planner: far or out of view -> ApexNav approaches; its early re-requests keep approaching
        until the object is close in view (or it arrives), then the gate's planner takes over."""
        self.gate.approach_planner = "mppi"
        self.observe(1, (-3.0, 0.0))
        self.statuses.append("insufficient")
        self.assertEqual(self.gate.handle(request(), step=1), APPROACH)
        far = {"pose": (-3.0, 0.0, 0.0), "masks": [mask(24)], "centroids": [OBJECT]}
        near = {"pose": (0.5, 0.0, 0.0), "masks": [mask(24)], "centroids": [OBJECT]}
        hidden = {"pose": (0.5, 0.0, 0.0), "masks": [], "centroids": []}
        self.assertFalse(self.gate.approach_close(request(), far))      # 5 m away
        self.assertFalse(self.gate.approach_close(request(), hidden))   # no detection
        self.assertTrue(self.gate.approach_close(request(), near))
        seen_here = {"pose": (-3.0, 0.0, 0.0), "masks": [], "centroids": []}
        self.gate.clusters[3]["approach"]["object"] = (-1.0, 0.0)            # 2 m away, evidence from here
        self.assertTrue(self.gate.approach_close(request(), seen_here))
        self.gate.clusters[3]["approach"]["object"] = OBJECT
        self.assertIsNone(self.gate.approach_handoff(request(), near))  # not a re-request: handle it
        self.gate.clusters[3]["approach"]["early"] = 5
        self.assertEqual(self.gate.approach_handoff(request(seq=2, approach_id=1, status=4, steps=8), far), "native")
        self.assertIsNone(self.gate.clusters[3]["approach"]["early"])   # the next new view re-requests
        self.assertEqual(self.gate.approach_handoff(request(seq=3, approach_id=1, status=4, steps=12), near),
                         "planner")
        self.assertEqual(self.gate.approach_handoff(request(seq=4, approach_id=1, status=1, steps=20), far),
                         "planner")                                     # arrived: take over wherever it is
        self.assertEqual(self.gate.counts["approach_steps"], 32)

    def test_approach_planner_needs_approach(self):
        with self.assertRaises(ValueError):
            VerificationGate({"enabled": False, "approach_planner": "mppi"}, "v")
        with self.assertRaises(ValueError):
            VerificationGate({"enabled": False, "approach": True, "on_error": "reject", "approach_planner": "x"}, "v")

    def test_last_mile_switch(self):
        """verification.last_mile: on by default with approach_planner, false turns it off, and false without
        approach_planner is a configuration error (it would be a silent no-op)."""
        base = {"enabled": False, "approach": True, "on_error": "reject", "approach_planner": "mppi"}
        self.assertTrue(VerificationGate(base, "v").last_mile)
        self.assertFalse(VerificationGate({**base, "last_mile": False}, "v").last_mile)
        self.assertFalse(VerificationGate({"enabled": False}, "v").last_mile)
        with self.assertRaises(ValueError):
            VerificationGate({"enabled": False, "last_mile": False}, "v")

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


class LiveFrameTest(unittest.TestCase):
    """LiveApproach._object_xz: planner-frame object point -> sim (x, z), whatever the two frames' offset."""

    def test_object_xz(self):
        from active_approach import geom as G
        from perception_gate.approach import LiveApproach
        live = LiveApproach.__new__(LiveApproach)
        live.G = G
        rng = np.random.default_rng(0)
        for _ in range(20):
            psi, yaw = rng.uniform(-math.pi, math.pi, 2)          # the agent's yaw in each frame
            pose_pf = np.array([*rng.uniform(-5, 5, 2), psi])
            position = np.array([rng.uniform(-5, 5), 0.1, rng.uniform(-5, 5)])
            ahead, left = rng.uniform(-3, 3, 2)
            # planner frame (ROS): x along yaw psi, left = +90 deg
            live.object_pf = pose_pf[:2] + ahead * np.array([math.cos(psi), math.sin(psi)]) \
                + left * np.array([-math.sin(psi), math.cos(psi)])
            want = position[[0, 2]] + ahead * G.forward(yaw) + left * G.forward(yaw + math.pi / 2)
            np.testing.assert_allclose(live._object_xz(position, yaw, pose_pf), want, atol=1e-9)
        # sim left of yaw 0 (looking along -z) is -x
        np.testing.assert_allclose(G.forward(math.pi / 2), [-1.0, 0.0], atol=1e-12)

    def test_goto_turns_toward_a_goal_behind(self):
        from active_approach import geom as G
        from active_approach import local_nav as L
        from active_approach import scene as S
        from perception_gate.approach import LiveApproach
        pos = np.array([1.0, 0.1, -2.0])
        live = LiveApproach.__new__(LiveApproach)
        live.G, live.S = G, S
        live.lmap = L.LocalMap(G.HeightMap(pos[[0, 2]], pos[1]))
        live.pf = L.LocalPathfinder(live.lmap, pos[1])
        live.pf.refresh(pos)
        live.last_action, live.last_pos, live.goto_stall = None, None, 0
        here = pos[[0, 2]]
        for yaw in (0.0, 2.0):
            self.assertEqual(live._goto(pos, yaw, here + 4.0 * G.forward(yaw)), "move_forward")
            self.assertIn(live._goto(pos, yaw, here + 4.0 * G.forward(yaw + 2.8)), ("turn_left", "turn_right"))
            self.assertEqual(live._goto(pos, yaw, here + 4.0 * G.forward(yaw + 1.2)), "turn_left")


class LiveSeedTest(unittest.TestCase):
    """LiveApproach._seed: the gate's bank views are named and counted through the diversity gate, never commit."""

    def test_seed_counts_distinct_views_only(self):
        from active_approach import decide as D
        from active_approach import geom as G
        from perception_gate.approach import LiveApproach
        live = LiveApproach.__new__(LiveApproach)
        live.D, live.G, live.target, live.log = D, G, "potted plant", None
        live.P = {**D.DEFAULTS, "GATE": "replace"}
        live.counted, live.belief, live.result, live.seen = [], None, None, []
        names = []

        def name(rgb, m, sam3):
            names.append(sam3)
            return {"options": ["lamp", "potted plant"], "probabilities": {"lamp": 0.05, "potted plant": 0.95},
                    "sam3": {"scores": {"potted plant": 0.8, "lamp": 0.1}}}
        live._name = name
        big = np.zeros((48, 64), np.uint8)
        big[4:44, 4:60] = 1                                  # 2240 px >= 400
        live.bank = [{"step": 3, "index": 0, "rgb": RGB, "mask": big, "camera": (0.0, 0.0), "centroid": (2.0, 0.0)},
                     {"step": 4, "index": 0, "rgb": RGB, "mask": big, "camera": (0.1, 0.0), "centroid": (2.0, 0.0)},
                     {"step": 9, "index": 0, "rgb": RGB, "mask": big, "camera": (2.0, 2.0), "centroid": (2.0, 0.0)},
                     {"step": 10, "index": 0, "rgb": RGB, "mask": mask(8), "camera": (2.0, -2.0), "centroid": (2.0, 0.0)}]
        pose = np.array([0.0, 0.1, 0.0])
        live._seed(pose, 0.0, (0.0, 0.0, 0.0))
        self.assertEqual(len(names), 3)                      # the 64 px view is not usable
        self.assertEqual(len(live.counted), 2)               # step 4 repeats step 3 (0.1 m, same bearing)
        self.assertEqual(sorted(f["step"] for f in live.counted), [-4, -2])
        self.assertIsNone(live.result)
        self.assertEqual(len(live.seen), 3)                  # SAM3 verifies over every usable view
        self.assertIsNotNone(live.belief)


def box(r0, r1, c0, c1, shape=(48, 64)):
    m = np.zeros(shape, bool)
    m[r0:r1, c0:c1] = True
    return m


class AssociateTest(unittest.TestCase):
    """approach.associate: the one rule (IoU >= ASSOC_MIN) matching detector masks (ApexNav's or SAM3's) to the
    tracked outline T."""

    def test_sconce_above_orchid_rejected(self):          # 693: SAM3's only candidate is the lamp above the orchid
        from perception_gate.approach import associate
        orchid, sconce = box(28, 44, 26, 38), box(4, 14, 28, 36)
        best, scores = associate([sconce], orchid)
        self.assertIsNone(best)
        self.assertEqual(scores[0][1], 0.0)

    def test_curtain_behind_pedestal_rejected(self):      # 259: the curtain (0.84) beside / behind the pedestal
        from perception_gate.approach import associate
        pedestal = box(20, 46, 26, 36)
        curtain = box(0, 48, 0, 26) | box(0, 18, 26, 64) | box(0, 48, 36, 64)   # around it, not over it
        best, scores = associate([curtain], pedestal)
        self.assertIsNone(best)
        self.assertLess(scores[0][1], 0.3)

    def test_whole_bed_around_bench_rejected(self):       # 304: the outline is the bench at the bed foot (GT couch)
        from perception_gate.approach import associate
        bench, bed = box(36, 46, 14, 50), box(4, 46, 4, 60)
        best, scores = associate([bed], bench)
        self.assertIsNone(best)
        self.assertEqual(scores[0][0], 1.0)               # the overlap coefficient would have accepted it
        self.assertLess(scores[0][1], 0.3)

    def test_same_size_object_accepted(self):
        from perception_gate.approach import associate
        chair = box(10, 40, 20, 44)
        best, scores = associate([box(0, 8, 0, 64), box(11, 41, 21, 45)], chair)
        self.assertEqual(best, 1)
        self.assertGreater(scores[1][1], 0.8)

    def test_tiny_fragment_rejected_whole_object_wins(self):   # a 487 px chair fragment vs the 8.5k whole chair
        from perception_gate.approach import associate
        chair, fragment = box(10, 40, 20, 44), box(12, 16, 22, 26)
        best, scores = associate([fragment], chair)
        self.assertIsNone(best)
        self.assertEqual(scores[0][0], 1.0)
        best, _ = associate([fragment, box(10, 40, 20, 44)], chair)
        self.assertEqual(best, 1)

    def test_ties_by_detector_score_and_empty_inputs(self):
        from perception_gate.approach import associate
        t = box(10, 30, 10, 30)
        self.assertEqual(associate([t, t], t)[0], 0)                         # no detector scores: the first
        self.assertEqual(associate([t, t], t, det_scores=[0.4, 0.9])[0], 1)  # else the higher score
        self.assertEqual(associate([], t), (None, []))
        self.assertIsNone(associate([t], np.zeros((48, 64), bool))[0])   # no outline: no detection


class LiveIdentityTest(unittest.TestCase):
    """LiveApproach._reference / _detect: the seed mask before O exists, SAM3 candidates through associate."""

    def _live(self):
        from perception_gate.approach import LiveApproach
        live = LiveApproach.__new__(LiveApproach)
        live.cloud, live.object_pf, live.target, live.tag, live.actions, live.url = \
            np.zeros((0, 3)), np.array([2.0, 0.0]), "potted plant", "t", 0, "http://x"
        return live

    def test_seed_is_nearest_centroid_within_seed_m(self):
        live = self._live()
        a, b = box(5, 15, 5, 15), box(30, 40, 30, 40)
        ref, source = live._reference(np.ones((48, 64)), [a, b], [(2.6, 0.0), (2.1, 0.1)])
        self.assertEqual(source, "seed")
        self.assertTrue((ref == b).all())
        ref, source = live._reference(np.ones((48, 64)), [a], [(4.0, 0.0)])
        self.assertEqual((source, int(ref.sum())), ("none", 0))

    def test_detect_takes_overlapping_candidate_only(self):
        import base64
        import cv2
        from unittest import mock
        live = self._live()
        orchid, sconce = box(28, 44, 26, 38), box(4, 14, 28, 36)

        def cand(m, word, score):
            return {"mask": base64.b64encode(cv2.imencode(".png", m.astype(np.uint8) * 255)[1]).decode(),
                    "word": word, "score": score, "scores": {word: score}}
        reply = {"ok": True, "candidates": [cand(sconce, "lamp", 0.9), cand(orchid[:, :] & box(30, 44, 0, 64),
                                                                            "potted plant", 0.6)]}
        with mock.patch("perception_gate.client._call", return_value=reply):
            det, log = live._detect(RGB, orchid)
            self.assertEqual(det["word"], "potted plant")
            self.assertEqual([round(x["overlap"], 2) for x in log], [0.0, 1.0])
            reply["candidates"] = reply["candidates"][:1]
            det, log = live._detect(RGB, orchid)
            self.assertIsNone(det)                        # the sconce alone: no detection this frame

    def test_detect_seeds_by_nearest_centre_without_outline(self):
        """No ApexNav seed and no O yet: the SAM3 candidate whose 3D centre is nearest the object point seeds."""
        import base64
        import types
        import cv2
        from unittest import mock
        from active_approach import geom as G
        live = self._live()
        live.sim, live.G = None, G
        # fake camera: a mask's points sit at x = its mean column / 10, z = 0
        live.S = types.SimpleNamespace(
            camera=lambda sim: (None, None),
            backproject=lambda d, m, *cam: np.array([[np.nonzero(m)[1].mean() / 10, 0.0, 0.0]]) if m.any()
            else np.zeros((0, 3)))
        far, near = box(10, 20, 2, 8), box(10, 20, 18, 24)        # centres x = 0.45 m and 2.05 m

        def cand(m, word):
            return {"mask": base64.b64encode(cv2.imencode(".png", m.astype(np.uint8) * 255)[1]).decode(),
                    "word": word, "score": 0.8, "scores": {word: 0.8}}
        reply = {"ok": True, "candidates": [cand(far, "lamp"), cand(near, "potted plant")]}
        empty = np.zeros((48, 64), bool)
        with mock.patch("perception_gate.client._call", return_value=reply):
            det, log = live._detect(RGB, empty, np.ones((48, 64), np.float32), np.array([2.0, 0.0]))
            self.assertEqual(det["word"], "potted plant")
            self.assertEqual([x["seed_dist"] for x in log], [1.55, 0.05])
            det, _ = live._detect(RGB, empty, np.ones((48, 64), np.float32), np.array([5.0, 0.0]))
            self.assertIsNone(det)                        # nothing within SEED_M


class NoEvidenceTest(unittest.TestCase):
    """No evidence is no verdict: an approach that never finds the object, or counts no frame, is undecided, and the
    gate defers the cluster (re-check rule) instead of rejecting it."""

    def _live(self, object_xz, yaw=0.0):
        import types
        from active_approach import geom as G
        from perception_gate.approach import LiveApproach
        live = LiveApproach.__new__(LiveApproach)
        position = np.array([0.0, 0.1, 0.0])
        noop = types.SimpleNamespace(update=lambda *a: None, set_object=lambda *a: None, refresh=lambda *a: None)
        from active_approach import decide as D
        from active_approach import size_opinion as SZ
        live.D, live.G, live.SZ, live.sim, live.log, live.target = D, G, SZ, None, None, "tv"
        live.P, live.size_views, live.size_info = dict(D.DEFAULTS), [], {}
        live.S = types.SimpleNamespace(agent_pose=lambda sim: (position.copy(), yaw, 0.0),
                                       camera=lambda sim: (None, None))
        live.hmap = live.lmap = live.pf = noop
        live.cloud, live.floor_y, live.counted, live.seen, live.belief = np.zeros((0, 3)), 0.1, [], [], None
        live.result, live.last_action, live.last_pos, live.stuck = None, None, None, 0
        live.started, live.seeded, live.goto, live.budget = False, True, False, 40
        live.actions = live.face_turns = 0
        live._object_xz = lambda *a: np.asarray(object_xz, float)
        live._detect = lambda *a: (None, [])           # SAM3 finds nothing
        return live

    def _step(self, live):
        return live.step(RGB, np.ones((48, 64), np.float32), [], [], (0.0, 0.0, 0.0))

    def test_facing_and_nothing_seen_is_undecided(self):
        from active_approach import geom as G
        live = self._live(1.0 * G.forward(0.2))      # 11 deg off: facing
        self.assertIsNone(self._step(live))
        self.assertEqual((live.result["status"], live.result["by"], live.result["end"], live.result["accept"]),
                         ("undecided", "not_found", "facing", False))

    def test_not_facing_turns_then_gives_up_undecided(self):
        from active_approach import geom as G
        live = self._live(1.0 * G.forward(math.pi / 2))   # 90 deg to the left
        self.assertEqual(self._step(live), "turn_left")
        live.face_turns = 12                              # FACE_TURNS used up
        self.assertIsNone(self._step(live))
        self.assertEqual((live.result["status"], live.result["by"]), ("undecided", "not_found"))

    def test_forced_without_counted_frames_is_undecided(self):
        live = self._live((1.0, 0.0))
        live._forced("planner_done")
        self.assertEqual((live.result["status"], live.result["by"], live.result["accept"]),
                         ("undecided", "empty", False))

    def test_meta_carries_p_consistent(self):
        from active_approach import decide as D
        live = self._live((1.0, 0.0))
        live.D, live.meta = D, {"target": "tv"}
        self.assertEqual(live._meta(), {"target": "tv", "p_consistent": 0.0})   # no belief yet: no evidence
        live.belief = type("B", (), {"alpha": np.array([9.0, 1.0])})()
        self.assertAlmostEqual(live._meta()["p_consistent"], D.p_consistent(live.belief))

    def test_gate_defers_and_reopens_after_cooldown_from_approach_end(self):
        t = ApproachTests("test_verdict_mapping")
        t.setUp()
        gate = t.gate
        gate.approach_planner = "mppi"
        t.observe(1, (0.0, 0.0))
        t.statuses.append("insufficient")
        self.assertEqual(gate.handle(request(), step=1), APPROACH)
        result = {"accept": False, "status": "undecided", "by": "not_found", "end": "facing", "actions": 20,
                  "counted": 0, "views": [], "seconds": 5.0}
        self.assertEqual(gate.finish_external_approach(request(), 21, result), DEFERRED)
        self.assertEqual(decisions(t.folder)[-1]["policy"], "approach_planner,not_found,defer")
        counts = gate.finish_episode()
        self.assertEqual((counts["approach_undecided"], counts["deferred"], counts["rejected"]), (1, 1, 0))
        self.assertEqual(t.observe(30, (1.0, 1.0)), [])      # cooldown counts from the approach's end (21)
        self.assertEqual(t.observe(46, (1.0, 1.0)), [3])     # cooldown over, new view: reopened
        self.assertEqual(gate.clusters[3]["state"], UNVERIFIED)


class StopDeltaTest(unittest.TestCase):
    """mppi.MppiV15: the stop tolerance DELTA_eff = DELTA (1 - P_STAR) / (1 - p), clipped to [0.2, 3] DELTA."""

    def test_formula(self):
        from active_approach.mppi import MppiV15
        from active_approach import decide as D
        m = MppiV15.__new__(MppiV15)
        d = MppiV15.DELTA
        self.assertEqual(MppiV15.P_STAR, D.DEFAULTS["P_STAR"])
        self.assertEqual(m.stop_delta(None), d)                      # offline harness: no belief key
        self.assertAlmostEqual(m.stop_delta(0.8), d)                 # at P_STAR: today's DELTA
        self.assertAlmostEqual(m.stop_delta(0.5), d * 0.2 / 0.5)
        self.assertAlmostEqual(m.stop_delta(0.9), 2 * d)
        self.assertAlmostEqual(m.stop_delta(0.0), 0.2 * d)           # no evidence yet
        self.assertAlmostEqual(m.stop_delta(0.999), 3 * d)           # clipped
        self.assertAlmostEqual(m.stop_delta(1.0), 3 * d)

    def test_execute_uses_delta_eff(self):
        import types
        from active_approach.mppi import MppiV15
        m = MppiV15.__new__(MppiV15)
        m._ws_seq, m.warm_info, m._w = None, [], 1.0
        m.top_cut = m.top_seen = m.bot_cut = m.bot_seen = False
        r = {"rollout": np.array([0] * 4), "diff": 0.0, "c0": 1.0, "cost": 1.0, "stay_cost": 1.05, "t*": 3,
             "info": None}

        def obs(meta):
            return types.SimpleNamespace(position=np.zeros(3), yaw=0.0, meta=meta)
        m.done = False
        self.assertIsNone(m.execute(obs({}), r))                     # 0.05 < DELTA 0.1: stop at the best view
        self.assertNotIn("delta", m.last)
        m.done = False
        self.assertEqual(m.execute(obs({"p_consistent": 0.0}), r), "move_forward")   # 0.05 >= 0.02: keep going
        self.assertEqual((m.last["p"], m.last["delta"]), (0.0, 0.02))


class LastMileTest(unittest.TestCase):
    """The gate's last mile after an accept (LiveApproach.start_last_mile): a reachable stop cell by the object, the
    stop rule (within STOP_M, facing O), and the no-progress rule (stop within NEAR_M, else hand back)."""

    def _live(self, position, yaw=0.0, cloud=None, wall=False):
        """A LiveApproach on a real local map: 14 m around (0, 0), all seen and free; O a 0.6 x 0.6 m box at x 2 m
        (sim x / z); ``wall``: a wall at x = 1.0 m from z = -0.5 to 2 m (between the agent and O's near face)."""
        import types
        from active_approach import decide as D
        from active_approach import geom as G
        from active_approach import local_nav as L
        from perception_gate.approach import LiveApproach
        hm = G.HeightMap(np.zeros(2), 0.0)
        hm.seen[:] = True
        if wall:
            ij, ok = hm.cells(np.stack([np.full(51, 1.0), np.linspace(-0.5, 2.0, 51)], 1))
            for di in (0, 1):
                hm.low[ij[ok, 0] + di, ij[ok, 1]] = 1.0
        live = LiveApproach.__new__(LiveApproach)
        live.D, live.G, live.L, live.sim, live.log, live.target = D, G, L, None, None, "chair"
        pos = np.array([position[0], 0.0, position[1]], float)
        live.S = types.SimpleNamespace(agent_pose=lambda sim: (pos.copy(), yaw, 0.0))
        live.lmap = L.LocalMap(hm)
        live.pf = L.LocalPathfinder(live.lmap, 0.0)
        if cloud is None:
            g = np.mgrid[1.8:2.4:0.05, 0.1:0.8:0.1, -0.3:0.3:0.05].reshape(3, -1).T
            cloud = g
        live.cloud, live.floor_y, live.seen, live.view_pts, live.P = cloud, 0.0, [], {}, dict(D.DEFAULTS)
        live.lmap.set_object(cloud)
        live.pf.refresh(pos)
        live.actions, live.started, live.blocked, live.goto_stall = 0, True, False, 0
        live.last_action = live.last_pos = None
        live.lm = live.last_mile = None
        live.result = {"accept": True, "status": "correct"}
        live.start_last_mile(40)
        return live, pos

    def test_stop_goal_is_reachable_and_within_stop_m(self):
        from perception_gate import approach as A
        live, pos = self._live((0.0, 0.0))
        goal, togo = live._stop_goal(pos[[0, 2]])
        self.assertIsNotNone(goal)
        self.assertLessEqual(live._object_distance(goal), A.STOP_M + 1e-6)
        self.assertGreaterEqual(live._object_distance(goal), 0.25)      # outside the inflated object
        self.assertTrue(np.isfinite(live.pf.cost[live.lmap.cell(goal)]))
        self.assertLess(goal[0], 1.8)                                   # the near side, the short way
        self.assertAlmostEqual(togo, 1.2, delta=0.4)

    def test_stop_goal_prefers_clearance(self):
        """With a wall beside the near face, the chosen cell keeps CLEAR_M from the wall when one does."""
        from perception_gate import approach as A
        live, pos = self._live((0.0, -1.5), wall=True)
        goal, _ = live._stop_goal(pos[[0, 2]])
        occupied, _, _ = live.lmap.grids()
        from scipy.ndimage import distance_transform_edt
        clear = distance_transform_edt(~(occupied & ~live.lmap.object)) * 0.1
        self.assertGreaterEqual(clear[live.lmap.cell(goal)], A.CLEAR_M - 1e-6)

    def test_far_drives(self):
        live, _ = self._live((0.0, 0.0))
        self.assertEqual(live._last_mile_step({}, np.array([0.0, 0.0, 0.0]), 0.0) in
                         ("move_forward", "turn_left", "turn_right"), True)
        self.assertIsNone(live.last_mile)

    def test_within_stop_m_faces_then_stops(self):
        from active_approach import geom as G
        live, _ = self._live((1.3, 0.0))
        here = np.array([1.3, 0.0, 0.0])
        yaw_to = G.bearing_yaw(here[[0, 2]], G.ObjectModel(live.cloud, 0.0).center)
        self.assertIn(live._last_mile_step({}, here, yaw_to + math.pi / 2), ("turn_left", "turn_right"))
        self.assertEqual(live._last_mile_step({}, here, yaw_to), "stop")
        self.assertEqual((live.last_mile["end"], live.last_mile["reason"]), ("stop", "reached"))

    def test_no_progress_far_hands_back(self):
        from perception_gate import approach as A
        live, _ = self._live((0.0, 0.0))
        here = np.array([0.0, 0.0, 0.0])
        live.lm["best"], live.lm["best_togo"], live.lm["stall"] = 1.0, 1.0, A.LM_STALL
        live.actions = 5
        self.assertIsNone(live._last_mile_step({}, here, 0.0))
        self.assertEqual((live.last_mile["end"], live.last_mile["reason"]), ("handback", "stall"))
        self.assertIsNone(live.step(RGB, None, [], [], None))           # handed back: no more actions

    def test_no_progress_near_stops(self):
        from active_approach import geom as G
        from perception_gate import approach as A
        live, _ = self._live((0.9, 0.0))                                # 0.9 m from O: > STOP_M, <= NEAR_M
        here = np.array([0.9, 0.0, 0.0])
        live.lm["best"], live.lm["best_togo"], live.lm["stall"] = 0.0, 0.0, A.LM_STALL
        live.actions = 5
        yaw_to = G.bearing_yaw(here[[0, 2]], G.ObjectModel(live.cloud, 0.0).center)
        self.assertEqual(live._last_mile_step({}, here, yaw_to), "stop")
        self.assertEqual((live.last_mile["end"], live.last_mile["reason"]), ("stop", "stall"))

    def test_budget_far_hands_back(self):
        live, _ = self._live((0.0, 0.0))
        live.actions = 40
        self.assertIsNone(live._last_mile_step({}, np.array([0.0, 0.0, 0.0]), 0.0))
        self.assertEqual(live.last_mile["reason"], "budget")

    def test_drives_to_a_stop_in_free_space(self):
        """Closed loop on the map (perfect steps): the last mile ends with "stop" within STOP_M, facing O."""
        from active_approach import geom as G
        from active_approach import scene as S
        from perception_gate import approach as A
        live, _ = self._live((-1.0, 0.5))
        here, yaw = np.array([-1.0, 0.0, 0.5]), math.pi   # facing away
        for _ in range(60):
            live.pf.refresh(here)
            a = live._last_mile_step({}, here, yaw)
            if a == "stop" or a is None:
                break
            if a == "move_forward":
                live.last_pos, live.last_yaw = here.copy(), yaw
                f = G.forward(yaw)
                here = here + np.array([f[0], 0.0, f[1]]) * S.FORWARD_M
            else:
                yaw = G.wrap(yaw + (1 if a == "turn_left" else -1) * G.LATTICE)
        self.assertEqual(a, "stop")
        self.assertLessEqual(live._object_distance(here[[0, 2]]), A.STOP_M)
        center = G.ObjectModel(live.cloud, 0.0).center
        self.assertLessEqual(math.degrees(abs(G.wrap(G.bearing_yaw(here[[0, 2]], center) - yaw))), A.FACE_DEG)

    def test_gate_defers_a_stuck_approach(self):
        t = ApproachTests("test_verdict_mapping")
        t.setUp()
        gate = t.gate
        gate.approach_planner = "mppi_v15"
        t.observe(1, (0.0, 0.0))
        t.statuses.append("insufficient")
        self.assertEqual(gate.handle(request(), step=1), APPROACH)
        result = {"accept": False, "status": "wrong", "by": "forced", "end": "stuck", "actions": 9,
                  "counted": 2, "views": [0, 3], "seconds": 5.0}
        self.assertEqual(gate.finish_external_approach(request(), 10, result), DEFERRED)
        self.assertEqual(decisions(t.folder)[-1]["policy"], "approach_planner,forced,defer")
        t.statuses.append("insufficient")
        self.assertEqual(gate.handle(request(seq=2), step=11), APPROACH)
        result = dict(result, accept=True, status="correct")             # a stuck approach may still accept
        self.assertEqual(gate.finish_external_approach(request(seq=2), 20, result), ACCEPTED)

    def test_goal_is_kept_unless_much_worse(self):
        """Hysteresis: a goal that stays valid is kept over a slightly cheaper one (no left / right flipping)."""
        from perception_gate import approach as A
        live, pos = self._live((0.0, 0.0))
        best, _ = live._stop_goal(pos[[0, 2]])
        cand = [best + 0.1 * np.array([i, j]) for i in range(-3, 4) for j in range(-3, 4)]
        other = next(c for c in cand if 0.15 < np.hypot(*(c - best)) <= 0.35   # another ring cell, a little dearer
                     and live._object_distance(c) <= A.STOP_M - 0.05)
        kept, _ = live._stop_goal(pos[[0, 2]], keep=other)
        self.assertLess(np.hypot(*(kept - other)), 0.08)
        far = np.array([2.1, 0.9])                    # the far side of O: much dearer, dropped
        if live._object_distance(far) <= A.STOP_M:
            new, _ = live._stop_goal(pos[[0, 2]], keep=far)
            self.assertLess(np.hypot(*(new - best)), 0.08)

    def test_o_frozen_in_last_mile(self):
        """O grows from detector masks during the approach; in the last mile only SAM3's mask seeding an empty O."""
        live, _ = self._live((0.0, 0.0))
        self.assertFalse(live._grows("sam3"))          # O exists: frozen
        self.assertFalse(live._grows("detector"))
        live.cloud = np.zeros((0, 3))
        self.assertTrue(live._grows("sam3"))           # first-view accept: seeded once, by either detector
        self.assertTrue(live._grows("detector"))
        self.assertFalse(live._grows("reproj"))
        live.lm = None                                 # the approach itself
        self.assertTrue(live._grows("detector"))


class SameObjectTest(unittest.TestCase):
    """A request for an object a gate approach already decided (another cluster id, same object) inherits the
    verdict: the identity rule (IoU of the request's mask with the decided object's depth-tested projection)."""

    def setUp(self):
        from active_approach import scene as S
        self.S = S
        self.cam = (np.zeros(3), np.eye(3))            # at the origin, looking down -z
        self.depth = np.full((S.HEIGHT, S.WIDTH), 3.0, np.float32)
        self.box = np.zeros((S.HEIGHT, S.WIDTH), bool)
        self.box[S.HEIGHT // 3: 2 * S.HEIGHT // 3, S.WIDTH // 3: S.WIDTH // 2] = True
        self.depth[self.box] = 2.0
        self.points = S.backproject(self.depth, self.box, *self.cam)
        t = ApproachTests("test_verdict_mapping")
        t.setUp()
        self.t, self.gate = t, t.gate

    def frame(self, mask):
        return {"depth": self.depth, "masks": [mask.astype(np.uint8)], "centroids": [(2.0, 0.0, 0.0)],
                "cam": self.cam}

    def test_rejected_object_rejects_a_new_cluster(self):
        from perception_gate.gate import REJECTED
        self.gate.remember_object(self.points, REJECTED)
        state, iou = self.gate.same_as_decided(OBJECT, self.frame(self.box))
        self.assertEqual(state, REJECTED)
        self.assertGreater(iou, 0.5)
        self.assertEqual(self.gate.handle(request(cluster_id=7), step=5, frame=self.frame(self.box)), REJECTED)
        self.assertEqual(decisions(self.t.folder)[-1]["policy"], "same_as_rejected")

    def test_accepted_object_is_inherited(self):
        self.gate.remember_object(self.points, ACCEPTED)
        self.assertEqual(self.gate.handle(request(cluster_id=8), step=5, frame=self.frame(self.box)), ACCEPTED)
        self.assertEqual(decisions(self.t.folder)[-1]["policy"], "same_as_accepted")

    def test_other_object_or_far_mask_is_not_the_same(self):
        from perception_gate.gate import REJECTED
        self.gate.remember_object(self.points, REJECTED)
        other = np.zeros_like(self.box)
        other[:, -self.S.WIDTH // 4:] = True                 # elsewhere in the frame
        self.assertIsNone(self.gate.same_as_decided(OBJECT, self.frame(other)))
        far = self.frame(self.box)
        far["centroids"] = [(6.0, 0.0, 0.0)]                 # the mask is not the requested object
        self.assertIsNone(self.gate.same_as_decided(OBJECT, far))
        self.assertIsNone(self.gate.same_as_decided(OBJECT, None))


class ObjectPointsTest(unittest.TestCase):
    """LiveApproach.object_points: the deciding (nearest verified) view's SAM3 instance, else O."""

    def test_nearest_verified_view_wins(self):
        from active_approach import decide as D
        from perception_gate.approach import LiveApproach
        live = LiveApproach.__new__(LiveApproach)
        live.D, live.P, live.cloud = D, dict(D.DEFAULTS), np.zeros((50, 3))

        def frame(step, dist):
            return {"step": step, "pose": {"distance": dist}, "mask": {"area": 0.2}}
        live.seen = [frame(0, 2.0), frame(1, 1.0), frame(2, 1.5)]
        live.view_pts = {0: np.ones((5, 3)), 2: np.full((4, 3), 2.0)}
        self.assertEqual(len(live.object_points()), 4)       # step 1 is nearest but has no instance: step 2
        live.view_pts = {}
        self.assertEqual(len(live.object_points()), 50)      # no instance: O
        live.seen = []
        self.assertEqual(len(live.object_points()), 50)

    def test_last_mile_measures_against_the_deciding_instance(self):
        """O merged a neighbour 2 m away; the last mile's distance is to the deciding view's instance."""
        t = LastMileTest("test_far_drives")
        merged = np.r_[np.mgrid[1.8:2.4:0.1, 0.1:0.8:0.2, -0.3:0.3:0.1].reshape(3, -1).T,
                       np.mgrid[0.4:0.6:0.1, 0.1:0.8:0.2, 1.0:1.2:0.1].reshape(3, -1).T]   # + a chair at (0.5, 1.1)
        live, _ = t._live((0.0, 0.0), cloud=merged)
        live.seen = [{"step": 3, "pose": {"distance": 1.0}, "mask": {"area": 0.2}}]
        live.view_pts = {3: merged[merged[:, 0] > 1.0]}
        live.lm["obj"] = None
        self.assertAlmostEqual(live._object_distance(np.zeros(2)), 1.8, delta=0.06)   # not 1.1 (the neighbour)


class SizeVetoTest(unittest.TestCase):
    """The couch size opinion is a veto (decide.fused_accept, size_opinion.likelihood): a complete small width
    rules the couch out, a large complete width (e.g. inflated by a merge, test2 ep 181) never confirms it."""

    def fused(self, belief, accept, width):
        from unittest import mock
        from active_approach import decide as D
        info = {}
        with mock.patch.object(D.SEL, "fuse_views", return_value={"accept": accept, "belief": belief}):
            out = D.fused_accept([{}], "couch", dict(D.DEFAULTS), {"W": width, "complete": True}, info)
        self.assertEqual(info["accept_after"], out)
        return out, info

    def test_large_width_never_accepts(self):
        # ep 181: chair narrowly ahead of couch; any couch-like or larger width must leave the reject
        belief = {"chair": 0.0965, "couch": 0.0957, "bed": 0.0004, "lamp": 0.0086, "tv": 0.0002}
        for width in (1.0, 1.2, 1.3, 1.5, 1.8, 2.1, 2.9, 4.0, 6.0):
            out, info = self.fused(belief, False, width)
            self.assertFalse(out, width)
            self.assertFalse(info["accept_before"])

    def test_large_width_never_lowers(self):
        # a real couch with a bed-sized or huge width keeps its accept (only "too small" carries evidence)
        belief = {"chair": 0.30, "couch": 0.32, "bed": 0.31, "tv": 0.07}
        for width in (1.5, 1.8, 2.1, 2.9, 4.0):
            out, info = self.fused(belief, True, width)
            self.assertTrue(out, width)
            self.assertFalse(info["applied"], width)

    def test_small_width_vetoes(self):
        belief = {"chair": 0.30, "couch": 0.40, "bed": 0.05, "tv": 0.05}
        out, info = self.fused(belief, True, 0.85)   # an armchair
        self.assertTrue(info["accept_before"])
        self.assertFalse(out)
        self.assertTrue(info["applied"])
        self.assertEqual(info["likelihood"]["couch"], 1.0)
        self.assertTrue(all(v >= 1.0 for v in info["likelihood"].values()))
