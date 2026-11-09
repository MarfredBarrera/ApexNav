"""Judges: the decision module the approach serves, with ApexNav's two request kinds.

    first(view)          the trigger frame (ApexNav's first /verify of a cluster)
    views(bank, mode)    one decision over the views gathered so far (ApexNav's /verify_views):
                         mode "early" (a stop on the way; may answer insufficient), "arrival" (the policy
                         has no viewpoint left; forced) or "failure" (budget, stuck, no path; forced)
Both return {"status": correct | wrong | insufficient, "source", ...}; correct = accept.

    jev_sam3  3DAwareVerification ``perception.server --selector jev_sam3`` (the closed-loop decision
              module, same server settings as the sync_jev_sam3 run): JEV-27B-VL's frozen gate decides
              only the uncertainty, SAM3 decides. first(): /verify (JEV unsure -> insufficient, else SAM3's
              raw top-1). views(): /verify_views over ApexNav's own ``select_views`` of the bank (<= 6 views,
              nearest first, dup 0.3 m / 15 deg), forced unless "early"; SAM3 decides with the fused belief
              (select.fuse_views). Each view is judged once (as CLOSER's evidence cache): its SAM3 scores
              (from naming() or an earlier reply) and its gate-word JEV probabilities (from an earlier reply)
              are sent back as cache, and a repeated request (same selected views, same mode) is answered
              locally. naming() (/name) names over the approach (HM3D) words for the belief replay.
    gt        GT visibility only (no model; Habitat dry runs): a view is sufficient when the object is
              mostly visible, in frame and large enough; then the scenario's role decides. Forced without a
              sufficient view -> reject (undecided counts as reject).

The object mask is the GT visible mask of the scenario's object (a perfect detector and tracker).
``HttpJudge`` (one frame -> status) stays for ``jev_filter`` (how the uncertain starts were selected).
"""
import json
import urllib.request

import numpy as np

# GT sufficiency: what a clear view of the object means for the gt judge, the oracle's
# best views and the scenario tags (start views must fail it)
GT_MIN_VIS, GT_MAX_TRUNC, GT_MIN_AREA = 0.6, 0.15, 0.01
MIN_MASK_PIXELS = 50      # below this the object counts as not seen (lost)
# ApexNav perception_gate DEFAULTS, as in the sync_jev_sam3 run
MULTIVIEW_MAX_VIEWS, DUP_DISTANCE_M, DUP_ANGLE_DEG = 6, 0.3, 15.0


def gt_sufficient(vis):
    return (vis["vis_frac"] >= GT_MIN_VIS and vis["trunc_frac"] <= GT_MAX_TRUNC
            and vis["area"] >= GT_MIN_AREA)


class GTJudge:
    name = "gt"

    def first(self, view):
        if not gt_sufficient(view["vis"]):
            return {"status": "insufficient", "source": "gt"}
        return {"status": "correct" if view["role"] == "target" else "wrong", "source": "gt"}

    def views(self, bank, mode):
        if any(gt_sufficient(v["vis"]) for v in bank):
            return {"status": "correct" if bank[0]["role"] == "target" else "wrong", "source": "gt",
                    "mode": mode, "n_views": len(bank)}
        if mode == "early":
            return {"status": "insufficient", "source": "gt", "mode": mode, "n_views": len(bank)}
        return {"status": "wrong", "source": "gt_forced", "mode": mode, "n_views": len(bank)}


class GTSeekJudge(GTJudge):
    """Planning proxy (Habitat only): the start is always unsure, so every policy must find a GT-sufficient
    view; ``resolved`` in the report = it banked one (before the budget), at that action cost."""
    name = "gt_seek"

    def first(self, view):
        return {"status": "insufficient", "source": "gt_seek"}


class JevSam3Judge:
    name = "jev_sam3"

    def __init__(self, url):
        from perception_gate.client import ping
        settings = ping(url, timeout=300.0).get("settings", {})
        if settings.get("selector") != "jev_sam3":
            raise RuntimeError(f"{url} serves selector {settings.get('selector')!r}, not jev_sam3")
        self.url, self.settings, self.cache, self.jev_cache, self.memo = url, settings, {}, {}, {}
        self.first_cache, self.first_path = {}, None

    def persist(self, path):
        """Keep first() replies in ``path`` (jsonl): every policy starts from the same frame, so the trigger
        verdict is computed once per scenario (JEV and SAM3 are deterministic)."""
        self.first_path = path
        if path.exists():
            for line in open(path):
                row = json.loads(line)
                self.first_cache[row["scenario"]] = row["reply"]

    def first(self, view):
        if view.get("scenario") in self.first_cache:
            return dict(self.first_cache[view["scenario"]], cached=True)
        reply = self._first(view)
        if view.get("scenario"):
            self.first_cache[view["scenario"]] = reply
            if self.first_path:
                with open(self.first_path, "a") as out:
                    out.write(json.dumps({"scenario": view["scenario"], "reply": reply}) + "\n")
        return reply

    def _first(self, view):
        from perception_gate.client import verify
        verdict = verify(self.url, view["rgb"], view["mask"], view["target"], timeout=300.0,
                         request_id=f"aa_{view['run']}_s{view['step']}")["verdict"]
        gate = verdict.get("gate") or {}
        return {"status": verdict["status"], "source": verdict.get("source"), "label": verdict.get("label"),
                "jev_score": gate.get("score"), "jev_top1": gate.get("top1"), "jev_threshold": gate.get("threshold"),
                "jev_contention_pct": gate.get("pct_contention"),
                "jev_probabilities": gate.get("probabilities"),
                "sam3_top": sorted((verdict.get("scores") or {}).items(), key=lambda kv: -kv[1])[:3]}

    def naming(self, view, sam3=False):
        """JEV's naming distribution of one frame (POST /name; logging only, never a decision):
        {probabilities (raw), top1, entropy, contention, pct_entropy, pct_contention, score, insufficient}."""
        import cv2

        from perception_gate.client import PROTOCOL, _call, encode_png
        # the same pose in the same scenario renders the same frame (other policies, the shared start segment):
        # reuse its outputs (JEV and SAM3 are deterministic)
        pose_key = (view["scenario"], round(view["camera"][0], 3), round(view["camera"][1], 3),
                    round(view.get("yaw", 0.0), 3), bool(sam3))
        memo = getattr(self, "pose_memo", None)
        if memo is None:
            memo = self.pose_memo = {}
        if pose_key in memo:
            out, jev_p, scores = memo[pose_key]
            key = (view["run"], int(view["step"]))
            if scores is not None:
                self.cache[key] = scores
            return out
        payload = {"protocol": PROTOCOL, "request_id": f"aa_{view['run']}_n{view['step']}", "target": view["target"],
                   "image": encode_png(cv2.cvtColor(np.ascontiguousarray(view["rgb"]), cv2.COLOR_RGB2BGR)),
                   "mask": encode_png((np.asarray(view["mask"]) > 0).astype(np.uint8) * 255), "sam3": bool(sam3),
                   "words": "approach"}
        reply = _call(self.url, "/name", payload, timeout=120.0)
        if not reply.get("ok"):
            raise RuntimeError(f"perception server /name: {reply.get('error')}")
        gate, scores = reply.get("gate") or {}, reply.get("sam3_scores")
        key = (view["run"], int(view["step"]))  # the same SAM3 scores views() would compute
        # the naming is over the approach (HM3D) words: never cached for /verify_views, whose JEV gate keeps its
        # own words (jev.name_opts)
        if scores is not None:
            self.cache[key] = scores
        out = {"options": reply["jev"]["options"], "probabilities": reply["jev"]["probabilities"],
               "folded": gate.get("probabilities"), "template": "bare-v1",
               **{k: gate.get(k) for k in ("top1", "entropy", "contention", "pct_entropy", "pct_contention", "score",
                                           "threshold", "insufficient")}}
        if scores is not None:
            others = [v for c, v in scores.items() if c != view["target"]]
            out["sam3"] = {"scores": scores, "margin": scores.get(view["target"], 0.0) - (max(others) if others else 0.0),
                           "top1": max(scores, key=scores.get) if scores else None}
        if len(memo) > 5000:
            memo.clear()
        memo[pose_key] = (out, None, scores)
        return out

    def views(self, bank, mode):
        from perception_gate.client import verify_views
        from perception_gate.gate import select_views
        chosen, best = select_views(bank, MULTIVIEW_MAX_VIEWS, DUP_DISTANCE_M, DUP_ANGLE_DEG)
        run = chosen[0]["run"]
        memo = (run, mode, tuple(v["step"] for v in chosen))
        if memo in self.memo:
            return dict(self.memo[memo], memo=True)
        payload = [{"rgb": v["rgb"], "mask": v["mask"], "input_elevation": 0.0, "step": v["step"], "index": 0,
                    "range_m": v["range_m"], "scores": self.cache.get((run, v["step"])),
                    "jev": self.jev_cache.get((run, v["step"]))} for v in chosen]
        reply = verify_views(self.url, payload, chosen[0]["target"], sam3="never", timeout=600.0,
                             request_id=f"aa_{run}_{mode}", forced=mode != "early")
        verdict = reply["verdict"]
        for v in verdict.get("per_view") or []:
            if v.get("step") is None:
                continue
            if v.get("scores") is not None:
                self.cache[(run, int(v["step"]))] = v["scores"]
            if v.get("probabilities") is not None:
                self.jev_cache[(run, int(v["step"]))] = v["probabilities"]
        jev, aggregate = verdict.get("jev") or {}, verdict.get("aggregate") or {}
        out = {"status": verdict["status"], "source": verdict.get("source"), "mode": mode,
                "views": [v["step"] for v in chosen], "best": best["step"], "n_views": len(chosen),
                "jev_calls": sum(not v.get("jev_cached") for v in verdict.get("per_view") or []),
                "jev_score": jev.get("score"), "jev_top1": jev.get("top1"), "jev_unsure": jev.get("insufficient"),
                "margin": aggregate.get("margin"), "vote": aggregate.get("vote"), "n_used": aggregate.get("n_used"),
                "sam3_labels": [v.get("label") for v in verdict.get("per_view") or []],
                "per_view_jev": [{"step": v.get("step"), "score": v.get("score"), "entropy": v.get("entropy"),
                                  "contention": v.get("contention")} for v in verdict.get("per_view") or []]}
        self.memo[memo] = out
        return out


def make_judge(name, url=None):
    if name == "gt":
        return GTJudge()
    if name == "gt_seek":
        return GTSeekJudge()
    if name == "jev_sam3":
        return JevSam3Judge(url or "http://3dawareverification-dev-1:8770")
    raise ValueError(f"unknown judge {name}")


class HttpJudge:
    """One frame -> {"status": correct|wrong|insufficient, ...} from a judge service (scripts/jev_server.py,
    threshold 0.4664); how ``jev_filter`` selected the uncertain starts."""

    def __init__(self, url="http://jev-judge:8771/judge", name="jev"):
        self.url, self.name = url, name

    def __call__(self, frame):
        import cv2

        from perception_gate.client import encode_png
        if frame["mask"].sum() < MIN_MASK_PIXELS:
            return {"status": "insufficient", "reason": "not_seen"}
        payload = {"target": frame["target"],
                   "image": encode_png(cv2.cvtColor(np.ascontiguousarray(frame["rgb"]), cv2.COLOR_RGB2BGR)),
                   "mask": encode_png((frame["mask"] > 0).astype(np.uint8) * 255)}
        request = urllib.request.Request(self.url, data=json.dumps(payload).encode(),
                                         headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=120) as response:
            reply = json.loads(response.read())
        if reply.get("status") not in ("correct", "wrong", "insufficient"):
            raise RuntimeError(f"judge {self.url}: bad reply {reply}")
        return reply
