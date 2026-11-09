"""Offline replay of the Dirichlet belief update (docs/active_approach_planners.md section 6). CPU only.

    python -m active_approach.belief videos/active_approach/runs/<run> [--policies ...] [--out <dir>]

Input: harness runs made with ``--judge jev_sam3 --log-frames --full-trajectory``: every in-view frame of the
whole trajectory has JEV's naming distribution (one /name call per frame) and, from 400 mask px, the SAM3
scores /verify_views would compute, so every jev_sam3 call can be replayed exactly:
    /verify        JevGate.decide on the frame's distribution; sure -> SAM3 raw top-1 == target
    /verify_views  JevGate.decide on the mean distribution of ApexNav's ``select_views`` (imported from
                   perception_gate) of the bank; unforced and unsure -> insufficient; else
                   select.aggregate_views: accept iff M >= margin and V >= 0.5
``perception.jev`` / ``perception.select`` are byte copies in videos/active_approach/replay_deps (md5.txt), with
the frozen gate (threshold 0.5076). ``validate`` replays every logged jev_sam3 call and counts mismatches.

The belief decides only WHEN to call jev_sam3 again (the start /verify was unsure; the start frame is evidence):
    today   no belief: /verify_views (unforced) on each judgeable frame that is new by ApexNav's approach_view
            rule (0.5 m closer or 30 deg around every judged frame); bank = judgeable frames
    B1      P-u, E1, w = m, view-cell pooling, T-gate (gate score on p_hat < 0.5076 and vacuity <= u*)
    B2      P-u, E2 (temperature T), w = r' n m (kernel novelty), T-dom (max(P_acc, P_rej) >= P*, S - a >= 1.5)
    B3      weighted SPRT, w = r' n m, |Lambda| >= L*
    ablations: B1/B2 without weights, B2 without novelty, B2 with E1/E4, B2 with P-cat, every variant with
    C-best (/verify on the frame with the largest r' m) instead of C-views
Still insufficient -> re-trigger only after S - S_last >= 1.0 with a new view cell. End of the trajectory ->
one forced /verify_views (margin 0.15 after budget / stuck / no path, else 0.0). An empty bank rejects.

Thresholds (u*, P*, T, L*) and the reliability model r are fitted with scene-grouped 5-fold CV; objective
wrong + 0.02 calls. Truth: the scenario's eye-checked role. Outputs <out>/belief_summary.csv,
belief_runs.jsonl, pareto.csv, validate.json.
"""
import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

DEPS = Path(__file__).resolve().parents[1] / "videos/active_approach/replay_deps"
sys.path.insert(0, str(DEPS))
from perception import jev as J  # noqa: E402  (byte copy of 3DAwareVerification perception/jev.py)
from perception import select as SEL  # noqa: E402

GATE = J.JevGate.load(DEPS / "jev_gate.json")
PRIOR_MASS = 2.0
EPS = 0.02
LAMBDA_CALLS = 0.02


# ------------------------------------------------------------------ data
def load_runs(run_dir, policies=None, scenarios=None):
    run_dir = Path(run_dir)
    info = {}
    if scenarios:
        info = {json.loads(l)["id"]: json.loads(l) for l in open(scenarios)}
    runs = []
    for result_path in sorted(run_dir.glob("jev_sam3/*/*/result.json")):
        result = json.load(open(result_path))
        if policies and result["policy"] not in policies:
            continue
        folder = result_path.parent
        if not (folder / "frames.jsonl").exists():
            continue
        frames = [json.loads(l) for l in open(folder / "frames.jsonl")]
        frames = [f for f in frames if f.get("jev")]
        steps = [json.loads(l) for l in open(folder / "steps.jsonl")]
        calls = [s["judge"] for s in steps if s.get("judge") and s["judge"].get("source") != "gt"]
        scen = info.get(result["id"], {})
        runs.append({"id": result["id"], "policy": result["policy"], "role": result["role"],
                     "target": result["target"], "scene": scen.get("scene", result["id"].split("_")[0]),
                     "source": result.get("source"), "result": result, "frames": frames, "calls": calls,
                     "end": (result.get("trajectory") or {}).get("end", result["end"])})
    return runs


def item(frame):
    """A frame as an ApexNav evidence item for select_views."""
    from perception_gate.evidence import view_quality
    pose, mask = frame["pose"], frame["mask"]
    cam = np.array([pose["position"][0], pose["position"][2]])
    b = pose["bearing"]
    centroid = cam - pose["distance"] * np.array([math.cos(b), math.sin(b)])
    clipped = any(mask[k] for k in ("top", "bottom", "left", "right"))
    return {"step": frame["step"], "index": 0, "camera": (float(cam[0]), float(cam[1])),
            "centroid": (float(centroid[0]), float(centroid[1])),
            "quality": view_quality(0.5, mask["area"], clipped), "frame": frame}


# ------------------------------------------------------------------ exact jev_sam3 replays
def verify(frame, target):
    status, _ = GATE.decide(frame["jev"]["probabilities"], target)
    if status == "insufficient":
        return "insufficient"
    scores = (frame["jev"].get("sam3") or {}).get("scores")
    if not scores:
        return None  # not replayable (no SAM3 scores below 400 px)
    return "correct" if SEL.top1(scores)[0] == target else "wrong"


def verify_views(frames, target, forced, margin):
    from perception_gate.gate import select_views
    if not frames:
        return "wrong"
    chosen, _ = select_views([item(f) for f in frames], 6, 0.3, 15.0)
    chosen = [c["frame"] for c in chosen]
    status, _ = GATE.decide(J.average([f["jev"]["probabilities"] for f in chosen]), target)
    if status == "insufficient" and not forced:
        return "insufficient"
    per_view = []
    for f in chosen:
        scores = (f["jev"].get("sam3") or {}).get("scores")
        m = f["mask"]
        per_view.append({"scores": scores, "area": m["area"], "clipped": any(m[k] for k in ("top", "bottom", "left", "right"))})
    agg = SEL.aggregate_views(per_view, target)
    ok = agg["margin"] is not None and agg["margin"] >= margin and agg["vote"] >= 0.5
    return "correct" if ok else "wrong"


def validate(runs):
    """Replay every logged jev_sam3 call from the frame logs; count status mismatches."""
    n = bad = skipped = 0
    examples = []
    for r in runs:
        by_step = {f["step"]: f for f in r["frames"]}
        for call in r["calls"]:
            mode = call.get("mode")
            if mode == "first":
                f = by_step.get(0)
                if f is None:
                    skipped += 1
                    continue
                got = verify(f, r["target"])
            elif call.get("views"):
                frames = [by_step.get(s) for s in call["views"]]
                if any(f is None for f in frames):
                    skipped += 1
                    continue
                margin = 0.15 if mode == "failure" else 0.0
                got = verify_views_exact(frames, r["target"], mode != "early", margin)
            else:
                continue
            if got is None:
                skipped += 1
                continue
            n += 1
            if got != call["status"]:
                bad += 1
                examples.append({"run": r["id"], "policy": r["policy"], "mode": mode, "logged": call["status"],
                                 "replay": got})
    return {"calls": n, "mismatches": bad, "skipped": skipped, "examples": examples[:10]}


def verify_views_exact(frames, target, forced, margin):
    """/verify_views on exactly these frames (as logged; no re-selection)."""
    status, _ = GATE.decide(J.average([f["jev"]["probabilities"] for f in frames]), target)
    if status == "insufficient" and not forced:
        return "insufficient"
    per_view = [{"scores": (f["jev"].get("sam3") or {}).get("scores"), "area": f["mask"]["area"],
                 "clipped": any(f["mask"][k] for k in ("top", "bottom", "left", "right"))} for f in frames]
    agg = SEL.aggregate_views(per_view, target)
    ok = agg["margin"] is not None and agg["margin"] >= margin and agg["vote"] >= 0.5
    return "correct" if ok else "wrong"


# ------------------------------------------------------------------ weights
def mask_weight(f):
    """m_v = clip((IoU - 0.2)/0.3, 0, 1); 0 below 400 px or when the mask's depth median is more than 0.5 m from
    the tracked cloud centre's distance; the start frame (no cloud yet) counts 1."""
    m = f["mask"]
    if m["px"] < 400:
        return 0.0
    if m.get("reprojection_iou") is None:
        return 1.0
    if abs(m["depth_median"] - f["pose"]["distance"]) > 0.5:
        return 0.0
    return float(np.clip((m["reprojection_iou"] - 0.2) / 0.3, 0, 1))


def rel_features(f, u0):
    p = f.get("pred") or {}
    area = max(p.get("area", f["mask"]["area"]), 1e-4)
    tr = p.get("trunc", 0.0)
    touches = float(any(f["mask"][k] for k in ("top", "bottom", "left", "right")))
    u0 = min(max(u0, 1e-3), 1 - 1e-3)
    return [1.0, math.log10(area), p.get("occ", 0.0), tr, max(0.0, tr - 0.5), p.get("facing", 0.5), touches,
            math.log(u0 / (1 - u0))]


def fit_logistic(X, y, w, l2=1e-2, iters=50):
    beta = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X @ beta))
        grad = X.T @ (w * (p - y)) / w.sum() + l2 * np.r_[0, beta[1:]]
        hess = (X.T * (w * p * (1 - p))) @ X / w.sum() + l2 * np.eye(len(beta))
        beta -= np.linalg.solve(hess, grad)
    return beta


def jev_correct(f, target, role):
    top = J.top1(J.fold(f["jev"]["probabilities"], target))
    return (top == target) if role == "target" else (top != target)


def cell(f):
    b = f["pose"]["bearing"]
    d = f["pose"]["distance"]
    return (int(math.floor((math.degrees(b) % 360) / 30)), 0 if d < 1.2 else 1 if d < 2 else 2 if d < 3 else 3)


# ------------------------------------------------------------------ belief
class Belief:
    def __init__(self, target, prior="u", evidence="E1", T=2.0, pooling="none", novelty=True, pcat=None):
        self.target, self.evidence, self.T, self.pooling, self.novelty = target, evidence, T, pooling, novelty
        self.options = None
        self.prior, self.pcat = prior, pcat
        self.frames, self.cells = [], defaultdict(list)
        self.Lambda = 0.0

    def _init(self, folded):
        self.options = list(folded)
        K = len(self.options)
        if self.prior == "cat" and self.pcat:
            pi = np.array([self.pcat.get(o, 0.0) + 0.5 for o in self.options])
            self.alpha0 = PRIOR_MASS * pi / pi.sum()
        else:
            self.alpha0 = np.full(K, PRIOR_MASS / K)

    def _e(self, folded):
        p = np.array([folded[o] for o in self.options])
        if self.evidence == "E2":
            q = p ** (1 / self.T)
            return q / q.sum()
        if self.evidence == "E3":
            q = np.where(p >= 0.05, p, 0)
            return q / q.sum() if q.sum() else p
        if self.evidence == "E4":
            h = J.entropy(dict(zip(self.options, p)))
            e = np.zeros_like(p)
            e[int(np.argmax(p))] = 1 - h
            return e
        return p

    def novelty_of(self, f):
        if not self.novelty or not self.frames:
            return 1.0
        k = 0.0
        for g, _, _ in self.frames:
            db = abs((f["pose"]["bearing"] - g["pose"]["bearing"] + math.pi) % (2 * math.pi) - math.pi)
            dd = f["pose"]["distance"] - g["pose"]["distance"]
            do = f["pose"]["offset_deg"] - g["pose"]["offset_deg"]
            k = max(k, math.exp(-(math.degrees(db) / 20) ** 2 - (dd / 0.5) ** 2 - (do / 15) ** 2))
        return 1 - k

    def add(self, f, w):
        folded = J.fold(f["jev"]["probabilities"], self.target)
        if self.options is None:
            self._init(folded)
        e = self._e(folded)
        self.frames.append((f, w, e))
        self.cells[cell(f)].append((w, e))
        p = np.array([folded[o] for o in self.options])
        t = self.options.index(self.target) if self.target in self.options else None
        if t is not None:
            rival = max(p[i] for i in range(len(p)) if i != t)
            self.Lambda += w * math.log((p[t] + EPS) / (rival + EPS))

    @property
    def alpha(self):
        if self.pooling == "cell":
            total = self.alpha0.copy()
            for entries in self.cells.values():
                ws = np.array([w for w, _ in entries])
                if ws.sum() <= 0:
                    continue
                e = sum(w * e for w, e in entries) / ws.sum()
                total += ws.max() * e
            return total
        return self.alpha0 + sum(w * e for _, w, e in self.frames)

    def signals(self):
        from scipy.special import betainc
        a = self.alpha
        S = a.sum()
        p_hat = dict(zip(self.options, a / S))
        score = GATE.assess(p_hat, self.target)["score"]
        t = self.options.index(self.target) if self.target in self.options else None
        out = {"S": S, "u": PRIOR_MASS / S, "gate_score": score, "p_acc": 0.0, "p_rej": 0.0}
        if t is not None:
            conf = [self.options.index(c) for c in J.contention_confusers(self.target) if c in self.options]
            if conf:
                c = max(conf, key=lambda i: a[i])
                out["p_acc"] = 1 - betainc(a[t], a[c], 0.5)
            n = max((i for i in range(len(a)) if i != t), key=lambda i: a[i])
            out["p_rej"] = betainc(a[t], a[n], 0.5)
        return out


# ------------------------------------------------------------------ one replay
def replay(run, variant, params, rel=None):
    """-> {decision correct/wrong, calls, frames_used, forced, fired_by}."""
    target, frames = run["target"], run["frames"]
    if not frames:
        return None
    first = run["calls"][0] if run["calls"] else None
    if first is None or first.get("mode") != "first":
        return None
    if first["status"] != "insufficient":
        return {"status": first["status"], "calls": 1, "frames": 1, "forced": False, "start_sure": True}
    u0 = first.get("jev_score") or 0.5
    kind = variant["kind"]
    calls, bank, judged = 1, [], []
    belief = Belief(target, variant.get("prior", "u"), variant.get("evidence", "E1"), params.get("T", 2.0),
                    variant.get("pooling", "none"), variant.get("novelty", True), variant.get("pcat"))
    last_S, last_cells = -math.inf, set()
    for i, f in enumerate(frames):
        weighted = variant.get("weights", "m")
        m = mask_weight(f)
        if weighted == "none":
            w = 1.0 if f["mask"]["px"] >= 400 else 0.0
        elif weighted == "m":
            w = m
        else:  # r' n m
            r = rel(f, u0) if rel else 0.75
            w = max(0.0, 2 * r - 1) * belief.novelty_of(f) * m
        if kind == "today":
            if not f["pose"]["judgeable"] or not (f["jev"].get("sam3") or {}).get("scores"):
                continue
            cam = np.array([f["pose"]["position"][0], f["pose"]["position"][2]])
            new = all(f["pose"]["distance"] <= j["pose"]["distance"] - 0.5 or
                      abs((f["pose"]["bearing"] - j["pose"]["bearing"] + math.pi) % (2 * math.pi) - math.pi)
                      >= math.radians(30) for j in judged)
            bank.append(f)
            if i == 0:
                judged.append(f)
                continue
            if new:
                judged.append(f)
                calls += 1
                status = verify_views(bank, target, False, 0.0)
                if status != "insufficient":
                    return {"status": status, "calls": calls, "frames": i + 1, "forced": False, "start_sure": False}
            continue
        belief.add(f, w)
        if w > 0 and (f["jev"].get("sam3") or {}).get("scores"):
            bank.append((f, w))
        if i == 0:
            last_S, last_cells = belief.alpha.sum(), {cell(f)}
            continue
        sig = belief.signals()
        if kind == "B1":
            fire = sig["gate_score"] < GATE.threshold and sig["u"] <= params["u"]
        elif kind == "B2":
            fire = max(sig["p_acc"], sig["p_rej"]) >= params["P"] and sig["S"] - PRIOR_MASS >= 1.5
        else:
            fire = abs(belief.Lambda) >= params["L"]
        if not fire or not bank:
            continue
        cells_now = {cell(g) for g, _ in bank}
        if not (sig["S"] - last_S >= 1.0 and cells_now - last_cells) and last_S > -math.inf and calls > 1:
            continue
        calls += 1
        last_S, last_cells = sig["S"], cells_now
        if variant.get("call", "views") == "best":
            best = max(bank, key=lambda x: x[1])[0]
            status = verify(best, target) or "insufficient"
        else:
            status = verify_views([g for g, _ in bank], target, False, 0.0)
        if status != "insufficient":
            return {"status": status, "calls": calls, "frames": i + 1, "forced": False, "start_sure": False}
    margin = 0.15 if run["end"] in ("budget", "stuck", "no_path") else 0.0
    pool = bank if kind == "today" else [g for g, _ in bank]
    if not pool:
        pool = [f for f in frames if (f["jev"].get("sam3") or {}).get("scores")]
    status = verify_views(pool, target, True, margin)
    return {"status": status, "calls": calls + 1, "frames": len(frames), "forced": True, "start_sure": False}


def correct(run, status):
    accept = status == "correct"
    return accept == (run["role"] == "target")


VARIANTS = {
    "today": {"kind": "today"},
    "B1": {"kind": "B1", "weights": "m", "pooling": "cell", "evidence": "E1"},
    "B1_noweights": {"kind": "B1", "weights": "none", "pooling": "cell", "evidence": "E1"},
    "B2": {"kind": "B2", "weights": "rnm", "evidence": "E2"},
    "B2_noweights": {"kind": "B2", "weights": "none", "pooling": "cell", "evidence": "E2", "novelty": False},
    "B2_nonovelty": {"kind": "B2", "weights": "rnm", "evidence": "E2", "novelty": False},
    "B2_E1": {"kind": "B2", "weights": "rnm", "evidence": "E1"},
    "B2_E4": {"kind": "B2", "weights": "rnm", "evidence": "E4"},
    "B2_Pcat": {"kind": "B2", "weights": "rnm", "evidence": "E2", "prior": "cat"},
    "B3": {"kind": "B3", "weights": "rnm"},
}
for _name in ("B1", "B2", "B3"):
    VARIANTS[_name + "_best"] = {**VARIANTS[_name], "call": "best"}
GRIDS = {"today": [{}], "B1": [{"u": u} for u in (0.3, 0.4, 0.5, 0.6, 0.7, 0.85)],
         "B2": [{"P": p, "T": t} for p in (0.8, 0.9, 0.95, 0.99) for t in (1.0, 2.0, 3.0)],
         "B3": [{"L": l} for l in (1.0, 2.0, 2.94, 4.0, 6.0)]}


def evaluate(runs, folds=5, seed=0):
    scenes = sorted({r["scene"] for r in runs})
    rng = np.random.default_rng(seed)
    order = list(rng.permutation(scenes))
    fold_of = {s: k % folds for k, s in enumerate(order)}
    # reliability: scene-grouped cross-fitted logistic on deployed (predicted) features
    rel_models = {}
    for k in range(folds):
        X, y, w = [], [], []
        for r in runs:
            if fold_of[r["scene"]] == k or not r["calls"]:
                continue
            u0 = r["calls"][0].get("jev_score") or 0.5
            n = len(r["frames"])
            for f in r["frames"]:
                X.append(rel_features(f, u0))
                y.append(float(jev_correct(f, r["target"], r["role"])))
                w.append(1.0 / max(n, 1))
        rel_models[k] = fit_logistic(np.array(X), np.array(y), np.array(w)) if X else None
    pcat = defaultdict(lambda: defaultdict(float))
    for r in runs:
        if r["frames"]:
            top = J.top1(J.fold(r["frames"][0]["jev"]["probabilities"], r["target"]))
            pcat[r["target"]][top] += 1
    rows, per_run, pareto = [], [], []
    for name, variant in VARIANTS.items():
        variant = dict(variant)
        if variant.get("prior") == "cat":
            variant["pcat"] = None  # filled per run below
        grid = GRIDS[variant["kind"]]
        results = {}
        for k in range(folds):
            beta = rel_models.get(k)
            rel = (lambda f, u0, b=beta: float(1 / (1 + np.exp(-np.array(rel_features(f, u0)) @ b)))) if beta is not None else None
            train = [r for r in runs if fold_of[r["scene"]] != k]
            test = [r for r in runs if fold_of[r["scene"]] == k]

            def cost(params, rs):
                total = 0.0
                for r in rs:
                    v = dict(variant, pcat=dict(pcat[r["target"]])) if variant.get("prior") == "cat" else variant
                    out = replay(r, v, params, rel)
                    if out is None or out["start_sure"]:
                        continue
                    total += (not correct(r, out["status"])) + LAMBDA_CALLS * out["calls"]
                return total
            best = min(grid, key=lambda p: cost(p, train))
            for r in test:
                v = dict(variant, pcat=dict(pcat[r["target"]])) if variant.get("prior") == "cat" else variant
                out = replay(r, v, best, rel)
                if out is not None:
                    results[(r["policy"], r["id"])] = (r, out, best)
        for policy in sorted({p for p, _ in results}) + ["all"]:
            items = [(r, o) for (p, _), (r, o, _) in results.items() if (policy == "all" or p == policy) and not o["start_sure"]]
            if not items:
                continue
            wrong = [not correct(r, o["status"]) for r, o in items]
            rows.append({"variant": name, "policy": policy, "n": len(items), "wrong": int(sum(wrong)),
                         "FA": sum(r["role"] == "confuser" and o["status"] == "correct" for r, o in items),
                         "FR": sum(r["role"] == "target" and o["status"] != "correct" for r, o in items),
                         "calls": round(float(np.mean([o["calls"] for _, o in items])), 2),
                         "frames": round(float(np.mean([o["frames"] for _, o in items])), 1),
                         "forced": round(float(np.mean([o["forced"] for _, o in items])), 3)})
        for (p, sid), (r, o, params) in results.items():
            per_run.append({"variant": name, "policy": p, "id": sid, "role": r["role"], "status": o["status"],
                            "correct": correct(r, o["status"]), "calls": o["calls"], "frames": o["frames"],
                            "forced": o["forced"], "start_sure": o["start_sure"], "params": params})
        # in-sample Pareto curve over the grid (reliability from the cross-fit models)
        for params in grid:
            outs = []
            for r in runs:
                beta = rel_models.get(fold_of[r["scene"]])
                rel = (lambda f, u0, b=beta: float(1 / (1 + np.exp(-np.array(rel_features(f, u0)) @ b)))) if beta is not None else None
                v = dict(variant, pcat=dict(pcat[r["target"]])) if variant.get("prior") == "cat" else variant
                o = replay(r, v, params, rel)
                if o is not None and not o["start_sure"]:
                    outs.append((r, o))
            if outs:
                pareto.append({"variant": name, "params": json.dumps(params),
                               "wrong": sum(not correct(r, o["status"]) for r, o in outs),
                               "calls": round(float(np.mean([o["calls"] for _, o in outs])), 2), "n": len(outs)})
    return rows, per_run, pareto


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", type=Path)
    parser.add_argument("--policies", nargs="*")
    parser.add_argument("--scenarios", default="videos/active_approach/scenarios_uncertain/scenarios.jsonl")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    out = args.out or args.run / "belief"
    out.mkdir(parents=True, exist_ok=True)
    runs = load_runs(args.run, args.policies, args.scenarios)
    check = validate(runs)
    json.dump(check, open(out / "validate.json", "w"), indent=1)
    print("validate", json.dumps(check)[:400])
    rows, per_run, pareto = evaluate(runs)
    for path, data in ((out / "belief_summary.csv", rows), (out / "pareto.csv", pareto)):
        with open(path, "w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    with open(out / "belief_runs.jsonl", "w") as stream:
        for row in per_run:
            stream.write(json.dumps(row) + "\n")
    for row in rows:
        if row["policy"] == "all":
            print(" ".join(f"{k}={v}" for k, v in row.items()))


if __name__ == "__main__":
    main()
