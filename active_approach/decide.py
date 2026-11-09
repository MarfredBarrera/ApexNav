"""Decision flow for the active approach, replayed offline on harness logs (CPU only).

    python -m active_approach.decide <run_dir> [<run_dir> ...] --out <dir> [--policies ...]

The flow (one tracked candidate):
  1. start frame -> jev_sam3 /verify (logged). Sure -> commit (accept iff SAM3's raw top-1 is the target).
  2. approach: every in-view frame is a candidate observation. It counts as evidence only if it is
     usable (facing the object, >= 400 mask px so SAM3 scored it, mask area >= MIN_AREA) and passes the
     pose diversity gate GATE: "pose" counts only frames DIST_M closer than, or ANGLE_DEG around, every
     counted frame; "replace" also lets a duplicate that reads better (area, x0.7 clipped, by REPLACE_GAIN)
     replace the frame it duplicates (CLOSER's eviction, by pose; CLIP whole-frame novelty was rejected,
     see gate_replay.py); "all" counts every usable frame. The start frame is the first counted frame.
  3. each counted frame updates a Dirichlet belief over JEV's 16 naming options (belief.Belief, E1/E2,
     uniform prior of mass 2), weighted by how big the object is: w = min(1, mask area / A_REF) (A_REF None:
     w = 1), so a far, tiny view adds little evidence.
  4. once N_MIN frames and W_MIN evidence mass are counted and the evidence is consistent (TRIGGER "consistent":
     P(JEV's top name outweighs its runner-up) >= P_STAR, whatever the names; "target": the older
     max(P_acc, P_rej) >= P_STAR of belief.Belief.signals), jev_sam3 verifies the counted views (one call)
     and the agent commits to its answer. The belief only says WHEN; SAM3 decides accept / reject (JEV's
     naming direction is unreliable: monitor for tv, flowers for a bouquet, stool for chair).
  5. trajectory over without a trigger -> the same verification, forced (margin FAIL_MARGIN after budget,
     stuck or no path).
Verification over the counted views (VIEWS: the 6 nearest, or ApexNav's select_views; REL_AREA > 0 first drops
views whose mask area is below REL_AREA x the largest counted view's): VERIFY "fused" (default since 2026-11-07,
as the jev_sam3 server): SAM3 accepts iff the target has the highest fused evidential belief over the views
(select.fuse_views); VERIFY "margin" (the older rule): the aggregate margin M >= MARGIN and vote V >= 0.5
(select.aggregate_views; VOTE "weighted" weights the vote like the margin, by mask area); JEV vetoes an accept unless
p_hat(target) >= VETO_K * max other p_hat of the belief (VETO_K 0: no veto). Without any view SAM3 could
score (the object never got big enough), JEV_FALLBACK lets the belief decide (accept iff the target is its
top option); otherwise that rejects.
TV_MONITOR folds JEV's "monitor" into "tv" for tv targets (HM3D's goal is tv_monitor), as jev.fold
folds armchair into chair.

Truth: the scenario role with the eye corrections in EYE_ROLE; EYE_AMBIGUOUS scenarios (labels that
contradict each other by eye) are reported separately and never used to choose parameters. Parameters
are chosen by leave-one-scene-out CV: maximise correct decisions, then fewer actions to the commit.
Writes <out>/decide_grid.csv (every config, all data), decide_cv.json and decide_runs.jsonl.
"""
import argparse
import csv
import itertools
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from active_approach import belief as B

J, SEL = B.J, B.SEL

# eye checks 2026-11-06 (outputs/apexnav_online/active_approach_2026-11-05/belief_iter/eye_*)
EYE_ROLE = {"q5QZSEeHe5g_413_sofa_oblique": "target",   # labelled "bench": a daybed / couch with back cushions
            "Dd4bFSTQ8gi_177_chair_truncated": "target",  # labelled "stool": a high-back counter-height chair
            # user decision 2026-11-07: flowers in a vase count as a potted plant (eye_bouquet sheets)
            "epi254_potted_plant_c0_s61": "target",        # labelled "bouquet": dark flower arrangement
            "sam3_epi259_potted_plant_c7_s268": "target",  # labelled "bouquet": same arrangement, other end
            "sam3_epi37_potted_plant_c21_s264": "target"}  # labelled "flower vase": orange flowers in a floor vase
# (sam3_epi37_potted_plant_c3_s425 red flowers in a glass vase and sam3_epi175_potted_plant_c17_s285 stems in a
# vase were ambiguous only against the flower confusers: their target labels now stand)
EYE_AMBIGUOUS = {"sam3_epi37_potted_plant_c5_s97",       # floor pot with a small plant on top (confuser)
                 "sam3_epi538_chair_c0_s155",            # child's bench by a kids' table (confuser)
                 "bxsVRursffK_38_plant_poor",            # unrecognisable crumpled mesh in a box (target)
                 "eF36g7L6Z9M_110_chair_far",            # bar stool labelled chair; other stools are confusers
                 "bxsVRursffK_135_chair_far",            # child's bench, same kids' corner as epi538 (confuser)
                 "wcojb4TFT35_433_chair_clear",          # single-seat wooden settle with a tall back (confuser)
                 "sam3_epi208_potted_plant_c7_s89",      # unreadable crumpled mesh in a pot (target)
                 "yr17PDCnDDW_141_plant_far"}            # unreadable mesh (target)
MAX_VIEWS = 6
# the formal flow (2026-11-07): GATE all, the 6 nearest views without tiny ones, P(consistent) >= 0.8, fused SAM3;
# JEV's per-view naming over the approach (HM3D) words (perception.jev.approach_opts) comes from the frame logs
DEFAULTS = {"DIST_M": 0.5, "ANGLE_DEG": 30.0, "MIN_AREA": 0.0, "N_MIN": 2, "P_STAR": 0.8, "EVIDENCE": "E1",
            "T": 2.0, "MARGIN": -0.1, "FAIL_MARGIN": -0.1, "VETO_K": 0.0, "TV_MONITOR": False, "JEV_FALLBACK": False,
            "A_REF": None, "W_MIN": 0.0, "GATE": "all", "REPLACE_GAIN": 1.2, "VIEWS": "nearest",
            "VOTE": "count", "REL_AREA": 0.1, "TRIGGER": "consistent", "VERIFY": "fused"}
GRID = {"GATE": ["pose", "replace", "all"], "VIEWS": ["nearest", "select"], "P_STAR": [0.8, 0.9, 0.95, 0.99, 2.0],
        "VETO_K": [0.0, 0.25, 0.5], "JEV_FALLBACK": [False, True], "MIN_AREA": [0.0, 0.005]}
DEFAULTS["ANGLE_DEG"] = 20.0


def truth(run):
    return EYE_ROLE.get(run["id"], run["role"])


def folded_jev(frame, target, tv_monitor):
    p = dict(frame["jev"]["probabilities"])
    if tv_monitor and target == "tv" and "monitor" in p:
        p["tv"] = p.get("tv", 0.0) + p["monitor"]
        p["monitor"] = 0.0
    return p


def usable(frame, P):
    if frame["step"] == 0:
        return True
    return (frame["pose"]["judgeable"] and bool((frame["jev"].get("sam3") or {}).get("scores"))
            and frame["mask"]["area"] >= P["MIN_AREA"])


def quality(frame):
    """How readable a view is: mask area, x0.7 when the mask touches the border (select.aggregate_views)."""
    clipped = any(frame["mask"][k] for k in ("top", "bottom", "left", "right"))
    return frame["mask"]["area"] * (0.7 if clipped else 1.0)


def same_view(frame, other, P):
    """The pose gate's duplicate test: not DIST_M closer and within ANGLE_DEG of ``other``."""
    db = abs((frame["pose"]["bearing"] - other["pose"]["bearing"] + math.pi) % (2 * math.pi) - math.pi)
    return not (frame["pose"]["distance"] <= other["pose"]["distance"] - P["DIST_M"]
                or math.degrees(db) >= P["ANGLE_DEG"])


def admit(frame, counted, P):
    """The diversity gate: (new counted list, changed). pose: only new views; replace: a duplicate that reads
    better (quality) replaces the view it duplicates (CLOSER's eviction, by pose); all: every usable frame."""
    if P["GATE"] == "all":
        return counted + [frame], True
    dups = [i for i, g in enumerate(counted) if same_view(frame, g, P)]
    if not dups:
        return counted + [frame], True
    if P["GATE"] == "replace":
        i = max(dups, key=lambda i: quality(counted[i]))
        if quality(frame) > quality(counted[i]) * P["REPLACE_GAIN"]:
            return counted[:i] + counted[i + 1:] + [frame], True
    return counted, False


def build_belief(counted, target, P):
    belief = B.Belief(target, evidence=P["EVIDENCE"], T=P["T"])
    for g in counted:
        belief.add(g, 1.0 if P["A_REF"] is None else min(1.0, g["mask"]["area"] / P["A_REF"]))
    return belief


def chosen_views(counted, P):
    if P["REL_AREA"] > 0:   # once a clear view exists, far tiny views are noise: keep views >= REL_AREA x the largest
        top = max(f["mask"]["area"] for f in counted)
        counted = [f for f in counted if f["mask"]["area"] >= P["REL_AREA"] * top]
    if P["VIEWS"] == "select":   # ApexNav's select_views (<= 6, best view + quality order, dup 0.3 m / 15 deg)
        from perception_gate.gate import select_views
        chosen, _ = select_views([B.item(f) for f in counted], MAX_VIEWS, 0.3, 15.0)
        return [c["frame"] for c in chosen]
    return sorted(counted, key=lambda f: (f["pose"]["distance"], f["step"]))[:MAX_VIEWS]


def logged_forced(run, views):
    """The status of a logged forced jev_sam3 call over exactly ``views`` (the live server scored SAM3 on views
    below the 400 px the frame log keeps scores for), else None."""
    steps = sorted(f["step"] for f in views)
    for call in run["calls"]:
        if call.get("mode") in ("arrival", "failure") and sorted(call.get("views") or []) == steps:
            return call["status"]
    return None


def weighted_vote(per_view, target):
    """The vote with select.aggregate_views' margin weights (min(1, area / FULL_VIEW_AREA), x CLIPPED_WEIGHT if
    clipped), so far, tiny views cannot outvote close, clear ones (the plain vote counts every view once)."""
    used = [v for v in per_view if v.get("scores") and any(x > 0 for x in v["scores"].values())]
    weights = [min(1.0, v["area"] / SEL.FULL_VIEW_AREA) * (SEL.CLIPPED_WEIGHT if v.get("clipped") else 1.0)
               for v in used]
    total = sum(weights)
    if total <= 0:
        return 0.0
    return sum(w for w, v in zip(weights, used) if SEL.top1(v["scores"])[0] == target) / total


def verify(counted, belief, target, margin, P, run=None):
    """jev_sam3 forced over the chosen views with the JEV belief veto (and fallback without SAM3 views)."""
    views = chosen_views(counted, P)
    per_view = [{"scores": (f["jev"].get("sam3") or {}).get("scores"), "area": f["mask"]["area"],
                 "clipped": any(f["mask"][k] for k in ("top", "bottom", "left", "right"))} for f in views]
    agg = SEL.aggregate_views(per_view, target)
    a = belief.alpha
    t = belief.options.index(target)
    rival = max(a[i] for i in range(len(a)) if i != t)
    if agg["margin"] is not None and P.get("VERIFY", "fused") == "fused":
        accept = SEL.fuse_views(per_view, target)["accept"]
        if accept and P["VETO_K"] > 0:
            accept = a[t] >= P["VETO_K"] * rival
        return "correct" if accept else "wrong"
    if agg["margin"] is None:  # no SAM3 scores in the frame log (< 400 px): the logged call over these views, if any
        logged = None if run is None else logged_forced(run, views)
        if logged is not None:
            return logged
        return "correct" if P["JEV_FALLBACK"] and a[t] >= rival else "wrong"
    vote = agg["vote"] if P["VOTE"] == "count" else weighted_vote(per_view, target)
    accept = agg["margin"] >= margin and vote >= 0.5
    if accept and P["VETO_K"] > 0:
        accept = a[t] >= P["VETO_K"] * rival
    return "correct" if accept else "wrong"


def p_consistent(belief):
    """P(JEV's top name outweighs its runner-up) under the Dirichlet belief, whatever the names (target-free)."""
    from scipy.special import betainc
    a = np.sort(np.asarray(belief.alpha, float))[::-1]
    return 1 - betainc(a[0], a[1], 0.5)


def sure(belief, P):
    """The trigger. target: max(P_acc, P_rej) >= P_STAR (target vs look-alike / vs any other name);
    consistent: p_consistent >= P_STAR (JEV names the object the same way across views)."""
    if P["TRIGGER"] == "consistent":
        return p_consistent(belief) >= P["P_STAR"]
    s = belief.signals()
    return max(s["p_acc"], s["p_rej"]) >= P["P_STAR"]


def flow(run, P):
    """One trajectory through the flow -> {status, by, step, counted, calls}."""
    target = run["target"]
    frames = sorted(run["frames"], key=lambda f: f["step"])
    first = next((c for c in run["calls"] if c.get("mode") == "first"), None)
    if first is not None and first["status"] != "insufficient":
        return {"status": first["status"], "by": "start", "step": 0, "counted": 1, "calls": 1}
    counted, belief, calls = [], None, 1
    for f in frames:
        if not usable(f, P):
            continue
        g = {**f, "jev": {**f["jev"], "probabilities": folded_jev(f, target, P["TV_MONITOR"])}}
        counted, changed = admit(g, counted, P)
        if not changed:
            continue
        belief = build_belief(counted, target, P)
        mass = len(counted) if P["A_REF"] is None else sum(min(1.0, c["mask"]["area"] / P["A_REF"]) for c in counted)
        if f["step"] == 0 or len(counted) < P["N_MIN"] or mass < P["W_MIN"] or target not in belief.options:
            continue
        if sure(belief, P):
            return {"status": verify(counted, belief, target, P["MARGIN"], P, run), "by": "belief", "step": f["step"],
                    "counted": len(counted), "calls": calls + 1}
    if not counted:
        return {"status": "wrong", "by": "empty", "step": None, "counted": 0, "calls": calls}
    if belief.options is None or target not in belief.options:
        return {"status": "wrong", "by": "forced", "step": None, "counted": len(counted), "calls": calls + 1}
    margin = P["FAIL_MARGIN"] if run["end"] in ("budget", "stuck", "no_path") else P["MARGIN"]
    return {"status": verify(counted, belief, target, margin, P, run), "by": "forced", "step": None,
            "counted": len(counted), "calls": calls + 1}


def actions(run, out):
    """Actions until the commit (the trajectory's actions when forced)."""
    if out["step"] is not None:
        return out["step"]
    return (run["result"].get("trajectory") or {}).get("actions", run["result"]["actions"])


def score(runs, P):
    rows = []
    for run in runs:
        out = flow(run, P)
        ok = (out["status"] == "correct") == (truth(run) == "target")
        rows.append({"policy": run["policy"], "id": run["id"], "scene": run["scene"], "truth": truth(run),
                     "logged": run["result"]["decision"], **out, "actions": actions(run, out), "correct": ok})
    return rows


def summary(rows):
    n = len(rows)
    return {"runs": n, "correct": sum(r["correct"] for r in rows),
            "FA": sum(not r["correct"] and r["truth"] != "target" for r in rows),
            "FR": sum(not r["correct"] and r["truth"] == "target" for r in rows),
            "actions": round(float(np.mean([r["actions"] for r in rows])), 2) if n else None,
            "calls": round(float(np.mean([r["calls"] for r in rows])), 2) if n else None,
            "triggered": sum(r["by"] == "belief" for r in rows)}


def configs():
    keys = list(GRID)
    for values in itertools.product(*(GRID[k] for k in keys)):
        yield {**DEFAULTS, **dict(zip(keys, values))}


def key(P):
    return json.dumps({k: P[k] for k in GRID}, sort_keys=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--scenarios", default="videos/active_approach/scenarios_uncertain/scenarios.jsonl")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--policies", nargs="*", help="only these approach policies")
    parser.add_argument("--override", nargs="*", type=Path, help="run dirs whose runs replace the same (policy, id)")
    parser.add_argument("--override-suffix", nargs="*", default=["_seekc", "_seek"],
                        help="policy-name suffixes stripped from the --override runs")
    parser.add_argument("--grid", type=json.loads, help='JSON {param: [values]} replacing GRID (e.g. a reduced grid)')
    args = parser.parse_args(argv)
    if args.grid:
        GRID.clear()
        GRID.update(args.grid)
    runs = [r for d in args.runs for r in B.load_runs(d, args.policies, args.scenarios)]
    if args.override:   # re-runs of some (policy, scenario) pairs, e.g. the "_seek" variants: they replace the base run
        base = {(r["policy"], r["id"]): r for r in runs}
        for d in args.override:
            for r in B.load_runs(d, None, args.scenarios):
                for suffix in args.override_suffix:   # later --override dirs win
                    r["policy"] = r["policy"].removesuffix(suffix)
                if (not args.policies or r["policy"] in args.policies) and (r["policy"], r["id"]) in base:
                    base[(r["policy"], r["id"])] = r
        runs = list(base.values())
    clean = [r for r in runs if r["id"] not in EYE_AMBIGUOUS]
    ambig = [r for r in runs if r["id"] in EYE_AMBIGUOUS]
    args.out.mkdir(parents=True, exist_ok=True)
    # every config on every clean run, cached per run for the CV
    table, grid_rows = {}, []
    for P in configs():
        rows = score(clean, P)
        table[key(P)] = (P, rows)
        grid_rows.append({**{k: P[k] for k in GRID}, **summary(rows)})
    with open(args.out / "decide_grid.csv", "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(grid_rows[0]))
        writer.writeheader()
        writer.writerows(grid_rows)
    # leave-one-scene-out CV
    scenes = sorted({r["scene"] for r in clean})
    held, chosen, worst, best = [], [], 0, 0
    for scene in scenes:
        def fit(item):
            rows = [r for r in item[1] if r["scene"] != scene]
            return (sum(r["correct"] for r in rows), round(-float(np.mean([r["actions"] for r in rows])), 6))
        top = max(fit(item) for item in table.values())
        tied = [item for item in table.values() if fit(item) == top]
        # ties (training scenes cannot tell the configs apart): held-out range over the tied configs
        outcomes = [sum(r["correct"] for r in rows if r["scene"] == scene) for _, rows in tied]
        worst, best = worst + min(outcomes), best + max(outcomes)
        P, rows = tied[0]
        held += [r for r in rows if r["scene"] == scene]
        chosen.append(key(P))
    best_key = max(set(chosen), key=chosen.count)
    P_best, rows_best = table[best_key]
    baseline = {**DEFAULTS, "P_STAR": 2.0, "GATE": "all", "VIEWS": "select"}  # today's forced call over the bank
    logged = [{"correct": (r["result"]["decision"] == "accept") == (truth(r) == "target"), "truth": truth(r),
               "actions": r["result"]["actions"], "calls": len(r["calls"]), "by": "logged"} for r in clean]
    cv = {"scenes": len(scenes), "runs_clean": len(clean), "runs_ambiguous": len(ambig),
          "logged": summary(logged), "baseline_forced_sam3": summary(score(clean, baseline)),
          "cv_held_out": summary(held), "cv_correct_range_over_ties": [worst, best],
          "most_chosen": json.loads(best_key),
          "most_chosen_times": chosen.count(best_key), "most_chosen_all_clean": summary(rows_best),
          "most_chosen_on_ambiguous": summary(score(ambig, P_best))}
    json.dump(cv, open(args.out / "decide_cv.json", "w"), indent=1)
    with open(args.out / "decide_runs.jsonl", "w") as stream:
        for r in rows_best + score(ambig, P_best):
            stream.write(json.dumps(r) + "\n")
    print(json.dumps(cv, indent=1))


if __name__ == "__main__":
    main()
