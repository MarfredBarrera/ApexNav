"""GT planning proxy over run logs (no VLM): did the trajectory produce a usable, GT-sufficient frame?

    python -m active_approach.proxy <run_dir>/<judge> [--ids-file F] [--policies ...]

Per run (steps.jsonl; run with --full-trajectory so the whole policy is logged):
  usable    facing (offset <= 20 deg) and >= 400 mask px (what decide.py counts as evidence)
  suff      usable and GT-sufficient (vis >= 0.6, trunc <= 0.15, area >= 1%)
  reach     any suff frame; a2r = actions to the first one
  reach2    two suff frames >= 0.5 m apart or >= 20 deg around the object (two independent clear views)
  trunc_end the last usable frame has GT trunc >= 0.3; bumps = forward steps that did not move
"""
import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from active_approach.judges import gt_sufficient


def one(path):
    rows, seen = [], set()
    for line in open(path / "steps.jsonl"):
        r = json.loads(line)
        if "vis" not in r:
            continue
        rows.append(r)
    res = json.load(open(path / "result.json"))
    suff, usable, bumps, prev = [], [], 0, None
    for r in rows:
        v = r["vis"]
        if prev is not None and prev.get("action") == "move_forward" and r["step"] != prev["step"]:
            if math.hypot(r["position"][0] - prev["position"][0], r["position"][2] - prev["position"][2]) < 0.05:
                bumps += 1
        prev = r
        if r["step"] in seen:
            continue
        seen.add(r["step"])
        if v["pixels"] >= 400 and r["facing_offset_deg"] <= 20:
            usable.append(r)
            if gt_sufficient(v):
                suff.append(r)
    two = False
    for i, a in enumerate(suff):
        for b in suff[i + 1:]:
            if math.hypot(a["position"][0] - b["position"][0], a["position"][2] - b["position"][2]) >= 0.5:
                two = True
    traj = res.get("trajectory") or {}
    return {"reach": bool(suff), "a2r": suff[0]["step"] if suff else None, "reach2": two,
            "usable": len(usable), "trunc_end": bool(usable and usable[-1]["vis"]["trunc_frac"] >= 0.3),
            "bumps": bumps, "actions": traj.get("actions", res["actions"]), "end": traj.get("end", res["end"]),
            "tag": res["tag"], "role": res["role"]}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("--ids-file", type=Path)
    ap.add_argument("--policies", nargs="*")
    ap.add_argument("--by-tag", action="store_true")
    args = ap.parse_args(argv)
    ids = set(open(args.ids_file).read().split()) if args.ids_file else None
    table = defaultdict(dict)
    for pdir in sorted(args.root.iterdir()):
        if args.policies and pdir.name not in args.policies:
            continue
        for run in sorted(pdir.iterdir()):
            if (run / "result.json").exists() and (ids is None or run.name in ids):
                table[pdir.name][run.name] = one(run)
    common = set.intersection(*(set(v) for v in table.values())) if table else set()
    print(f"{len(common)} common runs")
    hdr = f"{'policy':18s} {'reach':>6s} {'reach2':>6s} {'a2r':>5s} {'usable':>6s} {'truncE':>6s} {'bumps':>5s} {'acts':>5s} {'stuck/budget':>12s}"
    groups = [("all", lambda r: True)]
    if args.by_tag:
        tags = sorted({r["tag"] for v in table.values() for r in v.values()})
        groups += [(t, (lambda t: lambda r: r["tag"] == t)(t)) for t in tags]
    for gname, keep in groups:
        print(f"== {gname}")
        print(hdr)
        for p, runs in table.items():
            rs = [runs[i] for i in sorted(common) if keep(runs[i])]
            if not rs:
                continue
            a2r = [r["a2r"] for r in rs if r["a2r"] is not None]
            print(f"{p:18s} {sum(r['reach'] for r in rs):6d} {sum(r['reach2'] for r in rs):6d} "
                  f"{np.mean(a2r) if a2r else float('nan'):5.1f} {np.mean([r['usable'] for r in rs]):6.1f} "
                  f"{sum(r['trunc_end'] for r in rs):6d} {sum(r['bumps'] for r in rs):5d} "
                  f"{np.mean([r['actions'] for r in rs]):5.1f} {sum(r['end'] in ('stuck', 'budget', 'no_path') for r in rs):12d}"
                  f"   (n={len(rs)})")


if __name__ == "__main__":
    main()
