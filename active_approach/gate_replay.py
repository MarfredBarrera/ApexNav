"""Offline calibration of CLOSER's CLIP view-diversity gate (perception_gate.diversity) on logged harness runs.

    python -m active_approach.gate_replay candidates <run_dir> <out.json>
    python -m active_approach.gate_replay replay <run_dir> <features.npz> [--out <dir>]

Input: runs made before the gate (``--judge jev_sam3 --log-frames --full-trajectory``): every in-view frame
has JEV's distribution and (from 400 px) SAM3's scores, and every frame the old flow banked has a full-size
frames/NNN_rgb.jpg. Those banked frames are the gate's candidates (the old en-route rule kept only frames
0.5 m closer or 30 deg around; the live gate also sees the frames that rule dropped). ``candidates`` lists
them for embedding (CLIP features: perception.worker SAM3Server.embed, keyed "<policy>/<id>/<step>").

``replay`` decides every trajectory again with belief.py's exact jev_sam3 replays:
    old         bank = all candidates; each later candidate -> /verify_views (unforced) over ApexNav's
                select_views (<= 6, dup 0.3 m / 15 deg)  [the flow the runs were made with]
    gate K d    candidates offered in order to DiversityGate(K, d); only admitted ones are judged:
                /verify_views (unforced) over exactly the cached views
The start frame keeps its logged /verify status. Undecided at the end -> forced over the bank / cache
(margin 0.15 after budget / stuck / no path, else 0.0); an empty cache rejects. Truth: the scenario's role.
Per variant: wrong decisions, decisions that differ from ``old``, judged frames (one JEV + one SAM3 each),
/verify_views calls, decided before the forced call. Writes <out>/gate_replay.csv and gate_runs.jsonl.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from active_approach import belief as B
from perception_gate.diversity import DiversityGate

DELTAS = (0.0, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.12, 0.15, 0.18)
KS = (4, 6)


def candidates(run, run_dir):
    """The run's banked frames (full-size frame saved, JEV-named, facing the object or the start frame),
    in step order, each with its frames.jsonl row."""
    folder = Path(run_dir) / "jev_sam3" / run["policy"] / run["id"] / "frames"
    out = []
    for f in sorted(run["frames"], key=lambda f: f["step"]):
        if (folder / f"{f['step']:03d}_rgb.jpg").exists() and (f["step"] == 0 or f["pose"]["judgeable"]):
            out.append(f)
    return out


def key(run, frame):
    return f"{run['policy']}/{run['id']}/{frame['step']}"


def forced_margin(run):
    return 0.15 if run["end"] in ("budget", "stuck", "no_path") else 0.0


def first_status(run):
    first = next((c for c in run["calls"] if c.get("mode") == "first"), None)
    return None if first is None else first["status"]


def nearest_first(frames):
    return sorted(frames, key=lambda f: (f["pose"]["distance"], f["step"]))


def replay_old(run, cands):
    status, calls, bank = first_status(run), 0, []
    if status is not None and status != "insufficient":
        return {"status": status, "by": "first", "judged": 1, "calls": 0}
    for f in cands:
        bank.append(f)
        if f["step"] == 0:
            continue
        calls += 1
        got = B.verify_views(bank, run["target"], False, 0.0)
        if got != "insufficient":
            return {"status": got, "by": "early", "judged": len(bank), "calls": calls}
    got = B.verify_views(bank, run["target"], True, forced_margin(run)) if bank else "wrong"
    return {"status": got, "by": "forced", "judged": len(bank), "calls": calls + 1}


def replay_gate(run, cands, features, k, delta):
    gate, calls, judged = DiversityGate(k, delta), 0, 0
    status = first_status(run)
    for f in cands:
        if not gate.offer(f, features[key(run, f)])["admitted"]:
            continue
        judged += 1
        if f["step"] == 0:
            if status is not None and status != "insufficient":
                return {"status": status, "by": "first", "judged": judged, "calls": 0}
            continue
        calls += 1
        got = B.verify_views_exact(nearest_first(gate.items), run["target"], False, 0.0)
        if got != "insufficient":
            return {"status": got, "by": "early", "judged": judged, "calls": calls}
    if not gate.items:
        return {"status": "wrong", "by": "forced", "judged": judged, "calls": calls}
    got = B.verify_views_exact(nearest_first(gate.items), run["target"], True, forced_margin(run))
    return {"status": got, "by": "forced", "judged": judged, "calls": calls + 1}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("candidates")
    c.add_argument("run_dir", type=Path)
    c.add_argument("out", type=Path)
    r = sub.add_parser("replay")
    r.add_argument("run_dir", type=Path)
    r.add_argument("features", type=Path)
    r.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    runs = B.load_runs(args.run_dir)
    if args.cmd == "candidates":
        rows = [{"key": key(run, f), "rgb": str(Path(args.run_dir) / "jev_sam3" / run["policy"] / run["id"] / "frames"
                                               / f"{f['step']:03d}_rgb.jpg")}
                for run in runs for f in candidates(run, args.run_dir)]
        json.dump(rows, open(args.out, "w"), indent=1)
        print(f"{len(rows)} candidate frames in {len(runs)} runs -> {args.out}")
        return
    data = np.load(args.features)
    features = {k: data[k] for k in data.files}
    out = args.out or args.run_dir / "gate_replay"
    out.mkdir(parents=True, exist_ok=True)
    variants = [("old", None, None)] + [(f"gate K{k} d{d:.2f}", k, d) for k in KS for d in DELTAS]
    per_run, summary = [], []
    old = {}
    for name, k, d in variants:
        wrong = changed = judged = calls = early = 0
        for run in runs:
            cands = candidates(run, args.run_dir)
            got = replay_old(run, cands) if k is None else replay_gate(run, cands, features, k, d)
            ok = B.correct(run, got["status"])
            rid = (run["policy"], run["id"])
            if k is None:
                old[rid] = got["status"]
            wrong += not ok
            changed += got["status"] != old[rid]
            judged += got["judged"]
            calls += got["calls"]
            early += got["by"] != "forced"
            per_run.append({"variant": name, "policy": run["policy"], "id": run["id"], "role": run["role"],
                            "logged": run["result"]["decision"], **got, "correct": ok})
        n = len(runs)
        summary.append({"variant": name, "runs": n, "wrong": wrong, "changed_vs_old": changed,
                        "judged_per_run": round(judged / n, 2), "calls_per_run": round(calls / n, 2),
                        "decided_before_forced": early})
    with open(out / "gate_replay.csv", "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    with open(out / "gate_runs.jsonl", "w") as stream:
        for row in per_run:
            stream.write(json.dumps(row) + "\n")
    logged_ok = sum(B.correct(r, "correct" if r["result"]["decision"] == "accept" else "wrong") for r in runs)
    print(f"{len(runs)} runs; logged decisions wrong: {len(runs) - logged_ok}")
    for s in summary:
        print(f"{s['variant']:18s} wrong {s['wrong']:3d}  changed {s['changed_vs_old']:3d}  "
              f"judged/run {s['judged_per_run']:5.2f}  calls/run {s['calls_per_run']:5.2f}  "
              f"early {s['decided_before_forced']}")


if __name__ == "__main__":
    main()
