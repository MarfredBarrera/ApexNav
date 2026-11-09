"""Metrics tables and contact sheets for the active-approach tests.

    python -m active_approach.report videos/active_approach/runs/<name> [--policies ...] [--eye eye.tsv]

Writes <run>/summary.csv (judge x policy x subset), <run>/paired.csv (each policy vs stay and
random_ring on the same scenarios: b = this policy right & baseline wrong, c = the reverse, exact McNemar
p), and <run>/sheets/: before_after_<judge>_NN.jpg (start | each policy's decision frame) and
wrong_<judge>_<policy>_NN.jpg (every wrong decision: start | decision frame) for eye checks.

Metrics (decisions are accept / reject; an undecided run was forced, so every run decides):
    wrong        FA + FR: FA = confuser accepted, FR = target rejected
    start_sure   the start frame decided (JEV sure on /verify)
    unsure       starts the judge was unsure about: the population the approach is for
    resolved     unsure starts that became sure at a stop (an unforced /verify_views decided)
    forced       decided by the forced call (arrival / failure / no views)
    actions, path_m, stops   per run (unsure starts only)
``--eye`` (tsv: judge, policy, id, verdict in S/FS/FR/X) relabels eye-checked runs: X drops the scenario.
"""
import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

TILE_W, TILE_H, LINE_H = 320, 240, 14
ORDER = ("stay", "native", "random_ring", "oracle", "fba", "fba_tour", "svr", "svr_tour")


def tile(image, lines, colour=(255, 255, 255)):
    import cv2
    img = cv2.resize(image, (TILE_W, TILE_H))
    band = np.zeros((LINE_H * len(lines) + 4, TILE_W, 3), np.uint8)
    for k, text in enumerate(lines):
        cv2.putText(band, text[:52], (3, LINE_H * (k + 1) - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.38, colour, 1, cv2.LINE_AA)
    return np.vstack([img, band])


def grid(tiles, cols):
    h = max(t.shape[0] for t in tiles)
    tiles = [np.vstack([t, np.zeros((h - t.shape[0], TILE_W, 3), np.uint8)]) for t in tiles]
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]
    return np.vstack(rows)


def vis_line(v):
    dist = "-" if v.get("distance") is None else f"{v['distance']:.1f}m"
    return f"vis {v['vis_frac']:.2f} occ {v['occ_frac']:.2f} trunc {v['trunc_frac']:.2f} area {100 * v['area']:.1f}% {dist}"


def scenario_sheets(root, per_sheet=30, cols=5):
    """<root>/sheet_<role>_<tag>_NN.jpg from <root>/starts (BGR jpgs written by the generator)."""
    import cv2
    root = Path(root)
    scenarios = [json.loads(l) for l in open(root / "scenarios.jsonl")]
    groups = defaultdict(list)
    for s in scenarios:
        groups[(s["role"], s["tag"])].append(s)
    for (role, tag), items in sorted(groups.items()):
        for page in range(0, len(items), per_sheet):
            tiles = []
            for s in items[page:page + per_sheet]:
                img = cv2.imread(str(root / "starts" / f"{s['id']}.jpg"))
                tiles.append(tile(img, [s["id"], f"{role} {s['object_name']} -> '{s['target']}' [{tag}]",
                                        vis_line(s["gt"])]))
            cv2.imwrite(str(root / f"sheet_{role}_{tag}_{page // per_sheet:02d}.jpg"), grid(tiles, cols),
                        [cv2.IMWRITE_JPEG_QUALITY, 80])


def load_results(run, eye=None):
    results = [json.load(open(p)) for p in sorted(Path(run).glob("*/*/*/result.json"))]
    if not eye:
        return results
    drop = {i for (_, _, i), v in eye.items() if v == "X"}
    return [r for r in results if r["id"] not in drop]


def mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def summarize(results):
    rows = []
    groups = defaultdict(list)
    for r in results:
        for subset in ("all", r["role"], r.get("source") or "-"):
            groups[(r["judge"], r["policy"], subset)].append(r)
    mean = lambda xs: round(float(np.mean(xs)), 2) if len(xs) else None
    for (judge, policy, subset), rs in sorted(groups.items(), key=lambda kv: (kv[0][0], ORDER.index(kv[0][1])
                                                                           if kv[0][1] in ORDER else 99, kv[0][2])):
        unsure = [r for r in rs if not r["start_sure"]]
        tags = Counter(t for r in unsure for t in r["failure_tags"])
        rows.append({"judge": judge, "policy": policy, "subset": subset, "n": len(rs),
                     "wrong": sum(not r["correct"] for r in rs),
                     "FA": sum(r["role"] == "confuser" and r["decision"] == "accept" for r in rs),
                     "FR": sum(r["role"] == "target" and r["decision"] != "accept" for r in rs),
                     "start_sure": len(rs) - len(unsure), "unsure": len(unsure),
                     "wrong_unsure": sum(not r["correct"] for r in unsure),
                     "resolved": sum(r["resolved"] for r in unsure),
                     "forced": sum(not r["resolved"] for r in unsure),
                     "wrong_resolved": sum(not r["correct"] for r in unsure if r["resolved"]),
                     "wrong_forced": sum(not r["correct"] for r in unsure if not r["resolved"]),
                     "actions": mean([r["actions"] for r in unsure]), "path_m": mean([r["path_m"] for r in unsure]),
                     "stops": mean([r["stops"] for r in unsure]),
                     "tags": " ".join(f"{k}:{v}" for k, v in sorted(tags.items()))})
    return rows


def paired(results, baselines=("stay", "random_ring")):
    by = defaultdict(dict)
    for r in results:
        by[(r["judge"], r["id"])][r["policy"]] = r
    rows = []
    for judge in sorted({r["judge"] for r in results}):
        policies = sorted({r["policy"] for r in results if r["judge"] == judge},
                          key=lambda p: ORDER.index(p) if p in ORDER else 99)
        for base in baselines:
            for p in policies:
                if p == base:
                    continue
                pairs = [(v[p], v[base]) for (j, _), v in by.items() if j == judge and p in v and base in v]
                b = sum(x["correct"] and not y["correct"] for x, y in pairs)
                c = sum(y["correct"] and not x["correct"] for x, y in pairs)
                rows.append({"judge": judge, "policy": p, "baseline": base, "n": len(pairs),
                             "wrong": sum(not x["correct"] for x, _ in pairs),
                             "wrong_base": sum(not y["correct"] for _, y in pairs),
                             "b_policy_only_right": b, "c_base_only_right": c, "p_mcnemar": round(mcnemar(b, c), 4)})
    return rows


def decision_frame(run_dir):
    """The last saved frame of a run (the decision frame or the final one)."""
    import cv2
    frames = sorted((run_dir / "frames").glob("*_rgb.jpg"))
    if not frames:
        return None
    rgb = cv2.imread(str(frames[-1]))
    mask = cv2.imread(str(frames[-1]).replace("_rgb.jpg", "_mask.png"), 0) > 0
    return rgb, mask, frames[-1].name[:3]


def caption(r):
    v = r["verdict"]
    extra = ""
    if v.get("margin") is not None:
        extra = f" M{v['margin']:.2f} V{v['vote']:.2f}"
    elif v.get("jev_top1"):
        extra = f" jev:{v['jev_top1']}"
    return [f"{r['policy']}: {r['decision']} {'OK' if r['correct'] else 'WRONG'} by {r['decided_by']}{extra}",
            f"act {r['actions']} path {r['path_m']:.1f}m stops {r['stops']} end {r['end']}",
            vis_line(r["final_vis"])[:52], " ".join(r["failure_tags"]) or "-"]


def start_tile(run, judge, policy, r):
    import cv2

    from active_approach.scenarios import outline
    first = run / judge / policy / r["id"] / "frames"
    rgb = cv2.imread(str(first / "000_rgb.jpg"))
    if rgb is None:
        return tile(np.zeros((TILE_H, TILE_W, 3), np.uint8), [r["id"], "no start frame"])
    mask = cv2.imread(str(first / "000_mask.png"), 0) > 0
    return tile(outline(rgb, mask), [r["id"][:52], f"{r['role']} {r['object_name']} -> '{r['target']}' [{r['tag']}]",
                                     "START " + vis_line(r["start_vis"])[:44]])


def before_after(run, results, rows_per_sheet=8, only_unsure=True):
    import cv2

    from active_approach.scenarios import outline
    run = Path(run)
    by = defaultdict(dict)
    for r in results:
        by[(r["judge"], r["id"])][r["policy"]] = r
    (run / "sheets").mkdir(exist_ok=True)
    for judge in sorted({r["judge"] for r in results}):
        policies = sorted({r["policy"] for r in results if r["judge"] == judge},
                          key=lambda p: ORDER.index(p) if p in ORDER else 99)
        ids = sorted(i for (j, i), v in by.items() if j == judge
                     and not (only_unsure and all(x["start_sure"] for x in v.values())))
        for page in range(0, len(ids), rows_per_sheet):
            tiles = []
            for sid in ids[page:page + rows_per_sheet]:
                runs = by[(judge, sid)]
                any_r = next(iter(runs.values()))
                tiles.append(start_tile(run, judge, any_r["policy"], any_r))
                for p in policies:
                    r = runs.get(p)
                    frame = decision_frame(run / judge / p / sid) if r else None
                    if r is None or frame is None:
                        tiles.append(tile(np.zeros((TILE_H, TILE_W, 3), np.uint8), [p, "not run"]))
                        continue
                    rgb, mask, _ = frame
                    tiles.append(tile(outline(rgb, mask), caption(r), (255, 255, 255) if r["correct"] else (80, 80, 255)))
            cv2.imwrite(str(run / "sheets" / f"before_after_{judge}_{page // rows_per_sheet:02d}.jpg"),
                        grid(tiles, 1 + len(policies)), [cv2.IMWRITE_JPEG_QUALITY, 75])
        for p in policies:
            wrong = [r for r in results if r["judge"] == judge and r["policy"] == p and not r["correct"]]
            for page in range(0, len(wrong), 12):
                tiles = []
                for r in wrong[page:page + 12]:
                    tiles.append(start_tile(run, judge, p, r))
                    rgb, mask, _ = decision_frame(run / judge / p / r["id"])
                    tiles.append(tile(outline(rgb, mask), caption(r), (80, 80, 255)))
                cv2.imwrite(str(run / "sheets" / f"wrong_{judge}_{p}_{page // 12:02d}.jpg"), grid(tiles, 4),
                            [cv2.IMWRITE_JPEG_QUALITY, 75])


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", type=Path)
    parser.add_argument("--eye", type=Path, help="tsv judge, policy, id, S/FS/FR/X")
    parser.add_argument("--no-sheets", action="store_true")
    args = parser.parse_args(argv)
    eye = {}
    if args.eye:
        for line in open(args.eye):
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 4 and not line.startswith("#"):
                eye[tuple(parts[:3])] = parts[3]
    results = load_results(args.run, eye)
    rows = summarize(results)
    write_csv(args.run / "summary.csv", rows)
    pairs = paired(results)
    write_csv(args.run / "paired.csv", pairs)
    for row in rows:
        if row["subset"] in ("all", "target", "confuser"):
            print(" ".join(f"{k}={v}" for k, v in row.items()))
    for row in pairs:
        print(" ".join(f"{k}={v}" for k, v in row.items()))
    if not args.no_sheets:
        before_after(args.run, results)


if __name__ == "__main__":
    main()
