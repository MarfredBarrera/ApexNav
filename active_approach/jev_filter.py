"""Keep only starts the agent itself would judge uncertain (JEV "can't tell" signal, judges.py jev).

    python -m active_approach.jev_filter --scenarios A/scenarios.jsonl B/scenarios.jsonl \
        --out videos/active_approach/scenarios_uncertain --cap 30

Each start is re-rendered (scene.set_pose), its GT object mask outlined and judged by the jev judge; the reply
(u, entropy, look-alike contention, top-1, status) is stored as "jev_start". Kept: status insufficient
(u >= the frozen 45%-approach threshold). Real starts (source jev_real) were selected on the logged request
(detector mask); they are kept whatever the GT-mask score says, which is stored for comparison. Synthetic starts
are capped at --cap per (role, tag): scenes are visited in a seeded order and a group is no longer scored once
it is full; duplicates (same id) keep the
first file's. --drop: eye-check lists of starts to leave out; real starts whose eye labels conflict across runs are left
out too (dropped.tsv). Writes scenarios.jsonl, jev_scores.jsonl (every start scored) and sheets of the kept
starts.
"""
import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

from active_approach import scene as S
from active_approach.judges import HttpJudge
from active_approach.report import grid, tile, vis_line
from active_approach.scenarios import outline


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scenarios", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--judge-url")
    parser.add_argument("--cap", type=int, default=30)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--drop", type=Path, nargs="*", default=[],
                        help="eye-check tsv files (id, reason) of starts to leave out")
    parser.add_argument("--no-refill", action="store_true",
                        help="score no new starts (after the final eye check, so nothing unchecked comes in)")
    args = parser.parse_args(argv)
    dropped = {}
    for path in args.drop:
        for line in open(path).read().splitlines()[1:]:
            sid, reason = line.split("\t", 1)
            dropped[sid] = reason
    judge = HttpJudge(args.judge_url or "http://jev-judge:8771/judge")
    seen, items = set(), []
    for path in args.scenarios:
        for line in open(path):
            s = json.loads(line)
            if s["id"] not in seen and s["id"] not in dropped:
                seen.add(s["id"])
                items.append(s)
    # real requests whose eye labels disagree for the same object (goal in one run, not_goal in another)
    roles = defaultdict(set)
    for s in items:
        if s.get("source") == "jev_real":
            roles[(s["scene"], s["object_id"], s["target"])].add(s["role"])
    conflict = {k for k, v in roles.items() if len(v) > 1}
    for s in items:
        if s.get("source") == "jev_real" and (s["scene"], s["object_id"], s["target"]) in conflict:
            dropped[s["id"]] = "label conflict: the same object is goal in one run and not_goal in another"
    items = [s for s in items if s["id"] not in dropped]
    (args.out / "starts").mkdir(parents=True, exist_ok=True)
    scores_path = args.out / "jev_scores.jsonl"
    done = {}
    if scores_path.exists():
        for line in open(scores_path):
            d = json.loads(line)
            done[d["id"]] = d["jev_start"]
    log = open(scores_path, "a")
    by_scene = defaultdict(list)
    for s in items:
        by_scene[s["scene"]].append(s)
    uncertain = Counter()   # synthetic (role, tag) groups stop being scored once they hold --cap uncertain starts
    scenes = sorted(by_scene)
    random.Random(args.seed).shuffle(scenes)  # so the groups that fill up early are not from the same scenes
    for scene in scenes:
        its, sim = by_scene[scene], None
        for s in its:
            synthetic = s.get("source") != "jev_real"
            if synthetic and s["id"] not in done and (args.no_refill or uncertain[(s["role"], s["tag"])] >= args.cap):
                continue
            if s["id"] not in done or not Path(args.out / "starts" / f"{s['id']}.jpg").exists():
                sim = sim or S.make_sim(scene, args.gpu)
                st = s["start"]
                obs = S.set_pose(sim, st["position"], st["yaw"], st.get("pitch", 0.0))
                rgb = np.asarray(obs["rgb"])[..., :3]
                mask = np.asarray(obs["semantic"]).squeeze() == s["object_id"]
                if s["id"] not in done:
                    reply = judge({"rgb": rgb, "mask": mask, "target": s["target"]})
                    done[s["id"]] = {k: reply.get(k) for k in ("status", "u", "entropy", "con", "top1")}
                    log.write(json.dumps({"id": s["id"], "jev_start": done[s["id"]]}) + "\n")
                    log.flush()
                cv2.imwrite(str(args.out / "starts" / f"{s['id']}.jpg"), cv2.cvtColor(outline(rgb, mask), cv2.COLOR_RGB2BGR),
                            [cv2.IMWRITE_JPEG_QUALITY, 85])
            s["jev_start"] = done[s["id"]]
            if synthetic and s["jev_start"]["status"] == "insufficient":
                uncertain[(s["role"], s["tag"])] += 1
        if sim:
            sim.close()
        print(scene, len(done), flush=True)
    rng = random.Random(args.seed)
    real = [s for s in items if s.get("source") == "jev_real"]
    synth = [s for s in items if s.get("source") != "jev_real" and s.get("jev_start", {}).get("status") == "insufficient"]
    rng.shuffle(synth)
    groups = defaultdict(list)
    for s in synth:
        s["source"] = "synthetic"
        groups[(s["role"], s["tag"])].append(s)
    keep = real + [s for g in groups.values() for s in g[:args.cap]]
    keep.sort(key=lambda s: (s["source"], s["role"], s["tag"], s["id"]))
    with open(args.out / "scenarios.jsonl", "w") as f:
        for s in keep:
            f.write(json.dumps(s) + "\n")
    sheet_groups = defaultdict(list)
    for s in keep:
        sheet_groups[(s["source"], s["role"], s["tag"])].append(s)
    for (source, role, tag), its in sorted(sheet_groups.items()):
        for page in range(0, len(its), 30):
            tiles = []
            for s in its[page:page + 30]:
                j = s["jev_start"]
                extra = f"u {j['u']} jev:{j['top1']}" + (" AMBIG" if s.get("ambiguous_match") else "")
                tiles.append(tile(cv2.imread(str(args.out / "starts" / f"{s['id']}.jpg")),
                                  [s["id"][:50], f"{role} {s['object_name']} -> '{s['target']}' [{tag}]",
                                   vis_line(s["gt"]), extra]))
            cv2.imwrite(str(args.out / f"sheet_{source}_{role}_{tag}_{page // 30:02d}.jpg"), grid(tiles, 5),
                        [cv2.IMWRITE_JPEG_QUALITY, 80])
    with open(args.out / "dropped.tsv", "w") as f:
        f.write("id\treason\n" + "".join(f"{k}\t{v}\n" for k, v in sorted(dropped.items())))
    scored = [s for s in items if s.get("source") != "jev_real" and "jev_start" in s]
    print("synthetic scored", len(scored), "uncertain", len(synth))
    print(json.dumps({f"{k[0]}/{k[1]}/{k[2]}": v for k, v in
                      sorted(Counter((s["source"], s["role"], s["tag"]) for s in keep).items())}, indent=1))


if __name__ == "__main__":
    main()
