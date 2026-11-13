"""Build the last-mile benchmark: episodes.tsv + episodes/<set>_<ep>.npz (run inside the container).

    python -m lastmile.build            # writes lastmile/bench_data/
    python -m lastmile.build --publish  # + copies it to 3DAwareVerification/outputs/.../lastmile_bench_2026-11-11/

Start pose = agent pose at the first gate ACCEPT (first decisions.jsonl row with state 1; the pose is
that step's entry in records/*/steps.jsonl, logged in the planner frame relative to the episode start,
then mapped to the world frame and checked against a navmesh replay of the logged actions).
"""
import argparse
import collections
import csv
import glob
import json
import math
import os
import shutil
import sys

import numpy as np

from lastmile import bench, gt

V = "videos/"
ROOT3D = os.environ.get("ROOT3D", os.path.expanduser("~/codes/3DAwareVerification/outputs/apexnav_online"))
RUNS = {"test2": ("test2_nolm/gate", "test2_baseline/baseline", "test2_2026-11-10"),
        "heldout": ("heldout_final/gate", "heldout_baseline/baseline", "heldout_2026-11-10"),
        "tuning": ("tuning_final/gate", "tuning_baseline/baseline", "tuning_final_2026-11-11"),
        "fresh400": ("fresh400_final/gate", "fresh400_baseline/baseline", "fresh400_2026-11-11")}   # fresh400: TEST ONLY
LASTMILE_TSV = "postaccept_2026-11-11/lastmile.tsv"   # copied next to the data by --publish; staged here
MAX_STEPS = 500
# no clean semantic instance for the accepted object (checked against the approach cloud): override / use the cloud itself
# gate accepted and the agent was forced to STOP (step cap / stucking) within 1 m: successes
EXTRA_CAPSTOP = [("test2", 173), ("heldout", 70), ("heldout", 597), ("tuning", 179), ("tuning", 761), ("tuning", 809), ("tuning", 459)]
# fresh400 (TEST ONLY): post-accept failures whose accepted cluster overlaps a goal (101/184/254 are non-goal clusters; 435/719 are S now)
FRESH_FAIL = [36, 172, 230, 231, 235, 249, 540, 561, 661, 665, 669, 700, 748, 867, 959, 973]
N_FRESH_SUCCESS = 20
GT_OVERRIDE = {("heldout", 601): 616}          # cloud sits on 'sitting bench' (eye: wicker chair)
GT_PROXY = {("heldout", 322), ("test2", 312)}  # loveseat, no matching HM3D instance: GT = dilated perceived cloud
STRUCT = {"wall", "floor", "ceiling", "door", "window", "unknown", "stairs", "beam", "column", "doorframe", "wall,", "rug", "mat", "curtain"}
N_SUCCESS = 30


def eye_labels():
    out = {}
    fp = f"{ROOT3D}/fresh400_2026-11-11/manual1m.tsv"
    fp = fp if os.path.exists(fp) else "lastmile/bench_data/src/fresh400.manual1m.tsv"
    for r in csv.DictReader(open(fp), delimiter="\t"):
        out[("fresh400", int(r["episode"]), r["arm"])] = r["eye_label"]
    for s, (_, _, d) in RUNS.items():
        if s == "fresh400":
            continue
        p = f"{ROOT3D}/{d}/classes.tsv"
        if not os.path.exists(p):
            p = f"lastmile/bench_data/src/{d}.classes.tsv"
        for r in csv.DictReader(open(p), delimiter="\t"):
            out[(s, int(r["episode"]), r["run"])] = r["eye"]
    return out


def first_accept(d):
    rows = [json.loads(l) for l in open(d + "/verification/decisions.jsonl")]
    for r in rows:
        if r.get("state") == 1:
            return r
    return None


def steps_of(d):
    f = glob.glob(d + "/records/*/*/steps.jsonl")
    return [json.loads(l) for l in open(f[0]) if l.strip()] if f else []


def collect():
    eye = eye_labels()
    cands = {}
    for s, (g, b, _) in RUNS.items():
        for e in sorted(os.listdir(V + g)):
            if not e.startswith("epi"):
                continue
            n, d = int(e[3:]), f"{V}{g}/{e}"
            try:
                E = json.loads(open(d + "/episodes.jsonl").readline())
                acc = first_accept(d)
                steps = steps_of(d)
            except Exception:
                continue
            if acc is None or not steps:
                continue
            at = {x["step"]: x for x in steps}
            if acc["step"] not in at:
                continue
            st = at[acc["step"]]
            appr = []
            for p in glob.glob(f"{d}/verification/approach/{e}_c{acc['cluster']['id']}_s*/object_cloud.npz"):
                step = int(p.split("_s")[-1].split("/")[0])
                if step <= acc["step"]:
                    appr.append((step, p))
            cx, cy = acc["cluster"]["centroid"]
            x, y, yaw = st["pose"]
            gtinfo = acc.get("ground_truth") or {}
            cands[(s, n)] = dict(set=s, ep=n, dir=d, base=f"{V}{b}/{e}", target=E["target_label"], acc=acc, step=acc["step"],
                                 pose=st["pose"], pitch=st["camera_pitch"], centroid=(cx, cy), cdist=math.hypot(cx - x, cy - y),
                                 cloud=max(appr)[1] if appr else None, success=E["success"], final_state=E["final_state"],
                                 correct=gtinfo.get("is_correct_instance"), steps=steps,
                                 pipe_eye=eye.get((s, n, "pipeline")), base_eye=eye.get((s, n, "baseline")))
    return cands


def select(cands):
    lm = list(csv.DictReader(open(f"lastmile/bench_data/src/{LASTMILE_TSV}"), delimiter="\t"))
    sel = collections.OrderedDict()
    for r in lm:
        key = (r["set"], int(r["episode"]))
        tag = "FAIL" if r["class"].startswith("A_") else r["class"] if r["class"] in ("benchonly_1m", "budget") else None
        if tag and key in cands:
            sel[key] = (tag, r["class"], r["note"])
    pool = [c for k, c in cands.items() if k not in sel and c["success"] == 1.0 and c["pipe_eye"] == "S" and c["correct"]]
    pool.sort(key=lambda c: c["cdist"])
    far = [c for c in pool if c["cdist"] > 3.0]
    pick = {id(c): c for c in far[:: max(1, len(far) // 8)][:8]}
    rest = [c for c in pool if id(c) not in pick]
    step = len(rest) / (N_SUCCESS - len(pick))
    for i in range(N_SUCCESS - len(pick)):
        c = rest[int(i * step)]
        pick[id(c)] = c
    for c in pick.values():
        sel[(c["set"], c["ep"])] = ("SUCCESS", "S", "")
    return sel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--publish", action="store_true")
    ap.add_argument("--extra", action="store_true", help="only (re)build the FAIL_nostop episodes and merge them into episodes.tsv")
    a = ap.parse_args()
    out = bench.BENCH
    os.makedirs(out + "/episodes", exist_ok=True)
    if a.publish:
        publish(out)
        return
    import quaternion
    sys.path.insert(0, "scripts")
    from sample_episodes import iterator_episodes
    from scipy.spatial import cKDTree
    eps, _ = iterator_episodes("hm3dv2")
    cands = collect()
    sel = select(cands)
    if a.extra:      # gate accepted, agent reached the object but never issued STOP
        sel = collections.OrderedDict((k, ("SUCCESS_capstop", "capstop", "forced STOP (cap/stucking) within 1 m")) for k in EXTRA_CAPSTOP)
        for n in FRESH_FAIL:
            sel[("fresh400", n)] = ("FAIL", "fresh400_postaccept", "TEST ONLY")
        pool = sorted((c for k, c in cands.items() if k[0] == "fresh400" and k[1] not in FRESH_FAIL and c["success"] == 1.0
                       and c["pipe_eye"] == "S" and c["correct"]), key=lambda c: c["cdist"])
        step = len(pool) / N_FRESH_SUCCESS
        for i in range(N_FRESH_SUCCESS):
            sel[("fresh400", pool[int(i * step)]["ep"])] = ("SUCCESS", "S", "TEST ONLY")
    print(len(cands), "episodes with an accept;", collections.Counter(v[0] for v in sel.values()), flush=True)
    byscene = collections.defaultdict(list)
    for k in sel:
        byscene[eps[k[1]].scene_id].append(k)
    rows = []
    for sid, keys in byscene.items():
        scene_dir, short = sid.split("/")[-2], sid.split("/")[-1].split(".")[0]
        short = scene_dir.split("-", 1)[1]
        sim = bench.make_sim(short, render=False)
        so = gt.SceneObjects(scene_dir, short)
        for k in keys:
            c, (tag, cls, note) = cands[k], sel[k]
            X = eps[k[1]]
            sp, q = np.array(X.start_position), X.start_rotation
            Q = np.quaternion(q[3], q[0], q[1], q[2])
            rot = lambda v: quaternion.rotate_vectors(Q, np.asarray(v, float))
            p2w = lambda x, y: sp + rot([-y, 0, -x])
            # replay the logged actions with the navmesh (no renderer)
            import habitat_sim
            st = habitat_sim.AgentState()
            st.position, st.rotation = sp.astype(np.float32), Q
            sim.get_agent(0).set_state(st)
            err = 0.0
            for s in c["steps"]:
                if s["step"] > c["step"]:
                    break
                a_ = s["action"]
                if a_ in (1, 2, 3):
                    sim.get_agent(0).act(("move_forward", "turn_left", "turn_right")[a_ - 1])
            pos, yaw = bench.pose(sim)
            x, y, _ = c["pose"]
            cf = p2w(x, y)
            err = float(np.hypot(*(pos - cf)[[0, 2]]))
            # perceived goal cloud
            if c["cloud"]:
                gc = np.load(c["cloud"])["cloud"].astype(np.float32)
                src = "approach_cloud"
            else:
                w = p2w(*c["centroid"])
                gc = np.array([[w[0], pos[1], w[2]]], np.float32)
                src = "cluster_centroid"
            ctr = gc.mean(0)
            # GT accepted object: the semantic instance (goal ids or any same-category) nearest the perceived centroid
            ids = set(so.category_ids(X.object_category)) | {int(g.object_id) for g in X.goals}
            best = None
            for i in ids:
                Pi = so.points(i)
                Pi = Pi[np.abs(Pi[:, 1] - pos[1]) < 2.0]
                if len(Pi) == 0:
                    continue
                d = float(np.hypot(Pi[:, 0] - ctr[0], Pi[:, 2] - ctr[2]).min())
                if best is None or d < best[0]:
                    best = (d, i)
            fallback = ""
            if k in GT_OVERRIDE:
                best, fallback = (0.0, GT_OVERRIDE[k]), "gt_override"
            elif best is None or (best[0] > 1.0 and src == "approach_cloud" and k not in GT_PROXY):     # eye-verified real objects outside the category list (loveseat, vase, ...)
                near = np.hypot(so.P[:, 0] - ctr[0], so.P[:, 2] - ctr[2]) < 1.5
                near &= np.abs(so.P[:, 1] - pos[1]) < 2.0
                hx = {int(h): i for i, (h, nm) in so.objs.items() if not (STRUCT & set(nm.lower().split()))}
                cnt = collections.Counter(int(h) for h in so.K[near] if int(h) in hx)
                if cnt:
                    h, _ = cnt.most_common(1)[0]
                    gi_ = hx[h]
                    best, fallback = (0.0, gi_), f"gt_fallback:{so.objs[gi_][1]}"
            if k in GT_PROXY:
                fallback = "gt_proxy=perceived_cloud"
            row = dict(set=k[0], ep=k[1], tag=tag, scene=short, target=c["target"], cat=X.object_category,
                       accept_step=c["step"], remaining_steps=MAX_STEPS - c["step"], note=(note + " " + fallback).strip(), cls=cls)
            if best is None:
                row["note"] += " NO GT"
                rows.append(row)
                continue
            gi = best[1] if best else -1
            gp = so.points(gi) if gi >= 0 else gc
            if k in GT_PROXY:
                gp = gc
            gp = gp[np.abs(gp[:, 1] - pos[1]) < 2.0]
            if len(gp) > 30000:
                gp = gp[np.random.RandomState(0).choice(len(gp), 30000, replace=False)]
            d_acc, dy = bench.score_point(pos, gp)
            snap = np.array(sim.pathfinder.snap_point(pos))
            row.update(start_x=round(pos[0], 3), start_y=round(pos[1], 3), start_z=round(pos[2], 3), start_yaw=round(yaw, 4),
                       start_pitch=round(c["pitch"], 3), accept_dist=round(d_acc, 2), centroid_dist=round(c["cdist"], 2),
                       goal_source=src, gt_instance=f"{gi}:{so.objs[gi][1]}" if k not in GT_PROXY else "proxy:cloud", gt_is_goal=int(gi in {int(g.object_id) for g in X.goals}),
                       gt_vs_perceived_m=round(best[0], 2), native_bench=int(c["success"]), native_eye=c["pipe_eye"] or "",
                       baseline_eye=c["base_eye"] or "", pose_replay_err_m=round(err, 3), navmesh_snap_m=round(float(np.linalg.norm(snap - pos)), 3))
            np.savez_compressed(f"{out}/episodes/{k[0]}_{k[1]}.npz", position=pos, yaw=yaw, pitch=c["pitch"], goal_cloud=gc, gt_points=gp,
                                gt_instance=gi)
            rows.append(row)
            print(k, tag, row["accept_dist"], src, "replay_err", round(err, 3), flush=True)
        sim.close()
        del so
    cols = ["set", "ep", "tag", "scene", "target", "cat", "accept_step", "remaining_steps", "start_x", "start_y", "start_z", "start_yaw",
            "start_pitch", "accept_dist", "centroid_dist", "goal_source", "gt_instance", "gt_is_goal", "gt_vs_perceived_m",
            "native_bench", "native_eye", "baseline_eye", "cls", "pose_replay_err_m", "navmesh_snap_m", "note"]
    if a.extra:
        old = [r for r in csv.DictReader(open(out + "/episodes.tsv"), delimiter="\t")
               if (r["set"], int(r["ep"])) not in set(sel)]
        cols = list(old[0].keys()) if old else cols
        rows = old + rows
    else:
        rows.sort(key=lambda r: (r["tag"], r["set"], r["ep"]))
    with open(out + "/episodes.tsv", "w") as f:
        f.write("\t".join(cols) + "\n")
        for r in rows:
            f.write("\t".join(str(r.get(c_, "")) for c_ in cols) + "\n")


def publish(out):
    dst = os.path.join(ROOT3D, "lastmile_bench_2026-11-11")
    os.makedirs(dst, exist_ok=True)
    for name in ("episodes.tsv", "episodes", "results"):
        p = os.path.join(out, name)
        if os.path.isdir(p):
            shutil.copytree(p, os.path.join(dst, name), dirs_exist_ok=True)
        elif os.path.exists(p):
            shutil.copy(p, dst)


if __name__ == "__main__":
    main()
