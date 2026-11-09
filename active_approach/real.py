"""Real starts: re-create the JEV-uncertain ApexNav verification requests in Habitat (HM3D-v2 runs).

    python -m active_approach.real --requests videos/active_approach/jev_real/requests.jsonl \
        --out videos/active_approach/scenarios_jev_real --gpu 3

Input: one line per request (3DAwareVerification outputs/apexnav_online/active_approach_2026-11-05/
_real_requests.py): scene, episode id, the evidence step's planner pose (x, y, yaw), camera pitch, truth, target,
and <name>_input.jpg/_mask.png. The planner pose is in the episode's GPS frame (agent_world_pose:
x = -gps_z, y = -gps_x, yaw = compass), so world = start + R_start (-y, 0, -x), snapped to the navmesh.
The re-rendered frame is checked against the logged input (grey-level correlation) and the judged object is
the HM3D instance with most pixels under the request mask in the re-rendered semantic frame
(match_frac = its share of the mask's labelled pixels; ambiguous below 0.5 or a poor frame match).
Truth (eye label, else prior label; README of reliability_2026-11-05) gives the role: goal -> target,
not_goal -> confuser. Writes scenarios.jsonl (source jev_real; tags from GT visibility as in scenarios.py),
starts/<id>.jpg (re-rendered, outlined) and compare/<id>.jpg (logged | re-rendered) for eye checks.
"""
import argparse
import gzip
import json
import math
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from active_approach import scene as S
from active_approach.report import grid, tile, vis_line
from active_approach.scenarios import CONTENT, outline, tag_of

MIN_CORR, MIN_MATCH = 0.6, 0.5
STRUCTURE = {"wall", "floor", "ceiling", "door frame", "window frame", "unknown", "stairs", "beam", "pillar", "column", "railing", "baseboard"}


_EPISODES = {}


def episode(scene, index):
    """The run's episode: the record name's episode number (epiN_<scene>_<k>_<label>) is the index k within
    the scene's content file (its episode_id values repeat)."""
    if scene not in _EPISODES:
        _EPISODES[scene] = json.load(gzip.open(f"{CONTENT['hm3dv2']}/{scene}.json.gz"))["episodes"]
    eps = _EPISODES[scene]
    return eps[int(index)] if int(index) < len(eps) else None


def world_pose(ep, pose):
    import quaternion
    q = quaternion.from_float_array(np.array(ep["start_rotation"])[[3, 0, 1, 2]])
    local = np.array([-pose[1], 0.0, -pose[0]])
    pos = np.asarray(ep["start_position"]) + quaternion.rotate_vectors(q, local)
    fwd = quaternion.rotate_vectors(q, np.array([0.0, 0.0, -1.0]))
    start_yaw = math.atan2(-fwd[0], -fwd[2])
    return pos, start_yaw + pose[2]


def corr(a, b):
    a = cv2.resize(cv2.cvtColor(a, cv2.COLOR_RGB2GRAY), (160, 120)).astype(float).ravel()
    b = cv2.resize(cv2.cvtColor(b, cv2.COLOR_RGB2GRAY), (160, 120)).astype(float).ravel()
    return float(np.corrcoef(a, b)[0, 1])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--cache", default="videos/active_approach/cache")
    args = parser.parse_args(argv)
    src = args.requests.parent
    items = [json.loads(l) for l in open(args.requests)]
    for d in ("starts", "compare"):
        (args.out / d).mkdir(parents=True, exist_ok=True)
    report, keep = [], []
    by_scene = {}
    for it in items:
        if it["failure"]:
            report.append({"name": it["name"], "failure": it["failure"]})
        else:
            by_scene.setdefault(it["scene"], []).append(it)
    for scene, its in sorted(by_scene.items()):
        names = S.semantic_objects(scene)
        sim = S.make_sim(scene, args.gpu)
        try:
            for it in its:
                rec = {"name": it["name"], "failure": None}
                report.append(rec)
                if it["truth"] not in ("goal", "not_goal"):
                    rec["failure"] = "no_truth"
                    continue
                ep = episode(scene, it["episode_id"])
                if ep is None:
                    rec["failure"] = "no_episode"
                    continue
                logged = cv2.cvtColor(cv2.imread(str(src / f"{it['name']}_input.jpg")), cv2.COLOR_BGR2RGB)
                mask = cv2.imread(str(src / f"{it['name']}_mask.png"), 0) > 0
                def render(pose, pitch):
                    pos, yaw = world_pose(ep, pose)
                    pos = np.asarray(sim.pathfinder.snap_point(pos), float)
                    obs = S.set_pose(sim, pos, yaw, float(pitch or 0.0))
                    return pos, yaw, float(pitch or 0.0), obs, corr(np.asarray(obs["rgb"])[..., :3], logged)

                pos, yaw, pitch, obs, c = render(it["pose"], it["pitch"])
                rec["step"] = it["step"]
                if c < MIN_CORR:  # the judged view may be an earlier frame: the best-matching logged pose
                    best = max(((render(p, pp)[4], k) for k, p, pp in it.get("all_poses", [])), default=(c, None))
                    if best[0] > c:
                        k = best[1]
                        p, pp = next((p, pp) for kk, p, pp in it["all_poses"] if kk == k)
                        pos, yaw, pitch, obs, c = render(p, pp)
                        rec["step"] = k
                rgb = np.asarray(obs["rgb"])[..., :3]
                rec["corr"] = round(c, 3)
                sem = np.asarray(obs["semantic"]).squeeze()
                under = sem[mask]
                under = under[under > 0]
                if not len(under):
                    rec["failure"] = "no_instance_under_mask"
                    continue
                ids, counts = np.unique(under, return_counts=True)
                frac_of = dict(zip(ids.tolist(), (counts / counts.sum()).tolist()))
                objects = [i for i in ids.tolist() if names.get(i, "") not in STRUCTURE] or ids.tolist()
                oid = int(max(objects, key=frac_of.get))   # most pixels under the mask, walls/floors excluded
                frac = float(frac_of[oid])
                rec.update(object_id=oid, object_name=names.get(oid), match_frac=round(frac, 3))
                if it.get("instances"):  # online: instances the run itself saw under the mask
                    rec["logged_top_instance"] = int(it["instances"][0]["id"])
                surface = S.surfaces(scene, args.cache, [oid])[oid]
                vis = S.visibility(obs, surface, oid, *S.camera(sim))
                ambiguous = c < MIN_CORR or frac < MIN_MATCH or (
                    rec.get("logged_top_instance") not in (None, oid))
                if c < MIN_CORR:
                    rec["failure"] = "pose_mismatch"
                floor = np.asarray(sim.pathfinder.snap_point(np.array([surface.center[0], surface.bottom,
                                                                       surface.center[2]])), float)
                sid = f"{it['name']}"
                cv2.imwrite(str(args.out / "compare" / f"{sid}.jpg"), cv2.cvtColor(grid([
                    tile(outline(logged, mask), [f"logged {it['name']}", f"target {it['target']} truth {it['truth']}"]),
                    tile(outline(rgb, sem == oid), [f"re-rendered corr {c:.2f}", f"{names.get(oid)} #{oid} match {frac:.2f}"])],
                    2), cv2.COLOR_RGB2BGR))
                if rec["failure"]:
                    continue
                s = {"id": sid, "source": "jev_real", "dataset": "hm3dv2", "scene": scene,
                     "role": "target" if it["truth"] == "goal" else "confuser", "category": it["target"],
                     "target": it["target"], "object_id": oid, "object_name": names.get(oid),
                     "tag": tag_of(vis) or "poor", "tags": [], "ambiguous_match": bool(ambiguous),
                     "match_frac": round(frac, 3), "frame_corr": round(c, 3),
                     "start": {"position": [round(float(x), 4) for x in pos], "yaw": round(float(yaw), 4),
                               "pitch": pitch},
                     "floor_y": float(floor[1]) if np.all(np.isfinite(floor)) else float(pos[1]),
                     "object_center": [round(float(x), 3) for x in surface.center],
                     "gt": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in vis.items()},
                     "request": {**{k: it[k] for k in ("name", "set", "eye", "truth", "prior_label", "u", "jev_top1")}, "step": rec["step"]}}
                s["tags"] = [s["tag"]]
                keep.append(s)
                cv2.imwrite(str(args.out / "starts" / f"{sid}.jpg"),
                            cv2.cvtColor(outline(rgb, sem == oid), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])
        finally:
            sim.close()
        print(scene, len(keep), flush=True)
    with open(args.out / "scenarios.jsonl", "w") as f:
        for s in sorted(keep, key=lambda s: s["id"]):
            f.write(json.dumps(s) + "\n")
    with open(args.out / "conversion.jsonl", "w") as f:
        for r in report:
            f.write(json.dumps(r) + "\n")
    print(json.dumps(Counter(r["failure"] or "ok" for r in report)))
    print(json.dumps(Counter((s["role"], s["tag"]) for s in keep).most_common()))


if __name__ == "__main__":
    main()
