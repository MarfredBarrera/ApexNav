"""Scenario generator: poor start views of ObjectNav goal objects (and look-alike confusers).

    python -m active_approach.scenarios --dataset hm3dv2 --gpu 3 \
        --out videos/active_approach/scenarios_hm3dv2

For each val scene: goal object instances (from the episodes' goals) and non-goal
look-alikes of each goal category (LOOKALIKES, HM3D semantic names). For each object,
sample start poses on its floor (0.5-6 m away, facing it or with it near the frame edge),
render them, and measure GT visibility (scene.visibility). A start must show the object
(>= MIN_START_PIXELS) but not clearly (fails judges.gt_sufficient). Tag by the dominant
problem:
    occluded   occ_frac >= 0.4 and trunc_frac < 0.2
    truncated  trunc_frac >= 0.35 (object at the frame edge or too close)
    far        distance >= 3.5 m and area < 1% of the frame
    oblique    facing_cos < 0.35 (surface seen edge-on)
    poor       none of the above but still not sufficient
    clear      control: a sufficient start (approach must not break it)
One start per (object, tag), then at most --cap per (role, tag).

Output: <out>/scenarios.jsonl, <out>/starts/<id>.jpg (object outlined) and contact
sheets <out>/sheet_<role>_<tag>.jpg for eye checks.
"""
import argparse
import gzip
import json
import math
import os
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from active_approach import scene as S
from active_approach.judges import gt_sufficient
from active_approach.policies import facing, geodesic

CONTENT = {"hm3dv2": "data/datasets/objectnav/hm3d/v2/val/content",
           "ovon": "data/datasets/ovon/hm3d/val_unseen/content"}
# HM3D-v2 goal category -> the name the judges ask for (perception.client.HOUSEHOLD); OVON verbatim
TARGET_LABEL = {"chair": "chair", "bed": "bed", "plant": "potted plant", "toilet": "toilet",
                "tv_monitor": "tv", "sofa": "couch"}
# Non-goal HM3D objects that look like the goal category
LOOKALIKES = {"chair": {"stool", "bench", "ottoman", "footstool", "pouf"},
              "sofa": {"armchair", "bench", "ottoman", "chaise lounge", "loveseat chair"},
              "bed": {"sofa", "couch", "futon", "sofa bed"},
              "toilet": {"sink", "bathtub", "bidet", "washbasin"},
              "tv_monitor": {"picture", "mirror", "framed picture", "painting", "poster", "whiteboard"},
              "plant": {"vase", "flower vase", "flowers", "decorative plant", "decoration"}}
MIN_START_PIXELS, MIN_FACES = 400, 30
TAGS = ("occluded", "truncated", "far", "oblique", "poor", "clear")


def tag_of(vis):
    if vis["pixels"] < MIN_START_PIXELS:
        return None
    if gt_sufficient(vis):
        return "clear"
    if vis["occ_frac"] >= 0.4 and vis["trunc_frac"] < 0.2:
        return "occluded"
    if vis["trunc_frac"] >= 0.35:
        return "truncated"
    if vis["distance"] is not None and vis["distance"] >= 3.5 and vis["area"] < 0.01:
        return "far"
    if vis["facing_cos"] is not None and vis["facing_cos"] < 0.35:
        return "oblique"
    return "poor"


def scene_goals(dataset, path):
    """[(category, object_id or None, position)] of one content file."""
    data = json.load(gzip.open(path))
    out = []
    for key, goals in data["goals_by_category"].items():
        category = key.split(".basis.glb_", 1)[1]
        for goal in goals:
            oid = goal["object_id"]
            out.append((category, oid if isinstance(oid, int) else None, goal["position"]))
    return out


def object_centers(faces):
    centroid, _, area, oid = faces
    out = {}
    order = np.argsort(oid)
    ids, starts = np.unique(oid[order], return_index=True)
    for i, a, b in zip(ids, starts, list(starts[1:]) + [len(order)]):
        idx = order[a:b]
        w = area[idx][:, None]
        out[int(i)] = ((centroid[idx] * w).sum(0) / max(w.sum(), 1e-9), len(idx))
    return out


def outline(rgb, mask, colour=(255, 230, 0)):
    import cv2
    img = np.ascontiguousarray(rgb[..., :3]).copy()
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(img, contours, -1, colour, 2)
    return img


def objects_for_scene(dataset, scene, goals, names, centers, rng, per_category):
    """[(role, category, object id)]: goal instances and non-goal look-alikes per category."""
    by_cat = defaultdict(set)
    for category, oid, position in goals:
        if oid is None:  # OVON: nearest semantic object to the goal position
            near = [(np.linalg.norm(c - position), i) for i, (c, n) in centers.items() if n >= MIN_FACES]
            d, oid = min(near) if near else (math.inf, None)
            if d > 0.5:
                continue
        if oid in centers and centers[oid][1] >= MIN_FACES:
            by_cat[category].add(oid)
    goal_ids = set().union(*by_cat.values()) if by_cat else set()
    out = []
    for category, ids in sorted(by_cat.items()):
        for oid in rng.sample(sorted(ids), min(per_category, len(ids))):
            out.append(("target", category, oid))
        goal_names = {names.get(i) for i in ids}
        alikes = [i for i, n in names.items() if n in LOOKALIKES.get(category, ()) and n not in goal_names
                  and i not in goal_ids and centers.get(i, (None, 0))[1] >= MIN_FACES]
        for oid in rng.sample(sorted(alikes), min(per_category, len(alikes))):
            out.append(("confuser", category, oid))
    return out


def sample_starts(sim, surface, oid, rng, n):
    """[(position, yaw, vis)] of candidate start poses on the object's floor."""
    pf = sim.pathfinder
    floor = np.asarray(pf.snap_point(np.array([surface.center[0], surface.bottom, surface.center[2]])), float)
    if not np.all(np.isfinite(floor)) or not -0.3 <= surface.bottom - floor[1] <= 1.8:
        return [], None
    out = []
    for _ in range(n):
        p = np.asarray(pf.get_random_navigable_point_near(floor, 6.0, 100), float)
        if not np.all(np.isfinite(p)) or abs(p[1] - floor[1]) > 0.2:
            continue
        flat = np.hypot(*(p - surface.center)[[0, 2]])
        if not 0.5 <= flat <= 6.0 or geodesic(pf, p, floor) > 15.0:
            continue
        offset = 0.0 if rng.random() < 0.6 else rng.choice((-1, 1)) * math.radians(rng.uniform(25, 42))
        yaw = facing(p, surface.center) + offset
        obs = S.set_pose(sim, p, yaw)
        vis = S.visibility(obs, surface, oid, *S.camera(sim))
        out.append((p, yaw, vis, obs))
    return out, float(floor[1])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=sorted(CONTENT), default="hm3dv2")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--scenes", nargs="*", help="scene ids (default: every val scene)")
    parser.add_argument("--per-category", type=int, default=2, help="objects per (scene, category, role)")
    parser.add_argument("--samples", type=int, default=48, help="candidate starts per object")
    parser.add_argument("--clear-rate", type=float, default=0.25, help="objects that also get a clear control")
    parser.add_argument("--cap", type=int, default=30, help="scenarios per (role, tag)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cache", default="videos/active_approach/cache")
    args = parser.parse_args(argv)
    rng = random.Random(args.seed)
    (args.out / "starts").mkdir(parents=True, exist_ok=True)
    files = sorted(Path(CONTENT[args.dataset]).glob("*.json.gz"))
    if args.scenes:
        files = [f for f in files if f.name.split(".")[0] in args.scenes]
    import cv2
    found = []
    for path in files:
        scene = path.name.split(".")[0]
        goals = scene_goals(args.dataset, path)
        names = S.semantic_objects(scene)
        faces = S.object_faces(scene, args.cache)
        centers = object_centers(faces)
        objects = objects_for_scene(args.dataset, scene, goals, names, centers, rng, args.per_category)
        surfaces = S.surfaces(scene, args.cache, {oid for _, _, oid in objects})
        sim = S.make_sim(scene, args.gpu)
        try:
            for role, category, oid in objects:
                starts, floor_y = sample_starts(sim, surfaces[oid], oid, rng, args.samples)
                by_tag = defaultdict(list)
                for p, yaw, vis, obs in starts:
                    tag = tag_of(vis)
                    if tag:
                        by_tag[tag].append((p, yaw, vis, obs))
                if "clear" in by_tag and rng.random() >= args.clear_rate:
                    del by_tag["clear"]
                for tag, items in by_tag.items():
                    p, yaw, vis, obs = rng.choice(items)
                    sid = f"{scene}_{oid}_{category}_{tag}"
                    tags = [t for t, hit in (("occluded", vis["occ_frac"] >= 0.4), ("truncated", vis["trunc_frac"] >= 0.35),
                                             ("far", (vis["distance"] or 0) >= 3.5 and vis["area"] < 0.01),
                                             ("oblique", (vis["facing_cos"] or 1) < 0.35)) if hit]
                    found.append({"id": sid, "dataset": args.dataset, "scene": scene, "role": role,
                                  "category": category, "target": TARGET_LABEL.get(category, category),
                                  "object_id": oid, "object_name": names.get(oid), "tag": tag, "tags": tags or [tag],
                                  "start": {"position": [round(float(x), 4) for x in p], "yaw": round(float(yaw), 4)},
                                  "floor_y": floor_y, "object_center": [round(float(x), 3) for x in surfaces[oid].center],
                                  "gt": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in vis.items()}})
                    mask = np.asarray(obs["semantic"]).squeeze() == oid
                    cv2.imwrite(str(args.out / "starts" / f"{sid}.jpg"),
                                cv2.cvtColor(outline(obs["rgb"], mask), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])
        finally:
            sim.close()
        print(f"{scene}: {len(objects)} objects, {len(found)} scenarios so far", flush=True)
    keep = []
    groups = defaultdict(list)
    for s in found:
        groups[(s["role"], s["tag"])].append(s)
    for key in sorted(groups):
        keep += rng.sample(groups[key], min(args.cap, len(groups[key])))
    keep.sort(key=lambda s: s["id"])
    kept = {s["id"] for s in keep}
    for s in found:
        if s["id"] not in kept:
            (args.out / "starts" / f"{s['id']}.jpg").unlink(missing_ok=True)
    with open(args.out / "scenarios.jsonl", "w") as f:
        for s in keep:
            f.write(json.dumps(s) + "\n")
    from active_approach.report import scenario_sheets
    scenario_sheets(args.out)
    counts = Counter((s["role"], s["tag"]) for s in keep)
    print(json.dumps({f"{r}/{t}": n for (r, t), n in sorted(counts.items())}, indent=1))


if __name__ == "__main__":
    main()
