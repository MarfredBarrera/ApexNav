"""Add floor_ok / n_targets_same_floor to episodes.tsv (run inside the container).

Same floor = the instance has points within [-0.6, +2.2] m of the agent's base height (storeys are > 2.5 m apart;
the band covers floor-level furniture and wall-hung TVs).  n_targets_same_floor counts HM3D instances of the episode's
target category (plus the accepted GT instance) on the agent's floor.  floor_ok = the accepted object itself is on
the agent's floor (min |dy| of its GT points <= 2.2 and some point inside the band) OR at least one target instance is."""
import collections
import csv
import sys

import numpy as np

sys.path.insert(0, "scripts")
from sample_episodes import iterator_episodes
from lastmile import bench, gt

B = bench.BENCH
rows = list(csv.DictReader(open(f"{B}/episodes.tsv"), delimiter="\t"))
eps, _ = iterator_episodes("hm3dv2")
LO, HI = -0.6, 2.2
cache = {}
for r in rows:
    X = eps[int(r["ep"])]
    sd = X.scene_id.split("/")[-2]
    if sd not in cache:
        cache.clear()
        cache[sd] = gt.SceneObjects(sd, sd.split("-", 1)[1])
    so = cache[sd]
    z = np.load(f"{B}/episodes/{r['set']}_{r['ep']}.npz")
    y0 = float(z["position"][1])
    ids = set(so.category_ids(X.object_category)) | {int(g.object_id) for g in X.goals}
    same, details = 0, []
    for i in ids:
        P = so.points(i)
        dy = P[:, 1] - y0
        ok = bool(((dy > LO) & (dy < HI)).any())
        same += ok
        if not ok:
            details.append(round(float(np.abs(dy).min()), 1))
    gp = z["gt_points"]
    dyg = gp[:, 1] - y0
    acc_ok = bool(((dyg > LO) & (dyg < HI)).any())
    r["n_targets_same_floor"] = same
    r["floor_ok"] = int(acc_ok or same > 0)
    r["n_targets_total"] = len(ids)
    r["acc_obj_same_floor"] = int(acc_ok)
    print(r["set"], r["ep"], r["tag"], "targets", len(ids), "same", same, "acc_ok", acc_ok, "other-floor min|dy|", sorted(details)[:4], flush=True)
cols = list(rows[0].keys())
for c in ("floor_ok", "n_targets_same_floor", "n_targets_total", "acc_obj_same_floor"):
    if c not in cols:
        cols.append(c)
with open(f"{B}/episodes.tsv", "w") as f:
    f.write("\t".join(cols) + "\n")
    for r in rows:
        f.write("\t".join(str(r.get(c, "")) for c in cols) + "\n")
