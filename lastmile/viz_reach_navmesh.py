"""Which navmesh does the agent actually experience? Compares, per episode, (A) bench.make_sim's recomputed navmesh (r 0.18, h 0.88),
(B) the navmesh a plain habitat Simulator holds when configured like habitat-lab (Configuration.navmesh_settings = radius 0.18,
height 1.5 = habitat-lab AgentConfig default, ApexNav's yaml sets only the radius), (C) the scene's shipped .basis.navmesh.
Output: viz_reach/<set>_<ep>_navmeshes.png and viz_reach/navmesh_compare.json.   python -m lastmile.viz_reach_navmesh [--eps ..]"""
import argparse
import json
import os

import numpy as np
from scipy.spatial import cKDTree

from lastmile import bench, viz_reach as V

OUT = V.OUT


def plain_sim(scene, h, r=0.18, recompute=False):
    """Simulator as habitat-lab builds it (no explicit recompute); Configuration.navmesh_settings as habitat_simulator.py sets them."""
    import habitat_sim
    from active_approach import scene as S
    cfg = habitat_sim.SimulatorConfiguration()
    cfg.scene_dataset_config_file, cfg.scene_id = S.SCENE_CONFIG, S.scene_path(scene)
    cfg.gpu_device_id, cfg.enable_physics, cfg.allow_sliding, cfg.create_renderer = 3, False, False, False
    ag = habitat_sim.agent.AgentConfiguration()
    ag.radius, ag.height = r, h
    nm = habitat_sim.nav.NavMeshSettings()
    nm.set_defaults()
    nm.agent_radius, nm.agent_height, nm.agent_max_climb, nm.agent_max_slope = r, h, 0.2, 45.0
    nm.include_static_objects = False
    cfg.navmesh_settings = nm
    conf = habitat_sim.Configuration(cfg, [ag])
    return habitat_sim.Simulator(conf), nm


def describe(pf, p0, y0, gt):
    ns = pf.nav_mesh_settings if hasattr(pf, "nav_mesh_settings") else None
    st = None if ns is None else dict(radius=round(ns.agent_radius, 3), height=round(ns.agent_height, 3), climb=round(ns.agent_max_climb, 3))
    pp = np.asarray(pf.snap_point(p0), np.float32)
    aisl = pf.get_island(pp)
    out = dict(settings=st, num_islands=pf.num_islands, agent_island_area_m2=round(float(pf.island_area(aisl)), 2) if aisl >= 0 else None,
               accept_navigable=bool(pf.is_navigable(p0)), accept_snap_dist_m=round(float(np.linalg.norm(pp - p0)), 3), accept_island=int(aisl))
    g, _ = V.zone_geodesic(pf, V.navmesh_tris(pf), p0, y0, gt, cap=600)
    out.update(zone_geodesic_m=g["true_geodesic_m"], zone_candidates=g["n_candidates_tested"], zone_connected=g["n_candidates_connected"])
    return out


def main():
    import habitat_sim
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrow
    from active_approach import scene as S
    ap = argparse.ArgumentParser()
    ap.add_argument("--eps", default=",".join(f"{s}:{e}" for s, e in V.FAIL + V.CONTRAST))
    a = ap.parse_args()
    rows = {(r["set"], r["ep"]): r for r in bench.load_episodes()}
    want = sorted([tuple(x.split(":")) for x in a.eps.split(",")], key=lambda k: rows[k]["scene"])
    res, cur = {}, None
    for s, ep in want:
        r = rows[(s, ep)]
        e = bench.load_episode_data(r)
        gt, p0, y0 = np.asarray(e["gt_points"], np.float32), np.asarray(e["position"], np.float32), float(e["position"][1])
        if r["scene"] != cur:
            cur = r["scene"]
            sims = {"A bench recompute r.18 h.88": bench.make_sim(cur, 3, False), "B habitat-lab-like (no recompute) r.18 h1.5": plain_sim(cur, 1.5)[0]}
            pfs = {k: v.pathfinder for k, v in sims.items()}
            shipped = habitat_sim.PathFinder()
            shipped.load_nav_mesh(S.scene_path(cur).replace(".basis.glb", ".basis.navmesh"))
            pfs["C shipped .basis.navmesh"] = shipped
        tele = json.load(open(os.path.join(bench.BENCH, "viz_fail", f"{s}_{ep}_telemetry.json")))
        ext = tele["extent"]
        info = {k: describe(pf, p0, y0, gt) for k, pf in pfs.items()}
        res[f"{s}:{ep}"] = info
        tris = V.floor_triangles(S.scene_path(cur), y0)
        fig, axs = plt.subplots(2, 2, figsize=(14, 14 * ext["h"] / ext["w"] + 1.6), dpi=75)
        axs = axs.ravel()
        # panel 0: real scene geometry (horizontal surfaces within +-1.5 m of the agent floor), coloured by height
        from PIL import Image, ImageDraw
        T3, ty = V.floor_triangles(S.scene_path(cur), y0, 1.5, with_y=True)
        pf0 = pfs["A bench recompute r.18 h.88"]
        lo0, _ = pf0.get_bounds()
        X0g, Z0g, Wg, Hg = ext["x0"], ext["z0"], ext["w"], ext["h"]
        im = Image.new("RGB", (Wg, Hg), (28, 28, 34))
        dr = ImageDraw.Draw(im)
        cmap = plt.get_cmap("turbo")
        for i in np.argsort(ty):
            t = T3[i]
            if (t[:, 0] < X0g - 1).all() or (t[:, 0] > X0g + Wg * V.MPP + 1).all() or (t[:, 1] < Z0g - 1).all() or (t[:, 1] > Z0g + Hg * V.MPP + 1).all():
                continue
            dr.polygon([((x - X0g) / V.MPP, (z - Z0g) / V.MPP) for x, z in t], fill=tuple(int(255 * c) for c in cmap((ty[i] - y0 + 1.5) / 3)[:3]))
        ax = axs[0]
        ax.imshow(np.asarray(im), extent=[X0g, X0g + Wg * V.MPP, Z0g + Hg * V.MPP, Z0g], interpolation="nearest")
        ax.scatter(gt[::3, 0], gt[::3, 2], s=3, c="w", lw=0)
        yaw = float(e["yaw"])
        ax.add_patch(FancyArrow(p0[0], p0[2], -0.3 * np.sin(yaw), -0.3 * np.cos(yaw), width=0.06, head_width=0.18, length_includes_head=True, fc="#ffe600", ec="k", lw=0.6))
        ax.set_title("REAL scene geometry: upward-facing surfaces within +-1.5 m of agent floor, colour = height (turbo -1.5..+1.5 m); white = GT", fontsize=8)
        ax.tick_params(labelsize=6)
        for ax, (k, pf) in zip(axs[1:], pfs.items()):
            lo, _ = pf.get_bounds()
            ix0, iz0 = int(round((ext["x0"] - lo[0]) / V.MPP)), int(round((ext["z0"] - lo[2]) / V.MPP))
            W, H = ext["w"], ext["h"]
            X0, Z0 = lo[0] + ix0 * V.MPP, lo[2] + iz0 * V.MPP
            nav = np.asarray(pf.get_topdown_view(V.MPP, y0))[iz0:iz0 + H, ix0:ix0 + W]
            isl = np.where(np.asarray(pf.get_topdown_view(V.MPP, y0)), np.asarray(pf.get_topdown_island_view(V.MPP, y0)), -1)[iz0:iz0 + H, ix0:ix0 + W]
            H, W = nav.shape
            inb = ((tris[:, :, 0] > X0) & (tris[:, :, 0] < X0 + W * V.MPP) & (tris[:, :, 1] > Z0) & (tris[:, :, 1] < Z0 + H * V.MPP)).any(1)
            real = V.raster(tris[inb], X0, Z0, H, W) if inb.any() else np.zeros((H, W), bool)
            aisl = pf.get_island(np.asarray(pf.snap_point(p0), np.float32))
            img = np.full((H, W, 3), (28, 28, 34), np.uint8)
            img[real & ~nav] = (95, 95, 105)
            for j, i in enumerate(np.unique(isl[isl >= 0])):
                img[isl == i] = (205, 228, 240) if i == aisl else V.ISLAND_COLS[j % len(V.ISLAND_COLS)]
            ys, xs = np.mgrid[0:H, 0:W]
            wx, wz = X0 + (xs + 0.5) * V.MPP, Z0 + (ys + 0.5) * V.MPP
            g2 = gt[np.abs(gt[:, 1] - y0) <= bench.DY_MAX][:, [0, 2]]
            dist = cKDTree(g2).query(np.stack([wx.ravel(), wz.ravel()], 1))[0].reshape(H, W)
            zone = dist <= 0.99
            img[zone & (isl == aisl) & (isl >= 0)] = (70, 175, 95)
            img[zone & (isl != aisl) & (isl >= 0)] = (240, 150, 40)
            ax.imshow(img, extent=[X0, X0 + W * V.MPP, Z0 + H * V.MPP, Z0], interpolation="nearest")
            ax.contour(wx, wz, dist, levels=[0.99], colors="w", linewidths=0.8, linestyles="--")
            ax.scatter(gt[::3, 0], gt[::3, 2], s=3, c="#b6ff00", lw=0)
            yaw = float(e["yaw"])
            ax.add_patch(FancyArrow(p0[0], p0[2], -0.3 * np.sin(yaw), -0.3 * np.cos(yaw), width=0.06, head_width=0.18, length_includes_head=True, fc="#ffe600", ec="k", lw=0.6))
            d = info[k]
            ax.set_title(f"{k}: islands {d['num_islands']}, agent island {d['agent_island_area_m2']} m2, accept nav {d['accept_navigable']}; zone path: {d['zone_geodesic_m']} m ({d['zone_connected']}/{d['zone_candidates']})", fontsize=7)
            ax.tick_params(labelsize=6)
        fig.suptitle(f"{s}:{ep} {r['target']}   light blue = agent island, orange = zone on other island, green = zone on agent island, grey = scene floor mesh without navmesh", fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, f"{s}_{ep}_navmeshes.png"), dpi=75)
        plt.close(fig)
        print(s, ep, json.dumps(info), flush=True)
    json.dump(res, open(os.path.join(OUT, "navmesh_compare.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
