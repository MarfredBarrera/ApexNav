"""Reachability figures: why a last-mile episode cannot reach the 1 m success zone from the gate-accept pose.

One PNG per episode in bench_data/viz_reach/ (+ index.json). Navmesh = bench.make_sim's (radius 0.18, height 0.88,
climb 0.2, no renderer, so no GPU memory), same floor slice as viz_fail (its telemetry extent).
  python -m lastmile.viz_reach [--eps heldout:98,...] [--gpu 3]
"""
import argparse
import csv
import json
import os

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

from lastmile import bench, gt as gtmod

OUT = os.path.join(bench.BENCH, "viz_reach")
FAIL = [("heldout", "98"), ("heldout", "104"), ("heldout", "322"), ("test2", "312"), ("test2", "478")]
CONTRAST = [("tuning", "185"), ("heldout", "237"), ("test2", "253")]
NECK_M = 0.25      # a navigable neck narrower than one forward step (0.25 m) cannot be driven with 30-degree headings
MPP = 0.025
ISLAND_COLS = [(150, 120, 90), (130, 110, 150), (160, 150, 90), (110, 140, 150), (150, 110, 120), (120, 150, 110), (140, 130, 130), (100, 120, 90)]


def node_matrix(nd):
    """glTF node transform (same maths as gt.load_objects.mat)."""
    if "matrix" in nd:
        return np.array(nd["matrix"]).reshape(4, 4).T
    M = np.eye(4)
    if "scale" in nd:
        M = np.diag(list(nd["scale"]) + [1]) @ M
    if "rotation" in nd:
        x, y, z, w = nd["rotation"]
        R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                      [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                      [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
        M2 = np.eye(4)
        M2[:3, :3] = R
        M = M2 @ M
    if "translation" in nd:
        M3 = np.eye(4)
        M3[:3, 3] = nd["translation"]
        M = M3 @ M
    return M


def floor_triangles(glb, y0, tol=0.15, with_y=False):
    """(T,3,2) xz triangles of the scene mesh (habitat world frame, y-up) with near-horizontal normals within tol of the floor height."""
    js, bin_ = gtmod._glb(glb)
    comp = {5126: np.float32, 5121: np.uint8, 5123: np.uint16, 5125: np.uint32}
    nc = {"SCALAR": 1, "VEC3": 3}

    def acc(i):
        a = js["accessors"][i]
        bv = js["bufferViews"][a["bufferView"]]
        dt, n = np.dtype(comp[a["componentType"]]), nc[a["type"]]
        stride = bv.get("byteStride", dt.itemsize * n)
        raw = np.frombuffer(bin_, np.uint8, count=a["count"] * stride, offset=bv.get("byteOffset", 0) + a.get("byteOffset", 0))
        return raw.reshape(a["count"], stride)[:, :dt.itemsize * n].copy().view(dt).reshape(a["count"], n)

    tris, ys = [], []

    def walk(ni, P):
        nd = js["nodes"][ni]
        M = P @ node_matrix(nd)
        if "mesh" in nd:
            for pr in js["meshes"][nd["mesh"]]["primitives"]:
                if "indices" not in pr:
                    continue
                p = acc(pr["attributes"]["POSITION"]).astype(float)
                p = (M[:3, :3] @ p.T).T + M[:3, 3]
                t = p[acc(pr["indices"]).reshape(-1, 3).astype(np.int64)]       # (T,3,3)
                n = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
                n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
                ok = (np.abs(n[:, 1]) > 0.707) & (np.abs(t[:, :, 1].mean(1) - y0) <= tol)
                tris.append(t[ok][:, :, [0, 2]].astype(np.float32))
                ys.append(t[ok][:, :, 1].mean(1).astype(np.float32))
        for ch in nd.get("children", []):
            walk(ch, M)

    for r in js["scenes"][js.get("scene", 0)]["nodes"]:
        walk(r, np.eye(4))
    T = np.concatenate(tris) if tris else np.zeros((0, 3, 2), np.float32)
    return (T, np.concatenate(ys) if ys else np.zeros(0, np.float32)) if with_y else T


def raster(tris, X0, Z0, H, W):
    from PIL import Image, ImageDraw
    im = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(im)
    for t in tris:
        d.polygon([((x - X0) / MPP, (z - Z0) / MPP) for x, z in t], fill=1)
    return np.asarray(im, bool)


def geodesic(pf, p0, cands):
    import habitat_sim
    best, bp = None, None
    for c in cands:
        sp = habitat_sim.ShortestPath()
        sp.requested_start, sp.requested_end = np.asarray(p0, np.float32), np.asarray(c, np.float32)
        if pf.find_path(sp) and (best is None or sp.geodesic_distance < best):
            best, bp = sp.geodesic_distance, np.array(sp.points)
    return best, bp


def wedge(sim, p0, yaw, action="move_forward", deg=30):
    """Uses the real simulator step (what the agent experiences): from the accept position, at every heading yaw + k*deg, act(action).
    Returns (fully blocked: moved < 2 cm, partially moved)."""
    import habitat_sim
    from habitat_sim.utils.common import quat_from_angle_axis
    ag = sim.get_agent(0)
    st0 = ag.get_state()
    blocked = partial = 0
    amt = {"move_forward": bench.FORWARD_M, "move_forward_small": bench.SMALL_M}[action]
    for k in range(360 // deg):
        st = habitat_sim.AgentState()
        st.position, st.rotation = np.asarray(p0, np.float32), quat_from_angle_axis(yaw + np.radians(deg * k), np.array([0.0, 1.0, 0.0]))
        ag.set_state(st, reset_sensors=False)
        ag.act(action)
        q = np.array(ag.get_state().position)
        mv = np.hypot(q[0] - p0[0], q[2] - p0[2])
        if mv < 0.02:
            blocked += 1
        elif mv < amt - 0.01:
            partial += 1
    ag.set_state(st0, reset_sensors=False)
    return blocked, partial


def navmesh_tris(pf):
    V = np.asarray(pf.build_navmesh_vertices(), np.float32)
    return V[np.asarray(pf.build_navmesh_vertex_indices(), np.int64).reshape(-1, 3)]        # (T,3,3)


def zone_samples(T, gt, step=0.06):
    """dense points (all heights) on navmesh triangles that come within 1 m of the GT footprint: barycentric grid, step metres."""
    lo, hi = gt[:, [0, 2]].min(0) - 1.05, gt[:, [0, 2]].max(0) + 1.05
    bb0, bb1 = T[:, :, [0, 2]].min(1), T[:, :, [0, 2]].max(1)
    T = T[((bb1 >= lo) & (bb0 <= hi)).all(1)]
    pts = [T.reshape(-1, 3), T.mean(1)]
    for t in T:
        a, b, c = t
        n = int(max(abs(np.cross((b - a)[[0, 2]], (c - a)[[0, 2]])) / 2 / (step * step), 0))
        if n > 4:
            m = min(int(np.sqrt(n)) + 1, 120)
            u, v = np.meshgrid(np.linspace(0, 1, m), np.linspace(0, 1, m))
            k = (u + v) <= 1
            u, v = u[k], v[k]
            pts.append(a + u[:, None] * (b - a) + v[:, None] * (c - a))
    return np.concatenate(pts).astype(np.float32)


def zone_geodesic(pf, T, p0, y0, gt, cap=1500):
    """all-height navmesh candidates within 1 m (+score_point's |dy| rule) -> find_path from p0. Returns stats dict and the best path."""
    import habitat_sim
    from scipy.spatial import cKDTree as KD
    sam = zone_samples(T, gt)
    d = KD(gt[:, [0, 2]]).query(sam[:, [0, 2]])[0]
    near = sam[d <= bench.SUCCESS_M - 0.01]
    ok = np.array([bench.score_point(q, gt)[1] <= bench.DY_MAX for q in near], bool) if len(near) else np.zeros(0, bool)
    cands = near[ok]
    out = dict(n_candidates_all_heights=int(len(cands)))
    dyc = cands[:, 1] - y0
    out["zone_navmesh_height_range_m"] = [round(float(dyc.min()), 2), round(float(dyc.max()), 2)] if len(cands) else None
    out["zone_navmesh_pts_off_floor_gt_0.25m"] = int((np.abs(dyc) > 0.25).sum()) if len(cands) else 0
    if len(cands) > cap:
        cands = cands[np.random.RandomState(0).choice(len(cands), cap, replace=False)]
    best, bp, n_conn = None, None, 0
    p0s = np.asarray(pf.snap_point(p0), np.float32)
    for c in cands:
        c = np.array(pf.snap_point(c), np.float32)
        if not np.isfinite(c).all():
            continue
        sp = habitat_sim.ShortestPath()
        sp.requested_start, sp.requested_end = p0s, c
        if pf.find_path(sp):
            n_conn += 1
            if best is None or sp.geodesic_distance < best:
                best, bp = sp.geodesic_distance, np.array(sp.points)
    out["n_candidates_tested"] = int(len(cands))
    out["n_candidates_connected"] = n_conn
    out["true_geodesic_m"] = None if best is None else round(float(best), 3)
    out["path_max_height_dev_m"] = None if bp is None else round(float(np.abs(bp[:, 1] - y0).max()), 2)
    return out, bp


def deep_reach(sim, p0, y0, gt):
    """Independent reachability (every navmesh point of every height within 1 m of the GT object) + the oracle's candidate filter
    replayed (single-height slice, snap, |dy|<0.5) + the same test on a navmesh rebuilt with a smaller agent radius."""
    import habitat_sim
    from scipy.spatial import cKDTree as KD
    pf = sim.pathfinder
    out, bp = zone_geodesic(pf, navmesh_tris(pf), p0, y0, gt)
    out.update(gt_y_min_m=round(float(gt[:, 1].min() - y0), 2), gt_y_med_m=round(float(np.median(gt[:, 1]) - y0), 2), gt_y_max_m=round(float(gt[:, 1].max() - y0), 2))
    # oracle-like filter
    lo, _ = pf.get_bounds()
    res = 0.1
    view = pf.get_topdown_view(res, y0)
    iz, ix = np.nonzero(view)
    cells = np.stack([lo[0] + (ix + 0.5) * res, np.full(len(ix), y0), lo[2] + (iz + 0.5) * res], 1).astype(np.float32)
    dd = KD(gt[:, [0, 2]]).query(cells[:, [0, 2]])[0]
    cc = cells[dd <= 0.99]
    good = []
    for c in cc:
        sn = np.array(pf.snap_point(c))
        if np.isfinite(sn).all() and abs(sn[1] - y0) < 0.5 and np.hypot(sn[0] - c[0], sn[2] - c[2]) < 0.1:
            a, b = bench.score_point(sn, gt)
            if a <= 0.99 and b <= bench.DY_MAX:
                good.append(sn)
    out["oracle_filter_cands"] = len(good)
    ob = None
    for g in good[:2000]:
        sp = habitat_sim.ShortestPath()
        sp.requested_start, sp.requested_end = p0, g.astype(np.float32)
        if pf.find_path(sp) and (ob is None or sp.geodesic_distance < ob):
            ob = sp.geodesic_distance
    out["oracle_filter_geodesic_m"] = None if ob is None else round(float(ob), 3)
    nm = habitat_sim.nav.NavMeshSettings()
    nm.set_defaults()
    nm.agent_height, nm.agent_max_climb, nm.agent_max_slope = bench.CAMERA_HEIGHT, 0.2, 45.0
    nm.include_static_objects = False
    sweep = {}
    for rad in (0.10, 0.12, 0.14, 0.16):
        nm.agent_radius = rad
        sim.recompute_navmesh(pf, nm)
        rg, _ = zone_geodesic(pf, navmesh_tris(pf), p0, y0, gt, cap=400)
        sweep[f"{rad:.2f}"] = rg["true_geodesic_m"]
    nm.agent_radius = bench.RADIUS
    sim.recompute_navmesh(pf, nm)
    out["zone_geodesic_by_agent_radius_m"] = sweep
    out["true_geodesic_radius010_m"] = sweep["0.10"]
    out["min_agent_radius_connecting_m"] = next((float(r) for r in sorted(sweep) if sweep[r] is not None), None)
    return out, bp


def levels_png(T, y0, p0, yaw, gt, gc, X0, Z0, W, H, dist, wx, wz, path, title, out_png):
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrow
    from PIL import Image, ImageDraw
    cm = plt.get_cmap("turbo")
    band = T[np.abs(T[:, :, 1].mean(1) - y0) <= 1.5]
    band = band[np.argsort(band[:, :, 1].mean(1))]
    im = Image.new("RGB", (W, H), (28, 28, 34))
    dr = ImageDraw.Draw(im)
    for t in band:
        c = tuple(int(255 * x) for x in cm((t[:, 1].mean() - y0 + 1.5) / 3)[:3])
        dr.polygon([((x - X0) / MPP, (z - Z0) / MPP) for x, z in t[:, [0, 2]]], fill=c)
    hh = 9 * H / W
    fig = plt.figure(figsize=(9, hh + 0.7), dpi=100)
    ax = fig.add_axes([0.02, 0.02, 0.86, hh / (hh + 0.7) * 0.96])
    cax = fig.add_axes([0.9, 0.05, 0.015, hh / (hh + 0.7) * 0.8])
    ax.imshow(np.asarray(im), extent=[X0, X0 + W * MPP, Z0 + H * MPP, Z0], interpolation="nearest")
    sc = plt.cm.ScalarMappable(norm=plt.Normalize(-1.5, 1.5), cmap="turbo")
    fig.colorbar(sc, cax=cax).set_label("navmesh height - agent floor (m)", fontsize=7)
    cax.tick_params(labelsize=7)
    ax.contour(wx, wz, dist, levels=[bench.SUCCESS_M - 0.01], colors=["w"], linewidths=1.0, linestyles="--")
    ax.scatter(gt[::2, 0], gt[::2, 2], s=5, c="#b6ff00", edgecolors="k", linewidths=0.2)
    if path is not None:
        ax.plot(path[:, 0], path[:, 2], "-", c="k", lw=3)
        ax.plot(path[:, 0], path[:, 2], "-", c="w", lw=1.5)
    ax.add_patch(FancyArrow(p0[0], p0[2], -0.35 * np.sin(yaw), -0.35 * np.cos(yaw), width=0.07, head_width=0.2, length_includes_head=True, fc="#ffe600", ec="k", lw=0.8, zorder=5))
    ax.tick_params(labelsize=7)
    fig.text(0.02, 0.995, title, fontsize=9, weight="bold", va="top")
    fig.savefig(out_png, dpi=100)
    plt.close(fig)


def analyse(sim, row, tele, out_png, orc):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrow
    from active_approach import scene as S
    pf = sim.pathfinder
    e = bench.load_episode_data(row)
    gt, gc = np.asarray(e["gt_points"], np.float32), np.asarray(e["goal_cloud"], np.float32)
    p0, y0 = np.asarray(e["position"], np.float32), float(e["position"][1])
    ext = tele["extent"]
    lo, _ = pf.get_bounds()
    ix0, iz0 = int(round((ext["x0"] - lo[0]) / MPP)), int(round((ext["z0"] - lo[2]) / MPP))
    W, H = ext["w"], ext["h"]
    X0, Z0 = lo[0] + ix0 * MPP, lo[2] + iz0 * MPP
    nav = np.asarray(pf.get_topdown_view(MPP, y0))
    isl = np.asarray(pf.get_topdown_island_view(MPP, y0))
    isl = np.where(nav, isl, -1)[iz0:iz0 + H, ix0:ix0 + W]
    nav = nav[iz0:iz0 + H, ix0:ix0 + W]
    H, W = nav.shape
    ys, xs = np.mgrid[0:H, 0:W]
    wx, wz = X0 + (xs + 0.5) * MPP, Z0 + (ys + 0.5) * MPP
    # success zone: cells within 1 m of a GT point on the same storey (same rule as bench.score_point)
    g2 = gt[np.abs(gt[:, 1] - y0) <= bench.DY_MAX][:, [0, 2]]
    dist = cKDTree(g2).query(np.stack([wx.ravel(), wz.ravel()], 1))[0].reshape(H, W) if len(g2) else np.full((H, W), 9.0)
    zone = dist <= bench.SUCCESS_M - 0.01
    aisl = pf.get_island(p0)
    on_agent = zone & (isl == aisl)
    other = zone & (isl >= 0) & (isl != aisl)
    cell_area = MPP * MPP / 0.05 ** 2        # counts reported in 0.05 m cells
    # real scene floor from the mesh
    tris = floor_triangles(S.scene_path(row["scene"]), y0)
    inb = ((tris[:, :, 0] > X0 - 0.2) & (tris[:, :, 0] < X0 + W * MPP + 0.2) & (tris[:, :, 1] > Z0 - 0.2) & (tris[:, :, 1] < Z0 + H * MPP + 0.2)).any(1)
    real = raster(tris[inb], X0, Z0, H, W) if inb.any() else np.zeros((H, W), bool)
    hole = zone & ~nav
    real_zone = bool((real & hole).sum() * MPP * MPP > 0.05) if hole.any() else None
    # geodesic to the nearest in-zone navigable point on the agent's island
    geo, path, clear, clear_mid = None, None, None, None
    cand_idx = np.argwhere(on_agent)
    if len(cand_idx):
        sub = cand_idx[:: max(1, len(cand_idx) // 300)]
        cands = [np.array(pf.snap_point(np.array([wx[i, j], y0, wz[i, j]], np.float32)), np.float32) for i, j in sub]
        geo, path = geodesic(pf, p0, cands)
        if path is not None:
            edt = ndimage.distance_transform_edt(nav) * MPP
            seg = np.concatenate([np.linspace(path[k], path[k + 1], max(2, int(np.hypot(*(path[k + 1] - path[k])[[0, 2]]) / 0.025))) for k in range(len(path) - 1)]) if len(path) > 1 else path
            jj = np.clip(((seg[:, 0] - X0) / MPP).astype(int), 0, W - 1)
            ii = np.clip(((seg[:, 2] - Z0) / MPP).astype(int), 0, H - 1)
            clear = float(edt[ii, jj].min())
            far = np.hypot(seg[:, 0] - p0[0], seg[:, 2] - p0[2]) > 0.3
            clear_mid = round(float(edt[ii, jj][far].min()), 3) if far.any() else None            # distance from path to the nearest non-navigable cell; neck width ~ 2x
    nblk, npart = wedge(sim, p0, float(e['yaw']))
    b15, _p15 = wedge(sim, p0, float(e['yaw']), 'move_forward_small', 15)
    free15 = 24 - b15
    nblk10, npart10 = wedge(sim, p0, float(e['yaw']), 'move_forward_small')
    on_nav = bool(pf.is_navigable(p0))
    # edge test: distance of the accept pose to the nearest non-navigable point, and the movable headings after nudging it inward to >= 5 cm clearance
    res = 0.01
    lo_, _ = pf.get_bounds()
    v01 = np.asarray(pf.get_topdown_view(res, y0))
    jx, jz = int((p0[0] - lo_[0]) / res), int((p0[2] - lo_[2]) / res)
    sub = v01[max(jz - 40, 0):jz + 41, max(jx - 40, 0):jx + 41]
    e01 = ndimage.distance_transform_edt(np.pad(sub, 1, constant_values=True)[1:-1, 1:-1]) * res
    oz, ox = min(jz, 40), min(jx, 40)
    edge_clear = float(e01[oz, ox])
    zz, xx = np.nonzero(e01 >= 0.05)
    nudge_m, nudged_blocked = None, None
    if len(zz):
        dd = np.hypot(zz - oz, xx - ox)
        j = int(dd.argmin())
        q = np.array([lo_[0] + (max(jx - 40, 0) + xx[j] + 0.5) * res, p0[1], lo_[2] + (max(jz - 40, 0) + zz[j] + 0.5) * res], np.float32)
        nudge_m, nudged_blocked = round(float(dd[j] * res), 3), wedge(sim, q, float(e["yaw"]))[0]
    T_ = navmesh_tris(pf)
    deep, dpath = deep_reach(sim, p0, y0, gt)
    levels_png(T_, y0, p0, float(e["yaw"]), gt, gc, X0, Z0, W, H, dist, wx, wz, dpath if dpath is not None else path,
               f"{row['set']}:{row['ep']} {row['target']} - navmesh height levels (+-1.5 m of agent floor)", out_png.replace("_reach.png", "_levels.png"))
    n_on, n_oth = int(round(on_agent.sum() * cell_area)), int(round(other.sum() * cell_area))
    if n_on + n_oth == 0:
        v = "no navigable floor within 1 m" + (" (real floor mesh present there: navmesh defect?)" if real_zone else " (no floor mesh there either)" if real_zone is False else "")
    elif n_on == 0:
        v = f"zone navigable only on another island (agent island {pf.island_area(aisl):.1f} m2; other island(s) {sorted(set(int(i) for i in np.unique(isl[other])))} vs agent {aisl})"
    elif nblk >= 10 or not on_nav:
        v = (f"agent wedged at accept (a 0.25 m step can move in {12 - nblk} of 12 headings [{npart} only partially]; 0.10 m step: {12 - nblk10} of 12; "
             f"0.10 m on a 15 deg grid: {free15} of 24); pose is {edge_clear * 100:.0f} cm from the navmesh edge" +
             (f", nudging it {nudge_m * 100:.0f} cm inward leaves {12 - nudged_blocked} of 12 headings movable" if nudge_m is not None else "") + ("" if on_nav else "; accept pose off navmesh"))
    elif clear is not None and 2 * clear < NECK_M:
        v = f"connected but neck narrower than {NECK_M:.2f} m (min width {2 * clear:.2f} m)"
    else:
        v = f"connected, geodesic {geo:.2f} m, min clearance {clear:.2f} m ({nblk} of 12 headings fully blocked, {npart} partial)"
    tg = deep["true_geodesic_m"]
    if n_on + n_oth == 0 and tg is not None:
        v = (f"no navigable floor within 1 m on the agent's floor slice; zone IS navigable on stairs/other height (true path {tg:.2f} m, "
             f"height change up to {deep['path_max_height_dev_m']:.2f} m)")
    elif n_on == 0 and tg is None and deep["min_agent_radius_connecting_m"] is not None:
        v += f"; NAVMESH EROSION: unreachable at the agent's radius 0.18 m, connects at radius {deep['min_agent_radius_connecting_m']:.2f} m (opening ~{2 * deep['min_agent_radius_connecting_m']:.2f}-0.36 m wide)"
    stairs = deep["zone_navmesh_pts_off_floor_gt_0.25m"] > 0
    v2 = (f"GT y vs floor {deep['gt_y_min_m']:+.2f}..{deep['gt_y_max_m']:+.2f} m; zone navmesh heights {deep['zone_navmesh_height_range_m']}" +
          ("; navmesh off-floor (stair/landing) within 1 m" if stairs else "; no off-floor navmesh in zone"))
    v3 = ("true navmesh path (all heights): " + (f"{tg:.2f} m ({deep['n_candidates_connected']}/{deep['n_candidates_tested']} cands)" if tg is not None else f"NONE ({deep['n_candidates_tested']} cands; radius 0.10: {deep['true_geodesic_radius010_m']})") +
          f"; oracle slice-filter: {deep['oracle_filter_cands']} cands, geodesic {deep['oracle_filter_geodesic_m']}; oracle_status: {orc}")
    # ---- figure
    img = np.zeros((H, W, 3), np.uint8)
    img[:] = (28, 28, 34)
    img[real & ~nav] = (78, 78, 86)
    for k, i in enumerate(np.unique(isl[isl >= 0])):
        img[isl == i] = (205, 228, 240) if i == aisl else ISLAND_COLS[k % len(ISLAND_COLS)]
    img[on_agent] = (70, 175, 95)
    img[other] = (240, 150, 40)
    hh = 9 * H / W
    fig = plt.figure(figsize=(9, hh + 1.5), dpi=100)
    ax = fig.add_axes([0.02, 0.02, 0.96, hh / (hh + 1.5) * 0.97])
    ax.imshow(img, extent=[X0, X0 + W * MPP, Z0 + H * MPP, Z0], interpolation="nearest")
    ax.contour(wx, wz, dist, levels=[bench.SUCCESS_M - 0.01], colors=["w"], linewidths=1.0, linestyles="--")
    ax.scatter(gc[::2, 0], gc[::2, 2], s=3, c="#ff2a2a", lw=0)
    ax.scatter(gt[::2, 0], gt[::2, 2], s=5, c="#b6ff00", edgecolors="k", linewidths=0.2)
    if path is not None:
        ax.plot(path[:, 0], path[:, 2], "-", c="#00e5ff", lw=2.2)
    yaw = float(e["yaw"])
    ax.add_patch(FancyArrow(p0[0], p0[2], -0.35 * np.sin(yaw), -0.35 * np.cos(yaw), width=0.07, head_width=0.2, length_includes_head=True, fc="#ffe600", ec="k", lw=0.8, zorder=5))
    for lab, col, mk in [("agent island", (205, 228, 240), "s"), ("other islands", ISLAND_COLS[0], "s"), ("zone navigable, agent island", (70, 175, 95), "s"),
                         ("zone navigable, other island", (240, 150, 40), "s"), ("scene floor mesh, no navmesh", (78, 78, 86), "s")]:
        ax.plot([], [], mk, c=tuple(c / 255 for c in col), label=lab)
    ax.plot([], [], "--", c="w", label="1 m zone outline")
    ax.plot([], [], "o", c="#b6ff00", mec="k", ms=4, label="GT object")
    ax.plot([], [], "o", c="#ff2a2a", ms=3, label="perceived cloud")
    ax.plot([], [], "-", c="#00e5ff", label="geodesic to zone" + (f" {geo:.2f} m" if geo else " (none)"))
    ax.plot([], [], ">", c="#ffe600", mec="k", label="accept pose")
    ax.legend(loc="best", fontsize=6.5, framealpha=0.85, ncol=2)
    ax.tick_params(labelsize=7)
    fig.text(0.02, 0.995, f"{row['set']}:{row['ep']}   target: {row['target']}   scene {row['scene']}", fontsize=10, weight="bold", va="top")
    for k, (t, c) in enumerate([(v, "#8a1c1c"), (v2, "#1c3c8a"), (v3, "#1c6a2a")]):
        fig.text(0.02, 0.995 - (0.4 + 0.3 * k) / (hh + 1.5), t, fontsize=7.3, va="top", color=c)
    fig.savefig(out_png, dpi=100)
    plt.close(fig)
    return dict(set=row["set"], ep=row["ep"], target=row["target"], verdict=v, stairs_note=v2, reachability_note=v3, **deep, agent_island_area_m2=round(float(pf.island_area(aisl)), 2),
                zone_navigable_cells_on_agent_island=n_on, zone_navigable_cells_other_islands=n_oth,
                geodesic_m=None if geo is None else round(float(geo), 3), min_clearance_m=None if clear is None else round(clear, 3),
                wedged_headings=nblk, partial_headings=npart, blocked_headings_010m=nblk10, free_headings_010m_15deg_of_24=free15, min_clearance_after_start_m=clear_mid, accept_on_navmesh=on_nav, accept_edge_clearance_m=round(edge_clear, 3), nudge_inward_m=nudge_m, nudged_blocked_headings_025=nudged_blocked, real_floor_in_zone=real_zone, cell_size_m=0.05)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eps", default=",".join(f"{s}:{e}" for s, e in FAIL + CONTRAST))
    ap.add_argument("--gpu", type=int, default=3)
    a = ap.parse_args()
    bench.ACTIONS = "fine"            # adds the 0.10 m forward action to the sim's action space (for the wedge test only)
    want = [tuple(x.split(":")) for x in a.eps.split(",")]
    rows = {(r["set"], r["ep"]): r for r in bench.load_episodes()}
    os.makedirs(OUT, exist_ok=True)
    ora = {(r['set'], r['ep']): r for r in csv.DictReader(open(os.path.join(bench.BENCH, 'episodes.tsv')), delimiter='\t')}
    res, cur, sim = [], None, None
    for s, ep in sorted(want, key=lambda k: rows[k]["scene"]):
        r = rows[(s, ep)]
        if r["scene"] != cur:
            if sim is not None:
                sim.close()
            sim, cur = bench.make_sim(r["scene"], a.gpu, False), r["scene"]
        tele = json.load(open(os.path.join(bench.BENCH, "viz_fail", f"{s}_{ep}_telemetry.json")))
        out = analyse(sim, r, tele, os.path.join(OUT, f"{s}_{ep}_reach.png"), ora[(s, ep)]['oracle_status'] or 'none')
        out["group"] = "unreachable" if (s, ep) in FAIL else "contrast"
        print(out, flush=True)
        res.append(out)
    idx = os.path.join(OUT, "index.json")
    old = {(o["set"], o["ep"]): o for o in json.load(open(idx))} if os.path.exists(idx) else {}
    old.update({(o["set"], o["ep"]): o for o in res})
    json.dump(list(old.values()), open(idx, "w"), indent=1)


if __name__ == "__main__":
    main()
