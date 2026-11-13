"""Per-episode failure / success visualisations + telemetry for the hybrid (astar_ddppo) last-mile navigator.

  python -m lastmile.viz_fail [--eps set:ep,...] [--exclude-sets fresh400] [--out DIR] [--gpu 3]

Replays every episode of results/hybrid_v1.tsv exactly like bench.run_episode (start=accept, budget=remaining, std actions)
and writes to bench_data/viz_fail/:
  <set>_<ep>_top.png        annotated top-down panel (axes box + world extent in index.json "extent" lets a page overlay on it)
  <set>_<ep>_fpv.jpg        start / 3 mid / final RGB strip
  <set>_<ep>_frames.jpg     sprite sheet (<=60 tiles, 10 per row) for scrubbing
  <set>_<ep>_telemetry.json per-step records (record i = state after action i; record 0 = start; nav fields = decision for action i)
  index.json
"""
import argparse
import io
import json
import math
import os
import time
from collections import Counter

import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial import cKDTree

from lastmile import bench

TILE_W, TILE_H, COLS, MAXF = 160, 120, 10, 60
ANAME = {0: "STOP", 1: "FWD", 2: "LEFT", 3: "RIGHT"}


def geod_multi(pf, p, ends):
    import habitat_sim
    mp = habitat_sim.MultiGoalShortestPath()
    mp.requested_start = np.asarray(p, np.float32)
    mp.requested_ends = ends
    if pf.find_path(mp):
        return float(mp.geodesic_distance), [np.asarray(q) for q in mp.points]
    return None, None


def region_ends(pf, gt, y0, res=0.1, maxn=150):
    """navigable cells within 1 m of GT (same floor, |dy|<=DY_MAX): the std success region."""
    lo, _ = pf.get_bounds()
    view = pf.get_topdown_view(res, float(y0))
    iz, ix = np.nonzero(view)
    cells = np.stack([lo[0] + (ix + 0.5) * res, np.full(len(ix), y0), lo[2] + (iz + 0.5) * res], 1).astype(np.float32)
    d, _ = cKDTree(gt[:, [0, 2]]).query(cells[:, [0, 2]])
    c = cells[d <= bench.SUCCESS_M - 0.01]
    if len(c) > 1500:
        c = c[np.random.RandomState(0).choice(len(c), 1500, replace=False)]
    c = np.array([q for q in c if bench.score_point(q, gt)[1] <= bench.DY_MAX], np.float32).reshape(-1, 3)
    if len(c) > maxn:
        c = c[np.random.RandomState(1).choice(len(c), maxn, replace=False)]
    return [q for q in c]


def replay(sim, nav, e, budget, ends):
    gt = np.asarray(e["gt_points"], np.float32)
    gc = np.asarray(e["goal_cloud"], np.float32)
    pf = sim.pathfinder
    bench.place(sim, e["position"], e["yaw"], e["pitch"])
    info = dict(scene=e["scene"], ep=e["ep"], camera=bench.CAMERA, goal_cloud=gc, goal_estimate=gc.mean(0), budget_steps=budget,
                start_position=np.asarray(e["position"]), start_yaw=e["yaw"], start_pitch=e["pitch"])
    obs = bench.observe(sim, True, gc, e["pitch"])
    nav.reset(obs, info)
    table = bench.action_table()
    cxz = gc[:, [0, 2]]

    def rec(step, obs, act, coll, nfo):
        p = obs["pose"]["position"]
        g, _ = geod_multi(pf, p, ends) if ends else (None, None)
        r = dict(step=step, action=act, collided=coll, x=round(float(p[0]), 3), z=round(float(p[2]), 3), yaw=round(float(obs["pose"]["yaw"]), 3),
                 d_gt=round(bench.score_point(p, gt)[0], 3), geo_gt=None if g is None else round(g, 3),
                 d_cloud=round(float(np.hypot(cxz[:, 0] - p[0], cxz[:, 1] - p[2]).min()), 3))
        r.update(nfo)
        return r

    recs = [rec(0, obs, None, False, {})]
    rgbs = [obs["rgb"]]
    coll, stopped, steps = 0, False, 0
    n_dd0 = n_fb0 = n_rs0 = 0
    prev_plan, nev = None, 0
    pos0 = obs["pose"]["position"]
    while steps < budget:
        a = int(nav.act(obs))
        # navigator-side telemetry for this decision
        pl = nav.plan if nav.plan is not prev_plan else None
        evs = [ev for s, ev in nav.trace["events"][nev:]]
        nev = len(nav.trace["events"])
        n_dd, n_fb, n_rs = nav.n_dd, nav.n_fallback, nav.n_reselect
        nfo = dict(replan=n_rs > n_rs0, wp_x=None, wp_z=None, d_wp=None, rho=None, phi=None, path_len=None, path_n=None, mode="scripted", events=evs)
        if n_fb > n_fb0:
            nfo["mode"] = "fallback"          # no A* path: DD-PPO toward the nearest cloud point (or spin)
        elif n_dd > n_dd0 and pl is not None:
            nfo["mode"] = "ddppo"
        elif pl is not None:
            nfo["mode"] = "astar"
        if a == bench.STOP:
            nfo["mode"] = "stop"
        if n_dd > n_dd0 and nav.dbg:
            nfo["rho"], nfo["phi"] = nav.dbg[-1][0], nav.dbg[-1][1]
        if pl is not None:
            prev_plan = nav.plan
            pth = getattr(pl, "path", None)
            if pth is not None and len(pth):
                pth = np.asarray(pth, float)
                nfo["path_n"] = int(len(pth))
                nfo["path_len"] = round(float(np.linalg.norm(np.diff(pth, axis=0), axis=1).sum()), 2) if len(pth) > 1 else 0.0
        if nfo["mode"] == "ddppo" and nav.wp is not None:
            nfo["wp_x"], nfo["wp_z"] = round(float(nav.wp[0]), 3), round(float(nav.wp[1]), 3)
            nfo["d_wp"] = round(float(np.hypot(nav.wp[0] - obs["pose"]["position"][0], nav.wp[1] - obs["pose"]["position"][2])), 3)
        n_dd0, n_fb0, n_rs0 = n_dd, n_fb, n_rs
        if a == bench.STOP:
            stopped = True
            recs[-1]["stop_next"] = True
            recs[-1].update({k: v for k, v in nfo.items() if k in ("mode", "events")})
            break
        name, fwd = table[a]
        before = bench.pose(sim)[0]
        sim.get_agent(0).act(name)
        steps += 1
        after = bench.pose(sim)[0]
        c = False
        if fwd and np.linalg.norm(after - before) < 0.5 * fwd:
            coll += 1
            c = True
        obs = bench.observe(sim, True, gc, e["pitch"])
        recs.append(rec(steps, obs, ANAME[a], c, nfo))
        rgbs.append(obs["rgb"])
    p = bench.pose(sim)[0]
    d, dy = bench.score_point(p, gt)
    return dict(recs=recs, rgbs=rgbs, stopped=stopped, steps=steps, coll=coll, final_dist=round(d, 3), final_dy=round(dy, 2),
                success=int(stopped and d <= bench.SUCCESS_M and dy <= bench.DY_MAX), pos0=pos0)


def topdown(sim, e, R, gt, gc, ends, oracle_pts, title, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pf = sim.pathfinder
    y0 = float(e["position"][1])
    mpp = 0.025
    lo, _ = pf.get_bounds()
    view = pf.get_topdown_view(mpp, y0)
    tr = np.array([[r["x"], r["z"]] for r in R["recs"]])
    pts = np.concatenate([tr, gt[:, [0, 2]], gc[:, [0, 2]]] + ([np.array(oracle_pts)[:, [0, 2]]] if oracle_pts else []))
    x0, z0 = pts.min(0) - 1.5
    x1, z1 = pts.max(0) + 1.5
    ix0, ix1 = int(max(0, (x0 - lo[0]) / mpp)), int(min(view.shape[1], (x1 - lo[0]) / mpp))
    iz0, iz1 = int(max(0, (z0 - lo[2]) / mpp)), int(min(view.shape[0], (z1 - lo[2]) / mpp))
    crop = view[iz0:iz1, ix0:ix1]
    X0, Z0 = lo[0] + ix0 * mpp, lo[2] + iz0 * mpp
    H, W = crop.shape
    img = np.where(crop[..., None], np.array([225, 225, 225], np.uint8), np.array([45, 45, 50], np.uint8)).astype(np.uint8)
    # 1 m success ring tint
    ys, xs = np.mgrid[0:H, 0:W]
    wx, wz = X0 + (xs + 0.5) * mpp, Z0 + (ys + 0.5) * mpp
    ring = np.zeros((H, W), bool)
    sub = gt[::3]
    near = sub[(sub[:, 0] > X0 - 1) & (sub[:, 0] < X0 + W * mpp + 1) & (sub[:, 2] > Z0 - 1) & (sub[:, 2] < Z0 + H * mpp + 1)]
    if len(near):
        d, _ = cKDTree(near[:, [0, 2]]).query(np.stack([wx.ravel(), wz.ravel()], 1))
        ring = (d.reshape(H, W) <= 1.0) & crop
    img[ring] = (0.55 * img[ring] + 0.45 * np.array([120, 220, 120])).astype(np.uint8)
    Wp = 900
    mh = 864 * H / W
    Hp = int(round(mh + 90))
    fig = plt.figure(figsize=(9, Hp / 100), dpi=100)
    ax_box = [0.02, 20 / Hp, 0.96, mh / Hp]
    ax = fig.add_axes(ax_box)
    ax.imshow(img, extent=[X0, X0 + W * mpp, Z0 + H * mpp, Z0], interpolation="nearest")
    ax.set_xlim(X0, X0 + W * mpp)
    ax.set_ylim(Z0 + H * mpp, Z0)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.scatter(gt[::4, 0], gt[::4, 2], s=3, c="#1b9e1b", label="GT object", zorder=3)
    ax.scatter(gc[:, 0], gc[:, 2], s=6, c="red", label="perceived goal cloud", zorder=3)
    if oracle_pts:
        o = np.array(oracle_pts)
        ax.plot(o[:, 0], o[:, 2], "--", c="cyan", lw=2, label="oracle geodesic", zorder=4)
    n = len(tr)
    ax.scatter(tr[:, 0], tr[:, 1], c=np.arange(n), cmap="plasma", s=7, zorder=5)
    ax.plot(tr[:, 0], tr[:, 1], "-", c="k", lw=0.5, alpha=0.5, zorder=4)
    cx = [(r["x"], r["z"]) for r in R["recs"] if r["collided"]]
    if cx:
        cx = np.array(cx)
        ax.scatter(cx[:, 0], cx[:, 1], marker="x", c="orange", s=35, lw=1.5, label="collision", zorder=6)
    wps = {(r["wp_x"], r["wp_z"]) for r in R["recs"] if r.get("wp_x") is not None}
    if wps:
        w = np.array(list(wps))
        ax.scatter(w[:, 0], w[:, 1], s=9, c="purple", label="A* waypoints", zorder=6)
    for r, col, lab in ((R["recs"][0], "blue", "start"), (R["recs"][-1], "red", "final")):
        fx, fz = -math.sin(r["yaw"]), -math.cos(r["yaw"])
        ax.annotate("", xy=(r["x"] + 0.4 * fx, r["z"] + 0.4 * fz), xytext=(r["x"], r["z"]),
                    arrowprops=dict(arrowstyle="-|>", color=col, lw=2.5), zorder=8)
        ax.plot(r["x"], r["z"], "o", c=col, ms=6, label=lab, zorder=8)
    ax.legend(loc="lower right", fontsize=7, framealpha=0.8)
    fig.text(0.5, 0.995, title, ha="center", va="top", fontsize=9, wrap=True)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100)
    plt.close(fig)
    im = Image.open(buf).convert("RGB").quantize(96, dither=Image.NONE)
    im.save(out_png, optimize=True)
    ext = dict(x0=round(float(X0), 4), z0=round(float(Z0), 4), mpp=mpp, w=W, h=H,
               axes_px=[round(ax_box[0] * Wp), round((1 - ax_box[1] - ax_box[3]) * Hp), round(ax_box[2] * Wp), round(ax_box[3] * Hp)],
               image_px=[Wp, Hp], note="axes box [left, top, w, h] in the PNG; imshow keeps aspect so the map is centred inside it")
    # actual placement of the map inside the axes box (aspect equal)
    bb = ax.get_window_extent()
    pos = ax.get_position()
    # data -> pixel via the axes transform
    p0 = ax.transData.transform((X0, Z0))
    p1 = ax.transData.transform((X0 + W * mpp, Z0 + H * mpp))
    ext["map_px"] = [round(float(p0[0]), 1), round(float(Hp - p0[1]), 1), round(float(p1[0] - p0[0]), 1), round(float(p0[1] - p1[1]), 1)]
    return ext


def strip_and_sheet(R, base, set_, ep):
    rg, rec = R["rgbs"], R["recs"]
    n = len(rg)
    idx = sorted({0, n // 4, n // 2, (3 * n) // 4, n - 1}) if n > 1 else [0]
    if n >= 5:
        idx = [0, round((n - 1) / 4), round((n - 1) / 2), round(3 * (n - 1) / 4), n - 1]
    tiles = []
    for i in idx:
        im = Image.fromarray(rg[i][::2, ::2]).convert("RGB")
        d = ImageDraw.Draw(im)
        d.rectangle([0, 0, 150, 16], fill=(0, 0, 0))
        d.text((3, 2), f"step {rec[i]['step']} {rec[i]['action'] or 'start'}", fill=(255, 255, 0))
        tiles.append(np.asarray(im))
    Image.fromarray(np.concatenate(tiles, 1)).save(base + "_fpv.jpg", quality=70, optimize=True)
    fs = list(range(n)) if n <= MAXF else sorted({int(round(v)) for v in np.linspace(0, n - 1, MAXF)})
    rows = math.ceil(len(fs) / COLS)
    sheet = Image.new("RGB", (COLS * TILE_W, rows * TILE_H))
    for k, i in enumerate(fs):
        sheet.paste(Image.fromarray(rg[i]).convert("RGB").resize((TILE_W, TILE_H), Image.BILINEAR), ((k % COLS) * TILE_W, (k // COLS) * TILE_H))
    sheet.save(base + "_frames.jpg", quality=70, optimize=True)
    return dict(frame_steps=[rec[i]["step"] for i in fs], tile_w=TILE_W, tile_h=TILE_H, cols=COLS)


def diagnose(row, R, gc, gt, o_len, o_ok):
    recs = R["recs"]
    d = R["final_dist"]
    gcd = float(cKDTree(gt[:, [0, 2]]).query(gc[:, [0, 2]].mean(0))[0])
    if not o_ok:
        return "target off navmesh / unreachable (no navigable point within 1 m of GT reachable from start)"
    if R["success"]:
        return f"success: stopped {d:.2f} m from GT after {R['steps']} steps (oracle {o_len:.1f} m)"
    pre = f"perceived goal off by {gcd:.1f} m; " if gcd > 1.0 else ""
    if R["stopped"]:
        if d <= bench.SUCCESS_M:
            return pre + f"stopped within 1 m horizontally but vertical offset dy={R['final_dy']} m"
        return pre + f"stopped early at {d:.2f} m from GT (step {R['steps']}; perceived cloud centroid {gcd:.2f} m from GT)"
    cs = [(round(r["x"] * 2) / 2, round(r["z"] * 2) / 2) for r in recs if r["collided"]]
    if cs:
        (cp, cn), = Counter(cs).most_common(1)
        if cn >= 8 or R["coll"] >= 0.25 * max(R["steps"], 1):
            return pre + f"collision loop at ({cp[0]:.1f},{cp[1]:.1f}): {cn} collisions there, {R['coll']} total in {R['steps']} steps; ended {d:.2f} m from GT"
    md = min(r["d_gt"] for r in recs)
    if md <= bench.SUCCESS_M:
        return pre + f"reached {md:.2f} m but never stopped (stop rule not satisfied); ended {d:.2f} m, budget exhausted"
    tr = np.array([[r["x"], r["z"]] for r in recs])
    pl = float(np.linalg.norm(np.diff(tr, axis=0), axis=1).sum())
    return pre + f"budget exhausted wandering: travelled {pl:.1f} m (oracle {o_len:.1f} m), ended {d:.2f} m from GT, closest {md:.2f} m"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eps", default=None)
    ap.add_argument("--exclude-sets", default="fresh400")
    ap.add_argument("--out", default=os.path.join(bench.BENCH, "viz_fail"))
    ap.add_argument("--results", default=os.path.join(bench.BENCH, "results", "hybrid_v1.tsv"))
    ap.add_argument("--gpu", type=int, default=3)
    a = ap.parse_args()
    bench.LEVEL_DEPTH, bench.START = True, "accept"
    import csv
    res = {(r["set"], r["ep"]): r for r in csv.DictReader(open(a.results), delimiter="\t")}
    rows = [r for r in bench.load_episodes() if (r["set"], r["ep"]) in res and r["set"] not in a.exclude_sets.split(",")]
    if a.eps:
        rows = [r for r in rows if r["set"] + ":" + r["ep"] in a.eps.split(",")]
    os.makedirs(a.out, exist_ok=True)
    os.environ.setdefault("LM_SCAN", "0")      # hybrid_v1 was run without the initial 360 scan (verified: replay matches only with LM_SCAN=0)
    nav = bench.get_navigator("astar_ddppo")
    index, sim, cur, mism = [], None, None, []
    for r in sorted(rows, key=lambda r: r["scene"]):
        if r["scene"] != cur:
            if sim is not None:
                sim.close()
            sim, cur = bench.make_sim(r["scene"], a.gpu, True), r["scene"]
        e = bench.load_episode_data(r)
        gt, gc = np.asarray(e["gt_points"], np.float32), np.asarray(e["goal_cloud"], np.float32)
        pf = sim.pathfinder
        ends = region_ends(pf, gt, e["position"][1])
        o_len, o_pts = geod_multi(pf, e["position"], ends) if ends else (None, None)
        o_ok = o_len is not None and np.isfinite(o_len)
        t0 = time.time()
        R = replay(sim, nav, e, e["remaining"], ends if o_ok else [])
        h = res[(r["set"], r["ep"])]
        mm = (R["success"] != int(h["success"]) or abs(R["final_dist"] - float(h["final_dist"])) > 0.01 or R["steps"] != int(h["actions"]))
        if mm:
            mism.append(f"{r['set']}:{r['ep']} replay succ={R['success']} fd={R['final_dist']} steps={R['steps']} coll={R['coll']} | tsv succ={h['success']} fd={h['final_dist']} steps={h['actions']} coll={h['collisions']}")
        recs = R["recs"]
        diag = diagnose(r, R, gc, gt, o_len or 0, o_ok)
        gcd = float(cKDTree(gt[:, [0, 2]]).query(gc[:, [0, 2]].mean(0))[0])
        base = os.path.join(a.out, f"{r['set']}_{r['ep']}")
        title = (f"{r['set']}:{r['ep']}  {r['target']}  [{r['tag']}]  {'SUCCESS' if R['success'] else 'FAIL'}  final {R['final_dist']} m  "
                 f"steps {R['steps']}/{e['remaining']}  coll {R['coll']}  oracle {o_len if o_ok else float('nan'):.1f} m\n{diag}")
        title = "\n".join(__import__("textwrap").fill(t, 120) for t in title.split("\n"))
        ext = topdown(sim, e, R, gt, gc, ends, o_pts if o_ok else None, title, base + "_top.png")
        fr = strip_and_sheet(R, base, r["set"], r["ep"])
        modes = Counter(x["mode"] for x in recs[1:])
        runs, cur_run, mx = 0, 0, 0
        for x in recs:
            cur_run = cur_run + 1 if x["collided"] else 0
            mx = max(mx, cur_run)
        first = next((x["step"] for x in recs if x["d_gt"] <= bench.SUCCESS_M), None)
        summ = dict(modes=dict(modes), replans=sum(bool(x.get("replan")) for x in recs), fallbacks=modes.get("fallback", 0),
                    max_consec_collisions=mx, first_within_1m_step=first)
        tele = dict(set=r["set"], ep=r["ep"], extent=ext, oracle_path=[[round(float(p[0]), 3), round(float(p[2]), 3)] for p in (o_pts or [])],
                    gt_xz=[[round(float(p[0]), 3), round(float(p[2]), 3)] for p in gt[::8]],
                    goal_cloud_xz=[[round(float(p[0]), 3), round(float(p[2]), 3)] for p in gc[::2]],
                    summary=summ, steps=recs, **fr)
        json.dump(tele, open(base + "_telemetry.json", "w"), separators=(",", ":"))
        index.append(dict(set=r["set"], ep=r["ep"], tag=r["tag"], target=r["target"], success=R["success"], final_dist=R["final_dist"],
                          steps=R["steps"], budget=e["remaining"], collisions=R["coll"], goal_cloud_to_gt_m=round(gcd, 2),
                          oracle_len_m=round(o_len, 2) if o_ok else None, oracle_feasible=bool(o_ok), diagnosis=diag,
                          replay_matches_tsv=not mm, extent=ext, summary=summ,
                          top=os.path.basename(base) + "_top.png", fpv=os.path.basename(base) + "_fpv.jpg",
                          frames=os.path.basename(base) + "_frames.jpg", telemetry=os.path.basename(base) + "_telemetry.json"))
        print(f"{r['set']}:{r['ep']} {r['tag']} succ={R['success']} fd={R['final_dist']} steps={R['steps']} {'MISMATCH' if mm else 'ok'} {time.time()-t0:.0f}s | {diag}", flush=True)
        json.dump(index, open(os.path.join(a.out, "index.json"), "w"), indent=1)
    if sim is not None:
        sim.close()
    print("MISMATCHES", len(mism))
    for m in mism:
        print(m)


if __name__ == "__main__":
    main()
