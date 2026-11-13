"""Eye-check a navigator trajectory: top-down path + RGB frames.   python -m lastmile.viz --nav ddppo --start stuck --ep test2:382 --out /workspace/ApexNav/lastmile/bench_data/viz"""
import argparse
import os

import imageio
import numpy as np

from lastmile import bench


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nav", default="ddppo")
    ap.add_argument("--start", default="accept")
    ap.add_argument("--ep", required=True)
    ap.add_argument("--budget", type=int, default=150)
    ap.add_argument("--out", default=os.path.join(bench.BENCH, "viz"))
    a = ap.parse_args()
    bench.LEVEL_DEPTH, bench.START = a.nav in ("ddppo", "astar_ddppo"), a.start
    st, ep = a.ep.split(":")
    row = [r for r in bench.load_episodes() if r["set"] == st and r["ep"] == ep][0]
    e = bench.load_episode_data(row)
    sim = bench.make_sim(row["scene"], 3, True)
    nav = bench.get_navigator(a.nav)
    # replicate run_episode but keep frames
    bench.place(sim, e["position"], e["yaw"], e["pitch"])
    gc = np.asarray(e["goal_cloud"], np.float32)
    obs = bench.observe(sim, True, gc, e["pitch"])
    nav.reset(obs, dict(goal_cloud=gc, goal_estimate=gc.mean(0), budget_steps=a.budget, camera=bench.CAMERA))
    table = bench.action_table()
    path, frames = [obs["pose"]["position"].copy()], [obs["rgb"]]
    for i in range(a.budget):
        act = int(nav.act(obs))
        if act == bench.STOP:
            break
        sim.get_agent(0).act(table[act][0])
        obs = bench.observe(sim, True, gc, e["pitch"])
        path.append(obs["pose"]["position"].copy())
        if i % 10 == 9:
            frames.append(obs["rgb"])
    frames.append(obs["rgb"])
    path = np.array(path)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5, 5))
    gt = e["gt_points"]
    ax.scatter(gt[::10, 0], gt[::10, 2], s=2, c="tab:green", label="GT object")
    ax.scatter(gc[:, 0], gc[:, 2], s=8, c="tab:red", label="perceived goal")
    ax.plot(path[:, 0], path[:, 2], "-k", lw=1)
    ax.plot(*path[0][[0, 2]], "bo")
    ax.plot(*path[-1][[0, 2]], "rx")
    ax.set_aspect("equal")
    ax.legend(fontsize=6)
    os.makedirs(a.out, exist_ok=True)
    fig.savefig(f"{a.out}/{a.nav}_{a.start}_{st}_{ep}_path.png", dpi=90)
    n = min(len(frames), 6)
    idx = np.linspace(0, len(frames) - 1, n).astype(int)
    imageio.imwrite(f"{a.out}/{a.nav}_{a.start}_{st}_{ep}_frames.jpg", np.concatenate([frames[i][::2, ::2] for i in idx], 1))
    d, dy = bench.score_point(path[-1], e["gt_points"])
    print("viz", a.ep, "steps", len(path) - 1, "final_dist", round(d, 2), "dy", round(dy, 2))


if __name__ == "__main__":
    main()
