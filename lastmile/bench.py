"""Last-mile navigation benchmark harness (offline, habitat-sim).

An episode starts at the agent pose where ApexNav's perception gate ACCEPTED the target; the
navigator must drive to the accepted object and STOP within 1 m (horizontal) of it.

Navigator interface (what nav_ddppo.py etc. implement):

    class Navigator:
        def reset(self, obs, start_info): ...   # once per episode, obs = first observation
        def act(self, obs) -> int: ...          # STOP / FORWARD / LEFT / RIGHT

Actions (ApexNav's): STOP=0, FORWARD=1 (0.25 m), LEFT=2 (30 deg), RIGHT=3 (30 deg); no sliding.
obs (dict), what the real agent has:
    rgb    (480,640,3) uint8, or None if the bench runs with render=False
    depth  (480,640) float32, RAW METRES (0 = invalid), camera res, hfov 79, camera 0.88 m high
    pose   dict: position (3,) world xyz of the agent base, yaw (rad, ccw about +y, 0 looks along -z),
           pitch (rad, camera pitch), from odometry (== simulator state; no noise)
    goal_cloud (N,3) world-frame points of the PERCEIVED accepted object (what the agent knows)
start_info (dict): scene, ep, camera {hfov,width,height,height_m}, goal_cloud (same as obs),
    goal_estimate (3,) = cloud centroid, budget_steps (int), start_position, start_yaw, start_pitch.
Navigators must not use GT points; the harness scores with them (episode['gt_points']).

Run:  python -m lastmile.bench --nav oracle --budget remaining|150 [--eps tsv] [--render]
"""
import argparse
import csv
import json
import math
import os
import sys
import time

import numpy as np

STOP, FORWARD, LEFT, RIGHT = 0, 1, 2, 3
FORWARD_SMALL, LEFT_15, RIGHT_15 = 4, 5, 6      # only with --actions fine
ACTIONS = "std"                                   # "std" (ApexNav's) or "fine" (adds ids 4-6); set by --actions
LEVEL_DEPTH = False                               # extra 224x224 depth at pitch 0 (obs["depth_level"]); only DD-PPO uses it
LEVEL_RES = int(os.environ.get("LASTMILE_LEVEL_RES", 224))
LEVEL_HFOV = float(os.environ.get("LASTMILE_LEVEL_HFOV", 79.0))        # DCON gibson ckpt: 256 px, hfov 90, camera 1.25 m
LEVEL_H = float(os.environ.get("LASTMILE_LEVEL_H", 0.88))
SLIDING = bool(int(os.environ.get("LASTMILE_SLIDING", 0)))               # simulator allow_sliding (default off = ApexNav's)
SMALL_M, SMALL_DEG = 0.10, 15.0                   # fine-action sizes (--small-m)
HFOV, WIDTH, HEIGHT, CAMERA_HEIGHT = 79.0, 640, 480, 0.88
FORWARD_M, TURN_DEG, RADIUS = 0.25, 30.0, 0.18
ACT_TURN_DEG = float(os.environ.get("LASTMILE_TURN_DEG", 30.0))      # turn_left/right amount (DCON policy: 10)
SUCCESS_M, DY_MAX = 1.0, 2.0   # DY_MAX: same storey (HM3D floors are >2.5 m apart); TVs hang ~1.7 m above the base
# bench data (episodes.tsv, episodes/*.npz, results/) lives in lastmile/bench_data so the container sees it;
# `python -m lastmile.build --publish` / `lastmile.sync` mirrors it to
# 3DAwareVerification/outputs/apexnav_online/lastmile_bench_2026-11-11/ (not mounted in the container).
BENCH = os.environ.get("LASTMILE_BENCH", os.path.join(os.path.dirname(os.path.abspath(__file__)), "bench_data"))
CAMERA = dict(hfov=HFOV, width=WIDTH, height=HEIGHT, height_m=CAMERA_HEIGHT)


class Navigator:
    name = "navigator"

    def reset(self, obs, start_info):
        pass

    def act(self, obs):
        raise NotImplementedError


def action_table():
    """id -> (habitat action name, forward metres or 0); fine ids only exist with ACTIONS == 'fine'."""
    t = {FORWARD: ("move_forward", FORWARD_M), LEFT: ("turn_left", 0.0), RIGHT: ("turn_right", 0.0)}
    if ACTIONS == "fine":
        t.update({FORWARD_SMALL: ("move_forward_small", SMALL_M), LEFT_15: ("turn_left_15", 0.0),
                  RIGHT_15: ("turn_right_15", 0.0)})
    return t


COUNT_KEYS = {FORWARD: "n_fwd", LEFT: "n_left", RIGHT: "n_right", FORWARD_SMALL: "n_fwd_small", LEFT_15: "n_left15", RIGHT_15: "n_right15"}


def make_sim(scene, gpu=3, render=False):
    """habitat-sim with ApexNav's agent. render=False needs no GPU memory (no RGB/depth)."""
    import habitat_sim
    from active_approach import scene as S
    cfg = habitat_sim.SimulatorConfiguration()
    cfg.scene_dataset_config_file = S.SCENE_CONFIG
    cfg.scene_id = S.scene_path(scene)
    cfg.gpu_device_id = gpu
    cfg.enable_physics = False
    cfg.allow_sliding = SLIDING
    cfg.create_renderer = render
    agent = habitat_sim.agent.AgentConfiguration()
    agent.radius, agent.height = RADIUS, CAMERA_HEIGHT
    if render:
        specs = []
        for uuid, kind in (("rgb", habitat_sim.SensorType.COLOR), ("depth", habitat_sim.SensorType.DEPTH)):
            sp = habitat_sim.CameraSensorSpec()
            sp.uuid, sp.sensor_type, sp.resolution = uuid, kind, [HEIGHT, WIDTH]
            sp.position, sp.hfov = [0.0, CAMERA_HEIGHT, 0.0], HFOV
            specs.append(sp)
        if LEVEL_DEPTH:      # level (pitch 0) depth for VLFM/DD-PPO pointnav: 224x224, hfov 79, same height; far 20 so the policy's own 5 m clip applies
            sp = habitat_sim.CameraSensorSpec()
            sp.uuid, sp.sensor_type, sp.resolution = "depth_level", habitat_sim.SensorType.DEPTH, [LEVEL_RES, LEVEL_RES]
            sp.position, sp.hfov, sp.far = [0.0, LEVEL_H, 0.0], LEVEL_HFOV, 20.0
            specs.append(sp)
        agent.sensor_specifications = specs
    A, Sp = habitat_sim.agent.ActionSpec, habitat_sim.agent.ActuationSpec
    fine = {}
    if ACTIONS == "fine":
        fine = {"move_forward_small": A("move_forward", Sp(amount=SMALL_M)),
                "turn_left_15": A("turn_left", Sp(amount=SMALL_DEG)),
                "turn_right_15": A("turn_right", Sp(amount=SMALL_DEG))}
    agent.action_space = {**fine, "move_forward": A("move_forward", Sp(amount=FORWARD_M)),
                          "turn_left": A("turn_left", Sp(amount=ACT_TURN_DEG)),
                          "turn_right": A("turn_right", Sp(amount=ACT_TURN_DEG)),
                          "look_up": A("look_up", Sp(amount=TURN_DEG)),
                          "look_down": A("look_down", Sp(amount=TURN_DEG))}
    conf = habitat_sim.Configuration(cfg, [agent])
    nm = habitat_sim.nav.NavMeshSettings()      # = habitat-lab's default_agent_navmesh for ApexNav's config
    nm.set_defaults()
    nm.agent_radius, nm.agent_height, nm.agent_max_climb, nm.agent_max_slope = RADIUS, CAMERA_HEIGHT, 0.2, 45.0
    nm.include_static_objects = False
    sim = habitat_sim.Simulator(conf)
    sim.recompute_navmesh(sim.pathfinder, nm)
    return sim


def render_ok(sim):
    return "depth_level" in sim.get_agent(0)._sensors


def place(sim, position, yaw, pitch):
    import habitat_sim
    from active_approach import scene as S
    st = habitat_sim.AgentState()
    st.position = np.asarray(position, np.float32)
    st.rotation = S.yaw_quat(yaw)
    ag = sim.get_agent(0)
    ag.set_state(st, reset_sensors=True)
    for _ in range(int(round(abs(math.degrees(pitch)) / TURN_DEG))):
        ag.act("look_down" if pitch < 0 else "look_up")
    if LEVEL_DEPTH and render_ok(sim):
        import magnum as mn
        for name, sens in ag._sensors.items():        # look_up/down tilt every sensor: put the level one back to pitch 0
            if name == "depth_level":
                sens.node.rotation = mn.Quaternion()


def pose(sim):
    p = sim.get_agent(0).get_state().position
    q = sim.get_agent(0).get_state().rotation
    import quaternion
    f = quaternion.rotate_vectors(q, np.array([0.0, 0.0, -1.0]))
    return np.asarray(p, float), math.atan2(-f[0], -f[2])


def observe(sim, render, goal_cloud, pitch):
    p, yaw = pose(sim)
    o = {"rgb": None, "depth": None, "pose": {"position": p, "yaw": yaw, "pitch": pitch},
         "goal_cloud": goal_cloud}
    if render:
        ob = sim.get_sensor_observations()
        o["rgb"] = np.ascontiguousarray(ob["rgb"][..., :3])
        o["depth"] = np.asarray(ob["depth"], np.float32)
        if "depth_level" in ob:
            o["depth_level"] = np.asarray(ob["depth_level"], np.float32)
    return o


def score_point(p, gt):
    """(horizontal distance to the nearest GT point, smallest |dy| (point - agent base) among GT points
    within SUCCESS_M horizontally of p, else of the nearest point)."""
    d = np.hypot(gt[:, 0] - p[0], gt[:, 2] - p[2])
    i = int(d.argmin())
    near = d <= SUCCESS_M
    dy = np.abs(gt[near, 1] - p[1]).min() if near.any() else abs(gt[i, 1] - p[1])
    return float(d[i]), float(dy)


def run_episode(sim, nav, ep, budget, render=False):
    """ep: dict with position, yaw, pitch, goal_cloud, gt_points (scoring only), + ids."""
    place(sim, ep["position"], ep["yaw"], ep["pitch"])
    gc = np.asarray(ep["goal_cloud"], np.float32)
    gt = np.asarray(ep["gt_points"], np.float32)
    info = dict(scene=ep["scene"], ep=ep["ep"], camera=CAMERA, goal_cloud=gc,
                goal_estimate=gc.mean(0) if len(gc) else None, budget_steps=budget,
                start_position=np.asarray(ep["position"]), start_yaw=ep["yaw"], start_pitch=ep["pitch"])
    obs = observe(sim, render, gc, ep["pitch"])
    if getattr(nav, "uses_gt", False):      # only the oracle reference may see the GT points
        nav.set_gt(gt)
    nav.reset(obs, info)
    pos0, coll, mind, steps, stopped = obs["pose"]["position"], 0, 1e9, 0, False
    path = [pos0.copy()]
    table = action_table()
    counts = {COUNT_KEYS[k]: 0 for k in table}
    t0 = time.time()
    while steps < budget:
        a = int(nav.act(obs))
        if a == STOP:
            stopped = True
            break
        name, fwd = table[a]            # KeyError = action not available under the current --actions
        before = pose(sim)[0]
        sim.get_agent(0).act(name)
        steps += 1
        counts[COUNT_KEYS[a]] += 1
        after = pose(sim)[0]
        if fwd:
            if np.linalg.norm(after - before) < 0.5 * fwd:
                coll += 1
            path.append(after.copy())
        obs = observe(sim, render, gc, ep["pitch"])
        mind = min(mind, score_point(obs["pose"]["position"], gt)[0])
    p = pose(sim)[0]
    d, dy = score_point(p, gt)
    mind = min(mind, d)
    extra = getattr(nav, "info", {})
    extra = extra() if callable(extra) else dict(extra)
    extra = {k: v for k, v in extra.items() if k != "object_dist"}
    # scoring rule shared with native ApexNav: the episode ends with a STOP either issued by the agent or FORCED at the step cap
    # (native also force-stops on stucking; our navigators stop on their own stagnation rule). success_stopped = old rule (needs an own STOP).
    ok_end = (stopped or steps >= budget) and d <= SUCCESS_M and dy <= DY_MAX
    return dict(**extra, **counts, success=int(ok_end), success_stopped=int(stopped and d <= SUCCESS_M and dy <= DY_MAX), stopped=int(stopped),
                success_forced=int(ok_end), final_dist=round(d, 3), final_dy=round(dy, 2), min_dist=round(mind, 3), actions=steps,
                collisions=coll, within1m=int(d <= SUCCESS_M), secs=round(time.time() - t0, 1))


START = "accept"      # "accept" (gate accept pose) or "stuck" (pose where native ApexNav ended with STUCKING); set by --start


def load_episodes(tsv=None, subset=None):
    if tsv is None and START in ("stuck", "stillstuck"):
        tsv = os.path.join(BENCH, f"episodes_{START}.tsv")
    tsv = tsv or os.path.join(BENCH, "episodes.tsv")
    rows = list(csv.DictReader(open(tsv), delimiter="\t"))
    if START == "stuck":
        rows = [r for r in rows if r.get("rescue_eligible") == "1"]
    if subset:
        rows = [r for r in rows if r["tag"] in subset.split(",") or r["set"] + ":" + r["ep"] in subset.split(",")]
    return rows


def load_episode_data(row):
    z = np.load(os.path.join(BENCH, "episodes", f"{START + '_' if START != 'accept' else ''}{row['set']}_{row['ep']}.npz"))
    return dict(scene=row["scene"], ep=int(row["ep"]), position=z["position"], yaw=float(z["yaw"]),
                pitch=float(z["pitch"]), goal_cloud=z["goal_cloud"], gt_points=z["gt_points"],
                remaining=int(row["remaining_steps"]))


def get_navigator(name):
    if name == "oracle":
        from lastmile.oracle import OracleNavigator
        return OracleNavigator()
    mod = __import__(f"lastmile.nav_{name}", fromlist=["make"])
    return mod.make()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nav", required=True)
    ap.add_argument("--budget", default="remaining", help="'remaining' (rest of the 500-step episode) or an int cap")
    ap.add_argument("--subset", default=None, help="comma list of tags (FAIL,SUCCESS,benchonly_1m,budget) or set:ep")
    ap.add_argument("--exclude-sets", default=None, help="comma list of sets to skip, e.g. fresh400 (held-out test split)")
    ap.add_argument("--render", action="store_true", help="RGB/depth observations (needs GPU memory)")
    ap.add_argument("--gpu", type=int, default=3)
    ap.add_argument("--out", default=None)
    ap.add_argument("--actions", default="std", choices=["std", "fine"],
                    help="fine adds ids 4 FORWARD_SMALL (--small-m), 5 LEFT_15, 6 RIGHT_15 to the standard 0-3")
    ap.add_argument("--start", default="accept", choices=["accept", "stuck", "stillstuck"],
                    help="accept: gate accept pose (default); stuck: pose where native ApexNav ended with STUCKING (FAIL eps only), "
                         "budget = 500 - that step; stillstuck: pose at the first 'Still stuck at the same place' event after the accept")
    ap.add_argument("--small-m", type=float, default=0.10, help="FORWARD_SMALL step in metres (fine only)")
    ap.add_argument("--shard", default=None, help="i/n: run every n-th episode (sorted by scene, ep) starting at i")
    a = ap.parse_args()
    global ACTIONS, SMALL_M
    global START, LEVEL_DEPTH
    LEVEL_DEPTH = a.nav in ("ddppo", "astar_ddppo", "mapcheck", "slide", "clearance", "wedge")
    if LEVEL_DEPTH:
        a.render = True          # DD-PPO needs depth
    ACTIONS, SMALL_M, START = a.actions, a.small_m, a.start
    import importlib                      # `python -m lastmile.bench` runs as __main__: also configure the importable module
    canon = importlib.import_module("lastmile.bench")
    canon.ACTIONS, canon.SMALL_M, canon.START, canon.LEVEL_DEPTH = a.actions, a.small_m, a.start, LEVEL_DEPTH
    rows = load_episodes(subset=a.subset)
    if a.exclude_sets:
        rows = [r for r in rows if r["set"] not in a.exclude_sets.split(",")]
    if a.shard:
        i, n = map(int, a.shard.split("/"))
        rows = sorted(rows, key=lambda r: (r["scene"], r["set"], int(r["ep"])))[i::n]
    nav = get_navigator(a.nav)
    out = a.out or os.path.join(BENCH, "results", f"{a.nav}_{a.budget}.tsv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    res, sim, cur = [], None, None
    for r in sorted(rows, key=lambda r: r["scene"]):
        if r["scene"] != cur:
            if sim is not None:
                sim.close()
            sim, cur = make_sim(r["scene"], a.gpu, a.render), r["scene"]
            if hasattr(nav, "set_sim"):
                nav.set_sim(sim)
        ep = load_episode_data(r)
        budget = ep["remaining"] if a.budget == "remaining" else int(a.budget)
        m = run_episode(sim, nav, ep, budget, a.render)
        m.update(set=r["set"], ep=r["ep"], tag=r["tag"], budget=budget)
        res.append(m)
        print(json.dumps(m), flush=True)
    if sim is not None:
        sim.close()
    keys = ["set", "ep", "tag", "budget", "success", "success_forced", "stopped", "within1m", "final_dist", "final_dy",
            "min_dist", "actions", "collisions", "secs"] + sorted({k for m in res for k in m} - set(
        ["set", "ep", "tag", "budget", "success", "success_forced", "stopped", "within1m", "final_dist", "final_dy", "min_dist", "actions", "collisions", "secs"]))
    with open(out, "w") as f:
        f.write("\t".join(keys) + "\n")
        for m in res:
            f.write("\t".join(str(m.get(k, "")) for k in keys) + "\n")
    for tag in sorted({m["tag"] for m in res}):
        s = [m for m in res if m["tag"] == tag]
        print(f"{tag}: n={len(s)} success={sum(m['success'] for m in s)} actions={np.mean([m['actions'] for m in s]):.1f}")


if __name__ == "__main__":
    main()
