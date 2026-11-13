"""Standard PointNav sanity eval of DD-PPO checkpoints in OUR HM3D sim (no bench episodes involved).

  python -m lastmile.pointnav_sanity --make-pairs            # sample + save the start/goal pairs once
  python -m lastmile.pointnav_sanity --policy dcon --sliding 1 --turn 10 --sample 1 --out x.json

policy: vlfm (lastmile/weights/pointnav_weights.pth, 224px, depth 0.5-5) | dcon (gibson se-resneXt101-lstm1024, 256px, 0-10 m)
Goal: exact navigable GT point via GPS+compass (habitat IntegratedPointGoal polar), or --goal nonnav (an obstacle-embedded point
0.2-0.5 m beyond the navmesh, like a point ON an object).
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
PAIRS = os.path.join(HERE, "bench_data", "sanity_pairs.json")
STOP, FWD, LEFT, RIGHT = 0, 1, 2, 3


def make_sim(scene, gpu, sensor=None, sliding=False, turn=30.0, radius=0.18, agent_h=0.88):
    import habitat_sim
    from active_approach import scene as S
    cfg = habitat_sim.SimulatorConfiguration()
    cfg.scene_dataset_config_file = S.SCENE_CONFIG
    cfg.scene_id = S.scene_path(scene)
    cfg.gpu_device_id = gpu
    cfg.enable_physics = False
    cfg.allow_sliding = bool(sliding)
    cfg.create_renderer = sensor is not None
    ag = habitat_sim.agent.AgentConfiguration()
    ag.radius, ag.height = radius, agent_h
    if sensor:
        sp = habitat_sim.CameraSensorSpec()
        sp.uuid, sp.sensor_type = "depth", habitat_sim.SensorType.DEPTH
        sp.resolution = [sensor["res"], sensor["res"]]
        sp.position, sp.hfov, sp.far = [0.0, sensor["height"], 0.0], sensor["hfov"], 20.0
        ag.sensor_specifications = [sp]
    A, Sp = habitat_sim.agent.ActionSpec, habitat_sim.agent.ActuationSpec
    ag.action_space = {"move_forward": A("move_forward", Sp(amount=0.25)), "turn_left": A("turn_left", Sp(amount=turn)),
                       "turn_right": A("turn_right", Sp(amount=turn))}
    sim = habitat_sim.Simulator(habitat_sim.Configuration(cfg, [ag]))
    nm = habitat_sim.nav.NavMeshSettings()
    nm.set_defaults()
    nm.agent_radius, nm.agent_height, nm.agent_max_climb, nm.agent_max_slope = radius, agent_h, 0.2, 45.0
    nm.include_static_objects = False
    sim.recompute_navmesh(sim.pathfinder, nm)
    return sim


def geodesic(sim, a, b):
    import habitat_sim
    p = habitat_sim.ShortestPath()
    p.requested_start, p.requested_end = np.asarray(a, np.float32), np.asarray(b, np.float32)
    return float(p.geodesic_distance) if sim.pathfinder.find_path(p) else float("inf")


def pointgoal_polar(pos, rot, goal):
    """habitat IntegratedPointGoalGPSAndCompassSensor, POLAR 2D -> (rho, -phi); theta > 0 = goal to the left."""
    import quaternion
    d = np.asarray(goal, float) - np.asarray(pos, float)
    inv = rot.inverse()
    v = (inv * np.quaternion(0.0, *d) * inv.inverse()).imag
    return float(np.hypot(-v[2], v[0])), -float(np.arctan2(v[0], -v[2]))


def make_pairs(n_scenes=12, per=8, seed=0, gpu=3):
    rng = np.random.RandomState(seed)
    import csv
    scenes = sorted({r["scene"] for r in csv.DictReader(open(os.path.join(HERE, "bench_data", "episodes.tsv")), delimiter="\t")})
    scenes = list(rng.choice(scenes, n_scenes, replace=False))
    out = []
    for sc in scenes:
        sim = make_sim(sc, gpu)
        pf, got, tries = sim.pathfinder, 0, 0
        while got < per and tries < 5000:
            tries += 1
            a, b = (np.asarray(pf.get_random_navigable_point(), np.float32) for _ in range(2))
            if abs(a[1] - b[1]) > 0.5 or not pf.is_navigable(a) or not pf.is_navigable(b):
                continue
            g = geodesic(sim, a, b)
            if not (1.0 <= g <= 10.0) or not np.isfinite(g):
                continue
            yaw = float(rng.uniform(-math.pi, math.pi))
            # non-navigable "object-like" variant of the goal: an obstacle point 0.2-0.5 m beyond the navmesh edge
            nonnav = None
            for _ in range(400):
                ang, r = rng.uniform(0, 2 * math.pi), rng.uniform(0.2, 0.7)
                p = np.array([b[0] + r * math.cos(ang), b[1], b[2] + r * math.sin(ang)], np.float32)
                if pf.is_navigable(p):
                    continue
                s = np.asarray(pf.snap_point(p), np.float32)
                if np.isfinite(s[0]) and 0.2 <= np.hypot(s[0] - p[0], s[2] - p[2]) <= 0.5 and abs(s[1] - b[1]) < 0.3 \
                        and np.isfinite(geodesic(sim, a, s)):
                    nonnav = dict(point=p.tolist(), snapped=s.tolist())
                    break
            out.append(dict(scene=sc, start=a.tolist(), yaw=yaw, goal=b.tolist(), geo=g, nonnav=nonnav))
            got += 1
        sim.close()
        print(sc, got, flush=True)
    json.dump(out, open(PAIRS, "w"))
    print(len(out), "pairs ->", PAIRS)


class Policy:
    """Uniform wrapper: reset(), act(depth_m (H,W) float, rho, theta) -> action id."""

    def __init__(self, name, device, deterministic):
        self.name, self.dev, self.det = name, torch.device(device), deterministic
        if name == "vlfm":
            from lastmile.nav_ddppo import load_policy
            self.pol = load_policy(device=self.dev)
            self.res, self.lo, self.hi, self.hid = 224, 0.5, 5.0, 512
        else:
            from lastmile.ddppo_policy_dcon import load_ddppo_policy
            self.pol = load_ddppo_policy(os.path.join(HERE, "weights", "gibson-2plus-se-resneXt101-lstm1024.pth"), str(self.dev))
            self.res, self.lo, self.hi, self.hid = 256, 0.0, 10.0, 1024

    def reset(self):
        self.h = torch.zeros(1, self.pol.net.num_recurrent_layers, self.hid, device=self.dev)
        self.prev = torch.zeros(1, 1, dtype=torch.long, device=self.dev)
        self.first = True

    @torch.no_grad()
    def act(self, depth, rho, theta):
        x = torch.from_numpy(np.ascontiguousarray(depth, np.float32))[None, None].to(self.dev)
        if x.shape[-1] != self.res:
            x = torch.nn.functional.interpolate(x, size=(self.res, self.res), mode="nearest-exact")
        x = ((x.clamp(self.lo, self.hi) - self.lo) / (self.hi - self.lo))
        o = {"depth": x.permute(0, 2, 3, 1), "pointgoal_with_gps_compass": torch.tensor([[rho, theta]], dtype=torch.float32, device=self.dev)}
        masks = torch.tensor([[not self.first]], dtype=torch.bool, device=self.dev)
        if self.name == "vlfm":
            out = self.pol.act(o, self.h, self.prev, masks, deterministic=self.det)
            a, self.h = (out.actions, out.rnn_hidden_states) if hasattr(out, "actions") else out
        else:
            a, self.h = self.pol.act(o["depth"], o["pointgoal_with_gps_compass"], self.h, self.prev, masks, deterministic=self.det)
        self.prev, self.first = a.clone().view(1, 1), False
        return int(a.item())


def run(args):
    pairs = json.load(open(PAIRS))
    if args.goal == "nonnav":
        pairs = [p for p in pairs if p["nonnav"]]
    dev = f"cuda:{args.gpu}"
    pol = Policy(args.policy, dev, not args.sample)
    sensor = dict(res=args.res or (224 if args.policy == "vlfm" else 256), hfov=args.hfov or (79 if args.policy == "vlfm" else 90),
                  height=args.cam_h or (0.88 if args.policy == "vlfm" else 1.25))
    res, cur, sim = [], None, None
    import habitat_sim
    for i, pr in enumerate(pairs):
        if pr["scene"] != cur:
            if sim:
                sim.close()
            sim = make_sim(pr["scene"], args.gpu, sensor, args.sliding, args.turn, args.radius)
            cur = pr["scene"]
        ag = sim.get_agent(0)
        st = habitat_sim.AgentState()
        st.position = np.asarray(pr["start"], np.float32)
        import quaternion
        st.rotation = quaternion.from_rotation_vector([0.0, pr["yaw"] + (math.pi if args.flip_yaw else 0.0), 0.0])
        ag.set_state(st, reset_sensors=True)
        goal = np.asarray(pr["goal"] if args.goal == "exact" else pr["nonnav"]["point"], np.float32)
        gnav = np.asarray(pr["goal"] if args.goal == "exact" else pr["nonnav"]["snapped"], np.float32)
        pol.reset()
        p0 = np.asarray(pr["start"], float)
        geo0 = geodesic(sim, p0, gnav)
        path_len, coll, stopped, steps, forced = 0.0, 0, False, 0, []
        for steps in range(1, args.max_steps + 1):
            s = ag.get_state()
            d = np.asarray(sim.get_sensor_observations()["depth"], np.float32)
            if args.depth_miss == "far":
                d = np.where(d <= 0, sensor.get("far", 10.0), d)
            rho, theta = pointgoal_polar(s.position, s.rotation, goal)
            if args.flip_theta:
                theta = -theta
            if forced:
                a = forced.pop(0)
                pol.prev = torch.tensor([[a]], dtype=torch.long, device=pol.dev)     # keep the prev-action input consistent
            else:
                a = pol.act(d, rho, theta)
            if a == STOP and not args.ignore_stop:
                stopped = True
                break
            if a == STOP:
                a = LEFT
            b = np.asarray(ag.get_state().position, float)
            ag.act({FWD: "move_forward", LEFT: "turn_left", RIGHT: "turn_right"}[a])
            if a == FWD:
                m = np.linalg.norm(np.asarray(ag.get_state().position, float) - b)
                path_len += m
                coll += int(m < 0.125)
                if m < 0.125 and args.recover and not forced:
                    # collision recovery: turn ~30 deg towards the freer side of the depth image, then 2 steps forward
                    w = d.shape[1] // 2
                    left_free, right_free = np.median(d[:, :w]), np.median(d[:, w:])     # left of image = left of agent
                    k = max(1, int(round(30.0 / args.turn)))
                    forced = [LEFT if left_free >= right_free else RIGHT] * k + [FWD, FWD]
        pos = np.asarray(ag.get_state().position, float)
        g_end = geodesic(sim, pos, gnav)
        eu = float(np.hypot(pos[0] - goal[0], pos[2] - goal[2]))
        succ = stopped and g_end <= 0.2
        res.append(dict(i=i, scene=pr["scene"], geo0=geo0, steps=steps, stopped=int(stopped), g_end=g_end, eu_end=eu,
                        sr02=int(succ), sr1=int(stopped and (g_end <= 1.0 if args.goal == "exact" else eu <= 1.0)),
                        spl=float(succ * geo0 / max(geo0, path_len)), coll=coll, near1=int(g_end <= 1.0 if args.goal == "exact" else eu <= 1.0)))
        print(json.dumps(res[-1]), flush=True)
    if sim:
        sim.close()
    m = lambda k: float(np.mean([r[k] for r in res]))
    summ = dict(vars(args), n=len(res), SR02=m("sr02"), SR1=m("sr1"), near1=m("near1"), SPL=m("spl"), stopped=m("stopped"), coll=m("coll"), steps=m("steps"))
    print("SUMMARY", json.dumps(summ), flush=True)
    if args.out:
        json.dump(dict(summary=summ, eps=res), open(args.out, "w"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--make-pairs", action="store_true")
    ap.add_argument("--policy", default="dcon", choices=["vlfm", "dcon"])
    ap.add_argument("--sliding", type=int, default=0)
    ap.add_argument("--turn", type=float, default=10.0)
    ap.add_argument("--sample", type=int, default=1)
    ap.add_argument("--goal", default="exact", choices=["exact", "nonnav"])
    ap.add_argument("--radius", type=float, default=0.18)
    ap.add_argument("--res", type=int, default=0)
    ap.add_argument("--hfov", type=float, default=0)
    ap.add_argument("--cam-h", type=float, default=0)
    ap.add_argument("--depth-miss", default="zero", choices=["zero", "far"])
    ap.add_argument("--flip-theta", action="store_true")
    ap.add_argument("--flip-yaw", action="store_true")
    ap.add_argument("--recover", action="store_true")
    ap.add_argument("--ignore-stop", action="store_true")
    ap.add_argument("--max-steps", type=int, default=500)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--gpu", type=int, default=3)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    torch.manual_seed(0)
    make_pairs(gpu=a.gpu) if a.make_pairs else run(a)
