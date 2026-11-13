"""DD-PPO PointNav last-mile navigator (VLFM's pretrained pointnav_weights.pth, habitat-baselines PointNavResNetPolicy).

Interface (adapt to bench.py if it differs):
    nav = DDPPONavigator(goal_mode="nearest"|"centroid"|"footprint", stop_radius=0.8, device="cpu")
    nav.reset(obs, start_info)   # start_info["cloud"]: (N,3) world-frame object points (habitat xyz, y up)
    action = nav.act(obs)        # int: STOP=0, FORWARD=1 (0.25 m), LEFT=2 (30 deg), RIGHT=3
(bench.py: obs["pose"] dict + obs["goal_cloud"], start_info["goal_cloud"]; flat keys below also work.)
obs keys: "depth" (H,W) or (H,W,1) metres (ApexNav 640x480, hfov 79, 0 = invalid); agent pose either
    "position" (3,) + "yaw" (rad, CCW from above, yaw 0 looks along -z; active_approach.scene convention), or
    "position" + "rotation" (3x3 matrix / habitat quaternion). Optional "cloud" in obs overrides the stored cloud.
Policy input conventions (verified in VLFM code): depth resized to 224x224 (area interp), normalized
(d - 0.5) / (5 - 0.5) clipped to [0, 1] (habitat depth sensor, min .5 max 5; invalid/near = 0);
pointgoal = (rho, theta) with the goal in the agent frame (x forward, y left): theta > 0 = goal to the LEFT (CCW).
"""
import math
import os

import numpy as np
import torch

STOP, FORWARD, LEFT, RIGHT = 0, 1, 2, 3
WANT_STOP = -1   # LASTMILE_RH: act() returns this for a distance-based STOP (info['reason']): the caller confirms it or retargets
MIN_DEPTH, MAX_DEPTH = 0.5, 5.0
WEIGHTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights", "pointnav_weights.pth")


def load_policy(path=WEIGHTS, device="cpu"):
    from gym import spaces
    from gym.spaces import Dict as SpaceDict, Discrete
    from habitat_baselines.rl.ddppo.policy import PointNavResNetPolicy
    obs_space = SpaceDict({
        "depth": spaces.Box(0.0, 1.0, (224, 224, 1), np.float32),
        "pointgoal_with_gps_compass": spaces.Box(np.finfo(np.float32).min, np.finfo(np.float32).max, (2,), np.float32)})
    import pickle
    import types

    class _U(pickle.Unpickler):          # the checkpoint's config pickles vlfm dataclasses; stub them (vlfm not installed)
        def find_class(self, mod, name):
            return type(name, (), {}) if mod.startswith("vlfm") else super().find_class(mod, name)
    pm = types.ModuleType("stub_pickle")
    pm.Unpickler, pm.load = _U, pickle.load
    ckpt = torch.load(path, map_location="cpu", weights_only=False, pickle_module=pm)
    # hyper-parameters of the VLFM/DD-PPO checkpoint (config layout is incompatible with this habitat-baselines)
    pol = PointNavResNetPolicy(obs_space, Discrete(4), hidden_size=512, num_recurrent_layers=2, rnn_type="LSTM",
                               resnet_baseplanes=32, backbone="resnet18", normalize_visual_inputs=False)
    pol.load_state_dict(ckpt["state_dict"])
    return pol.to(device).eval()


def _map_kw():
    """LM_MAP_* overrides of the local map built by this navigator (defaults = the old map)."""
    kw = {}
    for k, name, cast in (("margin", "LM_MAP_MARGIN", float), ("radius", "LM_MAP_RADIUS", float), ("floor_h", "LM_MAP_FLOOR_H", float),
                          ("top_h", "LM_MAP_TOP_H", float), ("min_hits", "LM_MAP_MIN_HITS", int)):
        if os.environ.get(name) is not None:
            kw[k] = cast(os.environ[name])
    return kw


def _pose(obs):
    """bench.py obs: obs["pose"] = {position, yaw, pitch}; flat keys also accepted."""
    return {**obs, **obs["pose"]} if "pose" in obs else obs


def _yaw_of(obs):
    if "yaw" in obs:
        return float(obs["yaw"])
    r = obs["rotation"]
    if hasattr(r, "w"):                       # habitat / numpy-quaternion
        import quaternion
        r = quaternion.as_rotation_matrix(r)
    f = np.asarray(r, float) @ np.array([0.0, 0.0, -1.0])
    return math.atan2(-f[0], -f[2])


def _hull(p):
    p = sorted(map(tuple, np.unique(np.round(p, 3), axis=0)))
    if len(p) < 3:
        return np.array(p)
    cr = lambda o, a, b: (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lo, up = [], []
    for q in p:
        while len(lo) >= 2 and cr(lo[-2], lo[-1], q) <= 0:
            lo.pop()
        lo.append(q)
    for q in reversed(p):
        while len(up) >= 2 and cr(up[-2], up[-1], q) <= 0:
            up.pop()
        up.append(q)
    return np.array(lo[:-1] + up[:-1])


def _nearest_on_polygon(poly, pt):
    best, bd = poly[0], 1e18
    for a, b in zip(poly, np.roll(poly, -1, 0)):
        ab = b - a
        t = 0.0 if not ab.any() else float(np.clip((pt - a) @ ab / (ab @ ab), 0, 1))
        q = a + t * ab
        d = float(np.linalg.norm(q - pt))
        if d < bd:
            best, bd = q, d
    return best


class DDPPONavigator:
    def __init__(self, goal_mode="nearest", stop_radius=0.8, device="cpu", weights=WEIGHTS, max_steps=None):
        assert goal_mode in ("centroid", "nearest", "footprint", "freemap", "navmesh")
        self.goal_mode, self.stop_radius, self.max_steps = goal_mode, stop_radius, max_steps
        self.device = torch.device(device)
        self.kind = os.environ.get("LASTMILE_POLICY", "vlfm")                  # vlfm (default) | dcon (gibson se-resneXt101, 256 px, 0-10 m)
        if self.kind == "dcon":
            from lastmile.ddppo_policy_dcon import load_ddppo_policy
            self.policy = load_ddppo_policy(os.path.join(os.path.dirname(WEIGHTS), "gibson-2plus-se-resneXt101-lstm1024.pth"), str(self.device))
            self.res, self.dlo, self.dhi, self.hid = 256, 0.0, 10.0, 1024
        else:
            self.policy = load_policy(weights, self.device)
            self.res, self.dlo, self.dhi, self.hid = 224, MIN_DEPTH, MAX_DEPTH, 512
        self.cloud = None
        self.info = {}
        self.sample = bool(int(os.environ.get("LASTMILE_SAMPLE", 0)))            # sample actions (VLFM wrapper default) instead of argmax
        self.free_r = float(os.environ.get("LASTMILE_FREE_R", 0.9))             # freemap: free point must lie within this of the cloud
        self.sim = None
        # new behaviour, each behind its own switch (all default off)
        self.snap2 = bool(int(os.environ.get("LASTMILE_SNAP2", 0)))              # snap to observed-free cell in the agent's own FREE component
        self.snap2_r = float(os.environ.get("LASTMILE_SNAP2_R", 1.5))           # candidate cells within this of the cloud
        self.los_wp = bool(int(os.environ.get("LASTMILE_LOS", 0)))               # line-of-sight waypoint along the distance field
        self.wp_max, self.wp_min = float(os.environ.get("LASTMILE_WP_MAX", 1.0)), float(os.environ.get("LASTMILE_WP_MIN", 0.4))
        self.arrive = bool(int(os.environ.get("LASTMILE_ARRIVE", 0)))            # ignore policy STOP far from the goal; stagnation stop
        self.arrive_rho = float(os.environ.get("LASTMILE_ARRIVE_RHO", 0.3))
        self.stag_n, self.stag_gain = int(os.environ.get("LASTMILE_STAG_N", 40)), 0.1
        self.patch_mode = os.environ.get("LASTMILE_PATCH", "off")                # collision patches in OUR map: off | cur (old: 0.3 m wide, inflated) | tiny
        self.patch_ttl = int(os.environ.get("LASTMILE_PATCH_TTL", 20))
        self.probe = bool(int(os.environ.get("LASTMILE_PROBE", 0)))              # blocked FORWARD: turn through the headings, keep the first that moves
        self.diag = bool(int(os.environ.get("LASTMILE_DIAG", 0)))
        self.recover = bool(int(os.environ.get("LASTMILE_RECOVER", 0)))          # collision recovery: sidestep towards the freer side
        self.ignore_stop = float(os.environ.get("LASTMILE_IGNORE_STOP", 0))     # >0: policy STOP is ignored (recovery turn) while rho > this
        self.rec_turns = int(os.environ.get("LASTMILE_REC_TURNS", 2))
        # live-fix switches (default off = F_full_cur): no radius / snapped / stagnation STOP in the first GRACE steps unless the
        # nearest cloud point is within GRACE_R (the stop radius itself is LASTMILE_STOP_R, default 0.8)
        self.grace = int(os.environ.get("LASTMILE_GRACE", 0))
        self.grace_r = float(os.environ.get("LASTMILE_GRACE_R", 0.5))
        self.stag_d = float(os.environ.get("LASTMILE_STAG_D", 1.5))     # stagnation STOP only within this of the cloud (else keep trying until the cap)
        self.rh = bool(int(os.environ.get("LASTMILE_RH", 0)))           # receding horizon: distance-based STOPs are returned as WANT_STOP (see retarget)
        self.dist_q = float(os.environ.get("LASTMILE_DIST_Q", 0))      # >0: object distance = this percentile of the cloud distances (robust to stray points)
        self.wp_override = None                                          # LASTMILE_APEXWP: set_waypoint() xz replaces goal_xz + _waypoint for the pointgoal

    def retarget(self, cloud=None):
        """Receding horizon: a new goal cloud, fresh policy memory and stagnation clock, same local map / patches / steps."""
        if cloud is not None:
            self.cloud = np.asarray(cloud, float).reshape(-1, 3)
        self.hidden = torch.zeros(1, self.policy.net.num_recurrent_layers, self.hid, device=self.device)
        self.prev = torch.zeros(1, 1, device=self.device, dtype=torch.long)
        self.first, self.snap, self.snap_d, self.last_act = True, None, None, None
        self.best_rho, self.best_step = 1e9, self.steps

    def set_waypoint(self, xz):
        """Pointgoal override (world x, z; None = off): the DD-PPO pointgoal aims here instead of goal_xz / _waypoint. The stop
        rules stay on the object cloud."""
        self.wp_override = None if xz is None else np.asarray(xz, float)

    def _stop(self, reason):
        self.info["reason"] = reason
        return WANT_STOP if self.rh else STOP

    def set_sim(self, sim):          # privileged (goal_mode "navmesh" only)
        self.sim = sim

    def reset(self, obs=None, start_info=None):
        self.hidden = torch.zeros(1, self.policy.net.num_recurrent_layers, self.hid, device=self.device)
        self.prev = torch.zeros(1, 1, device=self.device, dtype=torch.long)
        self.first, self.steps, self.info = True, 0, {}
        self.lmap, self.snap, self.snap_d = None, None, None
        self.best_rho, self.best_step = 1e9, 0
        self.probe_turns, self.probing, self.pr_pos, self.n_probe = 0, False, None, 0
        self.prev_fwd_pos, self.coll_pos, self.coll_k, self.n_patch = None, None, 0, 0
        self.forced, self.last_pos, self.last_act, self.n_rec = [], None, None, 0
        self.reset_pose = (obs or {}).get("pose")
        si = start_info or {}
        cloud = si.get("goal_cloud", si.get("cloud"))
        self.cloud = None if cloud is None else np.asarray(cloud, float).reshape(-1, 3)

    # ------------------------------------------------------------------ goal
    def _freemap_goal(self, obs, pos_xz, c):
        """Nearest observed-FREE cell (agent's own connected non-occupied component, obstacle clearance >= radius) to the
        cloud, within free_r of it; tighten-only. Built from the agent's own depth/odometry (lastmile.localmap)."""
        from scipy import ndimage
        from lastmile.localmap import LocalMap
        if self.lmap is None:
            self.lmap = LocalMap(pos_xz, **_map_kw())
        if obs.get("depth") is not None:
            self.lmap.update(np.asarray(obs["position"], float), _yaw_of(obs), float(obs.get("pitch", 0.0)), obs["depth"])
        m = self.lmap
        o, dist = m._derived(False)
        free = m.free & ~o & (dist >= m.radius)
        r0, c0 = m.cell(pos_xz)
        lab, _ = ndimage.label(~o, structure=np.ones((3, 3), bool))
        if m.inside(r0, c0):
            comp = lab == lab[r0, c0]
            free = free & comp if lab[r0, c0] > 0 else free
        # window around the cloud
        lo, hi = c.min(0) - self.free_r, c.max(0) + self.free_r
        (r_a, c_a), (r_b, c_b) = m.cell(lo), m.cell(hi)
        r_a, c_a, r_b, c_b = max(r_a, 0), max(c_a, 0), min(r_b + 1, m.n), min(c_b + 1, m.n)
        if r_b <= r_a or c_b <= c_a:
            return None
        rr, cc = np.nonzero(free[r_a:r_b, c_a:c_b])
        if not len(rr):
            return None
        xz = m.world(rr + r_a, cc + c_a)
        cs = c[:: max(1, len(c) // 400)]
        dmin = np.sqrt(((xz[:, None, :] - cs[None]) ** 2).sum(-1)).min(1)
        ok = dmin <= self.free_r
        if not ok.any():
            return None
        i = int(np.argmin(np.where(ok, dmin, 1e9)))
        if self.snap is None or dmin[i] < self.snap_d - 1e-6:
            self.snap, self.snap_d = xz[i], float(dmin[i])
        return self.snap

    def _update_map(self, obs, pos_xz):
        from lastmile.localmap import LocalMap
        if self.lmap is None:
            self.lmap = LocalMap(pos_xz, **_map_kw())
        if obs.get("depth") is not None:
            self.lmap.update(np.asarray(obs["position"], float), _yaw_of(obs), float(obs.get("pitch", 0.0)), obs["depth"])
        self._patches(obs)
        return self.lmap

    def _patches(self, obs):
        """A FORWARD that did not move (< half a step) stamps a collision patch ahead (LASTMILE_PATCH)."""
        m = self.lmap
        m.now = self.steps
        if self.patch_mode == "off":
            return
        pos = np.asarray(obs["position"], float)
        if self.last_act == FORWARD and self.prev_fwd_pos is not None and np.linalg.norm(pos - self.prev_fwd_pos) < 0.125:
            if self.coll_pos is not None and np.linalg.norm(pos - self.coll_pos) < 0.1:
                self.coll_k += 1
            else:
                self.coll_k = 0
            self.coll_pos = pos.copy()
            m.patch_inflate = self.patch_mode == "cur"
            m.patch_ttl = self.patch_ttl if self.patch_mode == "tiny" else 10 ** 9
            m.mark_collision(pos, _yaw_of(obs), self.coll_k, tiny=self.patch_mode == "tiny")
            self.n_patch += 1
        self.prev_fwd_pos = pos.copy()

    def _snap2_goal(self, obs, pos_xz, c):
        """Nearest observed-free, radius-clear cell to the cloud inside the agent's OWN connected component of such cells
        (no pockets behind walls; unknown space does not connect). Tighten-only (never moves away from the object)."""
        from scipy import ndimage
        m = self._update_map(obs, pos_xz)
        o, dist = m._derived(True)
        base = m.free & ~o & (dist >= m.radius)
        r0, c0 = m.cell(pos_xz)
        rr = int(0.15 / m.res)
        if m.inside(r0, c0):
            base[max(r0 - rr, 0):r0 + rr + 1, max(c0 - rr, 0):c0 + rr + 1] |= (dist[max(r0 - rr, 0):r0 + rr + 1, max(c0 - rr, 0):c0 + rr + 1] >= 0.1)
        lab, _ = ndimage.label(base, structure=np.ones((3, 3), bool))
        if not (m.inside(r0, c0) and lab[r0, c0] > 0):
            return self.snap
        base = lab == lab[r0, c0]
        lo, hi = c.min(0) - self.snap2_r, c.max(0) + self.snap2_r
        (r_a, c_a), (r_b, c_b) = m.cell(lo), m.cell(hi)
        r_a, c_a, r_b, c_b = max(r_a, 0), max(c_a, 0), min(r_b + 1, m.n), min(c_b + 1, m.n)
        if r_b <= r_a or c_b <= c_a:
            return self.snap
        rr_, cc_ = np.nonzero(base[r_a:r_b, c_a:c_b])
        if not len(rr_):
            return self.snap
        xz = m.world(rr_ + r_a, cc_ + c_a)
        cs = c[:: max(1, len(c) // 400)]
        dmin = np.sqrt(((xz[:, None, :] - cs[None]) ** 2).sum(-1)).min(1)
        ok = dmin <= self.snap2_r
        if not ok.any():
            return self.snap
        i = int(np.argmin(np.where(ok, dmin, 1e9)))
        if self.snap is None or dmin[i] < self.snap_d - 1e-6:
            self.snap, self.snap_d = xz[i], float(dmin[i])
        return self.snap

    def _waypoint(self, pxz, goal):
        """Aim at the goal if it is in clear sight on the map (unseen cells never block, last 0.3 m exempt); otherwise follow
        the obstacle-aware distance field from the goal and aim at the farthest route point still in sight (<= wp_max, >= wp_min)."""
        from lastmile.localmap import Field
        m = self.lmap
        if np.linalg.norm(goal - pxz) <= self.wp_max or m.los(pxz, goal)[0]:
            return goal
        cost, _ = m.costs(pxz, relax=2)
        pad = 1.5
        lo, hi = np.minimum(pxz, goal) - pad, np.maximum(pxz, goal) + pad
        (r_a, c_a), (r_b, c_b) = m.cell(lo), m.cell(hi)
        win = (max(r_a, 0), min(r_b + 1, m.n), max(c_a, 0), min(c_b + 1, m.n))
        agent_rc, goal_rc = tuple(int(v) for v in m.cell(pxz)), tuple(int(v) for v in m.cell(goal))
        f = Field(cost, [goal_rc], win, m.res, agent_rc)
        path = f.path_from(agent_rc) if f.valid else []
        if len(path) < 2:
            return goal
        pts = m.world(np.array([p[0] for p in path]), np.array([p[1] for p in path]))
        along = np.linalg.norm(pts - pxz, axis=1)
        vis = m.los(pxz, pts)
        best = None
        for k in range(len(pts)):
            if along[k] > self.wp_max:
                break
            if vis[k] and along[k] >= self.wp_min:
                best = pts[k]
        if best is None:                        # nothing in sight beyond the floor: first path point that clears it
            k = int(np.argmax(along >= self.wp_min)) if (along >= self.wp_min).any() else len(pts) - 1
            best = pts[k]
        return best

    def goal_xz(self, pos_xz, obs=None):
        c = self.cloud[:, [0, 2]]
        if self.snap2:
            g = self._snap2_goal(obs, pos_xz, c)
            return c[np.argmin(np.linalg.norm(c - pos_xz, axis=1))] if g is None else g
        if self.goal_mode == "freemap":
            g = self._freemap_goal(obs, pos_xz, c)
            return c[np.argmin(np.linalg.norm(c - pos_xz, axis=1))] if g is None else g
        if self.goal_mode == "navmesh":
            n = c[np.argmin(np.linalg.norm(c - pos_xz, axis=1))]
            if self.snap is None and self.sim is not None:        # privileged GT navmesh snap, once
                y = float(obs["position"][1])
                sp = np.asarray(self.sim.pathfinder.snap_point(np.array([n[0], y, n[1]], np.float32)), float)
                self.snap = sp[[0, 2]] if np.isfinite(sp).all() else n
            return n if self.snap is None else self.snap
        if self.goal_mode == "centroid":
            return c.mean(0)
        if self.goal_mode == "nearest":
            return c[np.argmin(np.linalg.norm(c - pos_xz, axis=1))]
        h = _hull(c)
        if len(h) < 3:
            return c[np.argmin(np.linalg.norm(c - pos_xz, axis=1))]
        return _nearest_on_polygon(h, pos_xz)

    def _diag(self, obs, pxz, goal):
        """LASTMILE_DIAG=1: log the free-map goal vs the privileged navmesh-snapped goal (final-step values in info)."""
        c = self.cloud[:, [0, 2]]
        n = c[np.argmin(np.linalg.norm(c - pxz, axis=1))]
        fm = self._freemap_goal(obs, pxz, c)
        y = float(obs["position"][1])
        sp = np.asarray(self.sim.pathfinder.snap_point(np.array([n[0], y, n[1]], np.float32)), float)
        sp = sp[[0, 2]] if np.isfinite(sp).all() else n
        i = self.info
        i["d_nav_raw"] = round(float(np.linalg.norm(sp - n)), 2)
        i["d_fm_raw"] = round(float("nan") if fm is None else float(np.linalg.norm(fm - n)), 2)
        i["d_fm_nav"] = round(float("nan") if fm is None else float(np.linalg.norm(fm - sp)), 2)
        i["fm_none"] = int(fm is None)
        i["nav_to_cloud"] = round(float(np.linalg.norm(c - sp, axis=1).min()), 2)
        if fm is not None:
            from lastmile.localmap import LocalMap  # noqa
            r, cc = self.lmap.cell(fm)
            i["fm_to_cloud"] = round(float(np.linalg.norm(c - fm, axis=1).min()), 2)
            import habitat_sim
            p = habitat_sim.ShortestPath()
            p.requested_start = np.asarray(obs["position"], np.float32)
            p.requested_end = np.array([fm[0], y, fm[1]], np.float32)
            i["fm_geo"] = round(float(p.geodesic_distance) if self.sim.pathfinder.find_path(p) else 99.0, 2)
            p.requested_end = np.array([sp[0], y, sp[1]], np.float32)
            i["nav_geo"] = round(float(p.geodesic_distance) if self.sim.pathfinder.find_path(p) else 99.0, 2)

    def object_distance(self, pos_xz):
        d = np.linalg.norm(self.cloud[:, [0, 2]] - pos_xz, axis=1)
        return float(d.min() if self.dist_q <= 0 or len(d) < 20 else np.percentile(d, self.dist_q))

    # ------------------------------------------------------------------ step
    @torch.no_grad()
    def act(self, obs):
        obs = _pose(obs)
        if obs.get("depth") is None:
            raise ValueError("DDPPONavigator needs depth: run the bench with --render")
        gc = obs.get("goal_cloud", obs.get("cloud"))
        if gc is not None and len(gc):
            self.cloud = np.asarray(gc, float).reshape(-1, 3)
        if self.cloud is None or not len(self.cloud):
            return STOP
        pos = np.asarray(obs["position"], float)
        pxz, yaw = pos[[0, 2]], _yaw_of(obs)
        dist = self.object_distance(pxz)
        self.steps += 1
        self.info = {"object_dist": dist}
        early = self.steps <= self.grace and dist > self.grace_r       # just after the accept: no distance-based STOP
        if dist <= self.stop_radius and not early:
            return self._stop("radius")
        if self.max_steps and self.steps > self.max_steps:
            self.info["reason"] = "max_steps"
            return STOP
        goal = self.goal_xz(pxz, obs)
        if self.diag:
            self._diag(obs, pxz, goal)
        self.info["goal_to_cloud"] = round(float(np.linalg.norm(self.cloud[:, [0, 2]] - goal, axis=1).min()), 2)
        if self.arrive:
            rho_g = float(np.linalg.norm(goal - pxz))
            if rho_g < self.best_rho - self.stag_gain:
                self.best_rho, self.best_step = rho_g, self.steps
            elif self.steps - self.best_step >= self.stag_n and dist <= self.stag_d and not early:
                return self._stop("stagnation")
            if self.snap2 and rho_g <= 0.25 and dist <= 1.0 and not early:
                return self._stop("snapped")
        if self.los_wp and not self.snap2 and self.goal_mode != "freemap":
            self._update_map(obs, pxz)
        if self.wp_override is not None:
            goal = self.wp_override
        elif self.los_wp and self.lmap is not None:
            goal = self._waypoint(pxz, goal)
        d = goal - pxz
        fwd = np.array([-math.sin(yaw), -math.cos(yaw)])
        left = np.array([fwd[1], -fwd[0]])            # left of forward (habitat: x right, -z forward, y up)
        rho, theta = float(np.hypot(*d)), math.atan2(d @ left, d @ fwd)
        self.info.update(rho=rho, theta=theta)
        d0 = obs["depth_level"] if obs.get("depth_level") is not None else obs["depth"]   # bench --nav ddppo adds a level (pitch 0) depth
        depth = np.asarray(d0, np.float32).reshape(np.asarray(d0).shape[:2])
        x = torch.from_numpy(depth)[None, None]
        if self.kind == "dcon":
            if x.shape[-1] != self.res:
                x = torch.nn.functional.interpolate(x, size=(self.res, self.res), mode="nearest-exact")
            x = x[0, 0]
        else:
            x = torch.nn.functional.interpolate(x, size=(224, 224), mode="area")[0, 0]
        x = ((x.clamp(self.dlo, self.dhi) - self.dlo) / (self.dhi - self.dlo)) * (x > 0)   # invalid (0) stays 0
        o = {"depth": x[None, :, :, None].to(self.device),
             "pointgoal_with_gps_compass": torch.tensor([[rho, theta]], dtype=torch.float32, device=self.device)}
        if self.probe:
            moved = self.pr_pos is None or np.linalg.norm(pos - self.pr_pos) >= 0.125
            if self.last_act == FORWARD and not moved and not self.probing:
                self.probing, self.probe_turns = True, 0
                self.n_probe += 1
            elif self.probing and self.last_act == FORWARD and moved:
                self.probing = False                         # found a free heading and moved: hand back to the policy
            self.pr_pos = pos.copy()
            if self.probing:
                if self.last_act == FORWARD:                 # blocked again: next heading
                    a = LEFT
                    self.probe_turns += 1
                else:                                        # just turned: test it
                    a = FORWARD
                if self.probe_turns >= 12:
                    self.probing = False
                else:
                    self.prev, self.first, self.last_act = torch.tensor([[a]], dtype=torch.long, device=self.device), False, a
                    self.info["probes"] = self.n_probe
                    return a
        if self.recover:
            if self.last_act == FORWARD and self.last_pos is not None and np.linalg.norm(pos - self.last_pos) < 0.125 and not self.forced:
                dd = depth[:, : depth.shape[1]]
                w = dd.shape[1] // 2
                side = LEFT if np.median(dd[:, :w]) >= np.median(dd[:, w:]) else RIGHT     # image left = agent left
                self.forced = [side] * self.rec_turns + [FORWARD]
                self.n_rec += 1
            self.last_pos = pos.copy()
            if self.forced:
                a = self.forced.pop(0)
                self.prev, self.first, self.last_act = torch.tensor([[a]], dtype=torch.long, device=self.device), False, a
                self.info["recoveries"] = self.n_rec
                return a
        masks = torch.tensor([[not self.first]], dtype=torch.bool, device=self.device)
        if self.kind == "dcon":
            a, self.hidden = self.policy.act(o["depth"], o["pointgoal_with_gps_compass"], self.hidden, self.prev, masks, deterministic=not self.sample)
            a = a.view(1, 1)
        else:
            out = self.policy.act(o, self.hidden, self.prev, masks, deterministic=not self.sample)
            a, self.hidden = (out.actions, out.rnn_hidden_states) if hasattr(out, "actions") else out
        self.prev, self.first = a.clone(), False
        act = int(a.item())
        if act == STOP and self.arrive and rho_g > self.arrive_rho:      # policy STOP far from the (true) goal: ignore
            act = LEFT if self.steps % 2 else RIGHT
            self.prev = torch.tensor([[act]], dtype=torch.long, device=self.device)
        if act == STOP and self.recover and self.ignore_stop > 0 and rho > self.ignore_stop:
            act = LEFT if self.steps % 2 else RIGHT
            self.prev = torch.tensor([[act]], dtype=torch.long, device=self.device)
        self.last_act = act
        self.info["probes"] = self.n_probe
        self.info["recoveries"] = self.n_rec
        if act == STOP:
            self.info["reason"] = "policy"
        return act


def make():
    """bench.py hook (--nav ddppo); env LASTMILE_GOAL (nearest|centroid|footprint), LASTMILE_STOP_R, LASTMILE_DEVICE."""
    if os.environ.get("LASTMILE_SEED"):
        torch.manual_seed(int(os.environ["LASTMILE_SEED"]))
    nav = DDPPONavigator(os.environ.get("LASTMILE_GOAL", "nearest"), float(os.environ.get("LASTMILE_STOP_R", 0.8)),
                         os.environ.get("LASTMILE_DEVICE", "cpu"))
    nav.name = "ddppo"
    return nav
