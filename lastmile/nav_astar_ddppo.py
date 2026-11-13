"""Hybrid last-mile navigator: A* (lastmile.nav_astar.AStarNavigator on the persistent local map) sets the waypoints, DD-PPO
(lastmile.nav_ddppo, VLFM pointnav weights) drives to them with std actions.

Hook into the A* agent: only `steer()` (and `_recover()` for "no path") are overridden; the A* core (stop rule, initial scan, map update,
collision marking, goal region / object body, back-off, limit-cycle handling) is inherited unchanged.
  - waypoint = farthest point of the A* path within LOOK m (euclid) of the agent and in map line of sight; DD-PPO's pointgoal.
  - re-select the waypoint when it is reached (< 0.3 m), every N steps, after 2 collisions in a row, or when the path moved away from it.
  - DD-PPO's own STOP is ignored (STOP logit masked); the A* stop rule (perceived cloud within 0.8 m and visible) ends the episode.
  - no A* path: DD-PPO drives at the nearest point of the perceived cloud.
Env: LM_HYB_LOOK (1.5), LM_HYB_N (5), LM_HYB_REACH (0.3).  Needs the level depth (bench enables it for --nav astar_ddppo).
"""
import math

import numpy as np
import torch

from lastmile.localmap import FORWARD, LEFT, RIGHT, STOP, core_kwargs, env, fwd_vec, wrap, yaw_to
from lastmile.nav_astar import AStarNavigator
from lastmile.nav_ddppo import DDPPONavigator, MAX_DEPTH, MIN_DEPTH


class HybridNavigator(AStarNavigator):
    name = "astar_ddppo"

    def __init__(self, look=1.5, replan_n=5, reach=0.3, **kw):
        super().__init__(look=0.4, **kw)
        self.hlook, self.replan_n, self.reach = look, replan_n, reach
        self.dd = DDPPONavigator(device="cpu")
        self.pol = self.dd.policy
        self.info = {}

    def reset(self, obs, start_info):
        super().reset(obs, start_info)
        self.dd.reset(obs, start_info)
        self.wp, self.wp_age, self.obs, self.n_dd, self.n_fallback, self.n_reselect = None, 0, None, 0, 0, 0
        self.dbg = []

    def act(self, obs):
        self.obs = obs
        a = super().act(obs)
        self.info = {"n_dd": self.n_dd, "n_fallback": self.n_fallback, "n_reselect": self.n_reselect, "n_collisions_core": self.coll_total}
        return a

    # ---------------------------------------------------------------- DD-PPO drive (STOP masked)
    @torch.no_grad()
    def _drive(self, goal_xz, pos, yaw):
        d = np.asarray(goal_xz, float) - pos[[0, 2]]
        fwd = np.array([-math.sin(yaw), -math.cos(yaw)])
        left = np.array([fwd[1], -fwd[0]])
        rho, theta = float(np.hypot(*d)), math.atan2(d @ left, d @ fwd)
        d0 = self.obs.get("depth_level")
        d0 = self.obs["depth"] if d0 is None else d0
        depth = torch.from_numpy(np.asarray(d0, np.float32).reshape(np.asarray(d0).shape[:2]))[None, None]
        x = torch.nn.functional.interpolate(depth, size=(224, 224), mode="area")[0, 0]
        x = ((x.clamp(MIN_DEPTH, MAX_DEPTH) - MIN_DEPTH) / (MAX_DEPTH - MIN_DEPTH)) * (x > 0)
        o = {"depth": x[None, :, :, None], "pointgoal_with_gps_compass": torch.tensor([[rho, theta]], dtype=torch.float32)}
        dd = self.dd
        masks = torch.tensor([[not dd.first]], dtype=torch.bool)
        feats, dd.hidden, _ = self.pol.net(o, dd.hidden, dd.prev, masks)
        logits = self.pol.action_distribution(feats).logits.clone()
        logits[..., STOP] = -1e9
        a = int(logits.argmax(-1).item())
        dd.prev, dd.first = torch.tensor([[a]], dtype=torch.long), False
        self.n_dd += 1
        self.dbg.append((round(rho, 2), round(theta, 2), a, [round(float(v), 2) for v in logits[0]]))
        return a

    # ---------------------------------------------------------------- waypoint
    def _select(self, path, pxz):
        d = np.linalg.norm(path - pxz, axis=1)
        i0 = int(d.argmin())
        best = path[min(i0 + 1, len(path) - 1)]
        for j in range(i0 + 1, len(path)):
            if d[j] > self.hlook:
                break
            if self.m.los(pxz, path[j][None])[0]:
                best = path[j]
        return np.asarray(best, float)

    def steer(self, plan, pos, yaw, obs):
        pxz = pos[[0, 2]]
        path = plan.path
        self.wp_age += 1
        stale = (self.wp is None or np.linalg.norm(self.wp - pxz) < self.reach or self.wp_age >= self.replan_n
                 or self.coll_run >= 2 or (len(path) and np.linalg.norm(path - self.wp, axis=1).min() > 0.5))
        if stale and len(path):
            self.wp, self.wp_age = self._select(path, pxz), 0
            self.n_reselect += 1
        plan.err = 0.0
        if self.wp is None:
            return FORWARD
        return self._drive(self.wp, pos, yaw)

    def _recover(self, pxz):
        """No A* path: DD-PPO toward the nearest point of the perceived cloud."""
        self.n_fallback += 1
        pos = np.asarray(self.obs["pose"]["position"], float)
        yaw = float(self.obs["pose"]["yaw"])
        cxz = self.cloud_xz if getattr(self, "cloud_xz", None) is not None and len(self.cloud_xz) else self._cloud_xz(pos[1])
        if not len(cxz):
            return LEFT
        tgt = cxz[np.linalg.norm(cxz - pxz, axis=1).argmin()]
        return self._drive(tgt, pos, yaw)


def make():
    return HybridNavigator(look=env("LM_HYB_LOOK", 1.5), replan_n=env("LM_HYB_N", 5, int), reach=env("LM_HYB_REACH", 0.3), **core_kwargs())
