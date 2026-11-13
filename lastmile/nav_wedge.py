"""Wedge guard on top of F_full_cur (DD-PPO, VLFM weights; sliding OFF; deployable: own depth/odometry map only).
Before each FORWARD chosen by the policy, check the clearance of the step footprint on the agent's own local map
(clearance = distance of the swept points to the nearest observed obstacle minus the agent radius). If the step would end with
clearance < WEDGE_CMIN, or the previous forward was a partial move (displacement < WEDGE_PART) and this forward has the same heading,
a turn (LEFT/RIGHT) whose forward step has adequate clearance (preferring the one facing the goal) replaces it. At most WEDGE_MAXOVR
consecutive overrides, then the policy is released. WEDGE_FIRST=1: the same check is applied at the very first step (accept pose on the edge).
Env: WEDGE_CMIN (0.12), WEDGE_PART (0.2), WEDGE_MAXOVR (3), WEDGE_TURN_DEG (30), WEDGE_PARTIAL (1), WEDGE_CLEAR (1), WEDGE_FIRST (0).
"""
import math
import os

import numpy as np
import torch

from lastmile.nav_ddppo import DDPPONavigator, FORWARD, LEFT, RIGHT, STOP, _pose, _yaw_of


class WedgeNavigator(DDPPONavigator):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.cmin = float(os.environ.get("WEDGE_CMIN", 0.12))
        self.part = float(os.environ.get("WEDGE_PART", 0.2))
        self.maxovr = int(os.environ.get("WEDGE_MAXOVR", 3))
        self.turn = math.radians(float(os.environ.get("WEDGE_TURN_DEG", 30)))
        self.use_partial = bool(int(os.environ.get("WEDGE_PARTIAL", 1)))
        self.use_clear = bool(int(os.environ.get("WEDGE_CLEAR", 1)))
        self.first_guard = bool(int(os.environ.get("WEDGE_FIRST", 0)))
        self.best = bool(int(os.environ.get("WEDGE_BEST", 0)))       # no adequate heading: take the neighbour that is clearly better than the current one
        self.trace = bool(int(os.environ.get("WEDGE_TRACE", 0)))

    def reset(self, obs=None, start_info=None):
        super().reset(obs, start_info)
        self.w_prev = None            # (pos, yaw, partial?) of the last executed forward
        self.w_last_fwd = None
        self.n_ovr, self.run_ovr, self.n_part = 0, 0, 0

    def _clear(self, pxz, yaw):
        """Clearance of a 0.25 m forward step at heading yaw (m beyond the agent radius); +inf if the map is unavailable."""
        m = self.lmap
        if m is None:
            return float("inf")
        _, dist = m._derived(False)
        f = np.array([-math.sin(yaw), -math.cos(yaw)])
        pts = np.array([pxz + f * t for t in (0.0, 0.125, 0.25)])
        v = m.lookup(dist, pts, 99.0)
        return float(v[1:].min() - m.radius)

    def act(self, obs):
        o = _pose(obs)
        a = super().act(obs)
        pos, yaw = np.asarray(o["position"], float), _yaw_of(o)
        pxz = pos[[0, 2]]
        # was the previous executed action a partial forward?
        if self.w_last_fwd is not None:
            p0, y0 = self.w_last_fwd
            if np.linalg.norm(pos[[0, 2]] - p0) < self.part:
                self.w_prev = (y0, True)
                self.n_part += 1
            else:
                self.w_prev = None
            self.w_last_fwd = None
        if self.trace:
            cl = [round(self._clear(pxz, yaw + k * self.turn), 2) for k in (-2, -1, 0, 1, 2)]
            print(f"[wedge] t={self.steps} pos=({pos[0]:.2f},{pos[2]:.2f}) yaw={math.degrees(yaw):.0f} a={a} prevpart={self.w_prev is not None} "
                  f"edge={float(self.lmap.lookup(self.lmap._derived(False)[1], pxz[None], 99)[0]) - self.lmap.radius:.2f} clr(-60..60)={cl}" if self.lmap is not None else f"[wedge] t={self.steps} a={a}", flush=True)
        if a != FORWARD or self.lmap is None:
            self.run_ovr = 0
            return a
        steps_ok = self.steps > 1 or self.first_guard
        bad = False
        if steps_ok and self.use_clear and self._clear(pxz, yaw) < self.cmin:
            bad = True
        if self.use_partial and self.w_prev is not None:
            dy = abs((yaw - self.w_prev[0] + math.pi) % (2 * math.pi) - math.pi)
            if dy < math.radians(15):
                bad = True
        if bad and self.run_ovr < self.maxovr:
            th = self.info.get("theta", 0.0)
            cur = self._clear(pxz, yaw)
            cand = []
            for act, dyaw in ((LEFT, self.turn), (RIGHT, -self.turn)):
                c = self._clear(pxz, yaw + dyaw)
                if self.w_prev is not None and abs((yaw + dyaw - self.w_prev[0] + math.pi) % (2 * math.pi) - math.pi) < math.radians(15):
                    c = -1.0
                cand.append((c >= self.cmin or (self.best and c > cur + 0.04), -abs(th - dyaw), c, act))
            cand = [c for c in cand if c[0]]
            if cand:
                act = max(cand)[3]
                self.n_ovr += 1
                self.run_ovr += 1
                self.prev = torch.tensor([[act]], dtype=torch.long, device=self.device)
                self.last_act = act
                self.info["wedge_ovr"] = self.n_ovr
                self.info["wedge_partial"] = self.n_part
                return act
        self.run_ovr = 0
        self.w_last_fwd = (pxz.copy(), yaw)
        self.info["wedge_ovr"] = self.n_ovr
        self.info["wedge_partial"] = self.n_part
        return a


def make():
    if os.environ.get("LASTMILE_SEED"):
        torch.manual_seed(int(os.environ["LASTMILE_SEED"]))
    nav = WedgeNavigator(os.environ.get("LASTMILE_GOAL", "nearest"), float(os.environ.get("LASTMILE_STOP_R", 0.8)),
                         os.environ.get("LASTMILE_DEVICE", "cpu"))
    nav.name = "wedge"
    return nav
