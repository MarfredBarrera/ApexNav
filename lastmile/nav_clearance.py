"""Clearance-seeking last-mile planner (sliding OFF, deployable: own depth/odometry map only).
Plans on lastmile.localmap's occupancy with cost = step * (1 + W * clip((CT - d)/CT, 0, 1)), d = distance to the nearest observed obstacle,
so the route stays in the middle of free space and never hugs walls/corners. Goal = the reachable free cell within CL_GOAL_R of the
perceived object with the best clearance. Followed with lookahead waypoints by DD-PPO pointnav (CL_CTRL=ddppo, default) or a simple
turn/forward controller (CL_CTRL=simple). CL_RIDGE=1: first drive to the clearance ridge (the accept pose sits on the navmesh edge).
Env: CL_W (4), CL_CT (0.7), CL_GOAL_R (0.85), CL_LOOK (0.8), CL_RIDGE (0), CL_CTRL.
"""
import math
import os

import numpy as np
import torch

from lastmile.localmap import Field
from lastmile.nav_ddppo import DDPPONavigator, FORWARD, LEFT, RIGHT, STOP, _pose


class ClearanceNavigator(DDPPONavigator):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.w = float(os.environ.get("CL_W", 4.0))
        self.ct = float(os.environ.get("CL_CT", 0.7))
        self.goal_r = float(os.environ.get("CL_GOAL_R", 0.85))
        self.look = float(os.environ.get("CL_LOOK", 0.8))
        self.ridge = bool(int(os.environ.get("CL_RIDGE", 0)))
        self.ctrl = os.environ.get("CL_CTRL", "ddppo")
        self.snap2 = False
        self.los_wp = True
        self.arrive = True

    def reset(self, obs=None, start_info=None):
        super().reset(obs, start_info)
        self.cgoal, self.cgoal_step, self.ridge_done, self.field = None, -99, not self.ridge, None

    def _cost(self, m, pxz):
        _, hard = m.costs(pxz, relax=1)
        _, dist = m._derived(True)
        f = np.clip((self.ct - dist) / self.ct, 0, 1)
        unk = np.where(m.free | (m.hits >= m.min_hits), 0.0, 0.5)
        return np.where(hard, np.inf, 1.0 + self.w * f + unk).astype(np.float32), dist

    def _window(self, m, pts, pad=2.0):
        lo, hi = np.min(pts, 0) - pad, np.max(pts, 0) + pad
        (r_a, c_a), (r_b, c_b) = m.cell(lo), m.cell(hi)
        return max(r_a, 0), min(r_b + 1, m.n), max(c_a, 0), min(c_b + 1, m.n)

    def _pick_goal(self, m, pxz, c, cost, dist):
        win = self._window(m, np.vstack([pxz[None], c.min(0)[None], c.max(0)[None]]), 1.5)
        arc = tuple(int(v) for v in m.cell(pxz))
        f = Field(cost, [arc], win, m.res, arc)
        if not f.valid:
            return None
        r0, r1, c0, c1 = win
        reach = np.isfinite(f.dist) & (dist[r0:r1, c0:c1] >= m.radius)
        frees = reach & m.free[r0:r1, c0:c1]
        cs = c[:: max(1, len(c) // 300)]
        rr, cc = np.nonzero(frees)
        if len(rr):                                       # observed-free cells first; if none is near the object, optimistic unknown cells
            near = np.sqrt(((m.world(rr + r0, cc + c0)[:, None, :] - cs[None]) ** 2).sum(-1)).min(1)
            if near.min() > self.goal_r:
                rr, cc = np.nonzero(reach)
        else:
            rr, cc = np.nonzero(reach)
        if not len(rr):
            return None
        xz = m.world(rr + r0, cc + c0)
        cs = c[:: max(1, len(c) // 300)]
        dmin = np.sqrt(((xz[:, None, :] - cs[None]) ** 2).sum(-1)).min(1)
        cl = dist[rr + r0, cc + c0]
        ok = dmin <= self.goal_r
        if not ok.any():
            i = int(np.argmin(dmin))
            return xz[i] if dmin[i] <= 1.3 * self.goal_r else None      # nothing usable near the object: aim at the object itself
        sc = np.where(ok, np.minimum(cl, 0.8) - 0.5 * dmin, -1e9)
        return xz[int(np.argmax(sc))]

    def goal_xz(self, pos_xz, obs=None):
        c = self.cloud[:, [0, 2]]
        m = self._update_map(obs, pos_xz)
        cost, dist = self._cost(m, pos_xz)
        self._c, self._d = cost, dist
        if self.cgoal is None or self.steps - self.cgoal_step >= 10:
            g = self._pick_goal(m, pos_xz, c, cost, dist)
            self.cgoal, self.cgoal_step = g, self.steps
        if self.cgoal is None:
            return c[np.argmin(np.linalg.norm(c - pos_xz, axis=1))]
        return self.cgoal

    def _waypoint(self, pxz, goal):
        m, cost, dist = self.lmap, self._c, self._d
        win = self._window(m, np.vstack([pxz, goal]), 1.5)
        arc, grc = tuple(int(v) for v in m.cell(pxz)), tuple(int(v) for v in m.cell(goal))
        if not self.ridge_done:
            r, cc = arc
            if dist[r, cc] >= 0.3 or self.steps > 12:
                self.ridge_done = True
            else:
                fa = Field(cost, [arc], self._window(m, pxz[None], 1.6), m.res, arc)
                r0, r1, c0, c1 = fa.win
                ok = np.isfinite(fa.dist) & (dist[r0:r1, c0:c1] >= 0.35) & (fa.dist < 1.6)
                if ok.any():
                    rr, qq = np.nonzero(ok)
                    i = int(np.argmin(fa.dist[rr, qq]))
                    return m.world(np.array([rr[i] + r0]), np.array([qq[i] + c0]))[0]
                self.ridge_done = True
        f = Field(cost, [grc], win, m.res, arc)
        path = f.path_from(arc) if f.valid else []
        if len(path) < 2:
            return goal
        pts = m.world(np.array([p[0] for p in path]), np.array([p[1] for p in path]))
        seg = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
        k = int(np.searchsorted(seg, self.look))
        return pts[min(k, len(pts) - 1)]

    def act(self, obs):
        a = super().act(obs)
        if self.ctrl == "simple" and a != STOP and "theta" in self.info:
            th = self.info["theta"]
            a = LEFT if th > math.radians(20) else RIGHT if th < -math.radians(20) else FORWARD
            self.prev = torch.tensor([[a]], dtype=torch.long, device=self.device)
            self.last_act = a
        return a


def make():
    if os.environ.get("LASTMILE_SEED"):
        torch.manual_seed(int(os.environ["LASTMILE_SEED"]))
    nav = ClearanceNavigator(os.environ.get("LASTMILE_GOAL", "nearest"), float(os.environ.get("LASTMILE_STOP_R", 0.8)),
                             os.environ.get("LASTMILE_DEVICE", "cpu"))
    nav.name = "clearance"
    return nav
