"""MPPI last-mile navigator on the shared collision-aware local map (lastmile.localmap).

Every step K discrete action sequences (length H over FORWARD/LEFT/RIGHT) are rolled out with the real motion
model (0.25 m, 30 deg); a forward whose destination or midpoint is inside the inflated map leaves the agent in
place and costs C_BUMP. A rollout costs  min_t [ J(x_t) + C_ACT t + W_HEAD heading_err_t + bumps_t ]  where J is
the geodesic cost-to-go (Dijkstra from the goal region, so walls do not trap it) and heading_err is the angle to
the object once within 2.5 m. The categorical sampling distribution is updated by exponentiated-cost weights
(MPPI, ITERS iterations); the A* path follower's action sequence is always one of the samples (nominal), so
MPPI is never worse than A* on its own objective. Deterministic given LM_SEED."""
import math

import numpy as np

from lastmile.localmap import FORWARD, LEFT, RIGHT, TURN, FWD_M, LastMileCore, core_kwargs, env, fwd_vec, wrap, yaw_to
from lastmile.nav_astar import follow_actions, pick_action


class MPPINavigator(LastMileCore):
    name = "mppi"

    def __init__(self, K=96, H=10, iters=3, lam=0.4, c_act=0.02, c_bump=3.0, w_head=0.25, **kw):
        super().__init__(**kw)
        self.K, self.H, self.iters, self.lam = K, H, iters, lam
        self.c_act, self.c_bump, self.w_head = c_act, c_bump, w_head
        self.P = None

    def reset(self, obs, start_info):
        super().reset(obs, start_info)
        self.P = np.tile(np.array([0.5, 0.25, 0.25]), (self.H, 1))

    def steer(self, plan, pos, yaw, obs):
        m = self.m
        pxz = pos[[0, 2]]
        _, err = pick_action(pxz, yaw, plan.path)
        plan.err = err
        r0, r1, c0, c1 = plan.win
        J = plan.field.dist
        Jc = np.where(np.isfinite(J), J, 50.0)
        hard = self.m._derived(True)[1] < self.m.radius - 0.02      # real footprint only (see nav_astar.steer)
        cxz = self.objxz
        # nearest cloud point per window cell is too heavy; use the nearest cloud point to the agent (stable, cheap)
        _, tgt = self._nearest(pxz, cxz)
        nominal = follow_actions(pxz, yaw, plan.path, lambda q: bool(m.lookup(hard, q[None], True)[0]), self.H)
        K, H = self.K, self.H
        P = np.vstack([self.P[1:], self.P[-1:]]) if self.P is not None else np.tile(np.array([0.5, 0.25, 0.25]), (H, 1))
        P = 0.5 * P + 0.5 * np.tile(np.array([0.5, 0.25, 0.25]), (H, 1))
        best = None
        for it in range(self.iters):
            u = self.rng.random_sample((K, H))
            cum = np.cumsum(P, 1)
            A = 1 + (u[:, :, None] > cum[None, :, :2]).sum(2)          # 1 fwd, 2 left, 3 right
            A[0] = nominal
            cost = self._rollout(A, pxz, yaw, Jc, hard, (r0, c0), tgt)
            w = np.exp(-(cost - cost.min()) / self.lam)
            w /= w.sum()
            cnt = np.zeros((H, 3))
            for a in (1, 2, 3):
                cnt[:, a - 1] = ((A == a) * w[:, None]).sum(0)
            P = 0.7 * cnt + 0.3 * P
            P = np.clip(P, 0.02, None)
            P /= P.sum(1, keepdims=True)
            first = np.array([w[A[:, 0] == a].sum() for a in (1, 2, 3)])
            best = (A[int(cost.argmin())], first, cost.min(), cost[0])
        self.P = P
        a0 = int(np.argmax(first)) + 1
        if first[nominal[0] - 1] >= first.max() - 1e-9:
            a0 = nominal[0]
        if best[3] <= best[2] + 0.02:                                  # nominal essentially as good: keep it
            a0 = nominal[0]
        return a0

    def _rollout(self, A, pxz, yaw, Jc, hard, off, tgt):
        m = self.m
        K, H = A.shape
        x = np.tile(pxz, (K, 1))
        th = np.full(K, yaw)
        bump = np.zeros(K)
        best = np.full(K, np.inf)
        n_h = Jc.shape
        for t in range(H):
            a = A[:, t]
            fwd = a == FORWARD
            f = np.stack([-np.sin(th), -np.cos(th)], 1)
            dest = x + FWD_M * f
            mid = x + 0.5 * FWD_M * f
            blocked = self._blocked(hard, off, dest) | self._blocked(hard, off, mid)
            move = fwd & ~blocked
            bump += self.c_bump * (fwd & blocked)
            x = np.where(move[:, None], dest, x)
            th = th + np.where(a == LEFT, TURN, 0.0) - np.where(a == RIGHT, TURN, 0.0)
            j = self._lookup(Jc, off, x)
            d = np.linalg.norm(tgt - x, axis=1)
            herr = np.abs(wrap_arr(np.arctan2(-(tgt - x)[:, 0], -(tgt - x)[:, 1]) - th))
            head = np.where(d < 2.5, self.w_head * herr, 0.0)
            tot = j + self.c_act * (t + 1) + head + bump
            best = np.minimum(best, tot)
        return best

    def _idx(self, off, xz):
        r, c = self.m.cell(xz)
        return r - off[0], c - off[1]

    def _blocked(self, hard, off, xz):
        ra, ca = self.m.cell(xz)
        ok = self.m.inside(ra, ca)
        out = np.ones(len(ra), bool)
        out[ok] = hard[ra[ok], ca[ok]]
        return out

    def _lookup(self, Jc, off, xz):
        r, c = self._idx(off, xz)
        ok = (r >= 0) & (r < Jc.shape[0]) & (c >= 0) & (c < Jc.shape[1])
        out = np.full(len(r), 50.0)
        out[ok] = Jc[r[ok], c[ok]]
        return out


def wrap_arr(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def make():
    return MPPINavigator(K=env("LM_K", 96, int), H=env("LM_H", 10, int), iters=env("LM_ITERS", 3, int),
                         lam=env("LM_LAMBDA", 0.4), c_bump=env("LM_BUMP", 3.0), w_head=env("LM_HEAD", 0.25),
                         **core_kwargs())
