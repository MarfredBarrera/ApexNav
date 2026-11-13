"""A* last-mile navigator: geodesic cost-to-go on the collision-aware local map (lastmile.localmap), path following
with discrete actions (turn to the next waypoint within +-15 deg, then forward), replanned every step.
The search is Dijkstra/A* with a zero heuristic (uniform cost, multi-source from the goal region, which is the
same optimum as A* and also yields the cost-to-go field MPPI needs)."""
import heapq
import math

import numpy as np

from lastmile.localmap import Field, FWD_S, LEFT15, RIGHT15, FORWARD, LEFT, RIGHT, TURN, FWD_M, LastMileCore, core_kwargs, env, fwd_vec, wrap, yaw_to


def follow_actions(xz, yaw, path, hard_fn, horizon, look=0.4, tol=math.radians(15)):
    """Simulate the path follower for `horizon` steps; returns the action list (used as MPPI nominal)."""
    acts = []
    xz = np.array(xz, float)
    for _ in range(horizon):
        a, _ = pick_action(xz, yaw, path, look, tol)
        if a == FORWARD:
            nxt = xz + FWD_M * fwd_vec(yaw)
            if hard_fn(nxt):
                a = LEFT
            else:
                xz = nxt
        if a == LEFT:
            yaw += TURN
        elif a == RIGHT:
            yaw -= TURN
        acts.append(a)
    return acts


def pick_action(xz, yaw, path, look=0.4, tol=math.radians(15)):
    """(action, signed heading error) toward the waypoint ~look metres ahead along the path."""
    if len(path) < 2:
        return FORWARD, 0.0
    d = np.linalg.norm(path - xz, axis=1)
    i0 = int(d.argmin())
    seg = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(path[i0:], axis=0), axis=1))]
    j = i0 + int(np.searchsorted(seg, look))
    wp = path[min(j, len(path) - 1)]
    if np.linalg.norm(wp - xz) < 1e-3:
        wp = path[-1]
    err = wrap(yaw_to(wp - xz) - yaw)
    if err > tol:
        return LEFT, err
    if err < -tol:
        return RIGHT, err
    return FORWARD, err


class AStarNavigator(LastMileCore):
    name = "astar"

    def __init__(self, look=0.4, **kw):
        super().__init__(**kw)
        self.look = look

    def lattice(self, plan, pos, yaw, rmin=0.17, max_nodes=3000):
        """State-lattice A* over the real action set (fwd 0.25 / 0.10, turn 30 / 15 deg) on the footprint map.
        Heuristic = cost-to-go of a near-raw field (inflation 0.12), so a 1-heading doorway is found even when
        the inflated map has no gap. Returns (first action, path of (x, z, yaw)) or (None, None)."""
        m = self.m
        pxz = pos[[0, 2]]
        cost3, _ = m.costs(pxz, 3, with_patch=True)
        fld = Field(cost3, [plan.goal_rc], plan.win, m.res, plan.arc)
        if not fld.valid or not np.isfinite(fld.at(plan.arc)):
            return None, None
        dist = m._derived(True)[1]
        r0, c0 = fld.win[0], fld.win[2]
        J = np.where(np.isfinite(fld.dist), fld.dist, 50.0)
        n = m.n
        small = self.small_m

        def hval(x, z):
            r = int((z - m.o[1]) / m.res) - r0
            c = int((x - m.o[0]) / m.res) - c0
            if 0 <= r < J.shape[0] and 0 <= c < J.shape[1]:
                return J[r, c] / FWD_M
            return 50.0

        def free(x, z):
            r = int((z - m.o[1]) / m.res)
            c = int((x - m.o[0]) / m.res)
            return 0 <= r < n and 0 <= c < n and dist[r, c] >= rmin

        K = 24
        k0 = int(round(yaw / (math.pi / 12)))
        acts = [(FORWARD, FWD_M, 0), (FWD_S, small, 0), (LEFT, 0, 2), (RIGHT, 0, -2), (LEFT15, 0, 1), (RIGHT15, 0, -1)]
        start = (pxz[0], pxz[1], k0)
        key = lambda s: (int(round(s[0] / 0.025)), int(round(s[1] / 0.025)), s[2] % K)
        h0 = hval(pxz[0], pxz[1])
        openl = [(h0, 0.0, 0, start, None, None)]
        seen = {key(start): 0.0}
        tie, best, expanded = 0, (h0, 0.0, start, None, None), 0
        parent = {}
        goal_h = 0.15 / FWD_M
        while openl and expanded < max_nodes:
            f, g, _, st, first, par = heapq.heappop(openl)
            expanded += 1
            h = f - g
            if h < best[0] - 1e-9 or (abs(h - best[0]) < 1e-9 and g < best[1]):
                best = (h, g, st, first, par)
            if h <= goal_h:
                best = (h, g, st, first, par)
                break
            x, z, k = st
            yw = k * math.pi / 12
            fv = np.array([-math.sin(yw), -math.cos(yw)])
            for a, d, dk in acts:
                if d:
                    nx, nz = x + d * fv[0], z + d * fv[1]
                    steps = max(int(d / 0.05), 1)
                    if not all(free(x + (nx - x) * t / steps, z + (nz - z) * t / steps) for t in range(1, steps + 1)):
                        continue
                    ns = (nx, nz, k)
                else:
                    ns = (x, z, k + dk)
                ng = g + (1.6 if a == FWD_S else 1.0)
                if d:
                    r_ = int((ns[1] - m.o[1]) / m.res); c_ = int((ns[0] - m.o[0]) / m.res)
                    ng += 1.5 * max(0.0, 0.30 - dist[r_, c_]) / 0.1      # prefer the middle of a gap
                ky = key(ns)
                if seen.get(ky, 1e9) <= ng:
                    continue
                seen[ky] = ng
                tie += 1
                heapq.heappush(openl, (ng + hval(ns[0], ns[1]) + 0.01 * (abs(dk) > 0), ng, tie, ns, first if first is not None else a, st))
        first = best[3]
        if first is None:
            return None, None
        return first, best[2]

    def steer_fine(self, plan, pos, yaw):
        a, end = self.lattice(plan, pos, yaw)
        _, err = pick_action(pos[[0, 2]], yaw, plan.path, self.look)
        plan.err = err
        if a is not None:
            return a
        return self.steer_fine_old(plan, pos, yaw)

    def steer_fine_old(self, plan, pos, yaw):
        """Fine actions: 15 deg turns (30 deg when far off), 0.10 m steps near obstacles, 0.25 m steps in the open."""
        m = self.m
        pxz = pos[[0, 2]]
        _, err = pick_action(pxz, yaw, plan.path, self.look)
        plan.err = err
        tol = math.radians(8)
        if abs(err) > tol:
            big = abs(err) > math.radians(22)
            if big:
                return LEFT if err > 0 else RIGHT
            return LEFT15 if err > 0 else RIGHT15
        dist = m._derived(True)[1]
        f = fwd_vec(yaw)
        ahead = np.array([pxz + t * f for t in (0.1, 0.2, 0.3)])
        clear = m.lookup(dist, ahead, 0.0)
        patch_hit = plan.patch and m.lookup(m.occ(True) & ~m.occ(False), ahead[:2], False).any()
        if patch_hit:
            return self._side15(err)
        tight = clear.min() < 0.30 or self.coll_run > 0 or (self.last and self.last[0] == FWD_S and False)
        return FWD_S if tight else FORWARD

    def _side15(self, err):
        if self.last and self.last[0] in (LEFT15, RIGHT15):
            return self.last[0]                                   # keep turning the same way
        return LEFT15 if err >= 0 else RIGHT15

    def steer(self, plan, pos, yaw, obs):
        m = self.m
        pxz = pos[[0, 2]]
        path = plan.path
        if self.fine:
            return self.steer_fine(plan, pos, yaw)
        a, err = pick_action(pxz, yaw, path, self.look)
        plan.err = err
        if a == FORWARD:
            # only a hit on the real footprint blocks the step: a path through a narrow gap may sit inside the
            # soft inflation, and refusing it makes the agent dither between turn left / turn right
            dist = m._derived(True)[1]
            ahead = [pxz + t * fwd_vec(yaw) for t in (0.125, FWD_M)]
            if any(m.lookup(dist, p[None], 0.0)[0] < m.radius - 0.02 for p in ahead):
                a = LEFT if err >= 0 else RIGHT
        self._last_turn = a if a in (LEFT, RIGHT) else self._last_turn
        return a

    def _side(self, err):
        # keep turning the same way as the previous turn: alternating left/right dithers in a narrow gap
        if getattr(self, "_last_turn", None) in (LEFT, RIGHT) and self.last and self.last[0] == self._last_turn:
            return self._last_turn
        return LEFT if err >= 0 else RIGHT


def make():
    nav = AStarNavigator(look=env("LM_LOOK", 0.4), **core_kwargs())
    return nav
