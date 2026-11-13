"""Port of ApexNav's object-goal A* (src/planner/path_searching/src/astar2d.cpp, ExplorationManager::searchObjectPath /
trySearchObjectPathWithDistance / searchObjectPathExtreme) on a 2D grid, for the live last-mile navigator's waypoints.

Frame: whatever the Grid is in (LASTMILE_APEXWP_MAP=own: sim world (x, z); apex: planner frame). Paths come back in that frame.
The search is the C++ one: 12 headings x 0.25 m steps from continuous positions, a 0.1 m closed-set index, a 0.025 m ray check
per step (skipped within 0.25 m of the start), cost g + lambda_heu * diagonal heuristic, success = within success_dist (or a
neighbouring index cell) of the goal. Safety: NORMAL blocks inflated / occupied / unknown cells and anything outside the map,
OPTIMISTIC allows unknown, EXTREME allows anything inside the map. The only deviation: the 0.2 s budget is wall time here
(plus an expansion cap), not the C++ expansion budget.
"""
import heapq
import math
import os
import time

import numpy as np

NORMAL, OPTIMISTIC, EXTREME = 0, 1, 2
LADDER = (0.5, 0.70, 0.85)          # searchObjectPath success distances
LAMBDA_HEU = 1.0                    # astar.lambda_heu (algorithm.launch.py)
ASTAR_RES = 0.1                     # astar.resolution_astar: the closed-set index cell
STEP, N_DIR = 0.25, 12              # Astar2D::generateSteps
_ANG = np.arange(N_DIR) * (math.pi / 6)
STEPS = STEP * np.stack([np.cos(_ANG), np.sin(_ANG)], 1)
_L = np.arange(0.025, STEP, 0.025)  # ray samples along a step (C++ `for l = 0.025; l < len; l += 0.025`)
SAMPLES = (STEPS / STEP)[:, None, :] * _L[None, :, None]   # (12, k, 2) offsets from the current position


class Grid:
    """occ / infl / known boolean arrays [row, col] (col = first coordinate), cell (0, 0) corner at origin."""

    def __init__(self, origin, res, occ, infl, known):
        self.origin, self.res = np.asarray(origin, float), float(res)
        self.occ, self.infl, self.known = occ, infl, known
        self.shape = occ.shape
        self._safe = {}

    @classmethod
    def from_localmap(cls, lm, radius=None):
        """The navigator's own LocalMap (sim world x, z): occupied = hits (+ collision patches), inflated = within `radius`
        (default ApexNav's sdf_map.obstacles_inflation 0.18 = the LocalMap radius), known = free ray cells or occupied."""
        o, dist = lm._derived(True)
        r = lm.radius if radius is None else radius
        return cls(lm.o, lm.res, o, (dist < r) & ~o, lm.free | o)

    def index(self, pts):
        c = np.floor((np.asarray(pts, float) - self.origin) / self.res).astype(int)
        return c[..., 1], c[..., 0]

    def inside(self, r, c):
        return (r >= 0) & (r < self.shape[0]) & (c >= 0) & (c < self.shape[1])

    def safe_mask(self, mode):
        if mode not in self._safe:
            if mode == EXTREME:
                m = np.ones(self.shape, bool)
            elif mode == OPTIMISTIC:
                m = ~(self.infl | self.occ)
            else:
                m = ~(self.infl | self.occ) & self.known
            self._safe[mode] = m
        return self._safe[mode]

    def safe(self, pts, mode):
        """checkPointSafety for an array of points (outside the map = unsafe)."""
        r, c = self.index(pts)
        ok = self.inside(r, c)
        out = np.zeros(r.shape, bool)
        out[ok] = self.safe_mask(mode)[r[ok], c[ok]]
        return out

    def state(self, pt):
        """(occupied, unknown, inflated) at a point."""
        r, c = self.index(pt)
        if not self.inside(r, c):
            return True, True, True
        return bool(self.occ[r, c]), not bool(self.known[r, c]), bool(self.infl[r, c])


def diag_heu(a, b):
    dx, dy = abs(a[0] - b[0]), abs(a[1] - b[1])
    return (1.0 + 1e-6 * (dx + dy)) * (math.sqrt(2.0) * min(dx, dy) + abs(dx - dy))


def astar(grid, start, end, success_dist, max_time=0.2, mode=NORMAL, max_expand=30000, lam=LAMBDA_HEU):
    """Astar2D::astarSearchImpl. Returns the path [start ... end] (list of (x, y)) or None."""
    start, end = np.asarray(start, float), np.asarray(end, float)
    org = grid.origin

    def key(p):
        return (int(math.floor((p[0] - org[0]) / ASTAR_RES)), int(math.floor((p[1] - org[1]) / ASTAR_RES)))

    end_k = key(end)
    pos = [start]
    parent, g = [-1], [0.0]
    heap = [(lam * diag_heu(start, end), 0, 0)]
    open_map, closed, cnt = {key(start): 0}, set(), 1
    t0, it = time.monotonic(), 0
    while heap:
        _, _, i = heap[0]
        p = pos[i]
        k = key(p)
        if k in closed:                      # stale duplicate of an improved node
            heapq.heappop(heap)
            continue
        if (abs(k[0] - end_k[0]) <= 1 and abs(k[1] - end_k[1]) <= 1) or math.hypot(p[0] - end[0], p[1] - end[1]) < success_dist:
            out = [end, p]
            while parent[i] >= 0:
                i = parent[i]
                out.append(pos[i])
            return out[::-1]
        if it >= max_expand or time.monotonic() - t0 > max_time:
            return None
        heapq.heappop(heap)
        open_map.pop(k, None)
        closed.add(k)
        it += 1
        nbr = p + STEPS                                         # (12, 2)
        far = np.hypot(*(nbr - start).T) > 0.25                 # near the start the safety check is skipped
        ok = np.ones(N_DIR, bool)
        if far.any():
            chk = np.concatenate([nbr[:, None, :], p + SAMPLES], 1)       # (12, 1+k, 2): the end point, then the ray
            ok = np.where(far, grid.safe(chk.reshape(-1, 2), mode).reshape(N_DIR, -1).all(1), True)
        for d in np.nonzero(ok)[0]:
            q = nbr[d]
            qk = key(q)
            if qk in closed:
                continue
            tg = g[i] + STEP
            j = open_map.get(qk)
            if j is None:
                pos.append(q)
                parent.append(i)
                g.append(tg)
                j = len(pos) - 1
                open_map[qk] = j
            elif tg < g[j]:
                parent[j], g[j] = i, tg
            else:
                continue
            heapq.heappush(heap, (tg + lam * diag_heu(q, end), cnt, j))
            cnt += 1
    return None


def path_length(path):
    p = np.asarray(path, float)
    return float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum()) if len(p) > 1 else 0.0


def try_with_distance(grid, start, obj, dist, max_time=0.2, **kw):
    """trySearchObjectPathWithDistance: A* to the object within `dist`, walk back to the last cell that is not occupied,
    not unknown and not inflated, A* to it within 0.2. Returns (goal, path) or None."""
    path = astar(grid, start, obj, dist, max_time, NORMAL, **kw)
    if path is None:
        return None
    tmp = None
    for p in reversed(path):
        occ, unk, inf = grid.state(p)
        if not occ and not unk and not inf:
            tmp = np.asarray(p, float)
            break
    if tmp is None:                                            # C++ then searches towards (-1000, -1000): never reachable
        return None
    path2 = astar(grid, start, tmp, 0.2, max_time, NORMAL, **kw)
    return None if path2 is None else (tmp, path2)


def nearest_object_point(start, cloud_xy):
    c = np.asarray(cloud_xy, float).reshape(-1, 2)
    c = c[np.isfinite(c).all(1)]
    if not len(c):
        return None
    return c[np.argmin(np.linalg.norm(c - np.asarray(start, float), axis=1))]


def plan_object_path(grid, start, cloud_xy, max_time=0.2, **kw):
    """searchObjectPath (normal safety, 0.5 / 0.70 / 0.85 m ladder) then searchObjectPathExtreme (extreme safety, success 0.25,
    the goal is the object point). Returns dict(mode, path, goal, obj) or None."""
    start = np.asarray(start, float)
    obj = nearest_object_point(start, cloud_xy)
    if obj is None:
        return None
    for d in LADDER:
        r = try_with_distance(grid, start, obj, d, max_time, **kw)
        if r is not None:
            return dict(mode="normal_%.2f" % d, path=np.asarray(r[1], float), goal=r[0], obj=obj)
    path = astar(grid, start, obj, 0.25, max_time, EXTREME, **kw)
    if path is not None:
        return dict(mode="extreme", path=np.asarray(path, float), goal=obj, obj=obj)
    return None


def waypoint(grid, path, agent, max_d=1.0, los=True):
    """The farthest point of the path within max_d metres of the agent along the path (path resampled every 0.1 m), stopping
    at the first point not in clear sight over the grid's occupied cells (when `los`)."""
    p = np.asarray(path, float)
    if len(p) < 2:
        return p[-1]
    seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    ss = np.arange(0.0, min(s[-1], max_d) + 1e-9, 0.1)
    pts = np.stack([np.interp(ss, s, p[:, 0]), np.interp(ss, s, p[:, 1])], 1)
    if s[-1] <= max_d:
        pts = np.vstack([pts, p[-1]])
    best = pts[0]
    a = np.asarray(agent, float)
    for q in pts[1:]:
        if los and not clear_sight(grid, a, q):
            break
        best = q
    return best


def clear_sight(grid, a, b):
    n = int(max(np.linalg.norm(b - a) / 0.025, 2)) + 1
    t = np.linspace(0.0, 1.0, n)[:, None]
    r, c = grid.index(a + t * (b - a))
    m = grid.inside(r, c)
    return not bool(grid.occ[r[m], c[m]].any())


def enabled():
    return bool(int(os.environ.get("LASTMILE_APEXWP", 0)))


def cell_class(grid, p):
    """'occ' / 'infl' / 'unk' / 'free' / 'out' at a point."""
    r, c = grid.index(p)
    if not grid.inside(r, c):
        return "out"
    return "occ" if grid.occ[r, c] else "infl" if grid.infl[r, c] else "free" if grid.known[r, c] else "unk"


def cell_fractions(grid, pts):
    r, c = grid.index(np.asarray(pts, float).reshape(-1, 2))
    ok = grid.inside(r, c)
    n = max(len(r), 1)
    o, i_, k = np.zeros(len(r), bool), np.zeros(len(r), bool), np.zeros(len(r), bool)
    o[ok], i_[ok], k[ok] = grid.occ[r[ok], c[ok]], grid.infl[r[ok], c[ok]], grid.known[r[ok], c[ok]]
    return dict(n=int(len(r)), occ=round(o.sum() / n, 2), infl=round((i_ & ~o).sum() / n, 2), free=round((k & ~o & ~i_).sum() / n, 2),
                unk=round((~k).sum() / n, 2))
