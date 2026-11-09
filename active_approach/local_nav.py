"""Self-contained local navigation for the approach (no navmesh, no host map, no seed panorama).

Everything here is built from what any ObjectNav agent has: its own depth frames, odometry, the object
mask and collision feedback (a blocked ``move_forward`` does not move the agent; ``allow_sliding`` false).

    LocalMap        occupancy on a 10 cm grid from geom.HeightMap (5 cm, own depth only) + the object cloud +
                    collision marks: free (seen, nothing in 0.12-1.6 m), occupied, unknown
    plan(...)       A* from the agent to a goal set; unknown cells cost UNKNOWN_COST per cell (optimistic),
                    occupied cells inflated by the agent radius + margin are blocked
    LocalController viewpoint -> action: follow the A* path with lattice line-following (turn when the
                    path point LOOKAHEAD ahead is > 25 deg off, else step forward), replan every step;
                    a blocked step marks the cell ahead occupied. Same interface as policies.Controller.

The floor height is the agent's own (odometry y), not the scenario's.
"""
import heapq
import math

import numpy as np

from active_approach import geom as G
from active_approach import scene as S

CELL = 0.10
OBST_LO, OBST_HI = 0.12, 1.6          # heights (above the floor) that block the base
INFLATE = S.AGENT_RADIUS + 0.02       # obstacle inflation (m)
UNKNOWN_COST = float(__import__('os').environ.get('AA_UNKNOWN_COST', 1.0))   # per unknown cell, vs 1 free (3: long detours)
GOAL_RADIUS, FACE_TOLERANCE_DEG = 0.25, 15.0
LEASH_RATIO, LEASH_M = 2.0, 1.0       # a viewpoint whose path is longer than 2 x straight + 1 m is given up
VIEW_COST = float(__import__('os').environ.get('AA_VIEW_COST', 2.0))   # per step leaving the object > 30 deg off heading (swept .5/1/2/4)


class LocalMap:
    def __init__(self, hmap):
        self.hmap = hmap                                  # geom.HeightMap (5 cm) fed with every depth frame
        self.k = int(round(CELL / G.CELL))
        self.n = hmap.n // self.k
        self.origin = hmap.origin
        self.bumps = np.zeros((self.n, self.n), bool)     # cells a blocked forward step ran into
        self.object = np.zeros((self.n, self.n), bool)

    def cell(self, xz):
        ij = np.floor((np.asarray(xz, float) - self.origin) / CELL).astype(int)
        return tuple(np.clip(ij, 0, self.n - 1))

    def center(self, ij):
        return self.origin + (np.asarray(ij, float) + 0.5) * CELL

    def set_object(self, cloud):
        self.object[:] = False
        if len(cloud):
            ij = np.floor((cloud[:, [0, 2]] - self.origin) / CELL).astype(int)
            ok = (ij >= 0).all(1) & (ij < self.n).all(1)
            self.object[ij[ok, 0], ij[ok, 1]] = True

    def bump(self, position, yaw):
        """A forward step from ``position`` along ``yaw`` was blocked: the cell ahead is occupied."""
        f = G.forward(yaw)
        side = np.array([-f[1], f[0]])
        for t in (-0.1, 0.0, 0.1):
            ahead = np.asarray(position)[[0, 2]] + (S.AGENT_RADIUS + 0.12) * f + t * side
            self.bumps[self.cell(ahead)] = True

    def grids(self):
        """(occupied, free, unknown) on the 10 cm grid."""
        k, n = self.k, self.n
        top = self.hmap.top[: n * k, : n * k].reshape(n, k, n, k)
        seen = self.hmap.seen[: n * k, : n * k].reshape(n, k, n, k).any((1, 3))
        hit = ((top > OBST_LO) & (top < OBST_HI)).reshape(n, k, n, k)
        occupied = (hit.sum((1, 3)) >= 2) | self.bumps | self.object
        free = seen & ~occupied
        return occupied, free, ~seen & ~occupied

    def cost_grid(self, position_xz):
        from scipy.ndimage import binary_dilation
        occupied, free, unknown = self.grids()
        r = int(math.ceil(INFLATE / CELL))
        disk = np.hypot(*np.mgrid[-r:r + 1, -r:r + 1]) <= r + 1e-6
        blocked = binary_dilation(occupied, structure=disk)
        cost = np.where(unknown, UNKNOWN_COST, 1.0)
        cost[blocked] = np.inf
        here = self.cell(position_xz)                     # the agent's own cells: it stands there
        sl = (slice(max(0, here[0] - 1), here[0] + 2), slice(max(0, here[1] - 1), here[1] + 2))
        cost[sl] = np.where(np.isfinite(cost[sl]), cost[sl], 5.0)
        return cost

    def los_free(self, a_xz, b_xz):
        """No occupied (un-inflated) cell on the segment a -> b (the object's own cells excluded)."""
        occupied, _, _ = self.grids()
        occupied &= ~self.object
        n = max(2, int(np.linalg.norm(np.asarray(b_xz) - a_xz) / (CELL / 2)))
        for t in np.linspace(0, 1, n):
            if occupied[self.cell(np.asarray(a_xz) + t * (np.asarray(b_xz) - a_xz))]:
                return False
        return True


def plan(cost, start, goals):
    """A* on ``cost`` (inf blocked) from cell ``start`` to any cell of ``goals``; list of cells or None."""
    goals = set(map(tuple, goals))
    goals = {g for g in goals if np.isfinite(cost[g])}
    if not goals:
        return None
    garr = np.array(list(goals))

    def h(c):
        return float(np.min(np.hypot(garr[:, 0] - c[0], garr[:, 1] - c[1])))

    n0, n1 = cost.shape
    best = {start: 0.0}
    parent = {start: None}
    heap = [(h(start), 0.0, start)]
    moves = [(1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
             (1, 1, 1.414), (1, -1, 1.414), (-1, 1, 1.414), (-1, -1, 1.414)]
    while heap:
        _, g, c = heapq.heappop(heap)
        if c in goals:
            path = []
            while c is not None:
                path.append(c)
                c = parent[c]
            return path[::-1]
        if g > best.get(c, math.inf):
            continue
        for di, dj, w in moves:
            m = (c[0] + di, c[1] + dj)
            if not (0 <= m[0] < n0 and 0 <= m[1] < n1):
                continue
            step = w * cost[m]
            if not np.isfinite(step):
                continue
            ng = g + step
            if ng < best.get(m, math.inf):
                best[m], parent[m] = ng, c
                heapq.heappush(heap, (ng + h(m), ng, m))
    return None


def path_length(path):
    if not path or len(path) < 2:
        return 0.0
    p = np.asarray(path, float)
    return float(np.hypot(*np.diff(p, axis=0).T).sum() * CELL)


class LocalController:
    """policies.Controller without the navmesh. ``action(viewpoint, obs)``: next action, None when within
    GOAL_RADIUS of the viewpoint and facing its yaw. Raises RuntimeError('no path') when the goal is
    unreachable on the local map (the runner then forces the decision)."""
    LOOKAHEAD, HYSTERESIS = 0.5, math.radians(25)

    def __init__(self, sim, lmap, pathfinder=None):
        self.sim, self.lmap, self.pf = sim, lmap, pathfinder
        self.last = None          # (position, yaw, action) of the previous call, for bump detection
        self.fails = 0
        self.vp_key, self.vp_spent, self.vp_budget = None, 0, None   # the leash on the current viewpoint
        self.gave_up, self.last_error = None, None
        self.center = None

    def _face(self, obs, position, yaw):
        if obs is None or not len(obs.cloud):
            return None
        diff = G.wrap(G.bearing_yaw(np.asarray(position)[[0, 2]], obs.cloud[:, [0, 2]].mean(0)) - yaw)
        if abs(math.degrees(diff)) > FACE_TOLERANCE_DEG:
            return "turn_left" if diff > 0 else "turn_right"
        return None

    def _lattice(self, here, yaw, end):
        """Lattice lookahead: for each of the 12 headings yaw + k 30 deg, cost = turns + forward steps (each
        swept free) + cost-to-go (Dijkstra from ``end``, in 0.25 m steps) at the best point along up to 6 steps.
        The current heading wins ties, so a committed turn is never undone (no L/R dithering)."""
        lm, pf = self.lmap, self.pf
        n = lm.n
        dist, _ = pf._tree(end)
        togo = lambda xz: dist[lm.cell(xz)[0] * n + lm.cell(xz)[1]] * CELL / S.FORWARD_M
        best = (math.inf, 0, None)                # staying put is not an option while the goal is away
        for k in range(-6, 7):
            if k == -6:
                continue
            turns = abs(k)
            h = yaw + k * G.LATTICE
            f = G.forward(h)
            p = here
            for j in range(1, 7):
                mid, nxt = p + 0.5 * S.FORWARD_M * f, p + S.FORWARD_M * f
                if not (np.isfinite(pf.cost[lm.cell(mid)]) and np.isfinite(pf.cost[lm.cell(nxt)])):
                    break
                p = nxt
                c = turns + j + togo(p) + 1e-3 * turns
                if self.center is not None and VIEW_COST:   # keep the object in view while driving
                    off = abs(G.wrap(G.bearing_yaw(p, self.center) - h))
                    c += VIEW_COST * j * (off > math.radians(30))
                if c < best[0] - 1e-6:
                    best = (c, k, j)
        if best[2] is None:
            return None
        k = best[1]
        if k == 0:
            return "move_forward"
        return "turn_left" if k > 0 else "turn_right"

    def _bump_check(self, position):
        if self.last is not None and self.last[2] == "move_forward":
            moved = float(np.hypot(*(position - self.last[0])[[0, 2]]))
            if moved < 0.5 * S.FORWARD_M:
                self.lmap.bump(self.last[0], self.last[1])

    def action(self, viewpoint, obs=None):
        """A viewpoint that is unreachable or too costly is given up: the agent turns to face the object
        (tracked cloud centre) where it stands and arrives there (the policy then moves on)."""
        position, yaw, _ = S.agent_pose(self.sim)
        self._bump_check(position)
        self.center = obs.cloud[:, [0, 2]].mean(0) if obs is not None and len(obs.cloud) else None
        if self.gave_up is not None and np.hypot(*(np.asarray(viewpoint.position)[[0, 2]] - self.gave_up)) < 0.3:
            act = self._face(obs, position, yaw)
        else:
            self.gave_up = None
            try:
                act = self._action(viewpoint, position, yaw)
            except RuntimeError as error:
                self.gave_up = np.asarray(viewpoint.position)[[0, 2]]
                self.last_error = str(error)
                act = self._face(obs, position, yaw)
        self.last = (position.copy(), yaw, act)
        return act

    def _action(self, viewpoint, position, yaw):
        if viewpoint.plan:
            return viewpoint.plan.pop(0)
        here = np.asarray(position)[[0, 2]]
        goal = np.asarray(viewpoint.position)[[0, 2]]
        if np.hypot(*(goal - here)) > GOAL_RADIUS:
            lm = self.lmap
            pf = self.pf
            pf.refresh(position)
            cells = pf.path_cells(here, goal)
            if cells is None:
                occupied, free, unknown = lm.grids()
                self.debug = {"occupied": occupied, "free": free, "unknown": unknown, "cost": pf.cost,
                              "here": lm.cell(here), "goal": lm.cell(goal), "object": lm.object.copy(),
                              "bumps": lm.bumps.copy()}
                raise RuntimeError("no path on the local map")
            end = cells[-1]
            snapped = np.hypot(*(lm.center(end) - goal)) > GOAL_RADIUS
            if snapped and np.hypot(*(goal - here)) <= 0.6:
                cells = [lm.cell(here)]          # the goal is unreachable and we are as close as the snap allows
                end = cells[0]
            length = path_length(cells)
            straight = float(np.hypot(*(goal - here)))
            key = tuple(np.round(goal, 1))
            if self.vp_key is None or np.hypot(*(np.asarray(key) - self.vp_key)) > 0.3:
                self.vp_key, self.vp_spent = np.asarray(key), 0
                self.vp_budget = 2 * (length / S.FORWARD_M + 3) + 4
            self.vp_spent += 1
            if length > LEASH_RATIO * straight + LEASH_M or self.vp_spent > self.vp_budget:
                self.vp_key = None
                raise RuntimeError(f"viewpoint too costly (path {length:.1f} m, straight {straight:.1f} m, "
                                   f"spent {self.vp_spent})")
            if len(cells) > 1 and np.hypot(*(lm.center(end) - here)) > GOAL_RADIUS * 0.6:
                act = self._lattice(here, yaw, end)
                if act is not None:
                    return act
        diff = G.wrap(viewpoint.yaw - yaw)
        if abs(math.degrees(diff)) > FACE_TOLERANCE_DEG:
            return "turn_left" if diff > 0 else "turn_right"
        return None


class _Path:
    """habitat_sim.ShortestPath stand-in (its ``points`` cannot be set from Python)."""

    def __init__(self):
        self.requested_start = self.requested_end = None
        self.points, self.geodesic_distance = [], math.inf


class LocalPathfinder:
    """The habitat PathFinder calls the planners make (snap_point, find_path, is_navigable,
    try_step_no_sliding, distance_to_closest_obstacle), answered from the self-built LocalMap only, so the
    navmesh-era planners run unchanged without the navmesh. ``refresh(position)`` once per step (the map
    grows every frame). Unknown cells are navigable (optimistic: the controller re-plans as they are seen)."""
    ShortestPath = _Path

    def __init__(self, lmap, floor_y):
        self.lmap, self.floor_y = lmap, floor_y
        self.cost = None
        self._dist = None

    def refresh(self, position):
        self.cost = self.lmap.cost_grid(np.asarray(position)[[0, 2]])
        self._dist, self._trees, self._graph = None, {}, None

    def _tree(self, start):
        """Dijkstra from cell ``start`` over the whole grid (scipy, cached per start until refresh)."""
        if start not in self._trees:
            from scipy.sparse import coo_matrix
            from scipy.sparse.csgraph import dijkstra
            n = self.lmap.n
            if self._graph is None:
                idx = np.arange(n * n).reshape(n, n)
                rows, cols, w = [], [], []
                for di, dj, length in ((0, 1, 1.0), (1, 0, 1.0), (1, 1, 1.414), (1, -1, 1.414)):
                    a = idx[max(0, -di): n - max(0, di), max(0, -dj): n - max(0, dj)]
                    b = idx[max(0, di): n - max(0, -di) or None, max(0, dj): n - max(0, -dj) or None]
                    ca = self.cost.ravel()[a.ravel()]
                    cb = self.cost.ravel()[b.ravel()]
                    ok = np.isfinite(ca) & np.isfinite(cb)
                    c = length * 0.5 * (ca + cb)
                    rows += [a.ravel()[ok], b.ravel()[ok]]
                    cols += [b.ravel()[ok], a.ravel()[ok]]
                    w += [c[ok], c[ok]]
                self._graph = coo_matrix((np.concatenate(w), (np.concatenate(rows), np.concatenate(cols))),
                                         shape=(n * n, n * n)).tocsr()
            dist, pred = dijkstra(self._graph, indices=start[0] * n + start[1], return_predecessors=True)
            if len(self._trees) > 8:
                self._trees.clear()
            self._trees[start] = (dist, pred)
        return self._trees[start]

    def path_cells(self, a_xz, b_xz, tol=0.6):
        """Cells of the cheapest path a -> b; when b is blocked or cut off, -> the reachable cell nearest b within
        ``tol`` (the map grows: a goal planned in unknown space may turn out to be inside furniture). None if
        nothing reachable is that close."""
        n = self.lmap.n
        start = self.lmap.cell(a_xz)
        goal = self.lmap.cell(b_xz)
        dist, pred = self._tree(start)
        g = goal[0] * n + goal[1]
        if not np.isfinite(dist[g]):
            r = int(math.ceil(tol / CELL))
            i0, j0 = goal
            ii, jj = np.mgrid[max(0, i0 - r): min(n, i0 + r + 1), max(0, j0 - r): min(n, j0 + r + 1)]
            d2 = (ii - i0) ** 2 + (jj - j0) ** 2
            ok = np.isfinite(dist[(ii * n + jj).ravel()]).reshape(ii.shape) & (d2 <= r * r)
            if not ok.any():
                return None
            k = np.argmin(np.where(ok, d2, np.inf))
            g = int(ii.ravel()[k] * n + jj.ravel()[k])
        cells = [g]
        while cells[-1] != start[0] * n + start[1] and pred[cells[-1]] >= 0:
            cells.append(pred[cells[-1]])
        return [(c // n, c % n) for c in cells[::-1]]

    def _ok(self, ij):
        return bool(np.isfinite(self.cost[ij]))

    def is_navigable(self, point, max_y_delta=0.5):
        return self._ok(self.lmap.cell(np.asarray(point)[[0, 2]]))

    def snap_point(self, point, tol=0.3):
        """The point itself when its cell is not blocked (unknown counts as open), else the nearest open cell
        within ``tol``; NaN when none."""
        p = np.asarray(point, float)
        ij = self.lmap.cell(p[[0, 2]])
        if self._ok(ij):
            return np.array([p[0], self.floor_y, p[2]], np.float32)
        n, r = self.lmap.n, int(math.ceil(tol / CELL))
        ii, jj = np.mgrid[max(0, ij[0] - r): min(n, ij[0] + r + 1), max(0, ij[1] - r): min(n, ij[1] + r + 1)]
        d2 = (ii - ij[0]) ** 2 + (jj - ij[1]) ** 2
        ok = np.isfinite(self.cost[ii, jj]) & (d2 <= r * r)
        if not ok.any():
            return np.array([np.nan, np.nan, np.nan], np.float32)
        k = np.argmin(np.where(ok, d2, np.inf))
        x, z = self.lmap.center((ii.ravel()[k], jj.ravel()[k]))
        return np.array([x, self.floor_y, z], np.float32)

    def find_path(self, path):
        a = np.asarray(path.requested_start, float)
        b = np.asarray(path.requested_end, float)
        cells = self.path_cells(a[[0, 2]], b[[0, 2]])
        if cells is None:
            path.points, path.geodesic_distance = [], math.inf
            return False
        mid = [self.lmap.center(c) for c in cells[1:-1]]
        pts = [a] + [np.array([m[0], self.floor_y, m[1]]) for m in mid] + [b]
        path.points = pts
        path.geodesic_distance = float(sum(np.hypot(*(q - p)[[0, 2]]) for p, q in zip(pts, pts[1:])))
        return True

    def nav_grid(self, hmap):
        """Navigable cells on the 5 cm HeightMap grid (svpto.navigable_grid)."""
        k = self.lmap.k
        nav = np.repeat(np.repeat(np.isfinite(self.cost), k, 0), k, 1)
        out = np.zeros((hmap.n, hmap.n), bool)
        m = min(hmap.n, nav.shape[0])
        out[:m, :m] = nav[:m, :m]
        return out

    def try_step_no_sliding(self, a, b):
        return np.asarray(b if self.is_navigable(b) else a, np.float32)

    def distance_to_closest_obstacle(self, point, max_search=1.0):
        if self._dist is None:
            from scipy.ndimage import distance_transform_edt
            occupied, _, _ = self.lmap.grids()
            self._dist = distance_transform_edt(~occupied) * CELL
        return float(min(self._dist[self.lmap.cell(np.asarray(point)[[0, 2]])], max_search))


def track_mask(cloud, depth, cam_pos, cam_rot, tol=0.12, grow=6):
    """GT-free tracker (ablation T1): the fused object cloud splatted into the frame with a z-buffer, grown by
    ``grow`` px, kept where the measured depth agrees with the splat depth within ``tol`` m. Empty when the
    cloud is out of view or hidden."""
    import cv2
    mask = np.zeros(depth.shape, bool)
    if not len(cloud):
        return mask
    u, v, z = S.project(cloud, cam_pos, cam_rot)
    ok = (z > 0.1) & (u >= 0) & (u < S.WIDTH) & (v >= 0) & (v < S.HEIGHT)
    if not ok.any():
        return mask
    zbuf = np.full(depth.shape, np.inf, np.float32)
    np.minimum.at(zbuf, (v[ok].astype(int), u[ok].astype(int)), z[ok].astype(np.float32))
    k = np.ones((2 * grow + 1, 2 * grow + 1), np.uint8)
    near = cv2.erode(np.where(np.isfinite(zbuf), zbuf, 1e3).astype(np.float32), k)   # min splat depth nearby
    far = cv2.dilate(np.where(np.isfinite(zbuf), zbuf, -1e3).astype(np.float32), k)   # max splat depth nearby
    covered = near < 1e3
    d = np.asarray(depth, np.float32)
    return covered & (d > near - tol) & (d < far + tol)
