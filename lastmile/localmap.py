"""Collision-aware local map + shared last-mile logic for the A* and MPPI navigators.

LocalMap: 5 cm 2D grid built from depth + odometry (habitat frame: x right, y up, -z forward, yaw ccw about +y,
camera pitch about its x axis, negative = down).
  * obstacles: depth points whose height above the agent base is in [FLOOR_H, TOP_H] (no outlier filter, min hit
    count only), free space by ray-casting every depth pixel, collision patches (persistent) written when a
    FORWARD does not move the agent;
  * inflation by agent radius (+ margin) for the hard mask, clearance cost for the soft cost;
  * unknown cells are traversable with a cost penalty (optimistic);
  * grid line-of-sight, geodesic Dijkstra (scipy csgraph) on a cropped window.

LastMileCore (base of nav_astar / nav_mppi): collision bookkeeping, goal region + commitment, STOP rule,
limit-cycle detection and recovery. Subclasses implement steer(plan) -> action.

Env vars (read by from_env): LM_RADIUS, LM_MARGIN, LM_STOP_R, LM_GOAL_LO, LM_GOAL_HI, LM_UNKNOWN, LM_SEED, LM_DEBUG_DIR.
"""
import atexit
import math
import os

import numpy as np
from scipy import ndimage
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

STOP, FORWARD, LEFT, RIGHT = 0, 1, 2, 3
FWD_S, LEFT15, RIGHT15 = 4, 5, 6                      # bench --actions fine
FWD_M, TURN = 0.25, math.radians(30.0)
HFOV_DEG, WIDTH, HEIGHT = 79.0, 640, 480
INF = np.inf


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def fwd_vec(yaw):
    return np.array([-math.sin(yaw), -math.cos(yaw)])


def yaw_to(dxz):
    """Yaw that looks along the horizontal vector (dx, dz)."""
    return math.atan2(-dxz[0], -dxz[1])


def env(name, default, cast=float):
    v = os.environ.get(name)
    return default if v is None else cast(v)


class LocalMap:
    def __init__(self, center_xz, res=0.05, size_m=24.0, radius=0.18, margin=0.03, floor_h=0.06, top_h=1.6,
                 min_hits=2, max_range=5.0, stride=4, soft_d=0.5, soft_w=3.0, unknown_pen=1.0, cam_h=0.88):
        self.res, self.n = res, int(round(size_m / res))
        self.o = np.asarray(center_xz, float) - size_m / 2          # world (x, z) of cell (0, 0) corner
        self.radius, self.margin = radius, margin
        self.floor_h, self.top_h, self.min_hits, self.max_range, self.stride = floor_h, top_h, min_hits, max_range, stride
        self.soft_d, self.soft_w, self.unknown_pen, self.cam_h = soft_d, soft_w, unknown_pen, cam_h
        n = self.n
        self.hits = np.zeros((n, n), np.int16)
        self.free = np.zeros((n, n), bool)
        self.patch = np.zeros((n, n), bool)
        self.patch_t = np.zeros((n, n), np.int32)      # step at which a patch cell was written
        self.now, self.patch_ttl = 0, 10 ** 9
        self.visit_pen = np.zeros((n, n), np.float32)
        self.blacklist = np.zeros((n, n), bool)
        f = (WIDTH / 2) / math.tan(math.radians(HFOV_DEG) / 2)
        vv, uu = np.mgrid[0:HEIGHT:stride, 0:WIDTH:stride]
        self._uu, self._vv, self._f = uu.ravel().astype(np.float32), vv.ravel().astype(np.float32), f
        self._dirty = True
        self._cache = {}
        self.patch_inflate = True
        self.clear, self.clear_n = bool(env("LM_CLEAR", 0, int)), env("LM_CLEAR_N", 2, int)
        self.dbg = bool(env("LM_DEBUG_H", 0, int))       # per-cell hit-height range (diagnostics only)
        if self.dbg:
            self.hmin = np.full((n, n), 9.0, np.float32)
            self.hmax = np.full((n, n), -9.0, np.float32)

    # ------------------------------------------------------------ coordinates
    def cell(self, xz):
        c = np.floor((np.asarray(xz, float) - self.o) / self.res).astype(int)
        return c[..., 1], c[..., 0]                                   # (row = z, col = x)

    def world(self, r, c):
        return self.o + (np.stack([np.asarray(c), np.asarray(r)], -1) + 0.5) * self.res

    def inside(self, r, c):
        return (r >= 0) & (r < self.n) & (c >= 0) & (c < self.n)

    # ------------------------------------------------------------ sensing
    def update(self, pos, yaw, pitch, depth):
        d = np.asarray(depth, np.float32)
        d = d.reshape(d.shape[:2])[::self.stride, ::self.stride].ravel()
        ok = (d > 0.1) & (d < self.max_range) & np.isfinite(d)
        if not ok.any():
            return
        d, u, v = d[ok], self._uu[ok], self._vv[ok]
        loc = np.stack([(u - WIDTH / 2) * d / self._f, -(v - HEIGHT / 2) * d / self._f, -d], 1)
        cp, sp = math.cos(pitch), math.sin(pitch)
        loc = loc @ np.array([[1, 0, 0], [0, cp, sp], [0, -sp, cp]]).T          # Rx(pitch)
        cy, sy = math.cos(yaw), math.sin(yaw)
        Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
        w = loc @ Ry.T                                                          # camera frame -> world offsets
        h = w[:, 1] + self.cam_h                                                # height above agent base
        px, pz = pos[0] + w[:, 0], pos[2] + w[:, 2]
        band = (h >= self.floor_h) & (h <= self.top_h)
        r, c = self.cell(np.stack([px[band], pz[band]], 1))
        m = self.inside(r, c)
        np.add.at(self.hits, (r[m], c[m]), 1)
        band_cells = (r[m], c[m])
        if self.dbg:
            hb = h[band][m]
            np.minimum.at(self.hmin, (r[m], c[m]), hb)
            np.maximum.at(self.hmax, (r[m], c[m]), hb)
        # free space: rays from the camera to every (sub-sampled) valid pixel
        sub = slice(None, None, 2)
        x0 = np.array([pos[0], pos[2]])
        tgt = np.stack([px, pz], 1)[sub]
        L = np.linalg.norm(tgt - x0, axis=1)
        t = np.linspace(0.0, 1.0, 100)[None, :] * np.maximum(1 - 0.1 / np.maximum(L, 1e-3), 0)[:, None]
        pts = x0[None, None, :] + t[:, :, None] * (tgt - x0)[:, None, :]
        r, c = self.cell(pts.reshape(-1, 2))
        m = self.inside(r, c)
        self.free[r[m], c[m]] = True
        if self.clear:                       # a ray passing through a cell at 0.25-1.2 m height: one vote against its occupancy
            hs = (self.cam_h + t * (h[sub] - self.cam_h)[:, None]).reshape(-1)
            ok2 = m & (hs >= 0.25) & (hs <= 1.2)
            flat = np.unique(r[ok2] * self.n + c[ok2])
            keep_idx = band_cells[0] * self.n + band_cells[1]
            keep_val = self.hits.reshape(-1)[keep_idx].copy()
            if len(flat):
                self.hits.reshape(-1)[flat] = np.maximum(self.hits.reshape(-1)[flat] - self.clear_n, 0)
            self.hits.reshape(-1)[keep_idx] = keep_val        # cells hit in THIS frame keep their count
        self._dirty = True

    def mark_collision(self, pos, yaw, k=0, small=False, tiny=False):
        """Persistent occupied patch ahead of the agent after a blocked forward (grows with repeated blocks k).
        Habitat's no-sliding step moves up to the obstacle, so the surface is ~0.18 m ahead of the centre."""
        f = fwd_vec(yaw)
        l = np.array([f[1], -f[0]])
        d0, d1, hw = (0.14, 0.22, 0.10) if small else (0.18, 0.28, 0.15)
        if tiny:                                   # contact point only: ~0.1 m wide, 0.04 m deep, no growth with k
            d0, d1, hw, k = 0.17, 0.21, 0.05, 0
        ds = np.arange(d0, d1 + 0.12 * k + 1e-6, 0.03)
        ws = np.arange(-hw - 0.04 * k, hw + 0.04 * k + 1e-6, 0.03)
        pts = np.array([pos[0], pos[2]])[None, None] + ds[:, None, None] * f + ws[None, :, None] * l
        r, c = self.cell(pts.reshape(-1, 2))
        m = self.inside(r, c)
        self.patch[r[m], c[m]] = True
        self.patch_t[r[m], c[m]] = self.now
        self._dirty = True

    def add_visit_penalty(self, xz, rad=0.4, amount=2.0):
        r, c = self.cell(xz)
        rr = int(rad / self.res)
        sl = (slice(max(r - rr, 0), r + rr + 1), slice(max(c - rr, 0), c + rr + 1))
        self.visit_pen[sl] += amount
        self._dirty = True

    def blacklist_disc(self, xz, rad=0.6):
        r, c = self.cell(xz)
        rr = int(rad / self.res)
        sl = (slice(max(r - rr, 0), r + rr + 1), slice(max(c - rr, 0), c + rr + 1))
        self.blacklist[sl] = True

    # ------------------------------------------------------------ derived grids
    def occ(self, with_patch=True):
        o = self.hits >= self.min_hits
        return o | (self.patch & (self.now - self.patch_t <= self.patch_ttl)) if with_patch else o

    def _derived(self, with_patch=True):
        key = with_patch
        if self._dirty:
            self._cache = {}
            self._dirty = False
        if key not in self._cache:
            o = self.occ(with_patch)
            if with_patch and not self.patch_inflate:       # patches block their own cells only (not re-inflated by the radius)
                raw = self.hits >= self.min_hits
                dist = ndimage.distance_transform_edt(~raw) * self.res if raw.any() else np.full(o.shape, 99.0)
                dist = np.where(o & ~raw, 0.0, dist)
            else:
                dist = ndimage.distance_transform_edt(~o) * self.res if o.any() else np.full(o.shape, 99.0)
            self._cache[key] = (o, dist)
        return self._cache[key]

    def costs(self, agent_xz, relax=0, with_patch=True):
        """(per-metre cost (inf = blocked), hard-blocked mask). relax 0: radius+margin+soft cost; 1: radius only;
        2: radius only, no soft cost."""
        o, dist = self._derived(with_patch)
        inflate = self.radius + (self.margin if relax == 0 else 0.0)
        if relax >= 3:
            inflate = 0.12
        hard = dist < inflate
        r, c = self.cell(agent_xz)
        rr = int(0.3 / self.res)
        sl = (slice(max(r - rr, 0), r + rr + 1), slice(max(c - rr, 0), c + rr + 1))
        # the agent may already sit inside the inflation zone: only raw obstacles (dilated 0.1 m) block it nearby
        near = np.zeros_like(hard)
        near[sl] = True
        hard = np.where(near, dist < 0.10, hard)
        cost = np.ones(o.shape, np.float32)
        if relax < 2:
            cost += self.soft_w * np.clip((self.soft_d - dist) / (self.soft_d - inflate), 0, 1) ** 2
            known = self.free | o
            cost += np.where(known, 0.0, self.unknown_pen).astype(np.float32)
            cost += self.visit_pen
        cost = np.where(hard, INF, cost)
        return cost, hard

    # ------------------------------------------------------------ queries
    def los(self, a_xz, b_xz, ignore_near_b=0.3, with_patch=True):
        """Grid line of sight from a to each row of b (obstacles within ignore_near_b of b are ignored)."""
        b = np.atleast_2d(b_xz)
        o, _ = self._derived(with_patch)
        a = np.asarray(a_xz, float)
        L = np.linalg.norm(b - a, axis=1)
        n = int(max(L.max() / 0.025, 2)) + 1
        t = np.linspace(0.0, 1.0, n)[None, :]
        pts = a[None, None] + t[:, :, None] * (b - a)[:, None]
        r, c = self.cell(pts.reshape(-1, 2))
        m = self.inside(r, c)
        hit = np.zeros(len(r), bool)
        hit[m] = o[r[m], c[m]]
        hit = hit.reshape(len(b), n)
        keep = (t * L[:, None]) < (L[:, None] - ignore_near_b)
        return ~(hit & keep).any(1)

    def lookup(self, grid, xz, default):
        r, c = self.cell(xz)
        m = self.inside(r, c)
        out = np.full(len(np.atleast_1d(r)), default, grid.dtype if grid.dtype != bool else bool)
        out[m] = grid[r[m], c[m]]
        return out


class Field:
    """Dijkstra cost-to-go on a window of the grid (8-connected), from one or several source cells."""

    def __init__(self, cost, src_rc, window, res, agent_rc):
        r0, r1, c0, c1 = window
        sub = cost[r0:r1, c0:c1].copy()
        ar, ac = agent_rc[0] - r0, agent_rc[1] - c0
        if 0 <= ar < sub.shape[0] and 0 <= ac < sub.shape[1] and not np.isfinite(sub[ar, ac]):
            sub[ar, ac] = 1.0
        H, W = sub.shape
        self.win, self.res, self.shape = window, res, (H, W)
        ids = np.arange(H * W).reshape(H, W)
        ok = np.isfinite(sub)
        rows, cols, w = [], [], []
        for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
            a = ids[:H - dr, max(0, -dc):W - max(0, dc)]
            b = ids[dr:, max(0, dc):W - max(0, -dc)]
            va, vb = sub.ravel()[a.ravel()], sub.ravel()[b.ravel()]
            m = np.isfinite(va) & np.isfinite(vb)
            rows.append(a.ravel()[m])
            cols.append(b.ravel()[m])
            w.append(0.5 * (va[m] + vb[m]) * res * math.hypot(dr, dc))
        rows, cols, w = np.concatenate(rows), np.concatenate(cols), np.concatenate(w)
        g = coo_matrix((w, (rows, cols)), shape=(H * W, H * W)).tocsr()
        src = [(r - r0) * W + (c - c0) for r, c in src_rc if 0 <= r - r0 < H and 0 <= c - c0 < W and ok[r - r0, c - c0]]
        self.valid = bool(src)
        if not src:
            self.dist = np.full((H, W), INF)
            self.pred = np.full(H * W, -9999)
            self.source = np.full(H * W, -9999)
            return
        d, p, s = dijkstra(g, directed=False, indices=src, return_predecessors=True, min_only=True)
        self.dist, self.pred, self.source = d.reshape(H, W), p, s

    def at(self, rc):
        r, c = rc[0] - self.win[0], rc[1] - self.win[2]
        if 0 <= r < self.shape[0] and 0 <= c < self.shape[1]:
            return float(self.dist[r, c])
        return INF

    def source_rc(self, rc):
        r, c = rc[0] - self.win[0], rc[1] - self.win[2]
        s = int(self.source[r * self.shape[1] + c])
        if s < 0:
            return None
        return s // self.shape[1] + self.win[0], s % self.shape[1] + self.win[2]

    def path_from(self, rc, max_len=2000):
        """Cells from rc to the source following the predecessor tree (list of (row, col))."""
        W = self.shape[1]
        i = (rc[0] - self.win[0]) * W + (rc[1] - self.win[2])
        if not np.isfinite(self.at(rc)):
            return []
        out = []
        while i >= 0 and len(out) < max_len:
            out.append((i // W + self.win[0], i % W + self.win[2]))
            i = int(self.pred[i])
        return out


class Plan:
    pass


class LastMileCore:
    name = "lastmile"

    def __init__(self, stop_r=0.9, goal_lo=0.4, goal_hi=0.9, radius=0.18, margin=0.03, unknown_pen=1.0, seed=0,
                 debug_dir=None, hold_cost=1.5, face_turns=3, cloud_r=1.2, floor_h=0.06, top_h=1.6, fine=False, small_m=0.10,
                 scan=True, patch_ttl=120, turn_cap=8):
        self.floor_h, self.top_h = floor_h, top_h
        self.fine, self.small_m, self.scan, self.patch_ttl, self.turn_cap = fine, small_m, scan, patch_ttl, turn_cap
        self.escape_on = bool(env("LM_ESCAPE", 0, int))
        self.cloud_r = cloud_r
        self.stop_r, self.goal_lo, self.goal_hi = stop_r, goal_lo, goal_hi
        self.radius, self.margin, self.unknown_pen, self.seed = radius, margin, unknown_pen, seed
        self.debug_dir, self.hold_cost, self.face_turns = debug_dir, hold_cost, face_turns
        self.m = None
        self.trace = None
        atexit.register(self._flush)

    # ------------------------------------------------------------ episode
    def reset(self, obs, start_info):
        self._flush()
        info = start_info or {}
        pose = obs["pose"]
        cam = info.get("camera", {})
        self.m = LocalMap(np.asarray(pose["position"])[[0, 2]], radius=self.radius, margin=self.margin,
                          unknown_pen=self.unknown_pen, floor_h=self.floor_h, top_h=self.top_h, cam_h=cam.get("height_m", 0.88))
        cloud = info.get("goal_cloud")
        self.cloud = np.zeros((0, 3)) if cloud is None else np.asarray(cloud, float).reshape(-1, 3)
        self.budget = int(info.get("budget_steps", 500))
        self.rng = np.random.RandomState(self.seed)
        self.tag = f"{info.get('scene', 'scene')}_{info.get('ep', 0)}"
        self.step_i, self.last = 0, None
        self.m.patch_ttl = self.patch_ttl
        self.turn_run, self.scanned = 0, False
        self.probe, self.probe_cool, self.probe_n = None, 0, 0
        self.goal_rc = None
        self.queue, self.spin_done, self.relax, self.relax_until = [], 0, 0, -1
        self.coll_run, self.coll_total, self.fail_keys = 0, 0, set()
        self.visits, self.face_used = {}, 0
        self.hist, self.cool, self.esc_n = [], 0, 0
        self.trace = dict(pos=[], collisions=[], goals=[], path=None, stops=None, events=[], acts=[])
        self.plan = None

    def log(self, ev):
        if self.trace is not None:
            self.trace["events"].append((self.step_i, ev))

    # ------------------------------------------------------------ helpers
    def _cloud_xz(self, y):
        c = self.cloud
        if not len(c):
            return np.zeros((0, 2))
        near = np.abs(c[:, 1] - y) <= 1.4
        return (c[near] if near.any() else c)[:, [0, 2]]

    def _body(self, cxz, near=0.25, reach=1.8):
        """Occupied-map cells of the object itself: connected obstacle components that touch the cloud (a centroid
        cloud lies inside a sofa, so the surface the agent must reach is only known from the depth map)."""
        m = self.m
        o = m.occ(False)
        c0 = cxz.mean(0)
        r, c = m.cell(c0)
        R = int((reach + 1.0) / m.res)
        r0, r1, q0, q1 = max(r - R, 0), min(r + R, m.n), max(c - R, 0), min(c + R, m.n)
        sub = o[r0:r1, q0:q1]
        if not sub.any():
            return np.zeros((0, 2))
        lab, n = ndimage.label(sub, structure=np.ones((3, 3)))
        cr, cc = m.cell(cxz)
        want = set()
        rr = int(near / m.res)
        for a, b in zip(cr, cc):
            win = lab[max(a - r0 - rr, 0):max(a - r0 + rr + 1, 0), max(b - q0 - rr, 0):max(b - q0 + rr + 1, 0)]
            want |= set(np.unique(win[win > 0]).tolist())
        if not want:
            return np.zeros((0, 2))
        rows, cols = np.nonzero(np.isin(lab, list(want)))
        pts = m.world(rows + r0, cols + q0)
        d = np.min(np.linalg.norm(pts[:, None] - cxz[None], axis=2), axis=1) if len(cxz) * len(pts) < 4e6 else np.linalg.norm(pts - c0, axis=1)
        pts = pts[d <= reach]
        return pts[:: max(1, len(pts) // 1500)]

    def _nearest(self, xz, cxz):
        d = np.linalg.norm(cxz - xz, axis=1)
        i = int(d.argmin())
        return float(d[i]), cxz[i]

    # ------------------------------------------------------------ main step
    def act(self, obs):
        if obs.get("depth") is None:
            raise ValueError("last-mile navigators need depth: run the bench with --render")
        m, p = self.m, obs["pose"]
        pos, yaw = np.asarray(p["position"], float), float(p["yaw"])
        pxz = pos[[0, 2]]
        self.step_i += 1
        collided = False
        m.now = self.step_i
        if self.last is not None and self.last[0] in (FORWARD, FWD_S):
            if np.linalg.norm(pxz - self.last[1]) < (0.05 if self.last[0] == FORWARD else 0.03):
                collided = True
                key = (tuple(np.round(pxz, 2)), round(yaw, 2))
                k = sum(1 for kk in self.fail_keys if kk[0] == key[0])
                self.fail_keys.add(key)
                if not self.probe:
                    m.mark_collision(pos, yaw, min(k, 3), small=self.last[0] == FWD_S)
                self.coll_run += 1
                self.coll_total += 1
                self.trace["collisions"].append(pxz.copy())
                self.queue = [a for a in self.queue if a not in (FORWARD, FWD_S)]
            else:
                self.coll_run = 0
        m.update(pos, float(yaw), float(p.get("pitch", 0.0)), obs["depth"])
        gc = obs.get("goal_cloud")
        if gc is not None and len(gc):
            self.cloud = np.asarray(gc, float).reshape(-1, 3)
        self.trace["pos"].append(pxz.copy())
        cxz = self._cloud_xz(pos[1])
        if not len(cxz):
            return self._ret(STOP, pxz)
        ncl = len(cxz)
        cxz = np.concatenate([cxz, self._body(cxz)])
        self.objxz = cxz
        self.cloud_xz = cxz[:ncl]
        dist, near_pt = self._nearest(pxz, cxz)
        remaining = self.budget - self.step_i
        # ---- stop rule
        dcl = float(np.linalg.norm(self.cloud_xz - pxz, axis=1).min())
        if (dist <= self.stop_r or (collided and dist <= 1.0)) and dcl <= self.cloud_r:
            los = bool(m.los(pxz, near_pt[None])[0])
            if los:
                bearing = wrap(yaw_to(near_pt - pxz) - yaw)
                if abs(bearing) <= math.radians(60) or self.face_used >= self.face_turns or remaining < 2:
                    self.log(f"STOP d={dist:.2f}")
                    self.trace["stops"] = pxz.copy()
                    return self._ret(STOP, pxz)
                self.face_used += 1
                return self._ret(LEFT if bearing > 0 else RIGHT, pxz)
        # ---- initial 360 deg scan (the rescuer starts with an empty map); 12 x 30 deg sees what 24 x 15 deg would
        if self.scan and not self.scanned:
            self.scanned = True
            if remaining > 40:
                self.queue = [LEFT] * 11
                self.log("scan")
                return self._ret(LEFT, pxz)
        # ---- queued recovery actions
        if self.queue:
            a = self.queue.pop(0)
            return self._ret(a, pxz)
        # ---- empirical probe sweep: in a pocket the map can be wrong by 10 cm; try small forward steps on every heading
        if self.fine:
            self.hist.append(pxz.copy())
            self.hist = self.hist[-20:]
            if self.probe is not None:
                a = self._probe_step(pxz, yaw)
                if a is not None:
                    return self._ret(a, pxz)
            elif (len(self.hist) >= 20 and self.step_i >= self.probe_cool and remaining > 12 and self.scanned
                  and np.linalg.norm(np.array(self.hist) - pxz, axis=1).max() < 0.25):
                self.probe = dict()
                self.probe_n += 1
                self.log("probe sweep")
                a = self._probe_step(pxz, yaw)
                if a is not None:
                    return self._ret(a, pxz)
        # ---- limit cycles
        key = (tuple(np.round(pxz / 0.15).astype(int)), int(round(yaw / TURN)) % 12)
        if not collided:
            self.visits[key] = self.visits.get(key, 0) + 1
        if self.visits.get(key, 0) >= 3:
            self.visits[key] = 0
            m.add_visit_penalty(pxz, 0.35, 1.5)
            if self.goal_rc is not None:
                m.blacklist_disc(m.world(*self.goal_rc), 0.5)
            self.goal_rc = None
            self.log("limit-cycle")
        if self.coll_run >= 3:
            self.coll_run = 0
            if remaining > 8:
                self.queue = [LEFT] * 6 + [FORWARD, FORWARD]
                if self.goal_rc is not None:
                    m.blacklist_disc(m.world(*self.goal_rc), 0.4)
                    self.goal_rc = None
                self.log("back-off")
                return self._ret(self.queue.pop(0), pxz)
        # ---- stuck (dithering in a pocket): drive out along the clearest heading
        if not self.fine:
            self.hist.append(pxz.copy())
            self.hist = self.hist[-24:]
        if (self.escape_on and len(self.hist) >= 24 and self.step_i >= self.cool and remaining > 6
                and np.linalg.norm(np.array(self.hist) - pxz, axis=1).max() < 0.4):
            self.cool = self.step_i + 24
            self.queue = self._escape(pxz, yaw, near_pt)
            self.log(f"escape {self.queue}")
            if self.queue:
                return self._ret(self.queue.pop(0), pxz)
        # ---- plan
        plan = self._plan(pos, yaw, pxz, cxz)
        self.plan = plan
        if plan is None:
            return self._ret(self._recover(pxz), pxz)
        a = self.steer(plan, pos, yaw, obs)
        self.turn_run = self.turn_run + 1 if a in (LEFT, RIGHT, LEFT15, RIGHT15) else 0
        if self.turn_run > self.turn_cap:                              # dithering: stop turning, probe forward
            self.turn_run = 0
            self.log("turn cap -> probe")
            return self._ret(FWD_S if self.fine else FORWARD, pxz)
        if a == FORWARD and (tuple(np.round(pxz, 2)), round(yaw, 2)) in self.fail_keys:
            a = LEFT if plan.err >= 0 else RIGHT                     # never repeat a failed forward from this pose
        return self._ret(a, pxz)

    def _ret(self, a, pxz):
        if self.trace is not None:
            self.trace["acts"].append(a)
        self.last = (a, pxz.copy())
        return a

    # ------------------------------------------------------------ planning
    def _regions(self, cxz, level):
        m = self.m
        cell_r, cell_c = m.cell(cxz)
        ok = m.inside(cell_r, cell_c)
        mask = np.ones((m.n, m.n), bool)
        mask[cell_r[ok], cell_c[ok]] = False
        d, idx = ndimage.distance_transform_edt(mask, return_indices=True)
        d = d * m.res
        lo, hi = [(self.goal_lo, self.goal_hi), (0.3, 0.75), (0.3, 0.75), (0.0, 2.0)][level]
        return d, idx, lo, hi

    def _plan(self, pos, yaw, pxz, cxz):
        m = self.m
        ar, ac = m.cell(pxz)
        if not m.inside(np.array(ar), np.array(ac)):
            return None
        lo_r = min(ar, *(m.cell(cxz)[0])) - int(4.0 / m.res)
        hi_r = max(ar, *(m.cell(cxz)[0])) + int(4.0 / m.res)
        lo_c = min(ac, *(m.cell(cxz)[1])) - int(4.0 / m.res)
        hi_c = max(ac, *(m.cell(cxz)[1])) + int(4.0 / m.res)
        win = (max(lo_r, 0), min(hi_r, m.n), max(lo_c, 0), min(hi_c, m.n))
        relax = max(self.relax if self.step_i <= self.relax_until else 0, 0)
        for rl in range(relax, 3):
            for patch in (True,) if rl < 2 else (True, False):
                cost, hard = m.costs(pxz, rl, with_patch=patch)
                for level in range(4):
                    plan = self._try(cost, hard, win, cxz, (ar, ac), pxz, yaw, level)
                    if plan is not None:
                        plan.relax, plan.level, plan.patch = rl, level, patch
                        if rl > relax or not patch:
                            self.log(f"relaxed {rl} patch={patch}")
                        return plan
        return None

    def _try(self, cost, hard, win, cxz, arc, pxz, yaw, level):
        m = self.m
        d, idx, lo, hi = self._regions(cxz, level)
        r0, r1, c0, c1 = win
        reg = (d >= lo) & (d <= hi) & ~hard & ~m.blacklist
        if level < 3:                                                  # and not far from the perceived cloud itself
            cr_, cc_ = m.cell(self.cloud_xz)
            okc = m.inside(cr_, cc_)
            mk = np.ones((m.n, m.n), bool)
            mk[cr_[okc], cc_[okc]] = False
            reg &= (ndimage.distance_transform_edt(mk) * m.res) <= self.cloud_r - 0.05
        reg[:r0], reg[r1:], reg[:, :c0], reg[:, c1:] = False, False, False, False
        rr, cc = np.nonzero(reg)
        if not len(rr):
            return None
        if level < 2:                                                  # needs line of sight on the grid to the object
            tgt = m.world(idx[0][rr, cc], idx[1][rr, cc])
            wx = m.world(rr, cc)
            los = np.empty(len(rr), bool)
            for i in range(0, len(rr), 4000):
                # per-candidate segments to the nearest cloud cell, evaluated in chunks
                los[i:i + 4000] = self._los_pairs(wx[i:i + 4000], tgt[i:i + 4000])
            rr, cc = rr[los], cc[los]
            if not len(rr):
                return None
        if level == 3:                                                  # approach as close as possible
            f0 = Field(cost, [arc], win, m.res, arc)
            dd = f0.dist[rr - r0, cc - c0]
            ok = np.isfinite(dd)
            if not ok.any():
                return None
            i = int(np.argmin(np.where(ok, d[rr, cc], INF)))
            srcs = [(int(rr[i]), int(cc[i]))]
        else:
            srcs = list(zip(rr.tolist(), cc.tolist()))
        multi = Field(cost, srcs, win, m.res, arc)
        best = multi.source_rc(arc)
        cbest = multi.at(arc)
        if best is None or not np.isfinite(cbest):
            return None
        keep, field = best, multi
        if self.goal_rc is not None and self.goal_rc != best and self.goal_rc in set(srcs):
            fg = Field(cost, [self.goal_rc], win, m.res, arc)
            cg = fg.at(arc)
            if np.isfinite(cg) and cg <= cbest + self.hold_cost:
                keep, field, cbest = self.goal_rc, fg, cg
        if keep != self.goal_rc:
            self.trace["goals"].append(m.world(*keep))
        self.goal_rc = keep
        pl = Plan()
        pl.field, pl.cost, pl.hard, pl.win, pl.goal_rc, pl.arc, pl.cbest = field, cost, hard, win, keep, arc, cbest
        pl.cells = field.path_from(arc)
        pl.path = m.world(np.array([c[0] for c in pl.cells]), np.array([c[1] for c in pl.cells])) if pl.cells else np.zeros((0, 2))
        pl.err = 0.0
        self.trace["path"] = pl.path
        return pl

    def _los_pairs(self, a, b):
        """LOS for paired segments a[i] -> b[i] on the grid."""
        m = self.m
        o, _ = m._derived(True)
        L = np.linalg.norm(b - a, axis=1)
        n = int(max(L.max() / 0.05, 2)) + 1
        t = np.linspace(0.0, 1.0, n)[None, :]
        pts = a[:, None] + t[:, :, None] * (b - a)[:, None]
        r, c = m.cell(pts.reshape(-1, 2))
        ins = m.inside(r, c)
        hit = np.zeros(len(r), bool)
        hit[ins] = o[r[ins], c[ins]]
        hit = hit.reshape(len(a), n)
        keep = (t * L[:, None]) < (L[:, None] - 0.3)
        return ~(hit & keep).any(1)

    def _probe_targets(self, pxz, yaw):
        """Headings (absolute yaw, multiples of 15 deg from the current one) ordered by map clearance, largest first;
        trying the most open headings first keeps partial wall-hugging moves (which shift the agent) to a minimum."""
        m = self.m
        o = m.occ(False)
        out = []
        for i in range(24):
            h = yaw + i * math.pi / 12
            ts = np.arange(0.1, 1.3, 0.05)
            pts = pxz[None] + ts[:, None] * fwd_vec(h)
            r, c = m.cell(pts)
            ok = m.inside(r, c)
            occ = np.zeros(len(ts), bool)
            occ[ok] = o[r[ok], c[ok]]
            clear = float(ts[occ.argmax()]) if occ.any() else 1.3
            out.append((-clear, i))
        return [i for _, i in sorted(out)]

    def _probe_step(self, pxz, yaw=None):
        """Probe sweep with FORWARD (0.25; habitat refuses the 0.10 step in tight spots): visit headings, most open first;
        a heading that moves >= 0.1 m is followed for 2 steps, then the normal planner resumes."""
        p = self.probe
        if yaw is None:
            yaw = self._yaw
        if "targets" not in p:
            p["targets"], p["k"], p["moves"], p["q"], p["base"] = self._probe_targets(pxz, yaw), 0, 0, [], yaw
        if self.last is not None and self.last[0] == FORWARD and p.get("fwd_pending"):
            p["fwd_pending"] = False
            if np.linalg.norm(pxz - self.last[1]) >= 0.1:
                p["moves"] += 1
                if p["moves"] >= 2:
                    self.probe, self.probe_cool, self.hist = None, self.step_i + 25, []
                    return None
                p["fwd_pending"] = True
                return FORWARD
            if p["moves"] > 0:
                self.probe, self.probe_cool, self.hist = None, self.step_i + 25, []
                return None
            p["k"] += 1
        if p["q"]:
            a = p["q"].pop(0)
            if a == "F":
                p["fwd_pending"] = True
                return FORWARD
            return a
        if p["k"] >= len(p["targets"]):
            self.probe, self.probe_cool, self.hist = None, self.step_i + 60, []
            return None
        tgt = p["targets"][p["k"]]
        cur = int(round(wrap(yaw - p["base"]) / (math.pi / 12))) % 24
        d = (tgt - cur) % 24
        if d > 12:
            d -= 24
        q = []
        n30, n15 = divmod(abs(d), 2)
        q += [LEFT if d > 0 else RIGHT] * n30 + [LEFT15 if d > 0 else RIGHT15] * n15
        q.append("F")
        p["q"] = q
        a = p["q"].pop(0)
        if a == "F":
            p["fwd_pending"] = True
            return FORWARD
        return a

    # ------------------------------------------------------------ recovery
    def _escape(self, pxz, yaw, goal_xz):
        m = self.m
        o, _ = m._derived(True)
        k = self.esc_n
        self.esc_n += 1
        gdir = yaw_to(goal_xz - pxz)
        best = None
        for i in range(12):
            h = yaw + i * TURN
            f = fwd_vec(h)
            ts = np.arange(0.1, 1.6, 0.05)
            pts = pxz[None] + ts[:, None] * f
            r, c = m.cell(pts)
            ok = m.inside(r, c)
            occ = np.ones(len(ts), bool)
            occ[ok] = o[r[ok], c[ok]]
            clear = float(ts[occ.argmax()]) if occ.any() else 1.6
            turn = min(i, 12 - i)
            sc = min(clear, 1.2) + 0.4 * math.cos(wrap(h - gdir)) - 0.03 * turn + 0.05 * ((i * 7 + k * 5) % 12) / 12
            if best is None or sc > best[0]:
                best = (sc, i, clear)
        _, i, clear = best
        turns = [LEFT] * i if i <= 6 else [RIGHT] * (12 - i)
        return turns + [FORWARD] * int(max(1, min(4, (clear - 0.2) // FWD_M)))

    def _recover(self, pxz):
        m = self.m
        remaining = self.budget - self.step_i
        self.log("no path")
        if self.spin_done == 0 and remaining > 8:
            self.spin_done = 1
            self.queue = [LEFT] * 5                                    # sense all around, then replan
            return self.queue.pop(0)
        if self.spin_done == 1:
            self.spin_done = 2
            m.patch[:] = False                                         # patches may be wrong: forget them
            m.blacklist[:] = False
            m._dirty = True
            self.relax, self.relax_until = 1, self.step_i + 25
            return LEFT
        self.spin_done = 0
        self.queue = [LEFT] * 6 + [FORWARD]
        return self.queue.pop(0)

    def steer(self, plan, pos, yaw, obs):
        raise NotImplementedError

    # ------------------------------------------------------------ debug
    def _flush(self):
        if not self.debug_dir or self.trace is None or self.m is None or not self.trace["pos"]:
            return
        try:
            import cv2
            os.makedirs(self.debug_dir, exist_ok=True)
            m = self.m
            pts = np.array(self.trace["pos"])
            ext = [pts]
            if len(self.cloud):
                ext.append(self.cloud[:, [0, 2]])
            allp = np.concatenate(ext)
            lo, hi = allp.min(0) - 2.5, allp.max(0) + 2.5
            r0, c0 = m.cell(lo)
            r1, c1 = m.cell(hi)
            r0, c0 = max(r0, 0), max(c0, 0)
            r1, c1 = min(r1, m.n), min(c1, m.n)
            o, dist = m._derived(True)
            img = np.full((r1 - r0, c1 - c0, 3), 127, np.uint8)
            img[m.free[r0:r1, c0:c1]] = (255, 255, 255)
            img[(dist < self.radius + self.margin)[r0:r1, c0:c1] & ~o[r0:r1, c0:c1]] = (200, 190, 150)
            img[o[r0:r1, c0:c1]] = (0, 0, 0)
            img[m.patch[r0:r1, c0:c1]] = (0, 0, 220)
            s = 4
            img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_NEAREST)

            def px(xz):
                r, c = m.cell(xz)
                return int((c - c0 + 0.5) * s), int((r - r0 + 0.5) * s)
            for q in self.cloud[:, [0, 2]]:
                cv2.circle(img, px(q), 2, (0, 160, 0), -1)
            for a, b in zip(pts[:-1], pts[1:]):
                cv2.line(img, px(a), px(b), (255, 0, 0), 1)
            for q in self.trace["collisions"]:
                cv2.circle(img, px(q), 4, (0, 0, 255), 1)
            for g in self.trace["goals"]:
                cv2.drawMarker(img, px(g), (200, 0, 200), cv2.MARKER_CROSS, 8, 1)
            if self.trace["path"] is not None and len(self.trace["path"]):
                for a, b in zip(self.trace["path"][:-1], self.trace["path"][1:]):
                    cv2.line(img, px(a), px(b), (0, 140, 255), 1)
            if self.trace["stops"] is not None:
                cv2.circle(img, px(self.trace["stops"]), 5, (0, 0, 0), 2)
            cv2.circle(img, px(pts[0]), 4, (0, 200, 0), -1)
            fd = float(np.min(np.linalg.norm(self.cloud[:, [0, 2]] - pts[-1], axis=1))) if len(self.cloud) else -1
            cv2.putText(img, f"{self.tag} n={len(pts)} coll={self.coll_total} endd={fd:.2f}", (4, 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
            cv2.imwrite(os.path.join(self.debug_dir, f"{self.name}_{self.tag}.png"), img)
            with open(os.path.join(self.debug_dir, f"{self.name}_{self.tag}.log"), "w") as f:
                f.write("acts " + "".join(map(str, self.trace["acts"])) + "\n")
                f.write("\n".join(f"{s_} {e}" for s_, e in self.trace["events"]))
        except Exception as e:                                          # debugging must never break a run
            print("debug flush failed:", e)
        self.trace = None


def core_kwargs():
    return dict(stop_r=env("LM_STOP_R", 0.8), goal_lo=env("LM_GOAL_LO", 0.4), goal_hi=env("LM_GOAL_HI", 0.75),
                radius=env("LM_RADIUS", 0.18), margin=env("LM_MARGIN", 0.03), unknown_pen=env("LM_UNKNOWN", 1.0),
                seed=env("LM_SEED", 0, int), cloud_r=env("LM_CLOUD_R", 1.2), floor_h=env("LM_FLOOR_H", 0.06), top_h=env("LM_TOP_H", 1.6),
                fine=bool(env("LM_FINE", 0, int)), small_m=env("LM_SMALL_M", 0.10), scan=bool(env("LM_SCAN", 1, int)),
                patch_ttl=env("LM_PATCH_TTL", 120, int), turn_cap=env("LM_TURN_CAP", 8, int), debug_dir=os.environ.get("LM_DEBUG_DIR"))
