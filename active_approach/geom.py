"""Run-time geometry for the planners (docs/active_approach_planners.md section 0; no GT).

Inputs are what ApexNav has at run time: the fused object cloud (world points under the object mask),
depth frames (a local 2.5D height map), the navmesh for paths, the agent pose. World frame: Habitat
(y up); 2D quantities use (x, z).

    ObjectModel(cloud, floor_y)   z_lo/z_hi (5/95 % height above the floor), PCA footprint, centre
    HeightMap                     5 cm cells: max height of non-object depth points, the same below BODY_TOP, seen flag
    predict(...)                  area_hat, trunc_hat, occ_hat, facing_hat of a candidate pose
    d_fit(...)                    the fit distance (pitch 0, margin 3 deg, kappa of the height may be cut)
    lattice_candidates(...)       viewpoints on the 12 lattice yaws (yaw0 + k 30 deg) facing the centre
"""
import math

import numpy as np

from active_approach import scene as S

H_CAM = S.CAMERA_HEIGHT
BETA_V = math.atan((S.HEIGHT / 2) / S.FOCAL)          # 31.7 deg
BETA_H = math.radians(S.HFOV / 2)                     # 39.5 deg
MARGIN = math.radians(3.0)
CELL = 0.05
DEPTH_MAX = 4.99                                       # ApexNav's depth range (map_ros depth_filter_maxdist)
LATTICE = math.radians(S.TURN_DEG)
BODY_TOP = 1.6                                         # m above the floor: higher structure does not block the base


def forward(yaw):
    """Horizontal (x, z) unit vector the camera looks along (yaw 0 = -z, scene.facing convention)."""
    return np.array([-math.sin(yaw), -math.cos(yaw)])


def bearing_yaw(position_xz, target_xz):
    d = np.asarray(target_xz, float) - np.asarray(position_xz, float)
    return math.atan2(-d[0], -d[1])


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class ObjectModel:
    def __init__(self, cloud, floor_y):
        xz = cloud[:, [0, 2]]
        h = cloud[:, 1] - floor_y
        self.z_lo, self.z_hi = (float(np.percentile(h, 5)), float(np.percentile(h, 95))) if len(h) else (0.0, 1.0)
        self.xz = xz
        mean = xz.mean(0)
        cov = np.cov((xz - mean).T) if len(xz) > 2 else np.eye(2)
        w, v = np.linalg.eigh(cov)
        self.axes = v[:, ::-1].T                      # long axis first
        local = (xz - mean) @ self.axes.T
        lo, hi = local.min(0), local.max(0)
        self.extent = hi - lo                         # (L, S)
        self.center = mean + ((lo + hi) / 2) @ self.axes
        self.bbox = np.r_[xz.min(0), xz.max(0)]

    def robust_extent(self, q=(2.0, 98.0)):
        """Footprint (L, S) along the PCA axes between the ``q`` percentiles (2-98 %): ``extent`` without the
        stray points of a merge (size_opinion's width W). (0, 0) for fewer than 3 points."""
        if len(self.xz) < 3:
            return np.zeros(2)
        local = (self.xz - self.xz.mean(0)) @ self.axes.T
        lo, hi = np.percentile(local, q, axis=0)
        return hi - lo

    @property
    def height(self):
        return max(self.z_hi - self.z_lo, 0.05)

    def half_extent(self, direction):
        """Half width of the footprint along a horizontal unit ``direction``."""
        proj = (self.xz - self.center) @ direction
        return float(max(proj.max(), -proj.min())) if len(proj) else 0.0


def d_fit(model, yaw, kappa=0.25, top_clipped=False, side_clipped=False):
    """Distance from the object's near face at which, looking along ``yaw`` at pitch 0, the top fits,
    at most ``kappa`` of the height is cut at the bottom, and the width fits (5 deg margin)."""
    z_hi = model.z_hi
    if top_clipped:  # the top ran off a judged frame: z_hi is only a lower bound
        z_hi = z_hi + 0.2
    t = math.tan(BETA_V - MARGIN)
    need = [0.3]
    if z_hi > H_CAM:
        need.append((z_hi - H_CAM) / t)
    bottom = model.z_lo + kappa * model.height
    if bottom < H_CAM:
        need.append((H_CAM - bottom) / t)
    side = np.array([math.cos(yaw), -math.sin(yaw)])  # perpendicular to forward(yaw)
    half = model.half_extent(side) + (0.3 if side_clipped else 0.0)  # a side ran off a judged frame: W is a lower bound
    need.append(half / math.tan(BETA_H - math.radians(5)))
    return max(need)


class HeightMap:
    """Local 2.5D map around the object: per 5 cm cell the max height (above the floor) of non-object
    depth points (``top``, for occlusion), the max height of those below BODY_TOP (``low``, for collision: a wall
    or a table seen up to the ceiling has ``top`` above BODY_TOP but ``low`` in the base's height band), and
    whether the cell was seen (a hit or crossed by a depth ray)."""

    def __init__(self, center_xz, floor_y, half=7.0):
        self.origin = np.asarray(center_xz, float) - half
        self.n = int(2 * half / CELL)
        self.top = np.full((self.n, self.n), -np.inf, np.float32)
        self.low = np.full((self.n, self.n), -np.inf, np.float32)
        self.seen = np.zeros((self.n, self.n), bool)
        self.floor_y = floor_y
        self._navigable = {}

    def cells(self, xz):
        ij = np.floor((np.asarray(xz) - self.origin) / CELL).astype(int)
        ok = (ij >= 0).all(-1) & (ij < self.n).all(-1)
        return ij, ok

    def update(self, depth, mask, cam_pos, cam_rot, stride=6):
        valid = (depth > 0.05) & (depth < DEPTH_MAX) & ~mask
        pts = S.backproject(depth, valid, cam_pos, cam_rot, stride=stride)
        if not len(pts):
            return
        ij, ok = self.cells(pts[:, [0, 2]])
        ij, h = ij[ok], (pts[ok, 1] - self.floor_y).astype(np.float32)
        np.maximum.at(self.top, (ij[:, 0], ij[:, 1]), h)
        below = h < BODY_TOP
        np.maximum.at(self.low, (ij[below, 0], ij[below, 1]), h[below])
        self.seen[ij[:, 0], ij[:, 1]] = True
        # free space: cells crossed by the rays to every 4th hit
        cam = cam_pos[[0, 2]]
        far = pts[ok][::4][:, [0, 2]]
        if not len(far):
            return
        steps = np.arange(0.0, 1.0, CELL / max(np.linalg.norm(far - cam, axis=1).max(), CELL))
        ray = cam[None, None] + steps[None, :, None] * (far - cam)[:, None]
        rij, rok = self.cells(ray.reshape(-1, 2))
        self.seen[rij[rok, 0], rij[rok, 1]] = True

    def navigable(self, i, j, pathfinder):
        key = (i, j)
        if hasattr(pathfinder, "cost"):   # local_nav.LocalPathfinder: Predictor asks only about unseen cells, and
            return False                  # unseen low space may hide an occluder (half weight, gap 2 of the design doc)
        if key not in self._navigable:
            x, z = self.origin + (np.array([i, j]) + 0.5) * CELL
            self._navigable[key] = bool(pathfinder.is_navigable(np.array([x, self.floor_y, z], np.float32), 0.5))
        return self._navigable[key]


def _near_object(model, hmap, radius=0.15):
    from scipy.ndimage import binary_dilation
    occ = np.zeros((hmap.n, hmap.n), bool)
    ij, ok = hmap.cells(model.xz)
    occ[ij[ok, 0], ij[ok, 1]] = True
    r = int(math.ceil(radius / CELL))
    return binary_dilation(occ, iterations=r)


class Predictor:
    """``predict(position_xz, yaw)``: view statistics of the object from a candidate pose (pitch 0)."""

    def __init__(self, model, cloud, hmap, pathfinder, n_points=256, seed=0, extra=None):
        """``extra``: phantom points (unseen extent past a clipped border) that count for area and truncation
        but not for occlusion."""
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(cloud), size=min(n_points, len(cloud)), replace=False)
        self.points, self.model, self.hmap, self.pf = cloud[idx], model, hmap, pathfinder
        self.extent_points = self.points
        if extra is not None and len(extra):
            k = rng.choice(len(extra), size=min(n_points // 2, len(extra)), replace=False)
            self.extent_points = np.concatenate([self.points, extra[k]])
        self.near = _near_object(model, hmap)

    def predict(self, position_xz, yaw):
        import cv2
        x, z = position_xz
        cam = np.array([x, self.hmap.floor_y + H_CAM, z])
        f, right = forward(yaw), np.array([math.cos(yaw), -math.sin(yaw)])
        rel = self.extent_points - cam
        depth = rel[:, [0, 2]] @ f
        u = S.WIDTH / 2 + S.FOCAL * (rel[:, [0, 2]] @ right) / np.maximum(depth, 1e-3)
        v = S.HEIGHT / 2 - S.FOCAL * rel[:, 1] / np.maximum(depth, 1e-3)
        margin = S.FOCAL * math.tan(MARGIN)
        inside = (depth > 0.1) & (u >= margin) & (u < S.WIDTH - margin) & (v >= margin) & (v < S.HEIGHT - margin)
        trunc = 1.0 - float(inside.mean())
        occ = self._occlusion(cam, self.points)
        area = 0.0
        if inside.sum() >= 3:
            uv = np.stack([np.clip(u, 0, S.WIDTH), np.clip(v, 0, S.HEIGHT)], 1)[depth > 0.1].astype(np.float32)
            area = float(cv2.contourArea(cv2.convexHull(uv))) / (S.WIDTH * S.HEIGHT) * (1 - occ)
        ray = self.model.center - np.asarray(position_xz)
        ray = ray / max(np.linalg.norm(ray), 1e-6)
        facing = float(max(abs(ray @ self.model.axes[0]), abs(ray @ self.model.axes[1])))
        distance = float(np.linalg.norm(self.model.center - np.asarray(position_xz)))
        return {"area": area, "trunc": trunc, "occ": occ, "facing": facing, "distance": distance}

    def _occlusion(self, cam, points):
        """Fraction of camera->point rays blocked by a map cell higher than the ray (unknown,
        non-navigable cells: assumed 1.0 m high, half weight); cells near the footprint ignored."""
        hm = self.hmap
        rel = points - cam
        length = np.linalg.norm(rel[:, [0, 2]], axis=1)
        n = max(int(length.max() / CELL), 2)
        t = np.linspace(0.02, 0.98, n)
        samples = cam[None, None] + t[None, :, None] * rel[:, None]          # (P, n, 3)
        ij, ok = hm.cells(samples[..., [0, 2]].reshape(-1, 2))
        ij = ij.reshape(len(points), n, 2)
        ok = ok.reshape(len(points), n)
        ray_h = samples[..., 1] - hm.floor_y
        blocked = np.zeros(len(points))
        for p in range(len(points)):
            cells = ij[p][ok[p]]
            if not len(cells):
                continue
            keep = ~self.near[cells[:, 0], cells[:, 1]]
            cells, h = cells[keep], ray_h[p][ok[p]][keep]
            top, seen = hm.top[cells[:, 0], cells[:, 1]], hm.seen[cells[:, 0], cells[:, 1]]
            if np.any(seen & (top > h + 0.05)):
                blocked[p] = 1.0
                continue
            unknown = np.flatnonzero(~seen & (h < 1.0))
            for k in unknown[::4][:6]:
                if not hm.navigable(*cells[k], self.pf):
                    blocked[p] = 0.5
                    break
        return float(blocked.mean())


def lattice_candidates(model, yaw0, pathfinder, floor_y, kappa=0.25, extra=(0.0, 0.25, 0.5), top_clipped=False,
                       side_clipped=False, snap_tol=0.15):
    """[{position (x, y, z), yaw, k, r, d_fit}]: for each lattice yaw psi_k = yaw0 + k 30 deg the camera stands
    at centre - (half depth + d_fit + extra) forward(psi_k), looking at the centre (zero facing error);
    snapped to the navmesh, dropped when the snap moves it more than ``snap_tol``."""
    out = []
    for k in range(12):
        yaw = wrap(yaw0 + k * LATTICE)
        f = forward(yaw)
        fit = d_fit(model, yaw, kappa, top_clipped, side_clipped)
        base = model.half_extent(f)
        for e in extra:
            r = base + math.ceil(fit / 0.25) * 0.25 + e
            xz = model.center - r * f
            want = np.array([xz[0], floor_y, xz[1]], np.float32)
            q = np.asarray(pathfinder.snap_point(want), float)
            if not np.all(np.isfinite(q)) or np.hypot(*(q - want)[[0, 2]]) > snap_tol or abs(q[1] - floor_y) > 0.3:
                continue
            out.append({"position": q, "yaw": yaw, "k": k, "r": r, "d_fit": fit})
    return out


def action_cost(pathfinder, position, yaw, goal, goal_yaw):
    """Actions to reach ``goal`` and face ``goal_yaw``: geodesic / 0.25 plus the turns along the path
    polyline and the final face turn (30 deg each). inf without a path."""
    import habitat_sim
    path = getattr(pathfinder, "ShortestPath", habitat_sim.ShortestPath)()   # local_nav.LocalPathfinder: own class
    path.requested_start, path.requested_end = np.asarray(position, np.float32), np.asarray(goal, np.float32)
    if not pathfinder.find_path(path):
        return math.inf
    turns, heading = 0.0, yaw
    pts = [np.asarray(p, float) for p in path.points]
    for a, b in zip(pts, pts[1:]):
        if np.hypot(*(b - a)[[0, 2]]) < 0.05:
            continue
        h = bearing_yaw(a[[0, 2]], b[[0, 2]])
        turns += abs(wrap(h - heading)) / LATTICE
        heading = h
    turns += abs(wrap(goal_yaw - heading)) / LATTICE
    return path.geodesic_distance / S.FORWARD_M + round(turns)
