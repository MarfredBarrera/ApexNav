"""Self-contained approach planners (docs/selfcontained_view_planners.md): no navmesh, no host map. They read
only the Observation (own RGB-D frame, odometry, the object mask and the fused object cloud) and the self-built
local map through ``obs.pathfinder`` (local_nav.LocalPathfinder, with ``--selfcontained``).

    closer      N1, the strong naive baseline: face the object, drive the straight line to the fit distance
                r = clip(half extent + d_fit (phantom extents), 0.9, 2.2) m, stop facing it (one view)
    closer_rc   N2: N1 + one lattice arc of 30 deg around (the cheaper free side) for a second view
    irc         D1 image-reflex closer: image cues only: approach while the mask is small, stop as soon as the
                top/bottom border is touched with area >= 4% or the area reaches A_STOP
    oes         D3 occlusion escape: N1, but when the frame shows occluders in front of the object (pixels
                nearer than the object inside its widened box), move to the ring pose 30/60 deg toward the
                less occluded side whose sight lines are free on the local map (at most twice)
    tvd         D5 two-view disparity: N1, then a second ring view >= 45 deg from every judged bearing
    ccs         D7 clear-cone standoff: ring pose (any bearing within 90 deg, r in fit .. fit + 1) minimising
                actions + w_los x blocked sight lines + w_unk x unknown cells on them
Each is also registered with the Seek wrapper (``*_seek``: the native hand-off when < 2 views were judged).
"""
import math

import numpy as np

from active_approach import geom as G
from active_approach.policies import Viewpoint, facing, wrap

R_MIN, R_MAX = 0.9, 2.2


def _model(obs):
    from active_approach.round3 import phantom_model
    model, extra, info = phantom_model(obs)
    return model, info


def fit_radius(obs, model, info, u):
    """Distance from the object centre along unit ``u`` (centre -> camera) at which the whole object fits."""
    yaw = G.bearing_yaw(model.center + u, model.center)
    side = info.get("left") or info.get("right")
    fit = G.d_fit(model, yaw, top_clipped=bool(info.get("top")), side_clipped=bool(side))
    return float(np.clip(model.half_extent(u) + fit, R_MIN, R_MAX))


def unit(v):
    return v / max(np.linalg.norm(v), 1e-6)


def vp3(obs, xz, center, info):
    pos = np.array([xz[0], obs.floor_y, xz[1]])
    return Viewpoint(pos, G.bearing_yaw(xz, center), info)


def ring_pose(center, bearing, r):
    """xz at range r on bearing (rad, direction centre -> camera)."""
    return center + r * np.array([math.cos(bearing), math.sin(bearing)])


def bearing_of(center, xz):
    d = np.asarray(xz) - center
    return math.atan2(d[1], d[0])


def path_actions(obs, goal_xz):
    """Actions to reach goal_xz on the local map (0.25 m steps + ~1 turn per 45 deg of heading change)."""
    pf = obs.pathfinder
    a = np.array([obs.position[0], obs.floor_y, obs.position[2]])
    b = np.array([goal_xz[0], obs.floor_y, goal_xz[1]])
    path = pf.ShortestPath() if hasattr(pf, "ShortestPath") else None
    if path is None:
        import habitat_sim
        path = habitat_sim.ShortestPath()
    path.requested_start, path.requested_end = a.astype(np.float32), b.astype(np.float32)
    if not pf.find_path(path):
        return math.inf
    pts = [np.asarray(p, float)[[0, 2]] for p in path.points]
    heading, turns = obs.yaw, 0.0
    for p, q in zip(pts, pts[1:]):
        if np.hypot(*(q - p)) < 0.05:
            continue
        h = G.bearing_yaw(p, q)
        turns += abs(wrap(h - heading)) / G.LATTICE
        heading = h
    return path.geodesic_distance / 0.25 + turns


def sight_lines(obs, model, xz, n=5):
    """(blocked fraction, unknown fraction) of n sight lines from xz to points across the object's width."""
    lm = obs.pathfinder.lmap
    occupied, free, unknown = lm.grids()
    occupied = occupied & ~lm.object
    u = unit(model.center - xz)
    side = np.array([-u[1], u[0]])
    w = model.half_extent(side)
    blocked = unk = 0
    for t in np.linspace(-0.8, 0.8, n):
        target = model.center + t * w * side
        seg = target - xz
        length = np.linalg.norm(seg)
        hit = False
        cells = set()
        for s in np.arange(0.2, max(length - 0.25, 0.2), 0.05):
            cells.add(lm.cell(xz + s * seg / max(length, 1e-6)))
        for c in cells:
            if occupied[c]:
                hit = True
        blocked += hit
        unk += np.mean([unknown[c] for c in cells]) if cells else 0.0
    return blocked / n, unk / n


def occluder_sides(obs, pad=0.25, margin=0.15):
    """(occ_L, occ_R): pixels nearer than the object (median depth - margin) inside the mask's box widened by
    ``pad`` of its size, left / right of the mask centre, as a fraction of that box half."""
    mask = np.asarray(obs.mask, bool)
    if mask.sum() < 50:
        return 0.0, 0.0
    depth = np.asarray(obs.depth)
    ys, xs = np.nonzero(mask)
    d_obj = float(np.median(depth[mask]))
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    pw, ph = int(pad * (x1 - x0 + 1)), int(pad * (y1 - y0 + 1))
    X0, X1 = max(0, x0 - pw), min(mask.shape[1], x1 + pw + 1)
    Y0, Y1 = max(0, y0 - ph), min(mask.shape[0], y1 + ph + 1)
    box_d, box_m = depth[Y0:Y1, X0:X1], mask[Y0:Y1, X0:X1]
    occ = (~box_m) & (box_d > 0.05) & (box_d < d_obj - margin)
    cx = int(np.mean(xs)) - X0
    left, right = occ[:, :cx], occ[:, cx:]
    return float(left.mean()) if left.size else 0.0, float(right.mean()) if right.size else 0.0


class Closer:
    """N1. One stop on the straight line at the fit distance; the goal is re-estimated every 4 steps."""
    name, enroute, driver, reason = "closer", False, None, None
    EVERY = 4

    def reset(self, ctx):
        self.goal, self.age, self.stops = None, 0, 0

    def done_after(self):
        return 1

    def line_goal(self, obs):
        model, info = _model(obs)
        here = np.asarray(obs.position)[[0, 2]]
        u = unit(here - model.center)
        r = fit_radius(obs, model, info, u)
        if np.linalg.norm(here - model.center) <= r + 0.15:   # already close enough: just face it
            return vp3(obs, here, model.center, {"r": round(r, 2), "stop": "here"})
        return vp3(obs, model.center + r * u, model.center, {"r": round(r, 2)})

    def first_stop(self, obs):
        if self.goal is None or self.age >= self.EVERY:
            self.goal, self.age = self.line_goal(obs), 0
        self.age += 1
        return self.goal

    def after_first(self, obs):
        return None

    def next_viewpoint(self, obs):
        if not len(obs.cloud):
            return None
        if obs.arrived:
            self.stops += 1
            self.goal = None
        if self.stops == 0:
            return self.first_stop(obs)
        return self.after_first(obs)


class CloserArc(Closer):
    """N2: after N1's stop, one ring view SEP deg around on the cheaper reachable side."""
    name, SEP, MIN_SEP = "closer_rc", 30.0, None

    def after_first(self, obs):
        if self.stops > 1:
            return None
        if self.goal is None:
            model, info = _model(obs)
            here = np.asarray(obs.position)[[0, 2]]
            b0 = bearing_of(model.center, here)
            r = max(np.linalg.norm(here - model.center), R_MIN)
            best = None
            for sign in (1, -1):
                for sep in self.seps():
                    xz = ring_pose(model.center, b0 + sign * math.radians(sep), r)
                    if self.MIN_SEP and not self.novel(obs, model, xz):
                        continue
                    cost = path_actions(obs, xz)
                    if best is None or cost < best[0]:
                        best = (cost, xz, sep * sign)
            if best is None or not np.isfinite(best[0]) or best[0] > obs.remaining - 2:
                return None
            self.goal = vp3(obs, best[1], model.center, {"arc": best[2], "A": round(best[0], 1)})
        return self.goal

    def seps(self):
        return (self.SEP,)

    def novel(self, obs, model, xz):
        b = bearing_of(model.center, xz)
        return all(abs(wrap(b - bearing_of(model.center, np.asarray(j["camera"])))) >= math.radians(self.MIN_SEP)
                   for j in obs.judged)


class TwoView(CloserArc):
    """D5: the second view >= 45 deg from every judged bearing (45, 60 or 90 deg arcs, the cheapest)."""
    name, MIN_SEP = "tvd", 45.0

    def seps(self):
        return (45.0, 60.0, 90.0)


class ImageReflex:
    """D1: image cues only (mask area, borders, offset); the cloud only gives the direction to drive."""
    name, enroute, driver, reason = "irc", False, None, None
    A_STOP, A_TOUCH = 0.06, 0.04

    def reset(self, ctx):
        self.stopped = False

    def next_viewpoint(self, obs):
        if not len(obs.cloud) or (obs.arrived and self.stopped):
            return None
        center = obs.cloud[:, [0, 2]].mean(0)
        here = np.asarray(obs.position)[[0, 2]]
        mask = np.asarray(obs.mask, bool)
        area = float(mask.mean())
        offset = abs(wrap(G.bearing_yaw(here, center) - obs.yaw))
        if mask.sum() < 50 or offset > math.radians(20):
            return vp3(obs, here, center, {"irc": "face"})
        touch_tb = bool(mask[0].any() or mask[-1].any())
        dist = float(np.linalg.norm(here - center))
        if (touch_tb and area >= self.A_TOUCH) or area >= self.A_STOP or dist <= R_MIN:
            self.stopped = True
            return vp3(obs, here, center, {"irc": "stop", "area": round(area, 3)})
        goal = center + max(dist - 0.5, R_MIN) * unit(here - center)
        return vp3(obs, goal, center, {"irc": "approach", "area": round(area, 3)})


class OcclusionEscape(Closer):
    """D3: N1, with up to MAX_ESC sidesteps when the frame shows occluders in front of the object."""
    name, TRIGGER, MAX_ESC = "oes", 0.25, 2

    def reset(self, ctx):
        super().reset(ctx)
        self.escapes, self.escape = 0, None

    def next_viewpoint(self, obs):
        if not len(obs.cloud):
            return None
        if obs.arrived:
            if self.escape is not None:   # reached the escape pose: resume the line from here
                self.escape, self.goal = None, None
                return self.first_stop(obs)
            self.stops += 1
            self.goal = None
        if self.stops > 0:
            return None
        if self.escape is not None:
            return self.escape
        mask = np.asarray(obs.mask, bool)
        here = np.asarray(obs.position)[[0, 2]]
        model, info = _model(obs)
        offset = abs(wrap(G.bearing_yaw(here, model.center) - obs.yaw))
        if self.escapes < self.MAX_ESC and mask.sum() >= 200 and offset <= math.radians(20):
            occ_l, occ_r = occluder_sides(obs)
            if occ_l + occ_r > self.TRIGGER:
                esc = self.pick(obs, model, info, here, occ_l, occ_r)
                self.escapes += 1
                if esc is not None:
                    self.escape = esc
                    return esc
        return self.first_stop(obs)

    def pick(self, obs, model, info, here, occ_l, occ_r):
        b0 = bearing_of(model.center, here)
        r0 = fit_radius(obs, model, info, unit(here - model.center))
        right = np.array([math.cos(obs.yaw), -math.sin(obs.yaw)])     # image-right direction in (x, z)
        sides = sorted([(occ_l, -1), (occ_r, 1)])                       # less occluded first; -1 = left
        for _, s in sides:
            best = None
            for sep in (30.0, 60.0):
                for sign in (1, -1):
                    b = b0 + sign * math.radians(sep)
                    for r in (r0, r0 + 0.5):
                        xz = ring_pose(model.center, b, r)
                        if np.sign((xz - here) @ right) != s:
                            continue
                        blocked, unk = sight_lines(obs, model, xz)
                        if blocked > 0.2:
                            continue
                        cost = path_actions(obs, xz) + 5 * unk
                        if np.isfinite(cost) and cost <= obs.remaining - 4 and (best is None or cost < best[0]):
                            best = (cost, xz, sep * sign, r)
            if best is not None:
                return vp3(obs, best[1], model.center, {"escape": best[2], "r": round(best[3], 2),
                                                        "occ": [round(occ_l, 3), round(occ_r, 3)]})
        return None


class ClearCone(Closer):
    """D7: ring pose (bearings within 90 deg of the current one, r = fit .. fit + 1) minimising actions +
    W_LOS x blocked sight lines + W_UNK x unknown on them; re-chosen every 4 steps."""
    name, W_LOS, W_UNK = "ccs", 20.0, 4.0

    def line_goal(self, obs):
        model, info = _model(obs)
        here = np.asarray(obs.position)[[0, 2]]
        b0 = bearing_of(model.center, here)
        r0 = fit_radius(obs, model, info, unit(here - model.center))
        best = None
        for db in (0, 30, -30, 60, -60, 90, -90):
            for dr in (0.0, 0.5, 1.0):
                r = r0 + dr
                xz = ring_pose(model.center, b0 + math.radians(db), r)
                if db == 0 and np.linalg.norm(here - model.center) <= r + 0.15:
                    xz = here
                blocked, unk = sight_lines(obs, model, xz)
                cost = path_actions(obs, xz) + self.W_LOS * blocked + self.W_UNK * unk + 2 * dr
                if np.isfinite(cost) and (best is None or cost < best[0]):
                    best = (cost, xz, db, r, blocked)
        if best is None:
            return super().line_goal(obs)
        return vp3(obs, best[1], model.center, {"db": best[2], "r": round(best[3], 2), "blocked": best[4],
                                                "S": round(best[0], 2)})


POLICIES_SC = {"closer": Closer, "closer_rc": CloserArc, "tvd": TwoView, "irc": ImageReflex,
               "oes": OcclusionEscape, "ccs": ClearCone}


def _register():
    from active_approach.round3 import Seek
    out = dict(POLICIES_SC)
    for n, cls in POLICIES_SC.items():
        out[n + "_seek"] = (lambda n, cls: lambda: Seek(cls(), n + "_seek"))(n, cls)
    return out
