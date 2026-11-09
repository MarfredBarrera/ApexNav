"""Approach policies: (agent pose, object observation, maps) -> next viewpoint.

A policy implements

    reset(ctx)               once per run; ctx: scenario, pathfinder, views (GT ring, oracle only), sim
    next_viewpoint(obs) -> Viewpoint | None

``obs`` (Observation): the agent pose, the frame (rgb, depth, object mask), ``cloud`` (the object's world
points fused from every mask so far, 5 cm voxels; what ApexNav's object cluster holds), ``hmap`` (the
local 2.5D height map from every depth frame, geom.HeightMap), the navmesh ``pathfinder``, ``yaw0`` (the
start yaw: every reachable yaw is yaw0 + k 30 deg), ``judged`` (the judged views so far: camera x/z,
top_clipped), ``remaining`` (actions left) and ``arrived`` (the controller reached the policy's last
viewpoint, the agent faces the object and the judge was still unsure there). None ends the approach
(the runner then makes the forced decision). The shared Controller turns a viewpoint into discrete
actions (move_forward 0.25 m, turn 30 deg; greedy geodesic path on the navmesh, then a face turn).

Baselines:
    stay         never moves: forced decision on the start frame (the identity control)
    native       ApexNav's approach: 1.5 m standoff on the line of sight to the cluster centroid
                 (re-planned every step), then the close point (0.75 m), face turn, forced decision
    random_ring  a uniformly random lattice-aligned viewpoint at 1-2.5 m from the centre, twice
    oracle       GT: the nearest (geodesic) ring viewpoint whose GT view is clearly sufficient; up to 3
Planners (docs/active_approach_planners.md, run-time inputs only):
    fba          M1 fit-band approach: the lattice viewpoint on the current bearing at d_fit (kappa 0.25),
                 re-estimated every 4 steps from the growing cloud
    svr          M2a sure-view ring: argmax q (0.5 + 0.5 nov) - A/60 over 12 lattice yaws x 3 radii,
                 q = (1 - trunc_hat)(1 - occ_hat) min(1, area_hat / 0.04); "nothing to gain" -> None
    *_tour       M3 wrapper: after the base viewpoint, up to K=3 views in total, each next view
                 argmax q nov - A/60 with >= 45 deg bearing separation and A <= the remaining budget
"""
import math
import zlib
from dataclasses import dataclass, field

import numpy as np

from active_approach import geom as G
from active_approach import scene as S
from active_approach.judges import gt_sufficient

GOAL_RADIUS, FACE_TOLERANCE_DEG = 0.25, 15.0
A0 = 60.0                       # actions that cost a whole unit of view utility


@dataclass
class Viewpoint:
    position: np.ndarray    # on the navmesh
    yaw: float              # rad, 0 = -z
    info: dict = field(default_factory=dict)
    plan: list = None       # an explicit action sequence to get there (S2); the controller executes it first


@dataclass
class Observation:
    step: int
    position: np.ndarray
    yaw: float
    rgb: np.ndarray
    depth: np.ndarray
    mask: np.ndarray
    cloud: np.ndarray
    pathfinder: object
    hmap: object = None
    yaw0: float = 0.0
    floor_y: float = 0.0
    judged: list = field(default_factory=list)
    remaining: int = 40
    arrived: bool = False
    meta: dict = field(default_factory=dict)   # target, the trigger frame's verdict (JEV distribution)


def facing(position, target):
    d = np.asarray(target, float) - np.asarray(position, float)
    return math.atan2(-d[0], -d[2])


def wrap(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi


def geodesic(pathfinder, a, b):
    import habitat_sim
    path = getattr(pathfinder, "ShortestPath", habitat_sim.ShortestPath)()   # local_nav.LocalPathfinder: own class
    path.requested_start, path.requested_end = np.asarray(a, np.float32), np.asarray(b, np.float32)
    return path.geodesic_distance if pathfinder.find_path(path) else math.inf


class Controller:
    """Viewpoint -> next discrete action (None: arrived and facing). Non-holonomic: forward and turns only."""

    def __init__(self, sim):
        import habitat_sim
        self.sim = sim
        self.follower = habitat_sim.nav.GreedyGeodesicFollower(
            sim.pathfinder, sim.get_agent(0), goal_radius=GOAL_RADIUS,
            forward_key="move_forward", left_key="turn_left", right_key="turn_right")
        self.greedy = 0     # actions left that the greedy follower takes alone (after a blocked step)

    LOOKAHEAD, HYSTERESIS = 0.5, math.radians(25)

    def action(self, viewpoint, obs=None):
        """An explicit ``viewpoint.plan`` is executed first. Else lattice line-following: head for the geodesic path point ``LOOKAHEAD`` ahead; turn only when the
        heading is off by more than ``HYSTERESIS`` (a 30 deg lattice cannot follow an arbitrary direction, and
        a greedy follower zig-zags with a turn every step); otherwise step forward if the step is free. Near
        the goal, or when the step is blocked, habitat's greedy geodesic follower decides."""
        import habitat_sim
        if viewpoint.plan:
            return viewpoint.plan.pop(0)
        position, yaw, _ = S.agent_pose(self.sim)
        distance = np.hypot(*(viewpoint.position - position)[[0, 2]])
        if distance > GOAL_RADIUS:
            pf = self.sim.pathfinder
            path = habitat_sim.ShortestPath()
            path.requested_start = np.asarray(position, np.float32)
            path.requested_end = np.asarray(viewpoint.position, np.float32)
            if self.greedy > 0:
                self.greedy -= 1
            elif distance > 2 * self.LOOKAHEAD and pf.find_path(path) and len(path.points) > 1:
                pts = [np.asarray(p, float) for p in path.points]
                walked, ahead = 0.0, pts[-1]
                for a, b in zip(pts, pts[1:]):
                    seg = np.hypot(*(b - a)[[0, 2]])
                    if walked + seg >= self.LOOKAHEAD:
                        ahead = a + (b - a) * (self.LOOKAHEAD - walked) / max(seg, 1e-6)
                        break
                    walked += seg
                diff = wrap(math.atan2(-(ahead - position)[0], -(ahead - position)[2]) - yaw)
                if abs(diff) > self.HYSTERESIS:
                    return "turn_left" if diff > 0 else "turn_right"
                step = position + S.FORWARD_M * np.array([-math.sin(yaw), 0.0, -math.cos(yaw)])
                landed = np.asarray(pf.try_step_no_sliding(np.asarray(position, np.float32),
                                                           np.asarray(step, np.float32)), float)
                if np.hypot(*(landed - position)[[0, 2]]) > 0.8 * S.FORWARD_M:
                    return "move_forward"
                self.greedy = 4   # blocked: let the greedy follower get round the corner (no L/R dithering)
            action = self.follower.next_action_along(np.asarray(viewpoint.position, np.float32))
            if action is not None:
                return action
        diff = wrap(viewpoint.yaw - yaw)
        if abs(math.degrees(diff)) > FACE_TOLERANCE_DEG:
            return "turn_left" if diff > 0 else "turn_right"
        return None


def ring_views(sim, surface, object_id, floor_y, origin,
               radii=(0.75, 1.0, 1.5, 2.0, 2.5, 3.0), bearings=24):
    """GT-scored viewpoints on rings around the object (camera level, facing its centre).

    [{position, yaw, radius, bearing_deg, geodesic (from origin), vis, sufficient}];
    the agent's state is restored afterwards.
    """
    agent = sim.get_agent(0)
    saved = agent.get_state()
    pf, center, views, seen = sim.pathfinder, surface.center, [], set()
    for radius in radii:
        for k in range(bearings):
            angle = 2 * math.pi * k / bearings
            want = np.array([center[0] + radius * math.cos(angle), floor_y, center[2] + radius * math.sin(angle)])
            q = np.asarray(pf.snap_point(want), float)
            if not np.all(np.isfinite(q)) or np.hypot(*(q - want)[[0, 2]]) > 0.2 or abs(q[1] - floor_y) > 0.3:
                continue
            key = tuple(np.round(q, 2))
            if key in seen:
                continue
            seen.add(key)
            dist = geodesic(pf, origin, q)
            if not math.isfinite(dist):
                continue
            yaw = facing(q, center)
            obs = S.set_pose(sim, q, yaw)
            vis = S.visibility(obs, surface, object_id, *S.camera(sim))
            views.append({"position": q.tolist(), "yaw": yaw, "radius": radius,
                          "bearing_deg": round(math.degrees(angle)), "geodesic": dist,
                          "vis": vis, "sufficient": gt_sufficient(vis)})
    agent.set_state(saved, reset_sensors=True)
    return views


def view_score(vis):
    return vis["vis_frac"] * (1 - vis["trunc_frac"]) * min(1.0, vis["area"] / 0.01)


def view_bearing(camera_xz, center_xz):
    """Direction object centre -> camera (rad)."""
    d = np.asarray(camera_xz, float) - np.asarray(center_xz, float)
    return math.atan2(d[1], d[0])


def novelty(position_xz, center_xz, judged):
    if not judged:
        return 1.0
    b = view_bearing(position_xz, center_xz)
    return min(1.0, min(abs(wrap(b - view_bearing(j["camera"], center_xz))) for j in judged) / math.radians(60))


def quality(p):
    """M2a: GT-free view utility from predicted statistics."""
    return (1 - p["trunc"]) * (1 - p["occ"]) * min(1.0, p["area"] / 0.04)


class Stay:
    name = "stay"

    def reset(self, ctx):
        pass

    def next_viewpoint(self, obs):
        return None


class Native:
    name = "native"
    STANDOFF, CLOSE = 1.5, 0.75

    def reset(self, ctx):
        self.phase = 0      # 0: to the standoff, 1: to the close approach point, 2: done

    def next_viewpoint(self, obs):
        if not len(obs.cloud):
            return None
        center = obs.cloud.mean(0)
        if obs.arrived:
            self.phase += 1
        here = np.hypot(*(obs.position - center)[[0, 2]])
        if self.phase == 0 and here <= self.STANDOFF + GOAL_RADIUS:
            self.phase = 1
        if self.phase >= 2:
            return None
        radius = self.STANDOFF if self.phase == 0 else self.CLOSE
        bearing = (obs.position - center)[[0, 2]]
        bearing = bearing / max(np.linalg.norm(bearing), 1e-6)
        want = np.array([center[0] + radius * bearing[0], obs.position[1], center[2] + radius * bearing[1]])
        goal = np.asarray(obs.pathfinder.snap_point(want), float)
        if not np.all(np.isfinite(goal)):
            return None
        return Viewpoint(goal, facing(goal, center), {"phase": self.phase})


class Oracle:
    name = "oracle"
    TRIES = 3   # arrival is approximate (0.25 m, 15 deg): if a view still fails, take the next one

    def reset(self, ctx):
        views = ctx["views"]
        # robust first: a view just inside the GT thresholds often ends just outside them
        robust = [v for v in views if v["sufficient"] and v["vis"]["vis_frac"] >= 0.7
                  and v["vis"]["trunc_frac"] <= 0.05 and v["vis"]["area"] >= 0.015]
        rest = [v for v in views if v["sufficient"] and v not in robust]
        ranked = sorted(robust, key=lambda v: v["geodesic"]) + sorted(rest, key=lambda v: v["geodesic"])
        if not ranked and views:
            ranked = [max(views, key=lambda v: view_score(v["vis"]))]
        self.queue = [Viewpoint(np.asarray(v["position"]), v["yaw"]) for v in ranked[:self.TRIES]]

    def next_viewpoint(self, obs):
        if obs.arrived and self.queue:
            self.queue.pop(0)
        return self.queue[0] if self.queue else None


class RandomRing:
    """Uniformly random lattice-aligned viewpoints at 1-2.5 m from the cloud centre, ``VIEWS`` of them."""
    name = "random_ring"
    VIEWS = 2

    def reset(self, ctx):
        self.rng = np.random.default_rng(zlib.crc32(ctx["scenario"]["id"].encode()))
        self.goal, self.done = None, 0

    def next_viewpoint(self, obs):
        if obs.arrived:
            self.done, self.goal = self.done + 1, None
        if self.done >= self.VIEWS or not len(obs.cloud):
            return None
        if self.goal is None:
            center = G.ObjectModel(obs.cloud, obs.floor_y).center
            for k in self.rng.permutation(12):
                yaw = wrap(obs.yaw0 + k * G.LATTICE)
                r = self.rng.uniform(1.0, 2.5)
                xz = center - r * G.forward(yaw)
                want = np.array([xz[0], obs.floor_y, xz[1]], np.float32)
                q = np.asarray(obs.pathfinder.snap_point(want), float)
                if (np.all(np.isfinite(q)) and np.hypot(*(q - want)[[0, 2]]) <= 0.15
                        and math.isfinite(geodesic(obs.pathfinder, obs.position, q))):
                    self.goal = Viewpoint(q, yaw, {"k": int(k), "r": round(r, 2)})
                    break
            else:
                return None
        return self.goal


def _clipped(obs):
    """{top_clipped, side_clipped}: did the object run off the top / a side of any judged frame."""
    return {"top_clipped": any(j.get("top_clipped") for j in obs.judged),
            "side_clipped": any(j.get("side_clipped") for j in obs.judged)}


def _scored(obs, candidates, sep_deg=None, remaining=None):
    """Score lattice candidates (M2a): [(S, q, nov, A, pred, cand)], best first; with ``sep_deg`` (M3) S is
    q nov - A/A0 and candidates closer than ``sep_deg`` in bearing to a judged view are left out."""
    model = G.ObjectModel(obs.cloud, obs.floor_y)
    predictor = G.Predictor(model, obs.cloud, obs.hmap, obs.pathfinder)
    out = []
    for c in candidates:
        xz = c["position"][[0, 2]]
        nov = novelty(xz, model.center, obs.judged)
        if sep_deg is not None and obs.judged:
            b = view_bearing(xz, model.center)
            if min(abs(wrap(b - view_bearing(j["camera"], model.center))) for j in obs.judged) < math.radians(sep_deg):
                continue
        a = G.action_cost(obs.pathfinder, obs.position, obs.yaw, c["position"], c["yaw"])
        if not math.isfinite(a) or (remaining is not None and a > remaining):
            continue
        pred = predictor.predict(xz, c["yaw"])
        q = quality(pred)
        utility = q * nov if sep_deg is not None else q * (0.5 + 0.5 * nov)
        out.append((utility - a / A0, q, nov, a, pred, c))
    out.sort(key=lambda t: -t[0])
    return out, model, predictor


def _vp(entry, reason):
    s, q, nov, a, pred, c = entry
    return Viewpoint(c["position"], c["yaw"], {"reason": reason, "S": round(s, 3), "q": round(q, 3),
                                               "nov": round(nov, 2), "A": a, "k": c["k"], "r": round(c["r"], 2),
                                               "pred": {k: round(v, 3) for k, v in pred.items()}})


class FitBand:
    """M1: the lattice viewpoint on the current bearing at d_fit; re-estimated every 4 steps."""
    name = "fba"
    REPLAN = 4

    def reset(self, ctx):
        self.goal, self.planned = None, -999

    def next_viewpoint(self, obs):
        if obs.arrived:
            return None
        if not len(obs.cloud) or (self.goal is not None and obs.step - self.planned < self.REPLAN):
            return self.goal
        self.planned = obs.step
        model = G.ObjectModel(obs.cloud, obs.floor_y)
        cands = G.lattice_candidates(model, obs.yaw0, obs.pathfinder, obs.floor_y,
                                     **_clipped(obs))
        here = G.bearing_yaw(obs.position[[0, 2]], model.center)
        k_here = round(wrap(here - obs.yaw0) / G.LATTICE) % 12
        cands = [c for c in cands if math.isfinite(geodesic(obs.pathfinder, obs.position, c["position"]))]
        if not cands:
            return self.goal
        best = min(cands, key=lambda c: (min((c["k"] - k_here) % 12, (k_here - c["k"]) % 12), c["r"]))
        new = Viewpoint(best["position"], best["yaw"], {"k": best["k"], "r": round(best["r"], 2),
                                                        "d_fit": round(best["d_fit"], 2)})
        if self.goal is None or np.hypot(*(new.position - self.goal.position)[[0, 2]]) > 0.25:
            self.goal = new
        return self.goal


class SureViewRing:
    """M2a: the best lattice candidate by q (0.5 + 0.5 nov) - A/60; replan every 5 steps (switch on +0.1)."""
    name = "svr"
    REPLAN, SWITCH, NOTHING = 5, 0.1, 0.05

    def reset(self, ctx):
        self.goal, self.score, self.planned, self.reason = None, None, -999, None

    def next_viewpoint(self, obs):
        if obs.arrived:
            return None
        if not len(obs.cloud) or (self.goal is not None and obs.step - self.planned < self.REPLAN):
            return self.goal
        self.planned = obs.step
        model = G.ObjectModel(obs.cloud, obs.floor_y)
        cands = G.lattice_candidates(model, obs.yaw0, obs.pathfinder, obs.floor_y,
                                     **_clipped(obs))
        scored, model, predictor = _scored(obs, cands)
        if not scored:
            if self.goal is None:
                self.reason = "no_candidate"
            return self.goal
        best = scored[0]
        if self.goal is None:
            current = quality(predictor.predict(obs.position[[0, 2]], obs.yaw)) * 0.5
            if obs.judged and (obs.judged[-1].get("top_clipped") or obs.judged[-1].get("side_clipped")):
                current *= 0.5   # the cloud holds only the part in frame: a clipped view predicts no truncation
            if current >= best[0] - self.NOTHING:
                self.reason = "nothing_to_gain"
                return None
            self.goal, self.score = _vp(best, "plan"), best[0]
        elif best[0] > self.score + self.SWITCH and np.hypot(*(best[5]["position"] - self.goal.position)[[0, 2]]) > 0.25:
            self.goal, self.score = _vp(best, "switch"), best[0]
        return self.goal


class Tour:
    """M3: the base policy's viewpoint, then more views (>= 45 deg apart) up to K stops in total."""
    K, SEPARATION, SLACK = 3, 45.0, 1.25    # SLACK: executed legs take up to ~25% more actions than A

    def __init__(self, base, name=None, driver=None, enroute=None):
        self.base = base
        self.name = name or f"{base.name}_tour"
        self.driver = driver                      # None: the Controller; ("mpc", s): svpto.MPCDriver
        self.enroute = getattr(base, "enroute", False) if enroute is None else enroute

    @property
    def reason(self):
        return self._reason or getattr(self.base, "reason", None)

    def reset(self, ctx):
        self.base.reset(ctx)
        self.phase, self.stops, self.goal, self._reason = "base", 0, None, None

    def next_viewpoint(self, obs):
        if obs.arrived:
            self.stops += 1
            self.goal = None
        if self.phase == "base":
            vp = self.base.next_viewpoint(obs)
            if vp is not None:
                return vp
            self.phase = "tour"
        if self.stops >= self.K or not len(obs.cloud):
            return None
        if self.goal is None:
            model = G.ObjectModel(obs.cloud, obs.floor_y)
            cands = G.lattice_candidates(model, obs.yaw0, obs.pathfinder, obs.floor_y, extra=(0.0, 0.25),
                                         **_clipped(obs))
            scored, _, _ = _scored(obs, cands, sep_deg=self.SEPARATION, remaining=obs.remaining / self.SLACK)
            scored = [e for e in scored if e[0] > 0]   # worth its actions
            if not scored:
                self._reason = "no_candidate"
                return None
            self.goal = _vp(scored[0], f"tour{self.stops + 1}")
        return self.goal


def _svpto(kind):
    """S2 (+ M3 views) and its ablations, S3 (M2 + M3 driven by the visibility MPC) and its s = 0 ablation."""
    from active_approach import svpto
    variants = {"s2": {}, "s2_nofm": {"w_fm": 0.0}, "s2_wedge": {"wedge": True}}
    if kind in variants:
        return Tour(svpto.FirstClearView(**variants[kind]), name=kind)
    return Tour(SureViewRing(), name=kind, driver=("mpc", None if kind == "s3" else 0.0), enroute=True)


POLICIES = {"stay": Stay, "native": Native, "oracle": Oracle, "random_ring": RandomRing, "fba": FitBand,
            "svr": SureViewRing, "fba_tour": lambda: Tour(FitBand()), "svr_tour": lambda: Tour(SureViewRing()),
            "s2": lambda: _svpto("s2"), "s2_nofm": lambda: _svpto("s2_nofm"), "s2_wedge": lambda: _svpto("s2_wedge"),
            "s3": lambda: _svpto("s3"), "s3_abl": lambda: _svpto("s3_abl")}


def _round3():
    from active_approach.round3 import POLICIES3
    return POLICIES3


for _name in ("fba_tour3", "svr_tour3", "s2_tour3", "fba_tour3_seek", "svr_tour3_seek", "s2_tour3_seek",
              "fba_tour3_seekc", "svr_tour3_seekc", "s2_tour3_seekc", "jev_pilot"):
    POLICIES[_name] = (lambda n: lambda: _round3()[n]())(_name)


def _selfcontained():
    from active_approach.selfcontained import _register
    return _register()


for _name in ("closer", "closer_rc", "tvd", "irc", "oes", "ccs"):
    for _suffix in ("", "_seek"):
        POLICIES[_name + _suffix] = (lambda n: lambda: _selfcontained()[n]())(_name + _suffix)


def _mppi():
    from active_approach.mppi import POLICIES_MPPI
    return POLICIES_MPPI


for _name in ("mppi", "mppi_k0", "mppi_k4", "mppi_dial", "mppi_v2", "mppi_v3", "mppi_v4", "mppi_v5", "mppi_v6", "mppi_v7", "mppi_v8", "mppi_v9", "mppi_v10", "mppi_v11", "mppi_v15"):
    POLICIES[_name] = (lambda n: lambda: _mppi()[n]())(_name)
