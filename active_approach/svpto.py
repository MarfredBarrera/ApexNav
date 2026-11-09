"""SVPTO-inspired planners (docs/active_approach_planners.md section 4, revised with the paper: Wang et al.,
IEEE TIE 71(3) 2024). Run-time inputs only. Every term and parameter is in ``PARAMS``.

``Visibility`` scores a lattice pose (x, z, heading k; yaw = yaw0 + k 30 deg); costs in action units:
    FM  (eq. 3, lateral) sum over 3 sight lines (object centre and its two horizontal extremes) and
        i = 1..N_h-1 of p(l_i / gamma(m_i) - 1), m_i = p + (i/N_h) delta, l_i = (i/N_h)|delta| tan beta_FM,
        beta_FM = min(beta_h, atan((W/2 + 0.15)/|delta|) + 5 deg) (ablation "wedge": beta_h); gamma = ESDF of
        the height-map slice nearest the ray height (0.3/0.6/0.9/1.2 m), footprint +0.15 m excluded,
        unknown non-navigable cells at half weight
    EC  (eq. 4, longitudinal) p(od_min - d)^2 + p(d - rd_max)^2, d = distance to the nearest object point,
        od_min = d_fit, od_max = d_fit + 1.0, rd_max = (zeta_max - zeta)/zeta_max (od_max - od_min) + od_min;
        zeta = sum_j eta_j S_j over 4 depth bins of the current frame's 5 cm voxels (non-floor, non-object,
        0.1-1.8 m high, out to min(5 m, object + 0.5 m)), eta = 1, .75, .5, .25, EMA 0.5 over replans,
        zeta_max frozen (95th percentile over the 517 start frames, ``zeta-max``)
    OV  (eq. 5, re-aimed) after a forward step: p(cos beta_h - cos angle(heading, delta))
    C   safety p(0.25 - clearance)^2 / 0.25^2 (navmesh), w 10
J_vis = s (0.1 FM + 10 EC + 5 OV) (the paper's weight ratios). A pose is valid when EC = OV = 0 and the three
sight lines are clear (PARAMS valid_fm "los"; the design's FM = 0 is never met indoors, see PARAMS).
The lattice simulator steps with ``pathfinder.try_step_no_sliding`` (allow_sliding false) and exact 30 deg turns.

    S2  ``FirstClearView``: A* over lattice states to the valid poses whose proxy P_unsure is at least 0.1 below
        the current one, the last 4 poses before the goal valid; edge cost 1 + J_vis + C; heuristic
        max(0, |p - c| - rd_max)/0.25 + turns to bring the object within beta_h. The plan's actions are
        executed (Viewpoint.plan); replanned every 4 steps when FM on the remaining path changed.
        No goal within the cap -> None (forced decision, or M3's next view).
    S3  ``MPCDriver``: drives to M2/M3 viewpoints by a beam search over 8 actions (beam 64, discount 0.9) of
        1 + J_vis + C per step plus the remaining actions to the goal; executes 2 per plan; within 0.75 m the
        plain controller finishes. s = 0 is the ablation (M2 + M3, MPC-driven).

Proxy (round 1, refit without category offsets, which mixed category with real/synthetic naming; labels from
the 0.4664 gate): logit P_unsure = -1.282 - 1.416 log10(area) + 0.326 occ + 0.287 trunc
+ 0.279 max(0, trunc - 0.5) - 0.253 facing - 0.239 min(d, 6), on predicted statistics (geom.Predictor).

    python -m active_approach.svpto zeta-max --scenarios .../scenarios.jsonl --gpu G   # freeze zeta_max
"""
import heapq
import json
import math
import os

import numpy as np

from active_approach import geom as G
from active_approach import scene as S

PARAMS = {
    # FM (eq. 3)
    "n_h": 10, "fm_pad": 0.15, "fm_margin_deg": 5.0, "wedge": False,
    "slices": (0.3, 0.6, 0.9, 1.2), "near_object": 0.15, "unknown_weight": 0.5,
    # validity: FM == 0 ("zero", the design) is never met indoors (min FM over in-band lattice poses 3-8 on the
    # dry-run scenes: something always enters the cone); "los" = the three sight lines themselves are clear
    # (ESDF >= los_clear at every sample), FM stays a soft cost
    "valid_fm": "los", "los_clear": 0.05,
    # EC (eq. 4)
    "band_width": 1.0, "zeta_bins": 4, "zeta_z": (0.1, 1.8), "zeta_range": 5.0, "zeta_ema": 0.5,
    "zeta_max_file": "videos/active_approach/zeta_max.json",
    # weights (paper ratios lambda_FM 0.1, lambda_EC 10, lambda_OV 5), scale s
    "s": 1.0, "w_fm": 0.1, "w_ec": 10.0, "w_ov": 5.0,
    # safety
    "d_thr": 0.25, "w_c": 10.0,
    # S2
    "s2_gain": 0.1, "s2_last_valid": 4, "s2_cap": 20000, "s2_box": 5.0, "s2_replan": 4,
    # S3 / MPC
    "mpc_n": 8, "mpc_beam": 64, "mpc_execute": 2, "mpc_discount": 0.9, "mpc_w_goal": 1.0, "mpc_near": 0.75,
    "mpc_progress": 0.5, "mpc_fallback": 6,   # progress guard (harness addition, not in the design)
}
P = PARAMS
BETA_H = G.BETA_H


def p_(x):
    return max(0.0, x)


def p_unsure(pred):
    a = max(pred["area"], 1e-4)
    z = (-1.282 - 1.416 * math.log10(a) + 0.326 * pred["occ"] + 0.287 * pred["trunc"]
         + 0.279 * max(0.0, pred["trunc"] - 0.5) - 0.253 * pred["facing"] - 0.239 * min(pred["distance"], 6.0))
    return 1 / (1 + math.exp(-z))


def navigable_grid(hmap, pathfinder):
    if hasattr(pathfinder, "nav_grid"):   # local_nav.LocalPathfinder: the self-built map changes every step
        return pathfinder.nav_grid(hmap)
    if getattr(hmap, "nav", None) is None:
        ii, jj = np.meshgrid(np.arange(hmap.n), np.arange(hmap.n), indexing="ij")
        xz = hmap.origin + (np.stack([ii, jj], -1) + 0.5) * G.CELL
        nav = np.zeros((hmap.n, hmap.n), bool)
        for i in range(hmap.n):
            for j in range(hmap.n):
                nav[i, j] = pathfinder.is_navigable(np.array([xz[i, j, 0], hmap.floor_y, xz[i, j, 1]], np.float32), 0.5)
        hmap.nav = nav
    return hmap.nav


def yaw_rotation(yaw):
    """Camera -> world rotation of a level camera at ``yaw`` (scene.yaw_quat)."""
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def zeta(depth, mask, position, yaw, floor_y, d_object):
    """SVPTO eq. 4 complexity of one level frame: sum_j eta_j S_j over ``zeta_bins`` equal depth bins of the
    5 cm voxels of non-object points 0.1-1.8 m above the floor, out to min(5 m, d_object + 0.5)."""
    cam = np.array([position[0], floor_y + S.CAMERA_HEIGHT, position[2]])
    reach = min(P["zeta_range"], (d_object or P["zeta_range"]) + 0.5)
    valid = (depth > 0.05) & (depth < reach) & ~mask
    pts = S.backproject(depth, valid, cam, yaw_rotation(yaw), stride=2)
    h = pts[:, 1] - floor_y
    pts = pts[(h >= P["zeta_z"][0]) & (h <= P["zeta_z"][1])]
    if not len(pts):
        return 0.0
    vox = np.unique(np.floor(pts / 0.05).astype(np.int64), axis=0)
    rng = np.hypot((vox[:, 0] + 0.5) * 0.05 - cam[0], (vox[:, 2] + 0.5) * 0.05 - cam[2])
    n = P["zeta_bins"]
    j = np.clip((rng / reach * n).astype(int), 0, n - 1)
    eta = (n - np.arange(n)) / n
    return float(sum(eta[b] * (j == b).sum() for b in range(n)))


_ZETA_MAX = None


def zeta_max():
    global _ZETA_MAX
    if _ZETA_MAX is None:
        with open(P["zeta_max_file"]) as stream:
            _ZETA_MAX = float(json.load(stream)["zeta_max"])
    return _ZETA_MAX


class Visibility:
    """Per-pose SVPTO terms from the run-time map; ``s``, ``w_fm``, ``wedge`` override PARAMS (ablations)."""

    def __init__(self, obs, s=None, w_fm=None, wedge=None, keep_zeta=False, model=None):
        from scipy.ndimage import distance_transform_edt
        self.obs, self.pf, self.hmap = obs, obs.pathfinder, obs.hmap
        self.model = model or G.ObjectModel(obs.cloud, obs.floor_y)
        self.s = P["s"] if s is None else s
        self.w_fm = P["w_fm"] if w_fm is None else w_fm
        self.wedge = P["wedge"] if wedge is None else wedge
        hm = self.hmap
        near = G._near_object(self.model, hm, P["near_object"])
        nav = navigable_grid(hm, self.pf)
        self.esdf_unknown = distance_transform_edt(~(~hm.seen & ~nav & ~near)) * G.CELL
        self.esdf = {z: distance_transform_edt(~(hm.seen & (hm.top >= z) & ~near)) * G.CELL for z in P["slices"]}
        self.mid = (self.model.z_lo + self.model.z_hi) / 2
        clipped = {"top_clipped": any(j.get("top_clipped") for j in obs.judged),
                   "side_clipped": any(j.get("side_clipped") for j in obs.judged)}
        if model is not None:  # a phantom model (round3) already holds the unknown extent
            clipped = {}
        self.fit = [G.d_fit(self.model, G.wrap(obs.yaw0 + k * G.LATTICE), 0.25, **clipped) for k in range(12)]
        # zeta from the current frame, EMA over replans (kept on the height map, one per run)
        d_obj = float(np.min(np.linalg.norm(self.model.xz - obs.position[[0, 2]], axis=1))) if len(self.model.xz) else None
        z_now = zeta(obs.depth, obs.mask, obs.position, obs.yaw, obs.floor_y, d_obj) if obs.depth is not None else 0.0
        previous = getattr(hm, "zeta_ema", None)
        ema = z_now if previous is None else P["zeta_ema"] * z_now + (1 - P["zeta_ema"]) * previous
        if not keep_zeta:  # a planner's replan updates the EMA; logging (keep_zeta) does not
            hm.zeta_ema = ema
        self.zeta = min(ema, zeta_max())
        self.rd_scale = (zeta_max() - self.zeta) / zeta_max() if zeta_max() > 0 else 1.0
        self.sub = self.model.xz[::max(1, len(self.model.xz) // 200)]
        self.cache = {}
        self.los_max = 1.0

    def yaw(self, k):
        return G.wrap(self.obs.yaw0 + k * G.LATTICE)

    def band(self, k):
        od_min = self.fit[k]
        return od_min, od_min + self.rd_scale * P["band_width"]

    def _esdf(self, table, xz):
        ij, ok = self.hmap.cells(np.asarray(xz)[None])
        return float(table[ij[0, 0], ij[0, 1]]) if ok[0] else 10.0

    def terms(self, position, k):
        """{FM, EC, OV, C, valid, d}; OV is the in-view hinge (the caller applies it after forward steps)."""
        key = (round(position[0] / 0.05), round(position[2] / 0.05), k)
        if key in self.cache:
            return self.cache[key]
        xz = np.asarray(position)[[0, 2]]
        c = self.model.center
        delta = c - xz
        dist = max(float(np.linalg.norm(delta)), 1e-6)
        u = delta / dist
        perp = np.array([-u[1], u[0]])
        half_w = self.model.half_extent(perp)
        # FM (eq. 3): three sight lines, cone half-width from the (capped) FOV angle
        beta = BETA_H if self.wedge else min(BETA_H, math.atan((half_w + P["fm_pad"]) / dist)
                                             + math.radians(P["fm_margin_deg"]))
        fm, n, blocked = 0.0, P["n_h"], False
        for target in (c, c + half_w * perp, c - half_w * perp):
            ray = target - xz
            length = float(np.linalg.norm(ray))
            for i in range(1, n):
                lam = i / n
                m = xz + lam * ray
                l_i = lam * length * math.tan(beta)
                z = S.CAMERA_HEIGHT + lam * (self.mid - S.CAMERA_HEIGHT)
                zs = min(P["slices"], key=lambda s: abs(s - z))
                e, eu = self._esdf(self.esdf[zs], m), self._esdf(self.esdf_unknown, m)
                fm += p_(l_i / max(e, 0.01) - 1) + P["unknown_weight"] * p_(l_i / max(eu, 0.01) - 1)
                if lam <= self.los_max:
                    blocked |= min(e, eu) < P["los_clear"]
        # EC (eq. 4)
        d = float(np.min(np.linalg.norm(self.sub - xz, axis=1)))
        od_min, rd_max = self.band(k)
        ec = p_(od_min - d) ** 2 + p_(d - rd_max) ** 2
        # OV (eq. 5, re-aimed): the object in the FOV of the heading
        yaw = self.yaw(k)
        heading = np.array([-math.sin(yaw), -math.cos(yaw)])
        ov = p_(math.cos(BETA_H) - float(heading @ u))
        clear = float(self.pf.distance_to_closest_obstacle(np.asarray(position, np.float32), 1.0))
        cc = p_(P["d_thr"] - clear) ** 2 / P["d_thr"] ** 2
        clear_view = fm == 0 if P["valid_fm"] == "zero" else not blocked
        out = {"FM": fm, "EC": ec, "OV": ov, "C": cc, "LOS": not blocked, "valid": clear_view and ec == 0 and ov == 0,
               "d": d}
        self.cache[key] = out
        return out

    def cost(self, t, forward=True):
        """Action cost of arriving at a pose with terms ``t``: 1 + J_vis + safety."""
        vis = self.w_fm * t["FM"] + P["w_ec"] * t["EC"] + (P["w_ov"] * t["OV"] if forward else 0.0)
        return 1 + self.s * vis + P["w_c"] * t["C"]


def step(pathfinder, position, k, action, yaw0):
    """The lattice simulator: (position, k) after ``action``; None when a forward step is blocked."""
    if action == "turn_left":
        return position, (k + 1) % 12
    if action == "turn_right":
        return position, (k - 1) % 12
    yaw = G.wrap(yaw0 + k * G.LATTICE)
    target = position + S.FORWARD_M * np.array([-math.sin(yaw), 0.0, -math.cos(yaw)])
    landed = np.asarray(pathfinder.try_step_no_sliding(np.asarray(position, np.float32),
                                                       np.asarray(target, np.float32)), float)
    if np.hypot(*(landed - position)[[0, 2]]) < 0.05:
        return None
    return landed, k


def heading_index(yaw, yaw0):
    return int(round(G.wrap(yaw - yaw0) / G.LATTICE)) % 12


ACTIONS = ("move_forward", "turn_left", "turn_right")


def _allowed(history, action):
    if history and {history[-1], action} == {"turn_left", "turn_right"}:
        return False
    if action != "move_forward" and len(history) >= 6 and all(a != "move_forward" for a in history[-6:]):
        return False
    return True


class FirstClearView:
    """S2 (a viewpoint policy whose Viewpoint carries the action plan); ``variant``: Visibility overrides
    (ablations: {"w_fm": 0} without the clearance term, {"wedge": True} the paper's full-FOV wedge)."""
    name = "s2"
    enroute = True

    def __init__(self, **variant):
        self.variant = variant

    def reset(self, ctx):
        self.goal, self.path, self.checked, self.reason, self.p0 = None, None, -999, None, None
        self.stats = {"plans": 0, "expansions": 0}

    def next_viewpoint(self, obs):
        if obs.arrived:
            return None
        if not len(obs.cloud):
            return self.goal
        if self.goal is not None and obs.step - self.checked < P['s2_replan']:
            return self.goal
        self.checked = obs.step
        vis = self._visibility(obs)
        if self.goal is not None and self.path:
            # replan only when FM on the remaining path changed
            if [vis.terms(p, k)["FM"] > 0 for p, k in self.path] == self.fm_path and self.goal.plan:
                return self.goal
        plan = self._search(obs, vis)
        if plan is None:
            if self.goal is None:
                self.reason = "nothing_to_gain"
                return None
            return self.goal
        actions, poses, info = plan
        self.path, self.fm_path = poses, [vis.terms(p, k)["FM"] > 0 for p, k in poses]
        end_pos, end_k = poses[-1] if poses else (obs.position, heading_index(obs.yaw, obs.yaw0))
        from active_approach.policies import Viewpoint
        self.goal = Viewpoint(np.asarray(end_pos), vis.yaw(end_k), info, plan=list(actions))
        return self.goal

    def _valid(self, vis, t):
        return t["valid"]

    def _search(self, obs, vis):
        pf, yaw0 = obs.pathfinder, obs.yaw0
        predictor = G.Predictor(vis.model, obs.cloud, obs.hmap, pf, n_points=96)
        if self.p0 is None:
            self.p0 = p_unsure(predictor.predict(obs.position[[0, 2]], obs.yaw))
        want = self.p0 - P["s2_gain"]
        start = (np.asarray(obs.position, float), heading_index(obs.yaw, yaw0))
        c = vis.model.center

        def h(p, k):
            far = max(0.0, float(np.linalg.norm(c - p[[0, 2]])) - vis.band(k)[1]) / S.FORWARD_M
            off = abs(G.wrap(G.bearing_yaw(p[[0, 2]], c) - vis.yaw(k)))
            return far + max(0, math.ceil((off - BETA_H) / G.LATTICE))

        def key(p, k):
            return (round(p[0] / 0.1), round(p[2] / 0.1), k)

        counter = 0
        frontier = [(h(*start), 0.0, counter, start, None)]
        nodes, closed, expansions = {}, set(), 0
        punsure = {}
        while frontier and expansions < P['s2_cap']:
            f, cost, _, (p, k), parent = heapq.heappop(frontier)
            kk = key(p, k)
            if kk in closed:
                continue
            closed.add(kk)
            nodes[kk] = (p, k, parent)
            expansions += 1
            t = vis.terms(p, k)
            if self._valid(vis, t) and parent is not None:
                chain = self._chain(nodes, kk)
                if len(chain) >= 1 and all(self._valid(vis, vis.terms(q, j)) for q, j, _ in chain[-P['s2_last_valid']:]):
                    if kk not in punsure:
                        punsure[kk] = p_unsure(predictor.predict(p[[0, 2]], vis.yaw(k)))
                    if punsure[kk] <= want:
                        actions = [a for _, _, a in chain]
                        self.stats["plans"] += 1
                        self.stats["expansions"] += expansions
                        return actions, [(q, j) for q, j, _ in chain], {
                            "reason": "s2", "p_unsure": round(punsure[kk], 3), "p0": round(self.p0, 3),
                            "expansions": expansions, "actions": len(actions), "zeta": round(vis.zeta, 1)}
            for a in ACTIONS:
                hist = self._last_actions(nodes, kk)
                if not _allowed(hist, a):
                    continue
                nxt = step(pf, p, k, a, yaw0)
                if nxt is None:
                    continue
                q, j = nxt
                if np.linalg.norm(q[[0, 2]] - c) > P['s2_box'] or key(q, j) in closed:
                    continue
                tq = vis.terms(q, j)
                step_cost = vis.cost(tq, forward=a == "move_forward")
                counter += 1
                heapq.heappush(frontier, (cost + step_cost + h(q, j), cost + step_cost, counter, (q, j), (kk, a)))
        self.stats["expansions"] += expansions
        return None

    def _visibility(self, obs):
        return Visibility(obs, **self.variant)

    @staticmethod
    def _chain(nodes, kk):
        """[(position, k, action into it)] from the start's child to ``kk``."""
        out = []
        while True:
            p, k, parent = nodes[kk]
            if parent is None:
                break
            pk, a = parent
            out.append((p, k, a))
            kk = pk
        return out[::-1]

    @staticmethod
    def _last_actions(nodes, kk, n=6):
        out = []
        while len(out) < n:
            parent = nodes[kk][2]
            if parent is None:
                break
            kk, a = parent
            out.append(a)
        return out[::-1]


class Visibility3(Visibility):
    """Round-3 fix 3: band [d_fit - 0.25, max(od_min + 0.75, eq. 4 with od_max = d_fit + 1.25)]; sight lines
    checked for lambda <= 0.85; FM tolerance set per plan (``fm_tol``)."""

    def __init__(self, obs, model=None, **kw):
        super().__init__(obs, model=model, **kw)
        self.los_max, self.fm_tol = 0.85, math.inf

    def band(self, k):
        od_min = self.fit[k] - 0.25
        eq4 = od_min + self.rd_scale * (self.fit[k] + 1.25 - od_min)
        return od_min, max(od_min + 0.75, eq4)

    def valid3(self, t, use_fm=True):
        return t["LOS"] and t["EC"] == 0 and t["OV"] == 0 and (not use_fm or t["FM"] <= self.fm_tol)


class FirstClearView3(FirstClearView):
    """S2 with round-3 fix 3 and phantom extents: FM tolerance = the 30th percentile of FM over the in-band
    lattice poses; no goal -> again without the FM tolerance; still none -> M1 (fba3) takes over."""
    name = "s2v3"

    def reset(self, ctx):
        super().reset(ctx)
        self.use_fm, self.fallback, self.stats["pass_rates"] = True, None, []

    def _visibility(self, obs):
        from active_approach.round3 import phantom_model
        model, _, info = phantom_model(obs)
        vis = Visibility3(obs, model=model)
        fms, rates = [], {"n": 0, "LOS": 0, "EC": 0, "OV": 0}
        for k in range(12):
            od_min, rd_max = vis.band(k)
            f = G.forward(vis.yaw(k))
            base = model.half_extent(f)
            for r in np.arange(od_min, rd_max + 1e-6, 0.25):
                xz = model.center - (base + r) * f
                q = np.asarray(obs.pathfinder.snap_point(np.array([xz[0], obs.floor_y, xz[1]], np.float32)), float)
                if not np.all(np.isfinite(q)) or np.hypot(*(q - [xz[0], q[1], xz[1]])[[0, 2]]) > 0.15:
                    continue
                t = vis.terms(q, k)
                rates["n"] += 1
                for term in ("LOS", "EC", "OV"):
                    rates[term] += bool(t[term]) if term == "LOS" else t[term] == 0
                fms.append(t["FM"])
        vis.fm_tol = float(np.percentile(fms, 30)) if fms else math.inf
        rates["fm_tol"] = round(vis.fm_tol, 2)
        self.stats["pass_rates"].append(rates)
        self.last_rates = rates
        return vis

    def _valid(self, vis, t):
        return vis.valid3(t, self.use_fm)

    def next_viewpoint(self, obs):
        if self.fallback is not None:
            return self.fallback.next_viewpoint(obs)
        vp = None
        self.use_fm = True
        vp = super().next_viewpoint(obs)
        if vp is None and self.reason == "nothing_to_gain" and not obs.arrived:
            self.reason, self.use_fm, self.checked = None, False, -999
            vp = super().next_viewpoint(obs)
            if vp is None and self.reason == "nothing_to_gain":
                from active_approach.round3 import FitBand3
                self.reason = "s2_to_m1"
                self.fallback = FitBand3()
                self.fallback.reset({})
                return self.fallback.next_viewpoint(obs)
        if vp is not None:
            vp.info.setdefault("pass_rates", getattr(self, "last_rates", None))
            vp.info.setdefault("fm_used", self.use_fm)
        return vp


class MPCDriver:
    """S3 tracking: beam search over F/L/R towards a viewpoint with the visibility terms."""
    def __init__(self, sim, s=None):
        from active_approach.policies import Controller
        self.controller, self.sim = Controller(sim), sim
        self.s = s
        self.queue, self.goal_key = [], None
        self.togo, self.stalls, self.fallback = math.inf, 0, 0

    def action(self, viewpoint, obs=None):
        position, yaw, _ = S.agent_pose(self.sim)
        if (obs is None or not len(obs.cloud) or self.fallback > 0
                or np.hypot(*(viewpoint.position - position)[[0, 2]]) <= P['mpc_near']):
            self.queue, self.fallback = [], max(0, self.fallback - 1)
            return self.controller.action(viewpoint)
        key = tuple(np.round(viewpoint.position, 2))
        if self.queue and key == self.goal_key:
            return self.queue.pop(0)
        if key != self.goal_key:
            self.togo, self.stalls = math.inf, 0
        self.goal_key = key
        # progress guard (not in the design): the visibility terms can hold the agent near the object, away
        # from a goal behind an obstacle; two replans without 0.5 actions of progress hand 6 actions to the
        # plain controller
        togo = G.action_cost(obs.pathfinder, position, yaw, viewpoint.position, viewpoint.yaw)
        self.stalls = self.stalls + 1 if togo > self.togo - P["mpc_progress"] else 0
        self.togo = min(self.togo, togo)
        if self.stalls >= 2:
            self.stalls, self.fallback = 0, P["mpc_fallback"] - 1
            self.queue = []
            return self.controller.action(viewpoint)
        plan = self._beam(obs, viewpoint)
        if not plan:
            return self.controller.action(viewpoint)
        self.queue = plan[1:P['mpc_execute']]
        return plan[0]

    def _beam(self, obs, viewpoint):
        vis = Visibility(obs, s=self.s)
        pf, yaw0 = obs.pathfinder, obs.yaw0
        beam = [(0.0, np.asarray(obs.position, float), heading_index(obs.yaw, yaw0), [])]
        for t in range(P['mpc_n']):
            grown = []
            for cost, p, k, hist in beam:
                for a in ACTIONS:
                    if not _allowed(hist, a):
                        continue
                    nxt = step(pf, p, k, a, yaw0)
                    if nxt is None:
                        continue
                    q, j = nxt
                    grown.append((cost + P['mpc_discount'] ** t * vis.cost(vis.terms(q, j), a == "move_forward"), q, j,
                                  hist + [a]))
            if not grown:
                break
            seen, beam = set(), []
            for item in sorted(grown, key=lambda x: x[0]):
                kk = (round(item[1][0] / 0.1), round(item[1][2] / 0.1), item[2])
                if kk in seen:
                    continue
                seen.add(kk)
                beam.append(item)
                if len(beam) >= P['mpc_beam']:
                    break
        best, best_j = None, math.inf
        for cost, p, k, hist in beam:
            togo = G.action_cost(pf, p, vis.yaw(k), viewpoint.position, viewpoint.yaw)
            j = cost + P['mpc_w_goal'] * togo
            if j < best_j:
                best, best_j = hist, j
        return best


def main(argv=None):
    """``zeta-max``: freeze zeta_max = the 95th percentile of zeta over the start frames (pitch 0)."""
    import argparse
    from collections import defaultdict
    from pathlib import Path
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["zeta-max"])
    parser.add_argument("--scenarios", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--out", default=P["zeta_max_file"])
    args = parser.parse_args(argv)
    scenarios = [json.loads(l) for l in open(args.scenarios)]
    by_scene, values = defaultdict(list), []
    for s in scenarios:
        by_scene[s["scene"]].append(s)
    for scene, items in by_scene.items():
        sim = S.make_sim(scene, args.gpu)
        try:
            for s in items:
                obs = S.set_pose(sim, s["start"]["position"], s["start"]["yaw"], 0.0)
                position, yaw, _ = S.agent_pose(sim)
                depth = np.asarray(obs["depth"]).squeeze()
                mask = np.asarray(obs["semantic"]).squeeze() == s["object_id"]
                cloud = S.backproject(np.where(depth < G.DEPTH_MAX, depth, 0), mask, *S.camera(sim))
                d_obj = float(np.min(np.hypot(cloud[:, 0] - position[0], cloud[:, 2] - position[2]))) if len(cloud) else None
                values.append(zeta(depth, mask, position, yaw, s["floor_y"], d_obj))
        finally:
            sim.close()
    result = {"zeta_max": float(np.percentile(values, 95)), "n": len(values), "median": float(np.median(values)),
              "source": str(args.scenarios), "rule": "95th percentile of zeta over the start frames at pitch 0"}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(result, open(args.out, "w"), indent=1)
    print(result)


if __name__ == "__main__":
    main()
