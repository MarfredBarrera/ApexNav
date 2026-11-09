"""Round-3 planners (docs/active_approach_planners.md section 5). New policy names only; the round-1/2
policies are unchanged.

Fix 2, unknown extent past clipped borders (``phantom_model``). The last judged frame's mask touching an
image border means the object continues there:
    top     z_hi = max(z_hi_obs + 0.2, z_frustum + 0.1, z_lo + H_prior), z_frustum = the top image ray's
            height at the object's depth
    bottom  z_lo = max(0, min(z_lo_obs - 0.2, z_hi - H_prior))
    side    the footprint is extended along the image-plane direction by max(0.2, W_prior - W_obs)
Priors (H x W): chair 0.9 x 0.55, couch 0.85 x 2.0, bed 0.6 x 2.0, toilet 0.8 x 0.45, tv 0.7 x 1.1,
potted plant 1.0 x 0.5; the max of the target's and the contending look-alike c*'s (JEV's start
distribution). A side is dropped once the last judged frame shows it unclipped. Phantom points count
for d_fit, area and truncation, not for occlusion.

Fix 1, relative quality (M2c): q(p) = clip(log2(area_hat(p)/area_ref), 0, 3)/3 (1 - occ_hat)(1 - trunc_hat)
+ 0.5 max(0, occ_ref - occ_hat(p)); ref = the current pose (base planner) or the best view so far (tour).

Fix 7 (planning side only; the decision module is unchanged): the tour wants cheap, consistent views:
same band, the next lattice bearing (>= 30 deg from every judged view), S = q - A/60 + 0.01 nov with q >= 0,
K = 3 stops.

Fix 3 (S2): see ``FirstClearView3``.

    fba_tour3   M1 with phantoms + tour3
    svr_tour3   M2c with phantoms ("nothing to gain": best q < 0.15) + tour3
    s2_tour3    S2 with fix 3 (and phantoms) + tour3; no valid pose -> M1
"""
import math

import numpy as np

from active_approach import geom as G
from active_approach import scene as S
from active_approach.policies import (A0, FitBand, SureViewRing, Tour, Viewpoint, geodesic, novelty, view_bearing,
                                      wrap)

PRIORS = {"chair": (0.9, 0.55), "couch": (0.85, 2.0), "bed": (0.6, 2.0), "toilet": (0.8, 0.45), "tv": (0.7, 1.1),
          "potted plant": (1.0, 0.5)}
# JEV naming options that are one of the prior categories
ALIAS = {"armchair": "chair", "monitor": "tv", "sofa": "couch", "plant": "potted plant"}
# perception.jev.CONFUSERS (the look-alikes JEV names), copied: the apexnav container cannot import perception
CONFUSERS = {"couch": ["armchair", "chair", "bench", "bed", "cushion"],
             "chair": ["armchair", "stool", "bench", "couch", "table"],
             "bed": ["couch", "bench", "ottoman or pouf", "table", "clothes"],
             "tv": ["picture", "mirror", "window", "cabinet", "monitor"],
             "potted plant": ["vase", "lamp", "flowers", "sculpture", "curtain"],
             "toilet": ["sink", "bathtub", "toilet paper holder", "trash can"]}


def prior_hw(meta):
    """(H, W) prior: max of the target's and the contending look-alike's (from JEV's start distribution)."""
    target = meta.get("target")
    h, w = PRIORS.get(target, (0.8, 0.8))
    probabilities = ((meta.get("first") or {}).get("jev_probabilities")) or {}
    rivals = [c for c in CONFUSERS.get(target, []) if c in probabilities and not (target == "chair" and c == "armchair")]
    if rivals:
        c_star = max(rivals, key=lambda c: probabilities[c])
        c_star = ALIAS.get(c_star, c_star)
        if c_star in PRIORS:
            h, w = max(h, PRIORS[c_star][0]), max(w, PRIORS[c_star][1])
    return h, w


def phantom_model(obs):
    """(ObjectModel with the unknown extent, phantom points, info)."""
    model = G.ObjectModel(obs.cloud, obs.floor_y)
    info = {"top": False, "bottom": False, "left": False, "right": False}
    if not obs.judged or not len(obs.cloud):
        return model, np.zeros((0, 3)), info
    last = obs.judged[-1]
    clip = last.get("clip") or {}
    h_prior, w_prior = prior_hw(obs.meta)
    extra = []
    cam = np.asarray(last["camera"], float)
    yaw = last.get("yaw", obs.yaw)
    f, right = G.forward(yaw), np.array([math.cos(yaw), -math.sin(yaw)])
    z_lo, z_hi = model.z_lo, model.z_hi
    if clip.get("top"):
        depth = float((model.center - cam) @ f)
        z_frustum = G.H_CAM + max(depth, 0.1) * math.tan(G.BETA_V)
        z_hi = max(z_hi + 0.2, z_frustum + 0.1, z_lo + h_prior)
        info["top"] = True
    if clip.get("bottom"):
        z_lo = max(0.0, min(z_lo - 0.2, z_hi - h_prior))
        info["bottom"] = True
    xz = obs.cloud[:, [0, 2]]
    proj = (xz - model.center) @ right
    w_obs = float(proj.max() - proj.min()) if len(proj) else 0.0
    ext = max(0.2, w_prior - w_obs)
    mean_y = float(np.median(obs.cloud[:, 1]))
    for side, sign, edge in (("left", -1.0, proj.min()), ("right", 1.0, proj.max())):
        if not clip.get(side):
            continue
        info[side] = round(ext, 2)
        rim = obs.cloud[np.abs(proj - edge) < 0.08]
        for t in np.linspace(0.05, ext, 6):
            shifted = rim.copy()
            shifted[:, 0] += sign * t * right[0]
            shifted[:, 2] += sign * t * right[1]
            extra.append(shifted[::max(1, len(shifted) // 30)])
    if info["top"] or info["bottom"]:
        foot = obs.cloud[::max(1, len(obs.cloud) // 60)][:, [0, 2]]
        for zz in ((z_hi,) if info["top"] else ()) + ((z_lo,) if info["bottom"] else ()):
            extra.append(np.stack([foot[:, 0], np.full(len(foot), obs.floor_y + zz), foot[:, 1]], 1))
    extra = np.concatenate(extra) if extra else np.zeros((0, 3))
    if len(extra):
        model = G.ObjectModel(np.concatenate([obs.cloud, extra]), obs.floor_y)
    model.z_lo, model.z_hi = z_lo, z_hi
    return model, extra, info


def phantom_candidates(obs, model, extra=(0.0, 0.25, 0.5)):
    """geom.lattice_candidates on the phantom model (its z extent and footprint already hold the unknown extent)."""
    return G.lattice_candidates(model, obs.yaw0, obs.pathfinder, obs.floor_y, extra=extra)


def relative_q(pred, ref):
    gain = math.log2(max(pred["area"], 1e-5) / max(ref["area"], 1e-5))
    return (min(max(gain, 0.0), 3.0) / 3.0 * (1 - pred["occ"]) * (1 - pred["trunc"])
            + 0.5 * max(0.0, ref["occ"] - pred["occ"]))


class FitBand3(FitBand):
    """M1 on the phantom model."""
    name = "fba3"

    def next_viewpoint(self, obs):
        if obs.arrived:
            return None
        if not len(obs.cloud) or (self.goal is not None and obs.step - self.planned < self.REPLAN):
            return self.goal
        self.planned = obs.step
        model, _, info = phantom_model(obs)
        cands = phantom_candidates(obs, model)
        here = G.bearing_yaw(obs.position[[0, 2]], model.center)
        k_here = round(wrap(here - obs.yaw0) / G.LATTICE) % 12
        cands = [c for c in cands if math.isfinite(geodesic(obs.pathfinder, obs.position, c["position"]))]
        if not cands:
            return self.goal
        best = min(cands, key=lambda c: (min((c["k"] - k_here) % 12, (k_here - c["k"]) % 12), c["r"]))
        new = Viewpoint(best["position"], best["yaw"], {"k": best["k"], "r": round(best["r"], 2),
                                                        "d_fit": round(best["d_fit"], 2), "phantom": info})
        if self.goal is None or np.hypot(*(new.position - self.goal.position)[[0, 2]]) > 0.25:
            self.goal = new
        return self.goal


def _score3(obs, cands, ref_pred, predictor, model, sep_deg=None, remaining=None, tour=False):
    out = []
    for c in cands:
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
        q = relative_q(pred, ref_pred)
        s = (q - a / A0 + 0.01 * nov) if tour else (q * (0.5 + 0.5 * nov) - a / A0)
        out.append((s, q, nov, a, pred, c))
    out.sort(key=lambda t: -t[0])
    return out


def _vp3(entry, reason, info=None):
    s, q, nov, a, pred, c = entry
    return Viewpoint(c["position"], c["yaw"], {"reason": reason, "S": round(s, 3), "q": round(q, 3),
                                               "nov": round(nov, 2), "A": a, "k": c["k"], "r": round(c["r"], 2),
                                               "pred": {k: round(v, 3) for k, v in pred.items()},
                                               **({"phantom": info} if info else {})})


class SureViewRing3(SureViewRing):
    """M2c: relative quality against the current pose, phantoms; nothing to gain when the best q < 0.15."""
    name = "svr3"
    NOTHING_Q = 0.15

    def next_viewpoint(self, obs):
        if obs.arrived:
            return None
        if not len(obs.cloud) or (self.goal is not None and obs.step - self.planned < self.REPLAN):
            return self.goal
        self.planned = obs.step
        model, extra, info = phantom_model(obs)
        predictor = G.Predictor(model, obs.cloud, obs.hmap, obs.pathfinder, extra=extra)
        ref = predictor.predict(obs.position[[0, 2]], obs.yaw) if self.goal is None else self.ref
        scored = _score3(obs, phantom_candidates(obs, model), ref, predictor, model)
        if not scored:
            if self.goal is None:
                self.reason = "no_candidate"
            return self.goal
        best = scored[0]
        if self.goal is None:
            self.ref = ref
            if max(e[1] for e in scored) < self.NOTHING_Q:
                self.reason = "nothing_to_gain"
                return None
            self.goal, self.score = _vp3(best, "plan", info), best[0]
        elif best[0] > self.score + self.SWITCH and np.hypot(*(best[5]["position"] - self.goal.position)[[0, 2]]) > 0.25:
            self.goal, self.score = _vp3(best, "switch", info), best[0]
        return self.goal


class Tour3(Tour):
    """M3 for round 3: K = 3 stops; next views in the same band, >= 30 deg from every judged view, cheap and
    no worse than the best view so far (q >= 0 relative to it): S = q - A/60 + 0.01 nov."""
    K, SEPARATION, SLACK = 3, 30.0, 1.25

    def __init__(self, base, name):
        super().__init__(base, name=name)

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
            model, extra, info = phantom_model(obs)
            predictor = G.Predictor(model, obs.cloud, obs.hmap, obs.pathfinder, extra=extra)
            # the best view so far, by predicted readability, is the reference
            refs = [predictor.predict(np.asarray(j["camera"]), j.get("yaw", obs.yaw)) for j in obs.judged] or \
                [predictor.predict(obs.position[[0, 2]], obs.yaw)]
            ref = max(refs, key=lambda p: p["area"] * (1 - p["occ"]) * (1 - p["trunc"]))
            cands = phantom_candidates(obs, model, extra=(0.0, 0.25))
            scored = _score3(obs, cands, {**ref, "area": ref["area"] * 0.999}, predictor, model,
                             sep_deg=self.SEPARATION, remaining=obs.remaining / self.SLACK, tour=True)
            # q >= 0 always holds for the gain term; require the candidate to be no worse than the reference
            scored = [e for e in scored if e[4]["area"] * (1 - e[4]["occ"]) * (1 - e[4]["trunc"])
                      >= 0.7 * ref["area"] * (1 - ref["occ"]) * (1 - ref["trunc"])]
            if not scored:
                self._reason = "no_candidate"
                return None
            self.goal = _vp3(scored[0], f"tour{self.stops + 1}", info)
        return self.goal


def _s2_v3():
    from active_approach import svpto
    return svpto.FirstClearView3()


class Seek:
    """Evidence seeking: run ``base``; if it has nothing left while fewer than MIN_VIEWS frames were judged (the
    object never came into a usable view), hand the remaining budget to ApexNav's native approach (1.5 m
    standoff, then 0.75 m, facing the object) so the decision has evidence. Otherwise identical to ``base``."""
    MIN_VIEWS = 2

    def __init__(self, base, name):
        from active_approach.policies import Native
        self.base, self.name, self.native = base, name, Native()
        self.enroute = getattr(base, "enroute", False)
        self.driver = getattr(base, "driver", None)

    @property
    def reason(self):
        return "seek" if self.seeking else getattr(self.base, "reason", None)

    def reset(self, ctx):
        self.base.reset(ctx)
        self.native.reset(ctx)
        self.seeking = False

    def satisfied(self, obs):
        """Enough evidence to stop collecting: MIN_VIEWS judged views."""
        return len(obs.judged) >= self.MIN_VIEWS

    def next_viewpoint(self, obs):
        import dataclasses
        if not self.seeking:
            vp = self.base.next_viewpoint(obs)
            if vp is not None or self.satisfied(obs):
                return vp
            self.seeking = True
            obs = dataclasses.replace(obs, arrived=False)   # the base policy's last arrival is not a native one
        return self.native.next_viewpoint(obs)


class SeekConfident(Seek):
    """Seek until confident (the decision flow of decide.py): also keep approaching while the Dirichlet belief
    over the judged views' JEV distributions is not dominant (max(P_acc, P_rej) < P_STAR)."""
    P_STAR = 0.95

    def satisfied(self, obs):
        if len(obs.judged) < self.MIN_VIEWS:
            return False
        from active_approach import belief as B
        target = obs.meta.get("target")
        named = [j for j in obs.judged if j.get("jev")]
        if not target or not named:
            return True    # nothing to judge confidence by (no --log-frames naming): as Seek
        belief = B.Belief(target)
        for j in named:
            belief.add({"jev": {"probabilities": j["jev"]}, "pose": {"bearing": 0.0, "distance": 0.0,
                                                                   "offset_deg": 0.0}}, 1.0)
        if target not in belief.options:
            return True
        s = belief.signals()
        return max(s["p_acc"], s["p_rej"]) >= self.P_STAR


class JevPilot:
    """JEV as the approach controller (single image, forward + turn only): every step JEV is asked, on the frame
    with the object outlined, which move gives a clearer view of it (turn left / turn right / move forward). It
    stops when the Dirichlet belief over the JEV naming distributions of the usable frames it has seen
    (>= SAM3_PX mask px) is dominant (SeekConfident's rule), and needs --log-frames (that naming). When the object
    is out of view (< MIN_PIXELS) it turns toward the tracked cloud instead of asking."""
    name, enroute, driver, reason = "jev_pilot", False, None, None
    QUESTION = ("The robot must check whether the outlined object is a {target}. "
                "Which move gives the robot a clearer view of the outlined object?")
    OPTIONS = ("turn left", "turn right", "move forward")
    ACTIONS = {"turn left": "turn_left", "turn right": "turn_right", "move forward": "move_forward"}
    MIN_PIXELS, SAM3_PX, MIN_VIEWS, P_STAR = 50, 400, 2, 0.95

    def reset(self, ctx):
        self.url, self.target = ctx.get("judge_url"), ctx["scenario"]["target"]
        self.run, self.named, self.last = None, [], None
        if not self.url:
            raise RuntimeError("JevPilot needs the jev_sam3 judge (perception server /ask)")

    def confident(self):
        if len(self.named) < self.MIN_VIEWS:
            return False
        from active_approach import belief as B
        belief = B.Belief(self.target)
        for p in self.named:
            belief.add({"jev": {"probabilities": p}, "pose": {"bearing": 0.0, "distance": 0.0, "offset_deg": 0.0}}, 1.0)
        if self.target not in belief.options:
            return True
        s = belief.signals()
        return max(s["p_acc"], s["p_rej"]) >= self.P_STAR

    def act(self, obs):
        from active_approach.policies import facing, wrap
        from perception_gate.client import ask
        pixels = int(np.asarray(obs.mask).sum())
        if obs.meta.get("jev") and pixels >= self.SAM3_PX:
            self.named.append(obs.meta["jev"])
        if self.confident():
            self.last = {"stop": "confident", "views": len(self.named)}
            return None
        if pixels < self.MIN_PIXELS:
            if not len(obs.cloud):
                self.last = {"stop": "lost"}
                return None
            diff = wrap(facing(obs.position, obs.cloud.mean(0)) - obs.yaw)
            self.last = {"recover": round(math.degrees(diff), 1)}
            return "turn_left" if diff > 0 else "turn_right"
        reply = ask(self.url, obs.rgb, obs.mask, self.target, self.QUESTION.format(target=self.target), self.OPTIONS,
                    request_id=f"aa_pilot_{self.target}_s{obs.step}")
        choice = max(reply["probabilities"], key=reply["probabilities"].get)
        self.last = {"probabilities": reply["probabilities"], "choice": choice}
        return self.ACTIONS[choice]


POLICIES3 = {"fba_tour3": lambda: Tour3(FitBand3(), "fba_tour3"),
             "svr_tour3": lambda: Tour3(SureViewRing3(), "svr_tour3"),
             "s2_tour3": lambda: _s2_tour3()}
for _n in list(POLICIES3):
    POLICIES3[_n + "_seek"] = (lambda n: lambda: Seek(POLICIES3[n](), n + "_seek"))(_n)
    POLICIES3[_n + "_seekc"] = (lambda n: lambda: SeekConfident(POLICIES3[n](), n + "_seekc"))(_n)
POLICIES3["jev_pilot"] = JevPilot


def _s2_tour3():
    tour = Tour3(_s2_v3(), "s2_tour3")
    tour.enroute = "valid"   # en-route frames are judged (and banked) only at valid poses
    return tour
