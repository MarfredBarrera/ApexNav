"""Approach driven by the gate itself (verification.approach_planner: mppi | closer), with the decision flow of
active_approach/decide.py (formal flow 2026-11-07), in closed loop.

When the jev_sam3 server says the first view is insufficient, the eval loop does not hand the approach to
ApexNav's object navigation (state 3). It drives the agent with a self-contained planner and answers the
original request with one verdict (habitat_evaluation.run_gate_approach):

    approach = LiveApproach(sim, gate, cluster_id, pending)
    while (action := approach.step(rgb, depth, target_masks, target_centroids, pose_planner)) is not None:
        observations = env.step(action) ...
    accepted = approach.result["accept"]

Everything the planner conditions on is the agent's own RGB-D, odometry and detector masks (no navmesh, no
semantic GT). Object identity (2026-11-09, one rule): the object is the fused cloud O (5 cm voxels). Each frame its
reference outline T is O splatted into the frame with a depth test (local_nav.track_mask), so T is 3D-consistent;
before O exists, T is the ApexNav target mask whose world centroid is nearest the requested object point, within
SEED_M; without one, the SAM3 candidate whose 3D centre is (``_seed_pick``: the only centroid test, used only to
seed O, normally in the request frame). Every detector mask, ApexNav's per-step
target masks and SAM3 /detect candidates alike, is associated with T by ``associate``: the best IoU
|M & T| / |M | T| (ties: higher detector score), accepted when >= ASSOC_MIN, else no detection this frame. So a sconce
above the tracked orchid, a curtain behind the tracked pedestal, a whole bed around the tracked bench at its foot or
a small fragment of the tracked chair are all rejected; only a mask of about the object's own extent is accepted
(the overlap coefficient used before 2026-11-09 preferred containers and fragments). O grows only from the accepted detector mask, eroded by ERODE_PX against depth-edge flying pixels; the
projected outline T only steers the planner and never feeds O or the decision. The request's bank views carry no
depth, so they are named (``_seed``) but do not seed O. Each step logs the association scores of every
candidate (row "assoc") and the model's size against this frame's mask (row "model").

The planners are active_approach's: ``mppi`` (mppi.Mppi, the footprint MPPI) or ``mppi_v8`` (closer until it would stop, then MPPI for a better view), ``mppi_v10`` (one MPPI cost blending
closer's trajectory as a reference, weighted by distance to closer's goal, with the view score; the best in Stage B,
98/104, 2026-11-08), ``mppi_v11`` (v10 plus a cut-aware re-plan when the frame cuts the object top; 99/104), ``mppi_v15`` (one view
MPPI warm-started and, far out, guided by closer's drive; 99/104 with fewer actions, best on a second set), or
``closer`` (selfcontained.Closer driven by local_nav.LocalController), on a local map built from the agent's
own depth (local_nav.LocalMap / LocalPathfinder).

Decision flow (decide.flow, frozen DEFAULTS): SAM3 runs on every frame facing the object (FACE_DEG) with
|T| >= SAM3_PX (no pose-gate freshness test, so the closest views reach the verification). Every usable frame whose
mask is SAM3's accepted instance (or, at step 0, the seed mask) is named by JEV over the approach (HM3D) words with
SAM3 scores (POST /name); frames pass the pose diversity gate into the Dirichlet belief, and once >= 2 frames are
counted and P(consistent) >= 0.8 the fused SAM3 verdict over the 6 nearest usable views (all of them, self.seen;
REL_AREA 0.1) commits. The planner ending, the budget
or getting stuck forces the same verification. The first frame of the approach is the flow's start frame (step 0).
The planner sees the belief's P(consistent) (Observation.meta["p_consistent"], 0 before any counted frame): MPPI's
stop tolerance grows with it (mppi.MppiV15.stop_delta), so it keeps moving for a better view while the verdict is open.

Size opinion (2026-11-10, decide.DEFAULTS["SIZE"]; active_approach/size_opinion.py): the fused SAM3 belief is
multiplied by a per-class likelihood of the object's metric width W (longer robust side of O) and the verdict accepts
iff the target is still the top class. A veto only: too small rules the target out, nothing confirms it (no rival
loses mass; size_opinion.likelihood). Vacuous unless W is complete (a contributing view that is neither side-clipped
nor side-occluded, by depth, spans >= 0.9 W) and the target is in size_opinion.SIZE_TARGETS (couch: armchairs). Each
contributing step logs "size_view"; the commit logs "size" (W, extents, complete, per-view flags, likelihoods, belief
before / after) and saves O as object_cloud.npz.

Last mile (2026-11-10): after an accept the gate drives the agent to the object and issues the STOP itself
(``start_last_mile``; habitat_evaluation.run_gate_last_mile), instead of handing ApexNav a forced target. It
continues the approach that accepted (O, the local map and its collision marks carry over) or, after an accept on
the first view (``LiveApproach(..., last_mile=True)``), first finds the object as an approach does, O seeded once from
the first detector mask of it. O is frozen for the whole last mile (``_grows``: the object is verified, the last mile only
navigates; a mask merging the bed with the wall once inflated O to 4.3 m and stopped the agent early, dev epi710).
The last mile measures against the decided object (``object_points``: the deciding view's SAM3 instance, else O).
Goal: the cheapest reachable cell of the local map (inflated by the agent radius) within STOP_M of its nearest
point, preferring CLEAR_M clearance from other obstacles (wedges came from hugging them), kept while it stays
reachable and its path is at most GOAL_SWITCH_M longer than the best (a goal that flips between two cells turns
the agent left and right without end, dev epi587 / 876); driven by ``_goto``. Stop: the agent within STOP_M
(horizontal) of O's nearest point, or at the goal cell (GOAL_REACH_M), then it turns to face O and returns "stop". No progress
(LM_STALL actions without getting LM_PROGRESS_M closer, three blocked steps, no reachable goal) or the separate
budget LM_BUDGET used up: stop anyway within NEAR_M of O, else hand back to ApexNav (None; ``last_mile["end"]``
"handback"), which then plans without a forced target. Rows "last_mile" in approach.jsonl, the end in
"last_mile_end". The navmesh is never used.

No evidence is no verdict (2026-11-09): an approach that never finds the object (``not_found``: FACE_TURNS turns,
the budget, or facing the object point with nothing detected there, where more turns show nothing new) or ends with
no counted frame (``empty``) commits status ``undecided`` (accept False). The gate logs it as DEFERRED
(gate.finish_external_approach): ApexNav sets the cluster aside and keeps exploring, and the gate's re-check rule
reopens it on a new view (cooldown, max_checks), as for a reject.
"""
import json
import math
import os
import time

import cv2
import numpy as np

PLANNERS = ("mppi", "mppi_v8", "mppi_v10", "mppi_v11", "mppi_v15", "closer")   # active_approach.mppi.POLICIES_MPPI names, or closer
SEED_M = 1.0             # before O exists: the request's ApexNav mask is the one whose centroid is nearest the object
ASSOC_MIN = 0.3          # a detector mask is the object when its IoU with T reaches this
ERODE_PX = 3             # detector masks are eroded by this before back-projection (depth-edge flying pixels)
VOXEL_M, CLOUD_MAX = 0.05, 20000
FACE_TURNS = 12          # turns to find the object before the approach gives up
GOTO_M, GOTO_SLACK = 1.0, 0.3   # a request farther than GOTO_M + GOTO_SLACK: first drive to GOTO_M short of the object
GOTO_LOOK, GOTO_ALIGN, GOTO_STALL = 0.5, 20.0, 3   # pursuit lookahead (m), heading tolerance (deg), blocked steps
JUDGE_PX, SAM3_PX, FACE_DEG = 200, 400, 20.0
STUCK_STEPS = 3
MAX_IDLE = 4
# last mile after an accept (module docstring)
STOP_M, NEAR_M = 0.65, 1.0       # goal ring and stop: within STOP_M of O's nearest point; no progress: stop within NEAR_M
GOAL_REACH_M = 0.15              # ... or within this of the goal cell's centre
GOAL_SWITCH_M = 0.5              # the goal is kept unless a new one's path is this much shorter
CLEAR_M, CLEAR_W = 0.45, 4.0     # goal cells closer than CLEAR_M to a non-object obstacle cost CLEAR_W per m short
LM_BUDGET, LM_STALL, LM_PROGRESS_M, LM_FACE_TURNS = 40, 8, 0.1, 6
LAST_MILE_REACH_M = 4.0          # an accept on the first view starts the last mile only within this of the object


def _sam3_top(jev):
    """{"sam3_top": [(class, score), (class, score)]} of a named frame with SAM3 scores, else {} (logging)."""
    scores = (jev.get("sam3") or {}).get("scores")
    if not scores:
        return {}
    top = sorted(scores.items(), key=lambda kv: -kv[1])[:2]
    return {"sam3_top": [[k, round(float(v), 3)] for k, v in top]}


def _replay(frame):
    """A named frame as decide.py reads it (JEV probabilities, full SAM3 scores, pose, mask stats): logging."""
    jev = frame["jev"]
    return {"step": frame["step"], "pose": frame["pose"], "mask": frame["mask"],
            "jev": {"probabilities": {k: round(float(v), 5) for k, v in jev["probabilities"].items()},
                    "sam3": {"scores": {k: round(float(v), 4)
                                        for k, v in ((jev.get("sam3") or {}).get("scores") or {}).items()}}}}


def _seed_pick(distances, max_m=SEED_M):
    """The seed rule (before O exists): index of the smallest distance (None: unknown) within ``max_m``, else None."""
    near = [(d, i) for i, d in enumerate(distances) if d is not None and d <= max_m]
    return min(near)[1] if near else None


def associate(masks, reference, min_score=ASSOC_MIN, det_scores=None):
    """The one association rule: which of ``masks`` (bool images) is the object tracked by the outline ``reference``.
    -> (index of the accepted mask or None, [(overlap, iou)] per mask). The mask with the best IoU is accepted when
    its IoU >= ``min_score``; ties go to the higher ``det_scores`` (detector score) if given, else the first.
    overlap = |M & T| / min(|M|, |T|) is logged only."""
    ref = np.asarray(reference, bool)
    t = int(ref.sum())
    scores = []
    for m in masks:
        m = np.asarray(m, bool)
        inter, n = int(np.logical_and(m, ref).sum()), int(m.sum())
        union = n + t - inter
        scores.append((inter / min(n, t) if n and t else 0.0, inter / union if union else 0.0))
    if not scores:
        return None, scores
    det = [0.0] * len(scores) if det_scores is None else [float(d) for d in det_scores]
    best = min(range(len(scores)), key=lambda i: (-scores[i][1], -det[i], i))
    return (best if scores[best][1] >= min_score else None), scores


class LiveApproach:
    lm = last_mile = None   # the last mile's state and outcome (start_last_mile)

    def __init__(self, sim, url, target, object_pf, budget, planner, log_dir=None, tag="", meta=None, bank=None,
                 last_mile=False):
        from active_approach import decide as D
        from active_approach import geom as G
        from active_approach import local_nav as L
        from active_approach import scene as S
        from active_approach import size_opinion as SZ
        if planner not in PLANNERS:
            raise ValueError(f"approach_planner must be one of {PLANNERS}, not {planner!r}")
        self.D, self.G, self.L, self.S, self.SZ = D, G, L, S, SZ
        self.sim, self.url, self.target = sim, url, target
        self.object_pf = np.asarray(object_pf, float)
        self.budget, self.planner_name, self.tag = int(budget), planner, tag
        position, yaw, _ = S.agent_pose(sim)
        self.floor_y = float(position[1])
        self.hmap = G.HeightMap(position[[0, 2]], self.floor_y)
        self.lmap = L.LocalMap(self.hmap)
        self.pf = L.LocalPathfinder(self.lmap, self.floor_y)
        self.pf.refresh(position)
        self.yaw0 = yaw
        self.cloud = np.zeros((0, 3))
        self.judged = []
        self.meta = {"target": target, **(meta or {})}
        if planner != "closer":
            from active_approach.mppi import POLICIES_MPPI
            self.policy, self.controller = POLICIES_MPPI[planner](), None
        else:
            from active_approach.selfcontained import Closer
            self.policy, self.controller = Closer(), L.LocalController(sim, self.lmap, self.pf)
        self.goto, self.goto_stall = True, 0   # the find phase's drive toward a far object (_goto)
        self.policy.reset({"scenario": {"target": target}, "pathfinder": self.pf, "views": [], "sim": sim,
                           "surface": None, "judge_url": url})
        self.P = dict(D.DEFAULTS)
        # view diversity (2026-11-09): a frame counts only as a new viewpoint (>= DIST_M closer or >= ANGLE_DEG around
        # the object from every counted view); a duplicate that reads REPLACE_GAIN better replaces the one it repeats
        self.P["GATE"] = "pose"   # diversity for the JEV belief only; SAM3 verifies over every usable view (self.seen)
        # the gate's view bank of the object (every stored detection of it, also from before the request): named and
        # counted before the first live frame (_seed), so the fusion sees the views native jev_sam3 judged
        self.bank, self.seeded = list(bank or []), False
        self.counted, self.belief, self.seen = [], None, []
        # the size opinion (decide.fused_accept): per view that grew O, its side clip / occlusion flags and span
        self.size_views, self.size_info = [], {}
        self.view_pts = {}   # flow step -> the points of that frame's SAM3 instance (object_points)
        self.actions = self.flow_step = self.stuck = self.idle = self.face_turns = self.skips = 0
        self.arrived, self.started, self.last_action, self.last_pos = False, False, None, None
        self.result = None
        self.lm, self.last_mile = None, None   # the last mile's state and its outcome
        self.log = None
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            self.log_dir = log_dir
            self.log = open(os.path.join(log_dir, "approach.jsonl"), "a", encoding="utf-8")
        if last_mile:   # accepted already (first view): only find the object and drive the last mile
            self.seeded = True
            self.result = {"accept": True, "status": "correct", "by": "accepted", "end": None, "actions": 0}
            self.start_last_mile(budget)

    # ------------------------------------------------------------------
    def _reference(self, depth, masks, centroids):
        """This frame's reference outline T: O projected with a depth test, or (before O exists) the ApexNav target
        mask whose centroid is nearest the requested object point within SEED_M (``_seed_pick``). -> (T, source)."""
        if len(self.cloud):
            return self.L.track_mask(self.cloud, depth, *self.S.camera(self.sim)), "reproj"
        best = _seed_pick([None if c is None else math.hypot(c[0] - self.object_pf[0], c[1] - self.object_pf[1])
                           for c in centroids or []])
        if best is not None:
            return np.asarray(masks[best]).squeeze() > 0, "seed"
        return np.zeros(depth.shape, bool), "none"

    def _grow(self, depth, mask):
        """Add an accepted detector mask, eroded by ERODE_PX, to the object cloud O (VOXEL_M voxels)."""
        k = np.ones((2 * ERODE_PX + 1, 2 * ERODE_PX + 1), np.uint8)
        core = cv2.erode(mask.astype(np.uint8), k) > 0
        new = self.S.backproject(np.where(depth < self.G.DEPTH_MAX, depth, 0), core, *self.S.camera(self.sim))
        if len(new):
            merged = np.concatenate([self.cloud, new]) if len(self.cloud) else new
            _, keep = np.unique(np.floor(merged / VOXEL_M).astype(np.int64), axis=0, return_index=True)
            self.cloud = merged[np.sort(keep)][-CLOUD_MAX:]
        return new

    def _name(self, rgb, mask, sam3):
        from perception_gate.client import PROTOCOL, _call, encode_png
        payload = {"protocol": PROTOCOL, "request_id": f"ap_{self.tag}_n{self.flow_step}", "target": self.target,
                   "image": encode_png(cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2BGR)),
                   "mask": encode_png(mask.astype(np.uint8) * 255), "sam3": bool(sam3), "words": "approach"}
        reply = _call(self.url, "/name", payload, timeout=120.0)
        if not reply.get("ok"):
            raise RuntimeError(f"perception server /name: {reply.get('error')}")
        out = {"options": reply["jev"]["options"], "probabilities": reply["jev"]["probabilities"]}
        scores = reply.get("sam3_scores")
        if scores is not None:
            out["sam3"] = {"scores": scores}
        return out

    def _detect(self, rgb, reference, depth=None, seed_xz=None):
        """SAM3 as the approach detector (POST /detect): its full-frame instances of the target word or a look-alike
        (score >= 0.3), associated with the reference outline T (``associate``). With an empty T before O exists
        (no ApexNav seed), the seed rule instead: the candidate whose 3D centre (median of its depth points, sim
        x/z) is nearest ``seed_xz`` within SEED_M. -> ({mask, word, score, scores (SAM3 score of every vocabulary
        word on that segment)} or None, [per-candidate log])."""
        import base64
        from perception_gate.client import _call, encode_png
        payload = {"request_id": f"det_{self.tag}_{self.actions}", "target": self.target,
                   "image": encode_png(cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2BGR))}
        reply = _call(self.url, "/detect", payload, timeout=120.0)
        if not reply.get("ok"):
            raise RuntimeError(f"perception server /detect: {reply.get('error')}")
        cand = reply["candidates"]
        masks = [cv2.imdecode(np.frombuffer(base64.b64decode(c["mask"]), np.uint8), cv2.IMREAD_GRAYSCALE) > 127
                 for c in cand]
        best, scores = associate(masks, reference, det_scores=[c["score"] for c in cand])
        log = [{"word": c["word"], "score": round(float(c["score"]), 3), "px": int(m.sum()),
                "overlap": round(o, 3), "iou": round(u, 3)} for c, m, (o, u) in zip(cand, masks, scores)]
        if not reference.any() and seed_xz is not None and not len(self.cloud):
            cam, d = self.S.camera(self.sim), np.where(depth < self.G.DEPTH_MAX, depth, 0)
            dist = []
            for m, row in zip(masks, log):
                pts = self.S.backproject(d, m, *cam)
                dist.append(float(np.hypot(*(np.median(pts[:, [0, 2]], 0) - seed_xz))) if len(pts) else None)
                row["seed_dist"] = None if dist[-1] is None else round(dist[-1], 2)
            best = _seed_pick(dist)
        if best is None:
            return None, log
        c = cand[best]
        return {"mask": masks[best], "word": c["word"], "score": c["score"], "scores": c["scores"]}, log

    def _size(self):
        """The tracked object's width for the size opinion: W = the longer robust footprint side of O
        (ObjectModel.robust_extent), complete when a clean contributing view spans it (size_opinion.complete)."""
        if len(self.cloud) < 3:
            return {"W": None, "complete": False}
        model = self.G.ObjectModel(self.cloud, self.floor_y)
        robust = model.robust_extent()
        width = float(max(robust))
        return {"W": round(width, 3), "complete": self.SZ.complete(self.size_views, width),
                "extent_robust": [round(float(x), 3) for x in robust],
                "extent": [round(float(x), 3) for x in model.extent]}

    def _verify(self):
        """decide.verify over every usable view, with the size opinion (its details in self.size_info)."""
        self.size_info = {}
        return self.D.verify(self.seen, self.belief, self.target, self.P["MARGIN"], self.P, size=self._size(),
                             info=self.size_info)

    def _commit(self, status, by, end=None):
        self.result = {"accept": status == "correct", "status": status, "by": by, "end": end,
                       "actions": self.actions, "counted": len(self.counted),
                       "views": [f["step"] for f in self.counted]}
        # re-measurement later: the width, its per-view evidence, the opinion, and O itself (object_cloud.npz)
        self.result["size"] = {**self._size(), "switch": bool(self.P.get("SIZE")), **self.size_info,
                               "views": self.size_views}
        if self.log and len(self.cloud):
            np.savez_compressed(os.path.join(self.log_dir, "object_cloud.npz"),
                                cloud=self.cloud.astype(np.float32), floor_y=np.float32(self.floor_y))
        if self.seen:   # what the verification saw: the chosen views' SAM3 top-2 and JEV top-1
            chosen = self.D.chosen_views(self.seen, self.P)
            self.result["verified"] = [{"step": f["step"], "area": round(f["mask"]["area"], 4),
                                        "distance": round(f["pose"]["distance"], 2), **_sam3_top(f["jev"]),
                                        "jev": max(f["jev"]["probabilities"], key=f["jev"]["probabilities"].get)}
                                       for f in chosen]
        self._write({"event": "commit", **self.result})
        return None

    def close(self):
        if self.log:
            self.log.close()
            self.log = None

    def _forced(self, end):
        target = self.target
        if not self.counted:   # no view of the object was usable: no evidence either way
            return self._commit("undecided", "empty", end)
        if self.belief is None or target not in self.belief.options:
            return self._commit("wrong", "forced", end)
        return self._commit(self._verify(), "forced", end)

    def _write(self, row):
        if self.log:
            self.log.write(json.dumps(row, default=float) + "\n")
            self.log.flush()

    # ------------------------------------------------------------------
    def step(self, rgb, depth, masks, centroids, pose_pf):
        """This step's frame -> the next action name (move_forward / turn_left / turn_right), or None when the
        approach has committed (``self.result``); in the last mile also "stop", and None once handed back. ``masks`` / ``centroids``: the target-class detections and
        their world centroids (planner frame); ``pose_pf``: (x, y, yaw) in the planner frame."""
        D, G, S = self.D, self.G, self.S
        if self.result is not None and not self.driving_last_mile:
            return None
        position, yaw, pitch = S.agent_pose(self.sim)
        if abs(pitch) > 1e-3:
            raise RuntimeError(f"gate approach needs camera pitch 0, got {math.degrees(pitch):.0f} deg")
        self.blocked = False
        if self.last_action == "move_forward" and self.last_pos is not None:
            moved = float(np.hypot(*(position - self.last_pos)[[0, 2]]))
            self.stuck = self.stuck + 1 if moved < 0.02 else 0
            # collision feedback for every forward step, whichever phase issued it (a partial step is contact too)
            self.blocked = self.lmap.forward_feedback(self.last_pos, self.last_yaw, position)
        depth = np.asarray(depth, np.float32)
        # identity (module docstring): T = O projected (or the seed mask); ApexNav's target masks and SAM3's
        # candidates are associated with T by the same rule; SAM3's accepted instance wins, then ApexNav's
        reference, ref_source = self._reference(depth, masks, centroids)
        apex = [np.asarray(m).squeeze() > 0 for m in masks or []]
        best, apex_scores = associate(apex, reference)
        row_assoc = {"ref": ref_source, "ref_px": int(reference.sum()),
                     "apexnav": [{"px": int(m.sum()), "overlap": round(o, 3), "iou": round(u, 3)}
                                 for m, (o, u) in zip(apex, apex_scores)]}
        if best is not None:
            mask, source = apex[best], "detector"
            row_assoc["apexnav_chosen"] = best
        else:
            mask, source = reference, ("reproj" if ref_source == "reproj" else "none")
        # SAM3 runs on every frame facing the object (O's centre, else the requested point) whose outline T is big
        # enough to judge (always while still finding the object: then an empty T falls back to the seed rule)
        det = None
        ref = G.ObjectModel(self.cloud, self.floor_y).center if len(self.cloud) else \
            self._object_xz(position, yaw, pose_pf)
        here = position[[0, 2]]
        facing = math.degrees(abs(G.wrap(yaw - G.bearing_yaw(here, ref)))) <= FACE_DEG
        big = not self.started or int(reference.sum()) >= SAM3_PX
        row_skip = None if (facing and big) else {"facing": facing, "big": big}
        if row_skip is None and not (self.lm is not None and self.started):   # the last mile needs no verdict
            det, row_assoc["sam3"] = self._detect(rgb, reference, depth, ref)
        if det is not None:
            mask, source = det["mask"], "sam3"
        new = self._grow(depth, mask) if self._grows(source) else np.zeros((0, 3))
        cam = S.camera(self.sim)
        self.hmap.update(depth, mask, *cam)
        self.lmap.set_object(self.cloud)
        self.pf.refresh(position)
        row = {"step": self.actions, "position": [round(float(x), 3) for x in position], "yaw": round(yaw, 4),
               "mask_px": int(mask.sum()), "mask_source": source, "cloud": len(self.cloud), "assoc": row_assoc}
        if len(self.cloud):   # the object model's size vs this frame's accepted mask (footprint L x S, height; m)
            model = G.ObjectModel(self.cloud, self.floor_y)
            row["model"] = {"extent": [round(float(x), 2) for x in model.extent], "height": round(model.height, 2)}
            if len(new) > 2:
                seen = G.ObjectModel(new, self.floor_y)
                row["model"]["mask"] = {"extent": [round(float(x), 2) for x in seen.extent],
                                        "height": round(seen.height, 2)}
                # this view's width evidence: its own span (robust extent) and whether a side was cut or hidden
                view = {"step": self.actions, "span": round(float(max(seen.robust_extent())), 3),
                        **self.SZ.side_flags(mask, depth)}
                self.size_views.append(view)
                row["size_view"] = view
        if det is not None:
            row["sam3_det"] = {"word": det["word"], "score": round(det["score"], 3)}
        elif row_skip is not None:
            row["sam3_skip"] = row_skip
        if not self.seeded:
            self.seeded = True
            self._seed(position, yaw, pose_pf)

        if not self.started:   # find the object first: drive toward it while far, then turn toward it
            if mask.sum() >= JUDGE_PX and len(self.cloud):
                self.started = True
            elif self.face_turns >= FACE_TURNS or self.actions >= self.budget:
                self._write({**row, "event": "not_found"})
                return self._not_found("not_found", row)
            else:
                obj = G.ObjectModel(self.cloud, self.floor_y).center if len(self.cloud) else \
                    self._object_xz(position, yaw, pose_pf)
                dist = float(np.hypot(*(obj - here)))
                row["object_dist"] = round(dist, 2)
                if self.goto and dist > GOTO_M + GOTO_SLACK:
                    action = self._goto(position, yaw, obj + (here - obj) / dist * GOTO_M)
                    if action is not None:
                        return self._act(action, {**row, "event": "goto"}, position, yaw)
                    self.goto = False   # no path on the local map, or blocked: look for it from here
                diff = G.wrap(G.bearing_yaw(here, obj) - yaw)
                if math.degrees(abs(diff)) <= FACE_DEG:   # facing it and nothing there: turning shows nothing new
                    self._write({**row, "event": "not_found"})
                    return self._not_found("facing", row)
                self.face_turns += 1
                return self._act("turn_left" if diff > 0 else "turn_right", {**row, "event": "find"}, position, yaw)

        if self.lm is not None:
            return self._last_mile_step(row, position, yaw)

        # the flow: name usable frames, update the belief, commit when sure
        center = G.ObjectModel(self.cloud, self.floor_y).center if len(self.cloud) else None
        here = position[[0, 2]]
        offset = math.degrees(abs(G.wrap(yaw - G.bearing_yaw(here, center)))) if center is not None else 180.0
        px = int(mask.sum())
        judgeable = px >= JUDGE_PX and offset <= FACE_DEG
        first = not self.counted and self.belief is None
        commit = None
        # only SAM3's own instance mask (this frame) feeds the decision; ApexNav's mask and the projected outline
        # are the planner's guesses of where the object is and only steer MPPI (the request frame keeps its mask)
        if center is not None and (source == "sam3" or (first and source == "detector")) \
                and (first or (judgeable and px >= SAM3_PX)):
            if source == "sam3":   # JEV names SAM3's segment; SAM3's scores on that segment come with it
                jev = self._name(rgb, mask, sam3=False)
                if judgeable and px >= SAM3_PX:
                    jev["sam3"] = {"scores": det["scores"]}
            else:
                jev = self._name(rgb, mask, sam3=judgeable and px >= SAM3_PX)
            frame = {"step": 0 if first else self.flow_step, "jev": jev,
                     "pose": {"judgeable": judgeable, "distance": float(np.hypot(*(here - center))),
                              "bearing": float(math.atan2(*(here - center)[::-1]))},
                     "mask": {"area": float(mask.mean()), "top": bool(mask[0].any()), "bottom": bool(mask[-1].any()),
                              "left": bool(mask[:, 0].any()), "right": bool(mask[:, -1].any())}}
            row["named"] = {"top1": max(jev["probabilities"], key=jev["probabilities"].get),
                            "sam3": bool(jev.get("sam3")), **_sam3_top(jev)}
            row["frame"] = _replay(frame)   # everything a fusion rule needs, for offline replay
            commit = self._flow(frame)
            if self.result is None and self.belief is not None:
                row["p_consistent"] = round(float(D.p_consistent(self.belief)), 4)
            if source == "sam3" and len(new) >= 3:
                self.view_pts[frame["step"]] = new
            self._save(rgb, mask, frame["step"])
        self.flow_step += 1
        if commit is not None:
            self._write({**row, "event": "belief"})
            return self._commit(commit, "belief")

        # the planner
        if self.actions >= self.budget:
            return self._forced("budget")
        if self.stuck >= STUCK_STEPS:
            return self._forced("stuck")
        if judgeable and (first or self.arrived):
            self.judged.append({"camera": (float(here[0]), float(here[1])), "yaw": yaw,
                                "top_clipped": bool(mask[0].any()),
                                "side_clipped": bool(mask[:, 0].any() or mask[:, -1].any()),
                                "clip": {"top": bool(mask[0].any()), "bottom": bool(mask[-1].any()),
                                         "left": bool(mask[:, 0].any()), "right": bool(mask[:, -1].any())},
                                "step": self.actions, "area": float(mask.mean()), "jev": None})
        from active_approach.policies import Observation
        o = Observation(self.actions, position, yaw, rgb, depth, mask, self.cloud, self.pf, self.hmap, self.yaw0,
                        self.floor_y, self.judged, self.budget - self.actions, self.arrived, self._meta())
        self.arrived = False
        if self.controller is None:
            action = self.policy.act(o)
            row["planner"] = getattr(self.policy, "last", None)
            if action is None:
                self._write({**row, "event": "planner_done"})
                return self._forced("planner_done")
            return self._act(action, row, position, yaw)
        while True:   # closer: viewpoints, driven by the local controller (harness run.py)
            vp = self.policy.next_viewpoint(o)
            if vp is None:
                self._write({**row, "event": "planner_done"})
                return self._forced("planner_done")
            row["viewpoint"] = {"position": [round(float(x), 3) for x in vp.position], **vp.info}
            try:
                action = self.controller.action(vp, o)
            except Exception as error:
                row["error"] = str(error)
                if self.skips < 3:
                    self.skips += 1
                    o.arrived = self.arrived = True
                    continue
                self._write({**row, "event": "no_path"})
                return self._forced("no_path")
            if action is None:   # at the viewpoint, facing it: the next call sees this frame as an arrival
                self.idle += 1
                if self.idle > MAX_IDLE:
                    return self._forced("planner_done")
                o.arrived = self.arrived = True
                if judgeable:
                    self.judged.append({"camera": (float(here[0]), float(here[1])), "yaw": yaw,
                                        "top_clipped": bool(mask[0].any()),
                                        "side_clipped": bool(mask[:, 0].any() or mask[:, -1].any()),
                                        "clip": {"top": bool(mask[0].any()), "bottom": bool(mask[-1].any()),
                                                 "left": bool(mask[:, 0].any()), "right": bool(mask[:, -1].any())},
                                        "step": self.actions, "area": float(mask.mean()), "jev": None})
                continue
            self.idle = 0
            return self._act(action, row, position, yaw)

    # ------------------------------------------------------------------ last mile
    @property
    def driving_last_mile(self):
        return self.lm is not None and self.last_mile is None

    def start_last_mile(self, budget=LM_BUDGET):
        """After an accept: drive on to a stop pose by the object (module docstring). ``step`` then returns
        actions, "stop" (stop here, facing the object) or None (handed back: ``self.last_mile``)."""
        self.lm = {"start": self.actions, "budget": int(budget), "best": math.inf, "best_togo": math.inf,
                   "stall": 0, "face": 0, "stop": None}
        self.goto_stall = 0
        if self.started:
            self.budget = self.actions + int(budget)   # (the find phase's budget when the last mile starts it)
        else:
            self.budget = int(budget)
        self._write({"event": "last_mile_start", "budget": int(budget), "cloud": len(self.cloud)})

    def _not_found(self, why, row):
        if self.lm is not None:
            return self._end_last_mile("handback", why, row, None)
        return self._commit("undecided", "not_found", "not_found" if why == "not_found" else why)

    def object_points(self):
        """The decided object's points: the SAM3 instance of the deciding view (the nearest view the verification
        used that has one), else O. O can merge neighbours over an approach (a dining chair with its table and the
        other chairs, 3.6 m, dev epi396); the deciding view's instance is what was verified."""
        if self.seen:
            for f in self.D.chosen_views(self.seen, self.P):
                pts = self.view_pts.get(f["step"])
                if pts is not None:
                    return pts
        return self.cloud

    def _target(self):
        """The last mile's object points (object_points, fixed once O exists)."""
        if self.lm.get("obj") is None and len(self.cloud):
            self.lm["obj"] = self.object_points()
            self._write({"event": "last_mile_target", "points": len(self.lm["obj"]), "cloud": len(self.cloud)})
        return self.lm["obj"] if self.lm.get("obj") is not None else self.cloud

    def _object_distance(self, here):
        """Horizontal distance from ``here`` (x, z) to the last mile's object (``_target``)."""
        return float(np.min(np.hypot(*(self._target()[:, [0, 2]] - here).T)))

    def _grows(self, source):
        """Whether this frame's mask (``source``) grows O: a detector mask (SAM3's instance or the associated ApexNav
        mask) during the approach; in the last mile only the one that seeds an empty O (after a first-view accept:
        the accepted object as first found), then O is frozen. (SAM3 runs only facing the object, so a seed from
        SAM3 alone was never found while the find phase drove there, dev epi587.)"""
        detected = source in ("sam3", "detector")
        return detected if self.lm is None else detected and not len(self.cloud)

    def _stop_goal(self, here, keep=None):
        """The last mile's goal (x, z): among the cells reachable on the local map (inflated by the agent radius)
        within STOP_M of O's nearest point, the cheapest by path length plus CLEAR_W x the shortfall of its
        clearance from non-object obstacles below CLEAR_M, and its path length (m). ``keep`` (the previous goal)
        stays the goal while it is such a cell and its score is within GOAL_SWITCH_M of the best. (None, None)
        when no such cell is reachable."""
        from scipy.ndimage import distance_transform_edt
        from scipy.spatial import cKDTree
        lm, L = self.lmap, self.L
        dist, _ = self.pf._tree(lm.cell(here))
        dist = dist.reshape(lm.n, lm.n)
        ii, jj = np.nonzero(np.isfinite(dist))
        if not len(ii):
            return None, None
        centers = lm.origin + (np.c_[ii, jj] + 0.5) * L.CELL
        d_obj, _ = cKDTree(self._target()[:, [0, 2]]).query(centers)
        ok = d_obj <= STOP_M
        if not ok.any():
            return None, None
        occupied, _, _ = lm.grids()
        clear = distance_transform_edt(~(occupied & ~lm.object)) * L.CELL
        score = dist[ii, jj] * L.CELL + CLEAR_W * np.maximum(CLEAR_M - clear[ii, jj], 0.0)
        score = np.where(ok, score, np.inf)
        k = int(np.argmin(score))
        if keep is not None:
            j = np.flatnonzero((ii == lm.cell(keep)[0]) & (jj == lm.cell(keep)[1]))
            if len(j) and score[j[0]] <= score[k] + GOAL_SWITCH_M:
                k = int(j[0])
        return centers[k], float(dist[ii[k], jj[k]] * L.CELL)

    def _last_mile_step(self, row, position, yaw):
        """One step of the last mile (module docstring)."""
        G, lm = self.G, self.lm
        here = position[[0, 2]]
        d = self._object_distance(here)
        used = self.actions - lm["start"]
        goal, togo = self._stop_goal(here, lm.get("goal")) if lm["stop"] is None and d > STOP_M else (None, None)
        lm["goal"] = goal
        # progress: closer to O, or a shorter path to the stop cells (a detour around furniture)
        progress = d < lm["best"] - LM_PROGRESS_M or (togo is not None and togo < lm["best_togo"] - LM_PROGRESS_M)
        lm["best"] = min(lm["best"], d)
        if togo is not None:
            lm["best_togo"] = min(lm["best_togo"], togo)
        lm["stall"] = 0 if progress else lm["stall"] + (used > 0)
        center = G.ObjectModel(self._target(), self.floor_y).center
        diff = G.wrap(G.bearing_yaw(here, center) - yaw)
        row["last_mile"] = {"d": round(d, 3), "used": used, "stall": lm["stall"],
                            "togo": None if togo is None else round(togo, 2)}
        if lm["stop"] is None:
            why = None
            if d <= STOP_M or (goal is not None and np.hypot(*(goal - here)) <= GOAL_REACH_M):
                lm["stop"] = "reached"
            elif lm["stall"] >= LM_STALL:
                why = "stall"
            elif used >= lm["budget"]:
                why = "budget"
            else:
                action = None if goal is None else self._goto(position, yaw, goal)
                if action is not None:
                    row["last_mile"]["goal"] = [round(float(x), 2) for x in goal]
                    return self._act(action, {**row, "event": "last_mile"}, position, yaw)
                why = "no_goal" if goal is None else "blocked"
            if why is not None:
                if d > NEAR_M:
                    return self._end_last_mile("handback", why, row, d)
                lm["stop"] = why
        # stopping: face the object first
        if math.degrees(abs(diff)) > FACE_DEG and lm["face"] < LM_FACE_TURNS:
            lm["face"] += 1
            return self._act("turn_left" if diff > 0 else "turn_right", {**row, "event": "last_mile_face"},
                             position, yaw)
        return self._end_last_mile("stop", lm["stop"], row, d)

    def finish_last_mile(self, why):
        """End the last mile from outside (the episode's step limit): stop within NEAR_M, else hand back."""
        if not self.driving_last_mile:
            return
        position, _, _ = self.S.agent_pose(self.sim)
        d = self._object_distance(np.asarray(position, float)[[0, 2]]) if len(self.cloud) else None
        self._end_last_mile("stop" if d is not None and d <= NEAR_M else "handback", why, {}, d)

    def _end_last_mile(self, end, why, row, d):
        """-> "stop" (end "stop") or None (end "handback")."""
        self.last_mile = {"end": end, "reason": why, "actions": self.actions - self.lm["start"],
                          "distance": None if d is None else round(float(d), 3)}
        self._write({**row, "event": "last_mile_end", **{f"last_mile_{k}": v for k, v in self.last_mile.items()}})
        return "stop" if end == "stop" else None

    def _meta(self):
        """The planner's Observation.meta: self.meta plus the decision belief's P(consistent) (0 before any frame
        is counted: no evidence yet), which scales MPPI's stop tolerance (mppi.MppiV15.stop_delta)."""
        p = 0.0 if self.belief is None else float(self.D.p_consistent(self.belief))
        return {**self.meta, "p_consistent": p}

    def _seed(self, position, yaw, pose_pf):
        """Name the bank's views (oldest first) and count them through the diversity gate, without committing: the
        first live frame decides whether the belief is already sure. Their steps are negative (-n .. -1)."""
        bank = sorted(self.bank, key=lambda b: (b["step"], b.get("index", 0)))
        for k, item in enumerate(bank):
            mask = np.asarray(item["mask"]).squeeze() > 0
            px = int(mask.sum())
            if px < SAM3_PX:   # usable frames only (decide.usable: SAM3 scores need >= 400 px)
                continue
            here = self._to_xz(item["camera"], position, yaw, pose_pf)
            center = self._to_xz(item["centroid"], position, yaw, pose_pf)
            jev = self._name(item["rgb"], mask, sam3=True)
            step = k - len(bank)
            frame = {"step": step, "jev": jev,
                     "pose": {"judgeable": True, "distance": float(np.hypot(*(here - center))),
                              "bearing": float(math.atan2(*(here - center)[::-1]))},
                     "mask": {"area": float(mask.mean()), "top": bool(mask[0].any()), "bottom": bool(mask[-1].any()),
                              "left": bool(mask[:, 0].any()), "right": bool(mask[:, -1].any())}}
            self._flow(frame, commit=False)
            self._save(item["rgb"], mask, step)
            self._write({"event": "seed", "bank_step": int(item["step"]), "step": step, "mask_px": px,
                         "distance": round(frame["pose"]["distance"], 2),
                         "named": max(jev["probabilities"], key=jev["probabilities"].get), **_sam3_top(jev),
                         "counted": len(self.counted), "frame": _replay(frame)})

    def _flow(self, frame, commit=True):
        """decide.flow, one frame at a time: the commit status once the belief is sure, else None (always None
        without ``commit``: the bank seeds)."""
        D, P, target = self.D, self.P, self.target
        if not D.usable(frame, P):
            return None
        g = {**frame, "jev": {**frame["jev"], "probabilities": D.folded_jev(frame, target, P["TV_MONITOR"])}}
        self.seen.append(g)
        self.counted, changed = D.admit(g, self.counted, P)
        if not changed:
            return None
        self.belief = D.build_belief(self.counted, target, P)
        if not commit or frame["step"] == 0 or len(self.counted) < P["N_MIN"] or target not in self.belief.options:
            return None
        if D.sure(self.belief, P):
            return self._verify()
        return None

    def _goto(self, position, yaw, goal):
        """Pure pursuit along the local map's cheapest path to ``goal`` (sim x, z): turn toward the path point
        GOTO_LOOK ahead until within GOTO_ALIGN, else step forward. (LocalController's one-step lattice is not
        used: it prefers a step away over turning around, so it never reverses toward a goal behind.)
        None when there is no path or GOTO_STALL forward steps in a row were blocked."""
        G, lm, pf = self.G, self.lmap, self.pf
        here = np.asarray(position, float)[[0, 2]]
        if self.last_action == "move_forward" and self.last_pos is not None:   # (step() marked a blocked one)
            self.goto_stall = self.goto_stall + 1 if self.blocked else 0
        if self.goto_stall >= GOTO_STALL:
            return None
        cells = pf.path_cells(here, goal)
        if cells is None or len(cells) < 2:
            return None
        pts = np.array([lm.center(c) for c in cells])
        along = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(pts, axis=0).T))])
        target = pts[min(int(np.searchsorted(along, GOTO_LOOK)), len(pts) - 1)]
        diff = G.wrap(G.bearing_yaw(here, target) - yaw)
        if abs(diff) > math.radians(GOTO_ALIGN):
            return "turn_left" if diff > 0 else "turn_right"
        return "move_forward"

    def _object_xz(self, position, yaw, pose_pf):
        """The object point (planner frame) in the sim's (x, z) (``_to_xz``)."""
        return self._to_xz(self.object_pf, position, yaw, pose_pf)

    def _to_xz(self, point_pf, position, yaw, pose_pf):
        """A planner-frame point in the sim's (x, z), through the agent's pose in both frames: its offset ahead /
        to the left of the agent is frame-independent (both yaws turn left positive)."""
        d = np.asarray(point_pf, float)[:2] - np.asarray(pose_pf[:2], float)
        psi = float(pose_pf[2])
        ahead = d[0] * math.cos(psi) + d[1] * math.sin(psi)
        left = -d[0] * math.sin(psi) + d[1] * math.cos(psi)
        G = self.G
        return np.asarray(position, float)[[0, 2]] + ahead * G.forward(yaw) + left * G.forward(yaw + math.pi / 2)

    def _act(self, action, row, position, yaw):
        self.actions += 1
        self.last_action, self.last_pos, self.last_yaw = action, position.copy(), yaw
        self._write({**row, "action": action})
        return action

    def _save(self, rgb, mask, step):
        if not self.log:
            return
        folder = os.path.join(self.log_dir, "frames")
        os.makedirs(folder, exist_ok=True)
        img = cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2BGR).copy()
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, contours, -1, (0, 255, 0), 2)
        cv2.imwrite(os.path.join(folder, f"f{step:03d}_a{self.actions:03d}.jpg"), cv2.resize(img, (320, 240)),
                    [cv2.IMWRITE_JPEG_QUALITY, 80])
