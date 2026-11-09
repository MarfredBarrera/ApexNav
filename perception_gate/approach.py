"""Approach driven by the gate itself (verification.approach_planner: mppi | closer), with the decision flow of
active_approach/decide.py (formal flow 2026-11-07), in closed loop.

When the jev_sam3 server says the first view is insufficient, the eval loop does not hand the approach to
ApexNav's object navigation (state 3). It drives the agent with a self-contained planner and answers the
original request with one verdict (habitat_evaluation.run_gate_approach):

    approach = LiveApproach(sim, gate, cluster_id, pending)
    while (action := approach.step(rgb, depth, target_masks, target_centroids, pose_planner)) is not None:
        observations = env.step(action) ...
    accepted = approach.result["accept"]

Everything the planner conditions on is the agent's own RGB-D, odometry and the detector's target-class masks
(no navmesh, no semantic GT): the object mask is the union of this step's target detections whose world
centroid matches the tracked object point (planner frame), else the fused cloud reprojected into the frame
(local_nav.track_mask). The planners are active_approach's: ``mppi`` (mppi.Mppi, the footprint MPPI) or
``mppi_v8`` (closer until it would stop, then MPPI for a better view), ``mppi_v10`` (one MPPI cost blending
closer's trajectory as a reference, weighted by distance to closer's goal, with the view score; the best in Stage B,
98/104, 2026-11-08), ``mppi_v11`` (v10 plus a cut-aware re-plan when the frame cuts the object top; 99/104), ``mppi_v15`` (one view
MPPI warm-started and, far out, guided by closer's drive; 99/104 with fewer actions, best on a second set), or
``closer`` (selfcontained.Closer driven by local_nav.LocalController), on a local map built from the agent's
own depth (local_nav.LocalMap / LocalPathfinder).

Decision flow (decide.flow, frozen DEFAULTS): every usable frame (facing the object within 20 deg, >= 400 mask
px) is named by JEV over the approach (HM3D) words with SAM3 scores (POST /name); frames pass the diversity
gate (GATE all), update the Dirichlet belief, and once >= 2 frames are counted and P(consistent) >= 0.8 the
fused SAM3 verdict over the 6 nearest counted views (REL_AREA 0.1) commits. The planner ending, the budget
or getting stuck forces the same verification; no counted frame rejects. The first frame of the approach is
the flow's start frame (step 0).
"""
import json
import math
import os
import time

import cv2
import numpy as np

PLANNERS = ("mppi", "mppi_v8", "mppi_v10", "mppi_v11", "mppi_v15", "closer")   # active_approach.mppi.POLICIES_MPPI names, or closer
MATCH_M = 1.0            # a target detection belongs to the approached object within this (planner frame)
FACE_TURNS = 12          # turns to find the object before the approach gives up
JUDGE_PX, SAM3_PX, FACE_DEG = 200, 400, 20.0
STUCK_STEPS = 3
MAX_IDLE = 4


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class LiveApproach:
    def __init__(self, sim, url, target, object_pf, budget, planner, log_dir=None, tag="", meta=None):
        from active_approach import decide as D
        from active_approach import geom as G
        from active_approach import local_nav as L
        from active_approach import scene as S
        if planner not in PLANNERS:
            raise ValueError(f"approach_planner must be one of {PLANNERS}, not {planner!r}")
        self.D, self.G, self.L, self.S = D, G, L, S
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
        self.policy.reset({"scenario": {"target": target}, "pathfinder": self.pf, "views": [], "sim": sim,
                           "surface": None, "judge_url": url})
        self.P = dict(D.DEFAULTS)
        self.counted, self.belief = [], None
        self.actions = self.flow_step = self.stuck = self.idle = self.face_turns = self.skips = 0
        self.arrived, self.started, self.last_action, self.last_pos = False, False, None, None
        self.result = None
        self.log = None
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            self.log_dir = log_dir
            self.log = open(os.path.join(log_dir, "approach.jsonl"), "a", encoding="utf-8")

    # ------------------------------------------------------------------
    def _mask(self, depth, masks, centroids):
        """Target detections matching the object point (planner frame), else the reprojected cloud."""
        hits = [i for i, c in enumerate(centroids or [])
                if c is not None and math.hypot(c[0] - self.object_pf[0], c[1] - self.object_pf[1]) <= MATCH_M]
        if hits:
            mask = np.zeros(depth.shape, bool)
            for i in hits:
                mask |= np.asarray(masks[i]).squeeze() > 0
            self.object_pf = 0.7 * self.object_pf + 0.3 * np.mean([centroids[i] for i in hits], axis=0)
            return mask, "detector"
        if len(self.cloud):
            return self.L.track_mask(self.cloud, depth, *self.S.camera(self.sim)), "reproj"
        return np.zeros(depth.shape, bool), "none"

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

    def _commit(self, status, by, end=None):
        self.result = {"accept": status == "correct", "status": status, "by": by, "end": end,
                       "actions": self.actions, "counted": len(self.counted),
                       "views": [f["step"] for f in self.counted]}
        self._write({"event": "commit", **self.result})
        if self.log:
            self.log.close()
            self.log = None
        return None

    def _forced(self, end):
        D, target = self.D, self.target
        if not self.counted:
            return self._commit("wrong", "empty", end)
        if self.belief is None or target not in self.belief.options:
            return self._commit("wrong", "forced", end)
        return self._commit(D.verify(self.counted, self.belief, target, self.P["MARGIN"], self.P), "forced", end)

    def _write(self, row):
        if self.log:
            self.log.write(json.dumps(row, default=float) + "\n")
            self.log.flush()

    # ------------------------------------------------------------------
    def step(self, rgb, depth, masks, centroids, pose_pf):
        """This step's frame -> the next action name (move_forward / turn_left / turn_right), or None when the
        approach has committed (``self.result``). ``masks`` / ``centroids``: the target-class detections and
        their world centroids (planner frame); ``pose_pf``: (x, y, yaw) in the planner frame."""
        D, G, L, S = self.D, self.G, self.L, self.S
        if self.result is not None:
            return None
        position, yaw, pitch = S.agent_pose(self.sim)
        if abs(pitch) > 1e-3:
            raise RuntimeError(f"gate approach needs camera pitch 0, got {math.degrees(pitch):.0f} deg")
        if self.last_action == "move_forward" and self.last_pos is not None:
            moved = float(np.hypot(*(position - self.last_pos)[[0, 2]]))
            self.stuck = self.stuck + 1 if moved < 0.02 else 0   # (mppi and the controller mark the bump cells)
        depth = np.asarray(depth, np.float32)
        mask, source = self._mask(depth, masks, centroids)
        cam = S.camera(self.sim)
        new = S.backproject(np.where(depth < G.DEPTH_MAX, depth, 0), mask, *cam)
        if source == "reproj" and len(new):
            new = new[new[:, 1] - self.floor_y > 0.04]
        if len(new):
            merged = np.concatenate([self.cloud, new]) if len(self.cloud) else new
            _, keep = np.unique(np.floor(merged / 0.05).astype(np.int64), axis=0, return_index=True)
            self.cloud = merged[np.sort(keep)][-20000:]
        self.hmap.update(depth, mask, *cam)
        self.lmap.set_object(self.cloud)
        self.pf.refresh(position)
        row = {"step": self.actions, "position": [round(float(x), 3) for x in position], "yaw": round(yaw, 4),
               "mask_px": int(mask.sum()), "mask_source": source, "cloud": len(self.cloud)}

        if not self.started:   # find the object first: turn toward the approach point
            if mask.sum() >= JUDGE_PX and len(self.cloud):
                self.started = True
            elif self.face_turns >= FACE_TURNS or self.actions >= self.budget:
                self._write({**row, "event": "not_found"})
                return self._commit("wrong", "not_found", "not_found")
            else:
                diff = _wrap(math.atan2(self.object_pf[1] - pose_pf[1], self.object_pf[0] - pose_pf[0]) - pose_pf[2])
                self.face_turns += 1
                return self._act("turn_left" if diff > 0 else "turn_right", {**row, "event": "find"}, position, yaw)

        # the flow: name usable frames, update the belief, commit when sure
        center = G.ObjectModel(self.cloud, self.floor_y).center if len(self.cloud) else None
        here = position[[0, 2]]
        offset = math.degrees(abs(G.wrap(yaw - G.bearing_yaw(here, center)))) if center is not None else 180.0
        px = int(mask.sum())
        judgeable = px >= JUDGE_PX and offset <= FACE_DEG
        first = not self.counted and self.belief is None
        commit = None
        if center is not None and (first or (judgeable and px >= SAM3_PX)):
            jev = self._name(rgb, mask, sam3=judgeable and px >= SAM3_PX)
            frame = {"step": 0 if first else self.flow_step, "jev": jev,
                     "pose": {"judgeable": judgeable, "distance": float(np.hypot(*(here - center))),
                              "bearing": float(math.atan2(*(here - center)[::-1]))},
                     "mask": {"area": float(mask.mean()), "top": bool(mask[0].any()), "bottom": bool(mask[-1].any()),
                              "left": bool(mask[:, 0].any()), "right": bool(mask[:, -1].any())}}
            row["named"] = {"top1": max(jev["probabilities"], key=jev["probabilities"].get),
                            "sam3": bool(jev.get("sam3"))}
            commit = self._flow(frame)
            if self.result is None and self.belief is not None:
                row["p_consistent"] = round(float(D.p_consistent(self.belief)), 4)
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
                        self.floor_y, self.judged, self.budget - self.actions, self.arrived, dict(self.meta))
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

    def _flow(self, frame):
        """decide.flow, one frame at a time: the commit status once the belief is sure, else None."""
        D, P, target = self.D, self.P, self.target
        if not D.usable(frame, P):
            return None
        g = {**frame, "jev": {**frame["jev"], "probabilities": D.folded_jev(frame, target, P["TV_MONITOR"])}}
        self.counted, changed = D.admit(g, self.counted, P)
        if not changed:
            return None
        self.belief = D.build_belief(self.counted, target, P)
        if frame["step"] == 0 or len(self.counted) < P["N_MIN"] or target not in self.belief.options:
            return None
        if D.sure(self.belief, P):
            return D.verify(self.counted, self.belief, target, P["MARGIN"], P)
        return None

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
