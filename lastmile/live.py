"""Live takeover: the last-mile navigator of the bench (F_full_cur) driving the closed-loop ApexNav episode.

habitat_evaluation.run_takeover hands control to ``Takeover`` at the FIRST gate accept (verification.takeover=true;
default off). Everything below is the bench's own code path:
  navigator  lastmile.nav_ddppo.make() under the F_full_cur switches (FENV.sh: VLFM DD-PPO weights, sampled actions,
             free-map snap + line-of-sight waypoint + arrival rule, collision patches), goal = the perceived accepted
             object's cloud, stop = the navigator's own STOP (radius / snapped / stagnation / policy)
  obs        depth (480x640 raw metres, camera pitch known), depth_level (224x224, hfov 79, camera 0.88 m, pitch 0: an extra
             habitat-sim sensor, as bench.make_sim adds), pose (position, yaw, pitch), goal_cloud
Never hands back to ApexNav's navigation.
"""
import math
import os

import numpy as np

# F_full_cur (lastmile/_tmp/FENV.sh), sliding off; LASTMILE_DEVICE left to the caller
F_FULL_CUR = dict(LASTMILE_POLICY="vlfm", LASTMILE_SAMPLE="1", LASTMILE_SNAP2="1", LASTMILE_LOS="1", LASTMILE_ARRIVE="1",
                  LASTMILE_PATCH="cur", LASTMILE_GOAL="freemap")
# Stop-rule fix found on the 455-episode big bench (lastmile/_tmp/REPORT_live_fix.md; lastmile/bench_data_big), on by default in the
# live takeover, LASTMILE_TAKEOVER_FIX=0 gives plain F_full_cur: radius STOP at 0.5 m (not 0.8) to the 10th-percentile cloud distance
# (stray cloud points otherwise stop the agent 1.2-1.6 m from the object), no distance STOP in the first 5 steps unless within 0.5 m,
# stagnation STOP only within 1.0 m of the cloud (else keep trying to the cap)
STOP_FIX = dict(LASTMILE_GRACE="5", LASTMILE_GRACE_R="0.5", LASTMILE_STOP_R="0.5", LASTMILE_DIST_Q="10", LASTMILE_STAG_D="1.0")
STOP, FORWARD, LEFT, RIGHT = 0, 1, 2, 3
WANT_STOP = -1   # = nav_ddppo.WANT_STOP
LEVEL_RES, LEVEL_HFOV, LEVEL_H = 224, 79.0, 0.88   # bench.LEVEL_* defaults


def device():
    """LASTMILE_DEVICE, else the simulator's GPU when the run keeps CUDA for the MPPI approach (MPPI_GPU set), else cpu."""
    return os.environ.get("LASTMILE_DEVICE") or ("cuda:0" if os.environ.get("MPPI_GPU") is not None else "cpu")


def make_navigator():
    for k, v in F_FULL_CUR.items():
        os.environ[k] = v
    if os.environ.get("LASTMILE_TAKEOVER_FIX", "1") != "0":
        os.environ.update(STOP_FIX)
    os.environ["LASTMILE_DEVICE"] = device()
    from lastmile import nav_ddppo
    return nav_ddppo.make()


def ensure_level_sensor(sim):
    """Add the 224x224 level depth sensor to the live simulator (once per scene load). Call with the camera at pitch 0:
    look_up / look_down tilt every sensor, so a sensor added while tilted would stay tilted after levelling."""
    import habitat_sim
    if "depth_level" in sim._sensors:
        return
    sp = habitat_sim.CameraSensorSpec()
    sp.uuid, sp.sensor_type, sp.resolution = "depth_level", habitat_sim.SensorType.DEPTH, [LEVEL_RES, LEVEL_RES]
    sp.position, sp.hfov, sp.far = [0.0, LEVEL_H, 0.0], LEVEL_HFOV, 20.0
    sim.add_sensor(sp, 0)


def level_depth(sim):
    s = sim._sensors["depth_level"]
    s.draw_observation()
    return np.asarray(s.get_observation(), np.float32).reshape(LEVEL_RES, LEVEL_RES)


def pf_to_world_xz(point_pf, position, yaw, pose_pf):
    """A planner-frame (x, y) point in the sim's (x, z), through the agent's pose in both frames
    (perception_gate.approach.LiveApproach._to_xz)."""
    import math
    d = np.asarray(point_pf, float)[:2] - np.asarray(pose_pf[:2], float)
    psi = float(pose_pf[2])
    ahead = d[0] * math.cos(psi) + d[1] * math.sin(psi)
    left = -d[0] * math.sin(psi) + d[1] * math.cos(psi)
    f = np.array([-math.sin(yaw), -math.cos(yaw)])
    l = np.array([-math.sin(yaw + math.pi / 2), -math.cos(yaw + math.pi / 2)])
    return np.asarray(position, float)[[0, 2]] + ahead * f + left * l


class Takeover:
    """One episode's takeover: ``act(depth)`` -> bench action id (STOP/FORWARD/LEFT/RIGHT); ``info`` afterwards.

    LASTMILE_RH=1 (receding horizon; ``act(depth, last)`` with the step's execute_step output): the goal cloud is a world-frame
    target model O re-perceived from ApexNav's target detections every step (seeded from the accepted centroid, then IoU-associated
    with O reprojected), the navigator is retargeted every LASTMILE_RH_K (8) actions, and a distance-based STOP (radius / snapped /
    stagnation) needs evidence: O refined and seen within the last K steps, else a look chunk (<= RH_LOOK turns towards O, then a
    sweep) tries to re-see the target first; found -> retarget and re-evaluate, nothing -> the STOP stands."""

    RH_LOOK, RH_SEED_M, SWEEP = 12, 1.0, (LEFT, RIGHT, RIGHT, LEFT)

    def __init__(self, sim, cloud, budget, source):
        from active_approach import geom as G
        from active_approach import scene as S
        self.sim, self.S, self.G, self.budget, self.source = sim, S, G, int(budget), source
        self.cloud = np.asarray(cloud, np.float32).reshape(-1, 3)
        self.nav = make_navigator()
        self.actions = 0
        position, yaw, pitch = S.agent_pose(sim)
        self.start = dict(position=[round(float(v), 3) for v in position], yaw=round(float(yaw), 4))
        self._obs = None
        self.rh = bool(int(os.environ.get("LASTMILE_RH", 0)))
        self.nav.reset(self.obs(None), dict(goal_cloud=self.cloud, goal_estimate=self.cloud.mean(0), budget_steps=self.budget))
        if self.rh:
            self.k = int(os.environ.get("LASTMILE_RH_K", 8))
            self.refined = source == "approach_cloud"     # O is an object surface, not the accepted centroid
            self.t, self.last_seen, self.seen, self.with_masks = 0, None, 0, 0
            self.replans, self.looks, self.look_turns, self.look, self.rt_at, self.evidence = 0, 0, 0, None, 0, None
            # LASTMILE_APEXWP=1: the DD-PPO pointgoal is a waypoint on ApexNav-style A* path (lastmile/apex_astar.py) to O
            from lastmile import apex_astar
            self.apex = apex_astar.enabled()
            if self.apex:
                self.apex_map = os.environ.get("LASTMILE_APEXWP_MAP", "apex")   # apex: ApexNav's /grid_map topics; own: our LocalMap
                if self.apex_map not in ("own", "apex"):
                    raise ValueError("LASTMILE_APEXWP_MAP=%s" % self.apex_map)
                self.apex_src = None            # lastmile.apex_grid.TopicSource, set by habitat_evaluation.run_takeover
                self.apex_los = bool(int(os.environ.get("LASTMILE_APEXWP_LOS", 1)))
                self.apex_wp, self.apex_due, self.apex_blk, self.apex_ppos, self.apex_la = None, True, 0, None, None
                self.apex_stats = dict(apex_plans=0, apex_fail=0, apex_mode_counts={}, apex_path_len_last=None,
                                       apex_goal_to_O=None, apex_ms_max=0.0, apex_nogrid=0, apex_log=[])

    def obs(self, depth):
        position, yaw, pitch = self.S.agent_pose(self.sim)
        return {"rgb": None, "depth": depth, "depth_level": level_depth(self.sim),
                "pose": {"position": position, "yaw": yaw, "pitch": 0.0}, "goal_cloud": self.cloud}

    def act(self, depth, last=None):
        if self.rh:
            a = self._act_rh(depth, last)
        else:
            a = int(self.nav.act(self.obs(depth)))
        if a != STOP:
            self.actions += 1
        if self.rh and self.apex:
            self.apex_la = a
        return a

    # ------------------------------------------------------------------ receding horizon
    def _observe(self, last):
        """Match this frame's target detections to O (grow it), or seed / refine O from the one nearest the centroid."""
        from perception_gate.approach import ASSOC_MIN, _seed_pick, associate
        self.t += 1
        masks = [np.asarray(m).squeeze() > 0 for m in last["masks"]]
        self.with_masks += bool(masks)
        if not masks:
            return
        depth, i = np.asarray(last["depth"], np.float32), None
        if self.refined:
            ref = self.nav_track(depth)
            i, _ = associate(masks, ref) if ref.any() else (None, None)
        else:
            position, yaw, _ = self.S.agent_pose(self.sim)
            here = self.cloud[:, [0, 2]].mean(0)
            dist = [None if c is None else float(np.hypot(*(pf_to_world_xz(c, position, yaw, last["pose"]) - here)))
                    for c in last["centroids"]]
            i = _seed_pick(dist, self.RH_SEED_M)
            if i is not None:
                self.cloud = np.zeros((0, 3), np.float32)   # the singleton is replaced by the object's surface
        if i is None:
            return
        from perception_gate.approach import LiveApproach
        old = self.cloud
        if not self.refined:
            self.cloud = np.zeros((0, 3), np.float32)       # the singleton is replaced by the object's surface
        LiveApproach._grow(self, depth, masks[i])           # eroded mask -> backproject -> 5 cm voxels, capped
        if len(self.cloud):
            self.refined, self.last_seen, self.seen = True, self.t, self.seen + 1
        else:                                               # a mask without usable depth
            self.cloud = old

    def nav_track(self, depth):
        from active_approach import local_nav as L
        return L.track_mask(self.cloud, depth, *self.S.camera(self.sim))

    def _confirmed(self):
        return self.refined and self.last_seen is not None and self.t - self.last_seen <= self.k

    def _retarget(self):
        self.nav.retarget(self.cloud)
        self.replans, self.rt_at = self.replans + 1, self.actions
        if self.apex:
            self.apex_due = True

    def _look_action(self):
        """Turn towards O's centroid until within 15 deg, then sweep left/right."""
        position, yaw, _ = self.S.agent_pose(self.sim)
        d = self.cloud[:, [0, 2]].mean(0) - position[[0, 2]]
        fwd = np.array([-math.sin(yaw), -math.cos(yaw)])
        theta = math.atan2(d @ np.array([fwd[1], -fwd[0]]), d @ fwd)
        if abs(theta) > math.radians(15):
            return LEFT if theta > 0 else RIGHT
        return self.SWEEP[self.look_turns % len(self.SWEEP)]

    def _act_rh(self, depth, last):
        seen0 = self.last_seen
        self._observe(last)
        if self.look is not None:
            if self.last_seen != seen0:                     # re-seen: back to the navigator with the new O
                self.look, self.evidence = None, "look_seen"
                self._retarget()
            elif self.look >= self.RH_LOOK:                 # nothing: the STOP stands
                self.evidence = "look_empty"
                return STOP
            else:
                self.look += 1
                self.look_turns += 1
                return self._look_action()
        elif self.actions - self.rt_at >= self.k:
            self._retarget()
        if self.apex:
            self._apex_step(last)
        a = int(self.nav.act(self.obs(depth)))
        if a == WANT_STOP:
            if self._confirmed():
                self.evidence = self.evidence if self.evidence == "look_seen" else "seen"
                return STOP
            self.look, self.looks = 1, self.looks + 1
            self.look_turns += 1
            return self._look_action()
        if a == STOP:
            self.evidence = "policy"
        return a

    def world_to_pf(self, q, position, yaw, pose_pf):
        """Inverse of pf_to_world_xz: sim world (x, z) points -> planner-frame (x, y)."""
        q = np.asarray(q, float)
        d = q - np.asarray(position, float)[[0, 2]]
        f = np.array([-math.sin(yaw), -math.cos(yaw)])
        l = np.array([-math.cos(yaw), math.sin(yaw)])
        ahead, left = d @ f, d @ l
        psi = float(pose_pf[2])
        return np.asarray(pose_pf[:2], float) + ahead[..., None] * np.array([math.cos(psi), math.sin(psi)]) \
            + left[..., None] * np.array([-math.sin(psi), math.cos(psi)])

    def _apex_grid(self, last, position, yaw, pxz, O):
        """(grid, to_world, agent, O) in the planning frame: ApexNav's map in the planner frame, else our LocalMap in the world xz."""
        from lastmile import apex_astar as AA
        st = self.apex_stats
        if self.apex_map == "apex" and self.apex_src is not None and last is not None:
            pose_pf = last["pose"]
            agent_pf = np.asarray(pose_pf[:2], float)
            O_pf = self.world_to_pf(O, position, yaw, pose_pf)
            g = self.apex_src.grid(np.vstack([agent_pf[None], O_pf]))
            if g is not None:
                st["apex_src"] = "apex"
                to_world = lambda p: pf_to_world_xz(p, position, yaw, pose_pf)
                return g, to_world, agent_pf, O_pf
            st["apex_nogrid"] += 1
        st["apex_src"] = "own"
        lm = self.nav.lmap
        if lm is None:                                      # the navigator builds its map on its first act
            st["apex_nogrid"] += 1
            return None
        return AA.Grid.from_localmap(lm), (lambda p: np.asarray(p, float)), pxz, O

    def _apex_step(self, last=None):
        """Replan the waypoint (every retarget, on arrival within 0.3 m, after 2 blocked FORWARDs, or when there is none)."""
        import time
        from lastmile import apex_astar as AA
        position, yaw, _ = self.S.agent_pose(self.sim)
        pxz = np.asarray(position, float)[[0, 2]]
        if self.apex_la == FORWARD and self.apex_ppos is not None:
            self.apex_blk = self.apex_blk + 1 if np.linalg.norm(pxz - self.apex_ppos) < 0.125 else 0
        self.apex_ppos = pxz
        wp = self.apex_wp
        if not (self.apex_due or wp is None or self.apex_blk >= 2 or np.linalg.norm(pxz - wp) < 0.3):
            return
        O = self.cloud[:, [0, 2]]
        st, t0 = self.apex_stats, time.monotonic()
        built = self._apex_grid(last, position, yaw, pxz, O)
        if built is None:
            return
        self.apex_due, self.apex_blk = False, 0
        grid, to_world, agent, Og = built
        res = AA.plan_object_path(grid, agent, Og)
        st["apex_plans"] += 1
        st["apex_ms_max"] = round(max(st["apex_ms_max"], (time.monotonic() - t0) * 1000), 1)
        if res is None:
            st["apex_fail"] += 1
            self.apex_wp = None
            self.nav.set_waypoint(None)
            return
        mc = st["apex_mode_counts"]
        mc[res["mode"]] = mc.get(res["mode"], 0) + 1
        wp_g = AA.waypoint(grid, res["path"], agent, 1.0, self.apex_los)
        self.apex_wp = to_world(wp_g)
        self.nav.set_waypoint(self.apex_wp)
        st["apex_path_len_last"] = round(AA.path_length(res["path"]), 2)
        st["apex_goal_to_O"] = round(float(np.linalg.norm(Og - res["goal"], axis=1).min()), 2)
        if len(st["apex_log"]) < 6 or (st["apex_plans"] % 25 == 0 and len(st["apex_log"]) < 30):
            f = lambda v: [round(float(x), 2) for x in v]
            e = dict(step=self.actions, src=st["apex_src"], mode=res["mode"], agent=f(agent), path0=f(res["path"][0]),
                     path_end=f(res["path"][-1]), n=len(res["path"]), goal=f(res["goal"]), obj=f(res["obj"]),
                     wp=f(wp_g), wp_world=f(self.apex_wp), goal_to_O=st["apex_goal_to_O"], agent_cell=AA.cell_class(grid, agent))
            if len(st["apex_log"]) < 6:       # coordinate check: where O and our own occupied cells land on the planning grid
                e["O_cells"] = AA.cell_fractions(grid, Og)
                if st["apex_src"] == "apex" and self.nav.lmap is not None:
                    e["own_occ_on_apex_occ_or_infl"] = self._xcheck(grid, last, position, yaw, pxz)
            st["apex_log"].append(e)

    def _xcheck(self, grid, last, position, yaw, pxz):
        """Share of our LocalMap's occupied cells within 3 m of the agent that fall on ApexNav's occupied|inflated cells (and within 1 cell)."""
        lm = self.nav.lmap
        o, _ = lm._derived(False)
        r0, c0 = (int(v) for v in lm.cell(pxz))
        k = int(3.0 / lm.res)
        sl = (slice(max(r0 - k, 0), r0 + k + 1), slice(max(c0 - k, 0), c0 + k + 1))
        rr, cc = np.nonzero(o[sl])
        if not len(rr):
            return None
        w = lm.world(rr + sl[0].start, cc + sl[1].start)
        pf = self.world_to_pf(w, position, yaw, last["pose"])
        r, c = grid.index(pf)
        ok = grid.inside(r, c)
        hit = np.zeros(len(r), bool)
        blocked = grid.occ | grid.infl
        from scipy import ndimage
        near = ndimage.binary_dilation(blocked, iterations=1)
        hit[ok] = near[r[ok], c[ok]]
        return dict(n=int(len(r)), frac=round(float(hit.mean()), 3))

    def result(self):
        info = {k: v for k, v in (self.nav.info or {}).items()}
        out = dict(actions=self.actions, reason=info.get("reason"), object_dist=info.get("object_dist"),
                   cloud_points=int(len(self.cloud)), cloud_source=self.source, **{"start_" + k: v for k, v in self.start.items()})
        if self.rh:
            out.update(rh=True, rh_k=self.k, rh_replans=self.replans, rh_looks=self.looks, rh_look_turns=self.look_turns,
                       rh_seen_steps=self.seen, rh_frames_with_masks=self.with_masks, rh_frames=self.t,
                       rh_last_seen=self.last_seen, rh_cloud_points=int(len(self.cloud)), rh_refined=self.refined,
                       stop_evidence=self.evidence)
            if self.apex:
                out.update(apex_map=self.apex_map, **{k: v for k, v in self.apex_stats.items() if not k.startswith("_")})
        else:
            out["rh"] = False
        return out
