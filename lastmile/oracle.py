"""Oracle navigator (upper bound): uses GT object points and the navmesh. Not a valid baseline for
perception-limited navigators; it only validates the harness (should succeed ~100%)."""
import numpy as np

from lastmile import bench


class OracleNavigator(bench.Navigator):
    name = "oracle"
    uses_gt = True

    def set_sim(self, sim):
        self.sim = sim

    def set_gt(self, gt_points):
        self.gt = np.asarray(gt_points, np.float32)

    def reset(self, obs, info):
        import habitat_sim
        pf, p0 = self.sim.pathfinder, obs["pose"]["position"]
        from scipy.spatial import cKDTree
        lo, hi = pf.get_bounds()
        res = 0.1
        view = pf.get_topdown_view(res, float(p0[1]))          # (nz, nx) navigable cells at the agent's floor
        iz, ix = np.nonzero(view)
        cells = np.stack([lo[0] + (ix + 0.5) * res, np.full(len(ix), p0[1]), lo[2] + (iz + 0.5) * res], 1).astype(np.float32)
        tree = cKDTree(self.gt[:, [0, 2]])
        d, _ = tree.query(cells[:, [0, 2]])
        cands = cells[d <= 0.99]
        if len(cands) > 4000:
            cands = cands[np.random.RandomState(0).choice(len(cands), 4000, replace=False)]
        best = None
        self.goal = None
        good = []
        for c in cands:
            sn = np.array(pf.snap_point(c))
            if np.isfinite(sn).all() and abs(sn[1] - p0[1]) < 0.5 and np.hypot(sn[0] - c[0], sn[2] - c[2]) < 0.1:
                dd, dy = bench.score_point(sn, self.gt)
                if dd <= 0.99 and dy <= bench.DY_MAX:
                    good.append(sn)
        cands = good
        for s in cands:
            sp = habitat_sim.ShortestPath()
            sp.requested_start, sp.requested_end = p0.astype(np.float32), s
            if pf.find_path(sp) and (best is None or sp.geodesic_distance < best):
                best, self.goal = sp.geodesic_distance, s
        self.geo, self.ncand, self.plan = best, len(cands), []
        if self.goal is not None:
            self.plan = self._plan(np.asarray(p0, float), obs["pose"]["yaw"])
        self.plan_full = list(self.plan)
        self.expect, self.replans = None, 0
        self.feasible = bool(self.plan) or self._at_goal(p0)

    def info(self):
        return dict(oracle_cands=self.ncand, oracle_geo=round(self.geo, 2) if self.geo else "", oracle_plan=len(self.plan_full), oracle_reach_exhausted=int(getattr(self, "exhausted", False)))

    def _plan(self, p, yaw):
        plan = self._astar(p, yaw) or []
        if bench.ACTIONS == "fine":
            # the fine set is a superset: never do worse than the std set (guards against search-cap misses)
            std = self._astar(p, yaw, std_only=True) or []
            if std and (not plan or len(std) < len(plan)):
                plan = std
        return plan

    def _at_goal(self, p):
        d, dy = bench.score_point(p, self.gt)
        return d <= 0.99 and dy <= bench.DY_MAX

    def _astar(self, p0, yaw0, max_expand=300000, weight=1.2, std_only=False):
        """Exact action plan: A* over the real simulator's dynamics (set_state + act), 30-degree headings."""
        import heapq
        import habitat_sim
        ag = self.sim.get_agent(0)
        st0 = ag.get_state()
        g = self.goal
        table = bench.action_table()
        if std_only:
            table = {k: v for k, v in table.items() if k <= bench.RIGHT}
        step = bench.FORWARD_M
        res = 15.0 if (bench.ACTIONS == "fine" and not std_only) else 30.0

        deg0 = np.degrees(2 * np.arctan2(st0.rotation.y, st0.rotation.w))

        def key(p, rot):
            yaw = round((np.degrees(2 * np.arctan2(rot.y, rot.w)) - deg0) / res) % int(round(360 / res))      # heading index
            return (round(float(p[0]), 2), round(float(p[2]), 2), yaw)

        def h(p):
            return max(0.0, np.hypot(p[0] - g[0], p[2] - g[2]) - 0.1) / step * weight

        k0 = key(st0.position, st0.rotation)
        best, parent = {k0: 0}, {k0: None}
        heap = [(h(st0.position), 0, 0, st0.position.copy(), st0.rotation)]
        cnt, tie, plan = 0, 0, None
        while heap and cnt < max_expand:
            f, gc, _, p, rot = heapq.heappop(heap)
            kk = key(p, rot)
            if best.get(kk, 1e9) < gc:
                continue
            cnt += 1
            if self._at_goal(p):
                acts = []
                while parent[kk] is not None:
                    kk, a = parent[kk]
                    acts.append(a)
                plan = acts[::-1]
                break
            for a in table:
                s_ = habitat_sim.AgentState()
                s_.position, s_.rotation = p, rot
                ag.set_state(s_, reset_sensors=False)
                ag.act(table[a][0])
                n = ag.get_state()
                q = np.array(n.position)
                if table[a][1] and np.hypot(*(q - p)[[0, 2]]) < 0.005:      # fully blocked; partial steps are real motion
                    continue
                nk = key(q, n.rotation)
                if best.get(nk, 1e9) > gc + 1:
                    best[nk] = gc + 1
                    parent[nk] = (kk, a)
                    tie += 1
                    heapq.heappush(heap, (gc + 1 + h(q), gc + 1, tie, q, n.rotation))
        ag.set_state(st0, reset_sensors=False)
        self.exhausted = plan is None and not heap
        if plan is None and heap and weight < 3:          # expansion cap hit: retry greedier
            return self._astar(p0, yaw0, max_expand, weight=3.0, std_only=std_only)
        self.expanded = cnt
        return plan

    def act(self, obs):
        # closed loop: if the sim left the planned trajectory (clearance-0 starts are sensitive), replan from the actual pose
        if self.plan and self.expect is not None:
            p = obs["pose"]["position"]
            if np.hypot(*(p - self.expect)[[0, 2]]) > 0.03 and self.replans < 10:
                self.replans += 1
                self.plan = self._plan(np.asarray(p, float), obs["pose"]["yaw"])
        if not self.plan:
            return bench.STOP
        a = self.plan.pop(0)
        yaw = obs["pose"]["yaw"]
        p = np.asarray(obs["pose"]["position"], float)
        fm = bench.action_table()[a][1]
        self.expect = p + (fm * np.array([-np.sin(yaw), 0, -np.cos(yaw)]) if fm else 0)
        return a
