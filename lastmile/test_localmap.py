"""Unit tests on a synthetic 2D world (boxes with height ranges) rendered to depth, plus closed-loop runs of the
A* and MPPI navigators on it. Run: python -m unittest lastmile.test_localmap (inside the apexnav container)."""
import math
import unittest

import numpy as np

from lastmile import localmap as LM
from lastmile.nav_astar import AStarNavigator
from lastmile.nav_mppi import MPPINavigator

CAM_H, F = 0.88, (640 / 2) / math.tan(math.radians(79) / 2)


class World:
    """Boxes (x0, x1, z0, z1, y0, y1) in habitat coords (floor y = 0); kinematic agent radius 0.18, no sliding."""

    def __init__(self, boxes, pos, yaw=0.0, pitch=0.0):
        self.boxes = np.array(boxes, float).reshape(-1, 6)
        self.pos, self.yaw, self.pitch = np.array([pos[0], 0.0, pos[1]]), yaw, pitch

    def blocked(self, xz):
        b = self.boxes
        r = 0.18
        hit = (xz[0] > b[:, 0] - r) & (xz[0] < b[:, 1] + r) & (xz[1] > b[:, 2] - r) & (xz[1] < b[:, 3] + r) & (b[:, 4] < 0.3)
        return bool(hit.any())

    def act(self, a):
        if a == LM.FORWARD:
            n = self.pos[[0, 2]] + LM.FWD_M * LM.fwd_vec(self.yaw)
            if not self.blocked(n):
                self.pos[[0, 2]] = n
        elif a == LM.LEFT:
            self.yaw += LM.TURN
        elif a == LM.RIGHT:
            self.yaw -= LM.TURN

    def depth(self):
        h, w = 120, 160
        vv, uu = np.mgrid[0:h, 0:w]
        u, v = uu * 4 + 2, vv * 4 + 2
        d_cam = np.stack([(u - 320) / F, -(v - 240) / F, -np.ones_like(u, float)], -1)
        cp, sp = math.cos(self.pitch), math.sin(self.pitch)
        d = d_cam @ np.array([[1, 0, 0], [0, cp, sp], [0, -sp, cp]]).T
        cy, sy = math.cos(self.yaw), math.sin(self.yaw)
        d = d @ np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]]).T
        o = np.array([self.pos[0], CAM_H, self.pos[2]])
        t = np.full(d.shape[:2], 50.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            tf = np.where(d[..., 1] < 0, -o[1] / d[..., 1], np.inf)
            t = np.minimum(t, tf)
            for x0, x1, z0, z1, y0, y1 in self.boxes:
                lo, hi = np.zeros(d.shape[:2]), np.full(d.shape[:2], np.inf)
                for ax, a0, a1 in ((0, x0, x1), (1, y0, y1), (2, z0, z1)):
                    t0, t1 = (a0 - o[ax]) / d[..., ax], (a1 - o[ax]) / d[..., ax]
                    t0, t1 = np.where(np.isnan(t0), -np.inf, t0), np.where(np.isnan(t1), np.inf, t1)
                    lo, hi = np.maximum(lo, np.minimum(t0, t1)), np.minimum(hi, np.maximum(t0, t1))
                t = np.where((lo <= hi) & (lo > 0), np.minimum(t, lo), t)
        depth = t * np.abs(self._zc(d))
        depth[depth > 49] = 0.0
        return np.repeat(np.repeat(depth, 4, 0), 4, 1).astype(np.float32)

    def _zc(self, d):
        """Camera-frame z magnitude of each ray direction (planar depth = t * |z_cam|)."""
        cy, sy = math.cos(self.yaw), math.sin(self.yaw)
        cp, sp = math.cos(self.pitch), math.sin(self.pitch)
        dc = d @ np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])      # undo yaw
        dc = dc @ np.array([[1, 0, 0], [0, cp, sp], [0, -sp, cp]])      # undo pitch (R^T applied via transpose twice)
        return dc[..., 2]

    def obs(self, cloud):
        return {"depth": self.depth(), "pose": {"position": self.pos.copy(), "yaw": self.yaw, "pitch": self.pitch},
                "goal_cloud": cloud}


def wall_with_door(z=-3.0, door=(0.5, 1.5), x=(-5, 5)):
    return [(x[0], door[0], z - 0.1, z + 0.1, 0, 2.5), (door[1], x[1], z - 0.1, z + 0.1, 0, 2.5)]


def object_cloud(xz, half=0.2):
    g = np.linspace(-half, half, 5)
    return np.array([[xz[0] + a, 0.5, xz[1] + b] for a in g for b in g])


def run(nav, world, cloud, steps=120, budget=None):
    obs = world.obs(cloud)
    nav.reset(obs, {"goal_cloud": cloud, "budget_steps": budget or steps, "camera": {"height_m": CAM_H}})
    n = 0
    for _ in range(steps):
        a = nav.act(obs)
        if a == LM.STOP:
            break
        world.act(a)
        n += 1
        obs = world.obs(cloud)
    p = world.pos[[0, 2]]
    return n, float(np.linalg.norm(cloud[:, [0, 2]] - p, axis=1).min()), a == LM.STOP


def look_at(world, cloud, nav_map=None):
    m = LM.LocalMap(world.pos[[0, 2]])
    m.update(world.pos, world.yaw, world.pitch, world.depth())
    return m


class MapTests(unittest.TestCase):
    def test_wall_blocks(self):
        w = World(wall_with_door(door=(9, 9.1)), (0, 0), 0.0)         # solid wall 3 m ahead (-z)
        m = look_at(w, None)
        o = m.occ()
        r, c = m.cell(np.array([0.0, -2.92]))
        self.assertTrue(o[r, c])
        r, c = m.cell(np.array([0.0, -1.5]))
        self.assertFalse(o[r, c])
        self.assertTrue(m.free[r, c])

    def test_door_gap(self):
        w = World(wall_with_door(door=(0.0, 1.2)), (0.6, 0), 0.0)
        m = look_at(w, None)
        cost, hard = m.costs(np.array([0.6, 0.0]))
        r, c = m.cell(np.array([0.6, -3.0]))
        self.assertTrue(np.isfinite(cost[r, c]))                       # gap centre is passable
        r, c = m.cell(np.array([-1.0, -3.0]))
        self.assertTrue(hard[r, c])

    def test_low_obstacle_seen(self):
        w = World([(-0.3, 0.3, -2.1, -1.9, 0.0, 0.15)], (0, 0), 0.0)  # 15 cm step: the old 0.28 m band misses it
        m = look_at(w, None)
        r, c = m.cell(np.array([0.0, -2.0]))
        self.assertTrue(m.occ()[r, c])

    def test_collision_patch(self):
        m = LM.LocalMap(np.array([0.0, 0.0]))
        pos = np.array([0.0, 0.0, 0.0])
        m.mark_collision(pos, 0.0)
        r, c = m.cell(np.array([0.0, -0.25]))
        self.assertTrue(m.occ()[r, c])
        r, c = m.cell(np.array([0.3, -0.25]))
        self.assertFalse(m.occ()[r, c])
        cost, hard = m.costs(np.array([0.0, 0.0]))
        r, c = m.cell(np.array([0.0, -0.5]))
        self.assertTrue(hard[r, c])

    def test_los_through_wall(self):
        w = World(wall_with_door(door=(9, 9.1)), (0, 0), 0.0)
        m = look_at(w, None)
        self.assertFalse(m.los(np.array([0.0, 0.0]), np.array([[0.0, -4.0]]))[0])
        self.assertTrue(m.los(np.array([0.0, 0.0]), np.array([[0.0, -2.0]]))[0])


class NavTests(unittest.TestCase):
    def _door_case(self, mk):
        w = World(wall_with_door(door=(0.5, 1.5)) + [(-5, 5, 0.9, 1.0, 0, 2.5)], (-1.5, 0.0), 0.0)
        cloud = object_cloud((-1.5, -3.7))                              # directly behind the wall, door is 2.5 m to the right
        n, d, stopped = run(mk(), w, cloud, steps=150)
        return n, d, stopped, w

    def test_astar_door(self):
        n, d, stopped, w = self._door_case(lambda: AStarNavigator())
        self.assertTrue(stopped)
        self.assertLessEqual(d, 1.0)
        self.assertLess(w.pos[2], -2.9)                                  # went through the doorway

    def test_mppi_door(self):
        n, d, stopped, w = self._door_case(lambda: MPPINavigator())
        self.assertTrue(stopped)
        self.assertLessEqual(d, 1.0)

    def test_refuses_through_wall_stop(self):
        w = World(wall_with_door(door=(9, 9.1)), (0, -2.55), 0.0)       # 0.6 m from the wall, object 0.8 m behind it
        cloud = object_cloud((0.0, -3.7))
        nav = AStarNavigator()
        obs = w.obs(cloud)
        nav.reset(obs, {"goal_cloud": cloud, "budget_steps": 100, "camera": {"height_m": CAM_H}})
        for _ in range(3):
            a = nav.act(obs)
            self.assertNotEqual(a, LM.STOP)                              # 1.0 m euclid but a wall in between
            w.act(a)
            obs = w.obs(cloud)

    def test_low_obstacle_detour(self):
        w = World([(-1, 1, -2.0, -1.8, 0.0, 0.15)], (0, 0), 0.0)         # 15 cm high bar the native band ignores
        cloud = object_cloud((0.0, -4.0))
        n, d, stopped = run(AStarNavigator(), w, cloud, steps=80)
        self.assertTrue(stopped)
        self.assertLessEqual(d, 1.0)

    def test_deterministic(self):
        res = []
        for _ in range(2):
            w = World(wall_with_door(), (-1.5, 0.0), 0.0)
            res.append(run(MPPINavigator(seed=3), w, object_cloud((0.0, -4.0)), steps=60)[:2])
        self.assertEqual(res[0], res[1])


if __name__ == "__main__":
    unittest.main()
