import unittest

import numpy as np

from lastmile import apex_astar as A


def grid(n=160, res=0.05):
    occ = np.zeros((n, n), bool)
    return occ, res


def make(occ, res=0.05, infl_r=0.18, known=None):
    from scipy import ndimage
    dist = ndimage.distance_transform_edt(~occ) * res if occ.any() else np.full(occ.shape, 99.0)
    return A.Grid((0.0, 0.0), res, occ, (dist < infl_r) & ~occ, np.ones_like(occ) if known is None else known)


class T(unittest.TestCase):
    def test_open_room(self):
        g = make(grid()[0])
        r = A.plan_object_path(g, (1.0, 4.0), [(5.0, 4.0)])
        self.assertEqual(r["mode"], "normal_0.50")
        self.assertLess(np.linalg.norm(r["goal"] - [5.0, 4.0]), 0.55)
        self.assertLess(np.linalg.norm(r["path"][-1] - r["goal"]), 0.25)
        self.assertLess(A.path_length(r["path"]), 4.6)

    def test_wall_with_gap(self):
        occ, res = grid()
        occ[:, 78:82] = True            # wall at x=3.9..4.1, full height
        occ[60:90, 78:82] = False       # gap at y 3.0..4.5
        g = make(occ)
        r = A.plan_object_path(g, (1.0, 6.0), [(6.0, 6.0)])
        self.assertIsNotNone(r)
        self.assertTrue(r["mode"].startswith("normal"))
        p = r["path"]
        cross = p[np.argmin(np.abs(p[:, 0] - 4.0))]
        self.assertTrue(3.0 <= cross[1] <= 4.5, cross)
        wp = A.waypoint(g, p, p[0], 1.0)
        self.assertLessEqual(np.linalg.norm(wp - p[0]), 1.05)

    def test_goal_inside_inflated_backs_off(self):
        occ, res = grid()
        occ[70:90, 110:130] = True      # an object block x 5.5..6.5, y 3.5..4.5
        g = make(occ)
        r = A.plan_object_path(g, (1.0, 4.0), [(5.6, 4.0)])
        self.assertIsNotNone(r)
        occ_, unk, inf = g.state(r["goal"])
        self.assertFalse(occ_ or unk or inf)
        self.assertGreater(np.linalg.norm(r["goal"] - [5.6, 4.0]), 0.3)

    def test_unknown_blocks_then_extreme(self):
        occ, res = grid()
        known = np.ones_like(occ)
        known[:, 78:82] = False         # unknown band across the room
        g = make(occ, known=known)
        self.assertIsNone(A.astar(g, (1.0, 4.0), (6.0, 4.0), 0.5, mode=A.NORMAL))
        self.assertIsNotNone(A.astar(g, (1.0, 4.0), (6.0, 4.0), 0.5, mode=A.OPTIMISTIC))
        r = A.plan_object_path(g, (1.0, 4.0), [(6.0, 4.0)])
        self.assertEqual(r["mode"], "extreme")

    def test_extreme_through_wall(self):
        occ, res = grid()
        occ[:, 78:82] = True
        g = make(occ)
        self.assertIsNone(A.astar(g, (1.0, 4.0), (6.0, 4.0), 0.5, mode=A.OPTIMISTIC))
        r = A.plan_object_path(g, (1.0, 4.0), [(6.0, 4.0)])
        self.assertEqual(r["mode"], "extreme")
        self.assertLess(np.linalg.norm(r["path"][-1] - [6.0, 4.0]), 0.3)

    def test_outside_map_unsafe(self):
        g = make(grid()[0])
        self.assertIsNone(A.astar(g, (1.0, 4.0), (20.0, 4.0), 0.5, mode=A.EXTREME))

    def test_speed(self):
        occ, res = grid(480)
        occ[:, 200:204] = True
        occ[100:140, 200:204] = False
        g = make(occ)
        import time
        t = time.time()
        r = A.plan_object_path(g, (2.0, 6.0), [(20.0, 6.0)])
        self.assertIsNotNone(r)
        self.assertLess(time.time() - t, 0.6)


if __name__ == "__main__":
    unittest.main()
