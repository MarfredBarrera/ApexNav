"""python -m unittest active_approach.test_local_nav (no Habitat needed)."""
import math
import unittest

import numpy as np

from active_approach import geom as G
from active_approach import local_nav as L


def wall_map():
    """A 7 m map around (0, 0): everything seen and free, with a wall x = 1 m from z = -2 to 2 (1 m high)."""
    hm = G.HeightMap(np.zeros(2), 0.0)
    hm.seen[:] = True
    ij, ok = hm.cells(np.stack([np.full(81, 1.0), np.linspace(-2, 2, 81)], 1))
    for di in (0, 1):
        hm.top[ij[ok, 0] + di, ij[ok, 1]] = 1.0
        hm.low[ij[ok, 0] + di, ij[ok, 1]] = 1.0
    lm = L.LocalMap(hm)
    pf = L.LocalPathfinder(lm, 0.0)
    pf.refresh(np.zeros(3))
    return lm, pf


class LocalNavTest(unittest.TestCase):
    def test_path_goes_around_wall(self):
        lm, pf = wall_map()
        path = pf.ShortestPath()
        path.requested_start, path.requested_end = np.array([0.0, 0, 0]), np.array([2.0, 0, 0])
        self.assertTrue(pf.find_path(path))
        self.assertGreater(path.geodesic_distance, 4.0)          # around the 4 m wall, not through it
        for p in path.points[1:-1]:
            self.assertFalse(0.8 < p[0] < 1.3 and -2.0 < p[2] < 2.0)

    def test_snap_out_of_wall(self):
        lm, pf = wall_map()
        q = pf.snap_point(np.array([1.0, 0.0, 0.0]))
        self.assertTrue(np.all(np.isfinite(q)))
        self.assertGreaterEqual(abs(q[0] - 1.0), 0.2)

    def test_unknown_is_navigable_but_occupied_is_not(self):
        lm, pf = wall_map()
        self.assertFalse(pf.is_navigable(np.array([1.0, 0, 0])))
        self.assertTrue(pf.is_navigable(np.array([-1.0, 0, 0])))

    def test_bump_blocks_cell_ahead(self):
        lm, pf = wall_map()
        lm.bump(np.array([-2.0, 0.0, 0.0]), 0.0)      # yaw 0 looks along -z
        pf.refresh(np.array([-2.0, 0.0, 0.0]))
        self.assertFalse(pf.is_navigable(np.array([-2.0, 0.0, -0.3])))


    def test_wall_seen_above_body_top_is_occupied(self):
        """A wall seen up to 2.2 m: its max height is above the base's band, but its points below BODY_TOP block.
        (Before 2026-11-09 only the max height was tested, so walls and tables seen up to the ceiling were free.)"""
        hm = G.HeightMap(np.zeros(2), 0.0)
        z = np.linspace(-1.0, 1.0, 41)
        h = np.linspace(0.0, 2.2, 23)
        pts = np.array([[1.0, y, zz] for y in h for zz in z])
        orig = G.S.backproject
        G.S.backproject = lambda depth, valid, cam_pos, cam_rot, stride=2: pts
        try:
            hm.update(np.ones((4, 4), np.float32), np.zeros((4, 4), bool), np.array([0.0, 0.88, 0.0]), np.eye(3))
        finally:
            G.S.backproject = orig
        lm = L.LocalMap(hm)
        occupied, _, _ = lm.grids()
        self.assertTrue(occupied[lm.cell([1.0, 0.0])])
        self.assertGreater(float(hm.top[tuple(hm.cells(np.array([1.0, 0.0]))[0])]), G.BODY_TOP)

    def test_partial_forward_step_is_contact(self):
        """No sliding: a step that stopped short (0.18 of 0.25 m) touched something ahead of where it stopped."""
        lm, pf = wall_map()
        start, end = np.array([-2.0, 0.0, 0.0]), np.array([-2.0, 0.0, -0.18])   # yaw 0 looks along -z
        self.assertTrue(lm.forward_feedback(start, 0.0, end))
        pf.refresh(end)
        self.assertFalse(pf.is_navigable(np.array([-2.0, 0.0, -0.18 - 0.3])))
        self.assertTrue(pf.is_navigable(np.array([-2.0, 0.0, 0.5])))           # behind is still open
        lm2, _ = wall_map()
        self.assertFalse(lm2.forward_feedback(start, 0.0, np.array([-2.0, 0.0, -0.25])))   # a full step: no mark
        self.assertFalse(lm2.bumps.any())

    def test_inflation_covers_radius_plus_half_cell_diagonal(self):
        """Blocked within the agent radius + half a cell diagonal of an occupied cell's centre (0.224 m: the
        logged blocked steps), open at 0.283 m."""
        hm = G.HeightMap(np.zeros(2), 0.0)
        hm.seen[:] = True
        lm = L.LocalMap(hm)
        c = lm.cell([0.0, 0.0])
        lm.bumps[c] = True
        cost = lm.cost_grid(np.array([3.0, 3.0]))
        self.assertFalse(np.isfinite(cost[c[0] + 2, c[1] + 1]))    # sqrt(5) cells = 0.224 m
        self.assertTrue(np.isfinite(cost[c[0] + 2, c[1] + 2]))     # sqrt(8) cells = 0.283 m
        self.assertTrue(np.isfinite(cost[c[0] + 3, c[1]]))


if __name__ == "__main__":
    unittest.main()
