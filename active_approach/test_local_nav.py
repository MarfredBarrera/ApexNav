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


if __name__ == "__main__":
    unittest.main()
