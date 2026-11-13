import math
import unittest

import numpy as np

from lastmile import nav_ddppo as N


def obs(pos=(0, 0.0, 0), yaw=0.0, d=3.0):
    return {"depth": np.full((480, 640), d, np.float32),
            "pose": {"position": np.array(pos, float), "yaw": yaw, "pitch": 0.0}}


class T(unittest.TestCase):
    def test_actions_deterministic(self):
        cloud = np.array([[0.0, 1, -4.0], [0.2, 1, -4.2]])
        res = []
        for _ in range(2):
            nav = N.DDPPONavigator("nearest")
            nav.reset(None, {"cloud": cloud})
            res.append([nav.act(obs()) for _ in range(3)])
        self.assertEqual(res[0], res[1])
        self.assertTrue(all(a in (0, 1, 2, 3) for a in res[0]))

    def test_theta_sign(self):
        nav = N.DDPPONavigator("centroid")
        nav.reset(None, {"cloud": np.array([[-3.0, 1, 0.0]])})   # yaw 0 faces -z; x<0 is to the left
        nav.act(obs()); self.assertGreater(nav.info["theta"], 0)
        nav.reset(None, {"cloud": np.array([[3.0, 1, 0.0]])})
        nav.act(obs()); self.assertLess(nav.info["theta"], 0)

    def test_stop_radius_and_modes(self):
        cloud = np.array([[x, 1, z] for x in (-.3, .3) for z in (-2.3, -1.7)])
        nav = N.DDPPONavigator("footprint", stop_radius=0.8)
        nav.reset(None, {"cloud": cloud})
        self.assertEqual(nav.act(obs(pos=(0, 0, -1.0))), N.STOP)
        g = nav.goal_xz(np.array([0.0, 0.0]))
        self.assertAlmostEqual(g[1], -1.7, places=2)

    def test_retarget_keeps_map_resets_memory(self):
        nav = N.DDPPONavigator("freemap")
        nav.reset(None, {"cloud": np.array([[0.0, 1, -4.0]])})
        for _ in range(3):
            nav.act(obs())
        nav.lmap = "MAP"; nav.n_patch = 2
        nav.snap, nav.snap_d, nav.best_rho, nav.best_step = np.zeros(2), 1.0, 2.0, 1
        nav.hidden += 1.0
        steps = nav.steps
        new = np.array([[1.0, 1, -3.0]])
        nav.retarget(new)
        self.assertEqual((nav.lmap, nav.n_patch, nav.steps), ("MAP", 2, steps))
        self.assertTrue(nav.first and nav.snap is None and nav.snap_d is None and nav.best_rho > 1e8)
        self.assertEqual(nav.best_step, steps)
        self.assertEqual(float(nav.hidden.abs().sum()), 0.0)
        np.testing.assert_allclose(nav.cloud, new)

    def test_rh_returns_want_stop(self):
        cloud = np.array([[0.0, 1, -1.5]])
        for rh, want in ((False, N.STOP), (True, N.WANT_STOP)):
            nav = N.DDPPONavigator("nearest", stop_radius=0.8)
            nav.rh = rh
            nav.reset(None, {"cloud": cloud})
            self.assertEqual(nav.act(obs(pos=(0, 0, -1.0))), want)
            self.assertEqual(nav.info["reason"], "radius")


class _Nav:
    """Stub navigator: always wants a radius STOP; counts retargets."""
    info = {"reason": "radius", "object_dist": 0.4}
    rt = 0
    def act(self, o): return N_WANT
    def retarget(self, cloud=None): self.rt += 1


N_WANT = N.WANT_STOP


class RHGate(unittest.TestCase):
    def tk(self, refined, last_seen):
        from lastmile import live
        t = live.Takeover.__new__(live.Takeover)
        t.nav, t.rh, t.k, t.actions, t.cloud = _Nav(), True, 8, 0, np.array([[0.0, 1, -2.0]], np.float32)
        t.refined, t.t, t.last_seen, t.seen, t.with_masks = refined, 20, last_seen, 0, 0
        t.replans, t.looks, t.look_turns, t.look, t.rt_at, t.evidence = 0, 0, 0, None, 0, None
        t.S = type("S", (), {"agent_pose": staticmethod(lambda sim: (np.zeros(3), 0.0, 0.0))})
        t.sim, t.obs = None, lambda d: {}
        t._observe = lambda last: setattr(t, "t", t.t + 1)   # no detections
        return t

    def test_confirmed_stop(self):
        t = self.tk(True, 18)
        self.assertEqual(t.act(None, {}), N.STOP)
        self.assertEqual((t.evidence, t.looks), ("seen", 0))

    def test_unrefined_looks_then_stops(self):
        t = self.tk(False, None)
        acts = [t.act(None, {}) for _ in range(14)]
        self.assertEqual(acts[:12].count(N.STOP), 0)
        self.assertTrue(all(a in (N.LEFT, N.RIGHT) for a in acts[:12]))
        self.assertEqual(acts[12], N.STOP)
        self.assertEqual((t.evidence, t.looks, t.look_turns), ("look_empty", 1, 12))

    def test_look_finds_target_retargets(self):
        t = self.tk(False, None)
        t.act(None, {}); t.act(None, {})
        t._observe = lambda last: (setattr(t, "t", t.t + 1), setattr(t, "refined", True), setattr(t, "last_seen", t.t))
        self.assertEqual(t.act(None, {}), N.STOP)       # re-seen -> retarget -> navigator still wants STOP, now confirmed
        self.assertEqual((t.nav.rt, t.evidence), (1, "look_seen"))


if __name__ == "__main__":
    unittest.main()
