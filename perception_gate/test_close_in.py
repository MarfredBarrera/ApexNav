"""Terminal close-in decisions and loop (no simulator): python -m unittest perception_gate.test_close_in"""
import math
import unittest

import numpy as np

from active_approach import geom as G
from active_approach import scene as S
from params import FINAL_RESULT
from perception_gate import close_in as C
from perception_gate.gate import VerificationGate

FREE = math.inf
CLOUD = np.array([[0.0, 0.5, -1.0], [0.1, 0.5, -1.0]])   # 1 m ahead of the origin (yaw 0 looks along -z)
ORIGIN = np.zeros(3)


def decide(position=ORIGIN, yaw=0.0, cloud=CLOUD, free=FREE, blocks=0, used=0, budget=12):
    return C.decide(position, yaw, cloud, free, blocks, used, budget)


class DecideTests(unittest.TestCase):
    def test_stop_when_near(self):
        self.assertEqual(decide(position=np.array([0.0, 0.0, -0.65])), ("stop", "reached"))   # 0.35 m
        self.assertEqual(decide(position=np.array([0.0, 0.0, -0.55]))[0], "move_forward")      # 0.45 m

    def test_turns_toward_a_bearing_beyond_15_degrees(self):
        left = decide(yaw=-math.radians(30))   # object is to the left of a yaw turned right
        self.assertEqual(left[0], "turn_left")
        self.assertEqual(decide(yaw=math.radians(30))[0], "turn_right")
        self.assertEqual(decide(yaw=math.radians(10))[0], "move_forward")

    def test_forward_needs_free_space(self):
        need = S.FORWARD_M + S.AGENT_RADIUS + C.CLEAR_MARGIN_M
        self.assertEqual(decide(free=need + 0.01)[0], "move_forward")
        self.assertEqual(decide(free=need - 0.01), ("stop", "blocked"))

    def test_blocks_and_budget_stop(self):
        self.assertEqual(decide(blocks=2), ("stop", "blocked"))
        self.assertEqual(decide(blocks=1)[0], "move_forward")
        self.assertEqual(decide(used=12, budget=12), ("stop", "budget"))
        self.assertEqual(decide(budget=0), ("stop", "budget"))

    def test_free_distance_band_ignores_floor_off_axis_and_object(self):
        depth = np.full((S.HEIGHT, S.WIDTH), 4.0)
        self.assertEqual(C.free_distance(depth, np.zeros(depth.shape, bool)), 4.0)
        wall = depth.copy()
        wall[:, S.WIDTH // 2 - 10:S.WIDTH // 2 + 10] = 0.4   # a pole in the middle of the band
        self.assertAlmostEqual(C.free_distance(wall, np.zeros(wall.shape, bool)), 0.4)
        self.assertEqual(C.free_distance(wall, wall < 1.0), 4.0)   # ... unless it is the object
        side = depth.copy()
        side[:, :S.WIDTH // 2 - 100] = 1.0   # far off to the side: outside the footprint
        self.assertEqual(C.free_distance(side, np.zeros(side.shape, bool)), 4.0)
        floor = depth.copy()   # a flat floor: row v sees it at z = camera height * focal / (v - centre)
        rows = np.arange(S.HEIGHT // 2 + 1, S.HEIGHT)
        floor[rows] = (S.CAMERA_HEIGHT * S.FOCAL / (rows - S.HEIGHT / 2))[:, None]
        self.assertEqual(C.free_distance(floor, np.zeros(floor.shape, bool)), 4.0)

    def test_trigger(self):
        near, far = ORIGIN, np.array([5.0, 0.0, 5.0])
        self.assertEqual(C.trigger_reason(True, FINAL_RESULT.REACH_OBJECT, False, near, CLOUD), "stop")
        self.assertEqual(C.trigger_reason(True, FINAL_RESULT.STUCKING, False, near, CLOUD), "stucking")
        self.assertIsNone(C.trigger_reason(True, FINAL_RESULT.REACH_OBJECT, True, near, CLOUD))      # step cap
        self.assertIsNone(C.trigger_reason(False, FINAL_RESULT.REACH_OBJECT, False, near, CLOUD))    # not a STOP
        self.assertIsNone(C.trigger_reason(True, FINAL_RESULT.NO_FRONTIER, False, near, CLOUD))
        self.assertIsNone(C.trigger_reason(True, FINAL_RESULT.REACH_OBJECT, False, far, CLOUD))      # not that object
        self.assertIsNone(C.trigger_reason(True, FINAL_RESULT.REACH_OBJECT, False, near, None))      # no accept


class DriveTests(unittest.TestCase):
    def run_drive(self, budget=12, blocked=False, start=ORIGIN):
        """A sim stand-in: moves of 0.25 m along yaw, turns of 30 deg; a blocked robot does not move."""
        st = {"p": np.array(start, float), "yaw": 0.0, "log": []}

        def act(name):
            st["log"].append(name)
            if name == "move_forward" and not blocked:
                st["p"] = st["p"] + S.FORWARD_M * np.array([G.forward(st["yaw"])[0], 0.0, G.forward(st["yaw"])[1]])
            elif name != "move_forward":
                st["yaw"] += math.radians(30) * {"turn_left": 1, "turn_right": -1, "stop": 0}[name]
            return {"depth": np.full((S.HEIGHT, S.WIDTH), 4.0, np.float32)}

        view = lambda: (st["p"], st["yaw"], st["p"] + [0, S.CAMERA_HEIGHT, 0], np.eye(3))
        out = C.drive(CLOUD, act, view, {"depth": np.full((S.HEIGHT, S.WIDTH), 4.0, np.float32)},
                      budget)
        return out, st["log"]

    def test_walks_up_and_stops(self):
        out, log = self.run_drive()
        self.assertEqual((out["end"], log[-1]), ("reached", "stop"))
        self.assertLessEqual(out["end_d"], C.CLOSE_IN_STOP_M)
        self.assertEqual(out["actions"], len(log))

    def test_turns_first(self):
        out, log = self.run_drive(start=(1.0, 0.0, -1.0))   # the object is 90 deg to the left... at 1 m
        self.assertEqual(log[0], "turn_left")
        self.assertEqual(log[-1], "stop")

    def test_blocked_stops_after_two_blocks(self):
        out, log = self.run_drive(blocked=True)
        self.assertEqual((out["end"], out["blocks"], log), ("blocked", 2, ["move_forward", "move_forward", "stop"]))

    def test_budget(self):
        out, log = self.run_drive(budget=1)
        self.assertEqual((out["end"], log), ("budget", ["move_forward", "stop"]))


class GateTests(unittest.TestCase):
    def test_switch_and_counters(self):
        ping = lambda url, timeout: {"ok": True, "settings": {"selector": "vqf"}}   # (no server is contacted)
        self.assertFalse(VerificationGate({"enabled": True}, "v", ping=ping).close_in)
        gate = VerificationGate({"enabled": True, "close_in": True}, "v", ping=ping)
        gate.start_episode(1, "scene", "e", "chair")
        self.assertEqual((gate.counts["close_in"], gate.counts["close_in_actions"]), (0, 0))
        gate.remember_object(np.ones((5, 3)), 1)
        self.assertEqual(len(gate.accepted_cloud), 5)
        gate.log_close_in("stop", {"start_d": 0.5, "end_d": 0.3, "actions": 3, "blocks": 0, "end": "reached"})
        self.assertEqual((gate.counts["close_in"], gate.counts["close_in_actions"], gate.counts["close_in_end"]),
                         (1, 3, "reached"))
        self.assertEqual(gate.pop_events()[0]["event"], "close_in")


class ObjectPointsTests(unittest.TestCase):
    def test_backprojects_the_matching_mask(self):
        depth = np.full((S.HEIGHT, S.WIDTH), 2.0, np.float32)
        mask = np.zeros(depth.shape, np.uint8)
        mask[200:280, 300:340] = 1
        frame = {"depth": depth, "masks": [mask], "centroids": [(1.0, 1.0)], "cam": (np.zeros(3), np.eye(3))}
        pts = C.object_points(frame, (1.2, 1.0), 1.0)
        self.assertTrue(np.allclose(pts[:, 2], -2.0))
        self.assertIsNone(C.object_points(frame, (5.0, 5.0), 1.0))


if __name__ == "__main__":
    unittest.main()
