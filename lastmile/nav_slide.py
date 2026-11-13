"""Software sliding on top of the F_full_cur DD-PPO navigator (no privileged info: only own pose + 'forward did not displace').
Modes (LM_SL_MODE):
  escape: after LM_SL_TRIG consecutive blocked FORWARDs, sweep the 12 headings (nearest-to-goal first), test each with one FORWARD;
          take the first that displaces >= LM_SL_MINDISP, keep going LM_SL_COMMIT steps in it, then reset the policy state.
  stair : on a blocked FORWARD, test heading offsets (+-30, +-60, +-90 around the blocked heading, goal side first); after the first that
          displaces, turn back to the original heading and hand over to the policy (motion-level sliding).
"""
import math
import os

import numpy as np
import torch

from lastmile.nav_ddppo import DDPPONavigator, FORWARD, LEFT, RIGHT, _pose, _yaw_of

TURN = math.radians(float(os.environ.get("LASTMILE_TURN_DEG", 30.0)))


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class SlideNavigator(DDPPONavigator):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.mode = os.environ.get("LM_SL_MODE", "escape")
        self.trig = int(os.environ.get("LM_SL_TRIG", 2))
        self.mind = float(os.environ.get("LM_SL_MINDISP", 0.2))
        self.commit = int(os.environ.get("LM_SL_COMMIT", 4))
        self.cool_n = int(os.environ.get("LM_SL_COOL", 6))
        self.blocked_d = 0.125
        self.crawl_d = float(os.environ.get("LM_SL_CRAWL", 0.03))     # smallest displacement that counts as progress
        self.release_d = float(os.environ.get("LM_SL_RELEASE", 0.5))  # leave crawl mode once this far from the wedge point
        self.max_sweeps = int(os.environ.get("LM_SL_SWEEPS", 6))

    def reset(self, obs=None, start_info=None):
        super().reset(obs, start_info)
        self.sl = None
        self.nb, self.cool, self.n_esc, self.n_blk = 0, 0, 0, 0
        self.p_pos, self.p_act, self.theta = None, None, 0.0

    def _inject(self, a, obs, pxz):
        self.prev = torch.tensor([[a]], dtype=torch.long, device=self.device)
        self.first = False
        self.last_act = None
        self.p_act, self.p_pos = a, np.asarray(obs["position"], float).copy()
        self.info["esc"] = self.n_esc
        self.info["blk"] = self.n_blk
        return a

    def _turn_to(self, tgt, yaw):
        d = round(wrap(tgt - yaw) / TURN)
        return None if d == 0 else (LEFT if d > 0 else RIGHT)

    def act(self, obs):
        o = _pose(obs)
        pos = np.asarray(o["position"], float)
        yaw = _yaw_of(o)
        pxz = pos[[0, 2]]
        if self.cloud is None and o.get("goal_cloud") is not None:
            self.cloud = np.asarray(o["goal_cloud"], float).reshape(-1, 3)
        moved = None
        if self.p_act == FORWARD and self.p_pos is not None:
            moved = float(np.linalg.norm(pos - self.p_pos))
        sl = self.sl
        if moved is not None and sl is None:
            if moved < self.blocked_d:
                self.nb += 1
                self.n_blk += 1
            elif moved >= self.mind:
                self.nb = 0
        if self.cool > 0:
            self.cool -= 1
        if sl is None and self.cool == 0 and self.nb >= (1 if self.mode == "stair" else self.trig) and moved is not None and moved < self.blocked_d \
                and self.object_distance(pxz) > self.stop_radius:
            self.n_esc += 1
            th = self.theta                       # bearing to the goal at the last policy step (>0 left)
            if self.mode in ("escape", "crawl"):
                ks = sorted(range(-5, 7), key=lambda k: abs(wrap(k * TURN - th)))
                T = [wrap(yaw + k * TURN) for k in ks]
            else:
                s = 1 if th >= 0 else -1
                T = [wrap(yaw + k * TURN) for k in (s, -s, 2 * s, -2 * s, 3 * s, -3 * s)]
            sl = self.sl = dict(T=T, i=0, st="turn", h0=yaw, n=0, p0=pos.copy(), tabu=set(), vis=[pos.copy()], sw=0, pb=pos.copy(), hb=None)
        elif sl is not None and self.mode == "crawl" and sl["st"] in ("eval", "commit"):
            self._crawl_result(sl, pos, yaw, moved)
            sl = self.sl
        elif sl is not None and sl["st"] == "eval":                   # result of the test FORWARD
            if moved is not None and moved >= self.mind:
                if self.mode == "escape":
                    sl.update(st="commit", n=0)
                else:
                    sl.update(st="restore")
            else:
                sl["i"] += 1
                sl["st"] = "turn"
                if sl["i"] >= len(sl["T"]):
                    self._end_sl(); sl = None
        elif sl is not None and sl["st"] == "commit":
            sl["n"] += 1
            if (moved is not None and moved < self.blocked_d) or sl["n"] >= self.commit:
                self._end_sl(); sl = None
        if sl is not None and self.object_distance(pxz) <= self.stop_radius:
            self._end_sl(); sl = None
        if sl is not None:
            self._update_map(o, pxz)
            if sl["st"] == "restore":
                a = self._turn_to(sl["h0"], yaw)
                if a is None:
                    self._end_sl()
                else:
                    return self._inject(a, o, pxz)
            elif sl["st"] == "commit":
                if self.mode == "crawl":
                    sl["pb"] = pos.copy()
                return self._inject(FORWARD, o, pxz)
            else:
                a = self._turn_to(sl["T"][sl["i"]], yaw)
                if a is None:
                    sl["st"] = "eval"
                    sl["pb"] = pos.copy()
                    sl["hb"] = int(round(sl["T"][sl["i"]] / TURN)) % 12
                    return self._inject(FORWARD, o, pxz)
                return self._inject(a, o, pxz)
        a = super().act(obs)
        self.theta = self.info.get("theta", self.theta)
        self.p_act, self.p_pos = a, pos.copy()
        self.info["esc"] = self.n_esc
        self.info["blk"] = self.n_blk
        return a

    def _crawl_result(self, sl, pos, yaw, moved):
        """tabu bug-style crawl: tiny displacements count; headings that lead back to visited points are tabu."""
        key = (tuple(np.round(sl["pb"][[0, 2]] / 0.1).astype(int)), sl["hb"])
        if sl["st"] == "eval":
            ok = moved is not None and moved >= self.crawl_d
            if ok and any(np.linalg.norm(pos - v) < 0.03 for v in sl["vis"][:-1]):
                sl["tabu"].add(key)
                ok = False
            if ok:
                sl["vis"].append(pos.copy())
                sl["st"], sl["n"] = "commit", 0
                return
            sl["tabu"].add(key)
            sl["i"] += 1
            sl["st"] = "turn"
        else:                                                         # commit: keep going while it displaces
            sl["n"] += 1
            if moved is not None and moved >= self.crawl_d and sl["n"] < self.commit:
                sl["vis"].append(pos.copy())
                return
            sl["st"] = "turn"
            sl["i"] = len(sl["T"])
        if np.linalg.norm(pos - sl["p0"]) >= self.release_d or sl["sw"] >= self.max_sweeps:
            self._end_sl()
            return
        if sl["i"] >= len(sl["T"]):                                   # new sweep from the current heading, goal first
            sl["sw"] += 1
            th = self.theta
            ks = sorted(range(-5, 7), key=lambda k: abs(wrap(k * TURN - th)))
            sl["T"] = [wrap(yaw + k * TURN) for k in ks]
            sl["i"] = 0
        while sl["i"] < len(sl["T"]):                                 # skip tabu headings at this spot
            hb = int(round(sl["T"][sl["i"]] / TURN)) % 12
            if (tuple(np.round(pos[[0, 2]] / 0.1).astype(int)), hb) in sl["tabu"]:
                sl["i"] += 1
            else:
                break
        if sl["i"] >= len(sl["T"]):
            sl["T"], sl["i"] = sl["T"], 0
            sl["sw"] += 1
            sl["tabu"] = set()

    def _end_sl(self):
        self.sl = None
        self.cool = self.cool_n
        self.nb = 0
        if self.mode in ("escape", "crawl"):                    # fresh recurrent state: the policy forgets the wedge
            self.hidden = torch.zeros_like(self.hidden)
            self.first = True
            self.prev = torch.zeros_like(self.prev)


def make():
    if os.environ.get("LASTMILE_SEED"):
        torch.manual_seed(int(os.environ["LASTMILE_SEED"]))
    nav = SlideNavigator(os.environ.get("LASTMILE_GOAL", "nearest"), float(os.environ.get("LASTMILE_STOP_R", 0.8)),
                         os.environ.get("LASTMILE_DEVICE", "cpu"))
    nav.name = "slide"
    return nav
