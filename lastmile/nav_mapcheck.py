"""Diagnostics navigator: plain sampled DD-PPO (nearest cloud point) that also keeps the LocalMap and, at the accept pose
(first step) and the final pose, compares it with the GT navmesh (false obstacles, cut-off goal component). --nav mapcheck.
Env: MAPCHECK_DIR (png/npz output), MAPCHECK_STRIDE (cells, default 2)."""
import os

import numpy as np
from scipy import ndimage

from lastmile.nav_ddppo import DDPPONavigator, _yaw_of, _pose


def navgrid(sim, m, rc_box, y, stride):
    """navigable bool + distance to navmesh edge (capped 1.0) on every stride-th cell of the box."""
    r0, r1, c0, c1 = rc_box
    rr, cc = np.meshgrid(np.arange(r0, r1, stride), np.arange(c0, c1, stride), indexing="ij")
    xz = m.world(rr.ravel(), cc.ravel())
    pf = sim.pathfinder
    nav = np.zeros(len(xz), bool)
    ed = np.zeros(len(xz), np.float32)
    for i, (x, z) in enumerate(xz):
        p = np.array([x, y, z], np.float32)
        if pf.is_navigable(p, 0.2):
            nav[i] = True
            ed[i] = min(pf.distance_to_closest_obstacle(p, 1.0), 1.0)
    return rr, cc, nav.reshape(rr.shape), ed.reshape(rr.shape)


class MapCheckNavigator(DDPPONavigator):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self._info, self.stats, self.meta = {}, {}, {}
        os.environ.setdefault("LM_DEBUG_H", "1")

    @property
    def info(self):
        if getattr(self, "_in_act", True):
            return self._info
        if not getattr(self, "_final", False):          # bench reads info once, after the episode ended
            self._final_stats()
        return {**self._info, **self.stats}

    @info.setter
    def info(self, v):
        self._info = v

    def reset(self, obs=None, start_info=None):
        super().reset(obs, start_info)
        self.stats, self.meta, self._final, self.track = {}, dict(start_info or {}), False, []
        self.did_first = False

    def act(self, obs):
        o = _pose(obs)
        pos = np.asarray(o["position"], float)
        if o.get("depth") is not None:
            self._update_map(o, pos[[0, 2]])
            self._upd = True
        self.track.append(pos.copy())
        self._last_obs = o
        # avoid double-updating the map inside goal code (only goal_mode freemap/snap2 would): this nav uses nearest
        if not self.did_first:
            self.did_first = True
            self.stats.update(self._measure(o, "acc"))
        self._in_act = True
        try:
            return super().act(obs)
        finally:
            self._in_act = False

    def _final_stats(self):
        self.stats.update(self._measure(self._last_obs, "fin", save=True))
        self._final = True

    def _measure(self, o, tag, save=False):
        m = self.lmap
        pos = np.asarray(o["position"], float)
        pxz, y = pos[[0, 2]], float(pos[1])
        cl = self.cloud[:, [0, 2]]
        ob, dist = m._derived(True)
        lo = np.minimum(pxz, cl.min(0)) - 3.0
        hi = np.maximum(pxz, cl.max(0)) + 3.0
        for p in self.track:
            lo, hi = np.minimum(lo, p[[0, 2]] - 2.5), np.maximum(hi, p[[0, 2]] + 2.5)
        (ra, ca), (rb, cb) = m.cell(lo), m.cell(hi)
        box = (max(ra, 0), min(rb + 1, m.n), max(ca, 0), min(cb + 1, m.n))
        st = int(os.environ.get("MAPCHECK_STRIDE", 2))
        rr, cc, nav, ed = navgrid(self.sim, m, box, y, st)
        hard = dist[rr, cc] < (m.radius + m.margin)
        raw = ob[rr, cc]
        good = nav & (ed >= 0.25)                                  # navmesh-free, >=0.25 m from the navmesh edge
        known = m.free[rr, cc] | raw
        out = {}
        fb = hard & good
        anyn = hard & nav                                            # blocked although navmesh-free (any edge distance)
        out[f"{tag}_any_blocked"] = int(anyn.sum())
        out[f"{tag}_any_blocked_frac"] = round(float(anyn.sum() / max(nav.sum(), 1)), 4)
        out[f"{tag}_any_blocked_known_frac"] = round(float((anyn & known).sum() / max((nav & known).sum(), 1)), 4)
        out[f"{tag}_any_edge_lt10"] = int((anyn & (ed < 0.10)).sum())             # within 0.10 m of the navmesh edge (thin passages)
        fraw = raw & good
        out[f"{tag}_n_good"] = int(good.sum())
        out[f"{tag}_false_blocked"] = int(fb.sum())
        out[f"{tag}_false_blocked_frac"] = round(float(fb.sum() / max(good.sum(), 1)), 4)
        gk = good & known
        out[f"{tag}_false_blocked_known_frac"] = round(float((fb & known).sum() / max(gk.sum(), 1)), 4)
        out[f"{tag}_false_raw"] = int(fraw.sum())
        hmin, hmax = m.hmin[rr, cc], m.hmax[rr, cc]
        out[f"{tag}_fr_floor"] = int((fraw & (hmin < 0.15)).sum())                # a low hit (floor band)
        out[f"{tag}_fr_top"] = int((fraw & (hmin >= 0.15) & (hmax > 1.3)).sum())     # only high hits
        out[f"{tag}_fr_mid"] = int((fraw & (hmin >= 0.15) & (hmax <= 1.3)).sum())
        out[f"{tag}_fr_nearwall"] = int((fraw & (ed < 0.5)).sum())                  # within 0.5 m of the navmesh edge (edge smear)
        out[f"{tag}_fr_open"] = int((fraw & (ed >= 0.5)).sum())
        out[f"{tag}_halo_only"] = int((fb & ~raw).sum())                            # blocked only by inflation of a (false) hit
        # component test: is the navmesh-snapped goal reachable from the agent over the map's passable cells?
        n = cl[np.argmin(np.linalg.norm(cl - pxz, axis=1))]
        sp = np.asarray(self.sim.pathfinder.snap_point(np.array([n[0], y, n[1]], np.float32)), float)
        sp = sp[[0, 2]] if np.isfinite(sp).all() else n
        r0, c0 = m.cell(pxz)
        rg, cg = m.cell(sp)
        for nm, passable in (("unk", ~(dist < (m.radius + m.margin))), ("free", m.free & ~(dist < (m.radius + m.margin)))):
            pa = passable.copy()
            pa[max(r0 - 3, 0):r0 + 4, max(c0 - 3, 0):c0 + 4] = True
            lab, _ = ndimage.label(pa, structure=np.ones((3, 3), bool))
            near = lab[max(rg - 3, 0):rg + 4, max(cg - 3, 0):cg + 4]
            out[f"{tag}_goal_conn_{nm}"] = int(bool((near == lab[r0, c0]).any()) if m.inside(r0, c0) else False)
        if save and os.environ.get("MAPCHECK_DIR"):
            d = os.environ["MAPCHECK_DIR"]
            os.makedirs(d, exist_ok=True)
            np.savez_compressed(os.path.join(d, f"{self.meta.get('scene')}_{self.meta.get('ep')}.npz"), rr=rr, cc=cc, nav=nav, ed=ed,
                                hard=hard, raw=raw, known=known, hmin=hmin, hmax=hmax, track=np.array(self.track), cloud=cl, snap=sp,
                                pos=pxz, res=m.res, stride=st, origin=m.o)
        return out


def make():
    import torch
    if os.environ.get("LASTMILE_SEED"):
        torch.manual_seed(int(os.environ["LASTMILE_SEED"]))
    nav = MapCheckNavigator("nearest", 0.8, os.environ.get("LASTMILE_DEVICE", "cpu"))
    nav.name = "mapcheck"
    return nav
