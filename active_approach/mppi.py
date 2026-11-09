"""MPPI view planner (self-contained, RGB-D + odometry only).

Receding-horizon sampling planner over the robot's own action set (move_forward 0.25 m, turn 30 deg; no strafe,
no tilt). Every control step:
  1. the object geometry is the fused depth cloud of the masked pixels (``obs.cloud``) plus geometric phantoms
     where the current frame cuts the object (top cut: points up to the frustum top + 0.2 m; side cut: the rim
     shifted 0.3 m outwards). No category size priors, no VLM outputs;
  2. K action sequences of length H are sampled from a per-step categorical distribution and rolled out on the
     self-built occupancy grid (local_nav, inflated by the agent radius): a forward step whose swept cells are
     blocked leaves the agent in place and costs C_BUMP;
  3. each pose on a rollout is scored by the object's footprint in the projected camera frustum:
        J = (in-frame hull area / image area) x (unoccluded fraction) x completeness^KAPPA x facing
     completeness = in-frame hull area / full hull area (1 = the whole object is in frame), the unoccluded
     fraction from rays through the 2.5D height map (own depth), facing = soft penalty beyond 15 deg off centre;
     a rollout's value is max_t (log J_t - C_ACT t) - bumps;
  4. MPPI update (softmax of the values with temperature LAMBDA, ITERS iterations, warm-started by shifting);
     the executed action is the weighted majority at t = 0. The planner stops when no sampled rollout beats the
     current pose by more than DELTA (log units).
A blocked forward step (odometry did not move) marks the cell ahead on the local map, like LocalController.

    mppi     KAPPA 2 (footprint, the whole object should be in frame)
    mppi_k0  KAPPA 0 (pure footprint in the frustum)
    mppi_k4  KAPPA 4
    mppi_dial  KAPPA 2 with DIAL-MPC annealing (noise shrinks over iterations, grows along the horizon)
    mppi_v2  vertical-cut barrier from heights (category height prior until the top is seen), side-only
             completeness, cost-to-go to closer's fit pose, stop only on a usable view (class MppiV2)
    mppi_v4  MPPI around closer: closer's drive is the nominal sequence, sampled perturbations replace it only for a
             >= 10 % better view (class MppiV4)
    mppi_v5  v4 with a collision-aware nominal (cost-to-go descent on the lattice) and DELTA 0.3 (class MppiV5)
    mppi_v6  v5 with half the samples free (forward-biased) and back-out hysteresis (class MppiV6)
    mppi_v7  v6 with a frame-aware value: predicted usable but weak frames on the way cost (class MppiV7)
    mppi_v8  closer until it would stop, then v7's MPPI for a better view before the forced decision (class MppiV8)
    mppi_v3  v2 with the projection cost: bbox presence x horizontal centring x extent barriers (vectorised)
"""
import math

import numpy as np

from active_approach import geom as G
from active_approach import local_nav as L
from active_approach import scene as S

ACTIONS = ("move_forward", "turn_left", "turn_right")
IMG_AREA = float(S.WIDTH * S.HEIGHT)


def _hull_area_clipped(uv):
    """(full hull area, hull area inside the image) of 2D points ``uv`` (pixels)."""
    from scipy.spatial import ConvexHull
    if len(uv) < 3:
        return 0.0, 0.0
    try:
        hull = ConvexHull(uv)
    except Exception:
        return 0.0, 0.0
    poly = uv[hull.vertices]
    full = float(hull.volume)
    for axis, bound, keep_less in ((0, 0.0, False), (0, S.WIDTH, True), (1, 0.0, False), (1, S.HEIGHT, True)):
        if not len(poly):
            break
        out = []
        n = len(poly)
        for i in range(n):
            a, b = poly[i], poly[(i + 1) % n]
            ina = a[axis] <= bound if keep_less else a[axis] >= bound
            inb = b[axis] <= bound if keep_less else b[axis] >= bound
            if ina:
                out.append(a)
            if ina != inb:
                t = (bound - a[axis]) / (b[axis] - a[axis])
                out.append(a + t * (b - a))
        poly = np.asarray(out)
    if len(poly) < 3:
        return full, 0.0
    x, y = poly[:, 0], poly[:, 1]
    inside = 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))
    return full, min(inside, full)


class VoxelMap:
    """3D occupancy of the robot's own depth hits (10 cm voxels, 0-2.4 m above the floor), object pixels excluded.
    A voxel is occupied after >= 2 hits. Unlike the 2.5D max-height map it keeps overhangs (ceilings, lintels,
    table tops) from hiding what is under them."""
    CELL, NZ = 0.10, 24

    def __init__(self, center_xz, floor_y, half=8.0):
        self.origin = np.asarray(center_xz, float) - half
        self.n = int(2 * half / self.CELL)
        self.floor_y = floor_y
        self.hits = np.zeros((self.n, self.n, self.NZ), np.uint8)

    def update(self, depth, mask, position, yaw, stride=4):
        c, s = math.cos(yaw), math.sin(yaw)
        rot = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
        cam = np.array([position[0], self.floor_y + G.H_CAM, position[2]])
        valid = (depth > 0.05) & (depth < G.DEPTH_MAX) & ~mask
        pts = S.backproject(depth, valid, cam, rot, stride=stride)
        if not len(pts):
            return
        idx = self.index(pts[:, [0, 2]], pts[:, 1] - self.floor_y)
        ok = idx[3]
        np.add.at(self.hits, (idx[0][ok], idx[1][ok], idx[2][ok]), 1)
        np.minimum(self.hits, 50, out=self.hits)

    def index(self, xz, h):
        ij = np.floor((np.asarray(xz) - self.origin) / self.CELL).astype(int)
        k = np.floor(np.asarray(h) / self.CELL).astype(int)
        ok = (ij >= 0).all(-1) & (ij < self.n).all(-1) & (k >= 0) & (k < self.NZ)
        return ij[..., 0], ij[..., 1], k, ok

    def occupied(self):
        return self.hits >= 2


class Mppi:
    name, enroute, driver, reason = "mppi", False, None, None
    KAPPA = 2.0
    H, K, ITERS = 8, 192, 3
    LAMBDA, C_ACT, C_BUMP, DELTA = 0.15, 0.02, 1.0, 0.03
    PRIOR = np.array([0.5, 0.25, 0.25])
    EPS = 0.1
    N_POINTS = 300

    def reset(self, ctx):
        self.pi = np.tile(self.PRIOR, (self.H, 1))
        self.prev = None          # (position, yaw, action) of the last call: bump detection
        self.last = None
        self.rng = np.random.default_rng(0)
        self.done = False
        self.vox = None

    # ---------------- object geometry (RGB-D only) ----------------
    def points(self, obs):
        cloud = np.asarray(obs.cloud)
        idx = self.rng.choice(len(cloud), min(len(cloud), self.N_POINTS), replace=False)
        pts = [cloud[idx]]
        mask = np.asarray(obs.mask, bool)
        info = {}
        if mask.sum() >= 200:
            f, right = G.forward(obs.yaw), np.array([math.cos(obs.yaw), -math.sin(obs.yaw)])
            here = np.asarray(obs.position)[[0, 2]]
            xz = cloud[:, [0, 2]]
            if mask[0].any():   # top cut: the object reaches at least the frustum top at its depth
                depth = max(float(np.median((xz - here) @ f)), 0.3)
                top = obs.floor_y + G.H_CAM + depth * math.tan(G.BETA_V) + 0.2
                foot = cloud[idx][:: max(1, len(idx) // 60)]
                pts.append(np.stack([foot[:, 0], np.full(len(foot), top), foot[:, 2]], 1))
                info["top"] = True
            proj = (xz - here) @ right
            for side, sign, edge, col in (("left", -1, proj.min(), 0), ("right", 1, proj.max(), -1)):
                if mask[:, col].any():
                    rim = cloud[np.abs(proj - edge) < 0.08]
                    rim = rim[:: max(1, len(rim) // 20)]
                    for t in (0.1, 0.2, 0.3):
                        shifted = rim.copy()
                        shifted[:, 0] += sign * t * right[0]
                        shifted[:, 2] += sign * t * right[1]
                        pts.append(shifted)
                    info[side] = True
        return np.concatenate(pts), info

    def object_cells(self, vox, cloud):
        from scipy.ndimage import binary_dilation
        grid = np.zeros((vox.n, vox.n), bool)
        i, j, _, ok = vox.index(cloud[:, [0, 2]], np.zeros(len(cloud)))
        grid[i[ok], j[ok]] = True
        return binary_dilation(grid, iterations=1)

    # ---------------- view score ----------------
    def score(self, poses, pts, center, vox, obj):
        """log J for poses (M, 3) = (x, z, yaw)."""
        cam = poses[:, :2]
        yaw = poses[:, 2]
        f = np.stack([-np.sin(yaw), -np.cos(yaw)], 1)
        r = np.stack([np.cos(yaw), -np.sin(yaw)], 1)
        d = pts[None, :, [0, 2]] - cam[:, None]                      # (M, P, 2)
        zc = np.einsum("mpk,mk->mp", d, f)
        xc = np.einsum("mpk,mk->mp", d, r)
        yc = (pts[:, 1] - (vox.floor_y + G.H_CAM))[None]
        front = zc > 0.1
        with np.errstate(divide="ignore", invalid="ignore"):
            u = S.WIDTH / 2 + S.FOCAL * xc / zc
            v = S.HEIGHT / 2 - S.FOCAL * np.broadcast_to(yc, zc.shape) / zc
        inframe = front & (u >= 0) & (u < S.WIDTH) & (v >= 0) & (v < S.HEIGHT)
        vis = self.unoccluded(cam, pts, vox, obj)                    # (M, P)
        out = np.full(len(poses), math.log(1e-4))
        for m in range(len(poses)):
            if inframe[m].sum() < 3:
                continue
            uv = np.stack([u[m, front[m]], v[m, front[m]]], 1)
            full, inside = _hull_area_clipped(uv)
            if inside <= 0 or full <= 0:
                continue
            completeness = (inside / full) * front[m].mean()
            unocc = vis[m, inframe[m]].mean()
            off = abs(G.wrap(G.bearing_yaw(cam[m], center) - yaw[m]))
            facing = math.exp(-(max(0.0, math.degrees(off) - 15.0) / 10.0) ** 2)
            J = inside / IMG_AREA * unocc * completeness ** self.KAPPA * facing
            out[m] = math.log(J + 1e-4)
        return out

    def unoccluded(self, cam, pts, vox, obj, step=0.05):
        """(M, P) bool: the ray camera -> point crosses no occupied voxel (own depth; object columns excluded)."""
        res = np.ones((len(cam), len(pts)), bool)
        occ = vox.occupied() & ~obj[..., None]
        h_cam = G.H_CAM
        h_pt = pts[:, 1] - vox.floor_y
        for a in range(0, len(cam), 32):
            c = cam[a:a + 32]
            d = pts[None, :, [0, 2]] - c[:, None]
            length = np.linalg.norm(d, axis=-1)                         # (m, P)
            s = np.arange(0.25, max(float(length.max()) - 0.15, 0.3), step)
            frac = s[None, None] / np.maximum(length[..., None], 1e-3)  # (m, P, S)
            valid = s[None, None] < (length[..., None] - 0.15)
            xy = c[:, None, None] + frac[..., None] * d[:, :, None]
            ray_h = h_cam + frac * (h_pt[None, :, None] - h_cam)
            i, j, k, ok = vox.index(xy, ray_h)
            ok &= valid
            blocked = np.zeros(ok.shape, bool)
            blocked[ok] = occ[i[ok], j[ok], k[ok]]
            res[a:a + 32] = ~blocked.any(-1)
        return res

    # ---------------- rollouts ----------------
    def rollout(self, seqs, x0, yaw0, pf):
        """Poses (K, H+1, 3) and bump counts (K,) of action sequences (K, H) on the inflated local grid."""
        K, H = seqs.shape
        lm = pf.lmap
        pos = np.tile(x0, (K, 1)).astype(float)
        k = np.zeros(K, int)
        poses = np.zeros((K, H + 1, 3))
        poses[:, 0, :2], poses[:, 0, 2] = pos, yaw0
        bumps = np.zeros(K)
        for t in range(H):
            a = seqs[:, t]
            k = k + (a == 1) - (a == 2)
            yaw = yaw0 + k * G.LATTICE
            fw = a == 0
            if fw.any():
                f = np.stack([-np.sin(yaw[fw]), -np.cos(yaw[fw])], 1)
                ok = np.ones(fw.sum(), bool)
                for frac in (0.5, 1.0):
                    q = pos[fw] + frac * S.FORWARD_M * f
                    ij = np.clip(np.floor((q - lm.origin) / L.CELL).astype(int), 0, lm.n - 1)
                    ok &= np.isfinite(pf.cost[ij[:, 0], ij[:, 1]])
                moved = pos[fw] + S.FORWARD_M * f
                newp = pos[fw]
                newp[ok] = moved[ok]
                pos[fw] = newp
                b = bumps[fw]
                b[~ok] += 1
                bumps[fw] = b
            poses[:, t + 1, :2], poses[:, t + 1, 2] = pos, yaw
        return poses, bumps

    def evaluate(self, seqs, x0, yaw0, pf, pts, center, hmap, obj):
        poses, bumps = self.rollout(seqs, x0, yaw0, pf)
        K, T, _ = poses.shape
        key = np.stack([np.round(poses[..., 0] / 0.02), np.round(poses[..., 1] / 0.02),
                        np.round((poses[..., 2] - yaw0) / G.LATTICE) % 12], -1).reshape(-1, 3).astype(np.int64)
        uniq, inv = np.unique(key, axis=0, return_inverse=True)
        inv = np.asarray(inv).ravel()
        first = np.zeros(len(uniq), int)
        first[inv[::-1]] = np.arange(len(inv))[::-1]
        flat = poses.reshape(-1, 3)
        todo = [i for i in range(len(uniq)) if tuple(uniq[i]) not in self.cache]
        if todo:
            vals = self.score(flat[first[todo]], pts, center, hmap, obj)
            for i, val in zip(todo, vals):
                self.cache[tuple(uniq[i])] = val
        sc = np.array([self.cache[tuple(q)] for q in uniq])[inv.ravel()].reshape(K, T)
        gain = sc - self.C_ACT * np.arange(T)[None]
        tstar = gain.argmax(1)
        values = gain.max(1) - self.C_BUMP * bumps
        return values, tstar, sc

    def _values_v1(self, poses, yaw0, pts, center, obj):
        """v1's view value max_t (log J_t - C_ACT t) per rollout, scores cached per pose (no bump cost)."""
        K, T, _ = poses.shape
        key = np.stack([np.round(poses[..., 0] / 0.02), np.round(poses[..., 1] / 0.02),
                        np.round((poses[..., 2] - yaw0) / G.LATTICE) % 12], -1).reshape(-1, 3).astype(np.int64)
        uniq, inv = np.unique(key, axis=0, return_inverse=True)
        inv = np.asarray(inv).ravel()
        first = np.zeros(len(uniq), int)
        first[inv[::-1]] = np.arange(len(inv))[::-1]
        flat = poses.reshape(-1, 3)
        todo = [i for i in range(len(uniq)) if tuple(uniq[i]) not in self.cache]
        if todo:
            vals = self.score(flat[first[todo]], pts, center, self.vox, obj)
            for i, val in zip(todo, vals):
                self.cache[tuple(uniq[i])] = val
        sc = np.array([self.cache[tuple(q)] for q in uniq])[inv].reshape(K, T)
        gain = sc - self.C_ACT * np.arange(T)[None]
        return gain.max(1), gain.argmax(1), sc

    def plan(self, obs):
        pf, hmap = obs.pathfinder, self.vox
        pts, info = self.points(obs)
        center = G.ObjectModel(np.asarray(obs.cloud), obs.floor_y).center
        obj = self.object_cells(self.vox, np.asarray(obs.cloud))
        x0 = np.asarray(obs.position)[[0, 2]].astype(float)
        H = max(1, min(self.H, obs.remaining))
        self.cache = {}
        pi = self.pi[:H]
        best_seq, best_val, best_t, s0 = None, -np.inf, 0, None
        for it in range(self.ITERS):
            eps = self.noise(it, H)[:, None]
            probs = (1 - eps) * pi + eps / 3
            cum = probs.cumsum(1)
            seqs = (self.rng.random((self.K, H, 1)) > cum[None, :, :]).sum(-1).clip(0, 2)
            if best_seq is not None:
                seqs[0] = best_seq
            values, tstar, sc = self.evaluate(seqs, x0, obs.yaw, pf, pts, center, hmap, obj)
            s0 = sc[0, 0]
            i = int(values.argmax())
            if values[i] > best_val:
                best_seq, best_val, best_t = seqs[i].copy(), float(values[i]), int(tstar[i])
            w = np.exp((values - values.max()) / self.temperature(it))
            w /= w.sum()
            onehot = np.eye(3)[seqs]                                     # (K, H, 3)
            pi = np.einsum("k,kha->ha", w, onehot)
        return pi, best_seq, best_val, best_t, s0, info

    def noise(self, it, H):
        """Uniform-mixing weight per horizon step (the sampling noise of a categorical distribution)."""
        return np.full(H, self.EPS)

    def temperature(self, it):
        return self.LAMBDA

    # ---------------- policy interface (run.py: act) ----------------
    def act(self, obs):
        pf = obs.pathfinder
        if self.vox is None:
            self.vox = VoxelMap(np.asarray(obs.position)[[0, 2]], obs.floor_y)
        self.vox.update(np.asarray(obs.depth), np.asarray(obs.mask, bool), obs.position, obs.yaw)
        if self.prev is not None and self.prev[2] == "move_forward":   # blocked step: mark the cell ahead
            moved = float(np.hypot(*(np.asarray(obs.position) - self.prev[0])[[0, 2]]))
            if moved < 0.5 * S.FORWARD_M:
                pf.lmap.bump(self.prev[0], self.prev[1])
                pf.refresh(obs.position)
        if self.done or not len(obs.cloud) or not hasattr(pf, "lmap"):
            self.last = {"stop": "no_object" if not len(obs.cloud) else "done"}
            return None
        pi, seq, val, tstar, s0, info = self.plan(obs)
        center = G.ObjectModel(np.asarray(obs.cloud), obs.floor_y).center
        off = math.degrees(abs(G.wrap(G.bearing_yaw(np.asarray(obs.position)[[0, 2]], center) - obs.yaw)))
        self.last = {"J0": round(float(np.exp(s0)), 4), "V": round(val, 3), "t*": tstar,
                     "seq": "".join("FLR"[a] for a in seq), "phantom": info}
        floor = math.log(1e-4) + 1e-6
        if val <= floor and s0 <= floor:   # no rollout sees the object (lost / far): face it and approach
            diff = G.wrap(G.bearing_yaw(np.asarray(obs.position)[[0, 2]], center) - obs.yaw)
            action = ("turn_left" if diff > 0 else "turn_right") if off > 20.0 else "move_forward"
            self.last["fallback"] = True
        elif tstar == 0 or val - s0 < self.DELTA:
            if off > 20.0:   # end facing the object (a judgeable frame)
                diff = G.wrap(G.bearing_yaw(np.asarray(obs.position)[[0, 2]], center) - obs.yaw)
                action = "turn_left" if diff > 0 else "turn_right"
            else:
                self.done = True
                self.last["stop"] = "best_view"
                return None
        else:
            action = ACTIONS[int(pi[0].argmax())]
        self.pi = np.vstack([pi[1:], np.tile(self.PRIOR, (self.H - len(pi) + 1, 1))])
        self.pi = 0.5 * self.pi + 0.5 * self.PRIOR
        self.prev = (np.asarray(obs.position).copy(), obs.yaw, action)
        self.last["action"] = action
        return action


class MppiDial(Mppi):
    """DIAL-MPC-style annealing (Xue et al., ICRA 2025, arXiv 2409.15610): ITERS reverse-diffusion iterations with
    noise eps(i, h) = EPS0 exp(-i / (B1 ITERS) - (H - 1 - h) / (B2 H)) : decreasing over iterations, larger for
    later horizon steps (the near future is refined, the far future stays exploratory); the softmax temperature
    is annealed the same way over iterations. Fewer samples per iteration keep the total cost of mppi."""
    name = "mppi_dial"
    ITERS, K = 6, 96
    EPS0, B1, B2, EPS_MIN = 0.9, 0.5, 1.0, 0.02
    LAMBDA_MIN = 0.03

    def noise(self, it, H):
        h = np.arange(H)
        eps = self.EPS0 * np.exp(-it / (self.B1 * self.ITERS) - (H - 1 - h) / (self.B2 * H))
        return np.clip(eps, self.EPS_MIN, 1.0)

    def temperature(self, it):
        t = it / max(self.ITERS - 1, 1)
        return float(self.LAMBDA * (self.LAMBDA_MIN / self.LAMBDA) ** t)


class MppiV2(Mppi):
    """v2 (2026-11-07, after Stage B): the failures were driving too close (the object left the frame vertically,
    which neither a turn nor a reverse step can undo) and stopping at once on a poor first view.
    - Vertical cut as a (steep) barrier, from heights only: at pitch 0 and forward distance d to the object's near
      face the frame spans heights H_CAM -/+ d tan(BETA_V); cut = (part above + part below) / height. The bottom
      may lose BOT_CUT of the height (floor objects lose their base below 1.4 m anyway), the top TOP_CUT; beyond that
      the pose's score drops by exp(-(excess / 0.03)^2). Until the top has been seen inside a frame (mask >= 400 px,
      not touching the top row), z_hi = max(observed, z_lo + the category height prior, round3.prior_hw).
    - Side cuts only: completeness = in-frame hull area / hull area clipped to the image rows (a turn fixes them).
    - Terminal cost-to-go: C_TOGO per 0.25 m of local-map path from the rollout's last pose to closer's fit pose,
      so rollouts make progress when no view within the horizon is better.
    - Stop only on a usable view (>= 400 mask px, facing within 20 deg) that no rollout beats; never on a poor one."""
    name = "mppi_v2"
    TOP_CUT, BOT_CUT, CUT_SOFT = 0.15, 0.25, 0.03
    C_TOGO = 0.02
    USABLE_PX, FACE_DEG = 400, 20.0

    def reset(self, ctx):
        super().reset(ctx)
        self.top_seen = False

    def heights(self, obs, pts):
        from active_approach.round3 import prior_hw
        h = np.asarray(obs.cloud)[:, 1] - obs.floor_y
        z_lo, z_hi = float(np.percentile(h, 5)), float(np.percentile(np.r_[h, pts[:, 1] - obs.floor_y], 95))
        if not self.top_seen:
            z_hi = max(z_hi, z_lo + prior_hw(obs.meta or {})[0])
        return z_lo, z_hi

    def vertical_factor(self, zc, front):
        """exp(-(excess cut / CUT_SOFT)^2) per pose from the near-face distance (pitch 0)."""
        d = np.where(front.any(1), np.where(front, zc, np.inf).min(1), 0.05)
        t = math.tan(G.BETA_V)
        height = max(self.z_hi - self.z_lo, 0.05)
        top = np.maximum(0.0, self.z_hi - (G.H_CAM + d * t)) / height
        bot = np.maximum(0.0, (G.H_CAM - d * t) - self.z_lo) / height
        excess = np.maximum(0.0, top - self.TOP_CUT) + np.maximum(0.0, bot - self.BOT_CUT)
        return np.exp(-(excess / self.CUT_SOFT) ** 2), top, bot

    def score(self, poses, pts, center, vox, obj):
        cam = poses[:, :2]
        yaw = poses[:, 2]
        f = np.stack([-np.sin(yaw), -np.cos(yaw)], 1)
        r = np.stack([np.cos(yaw), -np.sin(yaw)], 1)
        d = pts[None, :, [0, 2]] - cam[:, None]
        zc = np.einsum("mpk,mk->mp", d, f)
        xc = np.einsum("mpk,mk->mp", d, r)
        yc = (pts[:, 1] - (vox.floor_y + G.H_CAM))[None]
        front = zc > 0.1
        with np.errstate(divide="ignore", invalid="ignore"):
            u = S.WIDTH / 2 + S.FOCAL * xc / zc
            v = S.HEIGHT / 2 - S.FOCAL * np.broadcast_to(yc, zc.shape) / zc
        inframe = front & (u >= 0) & (u < S.WIDTH) & (v >= 0) & (v < S.HEIGHT)
        vis = self.unoccluded(cam, pts, vox, obj)
        vert, _, _ = self.vertical_factor(zc, front)
        out = np.full(len(poses), math.log(1e-4))
        for m in range(len(poses)):
            if inframe[m].sum() < 3:
                continue
            uv = np.stack([u[m, front[m]], v[m, front[m]]], 1)
            full, inside = _hull_area_clipped(uv)
            rows = uv.copy()
            rows[:, 1] = np.clip(rows[:, 1], 0, S.HEIGHT)          # cut vertically only: the side reference
            full_rows, _ = _hull_area_clipped(rows)
            if inside <= 0 or full_rows <= 0:
                continue
            side = min(1.0, inside / full_rows) * front[m].mean()
            unocc = vis[m, inframe[m]].mean()
            off = abs(G.wrap(G.bearing_yaw(cam[m], center) - yaw[m]))
            facing = math.exp(-(max(0.0, math.degrees(off) - 15.0) / 10.0) ** 2)
            J = inside / IMG_AREA * unocc * side ** self.KAPPA * facing * vert[m]
            out[m] = math.log(J + 1e-4)
        return out

    def togo_steps(self, obs, xz):
        """Forward steps (0.25 m) of local-map path from each xz (N, 2) to closer's fit pose."""
        if self.goal_cell is None:
            return np.zeros(len(xz))
        lm, pf = obs.pathfinder.lmap, obs.pathfinder
        dist, _ = pf._tree(self.goal_cell)
        ij = np.clip(np.floor((xz - lm.origin) / L.CELL).astype(int), 0, lm.n - 1)
        steps = dist[ij[:, 0] * lm.n + ij[:, 1]] * L.CELL / S.FORWARD_M
        return np.where(np.isfinite(steps), steps, 40.0)

    def evaluate(self, seqs, x0, yaw0, pf, pts, center, hmap, obj):
        poses, bumps = self.rollout(seqs, x0, yaw0, pf)
        values, tstar, sc = self._values(poses, bumps, yaw0, pts, center, hmap, obj)
        values = values - self.C_TOGO * self.togo_steps(self.obs, poses[:, -1, :2])
        return values, tstar, sc

    def fit_goal(self, obs):
        """xz of the cost-to-go target: closer's fit pose (the agent's own position once inside the fit range)."""
        from active_approach.selfcontained import Closer
        return np.asarray(Closer().line_goal(obs).position)[[0, 2]]

    def view_ok(self, obs, pts):
        """Whether the current view may end the approach (beyond usable): v2 accepts any usable view."""
        return True

    def _values(self, poses, bumps, yaw0, pts, center, hmap, obj):
        K, T, _ = poses.shape
        key = np.stack([np.round(poses[..., 0] / 0.02), np.round(poses[..., 1] / 0.02),
                        np.round((poses[..., 2] - yaw0) / G.LATTICE) % 12], -1).reshape(-1, 3).astype(np.int64)
        uniq, inv = np.unique(key, axis=0, return_inverse=True)
        inv = np.asarray(inv).ravel()
        first = np.zeros(len(uniq), int)
        first[inv[::-1]] = np.arange(len(inv))[::-1]
        flat = poses.reshape(-1, 3)
        todo = [i for i in range(len(uniq)) if tuple(uniq[i]) not in self.cache]
        if todo:
            vals = self.score(flat[first[todo]], pts, center, hmap, obj)
            for i, val in zip(todo, vals):
                self.cache[tuple(uniq[i])] = val
        sc = np.array([self.cache[tuple(q)] for q in uniq])[inv].reshape(K, T)
        gain = sc - self.C_ACT * np.arange(T)[None]
        return gain.max(1) - self.C_BUMP * bumps, gain.argmax(1), sc

    def act(self, obs):
        pf = obs.pathfinder
        mask = np.asarray(obs.mask, bool)
        if self.vox is None:
            self.vox = VoxelMap(np.asarray(obs.position)[[0, 2]], obs.floor_y)
        self.vox.update(np.asarray(obs.depth), mask, obs.position, obs.yaw)
        if self.prev is not None and self.prev[2] == "move_forward":
            moved = float(np.hypot(*(np.asarray(obs.position) - self.prev[0])[[0, 2]]))
            if moved < 0.5 * S.FORWARD_M:
                pf.lmap.bump(self.prev[0], self.prev[1])
                pf.refresh(obs.position)
        if self.done or not len(obs.cloud) or not hasattr(pf, "lmap"):
            self.last = {"stop": "no_object" if not len(obs.cloud) else "done"}
            return None
        here = np.asarray(obs.position)[[0, 2]]
        center = G.ObjectModel(np.asarray(obs.cloud), obs.floor_y).center
        off = math.degrees(abs(G.wrap(G.bearing_yaw(here, center) - obs.yaw)))
        usable = int(mask.sum()) >= self.USABLE_PX and off <= self.FACE_DEG
        if usable and not mask[0].any():
            self.top_seen = True
        # closer's fit pose: the cost-to-go target
        self.obs = obs
        g = self.fit_goal(obs)
        self.goal_cell = pf.lmap.cell(g) if np.isfinite(pf.cost[pf.lmap.cell(g)]) else None
        pts, _ = self.points(obs)
        self.z_lo, self.z_hi = self.heights(obs, pts)
        pi, seq, val, tstar, s0, info = self.plan(obs)
        usable = usable and self.view_ok(obs, pts)
        stay = s0 - self.C_TOGO * float(self.togo_steps(obs, here[None])[0])
        self.last = {"J0": round(float(np.exp(s0)), 4), "V": round(val, 3), "stay": round(stay, 3), "t*": tstar,
                     "seq": "".join("FLR"[a] for a in seq), "phantom": info, "usable": usable,
                     "z": [round(self.z_lo, 2), round(self.z_hi, 2)], "top_seen": self.top_seen}
        if usable and val - stay < self.DELTA:
            self.done = True
            self.last["stop"] = "best_view"
            return None
        if val - stay < self.DELTA and off > self.FACE_DEG:   # nothing better ahead: face the object
            diff = G.wrap(G.bearing_yaw(here, center) - obs.yaw)
            action = "turn_left" if diff > 0 else "turn_right"
        else:
            action = ACTIONS[int(seq[0])] if getattr(self, "BEST_SEQ", False) else ACTIONS[int(pi[0].argmax())]
        self.pi = np.vstack([pi[1:], np.tile(self.PRIOR, (self.H - len(pi) + 1, 1))])
        self.pi = 0.5 * self.pi + 0.5 * self.PRIOR
        self.prev = (np.asarray(obs.position).copy(), obs.yaw, action)
        self.last["action"] = action
        return action


class MppiV3(MppiV2):
    """v3 (user's cost, 2026-11-07): keep the object's projection centred in the image and large, fully in frame.
    Per pose, from the projected object points (the top raised to the category height prior until it is seen):
      presence   in-frame area of the projection's bounding box / image area (larger is better)
      centring   c = (bbox centre u - W/2) / (W/2): horizontal only (pitch 0: the vertical position is set by
                 height and distance, not by the heading); factor exp(-W_CENTER c^2)
      extent     horizontal: farthest projected pixel from the centre <= EXTENT x half width (soft barrier);
                 vertical: v2's height barrier (top cut <= 15 %, bottom <= 25 %)
      J = presence x unoccluded fraction x exp(-W_CENTER c^2) x barriers, all vectorised over poses (no hulls)
    plus v2's cost-to-go to closer's fit pose and its stop rule (only on a usable view)."""
    name = "mppi_v3"
    W_CENTER, EXTENT, EXT_SOFT = 2.0, 0.95, 0.05
    H = 10
    DELTA = 0.1          # keep moving only for a >= 10 % better view (log units)
    BEST_SEQ = True      # execute the best rollout's first action (the weighted majority dithered L/R)
    ANCHOR_PAD, PCT = 0.75, 2.0

    def reset(self, ctx):
        super().reset(ctx)
        self.pi = np.tile(self.PRIOR, (self.H, 1))
        self.anchor = None

    def act(self, obs):
        """The object model is anchored to the trigger frame's object: cloud points farther (xz) than its half
        extent + ANCHOR_PAD from its centre are dropped (a GT-free mask bleeds onto neighbouring furniture)."""
        import dataclasses
        cloud = np.asarray(obs.cloud)
        if len(cloud) and self.anchor is None:
            model = G.ObjectModel(cloud, obs.floor_y)
            self.anchor = (model.center, 0.5 * float(np.hypot(*model.extent)) + self.ANCHOR_PAD)
        if self.anchor is not None and len(cloud):
            keep = np.hypot(*(cloud[:, [0, 2]] - self.anchor[0]).T) <= self.anchor[1]
            if keep.sum() >= 10:
                obs = dataclasses.replace(obs, cloud=cloud[keep])
        return super().act(obs)

    def fit_goal(self, obs):
        """The fit pose on the current bearing, also when the agent is inside it (it then backs out)."""
        from active_approach.selfcontained import _model, fit_radius, unit
        model, info = _model(obs)
        here = np.asarray(obs.position)[[0, 2]]
        u = unit(here - model.center)
        return model.center + fit_radius(obs, model, info, u) * u

    def view_ok(self, obs, pts):
        """From this position the object fits the frame: the predicted vertical cut is within the limits and its
        horizontal angular width fits the field of view (5 deg margin each side); otherwise only backing out helps
        (no tilt, no reverse), so a truncated close view never ends the approach."""
        cam = np.asarray(obs.position)[[0, 2]]
        f = G.forward(obs.yaw)
        zc = ((pts[:, [0, 2]] - cam) @ f)[None]
        vbar, _, _ = self.vertical_factor(zc, zc > 0.1)
        d = pts[:, [0, 2]] - cam
        center = np.median(d, axis=0)
        ang = np.arctan2(d[:, 0] * center[1] - d[:, 1] * center[0], d @ center)   # signed angle to the centre ray
        width = float(np.percentile(ang, 98) - np.percentile(ang, 2))
        return bool(vbar[0] >= 0.5 and width <= 2 * (G.BETA_H - math.radians(5)))

    def score(self, poses, pts, center, vox, obj):
        cam = poses[:, :2]
        yaw = poses[:, 2]
        f = np.stack([-np.sin(yaw), -np.cos(yaw)], 1)
        r = np.stack([np.cos(yaw), -np.sin(yaw)], 1)
        d = pts[None, :, [0, 2]] - cam[:, None]
        zc = np.einsum("mpk,mk->mp", d, f)
        xc = np.einsum("mpk,mk->mp", d, r)
        yc = (pts[:, 1] - (vox.floor_y + G.H_CAM))[None]
        front = zc > 0.1
        zs = np.where(front, zc, 1.0)
        u = S.WIDTH / 2 + S.FOCAL * xc / zs
        v = S.HEIGHT / 2 - S.FOCAL * np.broadcast_to(yc, zc.shape) / zs
        near = np.where(front, zc, np.inf).min(1)
        near = np.where(np.isfinite(near), near, 0.1)
        v_top = S.HEIGHT / 2 - S.FOCAL * (self.z_hi - G.H_CAM) / near           # the (prior) top at the near face
        uf = np.where(front, u, np.nan)
        vf = np.where(front, v, np.nan)
        with np.errstate(all="ignore"):
            u0, u1 = np.nanpercentile(uf, self.PCT, 1), np.nanpercentile(uf, 100 - self.PCT, 1)
            v0, v1 = np.minimum(np.nanpercentile(vf, self.PCT, 1), v_top), np.nanpercentile(vf, 100 - self.PCT, 1)
        u0, u1, v0, v1 = (np.nan_to_num(a, nan=S.WIDTH / 2) for a in (u0, u1, v0, v1))
        box_in = (np.clip(u1, 0, S.WIDTH) - np.clip(u0, 0, S.WIDTH)) * (np.clip(v1, 0, S.HEIGHT) - np.clip(v0, 0, S.HEIGHT))
        presence = np.maximum(box_in, 0.0) / IMG_AREA
        c = ((u0 + u1) / 2 - S.WIDTH / 2) / (S.WIDTH / 2)
        ext = np.maximum(np.abs(u0 - S.WIDTH / 2), np.abs(u1 - S.WIDTH / 2)) / (S.WIDTH / 2)
        hbar = np.exp(-(np.maximum(0.0, ext - self.EXTENT) / self.EXT_SOFT) ** 2)
        vbar, _, _ = self.vertical_factor(zc, front)
        inframe = front & (u >= 0) & (u < S.WIDTH) & (v >= 0) & (v < S.HEIGHT)
        vis = self.unoccluded(cam, pts, vox, obj)
        n_in = inframe.sum(1)
        unocc = np.where(n_in > 0, (vis & inframe).sum(1) / np.maximum(n_in, 1), 0.0)
        J = presence * unocc * np.exp(-self.W_CENTER * c ** 2) * hbar * vbar * front.mean(1) ** 2
        J = np.where(n_in >= 3, J, 0.0)
        return np.log(J + 1e-4)


class MppiV4(MppiV3):
    """v4: MPPI around closer. The nominal sequence is closer's own drive (lattice line-following of the local-map
    path to the fit pose on the current bearing, backing out when inside it, then turning to face the object;
    "stay" once there). K sequences are sampled around it (each step keeps the nominal action with 1 - EPS,
    else F / L / R / stay uniformly) and scored with v2's view score (in-frame hull area x unoccluded x side
    completeness^2 x facing x the vertical-cut barrier). A sampled sequence replaces the nominal only when its
    value beats the nominal's by DELTA (10 % better view); the nominal's "stay" with no better sample ends the
    approach. On easy starts it is closer; it deviates for occluded, oblique or truncated views."""
    name = "mppi_v4"
    EPS, K, ITERS, H = 0.3, 160, 2, 10
    FREE_FRAC = 0.0

    def backing_out(self, obs, pts):
        """Back out to the fit pose (the current view cannot fit the object; no tilt, no reverse)."""
        return not self.view_ok(obs, pts)
    DELTA, C_TOGO = 0.1, 0.0
    STAY = 3

    def score(self, poses, pts, center, vox, obj):
        return MppiV2.score(self, poses, pts, center, vox, obj)

    def rollout(self, seqs, x0, yaw0, pf):
        moving = np.where(seqs == self.STAY, -1, seqs)          # stay: neither turn nor step
        return Mppi.rollout(self, moving, x0, yaw0, pf)

    def nominal(self, obs, here, yaw0, goal, center, H):
        """closer's next H actions on the lattice model (0 F, 1 L, 2 R, 3 stay)."""
        pf = obs.pathfinder
        cells = pf.path_cells(here, goal) if goal is not None else None
        path = [pf.lmap.center(c) for c in cells] if cells else [goal if goal is not None else center]
        seq, pos, k = [], here.astype(float).copy(), 0
        for _ in range(H):
            yaw = yaw0 + k * G.LATTICE
            if goal is None or np.hypot(*(goal - pos)) <= 0.25:
                diff = G.wrap(G.bearing_yaw(pos, center) - yaw)
                a = self.STAY if abs(math.degrees(diff)) <= 15.0 else (1 if diff > 0 else 2)
            else:
                ahead = next((p for p in path if np.hypot(*(p - pos)) >= 0.5), path[-1])
                diff = G.wrap(G.bearing_yaw(pos, ahead) - yaw)
                a = 0 if abs(math.degrees(diff)) <= 20.0 else (1 if diff > 0 else 2)
            seq.append(a)
            if a == 0:
                pos = pos + S.FORWARD_M * G.forward(yaw)
            elif a in (1, 2):
                k += 1 if a == 1 else -1
        return np.array(seq)

    def plan(self, obs):
        pf, vox = obs.pathfinder, self.vox
        pts, info = self.points(obs)
        center = G.ObjectModel(np.asarray(obs.cloud), obs.floor_y).center
        obj = self.object_cells(vox, np.asarray(obs.cloud))
        here = np.asarray(obs.position)[[0, 2]].astype(float)
        H = max(1, min(self.H, obs.remaining))
        self.cache = {}
        # closer's goal (stay once inside the fit range); back out only when this view is cut vertically
        from active_approach.selfcontained import Closer
        backout = self.backing_out(obs, pts)
        g = self.fit_goal(obs) if backout else np.asarray(Closer().line_goal(obs).position)[[0, 2]]
        nom = self.nominal(obs, here, obs.yaw, g if np.isfinite(g).all() else None, center, H).astype(int)
        if not backout:   # the nominal's first action is closer's own (policy + local controller)
            first = self.closer_action(obs)
            if first != nom[0]:
                if first == self.STAY:
                    nom = np.full(H, self.STAY)
                else:
                    nxt = self.rollout(np.array([[first]]), here, obs.yaw, pf)[0][0, 1]
                    rest = self.nominal(obs, nxt[:2], nxt[2], g if np.isfinite(g).all() else None, center, H - 1)
                    nom = np.r_[[first], rest][:H].astype(int)
        # the incumbent: the committed plan shifted by one step (stay-padded), else the nominal
        inc = nom if self.plan_seq is None else np.r_[self.plan_seq[1:], [self.STAY]][:H]
        if len(inc) < H:
            inc = np.r_[inc, np.full(H - len(inc), self.STAY)]
        best_seq, best_val, nom_val, inc_val, s0 = inc.copy(), None, None, None, None
        for it in range(self.ITERS):
            keep = self.rng.random((self.K, H)) >= self.EPS
            seqs = np.where(keep, best_seq[None], self.rng.integers(0, 4, (self.K, H)))
            n_free = int(self.FREE_FRAC * self.K)
            if n_free:   # free samples (v1's forward-biased prior, no stay): multi-step escapes the nominal never tries
                cum = self.PRIOR.cumsum()
                seqs[-n_free:] = (self.rng.random((n_free, H, 1)) > cum[None, None, :]).sum(-1).clip(0, 2)
            seqs[0], seqs[1], seqs[2] = nom, inc, best_seq
            stopped = np.maximum.accumulate(seqs == self.STAY, axis=1)   # stay is absorbing: a plan that stops,
            seqs = np.where(stopped, self.STAY, seqs)                     # stops (no stay-then-move plans)
            values, tstar, sc = self.evaluate(seqs, here, obs.yaw, pf, pts, center, vox, obj)
            if nom_val is None:
                nom_val, inc_val, s0 = float(values[0]), float(values[1]), float(sc[0, 0])
            i = int(values.argmax())
            if best_val is None or values[i] > best_val:
                best_seq, best_val = seqs[i].copy(), float(values[i])
        # hysteresis: the nominal wins ties; a sample must beat the incumbent (and the nominal) by DELTA
        if inc_val >= nom_val + self.DELTA and best_val < inc_val + self.DELTA:
            chosen, cval = inc, inc_val
        elif best_val >= max(nom_val, inc_val) + self.DELTA:
            chosen, cval = best_seq, best_val
        else:
            chosen, cval = nom, nom_val
        return nom, nom_val, chosen, cval, s0, info

    def act(self, obs):
        import dataclasses
        cloud = np.asarray(obs.cloud)
        if len(cloud) and self.anchor is None:
            model = G.ObjectModel(cloud, obs.floor_y)
            self.anchor = (model.center, 0.5 * float(np.hypot(*model.extent)) + self.ANCHOR_PAD)
        if self.anchor is not None and len(cloud):
            keep = np.hypot(*(cloud[:, [0, 2]] - self.anchor[0]).T) <= self.anchor[1]
            if keep.sum() >= 10:
                obs = dataclasses.replace(obs, cloud=cloud[keep])
        pf = obs.pathfinder
        mask = np.asarray(obs.mask, bool)
        if self.vox is None:
            self.vox = VoxelMap(np.asarray(obs.position)[[0, 2]], obs.floor_y)
        self.vox.update(np.asarray(obs.depth), mask, obs.position, obs.yaw)
        if self.prev is not None and self.prev[2] == "move_forward":
            moved = float(np.hypot(*(np.asarray(obs.position) - self.prev[0])[[0, 2]]))
            if moved < 0.5 * S.FORWARD_M:
                pf.lmap.bump(self.prev[0], self.prev[1])
                pf.refresh(obs.position)
        if self.done or not len(obs.cloud) or not hasattr(pf, "lmap"):
            self.last = {"stop": "no_object" if not len(obs.cloud) else "done"}
            return None
        self.obs, self.goal_cell = obs, None
        pts, _ = self.points(obs)
        self.z_lo, self.z_hi = self.heights(obs, pts)
        here = np.asarray(obs.position)[[0, 2]]
        center = G.ObjectModel(np.asarray(obs.cloud), obs.floor_y).center
        off = math.degrees(abs(G.wrap(G.bearing_yaw(here, center) - obs.yaw)))
        if int(mask.sum()) >= self.USABLE_PX and off <= self.FACE_DEG and not mask[0].any():
            self.top_seen = True
        nom, nom_val, seq, val, s0, info = self.plan(obs)
        deviate = not np.array_equal(seq, nom)
        chosen = seq
        self.plan_seq = chosen if deviate else None
        self.last = {"J0": round(float(np.exp(s0)), 4), "V": round(val, 3), "Vnom": round(nom_val, 3),
                     "nom": "".join("FLRS"[a] for a in nom), "seq": "".join("FLRS"[a] for a in seq),
                     "deviate": bool(deviate), "z": [round(self.z_lo, 2), round(self.z_hi, 2)]}
        a = int(chosen[0])
        if a == self.STAY:
            if not deviate and int(mask.sum()) < self.USABLE_PX and obs.remaining > 1 and not self.retried:
                self.retried = True   # arrived without a usable view: one more planning round from the object side
            self.done = True
            self.last["stop"] = "nominal_done" if not deviate else "best_view"
            return None
        action = ACTIONS[a]
        self.prev = (np.asarray(obs.position).copy(), obs.yaw, action)
        self.last["action"] = action
        return action

    def reset(self, ctx):
        super().reset(ctx)
        from active_approach.selfcontained import Closer
        self.retried = False
        self.plan_seq = None
        self.closer = Closer()
        self.closer.reset(ctx)
        self.sim, self.ctrl = ctx.get("sim"), None
        self.c_arrived, self.c_idle, self.c_skips, self.c_done = False, 0, 0, False

    def closer_action(self, obs):
        """closer's next action as the harness would drive it (Closer.next_viewpoint + local_nav.LocalController),
        as an index (0 F, 1 L, 2 R; STAY when closer is done)."""
        import dataclasses
        if self.c_done:
            return self.STAY
        if self.ctrl is None:
            self.ctrl = L.LocalController(self.sim, obs.pathfinder.lmap, obs.pathfinder)
        if self.prev is not None:   # the action actually executed (MPPI may have deviated): bump detection
            self.ctrl.last = (self.prev[0], self.prev[1], self.prev[2])
        o = dataclasses.replace(obs, arrived=self.c_arrived)
        while True:
            vp = self.closer.next_viewpoint(o)
            if vp is None:
                self.c_done = True
                return self.STAY
            try:
                act = self.ctrl.action(vp, o)
            except Exception:
                self.c_skips += 1
                if self.c_skips > 3:
                    self.c_done = True
                    return self.STAY
                o = dataclasses.replace(o, arrived=True)
                continue
            if act is None:
                self.c_idle += 1
                if self.c_idle > 4:
                    self.c_done = True
                    return self.STAY
                o = dataclasses.replace(o, arrived=True)
                continue
            self.c_arrived, self.c_idle = False, 0
            return ACTIONS.index(act)


class MppiV5(MppiV4):
    """v5: v4 with a collision-aware nominal and rarer deviations. v4's nominal followed the path's look-ahead
    point and cut corners through inflated cells, so its rollout bumped, scored near the floor and nearly every
    sample "beat" it (v4 deviated almost every step). Here the nominal descends the local-map cost-to-go to the
    goal on the lattice (each step: the heading k x 30 deg minimising |k| turns + forward steps until blocked +
    the cost-to-go there, as local_nav.LocalController._lattice), and a sample must beat it by DELTA = 0.3."""
    name = "mppi_v5"
    DELTA = 0.3

    def nominal(self, obs, here, yaw0, goal, center, H):
        pf = obs.pathfinder
        lm = pf.lmap
        if goal is None:
            return super().nominal(obs, here, yaw0, goal, center, H)
        dist, _ = pf._tree(lm.cell(goal))
        togo = lambda xz: dist[lm.cell(xz)[0] * lm.n + lm.cell(xz)[1]] * L.CELL / S.FORWARD_M

        def free(p, h):
            f = G.forward(h)
            return all(np.isfinite(pf.cost[lm.cell(p + frac * S.FORWARD_M * f)]) for frac in (0.5, 1.0))

        seq, pos, k = [], here.astype(float).copy(), 0
        for _ in range(H):
            yaw = yaw0 + k * G.LATTICE
            if np.hypot(*(goal - pos)) <= 0.25 or not np.isfinite(togo(pos)):
                diff = G.wrap(G.bearing_yaw(pos, center) - yaw)
                a = self.STAY if abs(math.degrees(diff)) <= 15.0 else (1 if diff > 0 else 2)
            else:
                best = (math.inf, 0)
                for dk in range(-5, 7):
                    h, p, c = yaw + dk * G.LATTICE, pos, math.inf
                    for j in range(1, 5):
                        if not free(p, h):
                            break
                        p = p + S.FORWARD_M * G.forward(h)
                        c = min(c, abs(dk) + j + togo(p))
                    if c < best[0] - 1e-6:
                        best = (c, dk)
                dk = best[1]
                a = 0 if dk == 0 and np.isfinite(best[0]) else (1 if dk > 0 else 2)
                if not np.isfinite(best[0]):   # boxed in: face the object
                    diff = G.wrap(G.bearing_yaw(pos, center) - yaw)
                    a = self.STAY if abs(math.degrees(diff)) <= 15.0 else (1 if diff > 0 else 2)
            seq.append(a)
            if a == 0:
                pos = pos + S.FORWARD_M * G.forward(yaw)
            elif a in (1, 2):
                k += 1 if a == 1 else -1
        return np.array(seq)


class MppiV6(MppiV5):
    """v6: v5 with half the samples drawn freely (v1's forward-biased prior; v5's samples around a "stay" nominal
    almost all stopped within two steps, so it never found the sidestep around an occluder v1 found), and the
    back-out decision kept unless the object is in view (>= 200 mask px): with the object out of view the fit
    check flipped and v5 alternated turn left / right between closer's action and the back-out nominal."""
    name = "mppi_v6"
    FREE_FRAC = 0.5

    def reset(self, ctx):
        super().reset(ctx)
        self.backout = False

    def backing_out(self, obs, pts):
        if int(np.asarray(obs.mask, bool).sum()) >= 200:
            self.backout = not self.view_ok(obs, pts)
        return self.backout


class MppiV7(MppiV6):
    """v7: v6 with a frame-aware value. In the decision flow every usable frame on the way (facing within 20 deg,
    the object visible) is evidence and the flow commits as soon as JEV names it consistently, so a weak frame
    made while manoeuvring can decide (epi696: a turn made a 460 px view; JEV said chair twice; the fused SAM3 over
    it rejected). Each rollout pose t >= 1 predicted usable whose score is more than 3x worse than the rollout's best
    costs W_POOR x (log-gap - log 3)."""
    name = "mppi_v7"
    W_POOR, POOR_GAP = 0.1, math.log(3.0)

    def _values(self, poses, bumps, yaw0, pts, center, hmap, obj):
        values, tstar, sc = super()._values(poses, bumps, yaw0, pts, center, hmap, obj)
        d = center[None, None] - poses[..., :2]
        bearing = np.arctan2(-d[..., 0], -d[..., 1])
        off = np.abs((bearing - poses[..., 2] + np.pi) % (2 * np.pi) - np.pi)
        usable = (off <= math.radians(self.FACE_DEG)) & (sc > math.log(1e-3))
        gap = sc.max(1, keepdims=True) - sc - self.POOR_GAP
        poor = (usable[:, 1:] * np.maximum(0.0, gap[:, 1:])).sum(1)
        return values - self.W_POOR * poor, tstar, sc


class MppiV8(MppiV7):
    """v8: closer first, MPPI after. Until closer itself would stop, every action is closer's (Closer +
    local_nav.LocalController on the raw observation), so every frame and every decision the flow commits on the way
    is closer's. Only then, when the flow has not committed (closer's stop view would be a forced call; 7 of its 9
    Stage B errors were forced), v7's MPPI (free samples, frame-aware value, back-out) looks for a clearly better
    view with the remaining budget. Early MPPI deviations, made from far with a poor object model, were gambles."""
    name = "mppi_v8"
    PHASE2_DELTA = 0.1

    def reset(self, ctx):
        super().reset(ctx)
        self.phase2 = None

    def act(self, obs):
        if self.c_done:
            if self.phase2 is None:   # v1's MPPI (free samples, stop when nothing beats staying by 10 %), sharing
                self.phase2 = Mppi()  # the voxel map built during closer's phase
                self.phase2.reset({})
                self.phase2.DELTA = self.PHASE2_DELTA
                self.phase2.vox = self.vox
                self.phase2.prev = self.prev
            action = self.phase2.act(obs)
            self.last = {"phase": "mppi", **(self.phase2.last or {})}
            return action
        if self.vox is None:
            self.vox = VoxelMap(np.asarray(obs.position)[[0, 2]], obs.floor_y)
        self.vox.update(np.asarray(obs.depth), np.asarray(obs.mask, bool), obs.position, obs.yaw)
        a = self.closer_action(obs)
        if a == self.STAY:   # closer is done: MPPI from here
            return self.act(obs)
        action = ACTIONS[a]
        self.prev = (np.asarray(obs.position).copy(), obs.yaw, action)
        self.last = {"phase": "closer", "action": action}
        return action


class MppiV9(MppiV5):
    """v9 (2026-11-08): one MPPI cost blending a reference trajectory with the view score. The reference is closer's
    drive: its first action is the real closer's (Closer + LocalController), the rest v5's collision-aware lattice
    descent to closer's goal (stay once there). Per sampled sequence
        V = (1 - w) V_view - w W_REF track - C_BUMP bumps,
        track = mean_t (|p_t - ref_t|^2 / SIG_P^2 + wrap(yaw_t - ref_yaw_t)^2 / SIG_Y^2),
    with V_view v1's max_t (log J_t - C_ACT t) and w = clip(distance to closer's goal / D_BLEND, 0, 1) (0 once closer
    is done). Far from the object the reference dominates (closer's frames); near it the view score does. Samples:
    half around the reference (keep its action with 1 - EPS), half free (v1's prior); softmax update (LAMBDA), the
    executed action is the best sequence's first. Stops (v1's rule, DELTA) only when w < W_STOP and no sequence's
    view beats the current one."""
    name = "mppi_v9"
    H, K, ITERS, EPS, FREE_FRAC = 8, 160, 3, 0.3, 0.5
    W_REF, SIG_P, SIG_Y = 1.0, 0.25, math.radians(30.0)
    D_BLEND, W_STOP, DELTA = 1.5, 0.2, 0.1

    def reset(self, ctx):
        super().reset(ctx)
        self.w = 1.0

    def score(self, poses, pts, center, vox, obj):
        return Mppi.score(self, poses, pts, center, vox, obj)

    def reference(self, obs, here, center, H):
        first = self.closer_action(obs)
        goal = self.closer.goal
        g = np.asarray(goal.position)[[0, 2]] if goal is not None else None
        if first == self.STAY:
            return np.full(H, self.STAY), None
        nxt = self.rollout(np.array([[first]]), here, obs.yaw, obs.pathfinder)[0][0, 1]
        rest = self.nominal(obs, nxt[:2], nxt[2], g, center, H - 1) if H > 1 else np.zeros(0, int)
        return np.r_[[first], rest][:H].astype(int), g

    def blended(self, seqs, here, yaw0, pf, pts, center, obj, ref_poses, w):
        poses, bumps = self.rollout(seqs, here, yaw0, pf)
        v_view, tstar, sc = Mppi._values_v1(self, poses, yaw0, pts, center, obj)
        if w > 0:
            dp = ((poses[:, 1:, :2] - ref_poses[None, 1:, :2]) ** 2).sum(-1) / self.SIG_P ** 2
            dy = ((poses[:, 1:, 2] - ref_poses[None, 1:, 2] + np.pi) % (2 * np.pi) - np.pi) ** 2 / self.SIG_Y ** 2
            track = (dp + dy).mean(1)
        else:
            track = np.zeros(len(seqs))
        return (1 - w) * v_view - w * self.W_REF * track - self.C_BUMP * bumps, v_view, tstar, sc

    def act(self, obs):
        pf = obs.pathfinder
        mask = np.asarray(obs.mask, bool)
        if self.vox is None:
            self.vox = VoxelMap(np.asarray(obs.position)[[0, 2]], obs.floor_y)
        self.vox.update(np.asarray(obs.depth), mask, obs.position, obs.yaw)
        if self.prev is not None and self.prev[2] == "move_forward":
            moved = float(np.hypot(*(np.asarray(obs.position) - self.prev[0])[[0, 2]]))
            if moved < 0.5 * S.FORWARD_M:
                pf.lmap.bump(self.prev[0], self.prev[1])
                pf.refresh(obs.position)
        if self.done or not len(obs.cloud) or not hasattr(pf, "lmap"):
            self.last = {"stop": "no_object" if not len(obs.cloud) else "done"}
            return None
        here = np.asarray(obs.position)[[0, 2]].astype(float)
        center = G.ObjectModel(np.asarray(obs.cloud), obs.floor_y).center
        H = max(1, min(self.H, obs.remaining))
        ref, g = self.reference(obs, here, center, H)
        w = 0.0 if (self.c_done or g is None) else float(np.clip(np.hypot(*(g - here)) / self.D_BLEND, 0, 1))
        w = max(w, getattr(self, "_floor", 0.0))
        self.w = w
        ref_poses = self.rollout(ref[None], here, obs.yaw, pf)[0][0]
        pts, info = self.points(obs)
        obj = self.object_cells(self.vox, np.asarray(obs.cloud))
        self.cache = {}
        best_seq, best_val, best_view, best_t, s0 = None, -np.inf, None, 0, None
        pi = np.tile(self.PRIOR, (H, 1))
        n_free = int(self.FREE_FRAC * self.K)
        for it in range(self.ITERS):
            centre_seq = ref if best_seq is None else best_seq
            keep = self.rng.random((self.K, H)) >= self.EPS
            seqs = np.where(keep, centre_seq[None], self.rng.integers(0, 3, (self.K, H)))
            if it == 0:
                cum = self.PRIOR.cumsum()
                seqs[-n_free:] = (self.rng.random((n_free, H, 1)) > cum[None, None, :]).sum(-1).clip(0, 2)
            else:   # later iterations: free half from the updated distribution
                cum = pi.cumsum(1)
                seqs[-n_free:] = (self.rng.random((n_free, H, 1)) > cum[None]).sum(-1).clip(0, 2)
            seqs[0] = ref
            if best_seq is not None:
                seqs[1] = best_seq
            stopped = np.maximum.accumulate(seqs == self.STAY, axis=1)
            seqs = np.where(stopped, self.STAY, seqs)
            values, v_view, tstar, sc = self.blended(seqs, here, obs.yaw, pf, pts, center, obj, ref_poses, w)
            s0 = float(sc[0, 0])
            i = int(values.argmax())
            if values[i] > best_val:
                best_seq, best_val, best_view, best_t = seqs[i].copy(), float(values[i]), float(v_view[i]), int(tstar[i])
            wt = np.exp((values - values.max()) / self.LAMBDA)
            wt /= wt.sum()
            pi = np.einsum("k,kha->ha", wt, np.eye(4)[seqs])[:, :3] + 1e-6
            pi /= pi.sum(1, keepdims=True)
        off = math.degrees(abs(G.wrap(G.bearing_yaw(here, center) - obs.yaw)))
        self.last = {"w": round(w, 2), "J0": round(float(np.exp(s0)), 4), "V": round(best_val, 3),
                     "Vview": round(best_view, 3), "t*": best_t, "ref": "".join("FLRS"[a] for a in ref),
                     "seq": "".join("FLRS"[a] for a in best_seq)}
        if w < self.W_STOP and (best_t == 0 or best_view - s0 < self.DELTA or best_seq[0] == self.STAY):
            if off > 20.0:
                diff = G.wrap(G.bearing_yaw(here, center) - obs.yaw)
                action = "turn_left" if diff > 0 else "turn_right"
            else:
                self.done = True
                self.last["stop"] = "best_view"
                return None
        elif best_seq[0] == self.STAY:   # still far but the reference/best says stay: follow the view (turn to face)
            diff = G.wrap(G.bearing_yaw(here, center) - obs.yaw)
            if off <= 20.0:
                self.done = True
                self.last["stop"] = "ref_done"
                return None
            action = "turn_left" if diff > 0 else "turn_right"
        else:
            action = ACTIONS[int(best_seq[0])]
        self.prev = (np.asarray(obs.position).copy(), obs.yaw, action)
        self.last["action"] = action
        return action


class MppiV10(MppiV9):
    """v10: v9 with w >= W_MIN until closer itself is done. In v9 w fell to 0 as soon as the agent was within reach
    of closer's goal, so from near starts MPPI ignored closer's final facing turns and gambled on the view score of a
    truncated first view (epi574 bed, 284 chair: v9 wrong, closer and v8 right after one turn)."""
    name = "mppi_v10"
    W_MIN = 0.5

    def reference(self, obs, here, center, H):
        ref, g = super().reference(obs, here, center, H)
        self._floor = 0.0 if (self.c_done or g is None) else self.W_MIN
        return ref, g


class MppiV11(MppiV10):
    """v11: v10 plus a cut-aware re-plan when the frame cuts the object's top. At pitch 0 the frame top at near-face
    distance d is H_CAM + d tan(BETA_V); an object whose top is above it (TV on a tall dresser, plant on a cabinet)
    cannot be seen whole from closer, and v10's view MPPI (no vertical term, horizon 8) stops on the glimpse
    (ziup 179). Trigger (once): facing the object (<= 30 deg), >= 200 mask px touching the top image row, and the
    near-face distance at least BACK_MIN short of the distance where the estimated top fits (top estimate:
    max(highest cloud point + TOP_PAD, bottom + category height prior)). Then v2's MPPI takes over (vertical-cut
    barrier in the view score, horizon 14, best sequence executed, no pull toward closer's goal) until it stops on
    a usable view or BACK_STEPS run out; mppi_v4, the same score, was the only planner to get ziup 179 right."""
    name = "mppi_v11"
    TOP_PAD, BACK_MIN, BACK_STEPS = 0.15, 0.2, 25

    def reset(self, ctx):
        super().reset(ctx)
        self.cutplan, self.cut_used, self.cut_n, self.cut_info = None, False, 0, {}

    def top_cut(self, obs):
        """True when the frame cuts the object's top and backing off by >= BACK_MIN would fit it."""
        from active_approach.round3 import prior_hw
        mask = np.asarray(obs.mask, bool)
        cloud = np.asarray(obs.cloud)
        if mask.sum() < 200 or mask[0].sum() < 5 or len(cloud) < 10:
            return False
        here = np.asarray(obs.position)[[0, 2]].astype(float)
        center = G.ObjectModel(cloud, obs.floor_y).center
        if math.degrees(abs(G.wrap(G.bearing_yaw(here, center) - obs.yaw))) > 30.0:
            return False
        h = cloud[:, 1] - obs.floor_y
        z_lo = float(np.percentile(h, 5))
        z_hi = max(float(h.max()) + self.TOP_PAD, z_lo + prior_hw(obs.meta or {})[0])
        d_face = float(np.min(np.hypot(*(cloud[:, [0, 2]] - here).T)))
        d_need = (z_hi - G.H_CAM) / math.tan(G.BETA_V)
        self.cut_info = {"z_hi": round(z_hi, 2), "d_face": round(d_face, 2), "d_need": round(d_need, 2)}
        return d_need - d_face >= self.BACK_MIN

    def act(self, obs):
        if self.cutplan is None and not self.cut_used and not self.done and len(obs.cloud) \
                and hasattr(obs.pathfinder, "lmap") and self.top_cut(obs):
            self.cut_used = True
            if self.vox is None:
                self.vox = VoxelMap(np.asarray(obs.position)[[0, 2]], obs.floor_y)
            plan = MppiV2()
            plan.reset({})
            plan.H, plan.K, plan.C_TOGO, plan.BEST_SEQ, plan.DELTA = 14, 192, 0.0, True, 0.1
            plan.pi = np.tile(plan.PRIOR, (plan.H, 1))
            plan.vox, plan.prev = self.vox, self.prev
            self.cutplan = plan
        if self.cutplan is None:
            return super().act(obs)
        self.cut_n += 1
        action = self.cutplan.act(obs) if self.cut_n <= self.BACK_STEPS else None
        cloud = np.asarray(obs.cloud)
        if action in ("turn_left", "turn_right") and len(cloud):
            # while still too close, never turn into a facing view (the flow would commit on the cut glimpse)
            here = np.asarray(obs.position)[[0, 2]].astype(float)
            center = G.ObjectModel(cloud, obs.floor_y).center
            d_face = float(np.min(np.hypot(*(cloud[:, [0, 2]] - here).T)))
            new_yaw = obs.yaw + (G.LATTICE if action == "turn_left" else -G.LATTICE)
            off = math.degrees(abs(G.wrap(G.bearing_yaw(here, center) - new_yaw)))
            if off <= 20.0 and d_face < self.cut_info.get("d_need", 0.0) - self.BACK_MIN:
                action = "turn_right" if action == "turn_left" else "turn_left"
                self.cutplan.prev = (np.asarray(obs.position).copy(), obs.yaw, action)
                self.cutplan.done = False
        self.prev = self.cutplan.prev
        self.last = {"phase": "cut_replan", **self.cut_info, **(self.cutplan.last or {})}
        if action is None:
            self.done = True
        return action


def _tukey(x, a):
    """1 on [a, 1 - a], cosine taper to 0 at 0 and 1, 0 outside (torch)."""
    import torch
    lo = 0.5 * (1 - torch.cos(math.pi * torch.clamp(x, 0, a) / a))
    hi = 0.5 * (1 - torch.cos(math.pi * torch.clamp(1 - x, 0, a) / a))
    return torch.where((x < 0) | (x > 1), torch.zeros_like(x), torch.minimum(lo, hi))


class MppiV15(MppiV10):
    """The view planner (2026-11-09; v12 -> v14 (user's design) -> v15, Stage B 99/104 = v11 with fewer actions,
    subset B 103/120 best). One MPPI, one value, no phases and no triggers.
    - Object model = mass: the fused cloud voxel-thinned (one point per 5 cm voxel, about equal surface each), plus
      mass-matched phantoms over extents never seen: top (until a frame shows the top inside the image) up to
      max(seen top + TOP_PAD, bottom + category height prior), bottom down to max(floor, top - prior), and the sides
      cut in the latest frame (missing width from the width prior). A phantom band gets (missing extent / seen
      extent) of the points.
    - View score, per pose (torch-batched): every model point p with area a_p projects to (u, v) with window weight
      w = tukey(u / W) tukey(v / H) (1 in the central 1 - 2 WIN of each axis) and visibility o (ray through the own
      3D voxel map); C = sum a o w / sum a (fraction of the object's surface well inside the frame), size = sum of
      the in-window projected areas a f^2 / z^2 / image area, J = sat(size) x C^KAPPA x facing: the optimum is the
      largest view that keeps the whole object in the window.
    - Value of a plan: the flow may decide at any usable facing frame on the way, each with P_COMMIT (measured in
      Stage B: it committed at 33 % of the first admitted frames after the start, 22 % of all), so stopping at t is
      worth the expected log J of the frame the decision is made on, A_t + S_t log J_t (S_t = P(no decision before t)),
      minus C_ACT t and C_BUMP bumps, minus w W_REF track(closer's reference): closer's planned drive (its next action
      from Closer + LocalController, then the lattice descent to its goal) matters far out, w = clip(d_goal / D_BLEND)
      and non-increasing over the episode (a deliberate back-off for the view never hands control back to closer).
    - Biased-MPPI sampler (Trevisan & Alonso-Mora 2024), horizon H. Seeds: path-followed lattice paths (both ways round)
      to ring proposals that can be reached and faced from within H (bearings every 15 deg, RADII beyond the near face;
      the 3D visibility in the score decides occlusion, a 2D line of sight rejected raised objects over furniture),
      the best 2 N_PROP by rollout value; closer's drive and goal pose; the shifted last plan (kept unless beaten by
      DELTA); stay; face-it. Samples perturb the seeds and the softmax-updated categorical.
    - The best sequence is executed; it stops when staying is within DELTA of the best plan (staying deviates from
      the reference while far, so no early stop out there)."""
    name = "mppi_v15"
    H, K, ITERS, EPS = 16, 256, 3, 0.15
    KAPPA, WIN, TOP_PAD, SIDE_PAD, EDGE, SEEN_PAD = 2.0, 0.1, 0.2, 0.2, 4, 0.15
    LOOK, S0 = 0.4, 0.05
    N_POINTS, N_PROP, RADII = 400, 6, (0.5, 0.8, 1.1, 1.5, 2.0, 2.6)
    P_COMMIT, DELTA, C_ACT = 0.3, 0.1, 0.02

    def reset(self, ctx):
        super().reset(ctx)
        self.top_seen, self.bot_seen, self.top_cut, self.bot_cut = False, False, False, False
        self.side_ph, self.plan_seq, self._w_cap = None, None, 1.0
        import os
        import torch
        g = os.environ.get("MPPI_GPU")
        self.dev = torch.device(f"cuda:{g}" if g is not None and torch.cuda.is_available() else "cpu")

    def cut_flags(self, obs):
        """Sticky vertical-extent flags from facing frames (>= 400 mask px, <= 35 deg off): an extent is cut when the
        mask touches that image border (EDGE px band), and seen once a frame's border at the object's depth clears the
        cloud's extreme by SEEN_PAD (the tracked mask grows into new object pixels, so a taller object would have
        shown up there)."""
        mask = np.asarray(obs.mask, bool)
        cloud = np.asarray(obs.cloud)
        if mask.sum() < 400 or len(cloud) < 10:
            return
        here = np.asarray(obs.position)[[0, 2]]
        center = G.ObjectModel(cloud, obs.floor_y).center
        if math.degrees(abs(G.wrap(G.bearing_yaw(here, center) - obs.yaw))) > 35.0:
            return
        h = cloud[:, 1] - obs.floor_y
        depth = max(float(np.median((cloud[:, [0, 2]] - here) @ G.forward(obs.yaw))), 0.3)
        span = depth * math.tan(G.BETA_V)
        if mask[: self.EDGE].any():
            self.top_cut = True
        if G.H_CAM + span > float(np.percentile(h, 98)) + self.SEEN_PAD:
            self.top_seen = True
        if mask[-self.EDGE:].any():
            self.bot_cut = True
        if G.H_CAM - span < float(np.percentile(h, 2)) - self.SEEN_PAD or float(np.percentile(h, 2)) < 0.1:
            self.bot_seen = True

    def act(self, obs):
        self._w, self._ref_poses = 0.0, None
        self.cut_flags(obs)
        self._ref, self._extra_goals = None, []
        cloud = np.asarray(obs.cloud)
        if len(cloud) and hasattr(obs.pathfinder, "lmap") and not self.done:
            here = np.asarray(obs.position)[[0, 2]].astype(float)
            center = G.ObjectModel(cloud, obs.floor_y).center
            H = max(1, min(self.H, obs.remaining))
            try:
                ref, g = MppiV10.reference(self, obs, here, center, H)
            except Exception:
                ref, g = None, None
            if ref is not None and not (ref == self.STAY).all() and g is not None and np.isfinite(g).all():
                self._ref, self._extra_goals = ref, [g]
                self._w = float(np.clip(np.hypot(*(g - here)) / self.D_BLEND, 0.0, 1.0))
                self._ref_poses = self.rollout(ref[None], here, obs.yaw, obs.pathfinder)[0][0]
        self._w = self._w_cap = min(self._w, self._w_cap)
        return self.view_act(obs)

    # ---------------- object model (mass) ----------------
    def model_points(self, obs):
        from active_approach.round3 import prior_hw
        cloud = np.asarray(obs.cloud, float)
        mask = np.asarray(obs.mask, bool)
        _, idx = np.unique(np.floor(cloud / 0.05).astype(np.int64), axis=0, return_index=True)
        pts = cloud[idx]
        n_vox = len(pts)
        if n_vox > self.N_POINTS:
            pts = pts[self.rng.choice(n_vox, self.N_POINTS, replace=False)]
        area = 0.0025 * n_vox / len(pts)
        prior_h, prior_w = prior_hw(obs.meta or {})
        h = pts[:, 1] - obs.floor_y
        z_lo, z_hi = float(h.min()), float(h.max())
        hv = max(z_hi - z_lo, 0.1)
        here = np.asarray(obs.position)[[0, 2]]
        extra, info = [], {}
        n = len(pts)
        if self.top_cut and not self.top_seen:
            top = max(z_hi + self.TOP_PAD, z_lo + prior_h)
            m = int(min(2 * n, n * (top - z_hi) / hv))
            rim = pts[h >= z_hi - 0.15 * hv]
            if m and len(rim):
                foot = rim[self.rng.integers(0, len(rim), m)]
                extra.append(np.stack([foot[:, 0], obs.floor_y + self.rng.uniform(z_hi, top, m), foot[:, 2]], 1))
                info["top"] = round(top, 2)
        if self.bot_cut and not self.bot_seen and z_lo > 0.1:
            bot = max(0.02, min(z_lo - self.TOP_PAD, z_hi - prior_h))
            m = int(min(2 * n, n * (z_lo - bot) / hv))
            rim = pts[h <= z_lo + 0.15 * hv]
            if m > 0 and len(rim):
                foot = rim[self.rng.integers(0, len(rim), m)]
                extra.append(np.stack([foot[:, 0], obs.floor_y + self.rng.uniform(bot, z_lo, m), foot[:, 2]], 1))
                info["bot"] = round(bot, 2)
        if mask.sum() >= 200:   # side cuts of this frame (kept while the object is out of view)
            right = np.array([math.cos(obs.yaw), -math.sin(obs.yaw)])
            proj = (pts[:, [0, 2]] - here) @ right
            wv = max(float(proj.max() - proj.min()), 0.1)
            miss = max(self.SIDE_PAD, prior_w - wv)
            side = []
            for sign, edge, cols, side_name in ((-1, proj.min(), slice(0, self.EDGE), "left"),
                                                (1, proj.max(), slice(-self.EDGE, None), "right")):
                if mask[:, cols].any():
                    rim = pts[np.abs(proj - edge) < 0.08]
                    m = int(min(n, n * miss / wv))
                    if m and len(rim):
                        q = rim[self.rng.integers(0, len(rim), m)].copy()
                        t = sign * self.rng.uniform(0, miss, m)
                        q[:, 0] += t * right[0]
                        q[:, 2] += t * right[1]
                        side.append(q)
            self.side_ph = np.concatenate(side) if side else None
        if self.side_ph is not None:
            extra.append(self.side_ph)
            info["side"] = len(self.side_ph)
        allp = np.concatenate([pts] + extra) if extra else pts
        return allp, np.full(len(allp), area), info

    # ---------------- batched view score ----------------
    def _prepare(self, obs, pts, wts, center, obj):
        import torch
        occ = self.vox.occupied() & ~obj[..., None]
        T = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float32, device=self.dev)
        self.t_occ = torch.as_tensor(occ, device=self.dev).reshape(-1)
        self.t_pts, self.t_w, self.t_center = T(pts), T(wts), T(center)
        self.vcache, self.cache = {}, {}

    def _vis(self, cams):
        """(Q, P) float: the ray camera -> point crosses no occupied voxel."""
        import torch
        vox, P = self.vox, self.t_pts
        out = []
        h_pt = P[:, 1] - vox.floor_y
        org = torch.as_tensor(vox.origin, dtype=torch.float32, device=self.dev)
        for a in range(0, len(cams), 64):
            c = cams[a:a + 64]
            d = P[None, :, [0, 2]] - c[:, None]
            L = d.norm(dim=-1)
            s = torch.arange(0.25, max(float(L.max()) - 0.15, 0.3), 0.05, device=self.dev)
            frac = s[None, None] / L.clamp(min=1e-3)[..., None]
            valid = s[None, None] < (L[..., None] - 0.15)
            xy = c[:, None, None] + frac[..., None] * d[:, :, None]
            rh = G.H_CAM + frac * (h_pt[None, :, None] - G.H_CAM)
            ij = torch.floor((xy - org) / vox.CELL).long()
            k = torch.floor(rh / vox.CELL).long()
            ok = valid & (ij >= 0).all(-1) & (ij < vox.n).all(-1) & (k >= 0) & (k < vox.NZ)
            flat = (ij[..., 0].clamp(0, vox.n - 1) * vox.n + ij[..., 1].clamp(0, vox.n - 1)) * vox.NZ + k.clamp(0, vox.NZ - 1)
            blocked = (self.t_occ[flat] & ok).any(-1)
            out.append((~blocked).float())
        return torch.cat(out)

    def score_poses(self, poses):
        """log J for poses (M, 3) (x, z, yaw), numpy in and out."""
        import torch
        if not len(poses):
            return np.zeros(0)
        pk = [(round(x / 0.02), round(z / 0.02)) for x, z in poses[:, :2]]
        new = sorted({k for k in pk if k not in self.vcache})
        if new:
            cams = torch.as_tensor(np.array(new, float) * 0.02, dtype=torch.float32, device=self.dev)
            vis = self._vis(cams)
            for k, v in zip(new, vis):
                self.vcache[k] = v
        vis = torch.stack([self.vcache[k] for k in pk])                        # (M, P)
        ps = torch.as_tensor(poses, dtype=torch.float32, device=self.dev)
        cam, yaw = ps[:, :2], ps[:, 2]
        f = torch.stack([-torch.sin(yaw), -torch.cos(yaw)], 1)
        r = torch.stack([torch.cos(yaw), -torch.sin(yaw)], 1)
        d = self.t_pts[None, :, [0, 2]] - cam[:, None]
        zc = (d * f[:, None]).sum(-1)
        xc = (d * r[:, None]).sum(-1)
        yc = (self.t_pts[:, 1] - (self.vox.floor_y + G.H_CAM))[None]
        front = zc > 0.1
        z = zc.clamp(min=0.1)
        u = (S.WIDTH / 2 + S.FOCAL * xc / z) / S.WIDTH
        v = (S.HEIGHT / 2 - S.FOCAL * yc / z) / S.HEIGHT
        win = _tukey(u, self.WIN) * _tukey(v, self.WIN) * front.float() * vis
        a = self.t_w[None]
        C = (a * win).sum(1) / a.sum()
        size = ((a * win * S.FOCAL ** 2 / z ** 2).sum(1) / IMG_AREA).clamp(max=1.0)
        dc = self.t_center[None] - cam
        bearing = torch.atan2(-dc[:, 0], -dc[:, 1])
        off = torch.rad2deg(torch.abs(torch.remainder(bearing - yaw + math.pi, 2 * math.pi) - math.pi))
        facing = torch.exp(-((off - 15.0).clamp(min=0) / 10.0) ** 2)
        J = size / (size + self.S0) * C ** self.KAPPA * facing   # pixels beyond ~S0 of the image add little
        self._last_C = C
        return torch.log(J + 1e-4).cpu().numpy().astype(float)

    def values(self, seqs, here, yaw0, pf, center):
        poses, bumps = self.rollout(seqs, here, yaw0, pf)
        K, T, _ = poses.shape
        key = np.stack([np.round(poses[..., 0] / 0.02), np.round(poses[..., 1] / 0.02),
                        np.round((poses[..., 2] - yaw0) / G.LATTICE) % 12], -1).reshape(-1, 3).astype(np.int64)
        uniq, first, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
        todo = [i for i in range(len(uniq)) if tuple(uniq[i]) not in self.cache]
        if todo:
            vals = self.score_poses(poses.reshape(-1, 3)[first[todo]])
            for i, val in zip(todo, vals):
                self.cache[tuple(uniq[i])] = val
        sc = np.array([self.cache[tuple(q)] for q in uniq])[np.asarray(inv).ravel()].reshape(K, T)
        d = center[None, None] - poses[..., :2]
        bearing = np.arctan2(-d[..., 0], -d[..., 1])
        off = np.abs((bearing - poses[..., 2] + np.pi) % (2 * np.pi) - np.pi)
        usable = (off <= math.radians(self.FACE_DEG)) & (sc > math.log(1.5e-3))
        u = usable.astype(float) * self.P_COMMIT
        u[:, 0] = 0.0                                                       # the current frame is already seen
        S = np.cumprod(np.c_[np.ones((K, 1)), 1.0 - u[:, :-1]], axis=1)       # P(no decision before t)
        A = np.c_[np.zeros((K, 1)), np.cumsum((u * S * sc)[:, :-1], axis=1)]
        gain = A + S * sc - self.C_ACT * np.arange(T)[None]
        tstar = gain.argmax(1)
        value = gain.max(1) - self.C_BUMP * bumps
        if self._w > 0 and self._ref_poses is not None:   # deviation from closer's drive, weighted far out
            rp = self._ref_poses
            n = min(T, len(rp))
            dp = ((poses[:, 1:n, :2] - rp[None, 1:n, :2]) ** 2).sum(-1) / self.SIG_P ** 2
            dy = ((poses[:, 1:n, 2] - rp[None, 1:n, 2] + np.pi) % (2 * np.pi) - np.pi) ** 2 / self.SIG_Y ** 2
            value = value - self._w * self.W_REF * (dp + dy).mean(1)
        return value, tstar, sc, poses

    # ---------------- proposals (Biased-MPPI seeds) ----------------
    def proposals(self, obs, here, center, pts):
        pf, lm = obs.pathfinder, obs.pathfinder.lmap
        r_face = float(np.percentile(np.hypot(*(pts[:, [0, 2]] - center).T), 90))
        b = np.radians(np.arange(0, 360, 15))
        cand = np.array([center + (r_face + rr) * np.array([math.sin(t), math.cos(t)]) for rr in self.RADII for t in b])
        ij = np.floor((cand - lm.origin) / L.CELL).astype(int)
        ok = (ij >= 0).all(1) & (ij < lm.n).all(1)
        cand, ij = cand[ok], ij[ok]
        ok = np.isfinite(pf.cost[ij[:, 0], ij[:, 1]])
        cand = cand[ok]
        if not len(cand):
            return []
        yaw0 = obs.yaw
        k = np.round(np.array([G.wrap(G.bearing_yaw(c, center) - yaw0) for c in cand]) / G.LATTICE)
        poses = np.c_[cand, yaw0 + k * G.LATTICE]
        sc = self.score_poses(poses)
        dist_t, _ = pf._tree(lm.cell(here))
        cij = np.floor((cand - lm.origin) / L.CELL).astype(int)
        togo = dist_t[cij[:, 0] * lm.n + cij[:, 1]] * L.CELL / S.FORWARD_M
        turns = np.abs(k) + 2 * (togo > 0.5)
        # a seed can only score a proposal it reaches (and faces from) within the horizon
        head = np.array([G.bearing_yaw(here, c) for c in cand])
        k_out = np.round(np.abs([G.wrap(h - yaw0) for h in head]) / G.LATTICE)
        k_in = np.round(np.abs([G.wrap(G.bearing_yaw(c, center) - h) for c, h in zip(cand, head)]) / G.LATTICE)
        steps = np.where(togo > 0.5, togo + k_out + k_in, np.abs(k))
        togo = np.where(steps <= min(self.H, getattr(obs, "remaining", self.H)), togo, np.inf)
        rank = sc - self.C_ACT * (np.where(np.isfinite(togo), togo, 1e3) + turns)
        order = np.argsort(-rank)
        keep = []
        for i in order:
            if not np.isfinite(togo[i]):
                break
            if all(np.hypot(*(cand[i] - cand[j])) > 0.4 for j in keep):
                keep.append(i)
            if len(keep) >= 3 * self.N_PROP:
                break
        self.prop_info = [(round(float(cand[i][0]), 2), round(float(cand[i][1]), 2), round(float(sc[i]), 2)) for i in keep]
        return [cand[i] for i in keep]

    def seed_path(self, obs, here, yaw0, goal, center, H, long_way=False):
        """Lattice actions (0 F, 1 L, 2 R, 3 stay) that follow the local-map path here -> goal (pure pursuit on a
        LOOK m look-ahead point, turning the short way when off by > 15 deg), then turn to face the object and stay."""
        pf = obs.pathfinder
        cells = pf.path_cells(here, goal)
        if not cells:
            return None
        path = np.array([pf.lmap.center(c) for c in cells] + [np.asarray(goal, float)])
        seq, pos, k = [], here.astype(float).copy(), 0
        arrived = False
        if long_way:   # the initial rotation the other way round (does not sweep the view past the object)
            diff = G.wrap(G.bearing_yaw(pos, path[min(len(path) - 1, 2)]) - yaw0)
            n = int(round(abs(diff) / G.LATTICE))
            if n == 0:
                return None
            seq = [2 if diff > 0 else 1] * (12 - n)
            k = -(12 - n) if diff > 0 else (12 - n)
        for _ in range(H - len(seq)):
            yaw = yaw0 + k * G.LATTICE
            arrived = arrived or np.hypot(*(goal - pos)) <= 0.2
            if arrived:
                diff = G.wrap(G.bearing_yaw(pos, center) - yaw)
                a = self.STAY if abs(math.degrees(diff)) <= 15.0 else (1 if diff > 0 else 2)
            else:
                dd = np.hypot(*(path - pos).T)
                i0 = int(dd.argmin())
                ahead = next((q for q in path[i0:] if np.hypot(*(q - pos)) >= self.LOOK), path[-1])
                diff = G.wrap(G.bearing_yaw(pos, ahead) - yaw)
                a = 0 if abs(math.degrees(diff)) <= 15.0 else (1 if diff > 0 else 2)
            seq.append(a)
            if a == 0:
                pos = pos + S.FORWARD_M * G.forward(yaw)
            elif a in (1, 2):
                k += 1 if a == 1 else -1
        return np.array(seq[:H], int)

    # ---------------- the view planner ----------------
    def view_act(self, obs):
        pf = obs.pathfinder
        mask = np.asarray(obs.mask, bool)
        if self.vox is None:
            self.vox = VoxelMap(np.asarray(obs.position)[[0, 2]], obs.floor_y)
        self.vox.update(np.asarray(obs.depth), mask, obs.position, obs.yaw)
        if self.prev is not None and self.prev[2] == "move_forward":
            moved = float(np.hypot(*(np.asarray(obs.position) - self.prev[0])[[0, 2]]))
            if moved < 0.5 * S.FORWARD_M:
                pf.lmap.bump(self.prev[0], self.prev[1])
                pf.refresh(obs.position)
        if self.done or not len(obs.cloud) or not hasattr(pf, "lmap"):
            self.last = {"stop": "no_object" if not len(obs.cloud) else "done"}
            return None
        import os
        if os.environ.get("MPPI_DUMP"):
            import pickle
            pickle.dump({"obs": obs, "self": {k: v for k, v in self.__dict__.items() if k not in ("sim", "ctrl", "closer", "dev")}},
                        open(f"{os.environ['MPPI_DUMP']}_{obs.step}.pkl", "wb"))
        r = self.view_plan(obs)
        return self.execute(obs, r)

    def view_plan(self, obs):
        """One planning round; closer's planned drive (self._ref) is one of the seeds."""
        pf = obs.pathfinder
        here = np.asarray(obs.position)[[0, 2]].astype(float)
        cloud = np.asarray(obs.cloud)
        center = G.ObjectModel(cloud, obs.floor_y).center
        H = max(1, min(self.H, obs.remaining))
        pts, wts, info = self.model_points(obs)
        self._prepare(obs, pts, wts, center, self.object_cells(self.vox, cloud))
        seeds = []
        goals = list(self.proposals(obs, here, center, pts)) + [np.asarray(q, float) for q in getattr(self, "_extra_goals", [])]
        ref = self._ref
        for g in goals:
            for lw in (False, True):
                sp = self.seed_path(obs, here, obs.yaw, g, center, H, long_way=lw)
                if sp is not None:
                    seeds.append(sp)
        if len(seeds) > 2 * self.N_PROP:   # keep the seeds whose full rollout value is best
            v = self.values(np.array(seeds), here, obs.yaw, pf, center)[0]
            seeds = [seeds[i] for i in np.argsort(-v)[: 2 * self.N_PROP]]
        inc = j_inc = j_ref = None
        if self.plan_seq is not None and len(self.plan_seq) > 1:
            inc = np.r_[self.plan_seq[1:], np.full(H, self.STAY)][:H].astype(int)
            j_inc = len(seeds)
            seeds.append(inc)
        if ref is not None:
            j_ref = len(seeds)
            seeds.append(np.r_[ref, np.full(H, self.STAY)][:H].astype(int))
        j_stay = len(seeds)
        seeds.append(np.full(H, self.STAY))
        diff = G.wrap(G.bearing_yaw(here, center) - obs.yaw)
        seeds.append(np.r_[[1 if diff > 0 else 2], np.full(H - 1, self.STAY)][:H])   # face it and stay
        seeds = np.array(seeds)
        pi = np.tile(self.PRIOR, (H, 1))
        best_seq, best_val, best_t, inc_val, ref_val = None, -np.inf, 0, None, None
        for it in range(self.ITERS):
            n_seed = self.K // 2
            centres = seeds if best_seq is None else np.r_[seeds, best_seq[None]]
            base = centres[self.rng.integers(0, len(centres), n_seed)]
            keep = self.rng.random((n_seed, H)) >= self.EPS
            pert = np.where(keep, base, self.rng.integers(0, 3, (n_seed, H)))
            cum = pi.cumsum(1)
            free = (self.rng.random((self.K - n_seed, H, 1)) > cum[None]).sum(-1).clip(0, 2)
            seqs = np.r_[centres, pert, free][: self.K + len(centres)]
            stopped = np.maximum.accumulate(seqs == self.STAY, axis=1)
            seqs = np.where(stopped, self.STAY, seqs)
            vals, tstar, sc, poses_ = self.values(seqs, here, obs.yaw, pf, center)
            if it == 0:
                self.seed_info = [("".join("FLRS"[a] for a in seeds[j]), round(float(vals[j]), 2), int(tstar[j]),
                                   round(float(sc[j].max()), 2), [round(float(x), 2) for x in poses_[j, -1]])
                                  for j in range(len(seeds))]
            if it == 0:
                stay_val = float(vals[j_stay])
            if it == 0 and j_ref is not None:
                ref_val = float(vals[j_ref])
            if it == 0 and j_inc is not None:
                inc_val = float(vals[j_inc])
                vals[j_inc] += self.DELTA   # a new plan must beat the incumbent by the stop margin
            s0 = float(sc[0, 0])
            i = int(vals.argmax())
            if vals[i] > best_val:
                best_seq, best_val, best_t = seqs[i].copy(), float(vals[i]), int(tstar[i])
            wt = np.exp((vals - vals.max()) / self.LAMBDA)
            wt /= wt.sum()
            pi = np.einsum("k,kha->ha", wt, np.eye(4)[seqs])[:, :3] + 1e-3
            pi /= pi.sum(1, keepdims=True)
        return {"best_seq": best_seq, "best_val": best_val, "best_t": best_t, "s0": s0, "ref_val": ref_val, "stay_val": stay_val,
                "diff": diff, "info": info}

    def execute(self, obs, r):
        best_seq, best_val, best_t, s0, diff = r["best_seq"], r["best_val"], r["best_t"], r["s0"], r["diff"]
        off = math.degrees(abs(diff))
        self.last = {"phase": "view12", "J0": round(float(np.exp(s0)), 4), "V": round(best_val, 3), "t*": best_t,
                     "seq": "".join("FLRS"[a] for a in best_seq), "phantom": r["info"],
                     "props": getattr(self, "prop_info", []), "seeds": self.seed_info,
                     "cut": [self.top_cut, self.top_seen, self.bot_cut, self.bot_seen]}
        if r["ref_val"] is not None:
            self.last["deviate_from_closer"] = round(best_val - r["ref_val"], 2)
        if self._ref is not None:   # closer's planned drive (the warm start / far-out reference)
            self.last["ref"] = "".join("FLRS"[a] for a in self._ref)
            self.last["w"] = round(self._w, 2)
        stop = best_val - r["stay_val"] < self.DELTA or best_seq[0] == self.STAY or (self._w == 0.0 and best_t == 0)
        if stop:
            if off > self.FACE_DEG:
                action = "turn_left" if diff > 0 else "turn_right"
                self.plan_seq = None
            else:
                self.done = True
                self.last["stop"] = "best_view"
                return None
        else:
            action = ACTIONS[int(best_seq[0])]
            self.plan_seq = best_seq
        self.prev = (np.asarray(obs.position).copy(), obs.yaw, action)
        self.last["action"] = action
        return action


class MppiK0(Mppi):
    name, KAPPA = "mppi_k0", 0.0


class MppiK4(Mppi):
    name, KAPPA = "mppi_k4", 4.0


POLICIES_MPPI = {"mppi": Mppi, "mppi_k0": MppiK0, "mppi_k4": MppiK4, "mppi_dial": MppiDial, "mppi_v2": MppiV2, "mppi_v3": MppiV3, "mppi_v4": MppiV4, "mppi_v5": MppiV5, "mppi_v6": MppiV6, "mppi_v7": MppiV7, "mppi_v8": MppiV8, "mppi_v9": MppiV9, "mppi_v10": MppiV10, "mppi_v11": MppiV11, "mppi_v15": MppiV15}
