"""Terminal close-in (verification.close_in; env CLOSE_IN=1 in scripts/run_strategy.sh).

ApexNav stops 0.24-0.53 m from an object's nearest point, or ends as STUCKING 0.3-1.4 m from it; HM3D needs the agent
within 0.2 m of a goal viewpoint. When the planner is about to end the episode (STOP, or stucking) at an object the gate
accepted, the eval loop spends up to CLOSE_IN_BUDGET more actions on getting nearer to the accepted object's points, then
STOPs. It never hands back to ApexNav and never runs before the planner's own end, so everything earlier is native.

Per step (``decide``), d = horizontal distance from the agent centre to the nearest point of the object:
  d <= CLOSE_IN_STOP_M                     -> stop (reached)
  bearing to that point > TURN_DEG          -> turn toward it
  free depth ahead >= step + radius + margin -> move forward (the object's own pixels do not count as obstacles)
  else                                      -> blocked: stop after MAX_BLOCKS of them (a move that did not move counts too)
  actions used up                           -> stop (budget)
The object's points: the gate's accepted cloud (gate.accepted_cloud: the approach's object points, else the accepted
request frame's mask back-projected). Pure functions below; ``drive`` runs the loop through a caller-supplied ``act``.
"""
import math

import numpy as np

from active_approach import geom as G
from active_approach import scene as S

CLOSE_IN_STOP_M = 0.40      # stop within this of the object's nearest point (from the agent centre)
CLOSE_IN_BUDGET = 12        # actions before the final STOP
CLOSE_IN_NEAR_M = 3.0       # only for an accepted object this near the agent when the planner ends the episode
CLOSE_IN_FAR_M = 0.9        # ... and only when the planner stopped farther than this from it: ApexNav aims 0.5 m from the
                            # object (reach tolerance 0.2 m), so a nearer stop is a normal one and extra steps only cost
                            # SPL (600 HM3D-v2 eps, 2026-11-11: all triggers 0.0 m -> SR +0.2 SPL -0.4; > 0.9 m -> SR +0.2 SPL 0)
TURN_DEG = 15.0             # turn toward the object when its bearing is farther off than this
STEP_M = S.FORWARD_M
CLEAR_MARGIN_M = 0.05       # free space needed ahead: step + agent radius + this
MAX_BLOCKS = 2
BAND_BOTTOM_M, BAND_TOP_M = 0.15, G.BODY_TOP   # heights above the floor of what blocks the base (floor pixels do not)
OBJECT_PIXEL_M = 0.15       # a depth pixel this close (3D) to the object's points belongs to the object
TRIGGER_STATES = ("REACH_OBJECT", "STUCKING")


def distance_to(position, cloud):
    """Horizontal distance from ``position`` (x, y, z) to the nearest point of ``cloud`` (N, 3), and that point's (x, z)."""
    xz = np.asarray(cloud, float)[:, [0, 2]]
    d = np.hypot(*(xz - np.asarray(position, float)[[0, 2]]).T)
    k = int(np.argmin(d))
    return float(d[k]), xz[k]


def trigger_reason(planned_stop, final_state, step_cap, position, cloud):
    """Why a close-in starts at the planner's end of the episode ("stop" / "stucking"), or None. ``final_state``: params
    FINAL_RESULT value; ``step_cap``: the STOP is the step-cap's forced one; ``cloud``: the accepted object's points."""
    from params import FINAL_RESULT
    if not planned_stop or step_cap or cloud is None or len(cloud) == 0:
        return None
    if final_state == FINAL_RESULT.STUCKING:
        reason = "stucking"
    elif final_state == FINAL_RESULT.REACH_OBJECT:
        reason = "stop"
    else:
        return None
    return reason if CLOSE_IN_FAR_M < distance_to(position, cloud)[0] <= CLOSE_IN_NEAR_M else None


def object_pixels(depth, cam_pos, cam_rot, cloud):
    """Bool image of the depth pixels that lie on the object: back-projected within OBJECT_PIXEL_M of its points."""
    from scipy.spatial import cKDTree
    valid = (depth > 0.05) & (depth < S.DEPTH_MAX - 0.1)
    pts = S.backproject(depth, valid, cam_pos, cam_rot, stride=1)
    out = np.zeros(depth.shape, bool)
    if len(pts):
        near = cKDTree(np.asarray(cloud, float)).query(pts, distance_upper_bound=OBJECT_PIXEL_M)[0] < np.inf
        out[np.nonzero(valid)] = near   # backproject yields pixels in np.nonzero(valid) order
    return out


def free_distance(depth, ignore):
    """Free distance ahead (m) in the central footprint band of a level camera's depth image: the nearest depth of the
    pixels within the agent's half width that are above the floor and below BODY_TOP, not in ``ignore`` (the object's
    pixels), and valid. inf when the band holds nothing."""
    h, w = depth.shape
    v, u = np.mgrid[0:h, 0:w]
    z = depth
    x = (u - w / 2) * z / S.FOCAL
    height = S.CAMERA_HEIGHT - (v - h / 2) * z / S.FOCAL
    band = (z > 0.05) & (np.abs(x) <= S.AGENT_RADIUS) & (height > BAND_BOTTOM_M) & (height < BAND_TOP_M) & ~ignore
    return float(z[band].min()) if band.any() else math.inf


def decide(position, yaw, cloud, free_m, blocks, used, budget):
    """One step: (action, reason). action: "stop", "turn_left", "turn_right" or "move_forward" (module docstring).
    ``free_m``: free distance ahead (free_distance); ``blocks``: moves that did not move so far; ``used``: actions so far."""
    d, nearest = distance_to(position, cloud)
    if d <= CLOSE_IN_STOP_M:
        return "stop", "reached"
    if blocks >= MAX_BLOCKS:
        return "stop", "blocked"
    if used >= budget:
        return "stop", "budget"
    diff = G.wrap(G.bearing_yaw(np.asarray(position, float)[[0, 2]], nearest) - yaw)
    if abs(math.degrees(diff)) > TURN_DEG:
        return ("turn_left" if diff > 0 else "turn_right"), "turn"
    if free_m >= STEP_M + S.AGENT_RADIUS + CLEAR_MARGIN_M:
        return "move_forward", "move"
    return "stop", "blocked"   # nothing free ahead: no point in waiting for a second block


def drive(cloud, act, view, last, budget):
    """Run the close-in. ``act(name) -> last_step`` executes one action through the eval loop's execute_step (records,
    video, step counting); ``view() -> (position, yaw, cam_pos, cam_rot)`` of the agent now; ``last``: the last executed
    step (its "depth" in metres); ``budget``: actions before the final STOP. Always ends with STOP (unless the episode is
    over). Returns the log entry {start_d, end_d, actions, blocks, end}."""
    position, yaw, cam_pos, cam_rot = view()
    start_d = distance_to(position, cloud)[0]
    used = blocks = 0
    while True:
        position, yaw, cam_pos, cam_rot = view()
        free = free_distance(last["depth"], object_pixels(last["depth"], cam_pos, cam_rot, cloud))
        name, why = decide(position, yaw, cloud, free, blocks, used, budget)
        last = act(name)
        used += 1
        if name == "stop":
            break
        if name == "move_forward":
            moved = float(np.hypot(*(view()[0] - position)[[0, 2]]))
            blocks += moved < 0.05
    end_d = distance_to(view()[0], cloud)[0]
    return {"start_d": round(start_d, 3), "end_d": round(end_d, 3), "actions": used, "blocks": int(blocks), "end": why}


def object_points(frame, centroid, match_m):
    """The accepted request's object points (sim world frame): the frame's target masks whose planner-frame centroid is
    within ``match_m`` of the request's ``centroid``, back-projected with the frame's depth. ``frame``: gate.same_as_decided's
    {depth, masks, centroids, cam}. None when no mask matches."""
    c = np.asarray(centroid, float)
    parts = [S.backproject(frame["depth"], np.asarray(m).squeeze() > 0, *frame["cam"])
             for m, p in zip(frame["masks"], frame["centroids"])
             if p is not None and np.hypot(*(np.asarray(p[:2], float) - c)) <= match_m]
    pts = np.concatenate(parts) if parts else np.zeros((0, 3))
    return pts.astype(np.float32) if len(pts) >= 3 else None
