"""Run approach policies on scenarios in Habitat with the deployed decision flow.

    python -m active_approach.run --scenarios .../scenarios.jsonl --policies stay native fba_tour \
        --judge jev_sam3 --out videos/active_approach/runs/<name> --gpu G [--limit N | --ids ...]

Per run (judge, policy, scenario), as ApexNav's perception gate with the approach and view bank:
  1. the start frame (camera pitch 0: ApexNav's pitch after its initial spin; starts logged at -30 deg
     are re-rendered at 0 and flagged ``pitch_reset``) -> judge.first (/verify). Decided -> done.
  2. unsure: the policy gives viewpoints, the shared Controller drives there with move_forward 0.25 m /
     turn 30 deg only (no strafing, no tilt, allow_sliding false), counting every action. The judge is
     asked only when the agent is stationary at a viewpoint and faces the object (|yaw - bearing to the
     cloud centre| <= 20 deg, >= 200 mask px): the frame joins the bank and judge.views(bank, "early")
     decides or says unsure. En route (S2/S3), a frame is judged only if it is new by ApexNav's approach_view
     rule (0.5 m closer than or 30 deg around every judged view). Each view is judged once (judges.py caches).
  3. the policy has nothing left -> judge.views(bank, "arrival") (forced, M >= 0, V >= 0.5); the action
     budget (40, ApexNav approach_max_steps), no path or stuck -> "failure" (forced, M >= 0.15).
     An empty bank rejects.
Before step 1 the agent's map is seeded with a 12-turn depth panorama at the start (not counted: ApexNav
has its exploration map when it triggers); the height map then grows with every frame.

Logged in <out>/<judge>/<policy>/<id>/: steps.jsonl (pose, action, GT visibility, viewpoint, judge
reply), frames/NNN_rgb.jpg + NNN_mask.png for judged frames and the final one, result.json.
Existing result.json files are skipped (resume).
"""
import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from active_approach import geom as G
from active_approach import scene as S
from active_approach.judges import make_judge
from active_approach.policies import POLICIES, Controller, Observation, ring_views

STUCK_STEPS = 3
TOO_CLOSE_M = 0.8
JUDGE_MIN_PIXELS = 200
FRAME_MIN_PIXELS = 50     # --log-frames: a frame is "in view" from this many mask pixels
SAM3_MIN_PIXELS = 400     # judgeable frames (facing, >= 200 px) from this many mask px also get SAM3 scores (an exact
                          # /verify_views replay; the bank holds only these: decisions only when facing the object)
FACE_JUDGE_DEG = 20.0
MAX_IDLE_ARRIVALS = 4
TRACKER = "gt"            # --tracker reproj: the object mask after the trigger frame from the cloud (local_nav.track_mask)
NAME_USABLE = False       # --name-usable: JEV names only the frames decide.py can count (start, facing, >= 400 px)


def voxel_merge(cloud, points, size=0.05, cap=20000):
    if not len(points):
        return cloud
    merged = np.concatenate([cloud, points]) if len(cloud) else points
    _, keep = np.unique(np.floor(merged / size).astype(np.int64), axis=0, return_index=True)
    merged = merged[np.sort(keep)]
    return merged[-cap:]


def same_category_ids(names, scenario):
    """Other instances whose name says they are the scenario's category (for wrong_instance)."""
    category = scenario["category"]
    words = {category, category.replace("_", " "), scenario["target"], "tv" if category == "tv_monitor" else category}
    return {i for i, n in names.items() if n in words and i != scenario["object_id"]}


def judged_entry(view, yaw, record, jev=None):
    """What the planners know about a judged view (no GT): camera, yaw, which borders the mask touches, and
    JEV's raw naming distribution when the frame was named (--log-frames; round3.Seek's belief)."""
    return {"camera": view["camera"], "yaw": yaw, "top_clipped": view["top_clipped"],
            "side_clipped": view["side_clipped"], "clip": view["clip"], "step": view["step"],
            "area": view["area_fraction"], "jev": jev}


def predicted_features(o):
    """geom.Predictor statistics of the current pose (logged at judged poses for the proxy refit)."""
    if not len(o.cloud):
        return None
    model = G.ObjectModel(o.cloud, o.floor_y)
    pred = G.Predictor(model, o.cloud, o.hmap, o.pathfinder).predict(o.position[[0, 2]], o.yaw)
    return {k: round(v, 4) for k, v in pred.items()}


def judgeable_now(mask, offset):
    return int(mask.sum()) >= JUDGE_MIN_PIXELS and offset <= FACE_JUDGE_DEG


def reprojection_iou(cloud, mask, cam_pos, cam_rot, dilate=4):
    """IoU of the mask with the tracked cloud (before this frame) splatted into the frame."""
    import cv2
    if not len(cloud):
        return None
    pts = cloud[:: max(1, len(cloud) // 4000)]
    u, v, z = S.project(pts, cam_pos, cam_rot)
    ok = (z > 0.05) & (u >= 0) & (u < S.WIDTH) & (v >= 0) & (v < S.HEIGHT)
    splat = np.zeros((S.HEIGHT, S.WIDTH), np.uint8)
    splat[v[ok].astype(int), u[ok].astype(int)] = 1
    splat = cv2.dilate(splat, np.ones((2 * dilate + 1, 2 * dilate + 1), np.uint8)) > 0
    union = (splat | mask).sum()
    return round(float((splat & mask).sum() / union), 4) if union else None


def frame_row(scenario, policy, judge, step, record, view, rgb, depth, mask, cloud_before, cam, last_action, arrived,
              judgeable, hmap, o, out):
    """One in-view frame for the belief replay and the proxy refit (docs/active_approach_planners.md 6.6)."""
    import cv2
    import time as _time

    from active_approach.round3 import phantom_model
    from active_approach.scenarios import outline
    ys, xs = np.nonzero(mask)
    center = np.asarray(view["centroid"])
    here = np.asarray(view["camera"])
    row = {"run": view["run"], "scenario": scenario["id"], "policy": policy.name, "judge": judge.name, "step": step,
           "action": last_action, "time": _time.time(),
           "pose": {"position": record["position"], "yaw": record["yaw"], "pitch": 0.0,
                    "distance": round(view["range_m"], 3),
                    "bearing": round(float(np.arctan2(*(here - center)[::-1])), 4),
                    "offset_deg": record["facing_offset_deg"], "judgeable": bool(judgeable), "arrived": bool(arrived)},
           "mask": {"px": int(mask.sum()), "area": round(view["area_fraction"], 5),
                    "bbox": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())], **view["clip"],
                    "depth_median": round(float(np.median(depth[mask])), 3),
                    "reprojection_iou": reprojection_iou(cloud_before, mask, *cam), "cloud_points": len(cloud_before),
                    "thumb": f"frames/t{step:03d}.jpg"},
           "gt": {**{k: record["vis"][k] for k in ("vis_frac", "trunc_frac", "occ_frac", "area", "facing_cos", "distance")},
                  "role": scenario["role"], "object_id": scenario["object_id"], "mask_on_object": 1.0}}
    if len(o.cloud):
        model, extra, info = phantom_model(o)
        pred = G.Predictor(model, o.cloud, hmap, o.pathfinder, extra=extra).predict(o.position[[0, 2]], o.yaw)
        row["pred"] = {**{k: round(v, 4) for k, v in pred.items()},
                       "d_fit": round(G.d_fit(model, o.yaw), 3), "phantom": info}
    name_it = not NAME_USABLE or step == 0 or (bool(judgeable) and int(mask.sum()) >= SAM3_MIN_PIXELS)
    if hasattr(judge, "naming") and name_it:  # also fills the judge's per-view caches: /verify_views reuses these outputs
        row["jev"] = judge.naming(view, sam3=bool(judgeable) and int(mask.sum()) >= SAM3_MIN_PIXELS)
    cv2.imwrite(str(out / "frames" / f"t{step:03d}.jpg"),
                cv2.resize(cv2.cvtColor(outline(rgb, mask), cv2.COLOR_RGB2BGR), (160, 120)),
                [cv2.IMWRITE_JPEG_QUALITY, 70])
    return row


def seed_map(sim, scenario, hmap):
    """12-turn depth panorama at the start pose into the height map (object pixels excluded)."""
    for k in range(12):
        obs = S.set_pose(sim, scenario["start"]["position"], scenario["start"]["yaw"] + k * G.LATTICE)
        depth = np.asarray(obs["depth"]).squeeze()
        mask = np.asarray(obs["semantic"]).squeeze() == scenario["object_id"]
        hmap.update(depth, mask, *S.camera(sim))


def run_one(sim, scenario, surface, policy, judge, views, out, budget, names, log_frames=False, full=False,
            selfcontained=False):
    """``selfcontained``: no navmesh and no host map (local_nav): no seed panorama, the map is centred on the start
    and the floor is the agent's own height; every policy plans on local_nav.LocalPathfinder (the self-built
    occupancy map) and local_nav.LocalController drives.
``full``: after the first decision, keep following the policy to its end (no more judge calls) so the
    belief replay sees the whole trajectory; the result is the first decision and its cost."""
    import cv2
    out.mkdir(parents=True, exist_ok=True)
    (out / "frames").mkdir(exist_ok=True)
    oid, pf, floor_y = scenario["object_id"], sim.pathfinder, scenario["floor_y"]
    run_name = f"{policy.name}_{scenario['id']}"
    lmap = None
    if selfcontained:
        from active_approach import local_nav as L
        obs = S.set_pose(sim, scenario["start"]["position"], scenario["start"]["yaw"], 0.0)
        floor_y = float(S.agent_pose(sim)[0][1])
        hmap = G.HeightMap(np.asarray(scenario["start"]["position"], float)[[0, 2]], floor_y)
        lmap = L.LocalMap(hmap)
        pf = L.LocalPathfinder(lmap, floor_y)
        pf.refresh(scenario["start"]["position"])
    else:
        hmap = G.HeightMap(np.asarray(scenario["object_center"])[[0, 2]], floor_y)
        seed_map(sim, scenario, hmap)
    obs = S.set_pose(sim, scenario["start"]["position"], scenario["start"]["yaw"], 0.0)
    yaw0 = S.agent_pose(sim)[1]
    policy.reset({"scenario": scenario, "pathfinder": pf, "views": views, "sim": sim, "surface": surface,
                  "judge_url": getattr(judge, "url", None)})
    if selfcontained:
        if getattr(policy, "driver", None):
            raise ValueError(f"{policy.name}: the MPC driver needs the navmesh")
        controller = L.LocalController(sim, lmap, pf)
    elif getattr(policy, "driver", None):
        from active_approach.svpto import MPCDriver
        controller = MPCDriver(sim, *policy.driver[1:])
    else:
        controller = Controller(sim)
    enroute = bool(getattr(policy, "enroute", False))
    enroute_calls = enroute_decisions = valid_poses = moving_poses = 0
    last_action = None
    others = same_category_ids(names, scenario)
    cloud, bank, judged, calls = np.zeros((0, 3)), [], [], []
    path_length, stuck, idle, actions, stops = 0.0, 0, 0, 0, 0
    end, verdict, arrived, first_done, first_verdict = None, None, False, False, None
    decided = None
    log = open(out / "steps.jsonl", "w")
    frames_log = open(out / "frames.jsonl", "w") if log_frames else None
    framed, named = set(), {}
    record = None

    def save(step, rgb, mask):
        cv2.imwrite(str(out / "frames" / f"{step:03d}_rgb.jpg"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, 85])
        cv2.imwrite(str(out / "frames" / f"{step:03d}_mask.png"), mask.astype(np.uint8) * 255)

    def forced(mode):
        if decided is not None:
            return decided["verdict"]
        if not bank:
            return {"status": "wrong", "source": "no_views", "mode": mode}
        reply = judge.views(bank, mode)
        calls.append(reply)
        return reply

    step = 0
    last_center = np.asarray(scenario["start"]["position"], float)[[0, 2]]
    skips = 0
    while True:
        position, yaw, _ = S.agent_pose(sim)
        cam = S.camera(sim)
        sem = np.asarray(obs["semantic"]).squeeze()
        mask = sem == oid
        if TRACKER == "reproj" and step > 0 and len(cloud):   # T1: no GT after the trigger frame
            mask = L.track_mask(cloud, np.asarray(obs["depth"]).squeeze(), *S.camera(sim))
        depth = np.asarray(obs["depth"]).squeeze()
        rgb = np.asarray(obs["rgb"])[..., :3]
        vis = S.visibility(obs, surface, oid, *cam)
        cloud_before = cloud
        new_pts = S.backproject(np.where(depth < G.DEPTH_MAX, depth, 0), mask, *cam)
        if TRACKER == "reproj" and len(new_pts):   # the grown mask touches the floor under the object: drop it
            new_pts = new_pts[new_pts[:, 1] - floor_y > 0.04]
        cloud = voxel_merge(cloud, new_pts)
        hmap.update(depth, mask, *cam)
        if lmap is not None:
            lmap.set_object(cloud)
            pf.refresh(position)
        if len(cloud):
            center = last_center = G.ObjectModel(cloud, floor_y).center
        else:   # self-contained: never the GT centre
            center = last_center if selfcontained else np.asarray(scenario["object_center"])[[0, 2]]
        offset = math.degrees(abs(G.wrap(yaw - G.bearing_yaw(position[[0, 2]], center))))
        rows = np.flatnonzero(mask.any(1))
        view = {"rgb": rgb, "mask": mask, "target": scenario["target"], "role": scenario["role"], "vis": vis,
                "step": step, "index": 0, "run": run_name, "scenario": scenario["id"], "camera": (float(position[0]), float(position[2])),
                "yaw": float(yaw),
                "centroid": (float(center[0]), float(center[1])),
                "range_m": float(np.hypot(*(position[[0, 2]] - center))),
                "area_fraction": float(mask.mean()), "score": 0.5,
                "top_clipped": bool(len(rows) and rows[0] == 0),
                "side_clipped": bool(mask[:, 0].any() or mask[:, -1].any()),
                "clip": {"top": bool(mask[0].any()), "bottom": bool(mask[-1].any()), "left": bool(mask[:, 0].any()),
                         "right": bool(mask[:, -1].any())}}
        from perception_gate.evidence import mask_touches_border, view_quality
        view["quality"] = view_quality(0.5, view["area_fraction"], mask_touches_border(mask.astype(np.uint8)))
        record = {"step": step, "position": [round(float(x), 3) for x in position], "yaw": round(yaw, 4),
                  "vis": vis, "facing_offset_deg": round(offset, 1), "cloud_points": len(cloud),
                  "other_instance_pixels": max((int((sem == i).sum()) for i in others), default=0)}
        if log_frames and step not in framed and int(mask.sum()) >= FRAME_MIN_PIXELS:
            framed.add(step)
            row = frame_row(scenario, policy, judge, step, record, view, rgb, depth, mask, cloud_before,
                            cam, last_action, arrived, judgeable_now(mask, offset), hmap,
                            Observation(step, position, yaw, rgb, depth, mask, cloud, pf, hmap,
                                        yaw0, floor_y, judged, budget - actions, arrived,
                                        {"target": scenario["target"], "first": first_verdict}), out)
            if row.get("jev"):
                named[step] = row["jev"]["probabilities"]
            frames_log.write(json.dumps(row, default=float) + "\n")
        # the trigger frame is judged as ApexNav does (the detection, wherever it is in the frame); later
        # frames only when stationary at a viewpoint and facing the object
        judgeable = int(mask.sum()) >= JUDGE_MIN_PIXELS and (not first_done or offset <= FACE_JUDGE_DEG)
        if first_done and len(cloud) and last_action is not None:  # SVPTO validity of every pose on the way
            from active_approach.svpto import Visibility, heading_index
            o_vis = Observation(step, position, yaw, rgb, depth, mask, cloud, pf, hmap, yaw0, floor_y, judged)
            valid = Visibility(o_vis, keep_zeta=True).terms(position, heading_index(yaw, yaw0))["valid"]
            record["valid_pose"] = bool(valid)
            moving_poses += 1
            valid_poses += bool(valid)
        # en route (S2/S3 and their ablation): a frame right after a forward step, facing the object, whole
        # object across (no side clip), new by ApexNav's approach_view rule (0.5 m closer than or 30 deg
        # around every judged view) is judged like a stop (unforced)
        novel_view = False
        if (decided is None and enroute and first_done and not arrived and last_action == "move_forward" and judgeable
                and not view["side_clipped"] and judged):
            here = np.asarray(view["camera"])
            ranges = [float(np.hypot(*(np.asarray(j["camera"]) - center))) for j in judged]
            bearings = [math.atan2(*(np.asarray(j["camera"]) - center)[::-1]) for j in judged]
            b = math.atan2(*(here - center)[::-1])
            novel_view = all(view["range_m"] <= r - 0.5 or abs(G.wrap(b - bb)) >= math.radians(30)
                             for r, bb in zip(ranges, bearings))
            if getattr(policy, "enroute", None) == "valid":  # round 3 fix 7: bank only valid poses' frames
                novel_view = novel_view and bool(record.get("valid_pose"))
        if novel_view:
            bank.append(view)
            judged.append(judged_entry(view, yaw, record, named.get(step)))
            save(step, rgb, mask)
            verdict = judge.views(bank, "early")
            verdict["enroute"] = True
            calls.append(verdict)
            enroute_calls += 1
            record["judge"] = verdict
            record["pred"] = predicted_features(Observation(step, position, yaw, rgb, depth, mask, cloud, pf, hmap,
                                                            yaw0, floor_y, judged))
            if verdict["status"] != "insufficient":
                end, enroute_decisions = "resolved", enroute_decisions + 1
        elif arrived and judged and judged[-1]["step"] == step:
            stops += 1   # this very frame was just judged en route
        elif not first_done or arrived:
            record["judgeable"] = judgeable
            if judgeable:
                bank.append(view)
                judged.append(judged_entry(view, yaw, record, named.get(step)))
                save(step, rgb, mask)
                if decided is not None:
                    stops += 1
                elif not first_done:
                    verdict = judge.first(view)
                    verdict["mode"] = "first"
                    first_verdict = verdict
                else:
                    verdict = judge.views(bank, "early")
                    stops += 1
                if decided is None:
                    calls.append(verdict)
                    record["judge"] = verdict
                record["pred"] = predicted_features(Observation(step, position, yaw, rgb, depth, mask, cloud, pf,
                                                                hmap, yaw0, floor_y, judged))
                if decided is None and verdict["status"] != "insufficient":
                    end = "start" if not first_done else "resolved"
            elif first_done:
                stops += 1
            first_done = True
        if full and end in ("start", "resolved") and decided is None:
            decided = {"verdict": verdict, "end": end, "actions": actions, "path_m": round(path_length, 3),
                       "stops": stops, "step": step}
            record["decided"] = True
            end = None
        action = None
        if end is None:
            if actions >= budget:
                end, verdict = "budget", forced("failure")
            else:
                o = Observation(step, position, yaw, rgb, depth, mask, cloud, pf, hmap, yaw0, floor_y, judged,
                                budget - actions, arrived, {"target": scenario["target"], "first": first_verdict,
                                                            "jev": named.get(step)})
                arrived = False
                if hasattr(policy, "act"):   # a pilot picks the next action itself (e.g. round3.JevPilot)
                    action = policy.act(o)
                    record["pilot"] = getattr(policy, "last", None)
                    if action is None:
                        end, verdict = "policy_done", forced("arrival")
                elif (viewpoint := policy.next_viewpoint(o)) is None:
                    end, verdict = "policy_done", forced("arrival")
                else:
                    record["viewpoint"] = {"position": [round(float(x), 3) for x in viewpoint.position],
                                           "yaw": round(viewpoint.yaw, 4), **viewpoint.info}
                    try:
                        action = controller.action(viewpoint, o)
                        if getattr(controller, "last_error", None):
                            record["giveup"], controller.last_error = controller.last_error, None
                    except Exception as error:  # habitat_sim.errors.GreedyFollowerError
                        record["error"] = str(error)
                        if selfcontained and skips < 3:   # unreachable on the local map: the policy's next one
                            skips += 1
                            arrived, idle = True, idle + 1
                            record["skipped"] = True
                            record["action"] = None
                            log.write(json.dumps(record, default=float) + "\n")
                            last_action = None
                            continue
                        if getattr(controller, "debug", None):
                            np.savez_compressed(out / f"local_map_{step:03d}.npz", **controller.debug)
                        end, verdict = "no_path", forced("failure")
                    else:
                        if action is None:  # stationary at the viewpoint, facing it: judge this frame
                            arrived, idle = True, idle + 1
                            if idle > MAX_IDLE_ARRIVALS:
                                end, verdict = "policy_done", forced("arrival")
            if end is not None and verdict is not None and "judge" not in record:
                record["judge"] = verdict
        record["action"] = action
        log.write(json.dumps(record, default=float) + "\n")
        if end is not None:
            save(step, rgb, mask)
            break
        if action is None:  # judge the same frame next iteration
            last_action = None
            continue
        idle, last_action = 0, action
        obs = sim.step(action)
        actions += 1
        step += 1
        moved = float(np.linalg.norm((S.agent_pose(sim)[0] - position)[[0, 2]]))
        path_length += moved
        stuck = stuck + 1 if action == "move_forward" and moved < 0.02 else 0
        if stuck >= STUCK_STEPS:
            # judge where we stand if possible, then force
            arrived = False
            end = "stuck"
            position, yaw, _ = S.agent_pose(sim)
            verdict = forced("failure")
            log.write(json.dumps({"step": step, "position": [round(float(x), 3) for x in position],
                                  "judge": verdict, "action": None}, default=float) + "\n")
            save(step, np.asarray(obs["rgb"])[..., :3], np.asarray(obs["semantic"]).squeeze() == oid)
            break
    log.close()
    if frames_log:
        frames_log.close()
    trajectory = {"actions": actions, "path_m": round(path_length, 3), "stops": stops, "end": end}
    if decided is not None:
        verdict, end, actions, path_length, stops = (decided["verdict"], decided["end"], decided["actions"],
                                                     decided["path_m"], decided["stops"])

    target = scenario["role"] == "target"
    decision = "accept" if verdict["status"] == "correct" else "reject"
    final_vis = S.visibility(obs, surface, oid, *S.camera(sim))
    tags = []
    if end in ("stuck", "no_path", "budget"):
        tags.append(end)
    if verdict.get("mode") in ("arrival", "failure"):
        tags.append("forced")
    if verdict.get("source") == "no_views":
        tags.append("no_views")
    reason = getattr(policy, "reason", None)
    if reason:
        tags.append(reason)
    if final_vis["pixels"] < 50:
        tags.append("lost")
    if final_vis["distance"] is not None and final_vis["distance"] < TOO_CLOSE_M:
        tags.append("too_close")
    if final_vis["trunc_frac"] >= 0.3:
        tags.append("truncated")
    result = {"id": scenario["id"], "source": scenario.get("source"), "role": scenario["role"],
              "tag": scenario["tag"], "target": scenario["target"], "object_name": scenario["object_name"],
              "policy": policy.name, "judge": judge.name, "pitch_reset": scenario["start"].get("pitch", 0.0) != 0.0,
              "decision": decision, "correct": decision == ("accept" if target else "reject"),
              "decided_by": verdict.get("mode") if verdict.get("source") != "no_views" else "no_views",
              "start_sure": end == "start", "resolved": end in ("start", "resolved"),
              "end": end, "actions": actions, "path_m": round(path_length, 3), "stops": stops,
              "bank": len(bank), "judge_calls": len(calls), "enroute_calls": enroute_calls,
              "enroute_decisions": enroute_decisions,
              "valid_pose_fraction": round(valid_poses / moving_poses, 3) if moving_poses else None, "start_vis": json.loads(log_first(out))["vis"],
              "final_vis": final_vis, "decision_views": verdict.get("views"), "verdict": verdict,
              "failure_tags": tags, "trajectory": trajectory}
    json.dump(result, open(out / "result.json", "w"), indent=1, default=float)
    return result


def log_first(out):
    with open(out / "steps.jsonl") as stream:
        return stream.readline()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scenarios", type=Path, required=True)
    parser.add_argument("--policies", nargs="+", default=["stay", "native", "oracle"], choices=sorted(POLICIES))
    parser.add_argument("--judge", default="gt", choices=("gt", "gt_seek", "jev_sam3"))
    parser.add_argument("--judge-url", help="jev_sam3: perception.server URL (selector jev_sam3)")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=40, help="action budget (ApexNav approach_max_steps)")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--limit", type=int, help="first N scenarios")
    parser.add_argument("--ids", nargs="*", help="only these scenario ids")
    parser.add_argument("--ids-file", type=Path, help="only the ids in this file (one per line)")
    parser.add_argument("--cache", default="videos/active_approach/cache")
    parser.add_argument("--full-trajectory", action="store_true",
                        help="after the first decision keep following the policy (no more judge calls; for the replay)")
    parser.add_argument("--tracker", default="gt", choices=("gt", "reproj"),
                        help="object mask after the trigger frame: GT semantics or the GT-free reprojection tracker")
    parser.add_argument("--name-usable", action="store_true",
                        help="--log-frames: JEV /name only on frames decide.py counts (saves ~half the VLM calls)")
    parser.add_argument("--selfcontained", action="store_true",
                        help="no navmesh / host map: plan on the self-built local map (local_nav)")
    parser.add_argument("--log-frames", action="store_true",
                        help="frames.jsonl: every in-view frame (pose, mask, predicted geometry, GT visibility, JEV "
                             "naming via /name with --judge jev_sam3) and 160x120 thumbnails")
    args = parser.parse_args(argv)
    global NAME_USABLE, TRACKER
    NAME_USABLE, TRACKER = args.name_usable, args.tracker
    scenarios = [json.loads(l) for l in open(args.scenarios)]
    wanted = set(args.ids or []) | (set(open(args.ids_file).read().split()) if args.ids_file else set())
    if wanted:
        scenarios = [s for s in scenarios if s["id"] in wanted]
    if args.limit:
        scenarios = scenarios[:args.limit]
    judge = make_judge(args.judge, args.judge_url)
    args.out.mkdir(parents=True, exist_ok=True)
    if hasattr(judge, "persist"):
        judge.persist(args.out / f"first_{args.judge}.jsonl")
    by_scene = defaultdict(list)
    for s in scenarios:
        by_scene[s["scene"]].append(s)
    (args.out / "views").mkdir(parents=True, exist_ok=True)
    __import__("os").environ.setdefault("MPPI_GPU", str(args.gpu))   # mppi_v15 batched scoring on the same GPU
    for scene, items in by_scene.items():
        todo = [(s, p) for s in items for p in args.policies
                if not (args.out / args.judge / p / s["id"] / "result.json").exists()]
        if not todo:
            continue
        surfaces = S.surfaces(scene, args.cache, {s["object_id"] for s in items})
        names = S.semantic_objects(scene)
        sim = S.make_sim(scene, args.gpu)
        try:
            for s, p in todo:
                views = []
                if p == "oracle":
                    vpath = args.out / "views" / f"{s['id']}.json"
                    if vpath.exists():
                        views = json.load(open(vpath))
                    else:
                        views = ring_views(sim, surfaces[s["object_id"]], s["object_id"], s["floor_y"],
                                           s["start"]["position"])
                        json.dump(views, open(vpath, "w"))
                r = run_one(sim, s, surfaces[s["object_id"]], POLICIES[p](), judge, views,
                            args.out / args.judge / p / s["id"], args.steps, names, args.log_frames,
                            args.full_trajectory, args.selfcontained)
                print(f"{judge.name} {p:11s} {s['id']:36s} {r['decision']:6s} correct={r['correct']} "
                      f"by={r['decided_by']} actions={r['actions']} path={r['path_m']} end={r['end']} "
                      f"tags={r['failure_tags']}", flush=True)
        finally:
            sim.close()


if __name__ == "__main__":
    main()
