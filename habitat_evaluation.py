"""
Habitat ObjectNav Evaluation Script for HM3D/MP3D Datasets

This script evaluates object navigation performance using the Habitat simulator
with support for HM3D-v1, HM3D-v2, and MP3D datasets. It communicates with ROS for
real-time planning and decision making, incorporates vision-language models
for object detection and image-text matching, and generates comprehensive
evaluation metrics.

The pieces live in dedicated modules:
    habitat2ros/eval_node.py        - every ROS topic this process touches
    basic_utils/eval/video.py       - video frame composition
    basic_utils/eval/reporting.py   - running totals and record files
    basic_utils/eval/habitat_utils.py - pose/label helpers shared with replay
    basic_utils/record_episode/     - per-episode forensic recorder
    perception_gate/                - verify target clusters with perception.Verifier

Usage:
    # Run with HM3D-v1 dataset
    python habitat_evaluation.py --dataset hm3dv1

    # Run with HM3D-v2 dataset (default)
    python habitat_evaluation.py --dataset hm3dv2

    # Run with MP3D dataset
    python habitat_evaluation.py --dataset mp3d

    # Run with HM3D-OVON val_unseen (labels verbatim, frozen LLM answers)
    python habitat_evaluation.py --dataset ovon

    # Test specific episode
    python habitat_evaluation.py --dataset hm3dv2 test_epi_num=10

Author: Zager-Zhang
"""

# Standard library imports
import os
import signal
import time
from copy import deepcopy

# Third-party library imports
from hydra import initialize, compose
import numpy as np
import rclpy
from omegaconf import DictConfig, OmegaConf
import tqdm

# Habitat-related imports
import habitat
from habitat.config.default import patch_config
from habitat.config import read_write
from habitat.sims.habitat_simulator.actions import HabitatSimActions

# ROS message imports
from plan_env.msg import MultipleMasksWithConfidence

# Local project imports
from basic_utils.eval.habitat_utils import (
    agent_world_pose,
    episode_goal_positions_in_planner_frame,
    load_category_mapping,
    resolve_label,
    semantic_instances_for_masks,
    slugify_label,
)
from basic_utils.eval import ovon_dataset  # noqa: F401  registers dataset type OVON-v1
from basic_utils.eval.reporting import RunTotals
from basic_utils.eval.setup import (
    add_visualization_measurements,
    parse_dataset_arg,
    signal_handler,
)
from basic_utils.eval.video import EpisodeVideo
from basic_utils.failure_check.failure_check import check_failure, is_on_same_floor
from basic_utils.object_point_cloud_utils.object_point_cloud import (
    get_object_point_cloud,
)
from basic_utils.record_episode.episode_recorder import EpisodeRecorder
from habitat2ros.eval_node import HabitatEvalNode, query_planner_fusion_config
from llm.answer_reader.answer_reader import read_answer
from params import HABITAT_STATE, ACTION, EXPL_RESULT, FINAL_RESULT, result_dirname
from vlm.utils.get_itm_message import get_itm_message_cosine
from vlm.utils.get_object_utils import get_object


ACTION_TO_HABITAT = {
    ACTION.MOVE_FORWARD: "move_forward",
    ACTION.TURN_LEFT: "turn_left",
    ACTION.TURN_RIGHT: "turn_right",
    ACTION.TURN_DOWN: "look_down",
    ACTION.TURN_UP: "look_up",
    ACTION.STOP: "stop",
}

# Camera pitch change applied by a single look_up / look_down action
PITCH_STEP = np.pi / 6.0


def episode_is_feasible(episode):
    """True when at least one goal sits on the agent's starting floor."""
    return any(
        is_on_same_floor(height=goal.position[1], episode=episode)
        for goal in episode.goals
    )


def build_verification_gate(cfg, video_output_path):
    """The perception gate, or None when verification.enabled is off.

    perception_gate is imported only here, so a baseline run has no new
    dependency. Raises when the perception server is unreachable.
    """
    verification_cfg = cfg.get("verification", {})
    if not verification_cfg or not verification_cfg.get("enabled", False):
        return None
    from perception_gate import VerificationGate

    rgb_sensor = OmegaConf.select(
        cfg, "habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor"
    )
    camera = None
    if rgb_sensor is not None:
        camera = {
            key: rgb_sensor.get(key) for key in ("hfov", "width", "height", "position")
        }
        camera = OmegaConf.to_container(OmegaConf.create(camera), resolve=True)
    return VerificationGate(verification_cfg, video_output_path, camera=camera)


def execute_step(env, node, ctx, action, planned_action, count_steps, camera_pitch, label, room, llm_answer,
                 cld_with_score_msg, external_approach=False):
    """Execute one action and process its observation (detection, planner map, gate evidence, records).
    ``external_approach``: a step of an approach the gate drives itself (no ACTION_FINISH, no reopens).
    Returns {observations, rgb, depth (m), target masks / centroids (label 0), pose, pass_object}."""
    gate = ctx["gate"]
    sync = ctx["sync_step"]
    print(f"\n--------------Step: {count_steps}--------------")
    print(f"Finding [{label}]; Action: {action};")

    # Notify ROS system that action execution is starting
    node.publish_int32(node.state_pub, HABITAT_STATE.ACTION_EXEC)

    observations = env.step(action)

    # Keep the untouched observation: get_object() below overwrites
    # observations["rgb"] in place with the annotated image
    recorder = ctx["recorder"]
    rgb_raw = (
        observations["rgb"].copy()
        if recorder.enabled and recorder.capture_rgb
        else None
    )
    rgb_gate = None
    if gate:
        rgb_gate = rgb_raw if rgb_raw is not None else observations["rgb"].copy()

    # Calculate ITM cosine similarity score
    cosine = get_itm_message_cosine(observations["rgb"], label, room)
    print(f"Target related room: {room}")
    print(f"ITM cosine similarity: {cosine:.3f}")
    if not sync:
        node.publish_float64(node.itm_score_pub, cosine)

    # Detect objects in the current observation
    observations["rgb"], score_list, object_masks_list, label_list = get_object(
        label, observations["rgb"], ctx["detector_cfg"], llm_answer
    )
    detection_semantic_instances = semantic_instances_for_masks(
        observations.get("semantic"), object_masks_list
    )

    # Publish habitat observations to ROS
    observations["camera_pitch"] = camera_pitch
    node.msg_observations = deepcopy(observations)
    del observations["camera_pitch"]
    if not sync:
        node.ros_pub.habitat_publish_ros_topic(node.msg_observations)

    # Generate and publish object point clouds
    (
        cld_with_score_msg.point_clouds,
        detection_world_centroids,
    ) = get_object_point_cloud(
        ctx["cfg"], observations, object_masks_list, node,
        return_centroids=True,
    )
    cld_with_score_msg.confidence_scores = score_list
    cld_with_score_msg.label_indices = label_list
    if not sync:
        node.cld_with_score_pub.publish(cld_with_score_msg)
    step_pose = agent_world_pose(observations)

    # Keep target detections as verification evidence; a rejected cluster
    # seen anew after the cooldown is reopened for another check
    reopened = gate.observe(
        count_steps, rgb_gate, object_masks_list, label_list, score_list,
        detection_world_centroids, detection_semantic_instances, step_pose,
        camera_pitch, node.object_fusion, external_approach=external_approach,
    ) if gate else []
    if sync:
        # The reopens ride with this step, so the planner applies them before it
        # plans the next action
        node.publish_step_observation(
            count_steps, node.msg_observations, itm=cosine,
            clouds=cld_with_score_msg, reopen=reopened)
    else:
        for cluster_id in reopened:
            node.publish_verification_result(0, cluster_id, 0)

    info = env.get_metrics()
    ctx["video"].track(observations)

    recorder.log_step(
        step=count_steps,
        action=planned_action,
        pose=step_pose,
        camera_pitch=camera_pitch,
        rgb_raw=rgb_raw,
        rgb_annotated=observations["rgb"],
        score_list=score_list,
        label_list=label_list,
        itm=cosine,
        distance_to_goal=info["distance_to_goal"],
        final_state=node.final_state,
        expl_result=node.expl_result,
        object_fusion=node.object_fusion,
        object_masks=object_masks_list,
        detection_world_centroids=detection_world_centroids,
        detection_semantic_instances=detection_semantic_instances,
        verification=gate.pop_events() if gate else None,
    )
    ctx["video"].capture(
        observations, info, itm_score=cosine, step=count_steps, target=label
    )

    # Track if agent has passed close to the target
    pass_object = 1 if info["distance_to_goal"] <= ctx["success_distance"] else 0

    if ctx["step_delay"] > 0:
        time.sleep(ctx["step_delay"])

    # Notify ROS system that action execution is complete (not for a step of an approach the
    # gate drives: the planner is waiting for the verdict, and only integrates the step)
    if not external_approach:
        node.publish_int32(node.state_pub, HABITAT_STATE.ACTION_FINISH)
        if sync:
            node.publish_int32(node.action_finish_step_pub, count_steps)


    target = [i for i, l in enumerate(label_list) if int(l) == 0]
    ctx["last_step"] = {"observations": observations, "rgb": rgb_gate,
            "depth": np.asarray(observations["depth"], np.float32).squeeze() * ctx["max_depth"],
            "masks": [object_masks_list[i] for i in target],
            "centroids": [detection_world_centroids[i] if detection_world_centroids is not None else None
                          for i in target],
            "pose": step_pose, "pass_object": pass_object}
    return ctx["last_step"]


MIN_PLANNER_STEPS = 12   # the gate's planner gets at least this many actions after a native approach


def level_camera(env, node, ctx, count_steps, camera_pitch, label, room, llm_answer, cld_with_score_msg):
    """The gate's planners assume camera pitch 0 (ApexNav's pitch after its initial spin): look up / down
    until level. Returns (count_steps, camera_pitch, pass_object)."""
    passed = 0
    while abs(camera_pitch) > 1e-6 and count_steps < ctx["max_episode_steps"] - 1:
        planned = ACTION.TURN_UP if camera_pitch < 0 else ACTION.TURN_DOWN
        camera_pitch += PITCH_STEP if planned == ACTION.TURN_UP else -PITCH_STEP
        count_steps += 1
        out = execute_step(env, node, ctx, getattr(HabitatSimActions, ACTION_TO_HABITAT[planned]), planned,
                           count_steps, camera_pitch, label, room, llm_answer, cld_with_score_msg,
                           external_approach=True)
        passed = max(passed, out["pass_object"])
    return count_steps, camera_pitch, passed


def drive_last_mile(env, node, ctx, approach, count_steps, camera_pitch, label, room, llm_answer,
                    cld_with_score_msg):
    """Drive the gate's last mile (LiveApproach.start_last_mile) from the last executed step until it stops the
    agent (STOP: the episode ends here, final state REACH_OBJECT) or hands back. Returns (stopped, count_steps,
    pass_object)."""
    passed = 0
    last = ctx["last_step"]
    while not env.episode_over:
        if count_steps >= ctx["max_episode_steps"] - 1:   # the last step is the STOP
            approach.finish_last_mile("episode_end")
            name = "stop" if approach.last_mile["end"] == "stop" else None
        else:
            name = approach.step(last["rgb"], last["depth"], last["masks"], last["centroids"], last["pose"])
        if name is None:
            break
        planned = {"move_forward": ACTION.MOVE_FORWARD, "turn_left": ACTION.TURN_LEFT,
                   "turn_right": ACTION.TURN_RIGHT, "stop": ACTION.STOP}[name]
        if planned == ACTION.STOP:
            node.final_state = FINAL_RESULT.REACH_OBJECT
        count_steps += 1
        last = execute_step(env, node, ctx, getattr(HabitatSimActions, ACTION_TO_HABITAT[planned]), planned,
                            count_steps, camera_pitch, label, room, llm_answer, cld_with_score_msg,
                            external_approach=True)
        passed = max(passed, last["pass_object"])
        if planned == ACTION.STOP:
            break
    if env.episode_over and approach.driving_last_mile:
        approach.finish_last_mile("episode_over")
    lm = approach.last_mile or {}
    print(f"[verification] gate last mile: {lm.get('end')} ({lm.get('reason')}) after {lm.get('actions')} actions, "
          f"{lm.get('distance')} m from the object")
    return env.episode_over, count_steps, passed


def run_gate_last_mile(env, node, ctx, request, count_steps, camera_pitch, label, room, llm_answer,
                       cld_with_score_msg):
    """After an accept on the first view (gate.handle -> 1, approach_planner runs): the gate drives the last mile
    itself when the object is within LAST_MILE_REACH_M (a fresh LiveApproach in last-mile mode finds it first).
    None when it is farther or verification.last_mile is false (ApexNav's own navigation goes on to it); else (stopped, count_steps,
    camera_pitch, pass_object, steps taken)."""
    from perception_gate.approach import LAST_MILE_REACH_M, LM_BUDGET, LiveApproach
    gate = ctx["gate"]
    obj = (float(request["centroid_x"]), float(request["centroid_y"]))
    pose = ctx["last_step"]["pose"]
    if (not gate.last_mile or not ctx["sync_step"]
            or np.hypot(obj[0] - pose[0], obj[1] - pose[1]) > LAST_MILE_REACH_M):
        return None
    start_steps = count_steps
    count_steps, camera_pitch, passed = level_camera(env, node, ctx, count_steps, camera_pitch, label, room,
                                                     llm_answer, cld_with_score_msg)
    tag = f"epi{gate.episode['run_index']}_c{request['cluster_id']}_s{count_steps}_lastmile"
    log_dir = os.path.join(gate.log_dir, "approach", tag) if gate.log_dir else None
    budget = max(0, min(LM_BUDGET, ctx["max_episode_steps"] - 1 - count_steps))
    approach = LiveApproach(env.sim, gate.cfg["url"], gate.episode["target"], obj, budget, gate.approach_planner,
                            log_dir=log_dir, tag=tag, last_mile=True)
    try:
        stopped, count_steps, p = drive_last_mile(env, node, ctx, approach, count_steps, camera_pitch, label, room,
                                                  llm_answer, cld_with_score_msg)
        if len(approach.cloud):
            gate.remember_object(approach.object_points(), 1)
    finally:
        approach.close()
    return stopped, count_steps, camera_pitch, max(passed, p), count_steps - start_steps


def run_close_in(env, node, ctx, planned_action, step_cap, count_steps, camera_pitch, label, room, llm_answer,
                 cld_with_score_msg):
    """The terminal close-in (verification.close_in; perception_gate.close_in): the planner's action is STOP (at an
    object, or ApexNav's stucking end) and the gate accepted an object near the agent. Before that STOP the agent spends
    up to close_in_budget actions getting nearer to the accepted object, and ends with its own STOP (final state
    REACH_OBJECT). None when nothing triggers (the planner's STOP goes ahead as it is, also at the step cap);
    else (count_steps, camera_pitch, pass_object)."""
    from active_approach import scene as S
    from perception_gate import close_in
    gate = ctx["gate"]
    position = S.agent_pose(env.sim)[0]
    reason = close_in.trigger_reason(planned_action == ACTION.STOP, node.final_state, step_cap, position,
                                     gate.accepted_cloud)
    last_step = ctx.get("last_step")
    if reason is None or last_step is None:
        return None
    count_steps, camera_pitch, passed = level_camera(env, node, ctx, count_steps, camera_pitch, label, room,
                                                     llm_answer, cld_with_score_msg)
    # the budget's actions and the final STOP all stay below the step cap's forced STOP
    budget = max(0, min(int(gate.cfg["close_in_budget"]), ctx["max_episode_steps"] - 2 - count_steps))
    steps = {"move_forward": ACTION.MOVE_FORWARD, "turn_left": ACTION.TURN_LEFT, "turn_right": ACTION.TURN_RIGHT,
             "stop": ACTION.STOP}
    state = {"count": count_steps, "passed": passed, "last": ctx["last_step"]}

    def act(name):
        if name == "stop":
            node.final_state = FINAL_RESULT.REACH_OBJECT
        state["count"] += 1
        state["last"] = execute_step(env, node, ctx, getattr(HabitatSimActions, ACTION_TO_HABITAT[steps[name]]),
                                     steps[name], state["count"], camera_pitch, label, room, llm_answer,
                                     cld_with_score_msg, external_approach=True)
        state["passed"] = max(state["passed"], state["last"]["pass_object"])
        return state["last"]

    def view():
        position, yaw, _ = S.agent_pose(env.sim)
        return (position, yaw, *S.camera(env.sim))

    entry = close_in.drive(gate.accepted_cloud, act, view, state["last"], budget)
    gate.log_close_in(reason, entry)
    return state["count"], camera_pitch, state["passed"]


def run_gate_approach(env, node, ctx, request, count_steps, camera_pitch, label, room, llm_answer,
                      cld_with_score_msg):
    """An approach driven by the gate (verification.approach_planner; perception_gate.approach): the planner
    is waiting for the verdict on ``request``. Each step goes through execute_step (detector, planner map,
    gate evidence, records) without ACTION_FINISH. Returns (state 1 / -1 / 2, count_steps, camera_pitch,
    pass_object, stopped); 2 (deferred: the approach found no evidence) reaches the planner as a non-accept, so
    ApexNav drops the cluster and keeps exploring until the gate's re-check reopens it. After an accept the same
    planner drives the last mile (drive_last_mile; not with verification.last_mile false): ``stopped`` when it
    ended the episode with STOP, else the accept reaches the planner without a forced target."""
    from perception_gate.approach import LM_BUDGET, LiveApproach
    gate = ctx["gate"]
    if not ctx["sync_step"]:
        raise RuntimeError("verification.approach_planner needs sync_step")
    pending = gate.pending_approach(request["cluster_id"])
    run = gate.episode["run_index"]
    tag = f"epi{run}_c{request['cluster_id']}_s{count_steps}"
    log_dir = os.path.join(gate.log_dir, "approach", tag) if gate.log_dir else None
    start = time.time()
    count_steps, camera_pitch, passed = level_camera(env, node, ctx, count_steps, camera_pitch, label, room,
                                                     llm_answer, cld_with_score_msg)
    native = int(request.get("approach_steps", 0) or 0) if int(request.get("approach_id", 0) or 0) else 0
    approach = LiveApproach(env.sim, gate.cfg["url"], gate.episode["target"], pending["object"],
                            min(max(pending["max_steps"] - native, MIN_PLANNER_STEPS),
                                ctx["max_episode_steps"] - 1 - count_steps),
                            gate.approach_planner, log_dir=log_dir, tag=tag,
                            bank=gate.approach_bank(request["cluster_id"]))
    last = ctx["last_step"]   # the request's frame: the last executed step
    frame = {"rgb": last["rgb"], "depth": last["depth"]}
    detections, pose = (last["masks"], last["centroids"]), last["pose"]
    while True:
        name = approach.step(frame["rgb"], frame["depth"], detections[0], detections[1], pose)
        if name is None:
            break
        planned = {"move_forward": ACTION.MOVE_FORWARD, "turn_left": ACTION.TURN_LEFT,
                   "turn_right": ACTION.TURN_RIGHT}[name]
        count_steps += 1
        out = execute_step(env, node, ctx, getattr(HabitatSimActions, name), planned, count_steps, camera_pitch,
                           label, room, llm_answer, cld_with_score_msg, external_approach=True)
        passed = max(passed, out["pass_object"])
        frame = {"rgb": out["rgb"], "depth": out["depth"]}
        detections = (out["masks"], out["centroids"])
        pose = out["pose"]
        if env.episode_over:
            approach._forced("episode_over")
            break
    result = dict(approach.result, seconds=round(time.time() - start, 1))
    state = gate.finish_external_approach(request, count_steps, result, approach_log=log_dir)
    gate.remember_object(approach.object_points(), state)   # a later request for the same object inherits it
    print(f"[verification] gate approach ({gate.approach_planner}) cluster {request['cluster_id']}: "
          f"{result['status']} by {result['by']} after {result['actions']} actions")
    stopped = False
    try:
        if state == 1 and gate.last_mile and not env.episode_over:
            approach.start_last_mile(max(0, min(LM_BUDGET, ctx["max_episode_steps"] - 1 - count_steps)))
            stopped, count_steps, p = drive_last_mile(env, node, ctx, approach, count_steps, camera_pitch, label,
                                                      room, llm_answer, cld_with_score_msg)
            passed = max(passed, p)
    finally:
        approach.close()
    return state, count_steps, camera_pitch, passed, stopped or env.episode_over


def request_frame(env, ctx):
    """The request frame for gate.same_as_decided: the last executed step's depth, target masks / centroids and
    the camera (sim frame)."""
    from active_approach import scene as S
    last = ctx.get("last_step")
    if last is None:
        return None
    return {"depth": last["depth"], "masks": last["masks"], "centroids": last["centroids"], "cam": S.camera(env.sim)}


def run_episode(env, node, ctx, label, llm_answer, room):
    """Drive one episode to completion.

    Returns (count_steps, pass_object) - the step count and whether the agent
    ever came within the success radius of a goal during the episode.
    """
    count_steps = 0
    pass_object = 0.0
    camera_pitch = 0.0
    cld_with_score_msg = MultipleMasksWithConfidence()

    gate = ctx["gate"]
    # sync_step: each step's map inputs go out once, in one StepObservation, and the
    # next action is only taken once it is known to be planned from that step
    sync = ctx["sync_step"]
    last_progress = time.time()
    while rclpy.ok() and not env.episode_over:
        rclpy.spin_once(node, timeout_sec=0.1)

        # Perception gate: the planner holds its action until the cluster it
        # selected is verified. Answer the latched request; no sim step here
        if node.verification_request is not None:
            request, node.verification_request = node.verification_request, None
            if sync:
                node.wait_for_fusion(request.get("fusion_seq", 0))
            # A gate-off run never gets here (main() checks the planner flag);
            # accepting keeps a mismatched planner from waiting for its timeout
            handoff = gate.approach_handoff(request, ctx["last_step"]) if gate and gate.approach_planner else None
            if handoff == "native":
                # a re-request of an approach ApexNav drives, object not yet close in view: keep approaching
                node.publish_verification_result(request["seq"], request["cluster_id"], 3,
                                                 gate.pending_approach(request["cluster_id"]))
                last_progress = time.time()
                continue
            if handoff == "planner":
                state = 3
            else:
                state = gate.handle(request, node.object_fusion, count_steps,
                                    frame=request_frame(env, ctx) if gate.approach_planner or gate.close_in else None) if gate else 1
                if state == 3 and gate.approach_planner and not gate.approach_close(request, ctx["last_step"]):
                    state = "native"   # far or out of view: ApexNav's object navigation gets close first
            if state == "native":
                node.publish_verification_result(request["seq"], request["cluster_id"], 3,
                                                 gate.pending_approach(request["cluster_id"]))
                last_progress = time.time()
                continue
            if state == 3 and gate.approach_planner:
                # the gate drives the approach itself and answers the request with its verdict; the
                # planner integrates the approach steps (resume_step) and replans from the new pose
                state, count_steps, camera_pitch, passed, stopped = run_gate_approach(
                    env, node, ctx, request, count_steps, camera_pitch, label, room, llm_answer,
                    cld_with_score_msg)
                pass_object = max(pass_object, passed)
                if not stopped:   # (stopped: the gate's last mile ended the episode)
                    node.publish_verification_result(request["seq"], request["cluster_id"], state,
                                                     resume_step=max(count_steps, 1))
                last_progress = time.time()
                continue
            if state == 1 and gate and gate.approach_planner:
                # accepted on the first view: the gate drives the last mile itself when the object is near
                lm = run_gate_last_mile(env, node, ctx, request, count_steps, camera_pitch, label, room,
                                        llm_answer, cld_with_score_msg)
                if lm is not None:
                    stopped, count_steps, camera_pitch, passed, steps = lm
                    pass_object = max(pass_object, passed)
                    if not stopped:   # handed back: plain accept, or replan from the new pose without a lock
                        node.publish_verification_result(request["seq"], request["cluster_id"], 1,
                                                         resume_step=max(count_steps, 1) if steps else 0)
                    last_progress = time.time()
                    continue
            # state 3: approach the object first; the planner re-requests the cluster
            approach = gate.pending_approach(request["cluster_id"]) if gate and state == 3 else None
            node.publish_verification_result(request["seq"], request["cluster_id"], state, approach)
            last_progress = time.time()
            continue

        # Keep publishing observations, confidence, and trigger so the FSM
        # always has fresh odom and can re-trigger after episode transitions.
        # Not in sync_step mode: repeats were integrated again by the planner
        if not sync:
            node.publish_observations()

        # Skip episode if target is not on the same floor
        if not episode_is_feasible(env.current_episode):
            break

        # Parse action from decision system
        if node.global_action is None:
            if sync and time.time() - last_progress > ctx["sync_stall_timeout_s"]:
                raise RuntimeError(
                    f"[sync_step] no action for step {count_steps + 1} within "
                    f"{ctx['sync_stall_timeout_s']:.0f} s: planner stalled")
            continue
        if sync:
            planned = node.step_action
            if planned["step"] != count_steps:
                raise RuntimeError(
                    f"[sync_step] action planned from step {planned['step']}, expected {count_steps}")
            node.wait_for_fusion(planned["fusion_seq"])
            # The FSM verdict the action was published with (-1: none yet)
            if planned["final_state"] >= 0:
                node.final_state = planned["final_state"]
            if planned["expl_result"] >= 0:
                node.expl_result = planned["expl_result"]
            last_progress = time.time()
        step_cap = count_steps == ctx["max_episode_steps"] - 1
        if step_cap:
            node.global_action = ACTION.STOP
        planned_action = node.global_action
        node.global_action = None
        if gate and gate.close_in:
            closed = run_close_in(env, node, ctx, planned_action, step_cap, count_steps, camera_pitch, label, room,
                                  llm_answer, cld_with_score_msg)
            if closed is not None:   # the close-in ended the episode with its own STOP
                count_steps, camera_pitch, passed = closed
                pass_object = max(pass_object, passed)
                continue

        name = ACTION_TO_HABITAT.get(planned_action)
        if name is None:
            continue
        action = getattr(HabitatSimActions, name)
        if planned_action == ACTION.TURN_DOWN:
            camera_pitch -= PITCH_STEP
        elif planned_action == ACTION.TURN_UP:
            camera_pitch += PITCH_STEP

        count_steps += 1
        pass_object = max(pass_object, execute_step(
            env, node, ctx, action, planned_action, count_steps, camera_pitch, label, room, llm_answer,
            cld_with_score_msg)["pass_object"])

    return count_steps, pass_object


def main(
    cfg: DictConfig,
    node: HabitatEvalNode,
    step_delay: float = 0.0,
    dataset: str = "hm3dv2",
) -> None:
    category_to_coco, id_to_name = load_category_mapping()
    # mp3d: MP3D category -> detector name (HM3D/MP3D configs); none: the
    # episode's category verbatim (open-vocabulary OVON)
    label_mapping = cfg.get("label_mapping", "mp3d")
    if label_mapping not in ("mp3d", "none"):
        raise ValueError(f"label_mapping must be mp3d or none, not {label_mapping}")

    node.final_state = 0
    node.expl_result = 0
    cfg = patch_config(cfg)

    # Extract configuration parameters
    video_output_path = cfg.video_output_path.format(split=cfg.habitat.dataset.split)
    max_episode_steps = cfg.habitat.environment.max_episode_steps
    success_distance = cfg.habitat.task.measurements.success.success_distance

    llm_cfg = cfg.llm
    llm_answer_path = llm_cfg.llm_answer_path

    # Single test parameters
    env_num_once = cfg.test_epi_num  # Which episode to test for single run
    flag_once = env_num_once != -1  # Whether to run single test

    os.makedirs(os.path.dirname(llm_answer_path), exist_ok=True)
    os.makedirs(video_output_path, exist_ok=True)

    # Forensic recorder: buffers each episode in RAM, writes pose/trajectory/RGB
    # only for the failure categories listed in record.flush_on
    recorder = EpisodeRecorder(cfg.get("record", {}), video_output_path)
    recorder.set_context(dataset=dataset)
    planner_cfg_read = False

    # sync_step: deterministic simulator-planner step (launch the planner with
    # sync_step:=true too). A verification error or timeout aborts instead of
    # failing open, so timing never decides a verdict
    sync_step = bool(cfg.get("sync_step", False))
    if sync_step:
        node.enable_sync_step()
        if cfg.get("verification", {}).get("enabled", False):
            with read_write(cfg):
                cfg.verification.max_consecutive_errors = 1

    video = EpisodeVideo(cfg)
    gate = build_verification_gate(cfg, video_output_path)
    totals = RunTotals(
        video_output_path, cfg.record_file_name, cfg.continue_file_name, flag_once
    )

    env = habitat.Env(add_visualization_measurements(cfg))
    print("Environment creation successful")
    number_of_episodes = env.number_of_episodes

    if totals.num_total >= number_of_episodes:
        raise ValueError("Already finished all episodes.")

    ctx = {
        "cfg": cfg,
        "detector_cfg": cfg.detector,
        "recorder": recorder,
        "video": video,
        "gate": gate,
        "max_episode_steps": max_episode_steps,
        "success_distance": success_distance,
        "step_delay": step_delay,
        "sync_step": sync_step,
        "sync_stall_timeout_s": float(cfg.get("sync_stall_timeout_s", 600.0)),
        "max_depth": float(cfg.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.max_depth),
    }
    detector_cfg_plain = OmegaConf.to_container(cfg.detector, resolve=True)

    pbar = tqdm.tqdm(total=number_of_episodes)

    # Fast-forward to the resume point, or to the single episode under test
    env_count = env_num_once if flag_once else totals.num_total
    while env_count:
        pbar.update()
        env.current_episode = next(env.episode_iterator)
        env_count -= 1

    for _ in range(number_of_episodes - totals.num_total):
        node.publish_int32_array(
            node.progress_pub, [totals.num_total, number_of_episodes]
        )

        # Initialize episode variables
        node.global_action = None
        node.final_state = FINAL_RESULT.EXPLORE
        node.expl_result = EXPL_RESULT.EXPLORATION
        # Drop the previous episode's fusion snapshot; the planner re-inits its
        # object map on EPISODE_FINISH but the latch here would otherwise persist
        node.object_fusion = None
        node.verification_request = None
        node.step_action = None

        observations = env.reset()
        observations["camera_pitch"] = 0.0
        node.msg_observations = deepcopy(observations)
        del observations["camera_pitch"]

        label = env.current_episode.object_category
        if label_mapping == "mp3d":
            label = resolve_label(label, category_to_coco, id_to_name)

        # Get LLM answer and fusion threshold for the target object
        llm_answer, room, node.fusion_threshold = read_answer(
            llm_answer_path, llm_cfg.llm_response_path, label, llm_cfg.llm_client
        )

        # Begin buffering this episode. run_index is the 0-based iterator
        # position, i.e. exactly the value to pass back as test_epi_num
        recorder.start(
            scene_id=env.current_episode.scene_id,
            episode_id=env.current_episode.episode_id,
            run_index=env_num_once if flag_once else totals.num_total,
            label=label,
            llm_answer=llm_answer,
            room=room,
            fusion_threshold=node.fusion_threshold,
            detector_cfg=detector_cfg_plain,
            start_pose=agent_world_pose(observations),
            ground_truth_goals=episode_goal_positions_in_planner_frame(
                env.current_episode
            ),
            start_rgb=observations["rgb"],
        )
        if gate:
            gate.start_episode(
                run_index=env_num_once if flag_once else totals.num_total,
                scene=env.current_episode.scene_id,
                episode_id=env.current_episode.episode_id,
                label=label,
                goal_object_ids=[
                    getattr(goal, "object_id", None) for goal in env.current_episode.goals
                ],
                max_episode_steps=ctx["max_episode_steps"],
            )

        video.start_episode(observations)
        video.capture(observations, env.get_metrics(), step=0, target=label)

        # Start publishing basic information and trigger messages
        node.start_observation_timer()
        print("Agent is waiting in the environment!!!")
        node.wait_for_planner_ready(timeout=300.0 if sync_step else None)
        node.stop_observation_timer()
        # Requests latched before the planner reset belong to the last episode
        node.start_verification_episode()

        # The planner is alive by now, so its fusion arm can be read back once
        # and stamped into every record produced by this run
        gate_on = gate is not None
        if (recorder.enabled or gate_on or sync_step) and not planner_cfg_read:
            planner_cfg = query_planner_fusion_config(node, extra=("verification.approach",))
            # The approach flag joins the recorded planner config only in approach runs, so
            # other runs keep their records unchanged
            approach_on = bool(gate_on and gate.approach)
            planner_approach = (planner_cfg or {}).get("verification.approach")
            if planner_cfg is not None and not approach_on:
                planner_cfg.pop("verification.approach", None)
            recorder.set_context(planner_cfg=planner_cfg)
            planner_cfg_read = True
            # The approach (gate state 3) needs the planner's approach module, and the
            # planner's stop-backstop turn must not run without it
            if (planner_cfg is not None or approach_on) and bool(planner_approach) != approach_on:
                raise RuntimeError(
                    f"verification.approach is {approach_on} here but {planner_approach} in "
                    "the planner; launch it with verification_approach:=" + str(approach_on).lower()
                )
            # Both halves of the gate must agree, or the planner either never
            # asks or waits for answers that never come
            planner_gate = (planner_cfg or {}).get("verification.enabled")
            if (planner_cfg is not None or gate_on) and bool(planner_gate) != gate_on:
                raise RuntimeError(
                    f"verification.enabled is {gate_on} here but {planner_gate} in "
                    "the planner; launch it with verification_gate:=" + str(gate_on).lower()
                )
            planner_sync = (planner_cfg or {}).get("sync_step")
            if bool(planner_sync) != sync_step:
                raise RuntimeError(
                    f"sync_step is {sync_step} here but {planner_sync} in the planner; "
                    "launch it with sync_step:=" + str(sync_step).lower()
                )

        print("Agent is ready to go!!!!")

        count_steps, pass_object = run_episode(
            env, node, ctx, label, llm_answer, room
        )

        # Notify ROS system that current episode evaluation is complete, then
        # wait for the FSM to reset before the next episode starts
        node.publish_int32(node.state_pub, HABITAT_STATE.EPISODE_FINISH)
        node.wait_for_planner_reset()

        # Collect evaluation metrics
        info = env.get_metrics()
        success = info["success"]
        near_object = 1 if info["distance_to_goal"] <= success_distance else 0

        # Determine episode result
        if success == 1:
            result_text = "success"
        else:
            result_text = check_failure(
                env.current_episode,
                node.final_state,
                node.expl_result,
                count_steps,
                max_episode_steps,
                pass_object,
                near_object,
                recorder.latest_commit_ground_truth(),
            )

        # Flush the buffered episode. Done before the flag_once break below so
        # single-episode debug runs also produce a record
        record_dir = recorder.finish(
            result_text,
            {
                "success": success,
                "spl": info["spl"],
                "soft_spl": info["soft_spl"],
                "distance_to_goal": info["distance_to_goal"],
                "distance_to_goal_reward": info["distance_to_goal_reward"],
                "pass_object": pass_object,
                "near_object": near_object,
                "final_state": node.final_state,
                "expl_result": node.expl_result,
                "task_number": totals.num_total + 1,
                **({"verification": gate.finish_episode()} if gate else {}),
            },
        )
        if record_dir:
            print(f"Episode artifacts written to {record_dir}")

        scene_id = env.current_episode.scene_id
        episode_id = env.current_episode.episode_id
        totals.add(info, success)

        # Video name uses the run's episode number - the same "No.N task"
        # written to the record file - rather than the dataset episode id,
        # which repeats across episodes and would overwrite clips
        video_dir = (
            "videos"
            if flag_once
            else os.path.join(video_output_path, result_dirname(result_text))
        )
        video.save(video_dir, f"epi{totals.num_total}_{slugify_label(label)}")

        averages, run_totals = totals.report(result_text)

        if flag_once:
            break

        totals.persist(scene_id, episode_id, label, result_text, averages, run_totals)
        node.publish_float32_array(node.record_pub, totals.record_payload())

        pbar.update()
        env.current_episode = next(env.episode_iterator)
        time.sleep(0.1)  # wait a moment

    env.close()
    pbar.close()
    video.close()


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    rclpy.init()

    # Register habitat config search path plugin before Hydra init
    from habitat.config.default_structured_configs import (
        HabitatConfigPlugin,
        register_hydra_plugin,
    )
    register_hydra_plugin(HabitatConfigPlugin)

    try:
        node = HabitatEvalNode()
        dataset, step_delay, overrides = parse_dataset_arg()
        cfg_name = f"habitat_eval_{dataset}"
        # Compose the chosen config and pass through extra Hydra overrides
        with initialize(version_base=None, config_path="config"):
            cfg = compose(config_name=cfg_name, overrides=overrides)
        main(cfg, node, step_delay=step_delay, dataset=dataset)
    except Exception as e:
        print(f"Unexpected error occurred: {e}")
        rclpy.shutdown()
        os._exit(1)
    finally:
        rclpy.shutdown()
