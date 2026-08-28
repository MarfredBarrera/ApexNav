"""
Re-drive a recorded episode through Habitat without the ROS stack.

A recorded false positive stores the exact action sequence the planner produced,
so the episode can be replayed deterministically by feeding those actions back
into the simulator. That needs neither the planner nor the four VLM servers,
which makes it the cheap way to inspect what the agent actually saw before it
stopped at the wrong object.

Note this replays the trajectory that happened; it does not re-run the planner.
A fresh planner run over the same episode (test_epi_num=<run_index>) may diverge,
because the planner's A* search is bounded by wall-clock time and the ROS loop
cadence is not deterministic.

Usage:
    python -m basic_utils.record_episode.replay_episode <record_dir>
    python -m basic_utils.record_episode.replay_episode <record_dir> --save-rgb out/
    python -m basic_utils.record_episode.replay_episode <record_dir> \
        --save-depth depth/ --dump-clouds clouds/
"""

import argparse
import json
import math
import os

import numpy as np

from basic_utils.eval.habitat_utils import agent_world_pose


def camera_tf_matrix(pose, camera_height, pitch):
    """World-from-camera transform, including camera pitch.

    The camera frame produced by ``get_point_cloud`` is (forward, left, up).
    The planner's own Python path builds this transform with
    ``xyz_yaw_to_tf_matrix(pos, yaw)`` - yaw only - which is why the C++ side
    discards frames where the camera is tilted (map_ros.cpp gates on
    ``camera_pitch < 1.5``). Recording the pitch per step lets us do it
    properly here, so look_down frames become usable instead of being dropped.

    Positive pitch is look_up, matching the convention habitat_evaluation.py
    uses when it tracks camera_pitch across look_up / look_down actions.
    """
    x, y, yaw = pose
    cy, sy = np.cos(yaw), np.sin(yaw)
    rot_yaw = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])

    cp, sp = np.cos(pitch), np.sin(pitch)
    # Rotation about the camera's left axis; negative pitch tips forward down
    rot_pitch = np.array([[cp, 0.0, -sp], [0.0, 1.0, 0.0], [sp, 0.0, cp]])

    tf = np.eye(4)
    tf[:3, :3] = rot_yaw @ rot_pitch
    tf[:3, 3] = [x, y, camera_height]
    return tf


def depth_sensor_params(sensor_cfg):
    """Pull the depth intrinsics and range out of a habitat sensor config."""
    width = int(sensor_cfg["width"])
    height = int(sensor_cfg["height"])
    hfov = float(sensor_cfg["hfov"])
    return {
        "min_depth": float(sensor_cfg["min_depth"]),
        "max_depth": float(sensor_cfg["max_depth"]),
        "width": width,
        "height": height,
        "fx": width / (2 * math.tan(hfov * math.pi / 360.0)),
        "fy": height / (2 * math.tan(hfov / width * height * math.pi / 360.0)),
    }


def depth_frame(observations, params, metric=False):
    """The step's depth image as a 2D float32 array.

    Habitat hands back depth normalized to [0,1] because the eval configs set
    ``normalize_depth: true``. Converting to metres is the same arithmetic
    extract_object_cloud() applies, kept here so every consumer in this module
    shares one definition.
    """
    d = np.asarray(observations["depth"], dtype=np.float32)
    if d.ndim == 3:
        d = d[:, :, 0]
    if metric:
        d = d * (params["max_depth"] - params["min_depth"]) + params["min_depth"]
    return d


def depth_to_world_cloud(depth_metric, pose, camera_height, pitch, params,
                         pixel_stride=1, max_range_frac=0.95):
    """Unproject a metric depth frame into the planner's world frame.

    Takes depth already in metres - see depth_frame() - and returns an (N,3)
    float32 array. Points at or beyond max_range_frac of the sensor range are
    dropped, since normalized depth saturates there and the returned range is
    not a real measurement.
    """
    metric = np.asarray(depth_metric)
    vs, us = np.mgrid[0 : metric.shape[0] : pixel_stride, 0 : metric.shape[1] : pixel_stride]
    z = metric[vs, us]
    valid = z < params["max_depth"] * max_range_frac
    vs, us, z = vs[valid], us[valid], z[valid]
    if z.size == 0:
        return np.zeros((0, 3), dtype=np.float32)

    px = (us - metric.shape[1] // 2) * z / params["fx"]
    py = (vs - metric.shape[0] // 2) * z / params["fy"]
    cloud = np.stack((z, -px, -py), axis=-1)  # camera frame: forward, left, up

    tf = camera_tf_matrix(pose, camera_height, pitch)
    world = (tf[:3, :3] @ cloud.T).T + tf[:3, 3]
    return world.astype(np.float32)


ACTION_NAMES = {
    0: "stop",
    1: "move_forward",
    2: "turn_left",
    3: "turn_right",
    4: "look_down",
    5: "look_up",
}


def load_record(record_dir):
    """Read meta.json and steps.jsonl from a record directory."""
    with open(os.path.join(record_dir, "meta.json"), "r", encoding="utf-8") as handle:
        meta = json.load(handle)

    steps = []
    steps_path = os.path.join(record_dir, "steps.jsonl")
    with open(steps_path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                steps.append(json.loads(line))

    return meta, steps


def discover_record_dirs(path):
    """Return record directories represented by ``path``.

    A path containing ``meta.json`` is treated as one episode. Otherwise the
    immediate child directories containing ``meta.json`` are treated as a
    category batch, such as ``records/false_positive``.
    """
    path = os.path.abspath(path)
    if os.path.isfile(os.path.join(path, "meta.json")):
        return [path]
    if not os.path.isdir(path):
        raise FileNotFoundError(f"Record directory does not exist: {path}")
    record_dirs = []
    for name in sorted(os.listdir(path)):
        candidate = os.path.join(path, name)
        if os.path.isdir(candidate) and os.path.isfile(os.path.join(candidate, "meta.json")):
            record_dirs.append(candidate)
    if not record_dirs:
        raise FileNotFoundError(
            f"No episode records (directories containing meta.json) found in {path}"
        )
    return record_dirs


def record_matches_target(record_dir, target_episodes):
    """Match a record against configured run index, episode ID, or dirname."""
    selectors = [str(value) for value in (target_episodes or [])]
    if not selectors or "all" in {value.lower() for value in selectors}:
        return True
    with open(os.path.join(record_dir, "meta.json"), "r", encoding="utf-8") as handle:
        meta = json.load(handle)
    candidates = {
        os.path.basename(record_dir),
        str(meta.get("episode_id", "")),
        str(meta.get("run_index", "")),
    }
    return any(selector in candidates for selector in selectors)


def replay(record_dir, cfg):
    """Replay the recorded actions and regenerate the requested artifacts.

    Args:
        record_dir: a directory written by EpisodeRecorder
        cfg: the composed config/replay.yaml

    Returns the maximum absolute pose deviation from the recording.
    """
    # Imported here so --help works without a habitat install
    import habitat
    from habitat.config.default import patch_config
    from habitat.sims.habitat_simulator.actions import HabitatSimActions
    from hydra import initialize_config_dir, compose
    from habitat.config.default_structured_configs import (
        HabitatConfigPlugin,
        register_hydra_plugin,
    )

    meta, steps = load_record(record_dir)
    dataset = meta.get("dataset") or "hm3dv2"
    run_index = meta["run_index"]

    print(
        f"Replaying {meta['scene']} episode {meta['episode_id']} "
        f"(run_index {run_index}, target '{meta['target_label']}', "
        f"result '{meta.get('result')}') - {len(steps)} steps"
    )

    register_hydra_plugin(HabitatConfigPlugin)
    repo_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    with initialize_config_dir(
        version_base=None, config_dir=os.path.join(repo_root, "config")
    ):
        habitat_cfg = compose(config_name=f"habitat_eval_{dataset}")
    habitat_cfg = patch_config(habitat_cfg)

    env = habitat.Env(habitat_cfg)

    # Advance the iterator to the recorded position; ordering is deterministic
    # because the dataset iterator is configured non-cycling and non-shuffling
    for _ in range(run_index):
        env.current_episode = next(env.episode_iterator)

    observations = env.reset()
    if str(env.current_episode.episode_id) != str(meta["episode_id"]):
        print(
            f"WARNING: iterator landed on episode {env.current_episode.episode_id}, "
            f"recording says {meta['episode_id']}"
        )

    action_map = {
        0: HabitatSimActions.stop,
        1: HabitatSimActions.move_forward,
        2: HabitatSimActions.turn_left,
        3: HabitatSimActions.turn_right,
        4: HabitatSimActions.look_down,
        5: HabitatSimActions.look_up,
    }

    out_root = str(cfg.output_dir).format(record=os.path.basename(record_dir.rstrip("/")))
    gif_save = bool(cfg.visualization.gif.save)
    mp4_save = bool(cfg.visualization.mp4.save)
    decision_save = bool(cfg.visualization.decision_frame.save)
    if bool(cfg.visualization.decision_frame.only_false_positive):
        decision_save = decision_save and meta.get("result") == "false positive"
    save_rgb = os.path.join(out_root, "rgb") if (cfg.rgb.save or gif_save or mp4_save or decision_save) else None
    save_depth = os.path.join(out_root, "depth") if cfg.depth.save else None
    dump_clouds = os.path.join(out_root, "clouds") if cfg.clouds.save else None
    for path in (save_rgb, save_depth, dump_clouds):
        if path:
            os.makedirs(path, exist_ok=True)
    if any((save_rgb, save_depth, dump_clouds)):
        print(f"Writing artifacts under {out_root}")

    rgb_written = 0
    # The reset observation is frame zero, matching the recorder's on-disk
    # convention and giving the GIF a useful starting frame.
    if save_rgb:
        import cv2

        cv2.imwrite(
            os.path.join(save_rgb, "000000.jpg"),
            np.asarray(observations["rgb"])[:, :, ::-1],
        )
        rgb_written += 1

    sensor_cfg = habitat_cfg.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor
    params = depth_sensor_params(sensor_cfg)
    camera_height = float(sensor_cfg["position"][1])
    step_stride = max(1, int(cfg.step_stride))
    # Pitch is not observable from habitat's observations, so it comes from the
    # recording, which tracked it across look_up / look_down actions
    pitch_by_step = {int(e["step"]): float(e.get("camera_pitch") or 0.0) for e in steps}
    decision_step = None
    if decision_save:
        from basic_utils.record_episode.episode_visualization import find_target_decision_step

        decision_step = find_target_decision_step(steps)

    poses = [agent_world_pose(observations)]
    max_error = 0.0
    depth_written = 0
    clouds_written = 0

    for entry in steps:
        action = entry["action"]
        if action is None:
            print(f"  step {entry['step']}: no action recorded, stopping replay")
            break
        observations = env.step(action_map[action])
        pose = agent_world_pose(observations)
        poses.append(pose)

        error = float(np.max(np.abs(np.asarray(pose) - np.asarray(entry["pose"]))))
        max_error = max(max_error, error)
        if error > float(cfg.tolerance):
            print(
                f"  step {entry['step']} ({ACTION_NAMES.get(action, action)}): "
                f"pose deviates by {error:.5f}"
            )

        # A plain `continue` here would skip the episode_over check below, so
        # gate the writes instead of the loop body
        want_artifacts = entry["step"] % step_stride == 0
        want_decision_frame = decision_step is not None and entry["step"] == decision_step

        if save_rgb and (want_artifacts or want_decision_frame):
            import cv2

            cv2.imwrite(
                os.path.join(save_rgb, f"{entry['step']:06d}.jpg"),
                np.asarray(observations["rgb"])[:, :, ::-1],
            )
            rgb_written += 1

        if (save_depth or dump_clouds) and want_artifacts:
            # One depth read per step, shared by both consumers
            depth = depth_frame(observations, params, metric=bool(cfg.depth.metric))

            if save_depth:
                np.save(os.path.join(save_depth, f"{entry['step']:06d}.npy"), depth)
                depth_written += 1

            if dump_clouds:
                metric_depth = depth
                if not cfg.depth.metric:
                    metric_depth = depth_frame(observations, params, metric=True)
                cloud = depth_to_world_cloud(
                    metric_depth,
                    pose,
                    camera_height,
                    pitch_by_step.get(entry["step"], 0.0),
                    params,
                    pixel_stride=int(cfg.clouds.pixel_stride),
                    max_range_frac=float(cfg.clouds.max_range_frac),
                )
                np.save(os.path.join(dump_clouds, f"{entry['step']:06d}.npy"), cloud)
                clouds_written += 1

        if env.episode_over:
            break

    metrics = env.get_metrics()
    print(
        f"Replay finished: success={metrics['success']} "
        f"spl={metrics['spl']:.4f} distance_to_goal={metrics['distance_to_goal']:.4f}"
    )
    print(f"Recorded:        {meta.get('metrics', {})}")

    if depth_written or clouds_written or rgb_written:
        print(
            f"Wrote {depth_written} depth frames, {clouds_written} clouds, "
            f"{rgb_written} rgb frames"
        )

    if cfg.visualization.gif.save:
        from basic_utils.record_episode.episode_visualization import create_episode_gif

        gif_path = os.path.join(
            out_root, str(cfg.visualization.gif.filename)
        )
        _, gif_frames, decision_step, gate_step = create_episode_gif(
            record_dir,
            save_rgb or os.path.join(out_root, "rgb"),
            gif_path,
            steps=steps,
            fps=int(cfg.visualization.gif.fps),
        )
        print(
            f"Wrote episode GIF: {gif_path} ({gif_frames} frames, "
            f"FSM decision step={decision_step}, gate step={gate_step})"
        )

    if mp4_save:
        from basic_utils.record_episode.episode_visualization import create_episode_mp4

        mp4_path = os.path.join(out_root, str(cfg.visualization.mp4.filename))
        _, mp4_frames, decision_step, gate_step = create_episode_mp4(
            record_dir,
            save_rgb or os.path.join(out_root, "rgb"),
            mp4_path,
            steps=steps,
            fps=float(cfg.visualization.mp4.fps),
        )
        print(
            f"Wrote episode MP4: {mp4_path} ({mp4_frames} frames, "
            f"FSM decision step={decision_step}, gate step={gate_step})"
        )

    if decision_save:
        from basic_utils.record_episode.episode_visualization import create_decision_frame

        decision_output = os.path.join(
            out_root, str(cfg.visualization.decision_frame.filename)
        )
        decision_result = create_decision_frame(
            record_dir, save_rgb or os.path.join(out_root, "rgb"), decision_output, steps=steps
        )
        if decision_result:
            _, saved_step = decision_result
            print(f"Wrote target decision frame: {decision_output} (step {saved_step})")
        else:
            print("No target decision signal recorded; decision frame not written")

    recorded_traj = np.load(os.path.join(record_dir, "trajectory.npy"))
    replayed_traj = np.asarray(poses, dtype=np.float32)
    if recorded_traj.shape == replayed_traj.shape:
        traj_error = float(np.max(np.abs(recorded_traj - replayed_traj)))
        print(f"Max trajectory deviation: {traj_error:.6f}")
    else:
        print(
            f"Trajectory length mismatch: recorded {recorded_traj.shape}, "
            f"replayed {replayed_traj.shape}"
        )

    env.close()
    return max_error


def main():
    parser = argparse.ArgumentParser(
        description="Replay a recorded episode offline. Settings live in "
                    "config/replay.yaml; pass hydra-style key=value overrides.",
        add_help=True,
    )
    parser.add_argument(
        "record_dir",
        help="one episode directory or a category directory such as records/false_positive",
    )
    # Keep unknown args so hydra-style overrides pass through, matching how
    # habitat_evaluation.py handles its own config
    args, overrides = parser.parse_known_args()

    from hydra import initialize_config_dir, compose

    repo_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    with initialize_config_dir(
        version_base=None, config_dir=os.path.join(repo_root, "config")
    ):
        cfg = compose(config_name="replay", overrides=overrides)

    record_dirs = discover_record_dirs(args.record_dir)
    target_episodes = list(cfg.visualization.gif.target_episodes)
    selected = [
        record_dir
        for record_dir in record_dirs
        if record_matches_target(record_dir, target_episodes)
    ]
    if not selected:
        raise SystemExit(
            "No episode records matched visualization.gif.target_episodes="
            f"{target_episodes}"
        )
    if len(selected) > 1:
        print(f"Replaying {len(selected)} episode records")
    for record_dir in selected:
        replay(record_dir, cfg)


if __name__ == "__main__":
    main()
