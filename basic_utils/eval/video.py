"""
Video frame composition for an evaluation run.

Two styles are supported. "apexnav" draws the planner's own top-down view -
value map, occupancy, frontier and object clusters, TSP tour, planned path -
with the annotated RGB observation inset. "habitat" keeps the stock layout.

The apexnav style needs live planner state, which arrives on ROS topics, so a
PlannerVisListener runs on its own executor thread; the evaluation loop's spin
cadence would otherwise throttle map updates. Creating that listener also has a
side effect the planner depends on: the /grid_map/* publishers early-out when
nobody is subscribed, so subscribing turns that work back on.
"""

import threading

from habitat.utils.visualizations.maps import colorize_draw_agent_and_fit_to_height
from habitat.utils.visualizations.utils import (
    images_to_video,
    observations_to_image,
    overlay_frame,
)
from rclpy.executors import SingleThreadedExecutor

from basic_utils.eval.habitat_utils import agent_world_pose
from basic_utils.visualization.apexnav_frame import (
    FrameStyle,
    render_apexnav_frame,
    update_view_center,
)
from basic_utils.visualization.planner_vis_listener import PlannerVisListener


class EpisodeVideo:
    """Collects frames for one episode at a time and writes them out.

    A no-op when video recording is disabled, so the evaluation loop can call
    into it unconditionally.
    """

    def __init__(self, cfg):
        self.enabled = bool(cfg.need_video)
        self.style = cfg.get("video_style", "apexnav")
        self.fps = int(cfg.get("video_fps", 6))

        vis_cfg = cfg.get("visualization", {}) or {}
        self.frame_style = FrameStyle(vis_cfg)

        self.frames = []
        self.trail = []
        self._view_center = None

        self._listener = None
        self._executor = None
        self._thread = None

        if self.enabled and self.style == "apexnav":
            self._listener = PlannerVisListener(
                map_size=vis_cfg.get("map_size", [80.0, 80.0]),
                resolution=vis_cfg.get("map_resolution", 0.05),
            )
            self._executor = SingleThreadedExecutor()
            self._executor.add_node(self._listener)
            self._thread = threading.Thread(target=self._executor.spin, daemon=True)
            self._thread.start()
            print("ApexNav planner visualization listener started")

    def start_episode(self, observations):
        """Reset per-episode state and seed the trail with the reset pose."""
        self.frames = []
        self.trail.clear()
        self.trail.append(agent_world_pose(observations)[:2])
        self._view_center = None
        if self._listener is not None:
            self._listener.reset()
            self.frame_style.reset_value_scale()

    def track(self, observations):
        """Record the agent's position for this step's trail."""
        self.trail.append(agent_world_pose(observations)[:2])

    def capture(self, observations, info, itm_score=None, step=None, target=None):
        """Append one frame, if video is enabled."""
        if not self.enabled:
            return
        self.frames.append(
            self._build_frame(observations, info, itm_score, step, target)
        )

    def _build_frame(self, observations, info, itm_score, step, target):
        """Compose one video frame in the configured style."""
        if self._listener is None:
            frame = observations_to_image(observations, info)
            info.pop("top_down_map", None)
            return overlay_frame(frame, info)

        gt_map = None
        if self.frame_style.show_gt_map and "top_down_map" in info:
            gt_map = colorize_draw_agent_and_fit_to_height(info["top_down_map"], 320)

        pose = agent_world_pose(observations)
        self._view_center = update_view_center(
            self._view_center, pose[:2], self.frame_style
        )
        snapshot = self._listener.snapshot(
            center=self._view_center, extent=self.frame_style.map_extent_m
        )

        meta = {
            "pose": pose,
            "trail": self.trail,
            "target": target,
            "step": step,
            "itm": itm_score,
            "distance_to_goal": info.get("distance_to_goal"),
        }
        return render_apexnav_frame(
            snapshot,
            rgb=observations["rgb"],
            meta=meta,
            style=self.frame_style,
            gt_map=gt_map,
        )

    def save(self, output_dir, video_name):
        """Write the buffered frames as an mp4 and clear the buffer."""
        if self.enabled and self.frames:
            images_to_video(
                self.frames, output_dir, video_name, fps=self.fps, quality=9
            )
        self.frames = []

    def close(self):
        """Tear down the listener thread."""
        if self._executor is None:
            return
        self._executor.shutdown()
        self._thread.join(timeout=2.0)
        self._listener.destroy_node()
        self._executor = None
