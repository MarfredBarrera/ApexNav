"""
ROS 2 interface for the Habitat evaluation loop.

Owns every topic the evaluation process publishes or subscribes to, keeping
habitat_evaluation.py to the episode logic itself. The handshake with the C++
planner runs over /habitat/state and /ros/state; see params.py for the enums.
"""

import time
from copy import deepcopy

import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.parameter import parameter_value_to_python
from rclpy.parameter_client import AsyncParameterClient
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Int32, Int32MultiArray, Float32MultiArray, Float64

from plan_env.msg import (
    MultipleMasksWithConfidence,
    ObjectFusionState,
    StepAction,
    StepObservation,
    VerificationRequest,
    VerificationResult,
)

from habitat2ros.habitat_publisher import ROSPublisherNonNode, make_step_messages
from params import ROS_STATE


class HabitatEvalNode(Node):
    def __init__(self):
        super().__init__('habitat_eval_node')

        # QoS profile
        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        # State variables
        self.global_action = None
        self.ros_state = ROS_STATE.INIT
        self.final_state = 0
        self.expl_result = 0
        self.msg_observations = None
        self.fusion_threshold = 0.0
        self.object_fusion = None
        self.pub_timer = None
        # Latest /object/verification_request (perception gate), as a dict.
        # Requests with seq <= verification_seq_floor are from an earlier episode
        self.verification_request = None
        self.last_verification_seq = 0
        self.verification_seq_floor = 0
        # sync_step mode (enable_sync_step): one StepObservation per executed step
        self.sync_step = False
        self.step_action = None

        # Publishers
        self.obj_point_cloud_pub = self.create_publisher(
            PointCloud2, "habitat/object_point_cloud", qos)
        self.state_pub = self.create_publisher(Int32, "/habitat/state", qos)
        self.trigger_pub = self.create_publisher(PoseStamped, "/move_base_simple/goal", qos)
        self.itm_score_pub = self.create_publisher(Float64, "/blip2/cosine_score", qos)
        self.confidence_threshold_pub = self.create_publisher(
            Float64, "/detector/confidence_threshold", qos)
        self.cld_with_score_pub = self.create_publisher(
            MultipleMasksWithConfidence, "/detector/clouds_with_scores", qos)
        self.progress_pub = self.create_publisher(Int32MultiArray, "/habitat/progress", qos)
        self.record_pub = self.create_publisher(Float32MultiArray, "/habitat/record", qos)
        self.verification_result_pub = self.create_publisher(
            VerificationResult, "/object/verification_result", qos)

        # ROS Publisher for habitat topics
        self.ros_pub = ROSPublisherNonNode(self)

        # Subscribers
        self.create_subscription(Int32, "/habitat/plan_action", self.ros_action_callback, qos)
        self.create_subscription(Int32, "/ros/state", self.ros_state_callback, qos)
        self.create_subscription(Int32, "/ros/expl_state", self.ros_final_state_callback, qos)
        self.create_subscription(Int32, "/ros/expl_result", self.ros_expl_result_callback, qos)
        self.create_subscription(
            ObjectFusionState, "/object/fusion_state", self.object_fusion_callback, qos)
        self.create_subscription(
            VerificationRequest, "/object/verification_request",
            self.verification_request_callback, qos)

    def ros_action_callback(self, msg):
        self.global_action = msg.data

    def ros_state_callback(self, msg):
        self.ros_state = msg.data

    def ros_final_state_callback(self, msg):
        self.final_state = msg.data

    def ros_expl_result_callback(self, msg):
        self.expl_result = msg.data

    def object_fusion_callback(self, msg):
        """Latch the planner's latest per-cluster fused confidence snapshot.

        Kept as a plain dict so the recorder can serialize it directly. The
        clusters are flattened per object; update_seq lets a consumer tell
        whether a step saw a stale snapshot.
        """
        self.object_fusion = {
            "update_seq": int(msg.update_seq),
            "min_confidence": float(msg.min_confidence),
            "min_observation_num": int(msg.min_observation_num),
            "clusters": [
                {
                    "id": int(msg.ids[i]),
                    "best_label": int(msg.best_labels[i]),
                    "confidence": float(msg.confidences[i]),
                    "observation_num": int(msg.observation_nums[i]),
                    "centroid": [float(msg.centroids_x[i]), float(msg.centroids_y[i])],
                    "is_confident": bool(msg.is_confident[i]),
                    "verification": (
                        int(msg.verification[i]) if i < len(msg.verification) else 0
                    ),
                }
                for i in range(len(msg.ids))
            ],
        }

    def verification_request_callback(self, msg):
        """Latch the planner's request to verify a target cluster (perception gate).

        The FSM waits in WAIT_VERIFICATION until publish_verification_result
        answers with the same seq. The planner's seq never resets, so a request
        from before the current episode began is recognized and dropped.
        """
        if msg.seq <= self.verification_seq_floor:
            print(f"[verification] dropping stale request seq={msg.seq}")
            return
        self.last_verification_seq = max(self.last_verification_seq, int(msg.seq))
        self.verification_request = {
            "seq": int(msg.seq),
            "cluster_id": int(msg.cluster_id),
            "centroid_x": float(msg.centroid_x),
            "centroid_y": float(msg.centroid_y),
            "confidence": float(msg.confidence),
            "observation_num": int(msg.observation_num),
            "expl_result": int(msg.expl_result),
            "at_stop": bool(msg.at_stop),
            "fusion_seq": int(msg.fusion_seq),
            # an approach re-request (perception gate state 3; 0: ordinary request)
            "approach_id": int(msg.approach_id),
            "viewpoint_status": int(msg.viewpoint_status),
            "viewpoint_x": float(msg.viewpoint_x),
            "viewpoint_y": float(msg.viewpoint_y),
            "approach_steps": int(msg.approach_steps),
        }

    # ------------------------------------------------------------------
    # sync_step: deterministic simulator-planner step
    # ------------------------------------------------------------------

    def enable_sync_step(self):
        """Switch to sync_step: the planner gets one StepObservation per executed step and
        answers with a StepAction tagged with that step (see plan_env/msg)."""
        if self.sync_step:
            return
        self.sync_step = True
        reliable = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST, depth=10)
        self.step_obs_pub = self.create_publisher(
            StepObservation, "/habitat/step_observation", reliable)
        self.action_finish_step_pub = self.create_publisher(
            Int32, "/habitat/action_finish_step", reliable)
        self.create_subscription(
            StepAction, "/habitat/step_action", self.step_action_callback, reliable)

    def step_action_callback(self, msg):
        self.step_action = {
            "action": int(msg.action),
            "step": int(msg.step),
            "fusion_seq": int(msg.fusion_seq),
            "final_state": int(msg.final_state),
            "expl_result": int(msg.expl_result),
        }
        self.global_action = int(msg.action)

    def publish_step_observation(self, step, observations, itm=None, clouds=None, reopen=()):
        """Everything the planner's map needs from one executed step, in one message."""
        msg = StepObservation()
        msg.step = int(step)
        ros_time = self.get_clock().now().to_msg()
        msg.depth, msg.odom, msg.camera_pose = make_step_messages(
            self.ros_pub.bridge, ros_time, observations)
        msg.has_itm = itm is not None
        msg.itm_score = float(itm) if itm is not None else -1.0
        msg.has_clouds = clouds is not None
        if clouds is not None:
            msg.clouds = clouds
        msg.reopen_cluster_ids = [int(c) for c in reopen]
        self.step_obs_pub.publish(msg)

    def wait_for_fusion(self, seq, timeout=60.0):
        """Spin until the planner's fusion snapshot numbered ``seq`` has arrived, so the gate
        and the recorder see the snapshot the planner acted on, not whichever came first."""
        deadline = time.time() + timeout
        while seq > 0 and (self.object_fusion is None or self.object_fusion["update_seq"] < seq):
            if time.time() > deadline:
                raise RuntimeError(f"[sync_step] fusion snapshot {seq} did not arrive in {timeout:.0f} s")
            rclpy.spin_once(self, timeout_sec=0.05)

    def start_verification_episode(self):
        """Forget requests from earlier episodes (call once the planner is ready)."""
        self.verification_request = None
        self.verification_seq_floor = self.last_verification_seq

    def publish_verification_result(self, seq, cluster_id, state, approach=None):
        """Answer a request (state 1 accept, -1 reject, 2 defer, 3 approach ``approach``:
        ``VerificationGate.pending_approach``) or reopen a cluster (state 0)."""
        msg = VerificationResult()
        msg.seq = int(seq)
        msg.cluster_id = int(cluster_id)
        msg.state = int(state)
        if approach is not None:
            msg.approach_id = int(approach["approach_id"])
            msg.object_x = float(approach["object"][0])
            msg.object_y = float(approach["object"][1])
            msg.camera_x = float(approach["camera"][0])
            msg.camera_y = float(approach["camera"][1])
            msg.max_steps = int(approach["max_steps"])
            msg.side_only = bool(approach["side_only"])
        self.verification_result_pub.publish(msg)

    def publish_int32(self, publisher, data):
        msg = Int32()
        msg.data = data
        publisher.publish(msg)

    def publish_float64(self, publisher, data):
        msg = Float64()
        msg.data = float(data)
        publisher.publish(msg)

    def publish_int32_array(self, publisher, data_list):
        msg = Int32MultiArray()
        msg.data = data_list
        publisher.publish(msg)

    def publish_float32_array(self, publisher, data_list):
        msg = Float32MultiArray()
        msg.data = [float(x) for x in data_list]
        publisher.publish(msg)

    def publish_observations(self):
        """Push the latest observations, confidence threshold and FSM trigger.

        In sync_step mode only the pre-episode handshake calls this; the planner integrates
        the repeated step-0 observation once (it drops step ids it has already seen)."""
        if self.msg_observations is None:
            return
        if self.sync_step:
            self.publish_step_observation(0, self.msg_observations)
        else:
            self.ros_pub.habitat_publish_ros_topic(deepcopy(self.msg_observations))
        self.publish_float64(self.confidence_threshold_pub, self.fusion_threshold)
        self.trigger_pub.publish(PoseStamped())

    def publish_observations_callback(self):
        """Timer callback to publish habitat observations and trigger messages"""
        self.publish_observations()

    def start_observation_timer(self):
        """Start timer for publishing observations"""
        self.pub_timer = self.create_timer(0.25, self.publish_observations_callback)

    def stop_observation_timer(self):
        """Stop observation timer"""
        if self.pub_timer is not None:
            self.pub_timer.cancel()
            self.pub_timer = None

    def wait_for_planner_ready(self, timeout=None):
        """Block until the FSM has odometry and has accepted the trigger."""
        self.ros_state = ROS_STATE.INIT
        deadline = None if timeout is None else time.time() + timeout
        while self.ros_state in (ROS_STATE.INIT, ROS_STATE.WAIT_TRIGGER):
            if deadline is not None and time.time() > deadline:
                raise RuntimeError(f"planner not ready after {timeout:.0f} s")
            if self.ros_state == ROS_STATE.INIT:
                print("Waiting for ROS to get odometry...")
            else:
                print("Waiting for ROS trigger...")
            rclpy.spin_once(self, timeout_sec=0.1)

    def wait_for_planner_reset(self, attempts=50):
        """Give the FSM time to re-init after EPISODE_FINISH.

        Without this the next episode can start before the planner has rebuilt
        its maps, which corrupts the first few steps.
        """
        for _ in range(attempts):
            rclpy.spin_once(self, timeout_sec=0.1)
            if self.ros_state == ROS_STATE.INIT:
                return True
        return False


def query_planner_fusion_config(node, timeout=5.0, extra=()):
    """Read the planner's multi-view fusion parameters over the ROS param API.

    The fusion arm is chosen by a launch argument on the C++ side, so the
    evaluation process cannot know it from its own config. Reading it back here
    means every record states which arm produced it instead of relying on the
    operator to keep two configs in sync. Returns None if the planner does not
    answer - recording continues either way.
    """
    names = [
        "object.fusion_type",
        "object.min_observation_num",
        "object.use_observation",
        "verification.enabled",
        "sync_step",
        *extra,
    ]
    try:
        client = AsyncParameterClient(node, "exploration_node")
        if not client.wait_for_services(timeout_sec=timeout):
            print("[recorder] exploration_node parameters unavailable; fusion arm unknown")
            return None

        future = client.get_parameters(names)
        deadline = time.time() + timeout
        while not future.done() and time.time() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
        if not future.done():
            print("[recorder] timed out reading planner parameters; fusion arm unknown")
            return None

        values = [parameter_value_to_python(v) for v in future.result().values]
        cfg = dict(zip(names, values))
        cfg["multiview_fusion"] = bool(
            cfg.get("object.fusion_type") == 1
            and cfg.get("object.use_observation")
            and (cfg.get("object.min_observation_num") or 0) > 1
        )
        print(f"[recorder] planner fusion config: {cfg}")
        return cfg
    except Exception as exc:
        print(f"[recorder] could not read planner parameters: {exc}")
        return None
