
#include <exploration_manager/exploration_manager.h>
#include <exploration_manager/exploration_fsm.h>
#include <exploration_manager/exploration_data.h>
#include <vis_utils/planning_visualization.h>
#include <plan_env/map_ros.h>
#include <cstdlib>

using namespace std::chrono_literals;

namespace apexnav_planner {
void ExplorationFSM::init(rclcpp::Node::SharedPtr node)
{
  bool first_init = (node_ == nullptr);
  node_ = node;
  fp_.reset(new FSMParam);
  fd_.reset(new FSMData);

  // ROS entities must outlive every episode. Recreating this manager destroyed
  // MapROS subscriptions/timers while the executor could still reference them.
  if (first_init) {
    expl_manager_.reset(new ExplorationManager);
    expl_manager_->initialize(node_);
    visualization_.reset(new PlanningVisualization(node_));
  }
  else {
    exec_timer_->cancel();
    expl_manager_->reset();
    object_fusion_seq_ = 0;
  }
  fp_->vis_scale_ = expl_manager_->sdf_map_->getResolution() * FSMConstants::VIS_SCALE_FACTOR;

  state_ = ROS_STATE::INIT;
  awaited_step_ = 0;

  if (!first_init) {
    exec_timer_->reset();
    return;
  }

  /* Perception gate (off = exact baseline) */
  if (!node_->has_parameter("verification.enabled"))
    node_->declare_parameter("verification.enabled", false);
  if (!node_->has_parameter("verification.timeout_s"))
    node_->declare_parameter("verification.timeout_s", 300.0);
  // Gate state 3 (approach, server selector d1) and the backstop's turn to face the object
  if (!node_->has_parameter("verification.approach"))
    node_->declare_parameter("verification.approach", false);
  verification_enabled_ = node_->get_parameter("verification.enabled").as_bool();
  verification_timeout_s_ = node_->get_parameter("verification.timeout_s").as_double();
  approach_enabled_ = verification_enabled_ && node_->get_parameter("verification.approach").as_bool();
  expl_manager_->object_map2d_->setVerificationEnabled(verification_enabled_);
  RCLCPP_INFO(node_->get_logger(), "[Verification] gate %s (timeout %.0f s)%s",
      verification_enabled_ ? "ON" : "off", verification_timeout_s_,
      approach_enabled_ ? ", approach ON" : "");

  /* sync_step (declared by MapROS): map and FSM callbacks share one MutuallyExclusive
     group, odometry comes with each StepObservation, actions carry their step id */
  auto map_ros = expl_manager_->sdf_map_->mapROS();
  sync_step_ = map_ros->syncStep();
  rclcpp::CallbackGroup::SharedPtr group = sync_step_ ? map_ros->syncCallbackGroup() : nullptr;
  rclcpp::SubscriptionOptions sub_options;
  sub_options.callback_group = group;
  RCLCPP_INFO(node_->get_logger(), "[sync_step] %s", sync_step_ ? "ON" : "off");

  /* ROS2 Timer */
  exec_timer_ = node_->create_wall_timer(
      std::chrono::duration<double>(FSMConstants::EXEC_TIMER_DURATION),
      std::bind(&ExplorationFSM::FSMCallback, this), group);
  frontier_timer_ = node_->create_wall_timer(
      std::chrono::duration<double>(FSMConstants::FRONTIER_TIMER_DURATION),
      std::bind(&ExplorationFSM::frontierCallback, this), group);

  /* ROS2 Subscriber */
  trigger_sub_ = node_->create_subscription<geometry_msgs::msg::PoseStamped>(
      "/move_base_simple/goal", 10,
      std::bind(&ExplorationFSM::triggerCallback, this, std::placeholders::_1), sub_options);
  if (sync_step_) {
    map_ros->setStepHook([this](const nav_msgs::msg::Odometry& odom, int) {
      odometryCallback(std::make_shared<nav_msgs::msg::Odometry>(odom));
    });
    action_finish_step_sub_ = node_->create_subscription<std_msgs::msg::Int32>(
        "/habitat/action_finish_step", rclcpp::QoS(rclcpp::KeepLast(10)).reliable(),
        std::bind(&ExplorationFSM::actionFinishStepCallback, this, std::placeholders::_1),
        sub_options);
  }
  else {
    odom_sub_ = node_->create_subscription<nav_msgs::msg::Odometry>(
        "/odom_world", 10,
        std::bind(&ExplorationFSM::odometryCallback, this, std::placeholders::_1));
  }
  habitat_state_sub_ = node_->create_subscription<std_msgs::msg::Int32>(
      "/habitat/state", 10,
      std::bind(&ExplorationFSM::habitatStateCallback, this, std::placeholders::_1), sub_options);
  confidence_threshold_sub_ = node_->create_subscription<std_msgs::msg::Float64>(
      "/detector/confidence_threshold", 10,
      std::bind(&ExplorationFSM::confidenceThresholdCallback, this, std::placeholders::_1),
      sub_options);
  verify_result_sub_ = node_->create_subscription<plan_env::msg::VerificationResult>(
      "/object/verification_result", 10,
      std::bind(&ExplorationFSM::verificationResultCallback, this, std::placeholders::_1),
      sub_options);

  /* ROS2 Publisher */
  ros_state_pub_ = node_->create_publisher<std_msgs::msg::Int32>("/ros/state", 10);
  expl_state_pub_ = node_->create_publisher<std_msgs::msg::Int32>("/ros/expl_state", 10);
  action_pub_ = node_->create_publisher<std_msgs::msg::Int32>("/habitat/plan_action", 10);
  expl_result_pub_ = node_->create_publisher<std_msgs::msg::Int32>("/ros/expl_result", 10);
  object_fusion_pub_ =
      node_->create_publisher<plan_env::msg::ObjectFusionState>("/object/fusion_state", 10);
  robot_marker_pub_ = node_->create_publisher<visualization_msgs::msg::Marker>("/robot", 10);
  verify_request_pub_ = node_->create_publisher<plan_env::msg::VerificationRequest>(
      "/object/verification_request", 10);
  if (sync_step_)
    step_action_pub_ = node_->create_publisher<plan_env::msg::StepAction>(
        "/habitat/step_action", rclcpp::QoS(rclcpp::KeepLast(10)).reliable());
}

int ExplorationFSM::lastProcessedStep()
{
  return expl_manager_->sdf_map_->mapROS()->lastProcessedStep();
}

// The next action: an Int32 on /habitat/plan_action, or in sync_step mode a StepAction
// tagged with the step it was planned from and the fusion snapshot published before it
void ExplorationFSM::publishAction(int action)
{
  if (!sync_step_) {
    std_msgs::msg::Int32 action_msg;
    action_msg.data = action;
    action_pub_->publish(action_msg);
    return;
  }
  plan_env::msg::StepAction msg;
  msg.action = action;
  msg.step = lastProcessedStep();
  msg.fusion_seq = object_fusion_seq_;
  msg.final_state = fd_->final_result_;
  msg.expl_result = fd_->last_expl_result_;
  step_action_pub_->publish(msg);
}

void ExplorationFSM::actionFinishStepCallback(const std_msgs::msg::Int32::SharedPtr msg)
{
  if (state_ != ROS_STATE::WAIT_ACTION_FINISH)
    return;
  awaited_step_ = msg->data;
  transitState(PLAN_ACTION, "Habitat Finish Action (sync)");
}

// FSM between ROS and Habitat for action planning and execution
void ExplorationFSM::FSMCallback()
{
  exec_timer_->cancel();
  std_msgs::msg::Int32 ros_state_msg;
  ros_state_msg.data = state_;
  ros_state_pub_->publish(ros_state_msg);
  // sync_step: plan only from a map that has integrated the step just finished
  if (sync_step_ && state_ == ROS_STATE::PLAN_ACTION && lastProcessedStep() < awaited_step_) {
    exec_timer_->reset();
    return;
  }
  switch (state_) {
    case ROS_STATE::INIT: {
      // Wait for odometry and target confidence threshold
      if (!fd_->have_odom_ || !fd_->have_confidence_) {
        RCLCPP_WARN_THROTTLE(node_->get_logger(), *node_->get_clock(), 1000,
            "No odom || No target confidence threshold.");
        exec_timer_->reset();
        return;
      }
      // Go to WAIT_TRIGGER when prerequisites are ready
      clearVisMarker();
      transitState(ROS_STATE::WAIT_TRIGGER, "FSM");
      break;
    }

    case ROS_STATE::WAIT_TRIGGER: {
      // Do nothing but wait for trigger
      RCLCPP_WARN_THROTTLE(node_->get_logger(), *node_->get_clock(), 1000, "Wait for trigger.");
      break;
    }

    case ROS_STATE::FINISH: {
      if (!fd_->have_finished_) {
        fd_->have_finished_ = true;
        clearVisMarker();
        publishAction(ACTION::STOP);
      }
      RCLCPP_WARN_THROTTLE(node_->get_logger(), *node_->get_clock(), 1000, "Finish One Episode!!!");
      break;
    }

    case ROS_STATE::PLAN_ACTION: {
      // Initial action sequence: perform orientation calibration turns
      if (fd_->init_action_count_ < 1 + 12 + 1 + 12) {
        if (fd_->init_action_count_ < 1)
          fd_->newest_action_ = ACTION::TURN_DOWN;
        else if (fd_->init_action_count_ < 1 + 12)
          fd_->newest_action_ = ACTION::TURN_LEFT;
        else if (fd_->init_action_count_ < 1 + 12 + 1)
          fd_->newest_action_ = ACTION::TURN_UP;
        else
          fd_->newest_action_ = ACTION::TURN_LEFT;
        RCLCPP_WARN(node_->get_logger(), "Init Mode Process -----> (%d/26)", fd_->init_action_count_);
        fd_->init_action_count_++;
        transitState(ROS_STATE::PUB_ACTION, "FSM");
        updateFrontierAndObject();
      }
      else if (fd_->ap_active_ && expl_manager_->object_map2d_->takeApproachView(fd_->ap_cluster_id_)) {
        // Approach: the gate saw a new view of the cluster (passed with this step); ask now
        updateFrontierAndObject();
        endApproach(4, "new view");
      }
      else if (fd_->stop_face_pending_) {
        // The backstop turned to face the stop target last step: verify that fresh view
        // (nothing else was planned since, so the stop target is unchanged)
        fd_->stop_face_pending_ = false;
        updateFrontierAndObject();
        fd_->final_result_ = FINAL_RESULT::REACH_OBJECT;
        int stop_cluster = -1;
        Vector2d stop_pos;
        if (fd_->ap_active_)
          endApproach(1, "arrived, facing the object");
        else if (needsStopVerification(stop_cluster, stop_pos))
          requestVerification(stop_cluster, stop_pos, true);
        else
          finishPlanning();
      }
      else {
        // Main planning phase: determine robot pose and call action planner
        fd_->start_pt_ = fd_->odom_pos_;
        fd_->start_yaw_(0) = fd_->odom_yaw_;

        auto t1 = node_->get_clock()->now();
        fd_->final_result_ = callActionPlanner();
        double call_action_planner_time = (node_->get_clock()->now() - t1).seconds();
        RCLCPP_INFO_THROTTLE(node_->get_logger(), *node_->get_clock(), 10000,
            "[Calculating Time] Planning process time = %.3f s", call_action_planner_time);

        // An unverified target cluster must pass the perception gate before
        // the agent heads for it (and could stop at it); the action stays held
        int stop_cluster = -1;
        Vector2d stop_pos;
        if (fd_->ap_active_ && checkApproach()) {
          // the approach ended (or turns to face the object): handled
        }
        else if (needsVerification())
          requestVerification(expl_manager_->ed_->target_cluster_id_,
              expl_manager_->ed_->target_object_pos_, false);
        else if (needsStopVerification(stop_cluster, stop_pos)) {
          // Approach mode: first one turn to face the object, so the stop is judged on a
          // view of it (the stop stays the target; the next step verifies)
          int turn = ACTION::MOVE_FORWARD;
          if (approach_enabled_ && !facing(expl_manager_->ed_->target_object_pos_, turn)) {
            RCLCPP_WARN(node_->get_logger(), "[Verification] backstop: turning to face the object first");
            fd_->newest_action_ = turn;
            fd_->stop_face_pending_ = true;
            fd_->final_result_ = FINAL_RESULT::SEARCH_OBJECT;
            finishPlanning();
          }
          else
            requestVerification(stop_cluster, stop_pos, true);
        }
        else
          finishPlanning();
      }
      visualize();
      break;
    }

    case ROS_STATE::PUB_ACTION: {
      publishAction(fd_->newest_action_);
      transitState(ROS_STATE::WAIT_ACTION_FINISH, "FSM");
      break;
    }

    case ROS_STATE::WAIT_ACTION_FINISH: {
      exec_timer_->reset();
      break;
    }

    case ROS_STATE::WAIT_VERIFICATION: {
      // Nothing is published and the simulator does not step until the
      // verdict arrives (verificationResultCallback). Fail open on timeout
      double waited = node_->get_clock()->now().seconds() - fd_->verify_start_;
      if (waited > verification_timeout_s_) {
        // sync_step: timing must never decide a verdict, so no fail-open here
        if (sync_step_) {
          RCLCPP_FATAL(node_->get_logger(),
              "[sync_step] no verdict within %.0f s: aborting instead of failing open",
              verification_timeout_s_);
          std::exit(1);
        }
        finishVerification(1, true);
      }
      break;
    }
  }
  exec_timer_->reset();
}

// Publish the planning verdict and move on: act, or end the episode
void ExplorationFSM::finishPlanning()
{
  std_msgs::msg::Int32 expl_state_msg;
  expl_state_msg.data = fd_->final_result_;
  expl_state_pub_->publish(expl_state_msg);
  if (fd_->final_result_ == FINAL_RESULT::EXPLORE ||
      fd_->final_result_ == FINAL_RESULT::SEARCH_OBJECT)
    transitState(ROS_STATE::PUB_ACTION, "FSM");
  else
    transitState(ROS_STATE::FINISH, "FSM");
}

bool ExplorationFSM::needsVerification()
{
  int id = expl_manager_->ed_->target_cluster_id_;
  return verification_enabled_ && fd_->final_result_ == FINAL_RESULT::SEARCH_OBJECT && id >= 0 &&
         expl_manager_->object_map2d_->getVerification(id) == 0;
}

// Backstop at the stop itself: every stop path (incl. over-depth targets, which
// have no cluster) must end at something the gate accepted. Returns false when
// the stop is covered; otherwise sets cluster_id (the nearest non-rejected
// cluster to the target object, or -1 for a bare position) and pos
bool ExplorationFSM::needsStopVerification(int& cluster_id, Vector2d& pos)
{
  if (!verification_enabled_ || fd_->final_result_ != FINAL_RESULT::REACH_OBJECT)
    return false;
  auto ed = expl_manager_->ed_;
  auto obj_map = expl_manager_->object_map2d_;
  if (ed->target_cluster_id_ >= 0 && obj_map->getVerification(ed->target_cluster_id_) == 1)
    return false;  // the usual path: the target cluster was verified before the approach
  // A deferred cluster (2) is verified again here, from the stop
  pos = ed->target_object_pos_;
  cluster_id = obj_map->nearestCluster(pos, FSMConstants::STOP_MATCH_DISTANCE);
  if (cluster_id >= 0)
    return obj_map->getVerification(cluster_id) != 1;
  return !obj_map->isAcceptedPosition(pos, FSMConstants::STOP_MATCH_DISTANCE);
}

// approach_id 0: an ordinary request; else the outcome of that approach (viewpoint_status
// 1 arrived and facing the object, 2 no object path / target lost, 3 aborted (budget),
// 4 a new view on the way)
void ExplorationFSM::requestVerification(
    int cluster_id, const Vector2d& pos, bool at_stop, int approach_id, int viewpoint_status)
{
  plan_env::msg::VerificationRequest msg;
  msg.seq = ++verification_seq_;
  msg.cluster_id = cluster_id;
  msg.expl_result = fd_->last_expl_result_;
  msg.at_stop = at_stop;
  msg.fusion_seq = object_fusion_seq_;
  if (approach_id > 0) {
    msg.approach_id = approach_id;
    msg.viewpoint_status = viewpoint_status;
    msg.viewpoint_x = static_cast<float>(fd_->odom_pos_(0));  // where the agent is
    msg.viewpoint_y = static_cast<float>(fd_->odom_pos_(1));
    msg.approach_steps = fd_->ap_steps_;
  }
  msg.centroid_x = static_cast<float>(pos(0));
  msg.centroid_y = static_cast<float>(pos(1));
  vector<ObjectConfidenceInfo> infos;
  double min_confidence = 0.0;
  int min_observation_num = 0;
  expl_manager_->object_map2d_->getObjectConfidences(infos, min_confidence, min_observation_num);
  for (const auto& info : infos) {
    if (info.id != cluster_id)
      continue;
    // At the stop, keep the stop position: Python matches evidence to where the
    // agent is about to stop, not to the (possibly large) cluster's average
    if (!at_stop) {
      msg.centroid_x = static_cast<float>(info.average(0));
      msg.centroid_y = static_cast<float>(info.average(1));
    }
    msg.confidence = static_cast<float>(info.confidence);
    msg.observation_num = info.observation_num;
  }
  fd_->verify_seq_ = msg.seq;
  fd_->verify_start_ = node_->get_clock()->now().seconds();
  fd_->verify_cluster_id_ = cluster_id;
  fd_->verify_pos_ = pos;
  fd_->verify_at_stop_ = at_stop;
  fd_->verify_approach_id_ = approach_id;
  fd_->verify_arrival_ = approach_id > 0 && viewpoint_status == 1;
  RCLCPP_WARN(node_->get_logger(),
      "[Verification] request seq=%u cluster=%d (%.2f, %.2f) conf=%.3f%s", msg.seq, cluster_id,
      msg.centroid_x, msg.centroid_y, msg.confidence, at_stop ? " [at stop]" : "");
  verify_request_pub_->publish(msg);
  transitState(ROS_STATE::WAIT_VERIFICATION, "FSM");
}

// Apply a verdict on the pending request. Accepted (or fail-open): carry out the
// action already planned (approach, or the stop). Deferred: carry out the approach;
// the stop backstop asks again (at a stop, deferred counts as rejected). Rejected: drop the cluster (or
// cut the position out of over-depth targets) and replan from the same pose as
// if it had never been detected. Approach re-requests: accepted on the way -> replan (the
// cluster stays the forced target, so the agent goes on to it and stops); accepted on
// arrival -> stop; rejected -> drop the forced target and replan
void ExplorationFSM::finishVerification(int state, bool timed_out)
{
  int id = fd_->verify_cluster_id_;
  auto obj_map = expl_manager_->object_map2d_;
  bool approach = fd_->verify_approach_id_ > 0, arrival = fd_->verify_arrival_;
  fd_->verify_approach_id_ = 0;
  fd_->verify_arrival_ = false;
  if (approach || (id >= 0 && expl_manager_->forced_cluster_id_ == id)) {
    fd_->ap_active_ = false;
    if (state != 1) {
      expl_manager_->forced_cluster_id_ = -1;
      fd_->stop_face_pending_ = false;  // a rejected approach is not verified at the stop
    }
  }
  if (state == 2 && (id < 0 || fd_->verify_at_stop_))
    state = -1;
  if (id >= 0)
    obj_map->setVerification(id, state);
  else if (state == 1)
    obj_map->acceptPosition(fd_->verify_pos_);
  // A rejected stop must not be re-targeted through the over-depth cloud either
  if (state != 1 && (id < 0 || fd_->verify_at_stop_))
    obj_map->suppressPosition(fd_->verify_pos_);
  if (timed_out)
    RCLCPP_ERROR(node_->get_logger(),
        "[Verification] no verdict for cluster %d within %.0f s: fail open (accepted)", id,
        verification_timeout_s_);
  else
    RCLCPP_WARN(node_->get_logger(), "[Verification] cluster %d %s%s", id,
        state == 1 ? "ACCEPTED" : state == 2 ? "DEFERRED" : "REJECTED",
        fd_->verify_at_stop_ ? " [at stop]" : "");
  publishObjectFusionState();
  if ((state == 1 || state == 2) && (!approach || arrival)) {
    finishPlanning();
    return;
  }
  if (state == 1) {  // accepted on the way: plan this step anew, toward the object
    fd_->skip_stuck_check_once_ = true;
    transitState(ROS_STATE::PLAN_ACTION, "Verification");
    return;
  }
  // Without this the reach check in callActionPlanner could stop at the
  // rejected cluster's next_pos_
  fd_->final_result_ = FINAL_RESULT::EXPLORE;
  fd_->skip_stuck_check_once_ = true;
  transitState(ROS_STATE::PLAN_ACTION, "Verification");
}

void ExplorationFSM::verificationResultCallback(
    const plan_env::msg::VerificationResult::SharedPtr msg)
{
  auto obj_map = expl_manager_->object_map2d_;
  // state 0 reopens a rejected cluster (Python's re-check cooldown passed); for a cluster
  // under approach (3) it says a new view of it arrived
  if (msg->state == 0) {
    int verification = obj_map->getVerification(msg->cluster_id);
    if (verification == -1) {
      obj_map->setVerification(msg->cluster_id, 0);
      RCLCPP_WARN(node_->get_logger(), "[Verification] cluster %d reopened", msg->cluster_id);
      publishObjectFusionState();
    }
    else if (verification == 3) {
      obj_map->markApproachView(msg->cluster_id);
      RCLCPP_WARN(node_->get_logger(), "[Verification] cluster %d: new view during the approach",
          msg->cluster_id);
    }
    return;
  }
  if (state_ != ROS_STATE::WAIT_VERIFICATION || msg->seq != fd_->verify_seq_ ||
      msg->cluster_id != fd_->verify_cluster_id_) {
    RCLCPP_WARN(node_->get_logger(), "[Verification] ignoring stale result seq=%u cluster=%d", msg->seq,
        msg->cluster_id);
    return;
  }
  // state 3: insufficient view; approach the cluster (or keep approaching it)
  if (msg->state == 3) {
    if (!approach_enabled_ || msg->cluster_id < 0 || fd_->verify_at_stop_) {
      RCLCPP_ERROR(node_->get_logger(), "[Verification] protocol error: state 3 for cluster %d%s (approach %s)",
          msg->cluster_id, fd_->verify_at_stop_ ? " at the stop" : "", approach_enabled_ ? "on" : "off");
      if (sync_step_)
        std::exit(1);
      finishVerification(-1, false);
      return;
    }
    startApproach(*msg);
    return;
  }
  finishVerification(msg->state == 1 || msg->state == 2 ? msg->state : -1, false);
}

// True when the robot faces ``point`` (within ACTION_ANGLE / 1.9); else ``turn`` is the
// turn toward it
bool ExplorationFSM::facing(const Vector2d& point, int& turn)
{
  Vector2d pos(fd_->odom_pos_(0), fd_->odom_pos_(1));
  turn = ACTION::MOVE_FORWARD;
  if ((point - pos).norm() < 1e-3)
    return true;
  turn = decideNextAction(fd_->odom_yaw_, std::atan2(point(1) - pos(1), point(0) - pos(0)));
  return turn == ACTION::MOVE_FORWARD;
}

// Gate state 3: the cluster becomes the planner's forced target and ApexNav's own object
// navigation (searchObjectPath, per-step replanning) drives toward it. After an ordinary
// request the held action already heads for it (it was planned for this cluster), so it is
// carried out as the approach's first action; after a re-request (a new view that was still
// insufficient) the approach goes on and this step is planned anew
void ExplorationFSM::startApproach(const plan_env::msg::VerificationResult& msg)
{
  bool resume = fd_->verify_approach_id_ > 0;
  fd_->verify_approach_id_ = 0;
  fd_->verify_arrival_ = false;
  if (!resume || msg.approach_id != fd_->ap_approach_id_ || msg.cluster_id != fd_->ap_cluster_id_)
    fd_->ap_steps_ = 0;
  fd_->ap_active_ = true;
  fd_->ap_cluster_id_ = msg.cluster_id;
  fd_->ap_approach_id_ = msg.approach_id;
  fd_->ap_max_steps_ = msg.max_steps;
  fd_->ap_object_ = Vector2d(msg.object_x, msg.object_y);
  expl_manager_->forced_cluster_id_ = msg.cluster_id;
  auto obj_map = expl_manager_->object_map2d_;
  obj_map->setVerification(msg.cluster_id, 3);  // under approach: one cluster at a time
  obj_map->takeApproachView(msg.cluster_id);    // drop any stale mark
  publishObjectFusionState();
  Vector2d pos(fd_->odom_pos_(0), fd_->odom_pos_(1));
  RCLCPP_WARN(node_->get_logger(),
      "[Verification] cluster %d approach %d %s: object (%.2f, %.2f) %.2f m away, evidence camera "
      "%.2f m from it; %d/%d actions used",
      msg.cluster_id, msg.approach_id, resume ? "goes on" : "starts", msg.object_x, msg.object_y,
      (fd_->ap_object_ - pos).norm(), (Vector2d(msg.camera_x, msg.camera_y) - fd_->ap_object_).norm(),
      fd_->ap_steps_, msg.max_steps);
  if (!resume && fd_->final_result_ == FINAL_RESULT::SEARCH_OBJECT &&
      expl_manager_->ed_->target_cluster_id_ == msg.cluster_id && msg.max_steps > 0) {
    fd_->ap_steps_ = 1;
    finishPlanning();  // the held action, toward this cluster
    return;
  }
  fd_->skip_stuck_check_once_ = true;  // no step was taken since the last plan
  transitState(ROS_STATE::PLAN_ACTION, "Approach");
}

// After callActionPlanner during an approach: true when the approach ended here (arrival,
// no object path, budget) or the agent turns to face the object first; false to carry out
// the planned action as one more approach action
bool ExplorationFSM::checkApproach()
{
  auto ed = expl_manager_->ed_;
  if (fd_->final_result_ == FINAL_RESULT::REACH_OBJECT) {
    int turn = ACTION::MOVE_FORWARD;
    if (!facing(ed->target_object_pos_, turn)) {  // one turn, then verify (stop_face_pending_)
      RCLCPP_WARN(node_->get_logger(), "[Verification] approach: reached; turning to face the object");
      fd_->newest_action_ = turn;
      fd_->stop_face_pending_ = true;
      fd_->final_result_ = FINAL_RESULT::SEARCH_OBJECT;
      fd_->ap_steps_++;
      finishPlanning();
      return true;
    }
    endApproach(1, "arrived, facing the object");
    return true;
  }
  if (fd_->final_result_ == FINAL_RESULT::STUCKING)
    return false;  // the episode ends as without the gate (give-up stops are not verified)
  if (expl_manager_->forced_failed_ || ed->target_cluster_id_ != fd_->ap_cluster_id_ ||
      fd_->final_result_ != FINAL_RESULT::SEARCH_OBJECT) {
    endApproach(2, expl_manager_->forced_failed_ ? "no object path" : "target lost");
    return true;
  }
  if (fd_->ap_steps_ >= fd_->ap_max_steps_) {
    endApproach(3, "step budget used up");
    return true;
  }
  fd_->ap_steps_++;
  return false;
}

// The approach is over for now: the gate decides (arrival: on the fresh view at the stop)
void ExplorationFSM::endApproach(int status, const char* reason)
{
  fd_->ap_active_ = false;
  Vector2d pos(fd_->odom_pos_(0), fd_->odom_pos_(1));
  RCLCPP_WARN(node_->get_logger(),
      "[Verification] cluster %d approach %d ends (status %d: %s) at (%.2f, %.2f), %.2f m from the "
      "object, after %d/%d actions",
      fd_->ap_cluster_id_, fd_->ap_approach_id_, status, reason, pos(0), pos(1),
      (fd_->ap_object_ - pos).norm(), fd_->ap_steps_, fd_->ap_max_steps_);
  bool arrival = status == 1;
  requestVerification(fd_->ap_cluster_id_, arrival ? expl_manager_->ed_->target_object_pos_ : fd_->ap_object_,
      arrival, fd_->ap_approach_id_, status);
}

/**
 * @brief Plan the next action based on current state and environment
 * @return Final result indicating the planned action type and exploration state
 *
 * This is the core planning function that decides what action the robot should take next.
 * It handles obstacle avoidance, frontier exploration, object search, and stuck recovery.
 */
int ExplorationFSM::callActionPlanner()
{
  const double stucking_distance = FSMConstants::STUCKING_DISTANCE;
  const double reach_distance = FSMConstants::REACH_DISTANCE;
  const double soft_reach_distance = FSMConstants::SOFT_REACH_DISTANCE;

  bool frontier_change_flag = updateFrontierAndObject();

  int expl_res, final_res;
  Eigen::Vector2d current_pos = Eigen::Vector2d(fd_->start_pt_(0), fd_->start_pt_(1));
  Eigen::Vector2d last_pos = Eigen::Vector2d(fd_->last_start_pos_(0), fd_->last_start_pos_(1));
  double current_yaw = fd_->start_yaw_(0);
  fd_->last_start_pos_ = fd_->start_pt_;
  // Replanning after a rejected target: the pose did not change, which the
  // stuck detectors below must not count as a failed step
  bool after_rejection = fd_->skip_stuck_check_once_;
  fd_->skip_stuck_check_once_ = false;

  // Reach the object - check if close enough to target object
  if (fd_->final_result_ == FINAL_RESULT::SEARCH_OBJECT &&
      (current_pos - expl_manager_->ed_->next_pos_).norm() < reach_distance) {
    RCLCPP_ERROR(node_->get_logger(), "Reach the object successfully!!!");
    final_res = FINAL_RESULT::REACH_OBJECT;
    return final_res;
  }

  /*******  Escape-from-stuck logic START *******/
  // Detect if robot is stuck and initiate escape sequence
  int last_action = fd_->newest_action_;
  if (!after_rejection && !fd_->escape_stucking_flag_ &&
      (current_pos - last_pos).norm() < stucking_distance && last_action == ACTION::MOVE_FORWARD) {
    if (fd_->final_result_ == FINAL_RESULT::SEARCH_OBJECT &&
        (current_pos - expl_manager_->ed_->next_pos_).norm() < soft_reach_distance) {
      RCLCPP_ERROR(node_->get_logger(), "Reach the object successfully!!!");
      final_res = FINAL_RESULT::REACH_OBJECT;
      return final_res;
    }

    bool past_stucking_flag = false;
    for (auto stucking_point : fd_->stucking_points_) {
      Vector2d stucking_pos = Vector2d(stucking_point(0), stucking_point(1));
      double stucking_yaw = stucking_point(2);
      if ((stucking_pos - current_pos).norm() < stucking_distance &&
          fabs(stucking_yaw - current_yaw) < FSMConstants::ACTION_ANGLE) {
        past_stucking_flag = true;
        RCLCPP_ERROR(node_->get_logger(), "Still stuck at the same place");
        break;
      }
    }
    if (!past_stucking_flag) {
      fd_->escape_stucking_flag_ = true;
      fd_->escape_stucking_count_ = 0;
      fd_->escape_stucking_pos_ = current_pos;
      fd_->escape_stucking_yaw_ = current_yaw;
    }
  }

  if (fd_->escape_stucking_flag_ && (current_pos - last_pos).norm() >= stucking_distance) {
    RCLCPP_ERROR(node_->get_logger(), "Escaped from stuck state.");
    fd_->escape_stucking_flag_ = false;
  }

  if (fd_->escape_stucking_flag_) {
    RCLCPP_ERROR(node_->get_logger(), "Escaping stuck...");
    if (fd_->escape_stucking_count_ == 0)
      fd_->newest_action_ = ACTION::TURN_RIGHT;
    else if (fd_->escape_stucking_count_ == 1)
      fd_->newest_action_ = ACTION::MOVE_FORWARD;
    else if (fd_->escape_stucking_count_ == 2)
      fd_->newest_action_ = ACTION::TURN_RIGHT;
    else if (fd_->escape_stucking_count_ == 3)
      fd_->newest_action_ = ACTION::MOVE_FORWARD;
    else if (fd_->escape_stucking_count_ == 4)
      fd_->newest_action_ = ACTION::TURN_LEFT;
    else if (fd_->escape_stucking_count_ == 5)
      fd_->newest_action_ = ACTION::TURN_LEFT;
    else if (fd_->escape_stucking_count_ == 6)
      fd_->newest_action_ = ACTION::TURN_LEFT;
    else if (fd_->escape_stucking_count_ == 7)
      fd_->newest_action_ = ACTION::MOVE_FORWARD;
    else if (fd_->escape_stucking_count_ == 8)
      fd_->newest_action_ = ACTION::TURN_LEFT;
    else if (fd_->escape_stucking_count_ == 9)
      fd_->newest_action_ = ACTION::MOVE_FORWARD;
    else {
      // Failed to escape - mark area as occupied and add to stuck points
      RCLCPP_ERROR(node_->get_logger(), "Cannot escape stuck state.");
      fd_->escape_stucking_flag_ = false;
      expl_manager_->sdf_map_->setForceOccGrid(current_pos);
      double forward_distance = FSMConstants::FORWARD_DISTANCE;
      Eigen::Vector2d forward_pos = fd_->escape_stucking_pos_;
      forward_pos(0) += forward_distance * cos(fd_->escape_stucking_yaw_);
      forward_pos(1) += forward_distance * sin(fd_->escape_stucking_yaw_);
      expl_manager_->sdf_map_->setForceOccGrid(forward_pos);
      forward_distance = FSMConstants::FORWARD_DISTANCE * 2.0;
      forward_pos = fd_->escape_stucking_pos_;
      forward_pos(0) += forward_distance * cos(fd_->escape_stucking_yaw_);
      forward_pos(1) += forward_distance * sin(fd_->escape_stucking_yaw_);
      expl_manager_->sdf_map_->setForceOccGrid(forward_pos);
      fd_->dormant_frontier_flag_ = true;
      Vector3d stucking_point(
          fd_->escape_stucking_pos_(0), fd_->escape_stucking_pos_(1), fd_->escape_stucking_yaw_);
      fd_->stucking_points_.push_back(stucking_point);
    }

    if (fd_->escape_stucking_flag_) {
      fd_->escape_stucking_count_++;
      return fd_->final_result_;
    }
  }

  /*******  Decide whether to replan path (stability heuristic) START *******/
  // Use path stability to reduce oscillation between different frontier targets
  vector<Vector2d> last_next_best_path = expl_manager_->ed_->next_best_path_;
  Vector2d last_next_pos = expl_manager_->ed_->next_pos_;
  if (fd_->dormant_frontier_flag_) {
    fd_->replan_flag_ = true;
    fd_->dormant_frontier_flag_ = false;
  }
  else if (fd_->final_result_ == FINAL_RESULT::EXPLORE && !frontier_change_flag && !after_rejection)
    fd_->replan_flag_ = false;  // (after a rejection the kept path would lead to the rejected cluster)

  expl_res = expl_manager_->planNextBestPoint(fd_->start_pt_, fd_->start_yaw_(0));
  fd_->last_expl_result_ = expl_res;

  if (expl_res != EXPL_RESULT::EXPLORATION) {
    fd_->replan_flag_ = true;
  }
  if (expl_res == EXPL_RESULT::EXPLORATION && !fd_->replan_flag_) {
    expl_manager_->ed_->next_best_path_ = last_next_best_path;
    expl_manager_->ed_->next_pos_ = last_next_pos;
    fd_->replan_flag_ = true;
  }
  /*******  Decide whether to replan path (stability heuristic) END *******/

  // Publish exploration result to monitor
  std_msgs::msg::Int32 expl_result_msg;
  expl_result_msg.data = expl_res;
  expl_result_pub_->publish(expl_result_msg);

  // Determine current high-level state based on exploration results
  if (expl_res == EXPL_RESULT::EXPLORATION)
    final_res = FINAL_RESULT::EXPLORE;
  else if (expl_res == EXPL_RESULT::NO_COVERABLE_FRONTIER ||
           expl_res == EXPL_RESULT::NO_PASSABLE_FRONTIER)
    final_res = FINAL_RESULT::NO_FRONTIER;
  else
    final_res = FINAL_RESULT::SEARCH_OBJECT;

  if (final_res == FINAL_RESULT::NO_FRONTIER || expl_manager_->ed_->next_best_path_.empty()) {
    RCLCPP_WARN(node_->get_logger(), "No (passable) frontier");
    return final_res;
  }

  Eigen::Vector2d end_pos = expl_manager_->ed_->next_pos_;
  Eigen::Vector2d last_end_pos = fd_->last_next_pos_;
  fd_->last_next_pos_ = end_pos;
  double min_dist = (current_pos - end_pos).norm();
  RCLCPP_WARN(node_->get_logger(), "To the next point (%.2fm %.2fm), distance = %.2f m",
      end_pos(0), end_pos(1), min_dist);

  // Handling being stuck while exploring toward a specific frontier
  if (final_res == FINAL_RESULT::EXPLORE) {
    // Force dormant if very close to target but still exploring
    if (min_dist < FSMConstants::FORCE_DORMANT_DISTANCE) {
      RCLCPP_ERROR(node_->get_logger(), "Force set dormant frontier.");
      expl_manager_->frontier_map2d_->setForceDormantFrontier(end_pos);
      fd_->dormant_frontier_flag_ = true;
    }

    // Count consecutive times with same target position while stuck
    if (after_rejection) {
      // replan after a rejected target: no step was taken
    }
    else if ((end_pos - last_end_pos).norm() < 1e-3 &&
        (current_pos - last_pos).norm() < stucking_distance) {
      fd_->stucking_next_pos_count_++;
      if (fd_->stucking_next_pos_count_ > 8) {
        RCLCPP_ERROR(node_->get_logger(), "stucking_next_pos_count_ = %d",
            fd_->stucking_next_pos_count_);
      }
    }
    else
      fd_->stucking_next_pos_count_ = 0;

    // Mark frontier as dormant if stuck too long with same target
    if (fd_->stucking_next_pos_count_ >= FSMConstants::MAX_STUCKING_NEXT_POS_COUNT) {
      RCLCPP_ERROR(node_->get_logger(), "Set dormant frontier.");
      fd_->stucking_action_count_ = 0;
      fd_->stucking_next_pos_count_ = 0;
      expl_manager_->frontier_map2d_->setForceDormantFrontier(end_pos);
      fd_->dormant_frontier_flag_ = true;
    }
  }

  // Track consecutive stuck actions globally
  if (after_rejection) {
    // replan after a rejected target: no step was taken
  }
  else if ((current_pos - last_pos).norm() < stucking_distance) {
    fd_->stucking_action_count_++;
    if (fd_->stucking_action_count_ > 15) {
      RCLCPP_ERROR(node_->get_logger(), "Stucking action count = %d",
          fd_->stucking_action_count_);
    }
  }
  else
    fd_->stucking_action_count_ = 0;

  // If stuck for too long globally, terminate episode
  if (fd_->stucking_action_count_ >= FSMConstants::MAX_STUCKING_COUNT) {
    RCLCPP_ERROR(node_->get_logger(), "Stuck for too long, stopping episode.");
    final_res = FINAL_RESULT::STUCKING;
    return final_res;
  }

  // Plan specific action based on exploration result
  if (expl_res == EXPL_RESULT::SEARCH_EXTREME)
    fd_->newest_action_ =
        planNextBestAction(current_pos, current_yaw, expl_manager_->ed_->next_best_path_, false);
  else
    fd_->newest_action_ =
        planNextBestAction(current_pos, current_yaw, expl_manager_->ed_->next_best_path_);

  return final_res;
}

int ExplorationFSM::planNextBestAction(
    Vector2d current_pos, double current_yaw, const vector<Vector2d>& path, bool need_safety)
{
  const double local_distance = FSMConstants::LOCAL_DISTANCE;

  // Update target position based on path and local distance
  Vector2d local_pos = selectLocalTarget(current_pos, path, local_distance);
  fd_->local_pos_ = local_pos;

  // Compute the best step considering obstacles and safety
  Vector2d best_step;
  if ((current_pos - path.back()).norm() > FSMConstants::ACTION_DISTANCE && need_safety)
    best_step = computeBestStep(current_pos, current_yaw, local_pos);
  else
    best_step = local_pos;

  // Calculate target orientation from best step direction
  double target_yaw = std::atan2(best_step(1) - current_pos(1), best_step(0) - current_pos(0));
  return decideNextAction(current_yaw, target_yaw);
}

Vector2d ExplorationFSM::selectLocalTarget(
    const Vector2d& current_pos, const vector<Vector2d>& path, const double& local_distance)
{
  Vector2d target_pos = path.back();

  // Find the closest path point to current position as starting search index
  int start_path_id = 0;
  double min_dist = std::numeric_limits<double>::max();
  for (int i = 0; i < (int)path.size() - 1; i++) {
    Eigen::Vector2d pos = path[i];
    if ((pos - current_pos).norm() < min_dist) {
      min_dist = (pos - current_pos).norm();
      start_path_id = i + 1;
    }
  }

  // Select a local target position within the specified distance
  double len = (path[start_path_id] - current_pos).norm();
  for (int i = start_path_id + 1; i < (int)path.size(); i++) {
    len += (path[i] - path[i - 1]).norm();
    if (len > local_distance && (current_pos - path[i - 1]).norm() > 0.30) {
      target_pos = path[i - 1];
      break;
    }
  }

  return target_pos;
}

Vector2d ExplorationFSM::computeBestStep(
    const Vector2d& current_pos, double current_yaw, const Vector2d& target_pos)
{
  Vector2d best_step = target_pos;

  double min_cost = std::numeric_limits<double>::max();
  for (auto step : fp_->action_steps_) {
    double cost = computeActionTotalCost(current_pos, current_yaw, target_pos, step);
    if (cost < min_cost) {
      best_step = current_pos + step;
      min_cost = cost;
    }
  }

  return best_step;
}

// Compute total cost of taking a step towards target
// Considers distance-to-target, movement efficiency, and collision safety
double ExplorationFSM::computeActionTotalCost(const Vector2d& current_pos, double current_yaw,
    const Vector2d& target_pos, const Vector2d& step)
{
  const double traget_weight = FSMConstants::TARGET_WEIGHT;
  const double traget_close_weight1 = FSMConstants::TARGET_CLOSE_WEIGHT_1;
  const double traget_close_weight2 = FSMConstants::TARGET_CLOSE_WEIGHT_2;
  const double safety_weight = FSMConstants::SAFETY_WEIGHT;
  double cost = 0.0;

  // Distance-to-target cost
  Vector2d step_pos = current_pos + step;
  double target_cost = traget_weight * (step_pos - target_pos).norm();

  // Change-in-distance cost (negative if moving closer)
  double target_close_cost = (step_pos - target_pos).norm() - (current_pos - target_pos).norm();
  if (target_close_cost > 0)
    target_close_cost *= traget_close_weight1;
  else
    target_close_cost *= traget_close_weight2;

  // Safety distance cost
  double safety_cost = safety_weight * computeActionSafetyCost(current_pos, step);

  cost += target_cost + target_close_cost + safety_cost;
  return cost;
}

// Compute safety cost along the step using SDF distance to obstacles
// Returns higher cost for paths that go too close to obstacles
double ExplorationFSM::computeActionSafetyCost(const Vector2d& current_pos, const Vector2d& step)
{
  const double min_safe_distance = FSMConstants::MIN_SAFE_DISTANCE;
  const double sample_num = FSMConstants::SAMPLE_NUM;

  Vector2d dir = step;
  double len = dir.norm();
  dir.normalize();

  double safety_cost = 0.0;
  for (double l = len / sample_num; l < len; l += len / sample_num) {
    Vector2d ckpt = current_pos + l * dir;
    Vector2d grad;
    double dist_to_occ = expl_manager_->sdf_map_->getDistWithGrad(ckpt, grad);
    if (dist_to_occ < min_safe_distance)
      safety_cost += 1 / (dist_to_occ + 1e-2);
  }

  return safety_cost;
}

// Decide whether to turn or move forward based on yaw difference
// Uses action angle threshold to determine if orientation adjustment is needed
int ExplorationFSM::decideNextAction(double current_yaw, double target_yaw)
{
  wrapAngle(target_yaw);
  wrapAngle(current_yaw);
  double yaw_diff = target_yaw - current_yaw;
  wrapAngle(yaw_diff);

  int next_action;
  if (std::fabs(yaw_diff) > FSMConstants::ACTION_ANGLE / 1.9) {
    if (yaw_diff > 0)
      next_action = ACTION::TURN_LEFT;
    else
      next_action = ACTION::TURN_RIGHT;
  }
  else
    next_action = ACTION::MOVE_FORWARD;

  return next_action;
}

void ExplorationFSM::visualize()
{
  auto ed_ptr = expl_manager_->ed_;

  // Lambda function to convert 2D vectors to 3D for visualization
  auto vec2dTo3d = [](const vector<Eigen::Vector2d>& vec2d, double z = 0.15) {
    vector<Eigen::Vector3d> vec3d;
    for (auto v : vec2d) vec3d.push_back(Vector3d(v(0), v(1), z));
    return vec3d;
  };

  // Draw frontier
  static int last_ftr2d_num = 0;
  for (int i = 0; i < (int)ed_ptr->frontiers_.size(); ++i) {
    visualization_->drawCubes(vec2dTo3d(ed_ptr->frontiers_[i]), fp_->vis_scale_,
        visualization_->getColor(double(i) / ed_ptr->frontiers_.size(), 1.0), "frontier", i, 4);
  }
  for (int i = ed_ptr->frontiers_.size(); i < last_ftr2d_num; ++i) {
    visualization_->drawCubes({}, fp_->vis_scale_, Vector4d(0, 0, 0, 1), "frontier", i, 4);
  }
  last_ftr2d_num = ed_ptr->frontiers_.size();

  // Draw dormant frontier
  static int last_dftr2d_num = 0;
  for (int i = 0; i < (int)ed_ptr->dormant_frontiers_.size(); ++i) {
    visualization_->drawCubes(vec2dTo3d(ed_ptr->dormant_frontiers_[i]), fp_->vis_scale_,
        Vector4d(0, 0, 0, 1), "dormant_frontier", i, 4);
  }
  for (int i = ed_ptr->dormant_frontiers_.size(); i < last_dftr2d_num; ++i) {
    visualization_->drawCubes({}, fp_->vis_scale_, Vector4d(0, 0, 0, 1), "dormant_frontier", i, 4);
  }
  last_dftr2d_num = ed_ptr->dormant_frontiers_.size();

  // Draw object
  static int last_obj_num = 0;
  for (int i = 0; i < (int)ed_ptr->objects_.size(); ++i) {
    int label = ed_ptr->object_labels_[i];
    visualization_->drawCubes(vec2dTo3d(ed_ptr->objects_[i]), fp_->vis_scale_,
        visualization_->getColor(double(label) / 5.0, 1.0), "object", i, 4);
  }
  for (int i = ed_ptr->objects_.size(); i < last_obj_num; ++i) {
    visualization_->drawCubes({}, fp_->vis_scale_, Vector4d(0, 0, 0, 1), "object", i, 4);
  }
  last_obj_num = ed_ptr->objects_.size();

  // Draw next best path
  visualization_->drawLines(vec2dTo3d(ed_ptr->next_best_path_), fp_->vis_scale_,
      Vector4d(1, 0.2, 0.2, 1), "next_path", 1, 6);

  // Draw next local point
  vector<Vector2d> local_points;
  local_points.push_back(fd_->local_pos_);
  visualization_->drawSpheres(vec2dTo3d(local_points), fp_->vis_scale_ * 3,
      Vector4d(0.2, 0.2, 1.0, 1), "local_point", 1, 6);

  visualization_->drawLines(vec2dTo3d(ed_ptr->tsp_tour_), fp_->vis_scale_ / 1.25,
      Vector4d(0.2, 1, 0.2, 1), "tsp_tour", 0, 6);

  visualization_->drawSpheres(vec2dTo3d(fd_->traveled_path_), fp_->vis_scale_ * 1.5,
      Vector4d(2.0 / 255.0, 111.0 / 255.0, 197.0 / 255.0, 1), "traveled_path", 1, 6);
}

void ExplorationFSM::clearVisMarker()
{
  auto ed_ptr = expl_manager_->ed_;
  for (int i = 0; i < 500; ++i) {
    visualization_->drawCubes({}, fp_->vis_scale_, Vector4d(0, 0, 0, 1), "frontier", i, 4);
    visualization_->drawCubes({}, fp_->vis_scale_, Vector4d(0, 0, 0, 1), "dormant_frontier", i, 4);
    visualization_->drawCubes({}, fp_->vis_scale_, Vector4d(0, 0, 0, 1), "object", i, 4);
  }

  visualization_->drawLines({}, fp_->vis_scale_, Vector4d(0, 0, 1, 1), "next_path", 1, 6);
}

bool ExplorationFSM::updateFrontierAndObject()
{
  bool change_flag = false;
  auto frt_map = expl_manager_->frontier_map2d_;
  auto obj_map = expl_manager_->object_map2d_;
  auto ed = expl_manager_->ed_;
  const Eigen::Vector2d sensor_pos(fd_->odom_pos_(0), fd_->odom_pos_(1));

  change_flag = frt_map->isAnyFrontierChanged();
  frt_map->searchFrontiers();
  change_flag |= frt_map->dormantSeenFrontiers(sensor_pos, fd_->odom_yaw_);
  frt_map->getFrontiers(ed->frontiers_, ed->frontier_averages_);
  frt_map->getDormantFrontiers(ed->dormant_frontiers_, ed->dormant_frontier_averages_);
  obj_map->getObjects(ed->objects_, ed->object_averages_, ed->object_labels_);
  publishObjectFusionState();

  return change_flag;
}

// Publish each cluster's fused confidence and the acceptance gate in force, so
// an evaluation run can pinpoint the update at which a cluster crossed the
// threshold - i.e. the moment the planner committed to an object
void ExplorationFSM::publishObjectFusionState()
{
  if (!object_fusion_pub_)
    return;

  vector<ObjectConfidenceInfo> infos;
  double min_confidence = 0.0;
  int min_observation_num = 0;
  expl_manager_->object_map2d_->getObjectConfidences(infos, min_confidence, min_observation_num);

  plan_env::msg::ObjectFusionState msg;
  msg.header.stamp = node_->get_clock()->now();
  msg.header.frame_id = "world";
  msg.update_seq = ++object_fusion_seq_;
  msg.min_confidence = static_cast<float>(min_confidence);
  msg.min_observation_num = min_observation_num;

  for (const auto& info : infos) {
    msg.ids.push_back(info.id);
    msg.best_labels.push_back(info.best_label);
    msg.confidences.push_back(static_cast<float>(info.confidence));
    msg.observation_nums.push_back(info.observation_num);
    msg.centroids_x.push_back(static_cast<float>(info.average(0)));
    msg.centroids_y.push_back(static_cast<float>(info.average(1)));
    msg.is_confident.push_back(info.is_confident);
    msg.verification.push_back(static_cast<int8_t>(info.verification));
  }

  object_fusion_pub_->publish(msg);
}

// Receive Habitat state messages
void ExplorationFSM::habitatStateCallback(const std_msgs::msg::Int32::SharedPtr msg)
{
  // sync_step: ACTION_FINISH comes with its step id on /habitat/action_finish_step
  if (!sync_step_ && msg->data == HABITAT_STATE::ACTION_FINISH &&
      state_ == ROS_STATE::WAIT_ACTION_FINISH)
    transitState(PLAN_ACTION, "Habitat Finish Action");
  if (msg->data == HABITAT_STATE::EPISODE_FINISH)
    init(node_);
  return;
}

// Periodically update frontiers and visualize in idle states
void ExplorationFSM::frontierCallback()
{
  // sync_step: frontier updates only happen while planning, a fixed number of times
  if (sync_step_)
    return;
  if (state_ != ROS_STATE::WAIT_TRIGGER && state_ != ROS_STATE::FINISH)
    return;

  updateFrontierAndObject();
  visualize();
}

// Receive user trigger to start exploration
void ExplorationFSM::triggerCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg)
{
  if (state_ != ROS_STATE::WAIT_TRIGGER)
    return;
  fd_->trigger_ = true;
  std::cout << "Triggered!" << std::endl;
  transitState(PLAN_ACTION, "triggerCallback");
}

// Receive robot odometry and update traveled path + marker
void ExplorationFSM::odometryCallback(const nav_msgs::msg::Odometry::SharedPtr msg)
{
  fd_->odom_pos_(0) = msg->pose.pose.position.x;
  fd_->odom_pos_(1) = msg->pose.pose.position.y;
  fd_->odom_pos_(2) = msg->pose.pose.position.z;

  fd_->odom_orient_.w() = msg->pose.pose.orientation.w;
  fd_->odom_orient_.x() = msg->pose.pose.orientation.x;
  fd_->odom_orient_.y() = msg->pose.pose.orientation.y;
  fd_->odom_orient_.z() = msg->pose.pose.orientation.z;

  Eigen::Vector3d rot_x = fd_->odom_orient_.toRotationMatrix().block<3, 1>(0, 0);
  fd_->odom_yaw_ = atan2(rot_x(1), rot_x(0));

  fd_->have_odom_ = true;

  Vector2d odom_pos2d = Vector2d(fd_->odom_pos_(0), fd_->odom_pos_(1));
  if (fd_->traveled_path_.empty())
    fd_->traveled_path_.push_back(odom_pos2d);
  else if ((fd_->traveled_path_.back() - odom_pos2d).norm() > 1e-2)
    fd_->traveled_path_.push_back(odom_pos2d);

  publishRobotMarker();
}

void ExplorationFSM::publishRobotMarker()
{
  const double robot_height = FSMConstants::ROBOT_HEIGHT;
  const double robot_radius = FSMConstants::ROBOT_RADIUS;

  // Create robot body cylinder marker
  visualization_msgs::msg::Marker robot_marker;
  robot_marker.header.frame_id = "world";
  robot_marker.header.stamp = node_->get_clock()->now();
  robot_marker.ns = "robot_position";
  robot_marker.id = 0;
  robot_marker.type = visualization_msgs::msg::Marker::CYLINDER;
  robot_marker.action = visualization_msgs::msg::Marker::ADD;

  // Set cylinder position
  robot_marker.pose.position.x = fd_->odom_pos_(0);
  robot_marker.pose.position.y = fd_->odom_pos_(1);
  robot_marker.pose.position.z = fd_->odom_pos_(2) + robot_height / 2.0;

  // Set cylinder orientation
  robot_marker.pose.orientation.x = fd_->odom_orient_.x();
  robot_marker.pose.orientation.y = fd_->odom_orient_.y();
  robot_marker.pose.orientation.z = fd_->odom_orient_.z();
  robot_marker.pose.orientation.w = fd_->odom_orient_.w();

  // Set cylinder dimensions
  robot_marker.scale.x = robot_radius * 2;  // Diameter
  robot_marker.scale.y = robot_radius * 2;  // Diameter
  robot_marker.scale.z = robot_height;      // Height

  // Set cylinder color (blue)
  robot_marker.color.r = 50.0 / 255.0;
  robot_marker.color.g = 50.0 / 255.0;
  robot_marker.color.b = 255.0 / 255.0;
  robot_marker.color.a = 1.0;

  // Create direction arrow marker
  visualization_msgs::msg::Marker arrow_marker;
  arrow_marker.header.frame_id = "world";
  arrow_marker.header.stamp = node_->get_clock()->now();
  arrow_marker.ns = "robot_direction";
  arrow_marker.id = 1;
  arrow_marker.type = visualization_msgs::msg::Marker::ARROW;
  arrow_marker.action = visualization_msgs::msg::Marker::ADD;

  // Set arrow position
  arrow_marker.pose.position.x = fd_->odom_pos_(0);
  arrow_marker.pose.position.y = fd_->odom_pos_(1);
  arrow_marker.pose.position.z = fd_->odom_pos_(2) + robot_height;

  // Set arrow orientation
  arrow_marker.pose.orientation.x = fd_->odom_orient_.x();
  arrow_marker.pose.orientation.y = fd_->odom_orient_.y();
  arrow_marker.pose.orientation.z = fd_->odom_orient_.z();
  arrow_marker.pose.orientation.w = fd_->odom_orient_.w();

  // Set arrow dimensions
  arrow_marker.scale.x = robot_radius + 0.13;  // Arrow length
  arrow_marker.scale.y = 0.08;                 // Arrow width
  arrow_marker.scale.z = 0.08;                 // Arrow thickness

  // Set arrow color (green)
  arrow_marker.color.r = 10.0 / 255.0;
  arrow_marker.color.g = 255.0 / 255.0;
  arrow_marker.color.b = 10.0 / 255.0;
  arrow_marker.color.a = 1.0;

  // Publish both markers
  robot_marker_pub_->publish(robot_marker);
  robot_marker_pub_->publish(arrow_marker);
}

void ExplorationFSM::confidenceThresholdCallback(const std_msgs::msg::Float64::SharedPtr msg)
{
  if (fd_->have_confidence_)
    return;
  fd_->have_confidence_ = true;
  expl_manager_->sdf_map_->object_map2d_->setConfidenceThreshold(msg->data);
}

// Transition FSM state and log the change
void ExplorationFSM::transitState(ROS_STATE new_state, string pos_call)
{
  int pre_s = int(state_);
  state_ = new_state;
  std::cout << "[ " + pos_call + "]: from " + fd_->state_str_[pre_s] + " to " +
              fd_->state_str_[int(new_state)]
       << std::endl;
}
}  // namespace apexnav_planner
