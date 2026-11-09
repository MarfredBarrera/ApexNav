#ifndef _EXPL_DATA_H_
#define _EXPL_DATA_H_

#include <Eigen/Eigen>
#include <cstdint>
#include <iostream>
#include <vector>
#include <trajectory_manager/optimizer.h>

// Undefine uint macro from optimizer.h to avoid conflict with OpenCV
#ifdef uint
#undef uint
#endif

namespace apexnav_planner {

enum FINAL_RESULT { EXPLORE, SEARCH_OBJECT, STUCKING, NO_FRONTIER, REACH_OBJECT };

struct FSMData {
  FSMData()
  {
    trigger_ = false;
    have_odom_ = false;
    have_confidence_ = false;
    have_finished_ = false;
    static_state_ = true;
    state_str_ = { "INIT", "WAIT_TRIGGER", "PLAN_ACTION", "WAIT_ACTION_FINISH", "PUB_ACTION",
      "FINISH", "WAIT_VERIFICATION" };

    odom_pos_ = Eigen::Vector3d::Zero();
    odom_vel_ = Eigen::Vector3d::Zero();
    odom_omega_ = Eigen::Vector3d::Zero();
    odom_orient_ = Eigen::Quaterniond::Identity();
    odom_yaw_ = 0.0;
    start_pt_ = Eigen::Vector3d::Zero();
    start_vel_ = Eigen::Vector3d::Zero();
    start_yaw_ = Eigen::Vector3d::Zero();
    last_start_pos_ = Eigen::Vector3d(-100, -100, -100);
    last_next_pos_ = Eigen::Vector2d(-100, -100);
    newest_action_ = -1;
    init_action_count_ = 0;
    stucking_action_count_ = 0;
    stucking_next_pos_count_ = 0;
    traveled_path_.clear();

    final_result_ = -1;
    replan_flag_ = true;
    dormant_frontier_flag_ = false;
    escape_stucking_flag_ = false;
    escape_stucking_count_ = 0;
    stucking_points_.clear();

    local_pos_ = Eigen::Vector2d(0, 0);

    skip_stuck_check_once_ = false;
    verify_seq_ = 0;
    verify_start_ = 0.0;
    verify_cluster_id_ = -1;
    verify_pos_ = Eigen::Vector2d(0, 0);
    verify_at_stop_ = false;
    last_expl_result_ = -1;

    ap_active_ = false;
    ap_cluster_id_ = -1;
    ap_approach_id_ = 0;
    ap_steps_ = 0;
    ap_max_steps_ = 0;
    ap_object_ = Eigen::Vector2d::Zero();
    ap_camera_ = Eigen::Vector2d::Zero();
    ap_standoff_ = false;
    verify_arrival_ = false;
    verify_approach_id_ = 0;
    verify_stuck_ = false;
    ap_face_turns_ = 0;
    stop_face_pending_ = false;
  }
  // FSM data
  bool trigger_, have_odom_, have_confidence_;
  bool have_finished_;
  std::vector<string> state_str_;
  std::vector<Eigen::Vector2d> traveled_path_;

  // odometry state
  Eigen::Vector3d odom_pos_, odom_vel_, odom_omega_;
  Eigen::Quaterniond odom_orient_;
  double odom_yaw_;
  bool static_state_;  // Track if robot is static or moving

  Eigen::Vector3d start_pt_, start_vel_, start_yaw_;
  Eigen::Vector3d last_start_pos_;
  Eigen::Vector2d last_next_pos_;
  int newest_action_;
  int init_action_count_;
  int stucking_action_count_;
  int stucking_next_pos_count_;

  int final_result_;
  bool replan_flag_, dormant_frontier_flag_;
  bool escape_stucking_flag_;
  int escape_stucking_count_;
  Eigen::Vector2d escape_stucking_pos_;
  double escape_stucking_yaw_;
  std::vector<Eigen::Vector3d> stucking_points_;

  Eigen::Vector2d local_pos_;
  LocalTrajectory newest_traj_;  // Store latest planned trajectory

  // Perception gate (verification.enabled). A replan right after a verdict
  // happens at the same pose, which the stuck detectors would read as a
  // failed forward step, so they are skipped once
  bool skip_stuck_check_once_;
  uint32_t verify_seq_;  // seq of the pending VerificationRequest
  int last_expl_result_;  // EXPL_RESULT of the latest planNextBestPoint
  double verify_start_;  // node clock (s) when it was sent
  int verify_cluster_id_;  // cluster it asks about; -1: a bare position (stop backstop)
  Eigen::Vector2d verify_pos_;
  bool verify_at_stop_;  // sent by the stop backstop

  // Perception gate approach (state 3, verification.approach): the cluster is the planner's
  // forced target (ExplorationManager::forced_cluster_id_) and the usual object navigation
  // drives there; a new view on the way, the arrival, or a failure goes back to the gate
  bool ap_active_;
  int ap_cluster_id_, ap_approach_id_;
  int ap_steps_, ap_max_steps_;  // actions taken in this approach / allowed
  Eigen::Vector2d ap_object_;    // object point of the approach (evidence centroid), for the log
  Eigen::Vector2d ap_camera_;    // camera of the evidence view that issued the approach
  bool ap_standoff_;             // the approach arrived at the standoff, not at the object
  bool verify_arrival_;          // the pending request is the approach's arrival (at the stop)
  int verify_approach_id_;       // the pending request is about this approach (0: ordinary)
  bool verify_stuck_;            // ... ended because the agent got stuck: stop after the verdict
  int ap_face_turns_;            // turns toward the object after reaching it
  bool stop_face_pending_;  // the backstop turned to face the object; verify at the next step
};

struct FSMParam {
  FSMParam()
  {
    vis_scale_ = 0.1;
    replan_time_ = 0.2;
    replan_traj_end_threshold_ = 1.0;
    replan_frontier_change_delay_ = 0.5;
    replan_timeout_ = 2.0;

    const double step_length = 0.25;
    const double angle_increment = M_PI / 6;
    action_steps_.clear();
    for (int i = 0; i < 12; ++i) {
      double angle = i * angle_increment;
      Eigen::Vector2d step(step_length * cos(angle), step_length * sin(angle));
      action_steps_.push_back(step);
    }
  }
  double vis_scale_;
  std::vector<Eigen::Vector2d> action_steps_;
  // replan timing parameters (loaded from ros params in ExplorationFSM::init)
  double replan_time_;
  double replan_traj_end_threshold_;
  double replan_frontier_change_delay_;
  double replan_timeout_;
};

struct ExplorationData {
  ExplorationData()
  {
    frontiers_.clear();
    frontier_averages_.clear();
    dormant_frontiers_.clear();
    dormant_frontier_averages_.clear();
    objects_.clear();
    object_averages_.clear();
    object_labels_.clear();
    next_pos_ = Eigen::Vector2d(0, 0);
    next_best_path_.clear();
    tsp_tour_.clear();
    target_cluster_id_ = -1;
    target_object_pos_ = Eigen::Vector2d(0, 0);
  }
  std::vector<std::vector<Eigen::Vector2d>> frontiers_, dormant_frontiers_;
  std::vector<Eigen::Vector2d> frontier_averages_, dormant_frontier_averages_;
  std::vector<std::vector<Eigen::Vector2d>> objects_;
  std::vector<Eigen::Vector2d> object_averages_;
  std::vector<int> object_labels_;
  Eigen::Vector2d next_pos_;
  Eigen::Vector2d next_local_pos_;  // Local target position along path
  std::vector<Eigen::Vector2d> next_best_path_;
  std::vector<Eigen::Vector2d> tsp_tour_;
  // Object cluster the last SEARCH_* result navigates to; -1 for exploration,
  // over-depth targets and the extreme "others" cloud
  int target_cluster_id_;
  // Nearest point of the target object (next_pos_ is the approach point 0.5-0.85 m
  // away); the stop backstop verifies what is there
  Eigen::Vector2d target_object_pos_;
};

struct ExplorationParam {
  enum POLICY_MODE { DISTANCE, SEMANTIC, HYBRID, TSP_DIST };
  // params
  int policy_mode_;
  double sigma_threshold_, max_to_mean_threshold_, max_to_mean_percentage_;
  std::string tsp_dir_;
};

}  // namespace apexnav_planner

#endif
