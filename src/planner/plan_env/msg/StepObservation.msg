# One executed simulator step in sync_step mode: every map input of that step in one
# message, so the planner integrates it exactly once and in a fixed order
# (pose -> depth/occupancy -> value map with this step's ITM -> object clouds).

# 0 = the observation right after env.reset(); k = after the k-th executed action
int32 step

# Robot odometry (as on /habitat/odom) and camera pose (as on /habitat/sensor_pose)
nav_msgs/Odometry odom
nav_msgs/Odometry camera_pose

# Depth image (32FC1, normalized as on /habitat/camera_depth)
sensor_msgs/Image depth

# BLIP-2 ITM cosine score of this step (has_itm false at step 0)
bool has_itm
float64 itm_score

# Detector output of this step (has_clouds false at step 0, where nothing is detected)
bool has_clouds
MultipleMasksWithConfidence clouds

# Rejected clusters the perception gate reopens after this step (re-check cooldown passed)
int32[] reopen_cluster_ids
