# ObjectFusionState.msg
#
# Per-cluster snapshot of the object map's fused confidence, published every
# planner update. Lets an evaluation run reconstruct when a cluster's fused
# confidence crossed the acceptance threshold - i.e. the step at which the
# planner committed to an object.
#
# All the array fields are parallel and share one entry per object cluster.

std_msgs/Header header

# Increments once per planner update, so a consumer can tell whether the
# snapshot it latched is stale relative to its own step counter
uint32 update_seq

# Acceptance gate currently in force. min_confidence comes from the LLM's
# per-category threshold on /detector/confidence_threshold; min_observation_num
# is the object.min_observation_num launch parameter
float32 min_confidence
int32 min_observation_num

int32[] ids                 # cluster id
int32[] best_labels         # most confident semantic class, -1 if none
float32[] confidences       # fused confidence for label 0 (the target class)
int32[] observation_nums    # frames the target class was observed in
float32[] centroids_x
float32[] centroids_y
bool[] is_confident         # passes both halves of the acceptance gate
