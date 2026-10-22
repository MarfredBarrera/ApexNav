# VerificationRequest.msg
#
# The FSM is about to commit to a target cluster it has not verified yet (best,
# suspicious or extreme), or about to stop at a target no accepted cluster
# covers (at_stop: the centroid is the stop target position; cluster_id -1
# when no cluster is near, e.g. an over-depth target). It waits in
# WAIT_VERIFICATION until a VerificationResult with the same seq arrives, or
# verification.timeout_s.

uint32 seq                 # increments per request (never reset, so stale replies cannot match)
int32 cluster_id           # index into the object map (stable within an episode); -1: position
float32 centroid_x         # cluster centroid; with at_stop the stop target position; planner world frame
float32 centroid_y
float32 confidence         # fused confidence for label 0 (the target class)
int32 observation_num      # frames the target class was observed in
int32 expl_result          # EXPL_RESULT that selected the cluster
bool at_stop               # sent by the stop backstop (REACH_OBJECT)
