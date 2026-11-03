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
uint32 fusion_seq          # ObjectFusionState.update_seq published before this request (sync_step)
# Approach (perception gate state 3, server selector d1): a re-request about one approach.
# Ordinary requests leave all of these 0.
int32 approach_id          # 0: ordinary request; else the approach (1-based per cluster) the status refers to
int32 viewpoint_status     # 1 arrived (object reached) and facing the object; 2 no object path / target lost; 3 aborted (step budget); 4 early (new view during the approach)
float32 viewpoint_x        # the agent's position, planner world frame
float32 viewpoint_y
int32 approach_steps       # actions spent on this approach
