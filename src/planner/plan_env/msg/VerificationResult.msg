# VerificationResult.msg
#
# Reply to a VerificationRequest (same seq): state 1 accepts the cluster for
# good, -1 rejects it (excluded from every target list), 2 defers it (it stays a
# target and the agent may approach it, but the stop backstop verifies it again
# before stopping). state 0 with any seq reopens a rejected cluster, so the next
# selection requests it again; for a cluster under approach (3) it instead says a
# new view of it arrived (the planner ends the approach early, viewpoint_status 4).
# state 3 (approach, server selector d1): the view was insufficient; drive to a
# viewpoint nearer the object point along the line of sight (or to the side), face
# it, and send a VerificationRequest with approach_id and viewpoint_status set.

uint32 seq
int32 cluster_id
int8 state                 # -1 rejected, 0 unverified (reopen), 1 accepted, 2 deferred, 3 approach
# state 3 only (zero otherwise):
int32 approach_id          # 1-based approach index for this cluster
float32 object_x           # the point to approach (evidence centroid), planner world frame
float32 object_y
float32 camera_x           # camera of the insufficient view (line of sight object -> camera)
float32 camera_y
float32 min_range          # closest object distance of any view the gate judged for the cluster
int32 max_steps            # actions the planner may spend on this approach
bool side_only             # stop backstop: only the +-40 deg bearings
