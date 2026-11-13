# VerificationResult.msg
#
# Reply to a VerificationRequest (same seq): state 1 accepts the cluster for
# good, -1 rejects it (excluded from every target list), 2 defers it (it stays a
# target and the agent may approach it, but the stop backstop verifies it again
# before stopping). state 0 with any seq reopens a rejected cluster, so the next
# selection requests it again; for a cluster under approach (3) it instead says a
# new view of it arrived (the planner ends the approach early, viewpoint_status 4).
# state 3 (approach, server selector d1): the view was insufficient; the cluster becomes
# the planner's target and its usual object navigation drives toward it. A new view on
# the way (a state-0 message for the cluster), the arrival (after facing the object) or
# a failure is sent as a VerificationRequest with approach_id and viewpoint_status set;
# state 3 in reply to a new view means: keep approaching.

uint32 seq
int32 cluster_id
int8 state                 # -1 rejected, 0 unverified (reopen), 1 accepted, 2 deferred, 3 approach
# state 3 only (zero otherwise):
int32 approach_id          # 1-based approach index for this cluster
float32 object_x           # the point to approach (evidence centroid), planner world frame
float32 object_y
float32 camera_x           # camera of the insufficient view (line of sight object -> camera)
float32 camera_y
int32 max_steps            # actions the planner may spend on this approach
# Approach run outside the planner (gate approach_planner mppi/closer, sync_step only): the gate drove the
# agent itself and answers the original request with state 1 / -1 and resume_step = the last executed step.
# The planner first integrates every step up to resume_step (their StepObservations), then replans from the
# new pose. After an accept the gate also drives the last mile and stops the episode itself; an accept with
# resume_step means its last mile gave up far from the object: the cluster is accepted, not a forced target.
int32 resume_step          # 0: an ordinary reply
