# VerificationResult.msg
#
# Reply to a VerificationRequest (same seq): state 1 accepts the cluster for
# good, -1 rejects it (excluded from every target list). state 0 with any seq
# reopens a rejected cluster, so the next selection requests it again.

uint32 seq
int32 cluster_id
int8 state                 # -1 rejected, 0 unverified (reopen), 1 accepted
