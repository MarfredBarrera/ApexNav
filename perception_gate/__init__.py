"""Perception gate: a target cluster must pass 3DAwareVerification's
``perception.Verifier`` (served over HTTP) before the planner may stop at it.

The C++ FSM sends /object/verification_request when it selects an unverified
cluster and waits; ``VerificationGate.handle`` picks the best stored view of
that cluster, verifies it, and the verdict goes back on
/object/verification_result. See gate.py.
"""
from perception_gate.gate import VerificationGate

__all__ = ["VerificationGate"]
