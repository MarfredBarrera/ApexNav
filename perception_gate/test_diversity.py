"""CLOSER evidence cache / diversity gate (perception_gate.diversity)."""
import numpy as np
import pytest

from perception_gate.diversity import DiversityGate, diversity, novelty


def unit(*v):
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v)


def test_first_frame_always_admitted():
    gate = DiversityGate(k=4, delta=0.18)
    reply = gate.offer("a", unit(1, 0, 0))
    assert reply["admitted"] and reply["rho"] == float("inf") and gate.items == ["a"]


def test_near_duplicate_rejected_and_novel_admitted():
    gate = DiversityGate(k=4, delta=0.18)
    gate.offer("a", unit(1, 0, 0))
    near = gate.offer("b", unit(1, 0.1, 0))       # cos ~0.995 -> rho ~0.005
    assert not near["admitted"] and near["rho"] < 0.18
    far = gate.offer("c", unit(0, 1, 0))           # orthogonal -> rho 1
    assert far["admitted"] and far["rho"] == pytest.approx(1.0)
    assert gate.items == ["a", "c"]


def test_rho_must_exceed_delta():
    gate = DiversityGate(k=4, delta=0.5)
    gate.offer("a", unit(1, 0))
    assert not gate.offer("b", unit(0.5, np.sqrt(0.75)))["admitted"]   # rho exactly 0.5


def test_full_cache_keeps_most_diverse_subset():
    gate = DiversityGate(k=2, delta=0.0)
    gate.offer("x", unit(1, 0, 0))
    gate.offer("x2", unit(1, 0.3, 0))              # close to x
    reply = gate.offer("z", unit(0, 0, 1))         # far from both: replaces one of the close pair
    assert reply["admitted"] and reply["evicted"] in ("x", "x2")
    assert "z" in gate.items and len(gate.items) == 2
    before = list(gate.items)
    reply = gate.offer("w", unit(1, 0.2, 0.05))    # novel enough for delta 0 but adds no diversity
    assert not reply["admitted"] and gate.items == before


def test_helpers():
    assert novelty(unit(1, 0), []) == float("inf")
    assert diversity([unit(1, 0)]) == 0.0
    assert diversity([unit(1, 0), unit(0, 1)]) == pytest.approx(1.0)
