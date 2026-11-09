"""CLOSER's evidence cache with a CLIP view-diversity gate (arXiv 2610.28476, Sec. III-C2, Alg. 1).

Consecutive frames of one candidate are nearly identical; judging each wastes verifier calls and biases
the fused decision toward repeated views. A frame enters the cache (and is judged, once) only if

    rho_t = min_j (1 - cos(v_t, v_j)) > delta       (the first frame is always eligible)

over the cached CLIP features v_j. While fewer than K views are cached, an eligible frame is appended.
With K cached, the gate keeps the size-K subset of the K + 1 features with the largest mean pairwise
cosine distance; if that subset drops the new frame, the frame is discarded unjudged, otherwise the
dropped entry is evicted. CLOSER: K = 4, delta = 0.18 (CLIP ViT-L/14 on the full frame).

Features come from ``perception.server`` POST /embed (``client.embed``), L2-normalised.
"""
from itertools import combinations

import numpy as np

CACHE_K = 4
DELTA_VIEW = 0.18


def novelty(feature, features):
    """rho: smallest cosine distance from ``feature`` to ``features`` (inf when there are none)."""
    if not len(features):
        return float("inf")
    return float(np.min(1.0 - np.asarray(features) @ np.asarray(feature)))


def diversity(features):
    """Mean pairwise cosine distance of ``features`` (0 for fewer than two)."""
    features = np.asarray(features)
    if len(features) < 2:
        return 0.0
    return float(np.mean([1.0 - features[a] @ features[b] for a, b in combinations(range(len(features)), 2)]))


class DiversityGate:
    """The evidence cache of one tracked candidate. ``items`` are the cached views (any objects), in
    admission order, ``features`` their CLIP features."""

    def __init__(self, k=CACHE_K, delta=DELTA_VIEW):
        if k < 1:
            raise ValueError("cache capacity k must be >= 1")
        self.k, self.delta = k, delta
        self.items, self.features = [], []

    def __len__(self):
        return len(self.items)

    def offer(self, item, feature):
        """Admit ``item`` with CLIP ``feature`` if it passes the gate. Returns {admitted, rho, evicted}
        (``evicted``: the item the new one replaced, or None)."""
        feature = np.asarray(feature, dtype=np.float64)
        rho = novelty(feature, self.features)
        if self.items and not rho > self.delta:
            return {"admitted": False, "rho": rho, "evicted": None}
        if len(self.items) < self.k:
            self.items.append(item)
            self.features.append(feature)
            return {"admitted": True, "rho": rho, "evicted": None}
        pool = self.features + [feature]
        best = max(combinations(range(len(pool)), self.k), key=lambda s: diversity([pool[i] for i in s]))
        if len(pool) - 1 not in best:
            return {"admitted": False, "rho": rho, "evicted": None}
        drop = next(i for i in range(len(self.items)) if i not in best)
        evicted = self.items.pop(drop)
        self.features.pop(drop)
        self.items.append(item)
        self.features.append(feature)
        return {"admitted": True, "rho": rho, "evicted": evicted}
