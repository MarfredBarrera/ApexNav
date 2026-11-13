"""Metric-size opinion for the approach verdict (study 2026-11-10: armchairs accepted as "couch").

SAM3 cannot separate an armchair from a couch; the tracked object's metric width can, once the object has been
seen whole. W = the longer horizontal side of the fused cloud O (robust 2-98 % footprint extents along its PCA
axes, geom.ObjectModel.robust_extent). The opinion is one likelihood per class, multiplied into the fused SAM3
belief (decide.fused_accept):

    belief'(c) = belief(c) x likelihood(W | c) / sum,   accept iff the target is still the top class.

It is a VETO only (2026-11-10): metric size can rule the target out (too small for it) but never confirm it. A
measured W can be inflated by a merge with a neighbour (test2 ep 181: an armchair merged with a sectional read
1.30 m "complete" and the two-sided opinion flipped reject -> accept, a false stop). So the likelihoods are taken
relative to the target and one-sided:

    likelihood(target) = 1,   likelihood(c) = max(1, p(W | c) / p(W | target))   if W < SMALL_BELOW[target]
    likelihood(c) = 1 for every class                                          otherwise

Every class is multiplied by >= 1 and the target by 1, so the target's belief (and the accept) can only fall; and
only a width below the target's smallest typical size (the smallest median of its prior's components, couch: a
1.5 m loveseat) carries evidence, so a large W never shifts mass between classes.

It is VACUOUS (every likelihood 1, nothing changes) unless
  1. W is complete: some view that contributed to O is neither side-clipped (mask within CLIP_PX of the left or
     right image border) nor side-occluded (``side_flags``: the depth just outside the mask's left / right edge is
     closer than the object) and spans >= COMPLETE_FRAC x W. A clipped or occluded view gives only a lower bound
     on the width, and a lower bound must never count against a bigger class (offline, scoring it as a censored
     likelihood turned 6 rejects into false accepts);
  2. the target is in SIZE_TARGETS: a class whose size prior can exclude its smaller look-alikes.

Priors are fixed a priori from standard furniture dimensions (not fit to episodes); classes without a prior get a
generic log-uniform density over [0.1, 4] m, so a measured W only shifts mass between classes that have priors.
Measurement model (sensor, not class): W_obs = W r, log r ~ N(0, SIG); with probability PI part of the object is
unseen (W_obs = W U, U ~ U(0.3, 1)); with probability RHO O is inflated by a merge (W_obs = W V, V ~ U(1, 2)).
"""
import math

import numpy as np

SIG, PI, RHO = 0.12, 0.25, 0.10   # log-width noise (per-view spread), partly unseen, merge inflation
COMPLETE_FRAC = 0.9               # a whole view spans >= this fraction of W
CLIP_PX = 2                       # a mask within this many px of the left / right border is side-clipped
OCC_BAND = (2, 8)                 # side occlusion: depth this many px (from, to) outside the mask's edge, per row
OCC_EDGE_PX = 3                   # ... compared with the object's depth this many px inside the edge
OCC_MARGIN = 0.10                 # ... closer by this much (m) than both the object's mean and the row's edge
OCC_FRAC = 0.2                    # ... in at least this fraction of the mask's rows: that side is occluded
DEPTH_MIN, DEPTH_MAX = 0.05, 4.99


def _ln(m, s, w):
    """Log-normal density of w with median m and log-sd s."""
    return math.exp(-0.5 * ((math.log(w) - math.log(m)) / s) ** 2) / (w * s * math.sqrt(2 * math.pi))


# Width priors W (longer horizontal side, m): mixtures of (weight, median, log-sd). Sources: furniture standards
# and retail size ranges (Neufert-style tables).
#   chair: armchair / club / wingback 0.70-1.00 m (median 0.85); side / dining / office chair 0.40-0.60 (0.50)
#   couch: loveseat 1.32-1.83 m (median 1.50); 3-seat sofa 1.83-2.29 (2.10); sectional (2.9)
#   bed:   mattress + frame length 1.9-2.1 m (median 2.05)
SIZE_PRIORS = {"chair": ((0.5, 0.85, 0.15), (0.5, 0.50, 0.15)),
               "couch": ((0.3, 1.50, 0.10), (0.5, 2.10, 0.10), (0.2, 2.90, 0.15)),
               "bed": ((1.0, 2.05, 0.05),)}
# Targets whose verdict uses the size opinion. Evidence (offline replay of sync_jev_sam3_mppi_v15_assoc5 +
# heldout_assoc5, eye labels): couch: 3 of 5 armchair false accepts (eps 445, 283, held-out 648) become rejects,
# 0 false rejects on 10 real couch objects (fully seen couches >= 1.33 m, the accepted armchairs 0.78-1.11 m).
# bed: no rule found that helps without risk. chair: applying it broke 3 real chairs (O sometimes inflated by
# merges: W 1.5-2.6 m), so chair targets stay size-free.
SIZE_TARGETS = frozenset({"couch"})
# A width is evidence (too small for the target) only below this: the smallest component median of its prior
SMALL_BELOW = {cls: min(m for _, m, _ in comps) for cls, comps in SIZE_PRIORS.items()}

_GRID = [0.1 * math.exp(k * 0.01) for k in range(int(math.log(60) / 0.01))]   # W from 0.1 to 6 m, log-spaced


def _measurement(w_obs, w):
    """p(W_obs | true width w)."""
    p = (1 - PI - RHO) * _ln(w, SIG, w_obs)
    if 0.3 * w <= w_obs <= w:
        p += PI / (0.7 * w)
    if w <= w_obs <= 2 * w:
        p += RHO / w
    return p


def class_likelihood(cls, w_obs):
    """p(W_obs | cls): the measurement model integrated over the class's width prior (generic log-uniform density
    over [0.1, 4] m for a class without a prior)."""
    comps = SIZE_PRIORS.get(cls)
    if comps is None:
        return 1.0 / (w_obs * math.log(40)) if 0.1 <= w_obs <= 4 else 1e-6
    dlog = 0.01   # integral over log W of prior(W) W p(W_obs | W)
    return sum(sum(a * _ln(m, s, w) for a, m, s in comps) * w * dlog * _measurement(w_obs, w) for w in _GRID)


def likelihood(width, complete, target, classes):
    """The size opinion (a veto): {class: likelihood of the observed width} over ``classes``, relative to the target
    and one-sided (module docstring): 1 for the target, max(1, p(W | c) / p(W | target)) for the others. Vacuous
    (every value 1.0) unless the width is ``complete``, the target is in SIZE_TARGETS and W < SMALL_BELOW[target]."""
    if (not complete or width is None or not width > 0 or target not in SIZE_TARGETS
            or not float(width) < SMALL_BELOW[target]):
        return {c: 1.0 for c in classes}
    w = float(width)
    own = class_likelihood(target, w)
    return {c: 1.0 if c == target else max(1.0, class_likelihood(c, w) / own) for c in classes}


def update(belief, lik):
    """belief x likelihood, renormalised over the classes (unchanged when the product vanishes)."""
    post = {c: b * lik.get(c, 1.0) for c, b in belief.items()}
    total = sum(post.values())
    return {c: v / total for c, v in post.items()} if total > 0 else dict(belief)


def top(belief, target):
    """The top class (a tie goes to the target, as select.fuse_views)."""
    return max(belief, key=lambda c: (belief[c], c == target))


# ---------------------------------------------------------------------------------------------------------------
# completeness of W (live: from each contributing view's mask, depth and back-projected points)

def side_flags(mask, depth):
    """{clip_left, clip_right, occ_left, occ_right} of one view's object mask. Clipped: the mask comes within
    CLIP_PX of that image border. Occluded: in >= OCC_FRAC of the mask's rows, the depth OCC_BAND px outside that
    edge is closer by OCC_MARGIN than both the object's mean depth and the object's depth just inside the edge of
    that row (an occluder in front of the edge; the row's edge also keeps the floor beside an oblique object's
    near end from reading as an occluder)."""
    mask = np.asarray(mask, bool)
    depth = np.asarray(depth, np.float32)
    h, w = mask.shape
    cols = np.flatnonzero(mask.any(0))
    out = {"clip_left": False, "clip_right": False, "occ_left": False, "occ_right": False}
    if not len(cols):
        return out
    out["clip_left"], out["clip_right"] = bool(cols[0] < CLIP_PX), bool(cols[-1] >= w - CLIP_PX)
    valid = (depth > DEPTH_MIN) & (depth < DEPTH_MAX)
    inside = depth[mask & valid]
    if not len(inside):
        return out
    mean = float(inside.mean())
    a, b = OCC_BAND
    votes = {"left": [0, 0], "right": [0, 0]}   # [occluded rows, rows with a band]
    for r in np.flatnonzero(mask.any(1)):
        row, ok = depth[r], valid[r]
        xs = np.flatnonzero(mask[r])
        for side, edge, band in (("left", xs[0], slice(max(xs[0] - b, 0), max(xs[0] - a, 0))),
                                 ("right", xs[-1], slice(min(xs[-1] + a + 1, w), min(xs[-1] + b + 1, w)))):
            near = slice(edge, edge + OCC_EDGE_PX) if side == "left" else slice(max(edge - OCC_EDGE_PX + 1, 0), edge + 1)
            e = row[near][ok[near] & mask[r, near]]
            o = row[band][ok[band]]
            if not len(o) or not len(e):
                continue
            votes[side][1] += 1
            votes[side][0] += float(np.median(o)) < min(mean, float(np.median(e))) - OCC_MARGIN
    for side, (k, n) in votes.items():
        out[f"occ_{side}"] = bool(n and k / n >= OCC_FRAC)
    return out


def complete(views, width):
    """W is complete when some contributing view is neither side-clipped nor side-occluded and its span reaches
    COMPLETE_FRAC x W. ``views``: [{span, clip_left, clip_right, occ_left, occ_right}] (span None: unknown)."""
    if width is None or not width > 0:
        return False
    return any(v.get("span") is not None and v["span"] >= COMPLETE_FRAC * width
               and not any(v.get(k) for k in ("clip_left", "clip_right", "occ_left", "occ_right")) for v in views)
