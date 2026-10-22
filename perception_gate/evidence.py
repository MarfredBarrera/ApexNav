"""Verification evidence: target detections stored per place, ranked by readability.

``view_quality``, ``mask_touches_border`` and ``MATCH_DISTANCE_M`` mirror
basic_utils/record_episode/episode_visualization.py (``_view_quality``,
``_mask_touches_border``, ``COMMIT_MATCH_DISTANCE_M``), so the online gate and
the offline commit review rank views the same way.
"""
import math

# How far a detection may sit from a cluster's centroid and still count as a
# view of it.
MATCH_DISTANCE_M = 0.75
# A mask covering this fraction of the frame already shows the object well
# enough to identify, so larger views stop earning extra rank beyond it.
FULL_VIEW_AREA_FRACTION = 0.10


def view_quality(score, area_fraction, clipped=False):
    """Mask area dominates until the object fills a readable part of the frame;
    then the detector score breaks ties. A mask running off the frame yields to
    a whole view of comparable size."""
    readable = min(1.0, area_fraction / FULL_VIEW_AREA_FRACTION)
    quality = readable * max(float(score) if score is not None else 0.0, 0.05)
    return quality * (0.7 if clipped else 1.0)


def mask_touches_border(mask):
    """True when a mask runs off the frame, so the object is only partly shown."""
    binary = mask > 0
    return bool(
        binary[0].any() or binary[-1].any() or binary[:, 0].any() or binary[:, -1].any()
    )


class EvidenceStore:
    """Detections bucketed by the ``cell_m`` grid cell of their world centroid.

    Each cell keeps its ``per_cell`` most readable views plus its newest one, so
    one heavily observed object cannot evict the only view of another. Past
    ``max_total`` views, the most crowded cell loses its least readable
    non-newest view (a cell's last view is never dropped for another cell).
    """

    def __init__(self, cell_m=MATCH_DISTANCE_M, per_cell=8, max_total=256):
        self.cell_m, self.per_cell, self.max_total = cell_m, per_cell, max_total
        self.cells = {}

    def __iter__(self):
        for items in self.cells.values():
            yield from items

    def __len__(self):
        return sum(len(items) for items in self.cells.values())

    def add(self, item):
        """``item`` needs "centroid" (x, y), "quality" and "step"."""
        key = (math.floor(item["centroid"][0] / self.cell_m), math.floor(item["centroid"][1] / self.cell_m))
        items = self.cells.setdefault(key, [])
        items.append(item)
        if len(items) > self.per_cell + 1:
            self._drop_worst(items)
        while len(self) > self.max_total:
            crowded = max(self.cells.values(), key=len)
            if len(crowded) < 2:
                break
            self._drop_worst(crowded)

    @staticmethod
    def _drop_worst(items):
        newest = max(items, key=lambda e: e["step"])
        items.remove(min((e for e in items if e is not newest), key=lambda e: e["quality"]))
