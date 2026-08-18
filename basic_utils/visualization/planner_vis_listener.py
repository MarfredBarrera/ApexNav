"""
Planner Visualization Listener

Subscribes to the visualization topics published by the ApexNav C++ planner
(the same ones RViz / Foxglove render) and accumulates them into dense 2D
grids plus a marker table, so the evaluation script can rasterize a top-down
view of the planner's internal state for every video frame.

The map topics are published incrementally: each cycle only covers the region
the mapper touched since the last update. Within one cycle every cell of that
region appears in exactly one of the occupied / inflated / free / unknown
clouds, so writing cells as they arrive keeps the accumulated state grid exact
without any explicit clearing.

Note that all of the /grid_map/* publishers early-out when they have no
subscribers, so simply creating this listener turns that work back on in the
planner node.

Author: ApexNav contributors
"""

import threading

import numpy as np
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import PointCloud2
from visualization_msgs.msg import Marker

# Cell states of the accumulated occupancy grid
CELL_UNKNOWN = 0
CELL_FREE = 1
CELL_INFLATE = 2
CELL_OCCUPIED = 3

# Marker namespaces published by ExplorationFSM::visualize()
NS_FRONTIER = "frontier"
NS_DORMANT_FRONTIER = "dormant_frontier"
NS_OBJECT = "object"
NS_NEXT_PATH = "next_path"
NS_TSP_TOUR = "tsp_tour"
NS_LOCAL_POINT = "local_point"
NS_TRAVELED_PATH = "traveled_path"

# sensor_msgs/PointField datatype -> numpy dtype
_PC2_DTYPES = {
    1: np.int8,
    2: np.uint8,
    3: np.int16,
    4: np.uint16,
    5: np.int32,
    6: np.uint32,
    7: np.float32,
    8: np.float64,
}


def pointcloud2_to_xyzi(msg):
    """Parse a PointCloud2 into an (N, 3) float32 array of (x, y, intensity).

    Intensity is zero-filled for clouds that do not carry the field. Written
    against the raw buffer so no ROS point-cloud helper package is required.
    """
    count = msg.width * msg.height
    if count == 0:
        return np.zeros((0, 3), dtype=np.float32)

    names, formats, offsets = [], [], []
    for field in msg.fields:
        if field.name in names:
            continue
        names.append(field.name)
        formats.append(_PC2_DTYPES[field.datatype])
        offsets.append(field.offset)

    dtype = np.dtype(
        {
            "names": names,
            "formats": formats,
            "offsets": offsets,
            "itemsize": msg.point_step,
        }
    )
    raw = np.frombuffer(memoryview(msg.data), dtype=dtype, count=count)

    out = np.empty((raw.shape[0], 3), dtype=np.float32)
    out[:, 0] = raw["x"]
    out[:, 1] = raw["y"]
    out[:, 2] = raw["intensity"] if "intensity" in names else 0.0
    return out


def marker_points(msg):
    """Extract a marker's point list as an (N, 2) float32 array of (x, y)."""
    if not msg.points:
        return np.zeros((0, 2), dtype=np.float32)
    return np.array([(p.x, p.y) for p in msg.points], dtype=np.float32)


class PlannerSnapshot:
    """Immutable crop of the accumulated planner state, safe to render."""

    def __init__(self, state, value, cell_min, origin, resolution, markers):
        self.state = state  # (h, w) uint8, indexed [iy, ix]
        self.value = value  # (h, w) float32, -1 where no value
        self.cell_min = cell_min  # (ix0, iy0) cell index of crop corner
        self.origin = origin  # world coords of cell (0, 0) corner
        self.resolution = resolution
        self.markers = markers  # {(ns, id): dict(points, color, type, scale)}

    @property
    def empty(self):
        return self.state.size == 0

    def world_to_cell(self, xy):
        """Map world (N, 2) coordinates to fractional crop-cell coordinates."""
        xy = np.asarray(xy, dtype=np.float32).reshape(-1, 2)
        cells = (xy - np.asarray(self.origin, dtype=np.float32)) / self.resolution
        return cells - np.asarray(self.cell_min, dtype=np.float32)

    def markers_in(self, namespace):
        """All live markers of one namespace, ordered by marker id."""
        items = [(key[1], mk) for key, mk in self.markers.items() if key[0] == namespace]
        return [mk for _, mk in sorted(items, key=lambda kv: kv[0])]


class PlannerVisListener(Node):
    """Accumulates planner map + marker state from the planner's vis topics.

    Meant to be spun by its own executor on a background thread so that the
    evaluation loop's own callback cadence does not throttle map updates.
    """

    def __init__(
        self,
        map_size=(80.0, 80.0),
        resolution=0.05,
        node_name="apexnav_vis_listener",
    ):
        super().__init__(node_name)

        self.resolution = float(resolution)
        # SDFMap2D places its origin at -map_size / 2 and reports cell centers,
        # so this quantization reproduces the planner's own cell indices.
        self.origin = (-float(map_size[0]) / 2.0, -float(map_size[1]) / 2.0)
        self.nx = int(np.ceil(map_size[0] / self.resolution))
        self.ny = int(np.ceil(map_size[1] / self.resolution))

        self._lock = threading.Lock()
        self._state = np.zeros((self.ny, self.nx), dtype=np.uint8)
        self._value = np.full((self.ny, self.nx), -1.0, dtype=np.float32)
        self._markers = {}
        self._bbox = None  # [ix_min, iy_min, ix_max, iy_max] of known cells

        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=200,
        )

        self.create_subscription(
            PointCloud2, "/grid_map/occupied", self._make_state_cb(CELL_OCCUPIED), qos
        )
        self.create_subscription(
            PointCloud2,
            "/grid_map/occupied_inflate",
            self._make_state_cb(CELL_INFLATE),
            qos,
        )
        self.create_subscription(
            PointCloud2, "/grid_map/free", self._make_state_cb(CELL_FREE), qos
        )
        self.create_subscription(
            PointCloud2, "/grid_map/unknown", self._make_state_cb(CELL_UNKNOWN), qos
        )
        self.create_subscription(
            PointCloud2, "/grid_map/value_map", self._value_map_cb, qos
        )
        self.create_subscription(Marker, "/planning_vis/frontier", self._marker_cb, qos)
        self.create_subscription(
            Marker, "/planning_vis/viewpoints", self._marker_cb, qos
        )

    # ------------------------------------------------------------------
    # Accumulation
    # ------------------------------------------------------------------
    def _to_cells(self, xy):
        """World (N, 2) -> integer cell indices, dropping out-of-map points."""
        cells = np.floor(
            (xy - np.asarray(self.origin, dtype=np.float32)) / self.resolution
        ).astype(np.int32)
        inside = (
            (cells[:, 0] >= 0)
            & (cells[:, 0] < self.nx)
            & (cells[:, 1] >= 0)
            & (cells[:, 1] < self.ny)
        )
        return cells[inside], inside

    def _grow_bbox(self, cells):
        if cells.shape[0] == 0:
            return
        lo = cells.min(axis=0)
        hi = cells.max(axis=0)
        if self._bbox is None:
            self._bbox = [int(lo[0]), int(lo[1]), int(hi[0]), int(hi[1])]
        else:
            self._bbox[0] = min(self._bbox[0], int(lo[0]))
            self._bbox[1] = min(self._bbox[1], int(lo[1]))
            self._bbox[2] = max(self._bbox[2], int(hi[0]))
            self._bbox[3] = max(self._bbox[3], int(hi[1]))

    def _make_state_cb(self, state_value):
        def callback(msg):
            points = pointcloud2_to_xyzi(msg)
            if points.shape[0] == 0:
                return
            cells, _ = self._to_cells(points[:, :2])
            if cells.shape[0] == 0:
                return
            with self._lock:
                self._state[cells[:, 1], cells[:, 0]] = state_value
                if state_value != CELL_UNKNOWN:
                    # Only observed cells define the region worth rendering.
                    self._grow_bbox(cells)
                else:
                    # A cell that fell back to unknown must not keep a value.
                    self._value[cells[:, 1], cells[:, 0]] = -1.0

        return callback

    def _value_map_cb(self, msg):
        points = pointcloud2_to_xyzi(msg)
        if points.shape[0] == 0:
            return
        cells, inside = self._to_cells(points[:, :2])
        if cells.shape[0] == 0:
            return
        values = points[inside, 2]
        with self._lock:
            self._value[cells[:, 1], cells[:, 0]] = values

    def _marker_cb(self, msg):
        key = (msg.ns, msg.id)
        with self._lock:
            if msg.action == Marker.DELETEALL:
                self._markers.clear()
                return
            if msg.action == Marker.DELETE:
                self._markers.pop(key, None)
                return
            points = marker_points(msg)
            if points.shape[0] == 0:
                self._markers.pop(key, None)
                return
            self._markers[key] = {
                "points": points,
                "color": (msg.color.r, msg.color.g, msg.color.b, msg.color.a),
                "type": msg.type,
                "scale": msg.scale.x,
            }

    # ------------------------------------------------------------------
    # Access
    # ------------------------------------------------------------------
    def reset(self):
        """Drop all accumulated state, e.g. between episodes."""
        with self._lock:
            self._state.fill(CELL_UNKNOWN)
            self._value.fill(-1.0)
            self._markers.clear()
            self._bbox = None

    def _fixed_window(self, center, extent):
        """Cell bounds of a fixed-size world window centred on ``center``.

        The window keeps its exact cell size near the map border by sliding
        instead of truncating, so the rendered scale never changes between
        frames.
        """
        size_x = min(self.nx, max(2, int(round(extent[0] / self.resolution))))
        size_y = min(self.ny, max(2, int(round(extent[1] / self.resolution))))
        cx = int(np.floor((center[0] - self.origin[0]) / self.resolution))
        cy = int(np.floor((center[1] - self.origin[1]) / self.resolution))
        ix0 = min(max(0, cx - size_x // 2), self.nx - size_x)
        iy0 = min(max(0, cy - size_y // 2), self.ny - size_y)
        return ix0, iy0, ix0 + size_x, iy0 + size_y

    def snapshot(self, center=None, extent=None, pad_cells=20, min_cells=120):
        """Copy a region of the map for rendering.

        With ``center`` and ``extent`` (metres, scalar or (x, y)) the crop is a
        fixed-size window, which keeps the video's scale constant. Without
        them the crop tracks the explored area instead.

        Cropping inside the lock keeps the copy proportional to the window
        rather than to the full 80 m x 80 m grid.
        """
        with self._lock:
            if center is not None and extent:
                if np.isscalar(extent):
                    extent = (float(extent), float(extent))
                ix0, iy0, ix1, iy1 = self._fixed_window(center, extent)
            elif self._bbox is None:
                return PlannerSnapshot(
                    np.zeros((0, 0), dtype=np.uint8),
                    np.zeros((0, 0), dtype=np.float32),
                    (0, 0),
                    self.origin,
                    self.resolution,
                    {},
                )
            else:
                ix0, iy0, ix1, iy1 = self._bbox
                cx = (ix0 + ix1) // 2
                cy = (iy0 + iy1) // 2
                half_w = max((ix1 - ix0) // 2 + pad_cells, min_cells // 2)
                half_h = max((iy1 - iy0) // 2 + pad_cells, min_cells // 2)

                ix0 = max(0, cx - half_w)
                ix1 = min(self.nx, cx + half_w + 1)
                iy0 = max(0, cy - half_h)
                iy1 = min(self.ny, cy + half_h + 1)

            state = self._state[iy0:iy1, ix0:ix1].copy()
            value = self._value[iy0:iy1, ix0:ix1].copy()
            markers = {
                key: {
                    "points": mk["points"],
                    "color": mk["color"],
                    "type": mk["type"],
                    "scale": mk["scale"],
                }
                for key, mk in self._markers.items()
            }

        return PlannerSnapshot(
            state, value, (ix0, iy0), self.origin, self.resolution, markers
        )
