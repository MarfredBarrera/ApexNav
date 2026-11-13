"""ApexNav's own 2D grid map (C++ SDFMap2D, built over the whole episode) as seen through the MapROS visualisation topics.

MapROS::visCallback (250 ms wall timer, same callback group as the step callback in sync_step mode) publishes PointCloud2
/grid_map/occupied, /grid_map/occupied_inflate, /grid_map/free, /grid_map/unknown over the cumulative updated box (md_->update_min_..max_),
in the planner frame (map_ros/frame_id), one point per 5 cm cell at the cell centre ((id + 0.5) * res + origin). The publishers
build a cloud only while someone subscribes, so subscribing is all it takes (no C++ change). Cell classes match the A*'s checks:
occupied = occupancy > min_occupancy_log, inflated = occupancy_buffer_inflate == 1 and not occupied (so occupied | inflated =
the planner's blocked set), free = known and not occupied/inflated, unknown = never observed and not inflated.
"""
import numpy as np

from lastmile import apex_astar as AA

TOPICS = ("occupied", "occupied_inflate", "free", "unknown")
RES = 0.05


def xy(msg):
    """(N, 2) float64 x, y of a PointCloud2 with float32 x at offset 0 and y at offset 4."""
    f = {x.name: x.offset for x in msg.fields}
    assert f.get("x") == 0 and f.get("y") == 4, f
    n = msg.width * msg.height
    if not n:
        return np.zeros((0, 2))
    raw = np.frombuffer(bytes(msg.data), np.uint8).reshape(n, msg.point_step)
    return np.ascontiguousarray(raw[:, :8]).view(np.float32).reshape(n, 2).astype(np.float64)


class TopicSource:
    """Caches the latest message of each /grid_map topic; ``grid(extra)`` builds an apex_astar.Grid in the planner frame."""

    def __init__(self, node, wait_s=4.0):
        import rclpy
        from rclpy.qos import QoSProfile, ReliabilityPolicy
        from sensor_msgs.msg import PointCloud2
        self.node, self.msgs, self.n_recv = node, {}, 0
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE)
        self.subs = [node.create_subscription(PointCloud2, "/grid_map/" + t, lambda m, t=t: self._cb(t, m), qos) for t in TOPICS]
        import time
        t0 = time.monotonic()
        while len(self.msgs) < len(TOPICS) and time.monotonic() - t0 < wait_s:
            rclpy.spin_once(node, timeout_sec=0.1)
        self.frame = None if "occupied" not in self.msgs else self.msgs["occupied"].header.frame_id
        self._parsed = {}

    def _cb(self, topic, msg):
        self.msgs[topic] = msg
        self.n_recv += 1

    def ready(self):
        return len(self.msgs) == len(TOPICS)

    def close(self):
        for s in self.subs:
            self.node.destroy_subscription(s)

    def grid(self, extra):
        """Grid over the clouds' bounding box plus ``extra`` (M, 2) planner-frame points (agent, object), padded 1 m."""
        import rclpy
        rclpy.spin_once(self.node, timeout_sec=0.0)
        if not self.ready():
            return None
        pts = {}
        for t in TOPICS:
            m = self.msgs[t]
            key = (t, id(m))
            if key not in self._parsed:
                self._parsed = {k: v for k, v in self._parsed.items() if k[0] != t}
                self._parsed[key] = xy(m)
            pts[t] = self._parsed[key]
        allp = np.vstack([pts[t] for t in TOPICS] + [np.asarray(extra, float).reshape(-1, 2)])
        lo = np.floor(allp.min(0) / RES).astype(int) - int(1.0 / RES)
        hi = np.floor(allp.max(0) / RES).astype(int) + int(1.0 / RES)
        shape = (hi[1] - lo[1] + 1, hi[0] - lo[0] + 1)
        layers = {}
        for t in TOPICS:
            a = np.zeros(shape, bool)
            if len(pts[t]):
                ij = np.floor(pts[t] / RES).astype(int) - lo
                a[ij[:, 1], ij[:, 0]] = True
            layers[t] = a
        occ, infl = layers["occupied"], layers["occupied_inflate"] & ~layers["occupied"]
        known = layers["free"] | occ | infl
        return AA.Grid(lo * RES, RES, occ, infl, known)
