"""
ApexNav Video Frame Renderer

Rasterizes the planner state collected by :mod:`planner_vis_listener` into the
paper-style top-down figure: value map over the occupancy grid, inflated
obstacles, object clusters, frontier clusters, the TSP tour, the planned path,
the traveled trajectory and the agent pose, with the annotated RGB observation
inset in the corner.

Everything is drawn with OpenCV on RGB uint8 arrays so the output can be handed
straight to habitat's ``images_to_video``.

Author: ApexNav contributors
"""

import cv2
import numpy as np

from basic_utils.visualization.planner_vis_listener import (
    CELL_FREE,
    CELL_INFLATE,
    CELL_OCCUPIED,
    NS_DORMANT_FRONTIER,
    NS_FRONTIER,
    NS_LOCAL_POINT,
    NS_NEXT_PATH,
    NS_OBJECT,
    NS_TSP_TOUR,
)

# ---------------------------------------------------------------------------
# Palette (RGB)
# ---------------------------------------------------------------------------
COLOR_BACKDROP = (18, 20, 26)
COLOR_UNKNOWN = (208, 212, 219)
COLOR_FREE = (255, 255, 255)
COLOR_INFLATE = (176, 181, 189)
COLOR_OCCUPIED = (38, 42, 51)
COLOR_DORMANT = (137, 143, 152)
# Object clusters carry the planner's per-label colour, so they are set apart
# from the frontier rainbow by a dark outline and a centroid ring instead.
COLOR_OBJECT_EDGE = (26, 28, 34)
COLOR_TSP = (34, 190, 96)
COLOR_PATH = (232, 62, 62)
COLOR_GOAL = (58, 128, 246)
COLOR_TRAIL = (2, 111, 197)
COLOR_AGENT = (255, 165, 30)
COLOR_FOV = (255, 214, 120)
COLOR_TEXT = (238, 240, 245)
COLOR_TEXT_DIM = (158, 165, 178)

# "turbo" matches the colour map the RViz / Foxglove layouts already use for
# /grid_map/value_map; the warm ramps read better in printed figures.
_CMAPS = {
    "turbo": getattr(cv2, "COLORMAP_TURBO", cv2.COLORMAP_JET),
    "jet": cv2.COLORMAP_JET,
    "inferno": cv2.COLORMAP_INFERNO,
    "magma": cv2.COLORMAP_MAGMA,
    "plasma": cv2.COLORMAP_PLASMA,
    "viridis": cv2.COLORMAP_VIRIDIS,
    "hot": cv2.COLORMAP_HOT,
}

FONT = cv2.FONT_HERSHEY_DUPLEX
HEADER_H = 40


class FrameStyle:
    """Layout and layer toggles for the rendered frame."""

    def __init__(self, cfg=None):
        cfg = cfg if cfg is not None else {}

        def get(key, default):
            try:
                value = cfg[key] if key in cfg else default
            except TypeError:
                value = getattr(cfg, key, default)
            return default if value is None else value

        size = get("frame_size", [1280, 720])
        self.width = int(size[0])
        self.height = int(size[1])
        self.inset_width_frac = float(get("inset_width_frac", 0.28))
        self.max_px_per_cell = float(get("max_px_per_cell", 8.0))
        self.show_value_map = bool(get("show_value_map", True))
        self.show_inflation = bool(get("show_inflation", True))
        self.show_frontiers = bool(get("show_frontiers", True))
        self.show_objects = bool(get("show_objects", True))
        self.show_tsp_tour = bool(get("show_tsp_tour", True))
        self.show_planned_path = bool(get("show_planned_path", True))
        self.show_trail = bool(get("show_trail", True))
        self.show_fov = bool(get("show_fov", True))
        self.show_rgb_inset = bool(get("show_rgb_inset", True))
        self.show_gt_map = bool(get("show_gt_map", False))
        self.show_legend = bool(get("show_legend", True))
        self.camera_hfov_deg = float(get("camera_hfov_deg", 79.0))
        self.camera_range_m = float(get("camera_range_m", 5.0))
        self.agent_size_m = float(get("agent_size_m", 0.45))
        self.value_floor = float(get("value_floor", 0.02))
        self.value_cmap = _CMAPS.get(str(get("value_cmap", "turbo")), _CMAPS["turbo"])
        # Value-map normalization: "auto" stretches the colour ramp to the
        # highest value seen so far this episode (values usually top out
        # around 0.3, so a fixed 0..1 ramp would never reach red).
        self.value_norm = str(get("value_norm", "auto"))
        self.value_vmax = float(get("value_vmax", 1.0))
        self.value_vmax_floor = float(get("value_vmax_floor", 0.05))
        self._auto_vmax = 0.0
        # Fixed world window in metres; 0 falls back to fitting the explored area
        self.map_extent_m = float(get("map_extent_m", 24.0))
        self.map_follow_margin = float(get("map_follow_margin", 0.25))

    def reset_value_scale(self):
        """Forget the running value-map maximum, e.g. between episodes."""
        self._auto_vmax = 0.0


# ---------------------------------------------------------------------------
# Small drawing helpers
# ---------------------------------------------------------------------------
def _blend(canvas, mask, color, alpha=1.0):
    """Alpha-blend a flat color into ``canvas`` wherever ``mask`` is set."""
    if not mask.any():
        return
    if alpha >= 1.0:
        canvas[mask] = color
        return
    patch = canvas[mask].astype(np.float32)
    canvas[mask] = (patch * (1.0 - alpha) + np.array(color, np.float32) * alpha).astype(
        np.uint8
    )


def _to_rgb255(color):
    """visualization_msgs color (0-1 floats) -> RGB uint8 tuple."""
    return tuple(int(np.clip(c, 0.0, 1.0) * 255) for c in color[:3])


def _stamp_cells(canvas, cells, color, half=0):
    """Paint marker cells onto the cell-resolution canvas.

    ``cells`` are integer (N, 2) local crop indices in (ix, iy); ``half``
    thickens each cell into a (2 * half + 1) square, matching the chunkier
    cube markers RViz shows.
    """
    h, w = canvas.shape[:2]
    if cells.shape[0] == 0:
        return
    for dx in range(-half, half + 1):
        for dy in range(-half, half + 1):
            xs = cells[:, 0] + dx
            ys = cells[:, 1] + dy
            keep = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
            canvas[h - 1 - ys[keep], xs[keep]] = color


def _text(img, label, org, scale=0.5, color=COLOR_TEXT, thickness=1, shadow=True):
    if shadow:
        cv2.putText(
            img,
            label,
            (org[0] + 1, org[1] + 1),
            FONT,
            scale,
            (0, 0, 0),
            thickness + 1,
            cv2.LINE_AA,
        )
    cv2.putText(img, label, org, FONT, scale, color, thickness, cv2.LINE_AA)


def _panel(img, x0, y0, x1, y1, color=(0, 0, 0), alpha=0.55):
    """Darken a rectangle so overlaid text stays readable on any map colour."""
    x0, y0 = max(0, x0), max(0, y0)
    x1 = min(img.shape[1], x1)
    y1 = min(img.shape[0], y1)
    if x1 <= x0 or y1 <= y0:
        return
    region = img[y0:y1, x0:x1].astype(np.float32)
    img[y0:y1, x0:x1] = (
        region * (1.0 - alpha) + np.array(color, np.float32) * alpha
    ).astype(np.uint8)


# ---------------------------------------------------------------------------
# Map panel
# ---------------------------------------------------------------------------
def value_scale(snap, style):
    """Upper end of the value-map colour ramp for this frame.

    In "auto" mode this is the running maximum of the observed values (99.5th
    percentile, so a single outlier cell cannot wash out the whole ramp). It
    only ever grows within an episode, which keeps colours from flickering
    between frames as the value map fills in.
    """
    if style.value_norm != "auto":
        return max(style.value_vmax, 1e-6)

    if not snap.empty:
        values = snap.value[snap.value >= style.value_floor]
        if values.size:
            style._auto_vmax = max(
                style._auto_vmax, float(np.percentile(values, 99.5))
            )
    return max(style._auto_vmax, style.value_vmax_floor)


def _render_cell_canvas(snap, style, vmax):
    """Build the map image at map resolution (one pixel per grid cell)."""
    state = snap.state
    h, w = state.shape
    canvas = np.empty((h, w, 3), dtype=np.uint8)
    canvas[:] = COLOR_UNKNOWN

    flip = lambda grid: grid[::-1, :]  # noqa: E731 - row 0 must be max-y

    free = flip(state == CELL_FREE)
    canvas[free] = COLOR_FREE

    if style.show_value_map:
        value = flip(snap.value)
        valued = free & (value >= style.value_floor)
        if valued.any():
            scaled = np.clip(value / vmax, 0.0, 1.0)
            colored = cv2.applyColorMap(
                (scaled * 255.0).astype(np.uint8), style.value_cmap
            )[:, :, ::-1]
            # Fade out with the value itself so near-zero cells stay readable
            # as plain free space instead of tinting the whole floor plan.
            alpha = (0.15 + 0.70 * scaled)[valued][:, None]
            base = canvas[valued].astype(np.float32)
            canvas[valued] = (
                base * (1.0 - alpha) + colored[valued].astype(np.float32) * alpha
            ).astype(np.uint8)

    if style.show_inflation:
        _blend(canvas, flip(state == CELL_INFLATE), COLOR_INFLATE)

    canvas[flip(state == CELL_OCCUPIED)] = COLOR_OCCUPIED
    return canvas


def _marker_cells(snap, marker):
    """Marker world points -> integer local crop cell indices."""
    cells = snap.world_to_cell(marker["points"])
    return np.floor(cells).astype(np.int32)


def _stamp_markers(canvas, snap, style):
    """Overlay the cube-list layers (objects, frontiers) at cell resolution.

    Frontier and object clusters both arrive with rainbow marker colours, so
    object blobs are outlined in near-black to tell the two layers apart at a
    glance; frontiers stay as flat unoutlined boundary strips.
    """
    h, w = canvas.shape[:2]

    if style.show_frontiers:
        for marker in snap.markers_in(NS_DORMANT_FRONTIER):
            _stamp_cells(canvas, _marker_cells(snap, marker), COLOR_DORMANT, half=0)
        for marker in snap.markers_in(NS_FRONTIER):
            _stamp_cells(
                canvas, _marker_cells(snap, marker), _to_rgb255(marker["color"]), half=0
            )

    if style.show_objects:
        for marker in snap.markers_in(NS_OBJECT):
            mask = np.zeros((h, w), dtype=np.uint8)
            _stamp_cells(mask, _marker_cells(snap, marker), 255, half=1)
            if not mask.any():
                continue
            canvas[mask > 0] = _to_rgb255(marker["color"])
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(canvas, contours, -1, COLOR_OBJECT_EDGE, 2)


def _object_centroids(snap):
    """World-frame centroid of every object cluster."""
    centroids = []
    for marker in snap.markers_in(NS_OBJECT):
        points = marker["points"]
        if points.shape[0]:
            centroids.append(points.mean(axis=0))
    return centroids


def _line_segments(snap, marker, to_px):
    """LINE_LIST marker -> list of pixel-space (p0, p1) pairs."""
    px = to_px(marker["points"])
    return [
        ((int(px[i][0]), int(px[i][1])), (int(px[i + 1][0]), int(px[i + 1][1])))
        for i in range(0, len(px) - 1, 2)
    ]


def _draw_vector_layers(img, snap, style, to_px, meta):
    """Draw paths, tour, goal, trail and the agent in panel pixel space."""
    if style.show_tsp_tour:
        for marker in snap.markers_in(NS_TSP_TOUR):
            for p0, p1 in _line_segments(snap, marker, to_px):
                cv2.line(img, p0, p1, COLOR_TSP, 3, cv2.LINE_AA)

    if style.show_planned_path:
        for marker in snap.markers_in(NS_NEXT_PATH):
            for p0, p1 in _line_segments(snap, marker, to_px):
                cv2.line(img, p0, p1, COLOR_PATH, 3, cv2.LINE_AA)

    if style.show_objects:
        # A ring on each object cluster, so they never read as one more
        # frontier blob however the planner happened to colour them.
        centroids = _object_centroids(snap)
        if centroids:
            for point in to_px(np.asarray(centroids, dtype=np.float32)):
                center = (int(point[0]), int(point[1]))
                cv2.circle(img, center, 9, COLOR_OBJECT_EDGE, 2, cv2.LINE_AA)
                cv2.circle(img, center, 12, (255, 255, 255), 1, cv2.LINE_AA)

    for marker in snap.markers_in(NS_LOCAL_POINT):
        for point in to_px(marker["points"]):
            center = (int(point[0]), int(point[1]))
            cv2.circle(img, center, 7, COLOR_GOAL, -1, cv2.LINE_AA)
            cv2.circle(img, center, 7, (255, 255, 255), 1, cv2.LINE_AA)

    trail = meta.get("trail")
    if style.show_trail and trail is not None and len(trail) > 1:
        points = to_px(np.asarray(trail, dtype=np.float32))
        cv2.polylines(img, [points.reshape(-1, 1, 2)], False, COLOR_TRAIL, 3, cv2.LINE_AA)

    pose = meta.get("pose")
    if pose is not None:
        _draw_agent(img, snap, style, to_px, pose)


def _draw_agent(img, snap, style, to_px, pose):
    """Agent pose as a heading triangle, optionally with its camera FOV wedge."""
    x, y, yaw = float(pose[0]), float(pose[1]), float(pose[2])
    rot = np.array(
        [[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]], dtype=np.float32
    )

    if style.show_fov:
        half_fov = np.deg2rad(style.camera_hfov_deg) / 2.0
        angles = np.linspace(-half_fov, half_fov, 24)
        arc = np.stack(
            [np.cos(angles), np.sin(angles)], axis=1
        ) * style.camera_range_m
        wedge = np.vstack([np.zeros((1, 2), np.float32), arc]) @ rot.T + (x, y)
        px_wedge = to_px(wedge).reshape(-1, 1, 2)
        overlay = img.copy()
        cv2.fillPoly(overlay, [px_wedge], COLOR_FOV)
        cv2.addWeighted(overlay, 0.10, img, 0.90, 0.0, img)
        cv2.polylines(img, [px_wedge], True, COLOR_FOV, 1, cv2.LINE_AA)

    # Triangle of unit length with its nose at +x, scaled to agent_size_m
    body = np.array([[0.64, 0.0], [-0.36, 0.40], [-0.36, -0.40]], dtype=np.float32)
    triangle = body * style.agent_size_m @ rot.T + (x, y)
    px = to_px(triangle).reshape(-1, 1, 2)
    cv2.fillPoly(img, [px], COLOR_AGENT, cv2.LINE_AA)
    cv2.polylines(img, [px], True, (255, 255, 255), 1, cv2.LINE_AA)


def _render_map_panel(frame, snap, style, area, vmax):
    """Draw the top-down map centered in ``area`` and return a world->px map.

    ``area`` is (x0, y0, x1, y1) in frame pixels; returns None when there is
    nothing mapped yet.
    """
    if snap.empty:
        return None

    canvas = _render_cell_canvas(snap, style, vmax)
    _stamp_markers(canvas, snap, style)

    ax0, ay0, ax1, ay1 = area
    area_w = max(1, ax1 - ax0)
    area_h = max(1, ay1 - ay0)

    h, w = canvas.shape[:2]
    scale = min(area_w / float(w), area_h / float(h))
    scale = min(scale, style.max_px_per_cell)
    out_w = max(1, int(w * scale))
    out_h = max(1, int(h * scale))
    resized = cv2.resize(canvas, (out_w, out_h), interpolation=cv2.INTER_NEAREST)

    off_x = ax0 + (area_w - out_w) // 2
    off_y = ay0 + (area_h - out_h) // 2
    frame[off_y : off_y + out_h, off_x : off_x + out_w] = resized

    def to_px(world_xy):
        cells = snap.world_to_cell(world_xy)
        xs = off_x + cells[:, 0] * scale
        ys = off_y + (h - cells[:, 1]) * scale
        return np.stack([xs, ys], axis=1).round().astype(np.int32)

    return to_px


# ---------------------------------------------------------------------------
# Overlays: inset, legend, colorbar, header
# ---------------------------------------------------------------------------
def _paste(frame, image, top_left, label=None, border=2, border_color=(222, 226, 234)):
    """Drop an inset with a hairline border and a caption above it."""
    x, y = top_left
    h, w = image.shape[:2]
    if y + h > frame.shape[0] or x + w > frame.shape[1] or x < 0 or y < 0:
        return
    frame[y : y + h, x : x + w] = image
    cv2.rectangle(
        frame,
        (x - border, y - border),
        (x + w + border - 1, y + h + border - 1),
        border_color,
        border,
    )
    if label:
        _text(frame, label, (x, y - 10), 0.45, COLOR_TEXT_DIM)


CAPTION_H = 34


def _column_insets(style, rgb, gt_map):
    """The (image, caption) pairs that go in the side column, top to bottom."""
    items = []
    if style.show_rgb_inset and rgb is not None and rgb.size:
        items.append((rgb, "RGB + detections"))
    if style.show_gt_map and gt_map is not None and gt_map.size:
        items.append((gt_map, "ground-truth map"))
    return items


def _draw_insets(frame, style, items, x, col_w, budget):
    """Stack the side-column insets, shrinking them to fit ``budget`` pixels.

    Returns the next free y coordinate. Images keep their aspect ratio and are
    centered in the column, so a narrower stack still lines up with the legend.
    """
    y = HEADER_H + 32
    if not items:
        return y

    def stack_height(width):
        return sum(
            int(width * img.shape[0] / img.shape[1]) + CAPTION_H for img, _ in items
        )

    width = col_w
    while width > 140 and stack_height(width) > budget:
        width -= 8

    for img, caption in items:
        height = int(width * img.shape[0] / img.shape[1])
        resized = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
        _paste(frame, resized, (x + (col_w - width) // 2, y), label=caption)
        y += height + CAPTION_H

    return y


def _draw_header(frame, style, meta):
    _panel(frame, 0, 0, style.width, HEADER_H, alpha=0.62)
    target = meta.get("target", "?")
    _text(frame, f"Target: {target}", (16, 27), 0.66, COLOR_TEXT)

    chips = []
    if meta.get("step") is not None:
        chips.append(f"step {meta['step']}")
    if meta.get("itm") is not None:
        chips.append(f"ITM {meta['itm']:.3f}")
    if meta.get("distance_to_goal") is not None:
        chips.append(f"d(goal) {meta['distance_to_goal']:.2f} m")
    if meta.get("state"):
        chips.append(str(meta["state"]))
    if chips:
        line = "   |   ".join(chips)
        size = cv2.getTextSize(line, FONT, 0.55, 1)[0]
        _text(frame, line, (style.width - size[0] - 16, 26), 0.55, COLOR_TEXT_DIM)


def _draw_colorbar(frame, style, x, y, vmax, width=190, height=12):
    ramp = np.linspace(0, 255, width, dtype=np.uint8)[None, :].repeat(height, axis=0)
    bar = cv2.applyColorMap(ramp, style.value_cmap)[:, :, ::-1]
    frame[y : y + height, x : x + width] = bar
    cv2.rectangle(frame, (x - 1, y - 1), (x + width, y + height), (255, 255, 255), 1)
    top = f"{vmax:.2f}"
    top_w = cv2.getTextSize(top, FONT, 0.42, 1)[0][0]
    _text(frame, "0", (x - 3, y + height + 15), 0.42, COLOR_TEXT_DIM)
    _text(frame, top, (x + width - top_w, y + height + 15), 0.42, COLOR_TEXT_DIM)
    _text(frame, "semantic value", (x, y - 7), 0.45, COLOR_TEXT_DIM)


LEGEND_ROW_H = 19


def _legend_entries(style):
    """Legend rows as (label, color).

    A color of "rainbow" / "rainbow_outlined" marks the multi-hue cluster
    layers, drawn as a hue strip - outlined for objects, matching the map.
    """
    entries = []
    if style.show_frontiers:
        entries.append(("frontier clusters", "rainbow"))
        entries.append(("dormant frontier", COLOR_DORMANT))
    if style.show_objects:
        entries.append(("object clusters", "rainbow_outlined"))
    if style.show_tsp_tour:
        entries.append(("TSP tour", COLOR_TSP))
    if style.show_planned_path:
        entries.append(("planned path", COLOR_PATH))
    entries.append(("next viewpoint", COLOR_GOAL))
    if style.show_trail:
        entries.append(("traveled path", COLOR_TRAIL))
    if style.show_inflation:
        entries.append(("inflated obstacle", COLOR_INFLATE))
    entries.append(("occupied", COLOR_OCCUPIED))
    entries.append(("unknown", COLOR_UNKNOWN))
    return entries


def _legend_height(style, entries):
    return len(entries) * LEGEND_ROW_H + (58 if style.show_value_map else 20)


def _draw_legend(frame, style, x0, y0, box_w, vmax):
    entries = _legend_entries(style)
    row_h = LEGEND_ROW_H
    box_h = _legend_height(style, entries)
    # Keep the legend inside the frame even when the insets above are tall.
    y0 = min(y0, style.height - box_h - 14)
    _panel(frame, x0, y0, x0 + box_w, y0 + box_h, alpha=0.62)

    y = y0 + 22
    for label, color in entries:
        if isinstance(color, str):
            # Multi-hue layers: show a hue strip instead of one swatch.
            for i in range(6):
                hue = int(180 * i / 6)
                patch = np.uint8([[[hue, 220, 235]]])
                rgb = cv2.cvtColor(patch, cv2.COLOR_HSV2RGB)[0, 0].tolist()
                cv2.rectangle(
                    frame, (x0 + 12 + i * 3, y - 9), (x0 + 14 + i * 3, y), rgb, -1
                )
            if color == "rainbow_outlined":
                cv2.rectangle(
                    frame, (x0 + 11, y - 10), (x0 + 30, y + 1), COLOR_OBJECT_EDGE, 2
                )
        else:
            cv2.rectangle(frame, (x0 + 12, y - 9), (x0 + 30, y), color, -1)
            cv2.rectangle(frame, (x0 + 12, y - 9), (x0 + 30, y), (90, 95, 105), 1)
        _text(frame, label, (x0 + 38, y), 0.44, COLOR_TEXT, shadow=False)
        y += row_h

    if style.show_value_map:
        _draw_colorbar(frame, style, x0 + 14, y + 12, vmax, width=box_w - 40)


def update_view_center(center, position, style):
    """Centre of the fixed-size map window for this frame.

    The window is pinned where the episode started and only slides once the
    agent approaches its edge, so the scene stays put under a moving agent
    instead of panning every step. Returns None when no fixed extent is set,
    which makes the renderer fit the explored area instead.
    """
    if style.map_extent_m <= 0.0:
        return None

    x, y = float(position[0]), float(position[1])
    if center is None:
        return (x, y)

    keep = (style.map_extent_m / 2.0) * (1.0 - style.map_follow_margin)
    return (
        min(max(center[0], x - keep), x + keep),
        min(max(center[1], y - keep), y + keep),
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def render_apexnav_frame(snapshot, rgb=None, meta=None, style=None, gt_map=None):
    """Compose one video frame from a planner snapshot and the RGB observation.

    Args:
        snapshot: :class:`PlannerSnapshot` from ``PlannerVisListener.snapshot()``.
        rgb: annotated RGB observation (H, W, 3) uint8, or None.
        meta: dict with optional keys ``pose`` (x, y, yaw), ``trail`` (N, 2),
            ``target``, ``step``, ``itm``, ``distance_to_goal``, ``state``.
        style: :class:`FrameStyle`.
        gt_map: habitat's ground-truth top-down map image, or None.

    Returns:
        (height, width, 3) uint8 RGB frame.
    """
    style = style or FrameStyle()
    meta = meta or {}

    frame = np.empty((style.height, style.width, 3), dtype=np.uint8)
    frame[:] = COLOR_BACKDROP

    # Two-column layout: the map gets everything left of the inset column.
    margin = 18
    inset_w = int(style.width * style.inset_width_frac)
    has_column = style.show_rgb_inset or style.show_gt_map or style.show_legend
    column_x = style.width - inset_w - margin
    map_area = (
        0,
        HEADER_H,
        column_x - margin if has_column else style.width,
        style.height,
    )

    vmax = value_scale(snapshot, style)
    to_px = _render_map_panel(frame, snapshot, style, map_area, vmax)
    if to_px is not None:
        _draw_vector_layers(frame, snapshot, style, to_px, meta)
    else:
        center_x = (map_area[0] + map_area[2]) // 2
        _text(
            frame,
            "waiting for planner map...",
            (center_x - 130, style.height // 2),
            0.7,
            COLOR_TEXT_DIM,
        )

    items = _column_insets(style, rgb, gt_map)
    legend_h = _legend_height(style, _legend_entries(style)) + 14 if style.show_legend else 0
    budget = style.height - (HEADER_H + 32) - 14 - legend_h
    next_y = _draw_insets(frame, style, items, column_x, inset_w, budget)
    if style.show_legend:
        _draw_legend(frame, style, column_x, next_y, inset_w, vmax)
    _draw_header(frame, style, meta)
    return frame
