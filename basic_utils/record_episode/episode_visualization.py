"""Visualizations for recorded episode replays.

The recorder stores RGB observations and detector masks separately so the replay
can regenerate the RGB stream without contacting the detector/VLM services.
This module combines those artifacts into a compact forensic GIF.
"""

import json
import math
import os
import shutil
import subprocess

import cv2
import numpy as np
from PIL import Image


TARGET_STATE = 1  # params.FINAL_RESULT.SEARCH_OBJECT
COMMIT_COLOR = (255, 70, 220)
PANEL_BACKGROUND = (18, 18, 20)
# A mask covering this fraction of the frame already shows the object well
# enough to identify, so larger views stop earning extra rank beyond it.
FULL_VIEW_AREA_FRACTION = 0.10
# How far a recorded detection may sit from the committed cluster's centroid
# and still be accepted as one of the observations that built it.
COMMIT_MATCH_DISTANCE_M = 0.75
# How close a detection must be to where the agent stopped to stand in for a
# missing commit: an episode ends with the agent parked at its chosen target.
STOP_PROXIMITY_M = 3.0


def _primary_label(label):
    """Use the first readable name when a category has detector aliases."""
    if label is None:
        return None
    return str(label).split("|")[0].strip() or None


def _semantic_label_names(meta):
    """Rebuild the detector label-index vocabulary stored in an episode."""
    names = {}
    target = _primary_label((meta or {}).get("target_label"))
    if target:
        names[0] = target

    similar = (meta or {}).get("llm_answer")
    if isinstance(similar, list):
        for index, label in enumerate(similar, start=1):
            name = _primary_label(label) if isinstance(label, str) else None
            if name:
                names[index] = name
    return names


def _decision_steps(steps):
    """Return the FSM decision step and confidence-gate step, if recorded."""
    fsm_step = next(
        (int(entry["step"]) for entry in steps if entry.get("final_state") == TARGET_STATE),
        None,
    )
    gate_step = None
    for entry in steps:
        fusion = entry.get("object_fusion") or {}
        for cluster in fusion.get("clusters", []):
            if cluster.get("is_confident") and cluster.get("best_label") == 0:
                gate_step = int(entry["step"])
                break
        if gate_step is not None:
            break
    return fsm_step, gate_step


def _commit_for_entry(entry):
    """Read a persisted commit, with a best-effort fallback for old records."""
    if not entry:
        return None
    if entry.get("planner_commit"):
        return entry["planner_commit"]
    fusion = entry.get("object_fusion") or {}
    candidates = [
        cluster for cluster in fusion.get("clusters", [])
        if cluster.get("is_confident") and cluster.get("best_label") == 0
    ]
    if not candidates:
        return None
    cluster = min(
        candidates,
        key=lambda item: (-float(item.get("confidence", 0.0)), -int(item.get("observation_num", 0))),
    )
    return {
        "cluster_id": cluster.get("id"),
        "confidence": cluster.get("confidence"),
        "observation_num": cluster.get("observation_num"),
        "centroid": cluster.get("centroid"),
        "selection": "reconstructed_from_fusion_snapshot",
    }


def _draw_commit_map(frame, entry, commit, rect=None):
    """Draw a compact world-frame inset identifying the committed cluster.

    ``rect`` is an optional ``(left, top, side)`` placement; without it the
    inset is tucked into the frame's top-right corner.
    """
    centroid = commit.get("centroid")
    if not isinstance(centroid, list) or len(centroid) < 2:
        return
    fusion = entry.get("object_fusion") or {}
    clusters = [
        cluster for cluster in fusion.get("clusters", [])
        if cluster.get("best_label") == 0 and len(cluster.get("centroid") or []) >= 2
    ]
    pose = entry.get("pose") or []
    points = [tuple(map(float, centroid[:2]))]
    points.extend(tuple(map(float, cluster["centroid"][:2])) for cluster in clusters)
    if len(pose) >= 2:
        points.append((float(pose[0]), float(pose[1])))
    xs, ys = zip(*points)
    pad = max(0.75, max(max(xs) - min(xs), max(ys) - min(ys)) * 0.15)
    xmin, xmax, ymin, ymax = min(xs) - pad, max(xs) + pad, min(ys) - pad, max(ys) + pad
    if rect is None:
        side = min(156, frame.shape[0] - 48)
        left, top = frame.shape[1] - side - 8, 42
    else:
        left, top, side = (int(value) for value in rect)
    cv2.rectangle(frame, (left, top), (left + side, top + side), (25, 25, 25), -1)
    cv2.rectangle(frame, (left, top), (left + side, top + side), (245, 245, 245), 1)

    def project(point):
        x = left + int((point[0] - xmin) / max(xmax - xmin, 1e-6) * side)
        y = top + side - int((point[1] - ymin) / max(ymax - ymin, 1e-6) * side)
        return x, y

    for cluster in clusters:
        cv2.circle(frame, project(tuple(map(float, cluster["centroid"][:2]))), 4, (90, 180, 255), -1)
    selected = project(tuple(map(float, centroid[:2])))
    cv2.circle(frame, selected, 8, COMMIT_COLOR, 2)
    cv2.circle(frame, selected, 4, COMMIT_COLOR, -1)
    if len(pose) >= 2:
        agent = project((float(pose[0]), float(pose[1])))
        cv2.drawMarker(frame, agent, (255, 255, 255), cv2.MARKER_TILTED_CROSS, 10, 2)
        cv2.line(frame, agent, selected, COMMIT_COLOR, 1, cv2.LINE_AA)
    cv2.putText(frame, "world target map", (left + 5, top + 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)


def _load_steps(record_dir, steps=None):
    """Return the caller's already-parsed steps, or read them from the record."""
    if steps is not None:
        return steps
    with open(os.path.join(record_dir, "steps.jsonl"), "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _detection_mask_path(masks_dir, step, index, detection):
    """Resolve one detection's stored mask, tolerating older name conventions."""
    for name in (detection.get("mask"), f"{int(step):06d}_{int(index)}.png"):
        if not name:
            continue
        path = os.path.join(masks_dir, name)
        if os.path.exists(path):
            return path
    return None


def _cluster_centroid_at(entry, cluster_id, default=None):
    """Centroid of one fused cluster as of a step, if it existed by then."""
    for cluster in (entry.get("object_fusion") or {}).get("clusters", []):
        centroid = cluster.get("centroid") or []
        if int(cluster.get("id", -1)) == cluster_id and len(centroid) >= 2:
            return float(centroid[0]), float(centroid[1])
    return default


def _view_quality(score, area_fraction, clipped=False):
    """Rank observations by how usable they are for judging object identity.

    A confident detection on a handful of distant pixels proves nothing to a
    human reviewer, so mask area dominates the ranking until the object fills a
    readable part of the frame; past that point the detector score breaks ties.
    A mask running off the frame edge shows only part of the object, so it
    yields to a whole view of comparable size.
    """
    readable = min(1.0, area_fraction / FULL_VIEW_AREA_FRACTION)
    # An absent score must still let area order the candidates.
    quality = readable * max(float(score) if score is not None else 0.0, 0.05)
    return quality * (0.7 if clipped else 1.0)


def _mask_touches_border(mask):
    """True when a mask runs off the frame, so the object is only partly shown."""
    binary = mask > 0
    return bool(
        binary[0].any() or binary[-1].any() or binary[:, 0].any() or binary[:, -1].any()
    )


def _final_pose(steps):
    """Planar pose of the last recorded step, i.e. where the episode ended."""
    for entry in reversed(steps):
        pose = entry.get("pose") or []
        if len(pose) >= 2:
            return float(pose[0]), float(pose[1])
    return None


def find_commit_step(steps):
    """Return the step whose commitment decides the episode's outcome.

    The FSM can enter SEARCH_OBJECT before any cluster is confident, and can
    leave it again, so its first step is not necessarily the commitment that
    ended the episode. The recorder scores an episode on its *last* recorded
    commit (see ``EpisodeRecorder._build_summary``), which is also the target
    the agent was driving at when it stopped, so the review panel follows it.
    Records that never stored a commit fall back to the FSM decision step.
    """
    committed = [int(entry["step"]) for entry in steps if entry.get("planner_commit")]
    if committed:
        return committed[-1]
    return find_target_decision_step(steps)


def _resolve_commit(steps, commit_step):
    """Best available description of what the planner committed to.

    Records written before ``planner_commit`` existed - and runs where the FSM
    switched a step after the fusion snapshot last reported the cluster as
    confident - carry no commit on that entry itself, so fall back to the most
    recent confident target cluster at or before that step.
    """
    by_step = {int(entry["step"]): entry for entry in steps}
    commit = _commit_for_entry(by_step.get(commit_step))
    if commit:
        return commit
    earlier = [entry for entry in steps if int(entry["step"]) <= commit_step]
    for entry in reversed(earlier):
        commit = _commit_for_entry(entry)
        if commit:
            return dict(
                commit,
                selection="reconstructed_from_last_confident_snapshot",
                snapshot_step=int(entry["step"]),
            )
    return None


def find_commit_evidence(record_dir, steps=None, max_match_distance_m=COMMIT_MATCH_DISTANCE_M):
    """Find the recorded observation that best shows the committed target.

    The FSM commit step is a poor picture of the commitment. The planner acts on
    a fused world cluster assembled from earlier views, so at the exact step the
    FSM switches to SEARCH_OBJECT the target is often off-camera and the frame
    carries no masks at all - or only masks of unrelated candidates.

    This walks every step up to that commitment, keeps the target detections
    whose world centroid matches the committed cluster, and returns the clearest
    of them. ``matched_by`` says how the mask was linked: ``cluster_centroid``
    is real evidence; ``stop_proximity`` and ``target_label`` are the weaker
    fallbacks used for records that stored no commit, and the review panel
    labels them as such.

    Returns None only when the record contains no target commitment at all. The
    returned mapping always carries ``commit_step`` and the ``step`` whose RGB
    the review panel needs; ``mask_path`` is None when no stored mask supports
    the commitment.
    """
    steps = _load_steps(record_dir, steps)
    commit_step = find_commit_step(steps)
    if commit_step is None:
        return None
    commit = _resolve_commit(steps, commit_step)
    evidence = {
        "commit_step": commit_step,
        "decision_step": find_target_decision_step(steps),
        "commit": commit,
        "step": commit_step,
        "detection": None,
        "detection_index": None,
        "mask_path": None,
        "area_fraction": None,
        "match_distance_m": None,
        "matched_by": "none",
        "observation_candidates": 0,
    }

    cluster_id = None
    committed_centroid = None
    if commit and commit.get("cluster_id") is not None:
        cluster_id = int(commit["cluster_id"])
        centroid = commit.get("centroid") or []
        if len(centroid) >= 2:
            committed_centroid = (float(centroid[0]), float(centroid[1]))

    masks_dir = os.path.join(record_dir, "masks")
    matched, unmatched = [], []
    for entry in steps:
        step = int(entry["step"])
        if step > commit_step:
            break
        # Cluster centroids drift as observations accumulate, so compare each
        # detection against the cluster as it stood at that step.
        reference = (
            None if cluster_id is None
            else _cluster_centroid_at(entry, cluster_id, committed_centroid)
        )
        for index, detection in enumerate(entry.get("detections", [])):
            if detection.get("label_idx") != 0:
                continue
            mask_path = _detection_mask_path(masks_dir, step, index, detection)
            if mask_path is None:
                continue
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                continue
            area = int((mask > 0).sum())
            if not area:
                continue
            world = detection.get("world_centroid") or []
            candidate = {
                "step": step,
                "detection": detection,
                "detection_index": index,
                "mask_path": mask_path,
                "area_fraction": area / float(mask.size),
                "match_distance_m": None,
                "world_centroid": (
                    (float(world[0]), float(world[1])) if len(world) >= 2 else None
                ),
            }
            candidate["quality"] = _view_quality(
                detection.get("score"),
                candidate["area_fraction"],
                _mask_touches_border(mask),
            )
            if reference is None or candidate["world_centroid"] is None:
                unmatched.append(candidate)
                continue
            distance = math.hypot(
                candidate["world_centroid"][0] - reference[0],
                candidate["world_centroid"][1] - reference[1],
            )
            if distance <= max_match_distance_m:
                candidate["match_distance_m"] = round(distance, 3)
                matched.append(candidate)
            else:
                unmatched.append(candidate)

    # Detections tied to the committed cluster are the only true evidence.
    # Records with no commit metadata fall back to where the agent came to rest:
    # an episode ends parked at the target it chose, so target detections there
    # are the likeliest stand-in. Anything else is a clearly-flagged last resort,
    # so an old record still yields a reviewable frame instead of a blank wall.
    pool, matched_by = matched, "cluster_centroid"
    if not pool:
        stop_pose = _final_pose(steps)
        near_stop = [
            candidate for candidate in unmatched
            if stop_pose is not None and candidate["world_centroid"] is not None
            and math.hypot(
                candidate["world_centroid"][0] - stop_pose[0],
                candidate["world_centroid"][1] - stop_pose[1],
            ) <= STOP_PROXIMITY_M
        ]
        pool, matched_by = (near_stop, "stop_proximity") if near_stop else (unmatched, "target_label")
    if not pool:
        return evidence
    best = max(pool, key=lambda item: item["quality"])
    evidence.update(
        {
            "step": best["step"],
            "detection": best["detection"],
            "detection_index": best["detection_index"],
            "mask_path": best["mask_path"],
            "area_fraction": best["area_fraction"],
            "match_distance_m": best["match_distance_m"],
            "matched_by": matched_by,
            "observation_candidates": len(pool),
        }
    )
    return evidence


def find_target_decision_step(steps):
    """Return the primary target-decision step, preferring the FSM signal."""
    fsm_step, gate_step = _decision_steps(steps)
    return fsm_step if fsm_step is not None else gate_step


def committed_target_confidence(steps):
    """Return confidence history for the cluster selected at first commitment.

    Cluster IDs are stable within an episode, so this follows the actual fused
    target the planner selected rather than a different cluster that becomes
    confident later. ``None`` means the selected cluster was absent from that
    step's fusion snapshot.
    """
    decision_step = find_target_decision_step(steps)
    if decision_step is None:
        return None
    decision_entry = next(
        (entry for entry in steps if int(entry.get("step", -1)) == decision_step),
        None,
    )
    commit = _commit_for_entry(decision_entry)
    if not commit or commit.get("cluster_id") is None:
        return None
    cluster_id = int(commit["cluster_id"])
    history = []
    gate_step = None
    threshold = None
    for entry in steps:
        fusion = entry.get("object_fusion") or {}
        if threshold is None and fusion.get("min_confidence") is not None:
            threshold = float(fusion["min_confidence"])
        cluster = next(
            (
                item for item in fusion.get("clusters", [])
                if int(item.get("id", -1)) == cluster_id
            ),
            None,
        )
        confidence = None if cluster is None else float(cluster.get("confidence", 0.0))
        history.append((int(entry["step"]), confidence))
        if gate_step is None and cluster and cluster.get("is_confident"):
            gate_step = int(entry["step"])
    return {
        "cluster_id": cluster_id,
        "decision_step": decision_step,
        "gate_step": gate_step,
        "threshold": threshold,
        "history": history,
    }


def create_commit_confidence_plot(record_dir, output_path, steps=None):
    """Render the committed target cluster's fused confidence as a PNG."""
    if steps is None:
        with open(os.path.join(record_dir, "steps.jsonl"), "r", encoding="utf-8") as handle:
            steps = [json.loads(line) for line in handle if line.strip()]
    trace = committed_target_confidence(steps)
    if trace is None or not any(value is not None for _, value in trace["history"]):
        return None

    width, height = 960, 480
    left, right, top, bottom = 72, 28, 72, 62
    plot_width, plot_height = width - left - right, height - top - bottom
    frame = np.full((height, width, 3), 250, dtype=np.uint8)
    steps_x = [step for step, _ in trace["history"]]
    x_min, x_max = min(steps_x), max(steps_x)
    x_span = max(x_max - x_min, 1)
    values = [value for _, value in trace["history"] if value is not None]
    y_max = max(1.0, max(values) * 1.08, float(trace["threshold"] or 0.0) * 1.08)

    def point(step, confidence):
        x = left + round((step - x_min) / x_span * plot_width)
        y = top + round((y_max - confidence) / y_max * plot_height)
        return int(x), int(y)

    for fraction in np.linspace(0.0, 1.0, 5):
        value = y_max * (1.0 - fraction)
        y = top + round(fraction * plot_height)
        cv2.line(frame, (left, y), (left + plot_width, y), (220, 220, 220), 1)
        cv2.putText(frame, f"{value:.2f}", (8, y + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (70, 70, 70), 1, cv2.LINE_AA)
    cv2.rectangle(frame, (left, top), (left + plot_width, top + plot_height), (70, 70, 70), 1)

    threshold = trace["threshold"]
    if threshold is not None and 0.0 <= threshold <= y_max:
        _, y = point(x_min, threshold)
        cv2.line(frame, (left, y), (left + plot_width, y), (55, 150, 55), 1, cv2.LINE_AA)
        cv2.putText(frame, f"gate {threshold:.2f}", (left + 6, max(top + 16, y - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (45, 120, 45), 1, cv2.LINE_AA)

    previous = None
    for step, confidence in trace["history"]:
        if confidence is None:
            previous = None
            continue
        current = point(step, confidence)
        if previous is not None:
            cv2.line(frame, previous, current, (195, 55, 185), 2, cv2.LINE_AA)
        cv2.circle(frame, current, 3, (195, 55, 185), -1, cv2.LINE_AA)
        previous = current

    for step, color, label in (
        (trace["gate_step"], (45, 150, 45), "confident"),
        (trace["decision_step"], (190, 55, 185), "commit"),
    ):
        if step is None:
            continue
        x, _ = point(step, 0.0)
        cv2.line(frame, (x, top), (x, top + plot_height), color, 1, cv2.LINE_AA)
        cv2.putText(frame, f"{label} {step}", (min(x + 4, width - 145), top + 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    cv2.putText(frame, f"Committed target cluster #{trace['cluster_id']} confidence", (left, 31),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (35, 35, 35), 2, cv2.LINE_AA)
    cv2.putText(frame, "episode step", (left + plot_width // 2 - 38, height - 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (55, 55, 55), 1, cv2.LINE_AA)
    cv2.putText(frame, str(x_min), (left - 5, height - 42),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (70, 70, 70), 1, cv2.LINE_AA)
    cv2.putText(frame, str(x_max), (left + plot_width - 16, height - 42),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (70, 70, 70), 1, cv2.LINE_AA)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    cv2.imwrite(output_path, frame)
    return output_path, trace


def _blend_mask(frame, mask, color, alpha):
    """Blend one binary mask into an RGB frame and outline it."""
    mask = np.asarray(mask) > 0
    if mask.shape != frame.shape[:2]:
        mask = cv2.resize(
            mask.astype(np.uint8), (frame.shape[1], frame.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
    if not mask.any():
        return
    overlay = np.asarray(color, dtype=np.float32)
    frame[mask] = (frame[mask].astype(np.float32) * (1.0 - alpha) + overlay * alpha).astype(
        np.uint8
    )
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(frame, contours, -1, color, 2)


def _put_banner(frame, text, color):
    """Draw a high-contrast status banner at the top of a frame."""
    height = 34
    cv2.rectangle(frame, (0, 0), (frame.shape[1], height), color, -1)
    cv2.putText(
        frame, text, (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
        (255, 255, 255), 2, cv2.LINE_AA,
    )


def _annotate_frame(
    rgb, step, entry, masks_dir, decision_step, gate_step, target_label,
    label_names=None, show_commit_map=True,
):
    frame = np.asarray(rgb).copy()
    detections = (entry or {}).get("detections", [])
    commit = _commit_for_entry(entry)
    committed_index = None if not commit else commit.get("matched_detection_index")
    mask_items = []
    used_mask_paths = set()
    for idx, detection in enumerate(detections):
        mask_name = detection.get("mask") or f"{step:06d}_{idx}.png"
        mask_path = os.path.join(masks_dir, mask_name)
        if not os.path.exists(mask_path):
            fallback = os.path.join(masks_dir, f"{step:06d}_{idx}.png")
            if os.path.exists(fallback):
                mask_name, mask_path = os.path.basename(fallback), fallback
        if not os.path.exists(mask_path):
            continue
        mask_items.append((idx, detection, mask_path))
        used_mask_paths.add(os.path.abspath(mask_path))

    # Older records can contain masks even when the corresponding detection
    # metadata was truncated or absent. Render those masks too.
    if os.path.isdir(masks_dir):
        prefix = f"{step:06d}_"
        for mask_name in sorted(os.listdir(masks_dir)):
            if not (mask_name.startswith(prefix) and mask_name.endswith(".png")):
                continue
            mask_path = os.path.join(masks_dir, mask_name)
            if os.path.abspath(mask_path) in used_mask_paths:
                continue
            try:
                idx = int(mask_name[len(prefix):-4])
            except ValueError:
                continue
            mask_items.append((idx, {}, mask_path))

    for idx, detection, mask_path in mask_items:
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        try:
            label_idx = int(detection.get("label_idx"))
        except (TypeError, ValueError):
            label_idx = None
        is_target = label_idx == 0
        is_committed = committed_index == idx
        # Target candidates are yellow; other candidates rotate through readable colors.
        palette = [(70, 210, 255), (130, 220, 120), (220, 150, 80), (190, 120, 230)]
        color = (255, 70, 220) if is_committed else ((255, 190, 0) if is_target else palette[idx % len(palette)])
        _blend_mask(frame, mask, color, 0.58 if is_committed else (0.48 if is_target else 0.32))
        score = detection.get("score")
        label = (label_names or {}).get(label_idx)
        if label is None:
            label = "target" if is_target else f"candidate {idx}"
        if is_committed:
            label = f"COMMITTED: {label}"
        if score is not None:
            label += f" {float(score):.2f}"
        ys, xs = np.where(mask > 0)
        if len(xs):
            cv2.putText(frame, label, (int(xs.min()), max(18, int(ys.min()) - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 2, cv2.LINE_AA)

    title = f"step {step} | {len(detections)} candidate object(s)"
    if target_label:
        title += f" | goal: {target_label}"
    cv2.putText(frame, title, (10, frame.shape[0] - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, title, (10, frame.shape[0] - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (30, 30, 30), 1, cv2.LINE_AA)

    if step == decision_step and commit:
        ground_truth_text = ""
        ground_truth = commit.get("ground_truth_instance")
        if ground_truth:
            verdict = "GT INSTANCE" if ground_truth.get("is_correct_instance") else "GT OTHER"
            ground_truth_text = " | {} {:.0%}".format(
                verdict, float(ground_truth.get("goal_overlap_fraction", 0.0))
            )
        else:
            # Older records have only centroid-to-goal geometry. It is shown
            # as context, never presented as an object-instance verdict.
            geometry = commit.get("ground_truth_geometry") or commit.get("ground_truth")
            if geometry:
                ground_truth_text = " | goal centroid {:.2f}m".format(
                    float(geometry.get("nearest_goal_distance_m", 0.0))
                )
        _put_banner(
            frame,
            "COMMIT #{id} | conf {confidence:.2f} | views {views}".format(
                id=commit.get("cluster_id", "?"),
                confidence=float(commit.get("confidence", 0.0)),
                views=commit.get("observation_num", "?"),
            ) + ground_truth_text,
            (130, 45, 120),
        )
        if show_commit_map:
            _draw_commit_map(frame, entry or {}, commit)
    elif step == gate_step and step != decision_step:
        _put_banner(frame, "CONFIDENCE GATE — target candidate became confident", (40, 150, 70))
    return frame


def _load_annotated_frames(record_dir, rgb_dir, steps=None):
    """Load replay RGBs with stored mask overlays and decision markers."""
    if steps is None:
        with open(os.path.join(record_dir, "steps.jsonl"), "r", encoding="utf-8") as handle:
            steps = [json.loads(line) for line in handle if line.strip()]
    with open(os.path.join(record_dir, "meta.json"), "r", encoding="utf-8") as handle:
        meta = json.load(handle)

    frame_paths = sorted(
        os.path.join(rgb_dir, name)
        for name in os.listdir(rgb_dir)
        if name.lower().endswith((".jpg", ".jpeg", ".png"))
    )
    if not frame_paths:
        raise FileNotFoundError(f"No RGB frames found in {rgb_dir}")
    by_step = {int(entry["step"]): entry for entry in steps}
    decision_step, gate_step = _decision_steps(steps)
    label_names = _semantic_label_names(meta)
    masks_dir = os.path.join(record_dir, "masks")
    frames = []
    for path in frame_paths:
        step = int(os.path.splitext(os.path.basename(path))[0])
        bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        frame = _annotate_frame(
            rgb, step, by_step.get(step), masks_dir, decision_step, gate_step,
            str(meta.get("target_label", "")), label_names,
        )
        frames.append(frame)
    if not frames:
        raise ValueError(f"RGB frames in {rgb_dir} could not be decoded")
    return frames, decision_step, gate_step


def create_episode_gif(record_dir, rgb_dir, output_path, steps=None, fps=4):
    """Create a GIF from replay-generated RGBs and recorded detector masks."""
    frames, decision_step, gate_step = _load_annotated_frames(record_dir, rgb_dir, steps)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    # Pillow writes GIF delays explicitly in milliseconds. This avoids imageio
    # versions that silently emit zero-delay frames when given ``duration``.
    pil_frames = [Image.fromarray(frame, mode="RGB") for frame in frames]
    pil_frames[0].save(
        output_path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=max(10, int(round(1000.0 / max(1, int(fps))))),
        loop=0,
        optimize=False,
    )
    return output_path, len(frames), decision_step, gate_step


def create_episode_mp4(record_dir, rgb_dir, output_path, steps=None, fps=4):
    """Create an MP4 from replay-generated RGBs and recorded detector masks."""
    frames, decision_step, gate_step = _load_annotated_frames(record_dir, rgb_dir, steps)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    height, width = frames[0].shape[:2]
    # H.264 + yuv420p is broadly supported by VS Code/web video players.
    # OpenCV's mp4v encoder creates MPEG-4 Part 2 files that those players
    # commonly reject, so stream raw RGB frames through FFmpeg instead.
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        try:
            import imageio_ffmpeg

            ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        except ImportError as exc:
            raise RuntimeError("FFmpeg or imageio-ffmpeg is required for MP4 output") from exc
    even_width, even_height = width - width % 2, height - height % 2
    command = [
        ffmpeg, "-y", "-f", "rawvideo", "-vcodec", "rawvideo",
        "-pix_fmt", "rgb24", "-s", f"{even_width}x{even_height}",
        "-r", str(max(1.0, float(fps))), "-i", "-", "-an",
        "-c:v", "libx264", "-preset", "medium", "-crf", "18",
        # Baseline/avc1 is accepted by Electron-based VS Code players that
        # reject the default High profile even when the container is MP4.
        "-profile:v", "baseline", "-level", "3.0", "-bf", "0",
        "-pix_fmt", "yuv420p", "-tag:v", "avc1", "-movflags", "+faststart",
        output_path,
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        for frame in frames:
            process.stdin.write(frame[:even_height, :even_width].astype(np.uint8).tobytes())
        process.stdin.close()
        stderr = process.stderr.read().decode("utf-8", errors="replace")
        return_code = process.wait()
    finally:
        if process.stdin and not process.stdin.closed:
            process.stdin.close()
        if process.poll() is None:
            process.kill()
            process.wait()
    if return_code != 0:
        raise RuntimeError(f"FFmpeg failed to write {output_path}: {stderr[-1000:]}")
    return output_path, len(frames), decision_step, gate_step


def _text_size(text, scale, thickness):
    return cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)[0]


def _draw_text(canvas, text, origin, scale, color, thickness=1):
    cv2.putText(canvas, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, color,
                thickness, cv2.LINE_AA)


def _draw_panel_caption(canvas, text, left, top, width, height=28):
    """Label a panel on a dimmed strip so it stays readable over bright pixels."""
    strip = canvas[top:top + height, left:left + width]
    strip[:] = (strip.astype(np.float32) * 0.35).astype(np.uint8)
    _draw_text(canvas, text, (left + 8, top + height - 9), 0.5, (245, 245, 245), 1)


def _mask_bbox(mask):
    """Pixel bounds of a binary mask, or None when it is empty."""
    ys, xs = np.where(mask > 0)
    if not len(xs):
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def _fit_mask(mask, frame):
    """Match a stored mask to the replayed frame's resolution."""
    if mask is None:
        return None
    if mask.shape[:2] != frame.shape[:2]:
        mask = cv2.resize(
            mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST
        )
    return mask


def _zoom_crop(rgb, mask, box_side):
    """Upscale the masked object, outlined and with context, for identification."""
    bbox = _mask_bbox(mask)
    if bbox is None:
        return None
    x0, y0, x1, y1 = bbox
    # Context around the object matters as much as the object: a bed is told
    # from a couch by the room around it.
    pad = int(0.2 * max(x1 - x0 + 1, y1 - y0 + 1)) + 8
    x0, y0 = max(0, x0 - pad), max(0, y0 - pad)
    x1 = min(rgb.shape[1] - 1, x1 + pad)
    y1 = min(rgb.shape[0] - 1, y1 + pad)
    crop = rgb[y0:y1 + 1, x0:x1 + 1].copy()
    contours, _ = cv2.findContours(
        (mask[y0:y1 + 1, x0:x1 + 1] > 0).astype(np.uint8),
        cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
    )
    cv2.drawContours(crop, contours, -1, COMMIT_COLOR, 2)
    scale = min(box_side / crop.shape[1], box_side / crop.shape[0])
    # Nearest-neighbour on magnification keeps the pixels honest rather than
    # inventing detail the detector never saw.
    return cv2.resize(
        crop,
        (max(1, int(crop.shape[1] * scale)), max(1, int(crop.shape[0] * scale))),
        interpolation=cv2.INTER_NEAREST if scale > 1.0 else cv2.INTER_AREA,
    )


def _commit_review_lines(evidence, goal):
    """Header lines: the goal first, then what was committed and on what basis."""
    commit = evidence.get("commit") or {}
    lines = [(f'GOAL: {goal}', 1.0, 2, (255, 255, 255))]

    summary = "COMMIT at step {commit}".format(commit=evidence["commit_step"])
    if commit:
        summary += " | cluster #{id} | fused conf {conf:.2f} | {views} view(s)".format(
            id=commit.get("cluster_id", "?"),
            conf=float(commit.get("confidence", 0.0)),
            views=commit.get("observation_num", "?"),
        )
    if evidence["mask_path"]:
        summary += " | mask shown from step {step}".format(step=evidence["step"])
        detection = evidence.get("detection") or {}
        details = []
        if detection.get("score") is not None:
            details.append("det {:.2f}".format(float(detection["score"])))
        if evidence.get("match_distance_m") is not None:
            details.append("{:.2f} m from cluster".format(evidence["match_distance_m"]))
        if details:
            summary += " ({})".format(", ".join(details))
    lines.append((summary, 0.55, 1, (225, 225, 225)))

    decision_step = evidence.get("decision_step")
    if decision_step is not None and decision_step != evidence["commit_step"]:
        # The FSM can flag a target before any cluster is confident; say so, so
        # the step numbers here can be reconciled with the episode GIF banners.
        lines.append((
            "FSM first entered the target state at step {}; the commitment scored "
            "for this episode is the one above".format(decision_step),
            0.46, 1, (170, 170, 175),
        ))

    ground_truth = commit.get("ground_truth_instance")
    if ground_truth:
        correct = bool(ground_truth.get("is_correct_instance"))
        lines.append((
            "GROUND TRUTH: committed mask is {} ({:.0%} of its pixels are goal pixels)".format(
                "the goal instance" if correct else "NOT the goal instance",
                float(ground_truth.get("goal_overlap_fraction", 0.0)),
            ),
            0.55, 1, (90, 220, 120) if correct else (255, 90, 90),
        ))
    else:
        geometry = commit.get("ground_truth_geometry") or commit.get("ground_truth")
        if geometry:
            lines.append((
                "GROUND TRUTH (geometry only, not an identity check): nearest goal "
                "centroid {:.2f} m away".format(
                    float(geometry.get("nearest_goal_distance_m", 0.0))
                ),
                0.52, 1, (200, 200, 200),
            ))

    if evidence["matched_by"] == "stop_proximity":
        lines.append((
            "NOTE: this record stores no commit, so the mask is the clearest target "
            "detection at the spot where the agent stopped - the object it drove to, "
            "not a verified member of a fused cluster",
            0.46, 1, (255, 200, 90),
        ))
    elif evidence["matched_by"] == "target_label":
        lines.append((
            "NOTE: the mask could not be linked to the committed cluster (the record "
            "lacks detection world centroids or a recoverable commit); it is the "
            "clearest target-labelled detection before the commit",
            0.46, 1, (255, 200, 90),
        ))
    elif evidence["matched_by"] == "none":
        lines.append((
            "NOTE: the record stores no target mask before the commit, so there is "
            "no visual evidence to review - only the fused world geometry below",
            0.46, 1, (255, 200, 90),
        ))
    return lines


def _render_commit_review(rgb, mask, evidence, goal, entry):
    """Lay out the goal, the committed mask, and a zoom of it on one canvas."""
    view = rgb.copy()
    if mask is not None:
        _blend_mask(view, mask, COMMIT_COLOR, 0.32)
        bbox = _mask_bbox(mask)
        if bbox:
            cv2.rectangle(view, (bbox[0], bbox[1]), (bbox[2], bbox[3]), COMMIT_COLOR, 2)
    height, width = view.shape[:2]

    margin, line_gap = 16, 12
    header_lines = _commit_review_lines(evidence, goal)
    header_height = margin + sum(
        _text_size(text, scale, thickness)[1] + line_gap
        for text, scale, thickness, _ in header_lines
    )
    footer_height = 46
    box_side = height
    canvas = np.full(
        (header_height + height + footer_height, margin * 3 + width + box_side, 3),
        PANEL_BACKGROUND, dtype=np.uint8,
    )

    baseline = margin
    for text, scale, thickness, color in header_lines:
        baseline += _text_size(text, scale, thickness)[1]
        _draw_text(canvas, text, (margin, baseline), scale, color, thickness)
        baseline += line_gap

    top = header_height
    canvas[top:top + height, margin:margin + width] = view
    box_left = margin * 2 + width
    cv2.rectangle(canvas, (box_left, top), (box_left + box_side, top + box_side),
                  (38, 38, 42), -1)

    # The captions sit inside each panel, so reserve their strip before placing
    # content that would otherwise be printed under the text.
    caption_height = 28
    inner_top = top + caption_height
    inner_side = box_side - caption_height - margin
    crop = None if mask is None else _zoom_crop(rgb, mask, inner_side)
    if crop is not None:
        offset_x = box_left + (box_side - crop.shape[1]) // 2
        offset_y = inner_top + (box_side - caption_height - crop.shape[0]) // 2
        canvas[offset_y:offset_y + crop.shape[0], offset_x:offset_x + crop.shape[1]] = crop
        box_caption = "COMMITTED OBJECT (zoom)"
    else:
        commit = evidence.get("commit") or {}
        if commit:
            _draw_commit_map(
                canvas, entry or {}, commit,
                rect=(box_left + (box_side - inner_side) // 2, inner_top, inner_side),
            )
        box_caption = "COMMITTED CLUSTER IN THE WORLD MAP"

    _draw_panel_caption(canvas, "step {} camera view".format(evidence["step"]),
                        margin, top, width, caption_height)
    _draw_panel_caption(canvas, box_caption, box_left, top, box_side, caption_height)

    prompt = 'Does the outlined object match the goal "{}"? If not, the episode is a real false positive.'.format(goal)
    _draw_text(canvas, prompt, (margin, canvas.shape[0] - 16), 0.58, (255, 214, 90), 1)
    cv2.rectangle(canvas, (margin - 1, top - 1), (margin + width, top + height),
                  (90, 90, 95), 1)
    return canvas


def create_decision_frame(record_dir, rgb_dir, output_path, steps=None, raw_output_path=None):
    """Save a one-glance review panel for the planner's committed target.

    The panel answers exactly one question - does the object the planner
    committed to match the episode goal? - by pairing the frame that best shows
    the committed mask with a zoom of that mask and the goal in large type. The
    FSM commit step itself is usually not that frame: the commitment rests on a
    fused cluster built from earlier views, so rendering the commit step showed
    unrelated candidates or no masks at all.

    ``raw_output_path`` additionally saves that same camera frame untouched - no
    overlay, no text - for use as a clean image in write-ups or as input to
    another model.

    Returns ``(output_path, evidence)``, where ``evidence`` is the mapping from
    :func:`find_commit_evidence`, or None when the record holds no target
    decision.
    """
    steps = _load_steps(record_dir, steps)
    evidence = find_commit_evidence(record_dir, steps)
    if evidence is None:
        return None
    with open(os.path.join(record_dir, "meta.json"), "r", encoding="utf-8") as handle:
        meta = json.load(handle)
    goal = _primary_label(meta.get("target_label")) or str(meta.get("target_label", ""))

    step = int(evidence["step"])
    frame_path = os.path.join(rgb_dir, f"{step:06d}.jpg")
    bgr = cv2.imread(frame_path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(
            f"Committed-target evidence frame is missing: {frame_path}; the replay "
            "must save the RGB of the step that best shows the committed mask"
        )
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    if raw_output_path:
        os.makedirs(os.path.dirname(raw_output_path) or ".", exist_ok=True)
        if os.path.splitext(raw_output_path)[1].lower() == os.path.splitext(frame_path)[1].lower():
            # Copy rather than re-encode: a second JPEG pass would soften exactly
            # the detail the reviewer is trying to judge.
            shutil.copyfile(frame_path, raw_output_path)
        else:
            cv2.imwrite(raw_output_path, bgr)
        evidence["raw_path"] = raw_output_path
    mask = None
    if evidence["mask_path"]:
        mask = _fit_mask(cv2.imread(evidence["mask_path"], cv2.IMREAD_GRAYSCALE), rgb)

    by_step = {int(entry["step"]): entry for entry in steps}
    panel = _render_commit_review(
        rgb, mask, evidence, goal, by_step.get(int(evidence["commit_step"]))
    )
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    cv2.imwrite(output_path, cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))
    return output_path, evidence
