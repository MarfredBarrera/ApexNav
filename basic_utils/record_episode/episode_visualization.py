"""Visualizations for recorded episode replays.

The recorder stores RGB observations and detector masks separately so the replay
can regenerate the RGB stream without contacting the detector/VLM services.
This module combines those artifacts into a compact forensic GIF.
"""

import json
import os
import shutil
import subprocess

import cv2
import numpy as np
from PIL import Image


TARGET_STATE = 1  # params.FINAL_RESULT.SEARCH_OBJECT


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


def _draw_commit_map(frame, entry, commit):
    """Draw a compact world-frame inset identifying the committed cluster."""
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
    side, margin = min(156, frame.shape[0] - 48), 8
    left, top = frame.shape[1] - side - margin, 42
    cv2.rectangle(frame, (left, top), (left + side, top + side), (25, 25, 25), -1)
    cv2.rectangle(frame, (left, top), (left + side, top + side), (245, 245, 245), 1)

    def project(point):
        x = left + int((point[0] - xmin) / max(xmax - xmin, 1e-6) * side)
        y = top + side - int((point[1] - ymin) / max(ymax - ymin, 1e-6) * side)
        return x, y

    for cluster in clusters:
        cv2.circle(frame, project(tuple(map(float, cluster["centroid"][:2]))), 4, (90, 180, 255), -1)
    selected = project(tuple(map(float, centroid[:2])))
    cv2.circle(frame, selected, 8, (255, 70, 220), 2)
    cv2.circle(frame, selected, 4, (255, 70, 220), -1)
    if len(pose) >= 2:
        agent = project((float(pose[0]), float(pose[1])))
        cv2.drawMarker(frame, agent, (255, 255, 255), cv2.MARKER_TILTED_CROSS, 10, 2)
        cv2.line(frame, agent, selected, (255, 70, 220), 1, cv2.LINE_AA)
    cv2.putText(frame, "world target map", (left + 5, top + 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)


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


def create_decision_frame(record_dir, rgb_dir, output_path, steps=None):
    """Save the annotated RGB at the first recorded target-decision step."""
    if steps is None:
        with open(os.path.join(record_dir, "steps.jsonl"), "r", encoding="utf-8") as handle:
            steps = [json.loads(line) for line in handle if line.strip()]
    decision_step = find_target_decision_step(steps)
    if decision_step is None:
        return None
    frame_path = os.path.join(rgb_dir, f"{decision_step:06d}.jpg")
    bgr = cv2.imread(frame_path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(
            f"Decision frame RGB is missing: {frame_path}; "
            "the replay must save RGBs through the decision step"
        )
    with open(os.path.join(record_dir, "meta.json"), "r", encoding="utf-8") as handle:
        meta = json.load(handle)
    by_step = {int(entry["step"]): entry for entry in steps}
    _, gate_step = _decision_steps(steps)
    frame = _annotate_frame(
        cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), decision_step,
        by_step.get(decision_step), os.path.join(record_dir, "masks"),
        decision_step, gate_step, str(meta.get("target_label", "")),
        _semantic_label_names(meta), show_commit_map=False,
    )
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    cv2.imwrite(output_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    return output_path, decision_step
