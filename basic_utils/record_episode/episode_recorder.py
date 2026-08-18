"""
Per-episode forensic recorder for ApexNav evaluation runs.

The evaluation loop already knows everything needed to reconstruct a failure -
agent pose, the action taken, the RGB observation, the per-frame detection
scores - but none of it survives the episode. This module buffers that data in
RAM as the episode runs and writes it to disk only when the episode ends in one
of the configured failure categories (false positives, by default). Successful
episodes cost nothing but a single summary line.

Typical wiring:

    recorder = EpisodeRecorder(cfg.get("record", {}), output_root)
    recorder.start(...)          # after env.reset()
    recorder.log_step(...)       # after every env.step()
    recorder.finish(result_text, metrics)   # after check_failure()

Author: generated for false-positive analysis of HM3D-v2 runs
"""

import json
import os
import shlex
import subprocess
import time

import cv2
import numpy as np

from params import RESULT_TYPES


def _git_commit():
    """Short hash of the working tree, or None outside a git checkout."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            capture_output=True,
            text=True,
            timeout=5,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        pass
    return None


def _scene_token(scene_id):
    """Shorten a habitat scene path to its scene name.

    "data/scene_datasets/hm3d_v0.2/val/00827-BAbdmeyTvMZ/BAbdmeyTvMZ.basis.glb"
    -> "00827-BAbdmeyTvMZ"
    """
    parts = str(scene_id).split("/")
    if len(parts) >= 2:
        return parts[-2]
    return os.path.splitext(os.path.basename(str(scene_id)))[0]


def _jsonable(value):
    """Convert numpy scalars/arrays into plain python for json.dump."""
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    return value


class EpisodeRecorder:
    """Buffers one episode at a time and flushes it on configured outcomes.

    Args:
        cfg: the ``record:`` block of the hydra config (dict-like, may be empty)
        output_root: directory the run already writes record.txt/videos into
    """

    def __init__(self, cfg, output_root):
        cfg = cfg or {}

        def _get(key, default):
            try:
                value = cfg.get(key, default)
            except AttributeError:
                value = getattr(cfg, key, default)
            return default if value is None else value

        self.enabled = bool(_get("enabled", True))
        self.capture_rgb = bool(_get("capture_rgb", True))
        self.capture_annotated_rgb = bool(_get("capture_annotated_rgb", True))
        self.capture_masks = bool(_get("capture_masks", True))
        self.jpeg_quality = int(_get("jpeg_quality", 85))
        self.max_buffered_steps = int(_get("max_buffered_steps", 600))
        self.summary_file = str(_get("summary_file", "episodes.jsonl"))

        self.output_root = output_root
        self.records_dir = os.path.join(output_root, "records")
        self.summary_path = os.path.join(output_root, self.summary_file)

        self.flush_on = self._parse_flush_on(_get("flush_on", ["false positive"]))

        self.git_commit = _git_commit()
        self.dataset = None
        self.planner_cfg = None

        self._reset_buffers()

        if self.enabled:
            os.makedirs(self.output_root, exist_ok=True)

    @staticmethod
    def _parse_flush_on(raw):
        """Validate the flush_on setting against the known result categories."""
        if isinstance(raw, str):
            if raw.strip().lower() == "all":
                return set(RESULT_TYPES)
            raw = [raw]
        categories = {str(item) for item in raw}
        if categories == {"all"}:
            return set(RESULT_TYPES)
        unknown = sorted(categories - set(RESULT_TYPES))
        if unknown:
            raise ValueError(
                f"record.flush_on contains unknown result types {unknown}. "
                f"Valid values are {RESULT_TYPES} or \"all\"."
            )
        return categories

    def set_context(self, dataset=None, planner_cfg=None):
        """Attach run-level context stamped into every record.

        planner_cfg carries the fusion arm (object.fusion_type etc.) read from
        the planner node; it is unknown until the ROS side is up, so it is set
        separately from construction.
        """
        if dataset is not None:
            self.dataset = dataset
        if planner_cfg is not None:
            self.planner_cfg = planner_cfg

    def _reset_buffers(self):
        self._active = False
        self._header = {}
        self._steps = []
        self._poses = []
        self._rgb = []
        self._rgb_annot = []
        self._masks = []
        self._dropped_frames = 0
        self._start_wall = None

    # ------------------------------------------------------------------
    # Episode lifecycle
    # ------------------------------------------------------------------

    def start(
        self,
        scene_id,
        episode_id,
        run_index,
        label,
        llm_answer,
        room,
        fusion_threshold,
        detector_cfg,
        start_pose,
        start_rgb=None,
    ):
        """Begin buffering a new episode. Any previous buffer is discarded."""
        self._reset_buffers()
        if not self.enabled:
            return

        self._active = True
        self._start_wall = time.time()
        self._header = {
            "scene_id": str(scene_id),
            "scene": _scene_token(scene_id),
            "episode_id": str(episode_id),
            "run_index": int(run_index),
            "target_label": str(label),
            "llm_answer": _jsonable(llm_answer),
            "room": str(room),
            "fusion_threshold": float(fusion_threshold),
            "detector": _jsonable(detector_cfg),
            "dataset": self.dataset,
            "git_commit": self.git_commit,
        }
        self._poses.append(tuple(float(v) for v in start_pose))
        self._append_frames(start_rgb, None)

    def log_step(
        self,
        step,
        action,
        pose,
        camera_pitch,
        rgb_raw=None,
        rgb_annotated=None,
        score_list=None,
        label_list=None,
        itm=None,
        distance_to_goal=None,
        final_state=None,
        expl_result=None,
        object_fusion=None,
        object_masks=None,
    ):
        """Record one simulator step. Cheap enough to call unconditionally."""
        if not self._active:
            return

        scores = list(score_list or [])
        labels = list(label_list or [])
        masks = list(object_masks or [])

        # Detector masks cannot be regenerated from a replay without re-running
        # the VLM servers, so they are stored; PNG keeps them lossless and small
        mask_names = self._encode_masks(step, masks)

        detections = [
            {"score": float(s), "label_idx": int(l)} for s, l in zip(scores, labels)
        ]
        for idx, name in enumerate(mask_names):
            if idx < len(detections):
                detections[idx]["mask"] = name

        self._steps.append(
            {
                "step": int(step),
                "action": int(action),
                "pose": [float(v) for v in pose],
                "camera_pitch": float(camera_pitch),
                "itm": None if itm is None else float(itm),
                "distance_to_goal": (
                    None if distance_to_goal is None else float(distance_to_goal)
                ),
                "num_detections": len(detections),
                "detections": detections,
                # FSM verdict for this step: final_state is params.FINAL_RESULT,
                # expl_result is params.EXPL_RESULT. The step where final_state
                # first becomes SEARCH_OBJECT is where the planner committed
                "final_state": None if final_state is None else int(final_state),
                "expl_result": None if expl_result is None else int(expl_result),
                # Fused per-cluster confidence from the planner's object map,
                # alongside the acceptance gate that applied to it
                "object_fusion": object_fusion,
            }
        )
        self._poses.append(tuple(float(v) for v in pose))
        self._append_frames(rgb_raw, rgb_annotated)

    def _encode_masks(self, step, masks):
        """PNG-encode this step's detector masks, returning their filenames."""
        if not self.capture_masks or not masks:
            return []
        if len(self._rgb) >= self.max_buffered_steps:
            return []

        names = []
        for idx, mask in enumerate(masks):
            if mask is None:
                continue
            arr = np.asarray(mask)
            if arr.ndim == 3:
                arr = arr[:, :, 0]
            # PNG is lossless, so 0/1 masks survive without rescaling
            ok, buf = cv2.imencode(".png", arr.astype(np.uint8))
            if not ok:
                continue
            name = f"{int(step):06d}_{idx}.png"
            self._masks.append((name, buf.tobytes()))
            names.append(name)
        return names

    def _append_frames(self, rgb_raw, rgb_annotated):
        """JPEG-encode frames now so the buffer holds bytes, not arrays."""
        if len(self._rgb) >= self.max_buffered_steps:
            self._dropped_frames += 1
            return
        if self.capture_rgb:
            self._rgb.append(self._encode(rgb_raw))
        if self.capture_annotated_rgb:
            self._rgb_annot.append(self._encode(rgb_annotated))

    def _encode(self, rgb):
        if rgb is None:
            return None
        bgr = np.asarray(rgb)[:, :, ::-1]
        ok, buf = cv2.imencode(
            ".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
        )
        return buf.tobytes() if ok else None

    def finish(self, result_text, metrics):
        """Close the episode: always summarize, write artifacts only on a hit.

        Returns the record directory when one was written, else None.
        """
        if not self._active:
            return None

        summary = self._build_summary(result_text, metrics)
        self._append_summary(summary)

        record_dir = None
        if result_text in self.flush_on:
            try:
                record_dir = self._write_record(summary, metrics)
            except Exception as exc:  # never let recording kill a long run
                print(f"[EpisodeRecorder] failed to write record: {exc}")

        self._reset_buffers()
        return record_dir

    def discard(self):
        """Drop the buffer without writing anything."""
        self._reset_buffers()

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    def _build_summary(self, result_text, metrics):
        summary = {
            "run_index": self._header.get("run_index"),
            "scene": self._header.get("scene"),
            "episode_id": self._header.get("episode_id"),
            "target_label": self._header.get("target_label"),
            "result": result_text,
            "count_steps": len(self._steps),
            "elapsed_s": round(time.time() - self._start_wall, 2),
            "fusion_threshold": self._header.get("fusion_threshold"),
            "planner": self.planner_cfg,
        }
        for key in (
            "success",
            "spl",
            "soft_spl",
            "distance_to_goal",
            "pass_object",
            "near_object",
            "final_state",
            "expl_result",
        ):
            if key in metrics:
                summary[key] = _jsonable(metrics[key])
        return summary

    def _append_summary(self, summary):
        try:
            os.makedirs(os.path.dirname(self.summary_path) or ".", exist_ok=True)
            with open(self.summary_path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(summary) + "\n")
        except Exception as exc:
            print(f"[EpisodeRecorder] failed to append summary: {exc}")

    def _record_dir(self, result_text):
        name = "epi{run}_{scene}_{eid}_{label}".format(
            run=self._header["run_index"],
            scene=self._header["scene"],
            eid=self._header["episode_id"],
            label=_slug(self._header["target_label"]),
        )
        return os.path.join(self.records_dir, result_text, name)

    def _write_record(self, summary, metrics):
        record_dir = self._record_dir(summary["result"])
        os.makedirs(record_dir, exist_ok=True)

        meta = dict(self._header)
        meta.update(summary)
        meta["planner"] = self.planner_cfg
        meta["dropped_frames"] = self._dropped_frames
        meta["metrics"] = _jsonable(dict(metrics))
        meta["replay"] = {
            "live": (
                f"python habitat_evaluation.py --dataset {self.dataset or '<dataset>'} "
                f"test_epi_num={self._header['run_index']}"
            ),
            # result categories contain spaces, so the path needs quoting
            "offline": (
                "python -m basic_utils.record_episode.replay_episode "
                f"{shlex.quote(record_dir)}"
            ),
        }
        with open(os.path.join(record_dir, "meta.json"), "w", encoding="utf-8") as handle:
            json.dump(meta, handle, indent=2)

        with open(os.path.join(record_dir, "steps.jsonl"), "w", encoding="utf-8") as handle:
            for entry in self._steps:
                handle.write(json.dumps(entry) + "\n")

        poses = np.asarray(self._poses, dtype=np.float32).reshape(-1, 3)
        np.save(os.path.join(record_dir, "trajectory.npy"), poses)

        self._write_frames(record_dir, "rgb", self._rgb)
        self._write_frames(record_dir, "rgb_annot", self._rgb_annot)
        self._write_masks(record_dir)

        return record_dir

    def _write_masks(self, record_dir):
        """Write the buffered detector masks under masks/."""
        if not self._masks:
            return
        target = os.path.join(record_dir, "masks")
        os.makedirs(target, exist_ok=True)
        for name, payload in self._masks:
            with open(os.path.join(target, name), "wb") as handle:
                handle.write(payload)

    @staticmethod
    def _write_frames(record_dir, subdir, frames):
        if not any(frame is not None for frame in frames):
            return
        target = os.path.join(record_dir, subdir)
        os.makedirs(target, exist_ok=True)
        for idx, frame in enumerate(frames):
            if frame is None:
                continue
            with open(os.path.join(target, f"{idx:06d}.jpg"), "wb") as handle:
                handle.write(frame)


def _slug(label):
    """Filename-safe token for a target category (mirrors slugify_label)."""
    primary = str(label).split("|")[0].strip().lower()
    slug = "".join(c if c.isalnum() else "_" for c in primary).strip("_")
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug or "object"
