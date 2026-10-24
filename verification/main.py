"""Run SAM 3 analysis on a recorded false-positive episode.

Run from the repository root:

    python -m verification.main
    python -m verification.main path/to/episode --query "red chair"

Settings normally live in ``verification/config.yaml``. Command-line values
override that file, which is useful for an individual review run.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

try:  # Supports both `python -m verification.main` and `python verification/main.py`.
    from .detector import ObjectDetector
    from .visualization import VerificationFrame, write_verification_gif
except ImportError:  # pragma: no cover - used only by direct script execution.
    from detector import ObjectDetector
    from visualization import VerificationFrame, write_verification_gif


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
CONFIG_KEYS = {
    "episode_dir",
    "query",
    "output",
    "device",
    "gpu_id",
    "checkpoint_path",
    "confidence_threshold",
    "max_instances",
    "fps",
    "max_frames",
}
DEFAULT_CONFIG_PATH = Path(__file__).with_name("config.yaml")
REPO_ROOT = Path(__file__).resolve().parent.parent


def _frame_sort_key(path: Path) -> tuple:
    """Sort numeric recorder frame names before any non-numeric names."""
    try:
        return (0, int(path.stem))
    except ValueError:
        return (1, path.name)


def discover_rgb_frames(episode_dir: Path) -> List[Path]:
    """Return RGB frames from an EpisodeRecorder directory in episode order."""
    rgb_dir = episode_dir / "rgb"
    if not rgb_dir.is_dir():
        raise FileNotFoundError(
            f"Expected recorded RGB frames in {rgb_dir}. "
            "Pass a directory produced by EpisodeRecorder."
        )
    frames = sorted(
        (path for path in rgb_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES),
        key=_frame_sort_key,
    )
    if not frames:
        raise FileNotFoundError(f"No RGB frames found in {rgb_dir}")
    return frames


def _sample_frames(frames: Sequence[Path], max_frames: Optional[int]) -> List[Path]:
    """Evenly sample an episode only when configuration asks to cap it."""
    if max_frames is None or len(frames) <= max_frames:
        return list(frames)
    if max_frames < 1:
        raise ValueError("max_frames must be at least 1")
    if max_frames == 1:
        return [frames[0]]
    last_index = len(frames) - 1
    indices = [round(index * last_index / (max_frames - 1)) for index in range(max_frames)]
    return [frames[index] for index in indices]


def _episode_query(episode_dir: Path, query_override: Optional[str]) -> str:
    """Read the recorded target label unless the caller supplied a prompt."""
    if query_override:
        return query_override.strip()
    meta_path = episode_dir / "meta.json"
    if not meta_path.is_file():
        raise FileNotFoundError(
            f"{meta_path} is required to infer the target query. Set query in "
            "config.yaml or pass --query."
        )
    with meta_path.open("r", encoding="utf-8") as handle:
        meta = json.load(handle)
    target = str(meta.get("target_label", "")).split("|")[0].strip()
    if not target:
        raise ValueError(f"{meta_path} has no usable target_label; set query")
    return target


def analyse_episode(
    episode_dir: Path,
    query: str,
    *,
    confidence_threshold: float,
    max_instances: Optional[int],
    device: Optional[str],
    gpu_id: Optional[int],
    checkpoint_path: Optional[Path],
    max_frames: Optional[int],
) -> List[VerificationFrame]:
    """Run SAM 3 over the selected recorded frames and return their results."""
    frame_paths = _sample_frames(discover_rgb_frames(episode_dir), max_frames)
    detector = ObjectDetector(
        checkpoint_path=checkpoint_path,
        device=device,
        gpu_id=gpu_id,
        confidence_threshold=confidence_threshold,
        max_instances=max_instances,
    )
    analyzed_frames = []
    for position, frame_path in enumerate(frame_paths, start=1):
        result = detector.detect(frame_path, query)
        analyzed_frames.append(
            VerificationFrame(
                source_path=frame_path,
                annotated_image=result.annotated_image,
                query=query,
                instance_count=result.count,
                scores=result.scores,
            )
        )
        print(
            f"[{position:>4}/{len(frame_paths)}] {frame_path.name}: "
            f"{result.count} SAM 3 match(es)"
        )
    return analyzed_frames


def _load_config(config_path: Path) -> Dict[str, Any]:
    """Load the verifier configuration and reject likely misspelled settings."""
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file does not exist: {config_path}")
    try:
        import yaml
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "PyYAML is required to read verification/config.yaml. It is included "
            "by the project's core Docker environment."
        ) from error
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Configuration must be a YAML mapping: {config_path}")
    unknown = set(config).difference(CONFIG_KEYS)
    if unknown:
        raise ValueError(f"Unknown configuration keys: {sorted(unknown)}")
    return config


def _output_path(episode_dir: Path, query: str, requested: Optional[Any]) -> Path:
    """Return the GIF path, giving every episode its own output directory.

    ``output`` names the directory that collects review runs; each episode gets
    a subdirectory named after its record, so verifying a second episode adds to
    the collection instead of overwriting the first one's GIF. A configured path
    that still ends in ``.gif`` keeps its filename and contributes its parent as
    the collecting directory.
    """
    safe_query = re.sub(r"[^A-Za-z0-9._-]+", "_", query.strip()).strip("_") or "target"
    filename = f"sam3_verification_{safe_query}.gif"
    if not requested:
        # Without a configured destination the GIF stays beside the record it
        # was made from, which is already an episode-specific directory.
        return episode_dir / filename
    # Relative paths resolve against the repository root, not the process
    # working directory. Under docker/docker-compose.yml the repo is the
    # bind mount at /workspace/ApexNav, so a repo-relative path is the only
    # one that reaches the host. An absolute host path such as
    # /home/<user>/codes/ApexNav/... is NOT that mount inside the
    # container: it silently lands in the container's overlay layer and the
    # GIF is invisible on the host.
    path = Path(requested).expanduser()
    if path.suffix.lower() == ".gif":
        path, filename = path.parent, path.name
    root = path if path.is_absolute() else REPO_ROOT / path
    return root / episode_dir.name / filename


def _setting(args: argparse.Namespace, config: Dict[str, Any], name: str) -> Any:
    """Return a CLI value when present, otherwise the configured value."""
    value = getattr(args, name)
    return value if value is not None else config.get(name)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run standalone SAM 3 verification over one recorded false-positive episode."
    )
    parser.add_argument("episode_dir", nargs="?", type=Path, help="Overrides config episode_dir")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="YAML settings file (default: verification/config.yaml)",
    )
    parser.add_argument("--query", help="Overrides config query / meta.json target_label")
    parser.add_argument(
        "--output",
        type=Path,
        help="Overrides config output directory; the GIF lands in a per-episode subdirectory",
    )
    parser.add_argument("--fps", type=float, help="Overrides config GIF frame rate")
    parser.add_argument("--confidence-threshold", type=float, dest="confidence_threshold")
    parser.add_argument("--device", help='Overrides config SAM 3 device, e.g. "cuda"')
    parser.add_argument("--gpu-id", type=int, dest="gpu_id", help="Overrides config CUDA GPU number")
    parser.add_argument("--checkpoint-path", type=Path, dest="checkpoint_path")
    parser.add_argument("--max-frames", type=int, dest="max_frames")
    parser.add_argument(
        "--max-instances",
        type=int,
        dest="max_instances",
        help="Overrides config cap on instances kept per frame",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    config = _load_config(args.config.expanduser().resolve())

    episode_value = _setting(args, config, "episode_dir")
    if not episode_value:
        raise ValueError("Set episode_dir in config.yaml or pass it as a positional argument")
    episode_dir = Path(episode_value).expanduser().resolve()
    if not episode_dir.is_dir():
        raise FileNotFoundError(f"Episode directory does not exist: {episode_dir}")

    query = _episode_query(episode_dir, _setting(args, config, "query"))
    # Use `is None` rather than truthiness: a configured 0.0 is a meaningful
    # (if unwise) threshold and must not silently fall back to the default.
    threshold_value = _setting(args, config, "confidence_threshold")
    confidence_threshold = float(threshold_value) if threshold_value is not None else 0.5
    fps = float(_setting(args, config, "fps") or 4.0)
    max_frames_value = _setting(args, config, "max_frames")
    max_frames = int(max_frames_value) if max_frames_value is not None else None
    max_instances_value = _setting(args, config, "max_instances")
    max_instances = int(max_instances_value) if max_instances_value is not None else None
    gpu_id_value = _setting(args, config, "gpu_id")
    gpu_id = int(gpu_id_value) if gpu_id_value is not None else None
    checkpoint_value = _setting(args, config, "checkpoint_path")
    checkpoint_path = Path(checkpoint_value) if checkpoint_value else None
    output_path = _output_path(episode_dir, query, _setting(args, config, "output")).resolve()

    analyzed_frames = analyse_episode(
        episode_dir,
        query,
        confidence_threshold=confidence_threshold,
        max_instances=max_instances,
        device=_setting(args, config, "device"),
        gpu_id=gpu_id,
        checkpoint_path=checkpoint_path,
        max_frames=max_frames,
    )
    written_path = write_verification_gif(analyzed_frames, output_path, fps)
    print(f"SAM 3 verification GIF written to {written_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
