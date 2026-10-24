"""GIF rendering for standalone SAM 3 false-positive verification."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from PIL import Image, ImageDraw


@dataclass
class VerificationFrame:
    """One analyzed frame ready to be included in a verification GIF."""

    source_path: Path
    annotated_image: Image.Image
    query: str
    instance_count: int
    scores: Iterable[float]


def _label_frame(frame: VerificationFrame, position: int, total: int) -> Image.Image:
    """Add compact SAM 3 review context above one annotated frame."""
    scores_text = ", ".join(f"{score:.3f}" for score in list(frame.scores)[:3]) or "none"
    header = (
        f"frame {frame.source_path.stem} ({position}/{total}) | "
        f"query: {frame.query} | SAM 3 instances: {frame.instance_count} | "
        f"scores: {scores_text}"
    )
    header_height = 30
    labelled = Image.new(
        "RGB",
        (frame.annotated_image.width, frame.annotated_image.height + header_height),
        (20, 24, 31),
    )
    labelled.paste(frame.annotated_image, (0, header_height))
    ImageDraw.Draw(labelled).text((8, 8), header, fill=(245, 245, 245))
    return labelled


def write_verification_gif(
    frames: Sequence[VerificationFrame], output_path: Path, fps: float
) -> Path:
    """Write SAM 3 annotated frames to a looping GIF."""
    if not frames:
        raise ValueError("At least one analyzed frame is required to write a GIF")
    if fps <= 0:
        raise ValueError("fps must be greater than zero")

    gif_frames = [
        _label_frame(frame, position, len(frames))
        for position, frame in enumerate(frames, start=1)
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    gif_frames[0].save(
        output_path,
        save_all=True,
        append_images=gif_frames[1:],
        duration=max(10, round(1000 / fps)),
        loop=0,
        optimize=False,
    )
    return output_path
