"""Standalone SAM 3 object verification for collected episode frames.

This module deliberately does not use ApexNav's online YOLO, GroundingDINO, or
MobileSAM services.  It runs SAM 3 directly against an image and an
open-vocabulary text query, which makes it useful for examining saved false
positives after an episode has completed.
"""

from __future__ import annotations

import inspect
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
from PIL import Image, ImageDraw


ImageInput = Union[str, Path, Image.Image, np.ndarray]


@dataclass
class Detection:
    """One SAM 3 instance prediction in the original image coordinate system."""

    index: int
    confidence: float
    box_xyxy: Tuple[float, float, float, float]
    mask: np.ndarray
    mask_area_pixels: int
    mask_area_fraction: float


@dataclass
class ObjectDetectionResult:
    """The complete result of an :class:`ObjectDetector` query.

    ``segmented_image`` is the RGB source image with all non-target pixels set
    to black. ``annotated_image`` is RGB and contains a translucent mask,
    bounding box, instance number, SAM 3 confidence, and mask area for every
    detected target. ``masks``, ``combined_mask``, ``boxes_xyxy``, and
    ``scores`` are NumPy arrays ordered by descending SAM 3 confidence.
    ``raw_output`` retains every field returned by the installed SAM 3
    processor for deeper, version-specific inspection.
    """

    query: str
    segmented_image: Image.Image
    annotated_image: Image.Image
    masks: np.ndarray
    combined_mask: np.ndarray
    boxes_xyxy: np.ndarray
    scores: np.ndarray
    detections: Sequence[Detection]
    raw_output: Mapping[str, Any]

    @property
    def count(self) -> int:
        """Number of instances SAM 3 found for ``query``."""
        return len(self.detections)

    @property
    def best_detection(self) -> Optional[Detection]:
        """The highest-confidence instance, or ``None`` if no target was found."""
        return self.detections[0] if self.detections else None

    def verification_metadata(self) -> Dict[str, Any]:
        """Return JSON-friendly data useful when reviewing a false positive.

        The potentially large binary masks are intentionally left out here and
        remain available through ``masks`` / each ``Detection.mask``.
        """
        return {
            "query": self.query,
            "instance_count": self.count,
            "instances": [
                {
                    "index": detection.index,
                    "confidence": detection.confidence,
                    "box_xyxy": list(detection.box_xyxy),
                    "mask_area_pixels": detection.mask_area_pixels,
                    "mask_area_fraction": detection.mask_area_fraction,
                }
                for detection in self.detections
            ],
            "sam3_output_keys": sorted(self.raw_output.keys()),
        }


class ObjectDetector:
    """Run text-prompted SAM 3 segmentation on one collected episode image.

    The model is loaded lazily on the first call to :meth:`detect`, so scripts
    can import this module even in environments intended only for collection or
    review.  SAM 3 checkpoints are downloaded by its standard Hugging Face
    loading path unless ``checkpoint_path`` is provided.

    Args:
        checkpoint_path: Optional local SAM 3 checkpoint path.
        device: Device accepted by SAM 3, such as ``"cuda"`` or ``"cpu"``.
            Leaving it as ``None`` lets SAM 3 choose its default.
        gpu_id: CUDA GPU number to select with PyTorch before SAM 3 builds its
            model. This works around SAM 3's builder, which supports ``"cuda"``
            but does not move a model for a device string such as ``"cuda:1"``.
        confidence_threshold: Lowest SAM 3 score kept in the result.  SAM 3's
            decoder always emits a fixed number of object queries (200 in the
            pinned build), so a threshold of 0 reports every query as an
            instance; 0.5 is SAM 3's own default.  Lower it only somewhat when
            reviewing marginal false-positive cases.
        max_instances: Optional cap on how many instances are kept, applied
            after thresholding and score sorting.  Useful together with a
            lowered threshold to keep annotated output readable.
        mask_opacity: Opacity of masks painted into ``annotated_image``.
        model: An optional pre-built SAM 3 image model, mainly for reuse/tests.
        processor: An optional pre-built ``Sam3Processor``.  Supplying one
            avoids model construction and is useful for dependency injection.

    Numpy arrays are interpreted as RGB.  Pass ``image_color_space="bgr"`` to
    :meth:`detect` when providing an OpenCV BGR frame.
    """

    def __init__(
        self,
        *,
        checkpoint_path: Optional[Union[str, Path]] = None,
        device: Optional[str] = None,
        gpu_id: Optional[int] = None,
        confidence_threshold: float = 0.5,
        max_instances: Optional[int] = None,
        mask_opacity: float = 0.45,
        model: Optional[Any] = None,
        processor: Optional[Any] = None,
    ) -> None:
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be between 0 and 1")
        if not 0.0 <= mask_opacity <= 1.0:
            raise ValueError("mask_opacity must be between 0 and 1")
        if processor is not None and model is not None:
            raise ValueError("Pass either model or processor, not both")
        if max_instances is not None and int(max_instances) < 1:
            raise ValueError("max_instances must be at least 1")
        if gpu_id is not None and int(gpu_id) < 0:
            raise ValueError("gpu_id must be zero or greater")
        if gpu_id is not None and device not in (None, "cuda"):
            raise ValueError("gpu_id can only be used with device='cuda' or None")

        self.checkpoint_path = str(checkpoint_path) if checkpoint_path else None
        self.device = device
        self.gpu_id = int(gpu_id) if gpu_id is not None else None
        self.confidence_threshold = confidence_threshold
        self.max_instances = int(max_instances) if max_instances is not None else None
        self.mask_opacity = mask_opacity
        self._model = model
        self._processor = processor

    def __call__(self, image: ImageInput, target_query: str) -> ObjectDetectionResult:
        return self.detect(image, target_query)

    def detect(
        self,
        image: ImageInput,
        target_query: str,
        *,
        image_color_space: str = "rgb",
    ) -> ObjectDetectionResult:
        """Segment every SAM 3 match for ``target_query`` in ``image``.

        Args:
            image: Image path, RGB PIL image, or RGB NumPy ``H x W x C`` array.
            target_query: Short open-vocabulary text phrase, e.g. ``"red mug"``.
            image_color_space: ``"rgb"`` (default) or ``"bgr"`` for NumPy
                images created by OpenCV.
        """
        if not isinstance(target_query, str) or not target_query.strip():
            raise ValueError("target_query must be a non-empty string")

        pil_image = self._to_rgb_pil(image, image_color_space)
        processor = self._get_processor()
        # SAM 3's image processor omits the CUDA autocast wrapper that its
        # video path uses. Without it, its vision backbone can produce BF16
        # activations for FP32 linear layers (BFloat16/Float mismatch).
        with self._inference_autocast():
            state = processor.set_image(pil_image)
            raw_output = processor.set_text_prompt(
                state=state, prompt=target_query.strip()
            )

        masks, boxes, scores = self._normalise_output(raw_output, pil_image.size)
        keep = scores >= self.confidence_threshold
        masks, boxes, scores = masks[keep], boxes[keep], scores[keep]

        # SAM 3 normally orders by score already, but make the verifier's output
        # deterministic across SAM 3 versions.
        order = np.argsort(-scores, kind="stable")
        if self.max_instances is not None:
            order = order[: self.max_instances]
        masks, boxes, scores = masks[order], boxes[order], scores[order]
        detections = self._build_detections(masks, boxes, scores)

        return ObjectDetectionResult(
            query=target_query.strip(),
            segmented_image=self._segment_image(pil_image, masks),
            annotated_image=self._render(pil_image, detections, target_query.strip()),
            masks=masks,
            combined_mask=masks.any(axis=0)
            if len(masks)
            else np.zeros((pil_image.height, pil_image.width), dtype=bool),
            boxes_xyxy=boxes,
            scores=scores,
            detections=detections,
            raw_output=raw_output,
        )

    def _get_processor(self) -> Any:
        if self._processor is not None:
            return self._processor

        build_device = self._configure_cuda_device()
        try:
            from sam3.model.sam3_image_processor import Sam3Processor
            from sam3.model_builder import build_sam3_image_model
        except ModuleNotFoundError as error:
            raise ModuleNotFoundError(
                "SAM 3 is required for ObjectDetector. Use the project's core "
                "Docker environment, which installs the pinned sam3 package."
            ) from error

        build_kwargs: Dict[str, Any] = {}
        if self.checkpoint_path is not None:
            build_kwargs["checkpoint_path"] = self.checkpoint_path
        if build_device is not None:
            build_kwargs["device"] = build_device
        if self._model is None:
            self._model = build_sam3_image_model(**build_kwargs)
            if build_device is not None:
                # SAM 3's builder does not reliably move every submodule and
                # buffer onto the requested device, which leaves CPU weights
                # facing a CUDA input (torch.cuda.FloatTensor vs
                # torch.FloatTensor). Move the built model explicitly.
                try:
                    self._model.to(build_device)
                except AttributeError:
                    pass

        # The pinned SAM 3 processor supports this argument.  Checking the
        # signature keeps this verifier compatible with earlier SAM 3 builds.
        processor_kwargs: Dict[str, Any] = {}
        parameters = inspect.signature(Sam3Processor).parameters
        if "confidence_threshold" in parameters:
            processor_kwargs["confidence_threshold"] = self.confidence_threshold
        self._processor = Sam3Processor(self._model, **processor_kwargs)
        return self._processor

    def _configure_cuda_device(self) -> Optional[str]:
        """Select a CUDA GPU before SAM 3 uses its limited device interface."""
        if self.gpu_id is None:
            return self.device
        try:
            import torch
        except ModuleNotFoundError as error:
            raise ModuleNotFoundError(
                "PyTorch is required to select a CUDA GPU for SAM 3."
            ) from error
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"gpu_id={self.gpu_id} was requested, but PyTorch cannot access CUDA"
            )
        if self.gpu_id >= torch.cuda.device_count():
            raise ValueError(
                f"gpu_id={self.gpu_id} is unavailable; PyTorch sees "
                f"{torch.cuda.device_count()} CUDA device(s)"
            )
        torch.cuda.set_device(self.gpu_id)
        # SAM 3's builder only moves the model when device is exactly "cuda".
        return "cuda"

    def _inference_autocast(self) -> Any:
        """Use SAM 3's required BF16 CUDA inference context when on a GPU."""
        try:
            model_device = next(self._model.parameters()).device
        except (AttributeError, StopIteration):
            return nullcontext()
        if model_device.type != "cuda":
            return nullcontext()

        import torch

        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)

    @staticmethod
    def _to_rgb_pil(image: ImageInput, image_color_space: str) -> Image.Image:
        if image_color_space.lower() not in {"rgb", "bgr"}:
            raise ValueError("image_color_space must be either 'rgb' or 'bgr'")
        if isinstance(image, (str, Path)):
            with Image.open(image) as opened_image:
                return opened_image.convert("RGB")
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        if not isinstance(image, np.ndarray):
            raise TypeError("image must be a path, PIL image, or NumPy array")

        array = np.asarray(image)
        if array.ndim not in {2, 3}:
            raise ValueError("NumPy image must have shape HxW, HxWx3, or HxWx4")
        if array.ndim == 3 and array.shape[2] not in {1, 3, 4}:
            raise ValueError("NumPy image must have 1, 3, or 4 channels")
        if array.dtype != np.uint8:
            if np.issubdtype(array.dtype, np.floating) and array.size and array.max() <= 1:
                array = array * 255
            array = np.clip(array, 0, 255).astype(np.uint8)
        if array.ndim == 3 and array.shape[2] == 4:
            array = array[:, :, :3]
        if array.ndim == 3 and array.shape[2] == 3 and image_color_space.lower() == "bgr":
            array = array[:, :, ::-1]
        return Image.fromarray(array).convert("RGB")

    @staticmethod
    def _as_numpy(value: Any) -> np.ndarray:
        if value is None:
            return np.asarray([])
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
        if hasattr(value, "numpy"):
            # SAM 3's CUDA autocast can emit bfloat16 tensors, and torch has no
            # NumPy bridge for bfloat16, so upcast those to float32 first.
            if "bfloat16" in str(getattr(value, "dtype", "")):
                value = value.float()
            value = value.numpy()
        return np.asarray(value)

    @classmethod
    def _normalise_output(
        cls, raw_output: Mapping[str, Any], image_size: Tuple[int, int]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not isinstance(raw_output, Mapping):
            raise TypeError("SAM 3 returned an unexpected non-mapping output")
        missing = {"masks", "boxes", "scores"}.difference(raw_output)
        if missing:
            raise KeyError(f"SAM 3 output is missing required keys: {sorted(missing)}")

        width, height = image_size
        masks = cls._as_numpy(raw_output["masks"])
        boxes = cls._as_numpy(raw_output["boxes"])
        scores = cls._as_numpy(raw_output["scores"])

        if masks.size == 0:
            masks = np.empty((0, height, width), dtype=bool)
        elif masks.ndim == 4 and masks.shape[1] == 1:
            masks = masks[:, 0]
        elif masks.ndim == 2:
            masks = masks[None, :, :]
        if masks.ndim != 3:
            raise ValueError(f"Expected SAM 3 masks shaped NxHxW, got {masks.shape}")
        if masks.shape[1:] != (height, width):
            raise ValueError(
                "SAM 3 masks do not match the input image dimensions "
                f"({masks.shape[1:]} vs {(height, width)})"
            )
        masks = masks.astype(bool, copy=False)

        if boxes.size == 0:
            boxes = np.empty((0, 4), dtype=np.float32)
        else:
            boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
        scores = np.asarray(scores, dtype=np.float32).reshape(-1)
        if not (len(masks) == len(boxes) == len(scores)):
            raise ValueError(
                "SAM 3 returned a different number of masks, boxes, and scores "
                f"({len(masks)}, {len(boxes)}, {len(scores)})"
            )
        return masks, boxes, scores

    @staticmethod
    def _build_detections(
        masks: np.ndarray, boxes: np.ndarray, scores: np.ndarray
    ) -> List[Detection]:
        image_area = masks.shape[1] * masks.shape[2] if len(masks) else 0
        return [
            Detection(
                index=index,
                confidence=float(scores[index]),
                box_xyxy=tuple(float(value) for value in boxes[index]),
                mask=masks[index],
                mask_area_pixels=int(masks[index].sum()),
                mask_area_fraction=(float(masks[index].sum()) / image_area)
                if image_area
                else 0.0,
            )
            for index in range(len(masks))
        ]

    def _render(
        self, image: Image.Image, detections: Sequence[Detection], query: str
    ) -> Image.Image:
        canvas = np.asarray(image).copy()
        colors = ((230, 57, 70), (29, 161, 242), (42, 157, 143), (244, 162, 97))
        for detection in detections:
            color = np.asarray(colors[detection.index % len(colors)], dtype=np.float32)
            mask = detection.mask
            canvas[mask] = (
                (1.0 - self.mask_opacity) * canvas[mask] + self.mask_opacity * color
            ).astype(np.uint8)

        annotated = Image.fromarray(canvas, mode="RGB")
        draw = ImageDraw.Draw(annotated)
        for detection in detections:
            color = colors[detection.index % len(colors)]
            draw.rectangle(detection.box_xyxy, outline=color, width=3)
            label = (
                f"{query} #{detection.index + 1}  {detection.confidence:.3f}  "
                f"{detection.mask_area_fraction:.1%}"
            )
            left, top, _, _ = detection.box_xyxy
            label_box = draw.textbbox((left, top), label)
            draw.rectangle(label_box, fill=color)
            draw.text((left, top), label, fill=(255, 255, 255))
        return annotated

    @staticmethod
    def _segment_image(image: Image.Image, masks: np.ndarray) -> Image.Image:
        """Return the source pixels belonging to any detected target instance."""
        segmented = np.zeros_like(np.asarray(image))
        if len(masks):
            combined_mask = masks.any(axis=0)
            segmented[combined_mask] = np.asarray(image)[combined_mask]
        return Image.fromarray(segmented, mode="RGB")
