"""Config-driven detector composition and generic overlapping small-object tiles."""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np

from visualmap.detection.base import ObjectDetector
from visualmap.detection.types import Detection
from visualmap.detection.yolo_onnx import COCO80_NAMES
from visualmap.slam.types import Frame


def merge_detections(rows: list[Detection], threshold: float) -> list[Detection]:
    """Class-aware NMS: keep original model geometry, never average object boxes."""
    kept: list[Detection] = []
    for row in sorted(rows, key=lambda value: (-value.confidence, value.class_name, value.xyxy)):
        duplicate = False
        for earlier in kept:
            if earlier.class_name != row.class_name:
                continue
            x1, y1 = max(row.xyxy[0], earlier.xyxy[0]), max(row.xyxy[1], earlier.xyxy[1])
            x2, y2 = min(row.xyxy[2], earlier.xyxy[2]), min(row.xyxy[3], earlier.xyxy[3])
            intersection = max(0, x2 - x1) * max(0, y2 - y1)
            union = row.width * row.height + earlier.width * earlier.height - intersection
            if union > 0 and intersection / union >= threshold:
                duplicate = True
                break
        if not duplicate:
            kept.append(row)
    return kept


class TiledDetector(ObjectDetector):
    """Whole image plus four overlapping, coordinate-independent image crops.

    Crops cover the entire image and are based only on dimensions. Predictions
    touching an internal crop edge are rejected, preserving complete-object rays.
    The full-image pass always remains available for large and edge objects.
    """

    def __init__(
        self,
        detector: ObjectDetector,
        *,
        enabled: bool = True,
        tile_fraction: float = 0.65,
        every_n_frames: int = 1,
        nms_threshold: float = 0.5,
        edge_margin_px: int = 2,
    ) -> None:
        if not isinstance(enabled, bool):
            raise ValueError("tiling.enabled must be a boolean")
        if not math.isfinite(tile_fraction) or not 0.5 < tile_fraction < 1:
            raise ValueError("tile_fraction must be finite and in (0.5, 1)")
        if (
            isinstance(every_n_frames, bool)
            or not isinstance(every_n_frames, int)
            or every_n_frames < 1
        ):
            raise ValueError("tiling.every_n_frames must be a positive integer")
        if not math.isfinite(nms_threshold) or not 0 < nms_threshold < 1:
            raise ValueError("tiling.nms_threshold must be in (0, 1)")
        if (
            isinstance(edge_margin_px, bool)
            or not isinstance(edge_margin_px, int)
            or edge_margin_px < 0
        ):
            raise ValueError("edge_margin_px must be a nonnegative integer")
        self.detector, self.enabled = detector, enabled
        self.tile_fraction, self.every_n_frames = tile_fraction, every_n_frames
        self.nms_threshold, self.edge_margin_px = nms_threshold, edge_margin_px
        self.name = f"{detector.name}+tiles" if enabled else detector.name
        self.capabilities = {
            **getattr(detector, "capabilities", {}),
            "small_object_tiles": enabled,
            "tile_fraction": tile_fraction,
            "tiles_every_n_frames": every_n_frames,
        }

    def warmup(self) -> None:
        self.detector.warmup()

    def detect(self, frame: Frame) -> list[Detection]:
        rows = list(self.detector.detect(frame))
        if not self.enabled or frame.index % self.every_n_frames:
            return rows
        height, width = frame.image.shape[:2]
        tile_height, tile_width = (
            math.ceil(height * self.tile_fraction),
            math.ceil(width * self.tile_fraction),
        )
        margin = self.edge_margin_px
        for top in sorted({0, height - tile_height}):
            for left in sorted({0, width - tile_width}):
                crop = np.ascontiguousarray(
                    frame.image[top : top + tile_height, left : left + tile_width]
                )
                for row in self.detector.detect(replace(frame, image=crop)):
                    x1, y1, x2, y2 = row.xyxy
                    if (
                        (left > 0 and x1 <= margin)
                        or (top > 0 and y1 <= margin)
                        or (left + tile_width < width and x2 >= tile_width - margin)
                        or (top + tile_height < height and y2 >= tile_height - margin)
                    ):
                        continue
                    rows.append(
                        replace(
                            row,
                            xyxy=(x1 + left, y1 + top, x2 + left, y2 + top),
                            metadata={
                                **row.metadata,
                                "inference_view": "overlapping_tile",
                                "inference_crop_xyxy": [
                                    left,
                                    top,
                                    left + tile_width,
                                    top + tile_height,
                                ],
                            },
                        )
                    )
        return merge_detections(rows, self.nms_threshold)


class CombinedDetector(ObjectDetector):
    """Baseline plus real open-vocabulary observations with one stable class-id space."""

    def __init__(
        self,
        baseline: ObjectDetector,
        supplement: ObjectDetector,
        *,
        every_n_frames: int = 3,
        nms_threshold: float = 0.5,
    ) -> None:
        if (
            isinstance(every_n_frames, bool)
            or not isinstance(every_n_frames, int)
            or every_n_frames < 1
        ):
            raise ValueError("open_vocabulary.every_n_frames must be a positive integer")
        if not math.isfinite(nms_threshold) or not 0 < nms_threshold < 1:
            raise ValueError("combined nms_threshold must be in (0, 1)")
        self.baseline, self.supplement = baseline, supplement
        self.every_n_frames, self.nms_threshold = every_n_frames, nms_threshold
        self.name = f"{baseline.name}+{supplement.name}"
        base_config = getattr(baseline, "config", None)
        # Explicitly use the model's names: custom ONNX vocabularies need not be COCO.
        names = list(
            getattr(base_config, "class_names", None)
            or getattr(baseline, "capabilities", {}).get("class_names")
            or COCO80_NAMES
        )
        self._baseline_names = set(names)
        for name in getattr(supplement, "capabilities", {}).get("class_names", []):
            if name not in names:
                names.append(name)
        self._class_ids = {name: index for index, name in enumerate(names)}
        self.capabilities = {
            "backend": self.name,
            "class_names": names,
            "offline": True,
            "baseline": getattr(baseline, "capabilities", {"vocabulary_kind": "fixed_model_head"}),
            "open_vocabulary": getattr(supplement, "capabilities", {}),
            "supplement_every_n_frames": every_n_frames,
            "brand_recognition_supported": False,
            "accuracy_validated": False,
        }

    def warmup(self) -> None:
        self.baseline.warmup()
        self.supplement.warmup()

    def detect(self, frame: Frame) -> list[Detection]:
        rows = list(self.baseline.detect(frame))
        if frame.index % self.every_n_frames == 0:
            rows.extend(self.supplement.detect(frame))
        mapped = []
        for row in rows:
            if row.class_name not in self._class_ids:
                self._class_ids[row.class_name] = len(self._class_ids)
            metadata = dict(row.metadata)
            if metadata.get("classification_source") == "open_vocabulary_model":
                metadata["open_vocabulary_class_is_novel"] = (
                    row.class_name not in self._baseline_names
                )
                conflicts = []
                for other in rows:
                    if other.class_name == row.class_name:
                        continue
                    x1, y1 = max(row.xyxy[0], other.xyxy[0]), max(row.xyxy[1], other.xyxy[1])
                    x2, y2 = min(row.xyxy[2], other.xyxy[2]), min(row.xyxy[3], other.xyxy[3])
                    intersection = max(0, x2 - x1) * max(0, y2 - y1)
                    union = row.width * row.height + other.width * other.height - intersection
                    if union > 0 and intersection / union >= 0.65:
                        conflicts.append(
                            {
                                "class_name": other.class_name,
                                "confidence": other.confidence,
                                "source": other.metadata.get(
                                    "classification_source", "model_inference"
                                ),
                            }
                        )
                if conflicts:
                    metadata["classification_conflicts"] = conflicts
                    metadata["classification_reason"] = "overlapping_models_disagree"
            mapped.append(replace(row, class_id=self._class_ids[row.class_name], metadata=metadata))
        return merge_detections(mapped, self.nms_threshold)


def create_configured_detector(config) -> ObjectDetector:
    """One public entry for CLI and validation; enabled models fail explicitly."""
    from visualmap.detection.pipeline import create_detector

    kwargs = {
        name: getattr(config, name)
        for name in (
            "conf_threshold",
            "nms_threshold",
            "model_path",
            "input_width",
            "input_height",
            "pegboard_refinement",
            "pegboard_config",
        )
    }
    if config.class_names is not None:
        kwargs["class_names"] = config.class_names
    baseline = create_detector(config.detector, **kwargs)
    baseline.capabilities = {
        "backend": baseline.name,
        "vocabulary_kind": "fixed_model_head",
        "class_names": list(getattr(getattr(baseline, "config", None), "class_names", [])),
        "offline": True,
        "accuracy_validated": False,
        "brand_recognition_supported": False,
    }
    tiling = dict(config.tiling)
    if tiling.pop("enabled", False):
        baseline = TiledDetector(baseline, **tiling)
    supplement = dict(config.open_vocabulary)
    if not supplement.pop("enabled", False):
        return baseline
    interval = supplement.pop("every_n_frames", 3)
    supplement_tiling = dict(supplement.pop("tiling", {}))
    extra = create_supplemental_detector(supplement)
    if supplement_tiling.pop("enabled", False):
        extra = TiledDetector(extra, **supplement_tiling)
    return CombinedDetector(baseline, extra, every_n_frames=interval)


def create_supplemental_detector(options: dict) -> ObjectDetector:
    """Load one explicitly selected offline supplemental model."""
    options = dict(options)
    model_type = options.pop("type", "yolo_world_onnx")
    verifier = dict(options.pop("crop_verifier", {}))
    if model_type == "grounding_dino_onnx":
        from visualmap.detection.grounding_dino import GroundingDINOConfig, GroundingDINODetector

        detector = GroundingDINODetector(GroundingDINOConfig(**options))
    elif model_type == "yolo_world_onnx":
        from visualmap.detection.open_vocabulary import OpenVocabularyConfig, OpenVocabularyDetector

        detector = OpenVocabularyDetector(OpenVocabularyConfig(**options))
    else:
        raise ValueError(f"unsupported open_vocabulary.type: {model_type}")
    if verifier.pop("enabled", False):
        from visualmap.detection.crop_verifier import CLIPCropConfig, CLIPCropVerifier

        detector = CLIPCropVerifier(detector, CLIPCropConfig(**verifier))
    return detector
