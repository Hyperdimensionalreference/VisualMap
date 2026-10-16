"""Optional CLIP crop re-ranking for real detector proposals; review evidence only."""

from __future__ import annotations

import hashlib
import importlib
import math
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import cv2
import numpy as np

from visualmap.detection.base import ObjectDetector
from visualmap.detection.configured import merge_detections
from visualmap.detection.open_vocabulary import load_vocabulary
from visualmap.detection.types import Detection
from visualmap.slam.types import Frame


@dataclass
class CLIPCropConfig:
    model_path: str | Path
    vocabulary_path: str | Path
    runtime_path: str | Path | None = None
    allowed_classes: tuple[str, ...] = (
        "earbud case",
        "watch",
        "tablet",
        "monitor",
        "lamp",
        "power adapter",
        "power strip",
    )
    min_similarity: float = 0.25
    min_margin: float = 0.003
    intra_op_threads: int = 2

    def __post_init__(self) -> None:
        for name in ("min_similarity", "min_margin"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or not 0 <= value < 1:
                raise ValueError(f"{name} must be finite and in [0, 1)")
        if (
            not isinstance(self.allowed_classes, (tuple, list))
            or not self.allowed_classes
            or any(not isinstance(name, str) or not name for name in self.allowed_classes)
        ):
            raise ValueError("allowed_classes must be a nonempty list of class names")
        if (
            isinstance(self.intra_op_threads, bool)
            or not isinstance(self.intra_op_threads, int)
            or self.intra_op_threads < 1
        ):
            raise ValueError("intra_op_threads must be a positive integer")


def clip_image_blob(image: np.ndarray) -> np.ndarray:
    """Official CLIP short-edge resize, center crop, RGB and normalization."""
    height, width = image.shape[:2]
    scale = 224 / min(height, width)
    resized = cv2.resize(
        image, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_CUBIC
    )
    top, left = (resized.shape[0] - 224) // 2, (resized.shape[1] - 224) // 2
    rgb = cv2.cvtColor(resized[top : top + 224, left : left + 224], cv2.COLOR_BGR2RGB)
    normalized = (
        rgb.astype(np.float32) / 255 - np.array([0.48145466, 0.4578275, 0.40821073], np.float32)
    ) / np.array([0.26862954, 0.26130258, 0.27577711], np.float32)
    return np.ascontiguousarray(normalized.transpose(2, 0, 1)[None])


class CLIPCropVerifier(ObjectDetector):
    """Reject conflicting/background candidates and suggest a generic crop label.

    This verifies semantic similarity only. It cannot verify proposal boundaries,
    physical identity, volume or a calibrated class probability.
    """

    def __init__(self, detector: ObjectDetector, config: CLIPCropConfig) -> None:
        self.detector, self.config = detector, config
        self.name = f"{detector.name}+clip_crop_review"
        self._prompts, vectors, manifest = load_vocabulary(config.vocabulary_path)
        self._vectors = vectors[0]
        self._names = [row["class_name"] for row in self._prompts]
        if not set(config.allowed_classes) <= set(self._names):
            raise ValueError("allowed_classes must occur in the CLIP comparison vocabulary")
        assets = {row["file"]: row for row in manifest.get("assets", [])}
        model_path = Path(config.model_path)
        if hashlib.sha256(model_path.read_bytes()).hexdigest() != assets.get(
            model_path.name, {}
        ).get("sha256"):
            raise ValueError("CLIP image model hash mismatch")
        if config.runtime_path is not None:
            runtime_path = str(Path(config.runtime_path).resolve())
            if runtime_path not in sys.path:
                sys.path.insert(0, runtime_path)
        ort = importlib.import_module("onnxruntime")
        options = ort.SessionOptions()
        options.intra_op_num_threads = config.intra_op_threads
        options.inter_op_num_threads = 1
        self._session = ort.InferenceSession(
            str(model_path), options, providers=["CPUExecutionProvider"]
        )
        if {row.name for row in self._session.get_inputs()} != {"pixel_values"}:
            raise ValueError("incompatible CLIP image model inputs")
        self.rejected_predictions: list[dict] = []
        self.capabilities = {
            **getattr(detector, "capabilities", {}),
            "class_names": list(config.allowed_classes),
            "crop_verification": {
                "model": "CLIP ViT-B/32 quantized ONNX",
                "candidate_only": True,
                "calibrated_accuracy": False,
                "physical_boundary_verified": False,
                "min_similarity": config.min_similarity,
                "min_margin": config.min_margin,
                "comparison_classes": self._names,
                "rejected_predictions": self.rejected_predictions,
            },
        }

    def warmup(self) -> None:
        self.detector.warmup()
        self._session.run(
            ["image_embeds"], {"pixel_values": clip_image_blob(np.zeros((224, 224, 3), np.uint8))}
        )

    def detect(self, frame: Frame) -> list[Detection]:
        return self.verify(frame, self.detector.detect(frame))

    def verify(self, frame: Frame, proposals: list[Detection]) -> list[Detection]:
        accepted = []
        height, width = frame.image.shape[:2]
        for proposal in proposals:
            x1, y1, x2, y2 = map(round, proposal.xyxy)
            crop = frame.image[max(0, y1) : min(height, y2), max(0, x1) : min(width, x2)]
            evidence = {
                "source": "clip_crop_similarity",
                "model": "CLIP ViT-B/32",
                "proposal": proposal.to_dict(),
                "score_is_probability": False,
                "physical_boundary_verified": False,
                "accuracy_validated": False,
            }
            if not crop.size or min(crop.shape[:2]) < 8:
                self.rejected_predictions.append({**evidence, "reason": "crop_too_small"})
                continue
            vector = self._session.run(["image_embeds"], {"pixel_values": clip_image_blob(crop)})[
                0
            ][0]
            if vector.shape != (512,) or not np.isfinite(vector).all():
                raise ValueError("invalid CLIP image embedding")
            norm = np.linalg.norm(vector)
            if norm <= 1e-6:
                raise ValueError("zero CLIP image embedding")
            similarities = self._vectors @ (vector / norm)
            ordered = np.argsort(similarities)[::-1]
            index = int(ordered[0])
            name = self._names[index]
            margin = float(similarities[index] - similarities[ordered[1]])
            evidence.update(
                {
                    "top_candidates": [
                        {"class_name": self._names[k], "cosine_similarity": float(similarities[k])}
                        for k in ordered[:3]
                    ],
                    "top_margin": margin,
                }
            )
            if (
                name not in self.config.allowed_classes
                or similarities[index] < self.config.min_similarity
                or margin < self.config.min_margin
            ):
                self.rejected_predictions.append({**evidence, "reason": "crop_not_supported"})
                continue
            accepted.append(
                replace(
                    proposal,
                    class_id=index,
                    class_name=name,
                    metadata={
                        **proposal.metadata,
                        "classification_source": "open_vocabulary_model",
                        "classification_needs_review": True,
                        "classification_candidate_only": True,
                        "classification_reason": "clip_crop_review_candidate",
                        "classification_status": "clip_crop_review_candidate",
                        "crop_verification": evidence,
                        "confidence_basis": "original_proposal_score",
                        "full_object_boundary_verified": False,
                    },
                )
            )
        return merge_detections(accepted, 0.5)
