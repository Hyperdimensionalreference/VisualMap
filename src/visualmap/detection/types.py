"""Detection / semantic object types.

2D boxes are pixels (OpenCV: origin top-left). 3D positions are world
coordinates in map units (same convention as CameraPose: T_world←camera).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = ["Detection", "TrackedDetection", "SemanticObject", "SemanticResult"]


@dataclass(frozen=True)
class Detection:
    """One 2D detection in a single frame."""

    frame_id: int
    timestamp: float
    class_id: int
    class_name: str
    confidence: float
    # x1, y1, x2, y2 in pixels
    xyxy: tuple[float, float, float, float]
    # Distinguish raw model predictions, structural candidates and review evidence.
    metadata: dict = field(default_factory=dict)

    @property
    def cx(self) -> float:
        return 0.5 * (self.xyxy[0] + self.xyxy[2])

    @property
    def cy(self) -> float:
        return 0.5 * (self.xyxy[1] + self.xyxy[3])

    @property
    def width(self) -> float:
        return max(0.0, self.xyxy[2] - self.xyxy[0])

    @property
    def height(self) -> float:
        return max(0.0, self.xyxy[3] - self.xyxy[1])

    def to_dict(self) -> dict:
        return {
            "frame_id": self.frame_id,
            "timestamp": self.timestamp,
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": self.confidence,
            "xyxy": list(self.xyxy),
            "detection_evidence": self.metadata,
            "classification_needs_review": self.metadata.get("classification_needs_review", False),
            "classification_reason": self.metadata.get("classification_reason"),
            "classification_evidence": {
                "crop_verification": self.metadata.get("crop_verification"),
                "source": self.metadata.get("classification_source", "model_inference"),
                "status": self.metadata.get("classification_status", "unverified_model_prediction"),
                "needs_review": self.metadata.get("classification_needs_review", False),
                "reason": self.metadata.get("classification_reason"),
                "raw_model_class": self.metadata.get("raw_model_class", self.class_name),
                "original_model_class": self.metadata.get("original_model_class", self.class_name),
                "structure_evidence": self.metadata.get("structure_evidence"),
                "raw_model_detections": self.metadata.get("raw_model_detections", []),
                "user_confirmed": self.metadata.get("user_confirmed", False),
                "trained_pegboard_model": self.metadata.get("trained_pegboard_model", False),
                "accuracy_validated": False,
                "full_object_boundary_verified": self.metadata.get("full_object_boundary_verified"),
                "physical_object_membership_verified": self.metadata.get(
                    "physical_object_membership_verified"
                ),
                "class_semantics_validated": self.metadata.get("class_semantics_validated", False),
                "possible_object_types": self.metadata.get("possible_object_types", []),
                "score_is_probability": self.metadata.get("rule_score_is_probability"),
                "temporal_class_disagreement": self.metadata.get(
                    "temporal_class_disagreement", False
                ),
                "stabilized_class": self.metadata.get("stabilized_class", self.class_name),
                "temporal_reason": self.metadata.get("temporal_classification_reason"),
            },
        }


@dataclass
class TrackedDetection:
    """Detection with a stable track id across frames."""

    track_id: int
    detection: Detection
    hits: int = 1
    # Preserve the current model output when the temporal vote keeps another label.
    raw_detection: Detection | None = None
    # Share of recent confidence mass supporting the displayed class; not accuracy.
    class_consistency: float = 1.0
    class_observations: int = 1
    class_alternatives: tuple[Detection, ...] = ()

    @property
    def class_name(self) -> str:
        return self.detection.class_name

    def to_dict(self) -> dict:
        d = self.detection.to_dict()
        d["track_id"] = self.track_id
        d["hits"] = self.hits
        d["class_consistency"] = self.class_consistency
        d["class_observations"] = self.class_observations
        raw = self.raw_detection or self.detection
        d["raw_class_id"] = raw.class_id
        d["raw_class_name"] = raw.class_name
        d["raw_confidence"] = raw.confidence
        d["class_alternatives"] = [candidate.to_dict() for candidate in self.class_alternatives]
        return d


@dataclass
class SemanticObject:
    """A tracked object lifted into the world frame."""

    track_id: int
    class_id: int
    class_name: str
    # World position (map units). Use xyz=None if lift failed.
    xyz: np.ndarray | None
    # Camera-frame z (depth) if available
    depth: float | None
    first_frame: int
    last_frame: int
    observations: int
    confidence: float
    # Last 2D box
    xyxy: tuple[float, float, float, float]
    frame_id: int
    metadata: dict = field(default_factory=dict)

    @property
    def label(self) -> str:
        return f"{self.class_name} #{self.track_id:03d}"

    def to_dict(self) -> dict:
        return {
            "track_id": self.track_id,
            "label": self.label,
            "class_id": self.class_id,
            "class_name": self.class_name,
            "xyz": self.xyz.tolist() if self.xyz is not None else None,
            "depth": self.depth,
            "first_frame": self.first_frame,
            "last_frame": self.last_frame,
            "observations": self.observations,
            "confidence": self.confidence,
            "xyxy": list(self.xyxy),
            "frame_id": self.frame_id,
            **self.metadata,
        }


@dataclass
class SemanticResult:
    objects: list[SemanticObject] = field(default_factory=list)
    detections: list[Detection] = field(default_factory=list)
    tracks: list[TrackedDetection] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.objects)

    def summary(self) -> dict:
        by_class: dict[str, int] = {}
        for obj in self.objects:
            by_class[obj.class_name] = by_class.get(obj.class_name, 0) + 1
        return {
            "num_objects": len(self.objects),
            "num_detections": len(self.detections),
            "num_tracks": len(self.tracks),
            "by_class": by_class,
            **self.metadata,
        }
