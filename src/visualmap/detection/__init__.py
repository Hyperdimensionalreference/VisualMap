"""Detection / semantics package (V0.5)."""

from visualmap.detection.base import ObjectDetector
from visualmap.detection.configured import create_configured_detector
from visualmap.detection.demo import DemoDetector, DemoDetectorConfig
from visualmap.detection.lift import LiftConfig, lift_detection
from visualmap.detection.pipeline import SemanticPipeline, SemanticPipelineConfig, create_detector
from visualmap.detection.tracker import IoUTracker, TrackerConfig
from visualmap.detection.types import Detection, SemanticObject, SemanticResult, TrackedDetection

__all__ = [
    "ObjectDetector",
    "DemoDetector",
    "DemoDetectorConfig",
    "Detection",
    "TrackedDetection",
    "SemanticObject",
    "SemanticResult",
    "IoUTracker",
    "TrackerConfig",
    "LiftConfig",
    "lift_detection",
    "SemanticPipeline",
    "SemanticPipelineConfig",
    "create_detector",
    "create_configured_detector",
]
