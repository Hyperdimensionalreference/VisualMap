"""Object detector abstract interface (adapter for YOLO / custom engines)."""

from __future__ import annotations

from abc import ABC, abstractmethod

from visualmap.detection.types import Detection
from visualmap.slam.types import Frame


class ObjectDetector(ABC):
    """Detect objects in a BGR frame. Business code never binds a concrete model."""

    name: str = "base"

    @abstractmethod
    def detect(self, frame: Frame) -> list[Detection]:
        """Return 2D detections for this frame (may be empty)."""

    def warmup(self) -> None:  # noqa: B027
        """Optional: load weights / allocate buffers."""
