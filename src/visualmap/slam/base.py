"""SLAM backend abstract interface.

Business code depends only on this interface. Concrete engines
(ORB-SLAM3, OpenCV VO, DROID-SLAM, ...) live under ``slam.backends``.
See docs/adr/0002-slam-adapter-interface.md.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator

from visualmap.slam.types import CameraIntrinsics, Frame, MapResult


class SLAMBackend(ABC):
    """Process a monocular video stream into trajectory + sparse map."""

    name: str = "base"

    @abstractmethod
    def initialize(self, intrinsics: CameraIntrinsics) -> None:
        """Bind camera model before feeding frames."""

    @abstractmethod
    def process_frame(self, frame: Frame) -> None:
        """Ingest one frame. Implementations decide keyframe selection."""

    @abstractmethod
    def get_result(self) -> MapResult:
        """Return current trajectory and map points."""

    def run(self, frames: Iterator[Frame], intrinsics: CameraIntrinsics) -> MapResult:
        """Convenience: initialize + consume iterator + return result."""
        self.initialize(intrinsics)
        for frame in frames:
            self.process_frame(frame)
        return self.get_result()

    def shutdown(self) -> None:  # noqa: B027
        """Release resources. Default is a no-op."""
