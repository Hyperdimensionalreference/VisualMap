"""Depth estimator abstract interface.

Business code depends only on this interface. Concrete engines
(sparse triangulation, MiDaS, Depth Anything, ...) live as implementations.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from visualmap.depth.types import DepthResult
from visualmap.slam.types import CameraIntrinsics, MapResult


class DepthEstimator(ABC):
    """Produce per-pixel approximate depth from a SLAM result (+ optional frames)."""

    name: str = "base"

    @abstractmethod
    def estimate(
        self,
        map_result: MapResult,
        intrinsics: CameraIntrinsics | None = None,
    ) -> DepthResult:
        """Return depth maps / point depths for the given reconstruction."""
