"""Depth estimation types.

Depth is stored in **map units** (monocular scale-ambiguous) unless a metric
depth model is attached. Values are z-depth in the camera frame (OpenCV z forward).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = ["DepthMap", "DepthResult"]


@dataclass(frozen=True)
class DepthMap:
    """One depth map aligned to a source frame."""

    frame_id: int
    timestamp: float
    # HxW float32, z-depth in map units; invalid pixels are NaN or <= 0
    depth: np.ndarray
    valid_ratio: float = 0.0
    # Original image size this grid was built from (for full-res detections)
    source_width: int = 0
    source_height: int = 0

    def __post_init__(self) -> None:
        d = np.asarray(self.depth, dtype=np.float32)
        if d.ndim != 2:
            raise ValueError(f"depth must be HxW, got {d.shape}")
        object.__setattr__(self, "depth", d)

    @property
    def height(self) -> int:
        return int(self.depth.shape[0])

    @property
    def width(self) -> int:
        return int(self.depth.shape[1])

    def valid_mask(self) -> np.ndarray:
        return np.isfinite(self.depth) & (self.depth > 0)

    def to_dict(self) -> dict:
        valid = self.valid_mask()
        vals = self.depth[valid]
        return {
            "frame_id": self.frame_id,
            "timestamp": self.timestamp,
            "width": self.width,
            "height": self.height,
            "source_width": self.source_width,
            "source_height": self.source_height,
            "valid_ratio": float(valid.mean()) if self.depth.size else 0.0,
            "min": float(vals.min()) if vals.size else 0.0,
            "max": float(vals.max()) if vals.size else 0.0,
            "mean": float(vals.mean()) if vals.size else 0.0,
        }


@dataclass
class DepthResult:
    """Collection of depth maps + optional per-point depths."""

    maps: list[DepthMap] = field(default_factory=list)
    # map_point_id -> depth in its reference camera (map units)
    point_depths: dict[int, float] = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.maps)

    def summary(self) -> dict:
        ratios = [m.valid_ratio for m in self.maps]
        return {
            "num_depth_maps": len(self.maps),
            "mean_valid_ratio": float(np.mean(ratios)) if ratios else 0.0,
            "num_point_depths": len(self.point_depths),
            **self.metadata,
        }
