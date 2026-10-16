"""Depth estimation package.

Default implementation: sparse triangulation → grid depth (no neural model).
Optional: attach a metric/relative depth model later via the same interface.
"""

from visualmap.depth.base import DepthEstimator
from visualmap.depth.sparse import SparseDepthConfig, SparseDepthEstimator
from visualmap.depth.types import DepthMap, DepthResult

__all__ = [
    "DepthEstimator",
    "SparseDepthConfig",
    "SparseDepthEstimator",
    "DepthMap",
    "DepthResult",
    "create_depth_estimator",
]


def create_depth_estimator(name: str = "sparse_depth", **kwargs) -> DepthEstimator:
    key = (name or "sparse_depth").strip().lower()
    if key in ("sparse_depth", "sparse", "triangulation"):
        cfg_keys = {
            "downsample",
            "max_frames",
            "fill_radius",
            "min_points",
        }
        cfg = SparseDepthConfig(**{k: v for k, v in kwargs.items() if k in cfg_keys})
        return SparseDepthEstimator(cfg)
    raise ValueError(f"unknown depth estimator: {name!r}")
