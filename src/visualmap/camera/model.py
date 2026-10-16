"""Camera model: intrinsics container + pixel helpers.

The heavy math (undistort, project) stays thin for V0.1. Feature matching and
reprojection live in SLAM backends later.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import yaml

from visualmap.slam.types import CameraIntrinsics
from visualmap.utils.logging import get_logger

logger = get_logger(__name__)

__all__ = ["CameraModel", "load_intrinsics", "save_intrinsics"]


class CameraModel:
    """Wraps CameraIntrinsics with practical helpers."""

    def __init__(self, intrinsics: CameraIntrinsics) -> None:
        self.intrinsics = intrinsics

    @property
    def K(self) -> np.ndarray:
        return self.intrinsics.K

    @property
    def dist_coeffs(self) -> np.ndarray:
        return self.intrinsics.dist_coeffs

    @property
    def size(self) -> tuple[int, int]:
        """(width, height)."""
        return self.intrinsics.width, self.intrinsics.height

    def is_compatible_with(self, width: int, height: int) -> bool:
        return self.intrinsics.width == width and self.intrinsics.height == height

    def undistort_image(self, image: np.ndarray) -> np.ndarray:
        if not self.intrinsics.has_distortion:
            return image
        return cv2.undistort(image, self.K, self.dist_coeffs)

    def project(self, points_cam: np.ndarray) -> np.ndarray:
        """Project 3D camera-frame points to pixels. points: Nx3 -> Nx2."""
        pts = np.asarray(points_cam, dtype=np.float64).reshape(-1, 3)
        if np.any(pts[:, 2] <= 0):
            raise ValueError("points must have positive Z in camera frame")
        uv = (pts[:, :2] / pts[:, 2:3]) * np.array([self.intrinsics.fx, self.intrinsics.fy])
        uv += np.array([self.intrinsics.cx, self.intrinsics.cy])
        return uv

    def to_dict(self) -> dict:
        return self.intrinsics.to_dict()


def load_intrinsics(path: str | Path) -> CameraIntrinsics:
    """Load intrinsics from YAML.

    Accepted shapes:
      1) bare keys: fx, fy, cx, cy, width, height, dist
      2) nested: camera: { ... }
      3) opencv-style: camera_matrix + dist_coeffs + image_size
    """
    p = Path(path)
    data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    if "camera" in data:
        data = data["camera"]

    if "fx" in data:
        return CameraIntrinsics.from_dict(data)

    if "camera_matrix" in data:
        K = np.asarray(data["camera_matrix"], dtype=np.float64)
        if K.shape != (3, 3):
            raise ValueError("camera_matrix must be 3x3")
        dist = data.get("dist_coeffs", [0, 0, 0, 0, 0])
        size = data.get("image_size")
        if not size or len(size) != 2:
            raise ValueError("image_size [width, height] is required with camera_matrix")
        return CameraIntrinsics(
            fx=float(K[0, 0]),
            fy=float(K[1, 1]),
            cx=float(K[0, 2]),
            cy=float(K[1, 2]),
            width=int(size[0]),
            height=int(size[1]),
            dist=tuple(float(x) for x in np.asarray(dist).ravel()),
        )

    raise ValueError(f"unrecognized intrinsics format: {p}")


def save_intrinsics(intrinsics: CameraIntrinsics, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {"camera": intrinsics.to_dict()}
    p.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")
    logger.info("Saved intrinsics to %s", p)
    return p
