"""Two-view triangulation for sparse map points.

Produces world-frame landmarks from matched pixels in two cameras with known
T_world←camera poses. Points with negative depth or large reprojection error
are discarded.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from visualmap.slam.types import CameraIntrinsics, CameraPose, MapPoint
from visualmap.utils.logging import get_logger

logger = get_logger(__name__)

__all__ = ["TriangulationConfig", "triangulate_two_view", "reprojection_error"]


@dataclass
class TriangulationConfig:
    min_parallax_deg: float = 1.0
    max_reproj_error_px: float = 3.0
    min_depth: float = 0.1
    max_depth: float = 200.0


def triangulate_two_view(
    pts_a: np.ndarray,
    pts_b: np.ndarray,
    pose_a: CameraPose,
    pose_b: CameraPose,
    intrinsics: CameraIntrinsics,
    config: TriangulationConfig | None = None,
    start_id: int = 0,
    feature_ids_a: np.ndarray | None = None,
    feature_ids_b: np.ndarray | None = None,
    frame_a: int = -1,
    frame_b: int = -1,
) -> list[MapPoint]:
    """Triangulate matched pixels.

    pts_a / pts_b: Nx2 pixel coordinates (u, v) in the same order.
    pose_a / pose_b: T_world←camera for each view.
    """
    cfg = config or TriangulationConfig()
    pts_a = np.asarray(pts_a, dtype=np.float64).reshape(-1, 2)
    pts_b = np.asarray(pts_b, dtype=np.float64).reshape(-1, 2)
    if pts_a.shape != pts_b.shape or pts_a.shape[0] == 0:
        return []

    # Triangulate undistorted bearings, not distorted sensor pixels. Applying K
    # alone silently bends rays on cameras with Brown-Conrady calibration.
    normalized_a = cv2.undistortPoints(
        pts_a.reshape(-1, 1, 2), intrinsics.K, intrinsics.dist_coeffs
    ).reshape(-1, 2)
    normalized_b = cv2.undistortPoints(
        pts_b.reshape(-1, 1, 2), intrinsics.K, intrinsics.dist_coeffs
    ).reshape(-1, 2)

    # Normalized projection matrices: P = T_camera←world[:3].
    T_wa = pose_a.matrix
    T_wb = pose_b.matrix
    T_aw = np.linalg.inv(T_wa)
    T_bw = np.linalg.inv(T_wb)
    P_a = T_aw[:3, :]
    P_b = T_bw[:3, :]

    # Homogeneous linear triangulation (Hartley), batched in OpenCV.
    n = pts_a.shape[0]
    X_h = cv2.triangulatePoints(P_a, P_b, normalized_a.T, normalized_b.T)

    # Dehomogenize
    w = X_h[3, :]
    w[np.abs(w) < 1e-12] = 1e-12
    X = (X_h[:3, :] / w).T  # Nx3 world

    # Depths in each camera
    Xa = (T_aw[:3, :3] @ X.T + T_aw[:3, 3:4]).T
    Xb = (T_bw[:3, :3] @ X.T + T_bw[:3, 3:4]).T
    depth_a = Xa[:, 2]
    depth_b = Xb[:, 2]

    # Parallax
    ca = pose_a.camera_center
    cb = pose_b.camera_center
    baseline = float(np.linalg.norm(ca - cb))
    if baseline < 1e-9:
        return []

    points: list[MapPoint] = []
    next_id = start_id
    for i in range(n):
        if not np.all(np.isfinite(X[i])) or abs(X_h[3, i]) < 1e-12:
            continue
        if depth_a[i] < cfg.min_depth or depth_b[i] < cfg.min_depth:
            continue
        if depth_a[i] > cfg.max_depth or depth_b[i] > cfg.max_depth:
            continue

        # Parallax angle between rays
        ra = X[i] - ca
        rb = X[i] - cb
        na = np.linalg.norm(ra)
        nb = np.linalg.norm(rb)
        if na < 1e-9 or nb < 1e-9:
            continue
        cos_p = float(np.clip(np.dot(ra, rb) / (na * nb), -1.0, 1.0))
        parallax_deg = float(np.degrees(np.arccos(cos_p)))
        if parallax_deg < cfg.min_parallax_deg:
            continue

        err_a = reprojection_error(X[i], pose_a, intrinsics, pts_a[i])
        err_b = reprojection_error(X[i], pose_b, intrinsics, pts_b[i])
        if err_a > cfg.max_reproj_error_px or err_b > cfg.max_reproj_error_px:
            continue

        obs = {}
        if feature_ids_a is not None:
            obs[int(frame_a)] = int(feature_ids_a[i])
        if feature_ids_b is not None:
            obs[int(frame_b)] = int(feature_ids_b[i])

        points.append(MapPoint(id=next_id, xyz=X[i], observations=obs))
        next_id += 1

    logger.debug("Triangulated %d / %d candidates", len(points), n)
    return points


def reprojection_error(
    X_world: np.ndarray,
    pose: CameraPose,
    intrinsics: CameraIntrinsics,
    pixel: np.ndarray,
) -> float:
    """Pixel-space reprojection error of a world point in a given camera."""
    X_cam = pose.inverse().transform_point(np.asarray(X_world, dtype=np.float64))
    z = X_cam[2]
    if z <= 1e-9:
        return 1e9
    projected, _ = cv2.projectPoints(
        X_cam.reshape(1, 3),
        np.zeros(3),
        np.zeros(3),
        intrinsics.K,
        intrinsics.dist_coeffs,
    )
    u, v = projected.reshape(2)
    return float(np.hypot(u - pixel[0], v - pixel[1]))
