"""Sparse-to-grid depth from triangulated map points.

Projects map points into each keyframe (or selected frames), then densifies
via nearest-neighbor fill on a downsampled grid. This gives "大致距离"
without a neural depth model — good enough for V0.4 plumbing and for
later fusion with metric depth (V0.4+ models / V0.5 semantics).

Units: map units (monocular scale).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from visualmap.depth.base import DepthEstimator
from visualmap.depth.types import DepthMap, DepthResult
from visualmap.slam.coverage import landmark_world
from visualmap.slam.types import CameraIntrinsics, CameraPose, MapResult
from visualmap.utils.logging import get_logger

logger = get_logger(__name__)

__all__ = ["SparseDepthConfig", "SparseDepthEstimator"]


@dataclass
class SparseDepthConfig:
    # Downsample factor for the depth grid (1 = full res, 4 = 1/4)
    downsample: int = 2
    # Max keyframe poses to densify (0 = all poses with map visibility)
    max_frames: int = 12
    # Local minimum depth fill radius in grid cells (display approximation).
    fill_radius: int = 8
    min_points: int = 8


class SparseDepthEstimator(DepthEstimator):
    name = "sparse_depth"

    def __init__(self, config: SparseDepthConfig | None = None) -> None:
        self.config = config or SparseDepthConfig()

    def estimate(
        self,
        map_result: MapResult,
        intrinsics: CameraIntrinsics | None = None,
    ) -> DepthResult:
        intr = intrinsics or map_result.intrinsics
        if intr is None:
            raise ValueError("intrinsics required for sparse depth")

        pts = (
            np.stack([mp.xyz for mp in map_result.map_points], axis=0)
            if map_result.map_points
            else np.zeros((0, 3), dtype=np.float64)
        )
        if pts.shape[0] == 0:
            return DepthResult(
                maps=[],
                point_depths={},
                metadata={"method": self.name, "note": "no map points"},
            )

        # Point depths in a reference camera (median over first pose)
        point_depths = self._point_depths_in_cameras(map_result, pts)

        poses = self._select_poses(map_result)
        maps: list[DepthMap] = []
        quality = map_result.metadata.get("frame_quality", {})
        worlds = [landmark_world(map_result, point) for point in map_result.map_points]
        for pose in poses:
            row = quality.get(str(pose.frame_id), quality.get(pose.frame_id, {}))
            world = row.get("world_frame_id", map_result.metadata.get("world_frame_id"))
            selected = pts
            if len(map_result.metadata.get("submaps", [])) > 1:
                selected = pts[np.array([item == world for item in worlds])]
            if quality and row.get("pose_valid") is not True:
                selected = np.zeros((0, 3), dtype=np.float64)
            dm = self._depth_for_pose(pose, intr, selected)
            maps.append(dm)

        return DepthResult(
            maps=maps,
            point_depths=point_depths,
            metadata={
                "method": self.name,
                "units": "map_units",
                "downsample": self.config.downsample,
                "num_source_points": int(pts.shape[0]),
            },
        )

    def _select_poses(self, map_result: MapResult) -> list[CameraPose]:
        poses = list(map_result.trajectory)
        if not poses:
            return []
        n = self.config.max_frames
        if n <= 0 or n >= len(poses):
            return poses
        idx = np.linspace(0, len(poses) - 1, n, dtype=int)
        return [poses[i] for i in idx]

    def _point_depths_in_cameras(self, map_result: MapResult, pts: np.ndarray) -> dict[int, float]:
        """Depth of each map point in its first observing camera (or first pose)."""
        depths: dict[int, float] = {}
        if not map_result.trajectory:
            return depths
        by_frame = {pose.frame_id: pose for pose in map_result.trajectory}
        pose0 = map_result.trajectory.poses[0]
        for mp in map_result.map_points:
            reference = by_frame.get(min(mp.observations), pose0) if mp.observations else pose0
            inv = reference.inverse()
            Xc = inv.transform_point(mp.xyz)
            depths[mp.id] = float(Xc[2]) if Xc[2] > 0 else float("nan")
        return depths

    def _depth_for_pose(
        self,
        pose: CameraPose,
        intr: CameraIntrinsics,
        pts_world: np.ndarray,
    ) -> DepthMap:
        ds = max(1, int(self.config.downsample))
        h = max(1, intr.height // ds)
        w = max(1, intr.width // ds)

        grid = np.full((h, w), np.nan, dtype=np.float32)
        inv = pose.inverse()
        # X_cam = R^T (X_world - t_world)  via inverse pose
        Xc = inv.transform_point(pts_world)
        z = Xc[:, 2]
        valid = np.isfinite(Xc).all(axis=1) & (z > 1e-6)
        if not np.any(valid):
            return DepthMap(
                frame_id=pose.frame_id,
                timestamp=pose.timestamp,
                depth=grid,
                valid_ratio=0.0,
                source_width=intr.width,
                source_height=intr.height,
            )

        x = Xc[valid, 0] / z[valid]
        y = Xc[valid, 1] / z[valid]
        u = intr.fx * x + intr.cx
        v = intr.fy * y + intr.cy
        z_valid = z[valid]

        # Reject points outside the source image before quantizing grid cells.
        # Clipping their coordinates would create false depths along the border.
        in_frame = (u >= 0) & (u < intr.width) & (v >= 0) & (v < intr.height)
        u, v, z_valid = u[in_frame], v[in_frame], z_valid[in_frame]

        # Scale to downsampled grid
        u_s = np.clip((u / ds).astype(np.int32), 0, w - 1)
        v_s = np.clip((v / ds).astype(np.int32), 0, h - 1)

        # Keep nearest depth when several points hit the same cell
        for ui, vi, zi in zip(u_s, v_s, z_valid, strict=False):
            prev = grid[vi, ui]
            if not np.isfinite(prev) or zi < prev:
                grid[vi, ui] = np.float32(zi)

        if self.config.fill_radius > 0:
            grid = _nn_fill(grid, self.config.fill_radius)

        valid_ratio = float(np.isfinite(grid).mean())
        return DepthMap(
            frame_id=pose.frame_id,
            timestamp=pose.timestamp,
            depth=grid,
            valid_ratio=valid_ratio,
            source_width=intr.width,
            source_height=intr.height,
        )


def _nn_fill(grid: np.ndarray, radius: int) -> np.ndarray:
    """Fill holes from original samples within a bounded square radius.

    This visual approximation chooses the nearest surface depth in the local
    window. Filled cells never become new sources, and measured cells keep
    their original values.
    """
    import cv2

    out = grid.copy()
    valid = np.isfinite(grid)
    if radius <= 0 or not np.any(valid):
        return out

    source = np.where(valid, grid, np.inf)
    size = 2 * int(radius) + 1
    local_minimum = cv2.erode(
        source,
        np.ones((size, size), dtype=np.uint8),
        borderType=cv2.BORDER_CONSTANT,
        borderValue=float("inf"),
    )
    holes_with_support = ~valid & np.isfinite(local_minimum)
    out[holes_with_support] = local_minimum[holes_with_support]
    return out
