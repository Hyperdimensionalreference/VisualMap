"""Unit tests for sparse depth estimation and export."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from visualmap.depth import SparseDepthConfig, SparseDepthEstimator, create_depth_estimator
from visualmap.depth.sparse import _nn_fill
from visualmap.export import export_depth
from visualmap.slam.types import CameraIntrinsics, CameraPose, MapPoint, MapResult, Trajectory


def _intr() -> CameraIntrinsics:
    return CameraIntrinsics(fx=100.0, fy=100.0, cx=50.0, cy=40.0, width=100, height=80)


def _result() -> MapResult:
    traj = Trajectory()
    traj.append(CameraPose.identity(frame_id=0, timestamp=0.0))
    traj.append(CameraPose.from_rt(np.eye(3), [0.5, 0, 0], frame_id=1, timestamp=0.1))
    # Points in front of camera
    pts = [
        MapPoint(id=0, xyz=[0.0, 0.0, 5.0]),
        MapPoint(id=1, xyz=[0.2, -0.1, 6.0]),
        MapPoint(id=2, xyz=[-0.1, 0.1, 4.0]),
        MapPoint(id=3, xyz=[0.05, 0.05, 5.5]),
    ]
    return MapResult(trajectory=traj, map_points=pts, intrinsics=_intr())


def test_factory_sparse():
    est = create_depth_estimator("sparse_depth", downsample=2, max_frames=4)
    assert isinstance(est, SparseDepthEstimator)


def test_factory_unknown():
    with pytest.raises(ValueError):
        create_depth_estimator("nope")


def test_sparse_depth_maps():
    est = SparseDepthEstimator(SparseDepthConfig(downsample=2, max_frames=4, fill_radius=3))
    result = est.estimate(_result(), _intr())
    assert len(result.maps) == 2
    dm = result.maps[0]
    assert dm.depth.shape == (40, 50)  # 80/2, 100/2
    assert dm.valid_ratio > 0
    assert np.nanmin(dm.depth) > 0
    assert 0 in result.point_depths


def test_sparse_depth_no_points():
    empty = MapResult(trajectory=Trajectory(), map_points=[], intrinsics=_intr())
    est = SparseDepthEstimator()
    result = est.estimate(empty, _intr())
    assert len(result.maps) == 0


def test_sparse_depth_ignores_points_outside_image():
    traj = Trajectory([CameraPose.identity(frame_id=0)])
    result = MapResult(
        trajectory=traj,
        map_points=[MapPoint(id=0, xyz=[100.0, 0.0, 5.0])],
        intrinsics=_intr(),
    )
    depth = SparseDepthEstimator(SparseDepthConfig(fill_radius=0)).estimate(result, _intr())
    assert len(depth.maps) == 1
    assert np.isnan(depth.maps[0].depth).all()


def test_sparse_depth_does_not_project_another_independent_submap():
    mapping = MapResult(
        trajectory=Trajectory([CameraPose.identity(frame_id=0), CameraPose.identity(frame_id=1)]),
        map_points=[MapPoint(0, [0, 0, 2], observations={0: 0}),
                    MapPoint(1, [0, 0, 8], observations={1: 1})],
        intrinsics=_intr(),
        metadata={
            "submaps": [{"world_frame_id": "a"}, {"world_frame_id": "b"}],
            "frame_quality": {
                "0": {"pose_valid": True, "world_frame_id": "a"},
                "1": {"pose_valid": True, "world_frame_id": "b"},
            },
        },
    )
    depth = SparseDepthEstimator(SparseDepthConfig(fill_radius=0)).estimate(mapping)
    assert np.nanmin(depth.maps[0].depth) == 2
    assert np.nanmin(depth.maps[1].depth) == 8


def test_export_depth(tmp_path: Path):
    est = SparseDepthEstimator(SparseDepthConfig(max_frames=2))
    depth = est.estimate(_result(), _intr())
    written = export_depth(depth, tmp_path, run_name="d1")
    assert "depth_summary" in written
    assert (tmp_path / "d1" / "depth" / "depth_summary.json").is_file()
    npy = list((tmp_path / "d1" / "depth").glob("*.npy"))
    png = list((tmp_path / "d1" / "depth").glob("*.png"))
    assert len(npy) >= 1
    assert len(png) >= 1
    arr = np.load(npy[0])
    assert arr.ndim == 2


def test_export_depth_png_roundtrip_in_unicode_directory(tmp_path: Path):
    import cv2

    depth = SparseDepthEstimator(SparseDepthConfig(max_frames=2)).estimate(_result(), _intr())
    written = export_depth(depth, tmp_path / "三维深度输出", run_name="手机视频")
    png_path = written["depth_png_0"]
    decoded = cv2.imdecode(
        np.frombuffer(png_path.read_bytes(), dtype=np.uint8), cv2.IMREAD_UNCHANGED,
    )
    assert decoded is not None
    assert decoded.dtype == np.uint16
    assert decoded.shape == depth.maps[0].depth.shape
    assert np.max(decoded) == 65535


def test_depth_fill_does_not_propagate_beyond_original_sample_radius():
    grid = np.full((5, 7), np.nan, dtype=np.float32)
    grid[2, 1] = 4.0
    filled = _nn_fill(grid, radius=1)
    expected = np.zeros(grid.shape, dtype=bool)
    expected[1:4, 0:3] = True
    assert np.array_equal(np.isfinite(filled), expected)
    assert np.all(filled[expected] == 4.0)
    assert np.isnan(grid[2, 2])


def test_depth_fill_preserves_measured_samples_and_uses_local_source():
    grid = np.array([[2.0, np.nan, 7.0, np.nan, np.nan]], dtype=np.float32)
    filled = _nn_fill(grid, radius=1)
    np.testing.assert_equal(filled, [[2.0, 2.0, 7.0, 7.0, np.nan]])
