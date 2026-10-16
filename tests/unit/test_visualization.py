"""Visualization helpers (Open3D optional)."""

from __future__ import annotations

import numpy as np
import pytest

from visualmap.slam.types import CameraPose, MapPoint, MapResult, Trajectory
from visualmap.visualization import build_geometry, open3d_available, show_map


def _result() -> MapResult:
    traj = Trajectory()
    traj.append(CameraPose.from_rt(np.eye(3), [0, 0, 0]))
    traj.append(CameraPose.from_rt(np.eye(3), [1, 0, 0]))
    traj.append(CameraPose.from_rt(np.eye(3), [2, 0.5, 0]))
    pts = [MapPoint(id=i, xyz=[i * 0.1, 0.2, 3.0]) for i in range(5)]
    return MapResult(trajectory=traj, map_points=pts)


def test_open3d_available_flag():
    assert isinstance(open3d_available(), bool)


@pytest.mark.skipif(not open3d_available(), reason="open3d not installed")
def test_build_geometry():
    pcd, lineset, markers, frame, obj_mesh = build_geometry(_result())
    assert pcd is not None
    assert len(pcd.points) == 5
    assert len(lineset.points) == 3


@pytest.mark.skipif(not open3d_available(), reason="open3d not installed")
def test_show_map_raises_without_display_is_ok():
    # Just ensure import surface exists; interactive window not opened in CI.
    assert callable(show_map)
