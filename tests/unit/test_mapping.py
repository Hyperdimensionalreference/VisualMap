"""Tests for keyframe policy and two-view triangulation."""

from __future__ import annotations

import numpy as np

from visualmap.mapping import (
    KeyframePolicy,
    KeyframeSelector,
    TriangulationConfig,
    reprojection_error,
    triangulate_two_view,
)
from visualmap.slam.types import CameraIntrinsics, CameraPose, Frame


def _intr() -> CameraIntrinsics:
    return CameraIntrinsics(fx=200.0, fy=200.0, cx=100.0, cy=80.0, width=200, height=160)


def _frame(i: int) -> Frame:
    return Frame(index=i, timestamp=i / 30.0, image=np.zeros((160, 200, 3), dtype=np.uint8))


def test_keyframe_first_is_forced():
    sel = KeyframeSelector(KeyframePolicy(force_first=True))
    pose = CameraPose.identity()
    assert sel.should_insert(_frame(0), pose, n_matches=0, tracked=False) is True
    assert sel.count == 1


def test_keyframe_min_gap():
    policy = KeyframePolicy(min_frames_between=5, min_translation=0.0, min_rotation_deg=0.0)
    sel = KeyframeSelector(policy)
    sel.should_insert(_frame(0), CameraPose.identity(), 100, True)
    assert sel.should_insert(_frame(2), CameraPose.identity(), 100, True) is False
    assert sel.should_insert(_frame(5), CameraPose.identity(), 100, True) is True


def test_keyframe_translation_trigger():
    policy = KeyframePolicy(min_frames_between=1, min_translation=1.0, min_rotation_deg=1e9)
    sel = KeyframeSelector(policy)
    sel.should_insert(_frame(0), CameraPose.identity(), 100, True)
    small = CameraPose.from_rt(np.eye(3), [0.1, 0, 0])
    assert sel.should_insert(_frame(1), small, 100, True) is False
    big = CameraPose.from_rt(np.eye(3), [2.0, 0, 0])
    assert sel.should_insert(_frame(2), big, 100, True) is True


def test_triangulate_two_view_recovers_point():
    intr = _intr()
    # True 3D point in world
    X = np.array([0.5, -0.2, 4.0])
    pose_a = CameraPose.identity()
    pose_b = CameraPose.from_rt(np.eye(3), [1.0, 0.0, 0.0])

    def project(pose: CameraPose) -> np.ndarray:
        Xc = pose.inverse().transform_point(X)
        u = intr.fx * (Xc[0] / Xc[2]) + intr.cx
        v = intr.fy * (Xc[1] / Xc[2]) + intr.cy
        return np.array([u, v])

    pts_a = project(pose_a).reshape(1, 2)
    pts_b = project(pose_b).reshape(1, 2)

    cfg = TriangulationConfig(min_parallax_deg=0.0, max_reproj_error_px=2.0)
    points = triangulate_two_view(pts_a, pts_b, pose_a, pose_b, intr, config=cfg)
    assert len(points) == 1
    assert np.allclose(points[0].xyz, X, atol=1e-4)
    assert reprojection_error(points[0].xyz, pose_a, intr, pts_a[0]) < 1e-3


def test_triangulate_rejects_negative_depth():
    intr = _intr()
    # Point behind both cameras (z negative in cam)
    pose_a = CameraPose.identity()
    pose_b = CameraPose.from_rt(np.eye(3), [1.0, 0.0, 0.0])
    pts = np.array([[100.0, 80.0], [100.0, 80.0]])
    cfg = TriangulationConfig(min_parallax_deg=0.0)
    points = triangulate_two_view(pts, pts, pose_a, pose_b, intr, config=cfg)
    assert points == []


def test_triangulate_empty_input():
    intr = _intr()
    pose = CameraPose.identity()
    assert triangulate_two_view(np.zeros((0, 2)), np.zeros((0, 2)), pose, pose, intr) == []
