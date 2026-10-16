"""Unit tests for OpenCVVOBackend."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from visualmap.slam.backends.opencv_vo import OpenCVVOBackend, OpenCVVOConfig, create_backend
from visualmap.slam.types import CameraIntrinsics, Frame


def _intrinsics(w: int = 160, h: int = 120) -> CameraIntrinsics:
    return CameraIntrinsics(
        fx=120.0, fy=120.0, cx=w / 2.0, cy=h / 2.0, width=w, height=h
    )


def _textured_frame(index: int, w: int = 160, h: int = 120, shift: int = 0) -> Frame:
    """Synthetic frame with moving checker-like texture for ORB to track."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    rng = np.random.default_rng(1234 + index)
    noise = rng.integers(0, 255, size=(h, w), dtype=np.uint8)
    img[:, :, 0] = noise
    img[:, :, 1] = noise // 2
    img[:, :, 2] = 255 - noise
    # Strong moving corners
    for i in range(5):
        for j in range(4):
            x = 10 + i * 28 + shift
            y = 10 + j * 28
            color = (int((i * 40 + index * 3) % 255), 255, 80)
            cv2.rectangle(img, (x, y), (x + 16, y + 16), color, -1)
    return Frame(index=index, timestamp=index / 30.0, image=img)


def test_factory_opencv_vo():
    backend = create_backend("opencv_vo")
    assert isinstance(backend, OpenCVVOBackend)


def test_factory_unknown():
    with pytest.raises(ValueError):
        create_backend("nope")


def test_factory_orb_not_implemented():
    with pytest.raises(NotImplementedError):
        create_backend("orb_slam3")


def test_process_requires_initialize():
    backend = OpenCVVOBackend()
    with pytest.raises(RuntimeError):
        backend.process_frame(_textured_frame(0))


def test_first_frame_identity():
    backend = OpenCVVOBackend()
    backend.initialize(_intrinsics())
    backend.process_frame(_textured_frame(0))
    result = backend.get_result()
    assert len(result.trajectory) == 1
    pose = result.trajectory.poses[0]
    assert pose.frame_id == 0
    assert np.allclose(pose.matrix, np.eye(4))


def test_trajectory_grows_with_frames():
    cfg = OpenCVVOConfig(min_matches=8, min_inliers=5, n_features=800)
    backend = OpenCVVOBackend(cfg)
    backend.initialize(_intrinsics())
    for i in range(12):
        backend.process_frame(_textured_frame(i, shift=i * 2))
    result = backend.get_result()
    assert len(result.trajectory) == 12
    assert result.metadata["frames_processed"] == 12
    assert result.metadata["frames_tracked"] >= 1
    assert result.intrinsics is not None


def test_pose_translation_scale_normalized():
    cfg = OpenCVVOConfig(
        min_matches=8, min_inliers=5, n_features=800, default_translation_norm=0.5
    )
    backend = OpenCVVOBackend(cfg)
    backend.initialize(_intrinsics())
    for i in range(15):
        backend.process_frame(_textured_frame(i, shift=i * 3))
    result = backend.get_result()
    # At least one successful step should have finite positions
    pos = result.trajectory.positions()
    assert pos.shape[0] == 15
    assert np.all(np.isfinite(pos))
