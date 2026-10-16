from pathlib import Path

import numpy as np
import pytest

from visualmap.camera import CameraModel, load_intrinsics, save_intrinsics
from visualmap.slam.types import CameraIntrinsics


def test_project_positive_z():
    intr = CameraIntrinsics(fx=100, fy=100, cx=50, cy=40, width=100, height=80)
    model = CameraModel(intr)
    uv = model.project(np.array([[0.0, 0.0, 1.0]]))
    assert uv.shape == (1, 2)
    assert np.allclose(uv[0], [50, 40])


def test_project_rejects_nonpositive_z():
    intr = CameraIntrinsics(fx=100, fy=100, cx=50, cy=40, width=100, height=80)
    model = CameraModel(intr)
    with pytest.raises(ValueError):
        model.project(np.array([[0.0, 0.0, 0.0]]))


def test_save_load_roundtrip(tmp_path: Path):
    intr = CameraIntrinsics(
        fx=800.5, fy=801.5, cx=320, cy=240, width=640, height=480, dist=(0.1, -0.05, 0, 0, 0)
    )
    path = tmp_path / "cam.yaml"
    save_intrinsics(intr, path)
    loaded = load_intrinsics(path)
    assert loaded.fx == pytest.approx(800.5)
    assert loaded.has_distortion
    assert loaded.width == 640


def test_load_opencv_style(tmp_path: Path):
    path = tmp_path / "ocv.yaml"
    path.write_text(
        """
camera_matrix: [[600, 0, 320], [0, 610, 240], [0, 0, 1]]
dist_coeffs: [0.01, 0.02, 0, 0, 0]
image_size: [640, 480]
""",
        encoding="utf-8",
    )
    intr = load_intrinsics(path)
    assert intr.fx == 600
    assert intr.fy == 610
    assert intr.width == 640


def test_undistort_noop_without_distortion():
    intr = CameraIntrinsics(fx=100, fy=100, cx=50, cy=40, width=100, height=80)
    model = CameraModel(intr)
    img = np.zeros((80, 100, 3), dtype=np.uint8)
    out = model.undistort_image(img)
    assert out is img
