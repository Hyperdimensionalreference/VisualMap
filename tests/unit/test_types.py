import numpy as np
import pytest

from visualmap.slam.types import CameraIntrinsics, CameraPose, Frame, MapResult, Trajectory


def test_intrinsics_k_matrix():
    intr = CameraIntrinsics(fx=500, fy=510, cx=320, cy=240, width=640, height=480)
    K = intr.K
    assert K[0, 0] == 500
    assert K[1, 1] == 510
    assert K[0, 2] == 320
    assert K[1, 2] == 240
    assert K[2, 2] == 1


def test_intrinsics_rejects_bad_values():
    with pytest.raises(ValueError):
        CameraIntrinsics(fx=-1, fy=1, cx=0, cy=0, width=10, height=10)
    with pytest.raises(ValueError):
        CameraIntrinsics(fx=1, fy=1, cx=0, cy=0, width=0, height=10)
    with pytest.raises(ValueError):
        CameraIntrinsics(fx=1, fy=1, cx=0, cy=0, width=10, height=10, dist=(1, 2, 3))


def test_pose_identity_transform():
    pose = CameraPose.identity()
    p = np.array([1.0, 2.0, 3.0])
    assert np.allclose(pose.transform_point(p), p)


def test_pose_translation_maps_point():
    pose = CameraPose.from_rt(np.eye(3), np.array([10.0, 0.0, 0.0]))
    p = np.array([1.0, 0.0, 0.0])
    assert np.allclose(pose.transform_point(p), [11.0, 0.0, 0.0])
    assert np.allclose(pose.camera_center, [10.0, 0.0, 0.0])


def test_pose_inverse_roundtrip():
    R = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float64)
    pose = CameraPose.from_rt(R, np.array([1.0, 2.0, 3.0]), frame_id=7, timestamp=1.5)
    inv = pose.inverse()
    eye = (inv.matrix @ pose.matrix)
    assert np.allclose(eye, np.eye(4), atol=1e-10)
    assert inv.frame_id == 7


def test_pose_rejects_bad_matrix():
    with pytest.raises(ValueError):
        CameraPose(np.eye(3))
    with pytest.raises(ValueError):
        CameraPose(np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [1, 0, 0, 1]], dtype=float))


def test_frame_meta():
    img = np.zeros((48, 64, 3), dtype=np.uint8)
    f = Frame(index=3, timestamp=0.1, image=img)
    assert f.width == 64
    assert f.height == 48
    assert f.shape == (48, 64, 3)


def test_trajectory_positions():
    traj = Trajectory()
    traj.append(CameraPose.from_rt(np.eye(3), np.array([0.0, 0.0, 0.0])))
    traj.append(CameraPose.from_rt(np.eye(3), np.array([1.0, 0.0, 0.0])))
    pos = traj.positions()
    assert pos.shape == (2, 3)
    assert np.allclose(pos[1], [1, 0, 0])


def test_map_result_summary():
    result = MapResult(trajectory=Trajectory(), metadata={"stage": "test"})
    s = result.summary()
    assert s["num_poses"] == 0
    assert s["stage"] == "test"


def test_pose_dict_roundtrip():
    pose = CameraPose.from_rt(np.eye(3), np.array([1.0, 2.0, 3.0]), frame_id=2, timestamp=0.5)
    restored = CameraPose.from_dict(pose.to_dict())
    assert restored.frame_id == 2
    assert np.allclose(restored.matrix, pose.matrix)


@pytest.mark.parametrize("name", ["fx", "fy", "cx", "cy"])
@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_intrinsics_rejects_nonfinite_camera_values(name, value):
    values = dict(fx=500, fy=500, cx=320, cy=240, width=640, height=480)
    values[name] = value
    with pytest.raises(ValueError, match="finite"):
        CameraIntrinsics(**values)


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_intrinsics_rejects_nonfinite_distortion(value):
    with pytest.raises(ValueError, match="finite"):
        CameraIntrinsics(500, 500, 320, 240, 640, 480, (value, 0, 0, 0, 0))


@pytest.mark.parametrize(
    "image",
    [np.zeros((0, 10)), np.zeros((10, 0)), np.zeros((10, 10, 2)), np.full((10, 10), np.nan)],
)
def test_frame_rejects_malformed_images(image):
    with pytest.raises(ValueError):
        Frame(0, 0, image)


@pytest.mark.parametrize(
    "index,timestamp", [(-1, 0), (1.5, 0), (0, np.nan), (0, np.inf), (0, -0.1)]
)
def test_frame_rejects_invalid_identity_and_time(index, timestamp):
    with pytest.raises(ValueError):
        Frame(index, timestamp, np.zeros((10, 10), np.uint8))


@pytest.mark.parametrize("channels", [None, 1, 3, 4])
def test_frame_preserves_supported_image_representations(channels):
    shape = (10, 10) if channels is None else (10, 10, channels)
    image = np.zeros(shape, np.float32)
    frame = Frame(0, 0, image)
    assert frame.image is image


def test_pose_rejects_nonfinite_transform():
    mat = np.eye(4)
    mat[0, 3] = np.inf
    with pytest.raises(ValueError, match="finite"):
        CameraPose(mat)
