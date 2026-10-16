"""Geometric evidence regressions: background fill is not a 3D observation."""

import numpy as np
import pytest

from visualmap.depth.types import DepthMap
from visualmap.detection import Detection, LiftConfig, lift_detection
from visualmap.slam.types import CameraIntrinsics, CameraPose, MapPoint


def _intr():
    return CameraIntrinsics(100, 100, 50, 40, 100, 80)


def _det(frame=0):
    return Detection(frame, frame / 10, 39, "bottle", 0.97, (30, 20, 70, 60))


def _points(z=5.0, observed=True, frame=0):
    return [
        MapPoint(
            index, [x, 0, z], observations={frame: index, frame + 1: index} if observed else {}
        )
        for index, x in enumerate([-0.1, 0, 0.1])
    ]


def test_no_depth_does_not_invent_position_even_with_legacy_default():
    obj = lift_detection(_det(), CameraPose.identity(), _intr(), config=LiftConfig(default_depth=5))
    assert obj.xyz is None and obj.depth is None
    assert obj.metadata["position_status"] == "unavailable"
    assert obj.confidence == 0.97  # Category evidence never becomes geometric evidence.


def test_sparse_grid_background_fill_cannot_locate_a_foreground_object():
    filled = DepthMap(0, 0, np.full((80, 100), 25), source_width=100, source_height=80)
    obj = lift_detection(
        _det(), CameraPose.identity(), _intr(), filled, depth_method="sparse_depth"
    )
    assert obj.xyz is None
    assert (
        obj.metadata["evidence"]["grid_depth_rejected"] == "sparse_grid_fill_is_not_object_evidence"
    )


def test_sparse_position_uses_observed_world_points_in_current_camera():
    pose = CameraPose.from_rt(np.eye(3), [3, 2, 1], frame_id=0)
    points = [
        MapPoint(point.id, pose.transform_point(point.xyz), observations=point.observations)
        for point in _points()
    ]
    obj = lift_detection(_det(), pose, _intr(), map_points=points, depth_method="sparse_depth")
    assert obj.xyz == pytest.approx([3, 2, 6])
    assert obj.depth == pytest.approx(5)
    assert obj.metadata["position_source"] == "sparse_observed"
    assert obj.metadata["position_frame"] == 0
    assert obj.metadata["evidence"]["observed_point_count"] == 3
    assert obj.metadata["position_uncertainty"] > 0
    assert not obj.metadata["uncertainty"]["calibrated_error_bound"]


def test_projected_background_without_current_observation_is_rejected():
    obj = lift_detection(_det(8), CameraPose.identity(frame_id=8), _intr(), map_points=_points())
    assert obj.xyz is None
    assert obj.metadata["evidence"]["observed_point_count"] == 0


def test_projection_only_mode_is_explicitly_tentative():
    obj = lift_detection(
        _det(),
        CameraPose.identity(),
        _intr(),
        map_points=_points(observed=False),
        config=LiftConfig(allow_projected_points=True),
    )
    assert obj.xyz is not None
    assert obj.metadata["position_status"] == "tentative"
    assert obj.metadata["position_source"] == "sparse_projected"
    assert not obj.metadata["evidence"]["occlusion_verified"]


def test_one_observing_frame_cannot_claim_triangulated_support():
    points = _points()
    for point in points:
        point.observations = {0: point.id}
    assert lift_detection(_det(), CameraPose.identity(), _intr(), map_points=points).xyz is None


def test_dominant_depth_layer_rejects_isolated_background_outlier():
    points = _points() + [MapPoint(8, [0, 0, 30], observations={0: 8, 1: 8})]
    obj = lift_detection(_det(), CameraPose.identity(), _intr(), map_points=points)
    assert obj.depth == pytest.approx(5)
    assert obj.metadata["evidence"]["sample_count"] == 3
    assert 8 not in obj.metadata["evidence"]["point_ids"]


def test_disjoint_singleton_depth_layers_cannot_establish_position():
    points = [MapPoint(i, [0, 0, z], observations={0: i, 1: i}) for i, z in enumerate([2, 5, 20])]
    obj = lift_detection(_det(), CameraPose.identity(), _intr(), map_points=points)
    assert obj.xyz is None
    assert obj.metadata["evidence"]["reason"] == "no_supported_depth_layer"


def test_depth_map_must_be_from_same_frame():
    depth = DepthMap(1, 0.1, np.full((80, 100), 5))
    obj = lift_detection(_det(), CameraPose.identity(), _intr(), depth)
    assert obj.xyz is None
    assert obj.metadata["evidence"]["reason"] == "depth_frame_mismatch"


def test_background_pixels_at_box_edges_are_not_a_depth_fallback():
    depth = np.full((80, 100), np.nan)
    depth[20:22, 30:70] = 20
    obj = lift_detection(_det(), CameraPose.identity(), _intr(), DepthMap(0, 0, depth))
    assert obj.xyz is None


def test_downsampled_dense_depth_uses_source_dimensions():
    depth = np.full((40, 50), np.nan)
    depth[18:22, 23:27] = 6
    obj = lift_detection(
        _det(),
        CameraPose.identity(),
        _intr(),
        DepthMap(0, 0, depth, source_width=100, source_height=80),
    )
    assert obj.depth == pytest.approx(6)
    assert obj.xyz == pytest.approx([0, 0, 6])


def test_points_behind_camera_cannot_locate_object():
    obj = lift_detection(_det(), CameraPose.identity(), _intr(), map_points=_points(z=-5))
    assert obj.xyz is None


@pytest.mark.parametrize(
    "field", ["center_frac", "max_depth_relative_spread", "depth_cluster_gap_ratio"]
)
@pytest.mark.parametrize("value", [-0.1, 0, 1.01, float("nan"), float("inf"), True, "0.5"])
def test_lift_fraction_settings_reject_invalid_values(field, value):
    with pytest.raises(ValueError, match=field):
        LiftConfig(**{field: value})


@pytest.mark.parametrize("field", ["min_valid_pixels", "min_sparse_points"])
@pytest.mark.parametrize("value", [-1, 0, 1.5, float("nan"), float("inf"), True, "3"])
def test_lift_support_counts_require_positive_integers(field, value):
    with pytest.raises(ValueError, match=field):
        LiftConfig(**{field: value})


def test_lift_setting_boundaries_are_accepted():
    cfg = LiftConfig(
        center_frac=1,
        max_depth_relative_spread=1,
        depth_cluster_gap_ratio=1,
        min_valid_pixels=1,
        min_sparse_points=1,
    )
    assert cfg.center_frac == 1


@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf"), True, "5"])
def test_legacy_default_depth_still_requires_a_valid_configuration(value):
    with pytest.raises(ValueError, match="default_depth"):
        LiftConfig(default_depth=value)


@pytest.mark.parametrize("value", ["false", 0, 1, None])
def test_projected_points_flag_cannot_be_accidentally_enabled_by_string(value):
    with pytest.raises(ValueError, match="allow_projected_points"):
        LiftConfig(allow_projected_points=value)


def _distorted_pixel(camera_point, intr):
    """Independent Brown-Conrady forward equation for calibrated round-trip tests."""
    x, y = np.asarray(camera_point[:2]) / camera_point[2]
    k1, k2, p1, p2, k3 = (*intr.dist, *(0.0 for _ in range(5 - len(intr.dist))))
    r2 = x * x + y * y
    radial = 1 + k1 * r2 + k2 * r2**2 + k3 * r2**3
    distorted_x = x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
    distorted_y = y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
    return intr.fx * distorted_x + intr.cx, intr.fy * distorted_y + intr.cy


@pytest.mark.parametrize("dist", [(0.2, -0.05, 0.01, -0.015), (0.2, -0.05, 0.01, -0.015, 0.01)])
def test_calibrated_dense_depth_backprojects_distorted_video_pixels(dist):
    intr = CameraIntrinsics(300, 260, 320, 240, 640, 480, dist=dist)
    camera_point = np.array([2, 1, 5], dtype=float)
    u, v = _distorted_pixel(camera_point, intr)
    det = Detection(0, 0, 39, "bottle", 0.9, (u - 8, v - 8, u + 8, v + 8))
    pose = CameraPose.from_rt(np.eye(3), [1, 2, 3], frame_id=0)
    depth = DepthMap(0, 0, np.full((480, 640), 5), source_width=640, source_height=480)
    obj = lift_detection(det, pose, intr, depth)
    assert obj.xyz == pytest.approx([3, 3, 8], abs=1e-6)
    assert obj.depth == 5


def test_calibrated_sparse_points_project_into_distorted_detection_box():
    intr = CameraIntrinsics(300, 260, 320, 240, 640, 480, dist=(0.3, -0.02, 0.005, -0.01, 0.005))
    centre = np.array([3.25, 1.25, 5])
    u, v = _distorted_pixel(centre, intr)
    det = Detection(0, 0, 39, "bottle", 0.9, (u - 10, v - 10, u + 10, v + 10))
    points = [
        MapPoint(index, centre + [offset, 0, 0], observations={0: index, 1: index})
        for index, offset in enumerate([-0.01, 0, 0.01])
    ]
    # An invalid point must not enter the nonlinear projection routine.
    points.append(MapPoint(9, [float("nan"), 0, 5], observations={0: 9, 1: 9}))
    obj = lift_detection(det, CameraPose.identity(), intr, map_points=points)
    assert obj.xyz == pytest.approx(centre)
    assert obj.metadata["evidence"]["sample_count"] == 3
    assert obj.metadata["position_source"] == "sparse_observed"
