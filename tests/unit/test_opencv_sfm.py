"""Geometry regressions: observable bootstrap, one scale, persistent landmarks."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from visualmap.mapping.triangulation import reprojection_error, triangulate_two_view
from visualmap.slam.backends import create_backend
from visualmap.slam.backends.opencv_sfm import OpenCVSFMBackend, OpenCVSFMConfig, _Track
from visualmap.slam.types import CameraIntrinsics, CameraPose, Frame, MapPoint


def _camera(dist=(0.0, 0.0, 0.0, 0.0, 0.0)):
    return CameraIntrinsics(500.0, 510.0, 320.0, 240.0, 640, 480, dist)


def _scene(n=220):
    rng = np.random.default_rng(123)
    return np.column_stack(
        (rng.uniform(-2, 2, n), rng.uniform(-1.5, 1.5, n), rng.uniform(4, 10, n))
    )


def _project(xyz, pose, intrinsics):
    inv = pose.inverse()
    rvec, _ = cv2.Rodrigues(inv.R)
    xy, _ = cv2.projectPoints(xyz, rvec, inv.t, intrinsics.K, intrinsics.dist_coeffs)
    return xy.reshape(-1, 2)


class _ExactFeatureSFM(OpenCVSFMBackend):
    """Geometry oracle supplies real perspective correspondences, bypassing texture."""

    def __init__(self, xyz, poses):
        super().__init__(
            OpenCVSFMConfig(
                bootstrap_min_landmarks=50,
                kf_min_translation=0.1,
                bootstrap_min_parallax_deg=1.0,
                tri_min_parallax_deg=0.7,
                bootstrap_min_rotation_compensated_px=1.0,
            )
        )
        self.truth_points = xyz
        self.truth_poses = poses
        self.current_index = 0

    def process_frame(self, frame):
        self.current_index = frame.index
        super().process_frame(frame)

    def _add_features(self, gray, pose):
        if self._tracks or self._bootstrapped:
            return
        pixels = _project(self.truth_points, self.truth_poses[self.current_index], self._intrinsics)
        for fid, pixel in enumerate(pixels):
            self._tracks[fid] = _Track(fid, pixel.copy(), pixel.copy(), pose, pose.frame_id)
        self._next_track_id = len(pixels)

    def _track_features(self, gray):
        pixels = _project(self.truth_points, self.truth_poses[self.current_index], self._intrinsics)
        for fid, track in self._tracks.items():
            track.pixel = pixels[fid].copy()

    def _save_descriptors(self, gray):
        pass


def _run_exact(poses):
    backend = _ExactFeatureSFM(_scene(), poses)
    backend.initialize(_camera())
    for i in range(len(poses)):
        backend.process_frame(Frame(i, i / 30, np.zeros((480, 640), np.uint8)))
    return backend


def test_factory_registers_sfm_and_preserves_legacy_vo_options():
    for name in ("opencv_sfm", "pnp_vo", "sfm"):
        backend = create_backend(name, n_features=1200, unknown_option="unused")
        assert isinstance(backend, OpenCVSFMBackend)
        assert backend.config.n_features == 1200
        assert backend.config.extra == {"unknown_option": "unused"}
    with pytest.raises(RuntimeError):
        OpenCVSFMBackend().process_frame(Frame(0, 0, np.zeros((10, 10), np.uint8)))


def test_prolonged_loss_starts_independent_observable_submap_without_reusing_world():
    poses = [CameraPose.from_rt(np.eye(3), [i * 0.15, 0, 0], i) for i in range(9)]
    poses += [CameraPose.from_rt(np.eye(3), [(i - 9) * 0.16, 0.1, 0], i) for i in range(9, 20)]
    backend = _ExactFeatureSFM(_scene(), poses)
    backend.initialize(_camera())
    for index in range(9):
        backend.process_frame(Frame(index, index / 30, np.zeros((480, 640), np.uint8)))
    previous = backend.get_result()
    old_world = previous.metadata["world_frame_id"]
    old_points = {point.id: point.xyz.copy() for point in previous.map_points}
    old_pose = previous.trajectory.poses[8].matrix.copy()
    backend._lost_frames = backend.config.submap_after_lost_frames
    for index in range(9, len(poses)):
        backend.process_frame(Frame(index, index / 30, np.zeros((480, 640), np.uint8)))
    result = backend.get_result()
    assert result.metadata["scale_initializations"] == 2
    assert result.metadata["scale_consistency_scope"] == "within_world_frame_id"
    assert result.metadata["global_scale_consistent"] is False
    assert result.metadata["submaps_registered"] is False
    assert result.metadata["world_frame_id"] != old_world
    assert result.metadata["frame_quality"]["19"]["pose_valid"] is True
    np.testing.assert_array_equal(result.trajectory.poses[8].matrix, old_pose)
    for point in result.map_points:
        if point.id in old_points:
            np.testing.assert_array_equal(point.xyz, old_points[point.id])
            assert result.metadata["landmark_world_frame_ids"][str(point.id)] == old_world
    assert len(result.map_points) > len(old_points)
    assert result.metadata["geometry_coverage"]["last_valid_frame"] == 19


def test_recovery_submap_configuration_is_bounded_and_can_be_disabled():
    for name in (
        "relocalization_keyframes", "landmark_descriptor_history", "submap_after_lost_frames"
    ):
        for invalid in (0, -1, 1.5, True):
            with pytest.raises(ValueError, match=name):
                OpenCVSFMConfig(**{name: invalid})
    poses = [CameraPose.from_rt(np.eye(3), [i * 0.15, 0, 0], i) for i in range(10)]
    backend = _run_exact(poses)
    backend.config.recovery_submaps = False
    backend._lost_frames = 100
    backend.process_frame(Frame(9, 0.4, np.zeros((480, 640), np.uint8)))
    assert backend.get_result().metadata["scale_initializations"] == 1


def test_verified_keyframe_flow_recovers_deleted_incremental_tracks_in_existing_world():
    rng = np.random.default_rng(212)
    gray = cv2.GaussianBlur(rng.integers(0, 256, (480, 640), np.uint8), (5, 5), 0.8)
    backend = OpenCVSFMBackend(OpenCVSFMConfig(n_features=400))
    backend.initialize(_camera())
    backend._image_scale = 1.0
    backend._bootstrapped = True
    backend._world_frame_id = "retained-world"
    backend._add_features(gray, CameraPose.identity(0))
    for track in backend._tracks.values():
        x, y = track.pixel
        point = [(x - 320) * 5 / 500, (y - 240) * 5 / 510, 5]
        backend._landmarks[track.id] = MapPoint(track.id, point, observations={0: track.id})
        track.landmark_id = track.id
    backend._save_descriptors(gray)
    backend._tracks = {}
    moved = cv2.warpAffine(gray, np.float32([[1, 0, -5], [0, 1, 0]]), (640, 480))
    ok, quality = backend._relocalize_keyframes(moved, Frame(2, 2 / 30, moved))
    assert ok
    assert quality["relocalization_method"] == "verified_keyframe_flow_pnp"
    assert quality["reused_world_landmarks"] > 250
    np.testing.assert_allclose(backend._pose.t, [0.05, 0, 0], atol=0.002)
    assert backend._world_frame_id == "retained-world"
    assert all(track.landmark_id in backend._landmarks for track in backend._tracks.values())


def test_nonuniform_speed_is_estimated_in_one_bootstrap_scale():
    # Equal time gaps, radically different distances, and a stop. Normalizing
    # each recovered t to 1 cannot satisfy this trajectory or heldout pixels.
    centers = [0.0, 0.003, 0.01, 0.035, 0.12, 0.25, 0.29, 0.55, 0.55, 0.59, 0.90]
    poses = [CameraPose.from_rt(np.eye(3), [x, 0, 0], i) for i, x in enumerate(centers)]
    backend = _run_exact(poses)
    result = backend.get_result()
    assert result.metadata["scale_initializations"] == 1
    assert result.metadata["scale_consistent"] is True
    positions = result.trajectory.positions()
    scale = positions[-1, 0] / centers[-1]
    np.testing.assert_allclose(positions[:, 0], np.array(centers) * scale, atol=0.003)
    np.testing.assert_allclose(positions[:, 1:], 0, atol=0.003)
    assert np.linalg.norm(positions[8] - positions[7]) < 0.001
    last_quality = result.metadata["frame_quality"][str(len(poses) - 1)]
    assert last_quality["heldout_count"] >= 10
    assert last_quality["heldout_reprojection_median_px"] < 0.01
    assert last_quality["positive_depth_ratio"] == 1.0
    # Every reconstructed map point also predicts its corresponding raw
    # physical scene point. It was not scored solely on its generating pair.
    heldout_pose = result.trajectory.poses[-1]
    oracle_pixels = _project(_scene(), poses[-1], _camera())
    errors = [
        reprojection_error(
            point.xyz, heldout_pose, _camera(), oracle_pixels[point.observations[len(poses) - 1]]
        )
        for point in result.map_points
        if len(poses) - 1 in point.observations
    ]
    assert np.median(errors) < 0.01


def test_pure_rotation_never_invents_depth_or_translation():
    poses = []
    for i in range(20):
        R, _ = cv2.Rodrigues(np.array([0.0, i * 0.012, 0.0]))
        poses.append(CameraPose.from_rt(R, [0, 0, 0], i))
    result = _run_exact(poses).get_result()
    assert result.metadata["slam_status"] == "unobservable"
    assert result.metadata["scale_consistent"] is False
    assert result.metadata["scale_initializations"] == 0
    assert not result.map_points
    np.testing.assert_array_equal(result.trajectory.positions(), 0)
    assert not any(q["pose_valid"] for q in result.metadata["frame_quality"].values())


def test_tracked_landmarks_are_reused_instead_of_retriangulated_copies():
    poses = [CameraPose.from_rt(np.eye(3), [i * 0.12, 0, 0], i) for i in range(22)]
    backend = _run_exact(poses)
    points = backend.get_result().map_points
    assert 50 <= len(points) <= len(_scene())
    feature_ids = [next(iter(p.observations.values())) for p in points]
    assert len(set(feature_ids)) == len(points)
    assert sum(len(p.observations) >= 15 for p in points) >= 40
    assert len({p.id for p in points}) == len(points)


def test_pnp_holds_pose_on_failure_and_keeps_same_world_map():
    poses = [CameraPose.from_rt(np.eye(3), [i * 0.15, 0, 0], i) for i in range(9)]
    backend = _run_exact(poses)
    before = backend.get_result()
    last_pose = before.trajectory.poses[-1].matrix.copy()
    ids_before = [point.id for point in before.map_points]
    # Lose all feature associations and feed an uninformative image. The new
    # backend must not append a new normalized essential-matrix motion.
    backend._tracks = {}
    backend.current_index = 8
    OpenCVSFMBackend.process_frame(backend, Frame(9, 0.3, np.zeros((480, 640), np.uint8)))
    result = backend.get_result()
    np.testing.assert_array_equal(result.trajectory.poses[-1].matrix, last_pose)
    assert result.metadata["frame_quality"]["9"]["pose_valid"] is False
    assert result.metadata["scale_initializations"] == 1
    assert [point.id for point in result.map_points] == ids_before


def test_pnp_heldout_pixels_are_not_used_by_lm_and_expose_bad_predictions():
    backend = OpenCVSFMBackend()
    backend.initialize(_camera())
    xyz = _scene(200)
    truth = CameraPose.from_rt(np.eye(3), [0.2, -0.04, 0.1])
    pixels = _project(xyz, truth, _camera())
    # Only withheld pixels are corrupt: training residual stays tiny, while
    # independent prediction fails and must reject this frame.
    pixels[np.arange(len(xyz)) % 5 == 0] += [25.0, -15.0]
    ok, _, _, quality = backend._estimate_pose(xyz, pixels, 3, 0.1)
    assert ok is False
    assert quality["inlier_reprojection_rms_px"] < 0.01
    assert quality["heldout_reprojection_median_px"] > 20
    assert quality["reason"] == "pnp_validation_failed"


def test_rotation_after_bootstrap_changes_orientation_without_translation_step():
    poses = [CameraPose.from_rt(np.eye(3), [x, 0, 0], i) for i, x in enumerate([0, 0.1, 0.25, 0.4])]
    for i in range(4, 14):
        R, _ = cv2.Rodrigues(np.array([0.0, (i - 3) * 0.02, 0.0]))
        poses.append(CameraPose.from_rt(R, [0.4, 0, 0], i))
    result = _run_exact(poses).get_result()
    centers = result.trajectory.positions()
    np.testing.assert_allclose(centers[4:], np.repeat(centers[3:4], 10, axis=0), atol=0.005)
    np.testing.assert_allclose(result.trajectory.poses[-1].R, poses[-1].R, atol=0.005)
    assert result.metadata["scale_initializations"] == 1


def test_relocalization_estimates_pose_in_existing_world_and_reuses_point_ids():
    rng = np.random.default_rng(541)
    xyz = _scene(200)
    descriptors = rng.integers(0, 256, (len(xyz), 32), dtype=np.uint8)
    backend = OpenCVSFMBackend()
    backend.initialize(_camera())
    backend._bootstrapped = True
    backend._world_frame_id = "sfm_bootstrap_0"
    backend._landmarks = {
        i: MapPoint(i, point, descriptors[i].copy()) for i, point in enumerate(xyz)
    }
    R, _ = cv2.Rodrigues(np.array([0.01, -0.04, 0.02]))
    truth = CameraPose.from_rt(R, [0.5, -0.1, 0.08], 20)
    pixels = _project(xyz, truth, _camera())

    class _CurrentFeatures:
        def detectAndCompute(self, gray, mask):
            return [cv2.KeyPoint(float(x), float(y), 31) for x, y in pixels], descriptors

    backend._orb = _CurrentFeatures()
    ok, quality = backend._relocalize(
        np.zeros((480, 640), np.uint8), Frame(20, 0.7, np.zeros((480, 640), np.uint8))
    )
    assert ok is True
    assert quality["reused_world_landmarks"] == len(xyz)
    np.testing.assert_allclose(backend._pose.matrix, truth.matrix, atol=1e-6)
    assert set(t.landmark_id for t in backend._tracks.values()) == set(range(len(xyz)))
    assert backend.get_result().metadata["scale_initializations"] == 1
    assert backend.get_result().metadata["world_frame_id"] == "sfm_bootstrap_0"


def test_triangulation_and_reprojection_use_brown_conrady_calibration():
    intr = _camera((-0.18, 0.025, 0.004, -0.003, 0.002))
    xyz = _scene(100)
    a = CameraPose.identity(0)
    R, _ = cv2.Rodrigues(np.array([0.02, -0.04, 0.01]))
    b = CameraPose.from_rt(R, [0.7, 0.06, 0.04], 1)
    pixels_a, pixels_b = _project(xyz, a, intr), _project(xyz, b, intr)
    ids = np.arange(len(xyz))
    points = triangulate_two_view(
        pixels_a,
        pixels_b,
        a,
        b,
        intr,
        feature_ids_a=ids,
        feature_ids_b=ids,
        frame_a=0,
        frame_b=1,
    )
    assert len(points) == len(xyz)
    for point in points:
        fid = point.observations[0]
        np.testing.assert_allclose(point.xyz, xyz[fid], atol=1e-4)
        assert reprojection_error(point.xyz, b, intr, pixels_b[fid]) < 0.002


def test_real_klt_preserves_feature_id_through_forward_backward_check():
    rng = np.random.default_rng(404)
    textured = cv2.GaussianBlur(rng.integers(0, 255, (240, 320), dtype=np.uint8), (5, 5), 1.0)
    backend = OpenCVSFMBackend(OpenCVSFMConfig(n_features=400))
    backend.initialize(CameraIntrinsics(250, 250, 160, 120, 320, 240))
    backend.process_frame(Frame(0, 0, textured))
    # Compare features safely inside the image; translated boundary pixels
    # have no physical correspondence and the FB check should reject them.
    original = {
        fid: track.pixel.copy()
        for fid, track in backend._tracks.items()
        if 50 < track.pixel[0] < 270 and 50 < track.pixel[1] < 190
    }
    shifted = cv2.warpAffine(textured, np.float32([[1, 0, 3], [0, 1, 2]]), (320, 240))
    backend.process_frame(Frame(1, 1 / 30, shifted))
    surviving = [fid for fid in backend._tracks if fid in original]
    assert len(surviving) >= 0.8 * len(original)
    deltas = [backend._tracks[fid].pixel - original[fid] for fid in surviving]
    np.testing.assert_allclose(np.median(deltas, axis=0), [3, 2], atol=0.03)


def test_pnp_rejects_noisy_tiny_patch_even_when_reprojection_would_be_small():
    # Actual perspective observations plus 0.6px noise. Such a tiny support can
    # move the camera far away while keeping a subpixel fitting residual.
    rng = np.random.default_rng(7)
    xyz = np.column_stack(
        (rng.uniform(-0.002, 0.002, 40), rng.uniform(-0.0014, 0.0014, 40), rng.uniform(4, 9, 40))
    )
    truth = CameraPose.from_rt(np.eye(3), [0.3, 0.04, 0.0])
    pixels = _project(xyz, truth, _camera()) + rng.normal(0, 0.6, (40, 2))
    backend = OpenCVSFMBackend()
    backend.initialize(_camera())
    ok, pose, _, quality = backend._estimate_pose(xyz, pixels, 1, 1 / 30)
    assert not ok and pose is None
    assert quality["reason"] == "pnp_insufficient_image_support"
    assert quality["image_support_minor_span_ratio"] < 0.01


def test_pnp_support_guard_accepts_wide_planar_scene():
    # Image support matters; scene planarity itself must not disqualify a wall.
    rng = np.random.default_rng(19)
    xyz = np.column_stack((rng.uniform(-2, 2, 100), rng.uniform(-1.5, 1.5, 100), np.full(100, 6.0)))
    R, _ = cv2.Rodrigues(np.array([0.02, -0.07, 0.01]))
    truth = CameraPose.from_rt(R, [0.3, 0.04, 0.1])
    backend = OpenCVSFMBackend()
    backend.initialize(_camera())
    ok, pose, _, quality = backend._estimate_pose(xyz, _project(xyz, truth, _camera()), 1, 1 / 30)
    assert ok
    np.testing.assert_allclose(pose.matrix, truth.matrix, atol=1e-5)
    assert quality["image_support_area_ratio"] > backend.config.pose_min_image_area_ratio


def test_pnp_rechecks_final_support_after_lm_changes_ransac_inliers():
    # RANSAC initially accepts >=20 points, but LM pushes boundary inliers out.
    # This is observed with real OpenCV estimation, rather than a mocked mask.
    rng = np.random.default_rng(12)
    xyz = np.column_stack(
        (rng.uniform(-2, 2, 30), rng.uniform(-1.5, 1.5, 30), rng.uniform(4, 9, 30))
    )
    intr = CameraIntrinsics(500, 500, 320, 240, 640, 480)
    truth = CameraPose.from_rt(np.eye(3), [0.3, 0.04, 0.0])
    pixels = _project(xyz, truth, intr) + rng.normal(0, 2.5, (30, 2))
    backend = OpenCVSFMBackend()
    backend.initialize(intr)
    cv2.setRNGSeed(12)
    ok, pose, _, quality = backend._estimate_pose(xyz, pixels, 1, 1 / 30)
    assert not ok and pose is None
    assert quality["reason"] == "pnp_refined_support_failed"
    assert quality["inliers"] < backend.config.min_inliers


def test_sfm_resolution_change_holds_pose_without_reusing_wrong_scale_klt():
    poses = [CameraPose.from_rt(np.eye(3), [i * 0.15, 0, 0], i) for i in range(9)]
    backend = _run_exact(poses)
    before = backend.get_result()
    last_pose = before.trajectory.poses[-1].matrix.copy()
    world = before.metadata["world_frame_id"]
    OpenCVSFMBackend.process_frame(backend, Frame(9, 0.3, np.zeros((240, 320), np.uint8)))
    result = backend.get_result()
    np.testing.assert_array_equal(result.trajectory.poses[-1].matrix, last_pose)
    assert result.metadata["input_invalid_frames"] == 1
    assert result.metadata["world_frame_id"] == world
    assert result.metadata["scale_initializations"] == 1
    quality = result.metadata["frame_quality"]["9"]
    assert quality["status"] == "input_invalid"
    assert quality["pose_valid"] is False
    assert quality["reason"] == "camera_intrinsics_frame_size_mismatch"
    assert not backend._tracks and backend._prev_gray is None


def test_sfm_rejects_float_image_explicitly_but_frame_keeps_float_compatibility():
    frame = Frame(0, 0, np.zeros((480, 640), np.float32))
    backend = OpenCVSFMBackend()
    backend.initialize(_camera())
    backend.process_frame(frame)
    result = backend.get_result()
    assert (
        result.metadata["frame_quality"]["0"]["reason"]
        == "unsupported_sfm_image_dtype_requires_uint8"
    )
    assert result.metadata["frame_quality"]["0"]["pose_valid"] is False
    assert not result.metadata["scale_consistent"]


def test_invalid_size_before_bootstrap_cannot_contaminate_later_history():
    backend = OpenCVSFMBackend()
    backend.initialize(_camera())
    backend.process_frame(Frame(0, 0, np.zeros((480, 640), np.uint8)))
    backend.process_frame(Frame(1, 1 / 30, np.zeros((240, 320), np.uint8)))
    rng = np.random.default_rng(40)
    backend.process_frame(Frame(2, 2 / 30, rng.integers(0, 255, (480, 640), np.uint8)))
    assert backend._bootstrap_anchor.frame_id == 2
    assert [row[0] for row in backend._bootstrap_history] == [2]
    assert backend.get_result().metadata["scale_initializations"] == 0


@pytest.mark.parametrize("bad_kind", ["nonfinite_xyz", "nonfinite_pixels", "different_counts"])
def test_pnp_bad_correspondences_fail_before_opencv(bad_kind):
    xyz = _scene(40)
    pixels = _project(xyz, CameraPose.identity(), _camera())
    if bad_kind == "nonfinite_xyz":
        xyz[0, 0] = np.nan
    elif bad_kind == "nonfinite_pixels":
        pixels[0, 0] = np.inf
    else:
        pixels = pixels[:-1]
    backend = OpenCVSFMBackend()
    backend.initialize(_camera())
    ok, pose, _, quality = backend._estimate_pose(xyz, pixels, 0, 0)
    assert not ok and pose is None
    assert quality["reason"] == "invalid_map_correspondences"
