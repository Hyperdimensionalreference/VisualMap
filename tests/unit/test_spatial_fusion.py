"""Temporal geometry stays separate from category scores and video overlays."""

import numpy as np
import pytest

from visualmap.depth.types import DepthMap, DepthResult
from visualmap.detection import Detection, LiftConfig, SemanticPipeline, SemanticPipelineConfig
from visualmap.detection.types import SemanticObject, SemanticResult, TrackedDetection
from visualmap.mapping import build_semantic_map
from visualmap.slam.types import CameraIntrinsics, CameraPose, MapResult, Trajectory


def _intr():
    return CameraIntrinsics(100, 100, 50, 40, 100, 80)


def _det(frame, name="chair", confidence=0.9):
    return Detection(frame, frame / 10, 0, name, confidence, (30, 20, 70, 60))


def _run(depths, name="chair", config=None):
    traj = Trajectory(
        [CameraPose.identity(frame_id=i, timestamp=i / 10) for i in range(len(depths))]
    )
    result = MapResult(traj, intrinsics=_intr())
    maps = [
        DepthMap(i, i / 10, np.full((80, 100), z)) for i, z in enumerate(depths) if z is not None
    ]
    pipeline = SemanticPipeline(config=config)
    for frame in range(len(depths)):
        pipeline.process_detections([_det(frame, name)], result, DepthResult(maps=maps))
    return pipeline.result().objects[0], pipeline.result()


def test_missing_later_depth_retains_last_supported_position_and_source():
    obj, result = _run([5, None, None])
    assert obj.xyz == pytest.approx([0, 0, 5])
    assert obj.frame_id == obj.last_frame == 2
    assert obj.metadata["position_frame"] == 0
    assert obj.metadata["position_status"] == "retained"
    assert obj.metadata["position_source"] == "depth_median"
    assert obj.metadata["position_age_frames"] == 2
    assert obj.metadata["position_age_seconds"] == pytest.approx(0.2)
    assert obj.metadata["position_observations"] == 1
    assert result.metadata["num_retained_positions"] == 1


def test_static_object_fusion_reduces_frame_noise_without_claiming_accuracy():
    obj, result = _run([4.8, 5.2, 5.0])
    assert obj.xyz == pytest.approx([0, 0, 5.0])
    assert obj.metadata["position_uncertainty"] > 0
    assert obj.metadata["position_observations"] == 3
    assert obj.metadata["position_fusion"] == "robust_temporal_median"
    assert not result.metadata["position_accuracy_validated"]


def test_sudden_background_depth_jump_cannot_replace_stable_static_position():
    obj, _ = _run([5, 5.1, 30])
    assert obj.xyz[2] == pytest.approx(5.05)
    assert obj.metadata["position_status"] == "retained"
    assert obj.metadata["position_frame"] == 1
    assert obj.metadata["position_retained_reason"] == "inconsistent_position_jump"
    assert obj.metadata["latest_position_evidence"]["rejected_jump_distance"] > 20
    assert len(obj.metadata["position_history"]) == 2


def test_dynamic_object_uses_latest_position_without_static_averaging():
    obj, _ = _run([5, 6, 10], name="person")
    assert obj.xyz[2] == pytest.approx(10)
    assert obj.metadata["motion_policy"] == "latest_observation"
    assert obj.metadata["position_fusion"] == "latest_dynamic_observation"
    assert obj.metadata["position_observations"] == 3


def test_dynamic_missing_position_is_marked_old_instead_of_current():
    obj, _ = _run([5, None], name="person")
    assert obj.metadata["position_status"] == "retained"
    assert obj.metadata["position_age_seconds"] == pytest.approx(0.1)
    assert obj.metadata["motion_policy"] == "latest_observation"


def test_unpositioned_detections_remain_available_for_classification():
    obj, result = _run([None, None])
    assert obj.xyz is None
    assert obj.class_name == "chair"
    assert obj.confidence == pytest.approx(0.9)
    assert result.metadata["num_unpositioned_objects"] == 1
    assert obj.metadata["position_history"] == []


def test_position_history_keeps_endpoints_when_bounded():
    obj, _ = _run([5] * 8, config=SemanticPipelineConfig(max_position_history=3))
    history = obj.metadata["position_history"]
    assert len(history) == 3
    assert history[0]["frame_id"] == 0 and history[-1]["frame_id"] == 7
    assert obj.metadata["position_observations"] == 8


def test_semantic_map_exports_camera_rotation_and_only_selected_object_tracks():
    rotation = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], dtype=float)
    map_result = MapResult(
        Trajectory([CameraPose.from_rt(rotation, [0, 0, 0], frame_id=0)]),
        intrinsics=_intr(),
        metadata={"track_rate": 0.4, "scale": "monocular_arbitrary"},
    )
    obj = SemanticObject(1, 0, "chair", None, None, 0, 1, 2, 0.9, (30, 20, 70, 60), 1)
    semantics = SemanticResult(
        objects=[obj],
        tracks=[
            TrackedDetection(1, _det(0)),
            TrackedDetection(99, _det(0)),
            TrackedDetection(1, _det(1)),
        ],
    )
    semantic_map = build_semantic_map(map_result, semantics)
    exported = semantic_map.to_dict()
    assert exported["trajectory_rotations"] == [rotation.tolist()]
    assert [row["track_id"] for row in exported["object_observations"]] == [1, 1]
    assert [row["frame_id"] for row in exported["object_observations"]] == [0, 1]
    assert exported["metadata"]["intrinsics"] == _intr().to_dict()
    assert exported["metadata"]["track_rate"] == 0.4
    assert exported["summary"]["num_unpositioned_objects"] == 1
    assert "object_observations" not in exported["summary"]


@pytest.mark.parametrize("field", ["max_objects", "fusion_window"])
@pytest.mark.parametrize("value", [-1, 0, 1.5, float("nan"), float("inf"), True, "3"])
def test_pipeline_limits_require_positive_integers(field, value):
    with pytest.raises(ValueError, match=field):
        SemanticPipelineConfig(**{field: value})


@pytest.mark.parametrize("value", [-1, 0, 1, 2.5, float("nan"), float("inf"), True, "3"])
def test_pipeline_history_limit_requires_two_endpoint_slots(value):
    with pytest.raises(ValueError, match="max_position_history"):
        SemanticPipelineConfig(max_position_history=value)


@pytest.mark.parametrize("value", [-0.1, 0, 1.1, float("nan"), float("inf"), True, "0.5"])
def test_pipeline_position_jump_ratio_requires_finite_unit_interval(value):
    with pytest.raises(ValueError, match="max_position_jump_ratio"):
        SemanticPipelineConfig(max_position_jump_ratio=value)


def test_pipeline_setting_boundaries_are_accepted():
    cfg = SemanticPipelineConfig(
        fusion_window=1,
        max_position_history=2,
        max_position_jump_ratio=1,
        max_objects=1,
    )
    assert cfg.max_position_history == 2


@pytest.mark.parametrize("name,static_assumed", [("chair", True), ("person", False)])
def test_category_motion_policy_is_not_reported_as_measured_motion(name, static_assumed):
    obj, result = _run([5], name=name)
    assert obj.metadata["motion_policy_basis"] == "stable_and_latest_class_prior_only"
    assert obj.metadata["static_scene_assumed"] == static_assumed
    assert obj.metadata["motion_detection_available"] is False
    assert result.metadata["motion_detection_available"] is False
    assert result.metadata["motion_policy_limitation"] == "static_category_instances_can_also_move"


@pytest.mark.parametrize("value", [-0.1, 1.1, float("nan"), float("inf"), True, "0.15"])
def test_reprojection_box_margin_requires_finite_unit_interval(value):
    with pytest.raises(ValueError, match="reprojection_box_margin"):
        SemanticPipelineConfig(reprojection_box_margin=value)


def _changed_camera_run(translation, second_depth=None):
    trajectory = Trajectory(
        [
            CameraPose.identity(frame_id=0),
            CameraPose.from_rt(np.eye(3), translation, frame_id=1, timestamp=0.1),
        ]
    )
    mapping = MapResult(trajectory, intrinsics=_intr())
    maps = [DepthMap(0, 0, np.full((80, 100), 5))]
    if second_depth is not None:
        maps.append(DepthMap(1, 0.1, np.full((80, 100), second_depth)))
    pipeline = SemanticPipeline()
    for frame in range(2):
        pipeline.process_detections([_det(frame)], mapping, DepthResult(maps=maps))
    return pipeline.result()


def test_historical_position_outside_current_image_is_degraded_without_erasing_history():
    result = _changed_camera_run([10, 0, 0])
    obj = result.objects[0]
    assert obj.xyz == pytest.approx([0, 0, 5])
    assert obj.metadata["position_status"] == "inconsistent"
    assert obj.metadata["position_temporal_consistency"] == "contradicted"
    projection = obj.metadata["position_reprojection"]
    assert projection["frame_id"] == 1
    assert projection["pixel"] == pytest.approx([-150, 40])
    assert not projection["inside_box"] and not projection["inside_image"]
    assert obj.metadata["latest_position_evidence"]["cause_distinguishable"] is False
    assert obj.metadata["position_history"][0]["position_temporal_consistency"] == "consistent"
    assert obj.metadata["position_history"][0]["xyz"] == [0, 0, 5]
    assert result.metadata["num_inconsistent_positions"] == 1


def test_historical_position_behind_current_camera_is_contradicted():
    obj = _changed_camera_run([0, 0, 10]).objects[0]
    assert obj.metadata["position_temporal_consistency"] == "contradicted"
    assert obj.metadata["position_reprojection"]["reason"] == "behind_current_camera"
    assert obj.metadata["position_reprojection"]["pixel"] is None


def test_new_geometric_observation_reanchors_after_historical_contradiction():
    obj = _changed_camera_run([10, 0, 0], second_depth=5).objects[0]
    assert obj.xyz == pytest.approx([10, 0, 5])
    assert obj.metadata["position_temporal_consistency"] == "consistent"
    assert obj.metadata["position_status"] == "tentative"
    assert obj.metadata["position_reanchored"] is True
    assert obj.metadata["position_frame"] == 1
    history = obj.metadata["position_history"]
    assert [entry["xyz"] for entry in history] == [[0, 0, 5], [10, 0, 5]]
    assert all(entry["position_temporal_consistency"] == "consistent" for entry in history)


def test_missing_current_pose_cannot_verify_historical_position():
    mapping = MapResult(Trajectory([CameraPose.identity(frame_id=0)]), intrinsics=_intr())
    depth = DepthResult([DepthMap(0, 0, np.full((80, 100), 5))])
    pipeline = SemanticPipeline()
    pipeline.process_detections([_det(0)], mapping, depth)
    pipeline.process_detections([_det(1)], mapping, depth)
    obj = pipeline.result().objects[0]
    assert obj.metadata["position_status"] == "retained"
    assert obj.metadata["position_temporal_consistency"] == "unverified"
    assert obj.metadata["position_reprojection"]["reason"] == "missing_current_camera_pose"


@pytest.mark.parametrize("margin,status", [(0.15, "consistent"), (0, "contradicted")])
def test_reprojection_margin_tolerates_small_box_jitter(margin, status):
    mapping = MapResult(
        Trajectory([CameraPose.identity(frame_id=frame) for frame in range(2)]),
        intrinsics=_intr(),
    )
    depth = DepthResult([DepthMap(0, 0, np.full((80, 100), 5))])
    pipeline = SemanticPipeline(config=SemanticPipelineConfig(reprojection_box_margin=margin))
    pipeline.process_detections([_det(0)], mapping, depth)
    jittered = Detection(1, 0.1, 0, "chair", 0.9, (51, 20, 91, 60))
    pipeline.process_detections([jittered], mapping, depth)
    obj = pipeline.result().objects[0]
    assert obj.observations == 2
    assert obj.metadata["position_temporal_consistency"] == status


def test_historical_reprojection_uses_distorted_video_pixel_coordinates():
    intr = CameraIntrinsics(300, 260, 320, 240, 640, 480, dist=(0.2, -0.05, 0.01, -0.015, 0.01))
    # Brown-Conrady projection of normalized ray (0.4, 0.2), z=5.
    u, v = 442.7096, 294.08416
    mapping = MapResult(
        Trajectory([CameraPose.identity(frame_id=frame) for frame in range(2)]),
        intrinsics=intr,
    )
    depth = DepthResult([DepthMap(0, 0, np.full((480, 640), 5))])
    pipeline = SemanticPipeline(config=SemanticPipelineConfig(lift=LiftConfig(min_valid_pixels=1)))
    for frame in range(2):
        det = Detection(frame, frame / 10, 39, "bottle", 0.9, (u - 1, v - 1, u + 1, v + 1))
        pipeline.process_detections([det], mapping, depth)
    obj = pipeline.result().objects[0]
    assert obj.metadata["position_temporal_consistency"] == "consistent"
    assert obj.metadata["position_reprojection"]["pixel"] == pytest.approx([u, v], abs=1e-6)
