"""Object positions need coherent camera scale and independent frame evidence."""

import numpy as np
import pytest

from visualmap.depth.types import DepthMap, DepthResult
from visualmap.detection import Detection, SemanticPipeline, SemanticPipelineConfig
from visualmap.detection.lift import LiftConfig, lift_detection
from visualmap.detection.tracker import TrackerConfig
from visualmap.mapping import build_semantic_map
from visualmap.slam.types import CameraIntrinsics, CameraPose, MapPoint, MapResult, Trajectory


def _intr():
    return CameraIntrinsics(400, 400, 320, 240, 640, 480)


def _mapping(translations, quality=None, world_frame="test_world"):
    poses = [
        CameraPose.from_rt(np.eye(3), [x, 0, 0], frame_id=i, timestamp=i / 10)
        for i, x in enumerate(translations)
    ]
    quality = quality or {
        str(i): {"status": "bootstrap" if i < 2 else "tracked", "pose_valid": True}
        for i in range(len(poses))
    }
    return MapResult(
        Trajectory(poses),
        intrinsics=_intr(),
        metadata={
            "frame_quality": quality,
            "world_frame_id": world_frame,
            "scale_consistent": True,
            "backend": "consistent_vo",
        },
    )


def _det(frame, pose, xyz=(0.2, -0.1, 6), name="bottle", offset=(0, 0)):
    camera = pose.inverse().transform_point(np.array(xyz))
    intr = _intr()
    u = intr.fx * camera[0] / camera[2] + intr.cx + offset[0]
    v = intr.fy * camera[1] / camera[2] + intr.cy + offset[1]
    return Detection(frame, frame / 10, 39, name, 0.95, (u - 45, v - 65, u + 45, v + 65))


def _run(mapping, name="bottle", offsets=None, pipeline=None):
    pipeline = pipeline or SemanticPipeline(
        config=SemanticPipelineConfig(
            tracker=TrackerConfig(iou_threshold=0.1, motion_prediction=False)
        )
    )
    for i, pose in enumerate(mapping.trajectory):
        pipeline.process_detections(
            [_det(i, pose, name=name, offset=(offsets or {}).get(i, (0, 0)))], mapping
        )
    return pipeline


def test_uneven_camera_speed_recovers_object_centre_in_one_scale_with_heldout_view():
    mapping = _mapping([0, 0.03, 0.11, 0.43, 0.92, 1.1])
    pipeline = _run(mapping)
    obj = pipeline.result().objects[0]
    assert obj.xyz == pytest.approx([0.2, -0.1, 6], abs=1e-8)
    assert obj.metadata["position_source"] == "object_center_multiview"
    assert obj.metadata["position_relation_usable"] is True
    assert obj.metadata["world_frame_id"] == "test_world"
    evidence = obj.metadata["multiview_evidence"]
    assert evidence["validation_frames"]  # More than the defining two rays.
    assert evidence["observation_count"] == 6
    assert evidence["max_inlier_reprojection_px"] < 1e-8
    assert evidence["centre_bias_unknown"] is True
    assert evidence["calibrated_error_bound"] is False
    assert obj.metadata["position_uncertainty"] > 0  # Exact centres do not imply certainty.
    assert evidence["depth_sensitivity"] == obj.metadata["position_uncertainty"]
    assert obj.metadata["uncertainty"]["method"] == "reprojection_and_parallax_sensitivity"
    snapshots = obj.metadata["position_snapshots"]
    assert [row["frame_id"] for row in snapshots] == list(range(6))
    assert all(row["xyz"] is None for row in snapshots[:3])
    assert snapshots[-1]["camera_xyz"] == pytest.approx([-0.9, -0.1, 6])


@pytest.mark.parametrize("structural_conflict", [True, False])
def test_review_metadata_gates_mixed_frame_relations_with_actual_multiview_position(
    structural_conflict,
):
    mapping = _mapping([0, 0.03, 0.11, 0.43, 0.92, 1.1])
    pipeline = SemanticPipeline(
        config=SemanticPipelineConfig(
            tracker=TrackerConfig(iou_threshold=0.1, motion_prediction=False),
        )
    )
    for i, pose in enumerate(mapping.trajectory):
        seed = _det(i, pose, name="chair")
        detection = Detection(
            i,
            i / 10,
            56,
            "chair",
            0.55,
            seed.xyxy,
            metadata={
                "classification_source": "model_inference",
                "classification_status": "needs_review",
                "classification_needs_review": True,
                "classification_structural_conflict": structural_conflict,
                "classification_reason": "mixed_foreground_and_perforated_background"
                if structural_conflict
                else "unverified_model_prediction",
                "raw_model_class": "chair",
            },
        )
        pipeline.process_detections([detection], mapping)
    obj = pipeline.result().objects[0]
    # Execute the complete non-short-circuited usable-position branch. This
    # caught a raw-provenance variable being read before its assignment.
    assert obj.xyz == pytest.approx([0.2, -0.1, 6], abs=1e-8)
    assert obj.metadata["position_source"] == "object_center_multiview"
    assert obj.metadata["position_relation_usable"] is (not structural_conflict)
    assert obj.metadata["position_snapshots"][-1]["position_relation_usable"] is (
        not structural_conflict
    )
    assert obj.metadata["classification_needs_review"] is True
    assert obj.metadata["classification_evidence"]["raw_model_class"] == "chair"


def test_third_frame_must_validate_two_ray_hypothesis():
    mapping = _mapping([0, 0.3, 0.7])
    obj = _run(mapping, offsets={2: (0, 55)}).result().objects[0]
    assert obj.xyz is None
    assert not obj.metadata["position_relation_usable"]
    assert obj.metadata["multiview_evidence"]["reason"] == "object_centre_reprojection_disagreement"


def test_latest_dynamic_class_vetoes_a_delayed_static_class_vote_and_its_prior():
    mapping = _mapping([0, 0.1, 0.2, 0.4, 0.5])
    pipeline = SemanticPipeline()
    first_id = None
    for i, pose in enumerate(mapping.trajectory):
        seed = _det(i, pose)
        dynamic = i == 4
        detection = Detection(
            i, i / 10, 0 if dynamic else 56,
            "person" if dynamic else "chair", 0.95, seed.xyxy,
        )
        track = pipeline.process_detections([detection], mapping)[0]
        first_id = track.track_id if first_id is None else first_id
        assert track.track_id == first_id
    obj = pipeline.result().objects[0]
    assert obj.class_name == "chair"  # The vote has not switched after one frame.
    assert obj.metadata["classification_evidence"]["latest_raw_class"] == "person"
    assert obj.metadata["classification_needs_review"] is True
    assert obj.metadata["static_scene_assumed"] is False
    assert obj.metadata["latest_raw_dynamic_class"] == "person"
    assert obj.metadata["position_relation_usable"] is False
    assert obj.metadata["multiview_evidence"]["reason"] == (
        "dynamic_category_excluded_from_static_triangulation"
    )
    assert len(pipeline._geometry_observations[first_id]) == 1
    assert not pipeline._position_samples[first_id]


@pytest.mark.parametrize("membership", [None, False, True])
def test_structure_fit_cannot_make_a_relation_without_verified_entity_membership(membership):
    mapping = _mapping([0, 0.03, 0.11, 0.43, 0.92, 1.1])
    pipeline = SemanticPipeline(
        config=SemanticPipelineConfig(
            tracker=TrackerConfig(iou_threshold=0.1, motion_prediction=False)
        )
    )
    for i, pose in enumerate(mapping.trajectory):
        seed = _det(i, pose)
        detection = Detection(
            i, i / 10, 1001, "perforated_panel", 0.8, seed.xyxy,
            metadata={
                "classification_source": "image_structure_rule",
                "classification_status": "structural_candidate",
                "classification_needs_review": True,
                "physical_object_membership_verified": membership,
            },
        )
        pipeline.process_detections([detection], mapping)
    obj = pipeline.result().objects[0]
    assert obj.xyz is not None  # Visible support can remain available for review.
    assert obj.metadata["position_relation_usable"] is (membership is True)
    assert obj.metadata["classification_evidence"]["physical_object_membership_verified"] is (
        membership
    )


def test_low_parallax_is_rejected_even_when_centres_reproject_perfectly():
    obj = _run(_mapping([0, 0.002, 0.006, 0.009])).result().objects[0]
    assert obj.xyz is None
    assert obj.metadata["multiview_evidence"]["reason"] == "insufficient_object_parallax"


def test_no_baseline_cannot_triangulate_object_from_box_dimensions():
    obj = _run(_mapping([0, 0, 0, 0])).result().objects[0]
    assert obj.xyz is None
    assert obj.metadata["multiview_evidence"]["reason"] == "no_translational_baseline"


def test_unknown_legacy_camera_scale_never_grants_relation_trust():
    mapping = _mapping([0, 0.3, 0.7, 1])
    mapping.metadata.clear()
    obj = _run(mapping).result().objects[0]
    assert obj.xyz is None
    assert obj.metadata["pose_trusted"] is False
    assert not obj.metadata["position_relation_usable"]


@pytest.mark.parametrize(
    "status,valid",
    [
        ("tracking_lost", False),
        ("pure_rotation", True),
        ("held", False),
        ("bootstrap_pending", False),
    ],
)
def test_held_rotation_or_lost_camera_frames_do_not_add_object_rays(status, valid):
    quality = {str(i): {"status": "tracked", "pose_valid": True} for i in range(4)}
    quality["2"] = {"status": status, "pose_valid": valid}
    mapping = _mapping([0, 0.3, 0.7, 1], quality)
    obj = _run(mapping).result().objects[0]
    assert obj.metadata["multiview_evidence"]["inlier_frames"] == [0, 1, 3]
    snapshot = obj.metadata["position_snapshots"][2]
    assert snapshot["position_relation_usable"] is False
    assert snapshot["pose_trusted"] is False


def test_dynamic_categories_are_never_triangulated_as_static_objects():
    obj = _run(_mapping([0, 0.3, 0.7, 1]), name="person").result().objects[0]
    assert obj.xyz is None
    assert (
        obj.metadata["multiview_evidence"]["reason"]
        == "dynamic_category_excluded_from_static_triangulation"
    )


def test_majority_background_layer_cannot_take_over_foreground_object():
    det = Detection(0, 0, 39, "bottle", 0.9, (275, 175, 365, 305))
    points = [
        MapPoint(i, [offset, 0, z], observations={0: i, 1: i})
        for i, (offset, z) in enumerate(
            [(x, 5) for x in [-0.05, 0, 0.05]] + [(x, 25) for x in np.linspace(-0.2, 0.2, 9)]
        )
    ]
    obj = lift_detection(det, CameraPose.identity(), _intr(), map_points=points)
    assert obj.xyz is None
    assert obj.metadata["evidence"]["reason"] == "ambiguous_foreground_background_depth_layers"
    anchored = lift_detection(
        det, CameraPose.identity(), _intr(), map_points=points, target_depth=5
    )
    assert anchored.depth == 5
    assert anchored.metadata["evidence"]["point_ids"] == [0, 1, 2]


def test_persistent_landmarks_must_remain_inside_other_observed_object_boxes():
    intr = _intr()
    det0 = Detection(0, 0, 39, "bottle", 0.9, (275, 175, 365, 305))
    det1 = Detection(1, 0.1, 39, "bottle", 0.9, (240, 175, 320, 305))
    pose0, pose1 = (
        CameraPose.identity(frame_id=0),
        CameraPose.from_rt(np.eye(3), [0.4, 0, 0], frame_id=1),
    )
    # At z=25 the wall has little parallax and leaves the inner object support.
    points = [
        MapPoint(i, [x, 0, 25], observations={0: i, 1: i}) for i, x in enumerate([0.85, 0.9, 0.95])
    ]
    current = lift_detection(det0, pose0, intr, map_points=points)
    assert current.xyz is not None
    constrained = lift_detection(
        det0, pose0, intr, map_points=points, object_observations=[(det1, pose1)]
    )
    assert constrained.xyz is None
    assert (
        constrained.metadata["evidence"]["reason"] == "landmarks_not_consistent_with_object_track"
    )


def test_clipped_box_centre_does_not_add_a_false_ray():
    mapping = _mapping([0, 0.3, 0.7])
    pipeline = SemanticPipeline(
        config=SemanticPipelineConfig(tracker=TrackerConfig(iou_threshold=0.01))
    )
    for i, pose in enumerate(mapping.trajectory):
        det = _det(i, pose)
        if i == 2:
            det = Detection(
                i,
                i / 10,
                det.class_id,
                det.class_name,
                det.confidence,
                (det.xyxy[0], 0, det.xyxy[2], det.xyxy[3]),
            )
        pipeline.process_detections([det], mapping)
    obj = pipeline.result().objects[0]
    assert obj.xyz is None
    assert obj.metadata["multiview_evidence"]["current_box_clipped"] is True
    assert (
        obj.metadata["multiview_evidence"]["reason"]
        == "clipped_box_centre_is_not_a_stable_object_ray"
    )


def test_current_lost_pose_keeps_historical_xyz_but_marks_snapshot_unusable():
    quality = {str(i): {"status": "tracked", "pose_valid": True} for i in range(4)}
    quality["3"] = {"status": "tracking_lost", "pose_valid": False}
    mapping = _mapping([0, 0.3, 0.7, 1], quality)
    pipeline = _run(mapping)
    obj = pipeline.result().objects[0]
    assert obj.xyz is not None
    assert obj.metadata["position_status"] == "retained"
    assert obj.metadata["position_temporal_consistency"] == "unverified"
    assert not obj.metadata["position_relation_usable"]
    sm = build_semantic_map(mapping, pipeline.result()).to_dict()
    geometries = [row["geometry"] for row in sm["object_observations"]]
    assert geometries[0]["xyz"] is None
    assert geometries[2]["position_relation_usable"] is True
    assert geometries[3]["pose_valid"] is False
    assert geometries[3]["position_relation_usable"] is False
    assert sm["metadata"]["slam_backend"] == "consistent_vo"
    assert sm["metadata"]["scale_consistent"] is True


def test_different_world_frames_never_fuse_even_with_same_object_track():
    mapping = _mapping([0, 0.3, 0.7, 1])
    pipeline = SemanticPipeline()
    for i in range(3):
        pipeline.process_detections([_det(i, mapping.trajectory.poses[i])], mapping)
    assert pipeline.result().objects[0].xyz is not None
    mapping.metadata["world_frame_id"] = "new_local_map"
    pipeline.process_detections([_det(3, mapping.trajectory.poses[3])], mapping)
    obj = pipeline.result().objects[0]
    assert obj.xyz is None
    assert obj.metadata["world_frame_id"] == "new_local_map"
    assert obj.metadata["position_snapshots"][-1]["xyz"] is None


def test_multiview_centre_cannot_be_validated_by_an_incompatible_background_surface():
    mapping = _mapping([0, 0.3, 0.7])
    # All box-centre rays intersect z=6; the only mapped surface sits at z=30.
    # It appears in the current box but must not become this object's coordinate.
    final_pose = mapping.trajectory.poses[2]
    final_det = _det(2, final_pose)
    camera_x = (final_det.cx - _intr().cx) / _intr().fx * 30
    camera_y = (final_det.cy - _intr().cy) / _intr().fy * 30
    mapping.map_points = [
        MapPoint(
            i,
            final_pose.transform_point([camera_x + offset, camera_y, 30]),
            observations={1: i, 2: i},
        )
        for i, offset in enumerate([-0.01, 0, 0.01])
    ]
    pipeline = SemanticPipeline()
    for i, pose in enumerate(mapping.trajectory):
        pipeline.process_detections(
            [_det(i, pose)], mapping, DepthResult(metadata={"method": "sparse_depth"})
        )
    obj = pipeline.result().objects[0]
    assert obj.metadata["multiview_evidence"]["observation_count"] == 3
    assert obj.xyz is None
    assert not obj.metadata["position_relation_usable"]
    assert obj.metadata["evidence"]["reason"] == "object_centre_and_surface_depth_disagree"


def test_each_contradicting_detection_gets_its_own_snapshot_instead_of_final_xyz():
    mapping = MapResult(
        Trajectory(
            [
                CameraPose.identity(frame_id=0),
                CameraPose.from_rt(np.eye(3), [10, 0, 0], frame_id=1),
            ]
        ),
        intrinsics=_intr(),
    )
    depth = DepthResult([DepthMap(0, 0, np.full((480, 640), 6))])
    pipeline = SemanticPipeline()
    for i in range(2):
        pipeline.process_detections(
            [Detection(i, i / 10, 39, "bottle", 0.9, (275, 175, 365, 305))], mapping, depth
        )
    snapshots = pipeline.result().objects[0].metadata["position_snapshots"]
    assert [row["frame_id"] for row in snapshots] == [0, 1]
    assert snapshots[0]["temporal_consistency"] == "consistent"
    assert snapshots[1]["temporal_consistency"] == "contradicted"
    assert snapshots[1]["status"] == "inconsistent"
    assert snapshots[1]["position_relation_usable"] is False
    exported = build_semantic_map(mapping, pipeline.result()).to_dict()
    assert exported["object_observations"][0]["geometry"]["temporal_consistency"] == "consistent"
    assert exported["object_observations"][1]["geometry"]["temporal_consistency"] == "contradicted"


def _persistent_surface_mapping(translations=(0, 0.002, 0.006, 0.009), observations=None):
    mapping = _mapping(translations)
    mapping.map_points = [
        MapPoint(
            i,
            [x, -0.1, 6],
            observations=(
                observations
                if observations is not None
                else {frame: i for frame in range(len(translations))}
            ),
        )
        for i, x in enumerate([0.18, 0.2, 0.22])
    ]
    return mapping


def test_persistent_observed_landmarks_in_three_trusted_centre_boxes_support_surface_relationship():
    pipeline = _run(_persistent_surface_mapping())
    obj = pipeline.result().objects[0]
    assert obj.metadata["multiview_evidence"]["reason"] == "insufficient_object_parallax"
    assert obj.xyz == pytest.approx([0.2, -0.1, 6])
    assert obj.metadata["position_source"] == "persistent_surface_multiview"
    assert obj.metadata["position_relation_kind"] == "surface_support"
    assert obj.metadata["position_relation_usable"] is True
    assert obj.metadata["center_bias_unknown"] is True
    assert obj.metadata["uncertainty"]["calibrated_error_bound"] is False
    evidence = obj.metadata["evidence"]
    assert evidence["persistent_support_point_ids"] == [0, 1, 2]
    assert evidence["persistent_support_min_views"] == 4
    assert all(len(frames) >= 3 for frames in evidence["persistent_support_point_frames"].values())
    snapshots = obj.metadata["position_snapshots"]
    assert [row["position_relation_usable"] for row in snapshots] == [False, False, True, True]
    assert snapshots[-1]["position_relation_kind"] == "surface_support"
    assert snapshots[-1]["surface_support_evidence"]["persistent_support_qualified"] is True


@pytest.mark.parametrize("observations", [{0: 0, 3: 0}, {3: 0, 99: 0}])
def test_projection_overlap_is_not_three_frames_of_persistent_object_observation(observations):
    obj = _run(_persistent_surface_mapping(observations=observations)).result().objects[0]
    assert obj.metadata["position_source"] != "persistent_surface_multiview"
    assert obj.metadata["position_relation_usable"] is False


def test_repeated_camera_pose_does_not_create_new_world_view_evidence_for_surface():
    obj = _run(_persistent_surface_mapping(translations=(0, 0, 0, 0))).result().objects[0]
    assert obj.metadata["position_source"] == "sparse_observed"
    assert obj.metadata["position_relation_usable"] is False


def test_persistent_landmarks_at_box_edges_do_not_qualify_as_central_surface_support():
    mapping = _persistent_surface_mapping()
    pipeline = SemanticPipeline()
    for i, pose in enumerate(mapping.trajectory):
        det = _det(i, pose)
        # Same points remain in all full boxes but first two shifted centres
        # never directly support the points in their central subboxes.
        if i < 2:
            det = Detection(
                i,
                i / 10,
                39,
                "bottle",
                0.95,
                tuple(
                    value + 22 if axis % 2 == 0 else value for axis, value in enumerate(det.xyxy)
                ),
            )
        pipeline.process_detections([det], mapping)
    obj = pipeline.result().objects[0]
    assert obj.metadata["position_relation_usable"] is False
    assert obj.metadata["position_source"] != "persistent_surface_multiview"


def test_surface_persistence_requires_trusted_camera_world_scale():
    mapping = _persistent_surface_mapping()
    mapping.metadata.clear()
    obj = _run(mapping).result().objects[0]
    assert obj.xyz is not None
    assert obj.metadata["position_source"] == "sparse_observed"
    assert obj.metadata["position_relation_usable"] is False


def test_persistent_background_layer_cannot_override_an_independent_centre_depth():
    mapping = _mapping([0, 0.3, 0.7])
    mapping.map_points = [
        MapPoint(i, [x, -0.1, 30], observations={0: i, 1: i, 2: i})
        for i, x in enumerate([0.18, 0.2, 0.22])
    ]
    pipeline = SemanticPipeline(config=SemanticPipelineConfig(lift=LiftConfig(center_frac=0.65)))
    for i, pose in enumerate(mapping.trajectory):
        det = _det(i, pose)
        det = Detection(
            i, i / 10, 39, "bottle", 0.95, (det.cx - 130, det.cy - 65, det.cx + 130, det.cy + 65)
        )
        pipeline.process_detections([det], mapping)
    obj = pipeline.result().objects[0]
    assert obj.metadata["position_status"] in {"unavailable", "retained"}
    assert (
        obj.metadata.get("latest_position_evidence", obj.metadata["evidence"])["reason"]
        == "object_centre_and_surface_depth_disagree"
    )
    assert obj.metadata["position_relation_usable"] is False


@pytest.mark.parametrize(
    "field,value",
    [
        ("min_multiview_observations", 2),
        ("min_multiview_observations", True),
        ("min_multiview_parallax_deg", 0),
        ("min_multiview_parallax_deg", 90),
        ("max_multiview_reprojection_px", float("nan")),
        ("max_center_surface_depth_ratio", 1.01),
    ],
)
def test_multiview_configuration_rejects_unusable_thresholds(field, value):
    with pytest.raises(ValueError, match=field):
        SemanticPipelineConfig(**{field: value})
