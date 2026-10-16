"""Occupancy is a same-frame visual hypothesis, never a free-space proof."""

import json
from copy import deepcopy

import numpy as np
import pytest

from visualmap.detection.types import Detection, SemanticObject, SemanticResult, TrackedDetection
from visualmap.mapping.occupancy import attach_occupancy
from visualmap.slam.types import CameraIntrinsics, CameraPose, MapPoint, MapResult, Trajectory


def fixture(frame=2, pose=None, points=None):
    pose = pose or CameraPose.identity(frame_id=frame)
    xyz = pose.transform_point(np.array([0.0, 0.0, 5.0]))
    snapshot = {
        "frame_id": frame,
        "xyz": xyz.tolist(),
        "camera_xyz": [0.0, 0.0, 5.0],
        "position_frame": frame,
        "world_frame_id": "world-a",
        "status": "estimated",
        "pose_valid": True,
        "pose_trusted": True,
        "temporal_consistency": "consistent",
        "position_relation_usable": True,
        "uncertainty": 0.03,
        "support_evidence": {"point_ids": [10, 11, 12]},
    }
    obj = SemanticObject(
        7,
        39,
        "bottle",
        xyz,
        5.0,
        frame,
        frame,
        3,
        0.9,
        (270, 190, 370, 290),
        frame,
        {"position_snapshots": [snapshot]},
    )
    mapping = MapResult(
        Trajectory([pose]),
        points or [],
        CameraIntrinsics(500, 500, 320, 240, 640, 480),
        {
            "world_frame_id": "world-a",
            "scale_consistent": True,
            "frame_quality": {
                str(frame): {"status": "tracked", "pose_valid": True, "world_frame_id": "world-a"}
            },
        },
    )
    record = {
        "track_id": 7,
        "frame_id": frame,
        "class_name": "bottle",
        "raw_class_name": "bottle",
        "xyxy": [270, 190, 370, 290],
        "classification_evidence": {"source": "model_inference", "needs_review": False},
    }
    return mapping, SemanticResult(objects=[obj]), [record], snapshot


def estimate(mapping, semantics, records):
    assert attach_occupancy(mapping, semantics, records) is semantics
    return semantics.objects[0].metadata["position_snapshots"][0]["occupancy"]


def test_box_has_expected_projected_size_and_explicit_unverified_policy():
    mapping, semantics, records, snapshot = fixture()
    original = deepcopy(snapshot)
    occupancy = estimate(mapping, semantics, records)
    assert occupancy["status"] == "inferred_unverified"
    assert occupancy["available"] is True
    assert occupancy["min"] == pytest.approx([-0.525, -0.525, 4.75])
    assert occupancy["max"] == pytest.approx([0.525, 0.525, 5.25])
    assert occupancy["size"] == pytest.approx([1.05, 1.05, 0.5])
    assert occupancy["volume_map_units3"] == pytest.approx(1.05 * 1.05 * 0.5)
    assert occupancy["candidate_overlap_usable"] is True
    for field in (
        "collision_usable",
        "bounds_are_guaranteed",
        "free_space_certified",
        "collision_clearance_certified",
        "volume_verified",
    ):
        assert occupancy[field] is False
    assert {key: value for key, value in snapshot.items() if key != "occupancy"} == original
    assert semantics.objects[0].metadata["occupancy"] == occupancy
    assert semantics.objects[0].metadata["occupancy"] is not occupancy
    json.dumps(semantics.objects[0].to_dict(), allow_nan=False)


def multiview_fixture():
    mapping, semantics, records, first = fixture()
    records[0]["xyxy"] = [290, 200, 350, 280]
    current = deepcopy(first)
    current.update(frame_id=4, position_frame=4, camera_xyz=[-1.0, 0.0, 5.0])
    semantics.objects[0].metadata["position_snapshots"].append(current)
    semantics.objects[0].frame_id = 4
    mapping.trajectory.append(CameraPose.from_rt(np.eye(3), [1, 0, 0], frame_id=4))
    mapping.metadata["frame_quality"]["4"] = {
        "status": "tracked",
        "pose_valid": True,
        "world_frame_id": "world-a",
    }
    records.append({**deepcopy(records[0]), "frame_id": 4, "xyxy": [170, 190, 270, 290]})
    return mapping, semantics, records, first, current


def test_past_independent_view_carves_proxy_and_records_assumption_sensitivity():
    mapping, semantics, records, first, current = multiview_fixture()
    attach_occupancy(mapping, semantics, records)
    volume = current["occupancy"]
    assert volume["multi_view"]["status"] == "constrained"
    assert volume["multi_view"]["frame_ids"] == [2, 4]
    assert volume["proxy_volume_map_units3"] < 0.5
    assert volume["aabb_volume_map_units3"] >= volume["proxy_volume_map_units3"]
    assert volume["volume_verified"] is False
    values = volume["volume_sensitivity"]["proxy_volumes_map_units3"]
    assert values == sorted(values)
    assert first["occupancy"]["multi_view"]["frame_ids"] == [2]
    assert first["occupancy"]["multi_view"]["future_views_used"] is False
    # Every shape point must satisfy the earlier, expanded image box.
    vertices = np.asarray(volume["shape_vertices"])
    pixels = vertices[:, :2] / vertices[:, 2:3] * [500, 500] + [320, 240]
    assert np.all(pixels[:, 0] >= 290 - 60 * 0.08 - 1e-5)
    assert np.all(pixels[:, 0] <= 350 + 60 * 0.08 + 1e-5)
    json.dumps(volume, allow_nan=False)


def test_past_volume_from_another_submap_never_carves_current_shape():
    mapping, semantics, records, first, current = multiview_fixture()
    first["world_frame_id"] = "world-old"
    mapping.metadata["frame_quality"]["2"]["world_frame_id"] = "world-old"
    attach_occupancy(mapping, semantics, records)
    assert current["occupancy"]["multi_view"]["frame_ids"] == [4]


def test_degenerate_or_wrong_track_view_does_not_produce_empty_certain_volume():
    mapping, semantics, records, _, current = multiview_fixture()
    records[0]["xyxy"] = [10, 10, 60, 60]
    attach_occupancy(mapping, semantics, records)
    result = current["occupancy"]
    assert result["available"] is True
    assert result["proxy_volume_map_units3"] > 0
    assert result["multi_view"]["rejected_frame_ids"] == [2]
    assert result["candidate_overlap_usable"] is False


def test_convex_volume_is_rotation_invariant_and_clipping_preserves_closed_solid():
    from visualmap.mapping.visual_hull import clip_faces, frustum_faces, serialize_shape

    vertices = np.array(
        [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]],
        dtype=float,
    )
    faces = frustum_faces(vertices)
    assert serialize_shape(faces)[2] == pytest.approx(1)
    clipped = clip_faces(faces, np.array([-1, 0, 0]), 0.5)
    assert serialize_shape(clipped)[2] == pytest.approx(0.5)
    angle = np.radians(37)
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]]
    )
    transformed = vertices @ rotation.T + [100, -5, 13]
    assert serialize_shape(frustum_faces(transformed))[2] == pytest.approx(1)
    shifted_vertices, shifted_faces, shifted_volume = serialize_shape(frustum_faces(vertices + 1e8))
    assert len(shifted_vertices) == 8
    assert len(shifted_faces) == 6
    assert all(len(set(face)) == 4 for face in shifted_faces)
    assert shifted_volume == pytest.approx(1)
    tiny_vertices, tiny_faces, tiny_volume = serialize_shape(frustum_faces(vertices * 1e-10))
    assert len(tiny_vertices) == 8
    assert len(tiny_faces) == 6
    assert tiny_volume == pytest.approx(1e-30, abs=1e-40)


def test_rotated_translated_camera_produces_world_axis_aligned_bounds():
    rotation = np.array([[0.0, 0, 1], [0, 1, 0], [-1, 0, 0]])
    pose = CameraPose.from_rt(rotation, [2, 3, 4], frame_id=2)
    mapping, semantics, records, _ = fixture(pose=pose)
    box = estimate(mapping, semantics, records)
    assert box["center"] == pytest.approx([7, 3, 4])
    assert box["size"] == pytest.approx([0.5, 1.05, 1.05])
    corners = np.asarray(box["corners"])
    assert corners.shape == (8, 3)
    assert np.min(corners, axis=0) == pytest.approx(box["min"])
    assert np.max(corners, axis=0) == pytest.approx(box["max"])


def test_adjustable_thickness_changes_hidden_extent_without_class_size_lookup():
    mapping, semantics, records, _ = fixture()
    attach_occupancy(mapping, semantics, records, inferred_thickness_ratio=0.8)
    thick = semantics.objects[0].metadata["occupancy"]
    assert thick["inferred_thickness"] == pytest.approx(0.8)
    records[0]["class_name"] = records[0]["raw_class_name"] = "refrigerator"
    attach_occupancy(mapping, semantics, records, inferred_thickness_ratio=0.8)
    assert semantics.objects[0].metadata["occupancy"]["size"] == thick["size"]


def test_position_scatter_is_not_mislabeled_as_hidden_solid_thickness():
    mapping, semantics, records, snapshot = fixture()
    snapshot["uncertainty"] = 100
    box = estimate(mapping, semantics, records)
    assert box["inferred_thickness"] == pytest.approx(0.5)
    assert box["position_uncertainty"] == 100
    assert box["position_error_included_in_extent"] is False


def test_future_object_metadata_and_final_position_cannot_change_earlier_box():
    mapping, semantics, records, _ = fixture()
    first = deepcopy(estimate(mapping, semantics, records))
    obj = semantics.objects[0]
    obj.xyz = np.array([100, 200, 300])
    obj.xyxy = (0, 0, 640, 480)
    obj.class_name = "person"
    obj.metadata["classification_evidence"] = {"needs_review": True}
    assert estimate(mapping, semantics, records) == first


def test_no_exact_detection_does_not_borrow_a_future_box():
    mapping, semantics, records, _ = fixture()
    records[0]["frame_id"] = 3
    box = estimate(mapping, semantics, records)
    assert box["available"] is False
    assert box["reason"] == "no_same_frame_detection"
    assert box["corners"] == []


def test_support_observations_from_future_and_other_world_are_not_counted():
    points = [
        MapPoint(10, [0, 0, 5], observations={1: 0, 2: 1}),
        MapPoint(11, [0, 0, 5], observations={2: 0, 3: 1}),
        MapPoint(12, [0, 0, 5], observations={0: 0, 2: 1}),
        MapPoint(13, [0, 0, 5], observations={0: 0, 1: 1}),
    ]
    mapping, semantics, records, snapshot = fixture(points=points)
    mapping.metadata["frame_quality"]["0"] = {"world_frame_id": "other-world"}
    snapshot["support_evidence"]["point_ids"] = [10, 11, 12, 13, 999, True, [1]]
    box = estimate(mapping, semantics, records)
    assert box["support_point_ids"] == [10]
    assert box["observed_support_count"] == 1
    # Final-map landmark coordinates may have been refined later. Do not use
    # them to replace snapshot geometry or grow historical bounds.
    points[0].xyz[:] = [1e8, -1e8, 1e8]
    assert estimate(mapping, semantics, records)["size"] == box["size"]


@pytest.mark.parametrize(
    "changes",
    [
        {"pose_valid": False},
        {"pose_trusted": False},
        {"position_frame": 1},
        {"status": "retained"},
        {"xyz": None},
        {"xyz": [0, 0, float("nan")]},
        {"xyz": [0, 0, -5], "camera_xyz": [0, 0, -5]},
        {"world_frame_id": "different-world"},
        {"camera_xyz": [0, 0, 8]},
        {"frame_id": True},
    ],
)
def test_unreliable_geometry_never_generates_a_fake_volume(changes):
    mapping, semantics, records, snapshot = fixture()
    snapshot.update(changes)
    box = estimate(mapping, semantics, records)
    assert box["available"] is False
    assert box["volume_map_units3"] is None
    assert box["collision_usable"] is False


@pytest.mark.parametrize(
    "bbox",
    [
        [10, 20, 10, 50],
        [0, 0, 650, 480],
        [0, 0, 640, float("inf")],
        [1, 2],
        None,
    ],
)
def test_bad_boxes_are_not_invented_or_silently_clipped(bbox):
    mapping, semantics, records, _ = fixture()
    records[0]["xyxy"] = bbox
    assert estimate(mapping, semantics, records)["available"] is False


def test_unverified_structural_candidate_is_not_a_physical_occupancy_box():
    mapping, semantics, records, _ = fixture()
    records[0]["classification_evidence"] = {
        "source": "image_structure_rule",
        "physical_object_membership_verified": False,
    }
    box = estimate(mapping, semantics, records)
    assert box["reason"] == "structural_candidate_object_membership_unverified"
    assert box["available"] is False


@pytest.mark.parametrize(
    "conflict",
    [
        {"classification_needs_review": True},
        {"classification_evidence": {"temporal_class_disagreement": True}},
        {"detection_evidence": {"classification_structural_conflict": True}},
        {"raw_class_name": "person"},
        {"raw_class_name": "chair"},
    ],
)
def test_conflicts_or_dynamic_raw_class_cannot_authorize_overlap_comparison(conflict):
    mapping, semantics, records, _ = fixture()
    records[0].update(conflict)
    box = estimate(mapping, semantics, records)
    assert box["available"] is True
    assert box["candidate_overlap_usable"] is False
    assert box["collision_usable"] is False


def test_clipped_boxes_and_unverified_position_disable_overlap_comparison():
    mapping, semantics, records, snapshot = fixture()
    records[0]["xyxy"][0] = 0
    box = estimate(mapping, semantics, records)
    assert box["available"] is True
    assert box["candidate_overlap_usable"] is False
    records[0]["xyxy"][0] = 270
    snapshot["position_relation_usable"] = False
    assert estimate(mapping, semantics, records)["candidate_overlap_usable"] is False


def test_same_frame_quality_and_rotation_are_checked_independently():
    mapping, semantics, records, _ = fixture()
    mapping.metadata["frame_quality"]["2"]["status"] = "lost"
    assert estimate(mapping, semantics, records)["available"] is False
    mapping.metadata["frame_quality"]["2"]["status"] = "tracked"
    mapping.trajectory.poses[0] = CameraPose.from_rt(np.eye(3) * 2, [0, 0, 0], frame_id=2)
    assert estimate(mapping, semantics, records)["reason"] == "invalid_pose_rotation"


def test_current_object_cannot_borrow_occupancy_from_a_previous_snapshot():
    mapping, semantics, records, _ = fixture()
    semantics.objects[0].frame_id = 3
    attach_occupancy(mapping, semantics, records)
    assert semantics.objects[0].metadata["occupancy"]["available"] is False
    assert semantics.objects[0].metadata["occupancy"]["frame_id"] == 3


def test_live_tracks_and_saved_records_have_identical_occupancy():
    mapping, semantics, records, _ = fixture()
    detection = Detection(2, 0.2, 39, "bottle", 0.9, (270, 190, 370, 290))
    semantics.tracks = [TrackedDetection(7, detection)]
    saved = [track.to_dict() for track in semantics.tracks]
    attach_occupancy(mapping, semantics)
    live = deepcopy(semantics.objects[0].metadata["occupancy"])
    attach_occupancy(mapping, semantics, saved)
    assert semantics.objects[0].metadata["occupancy"] == live


def test_duplicate_tracking_rows_do_not_silently_choose_one_membership():
    mapping, semantics, records, _ = fixture()
    records.append({**records[0], "class_name": "person"})
    assert estimate(mapping, semantics, records)["reason"] == "ambiguous_same_frame_detection"


@pytest.mark.parametrize("ratio", [0, -1, 5, float("nan"), float("inf"), True, "0.5"])
def test_invalid_thickness_policy_is_rejected(ratio):
    mapping, semantics, records, _ = fixture()
    with pytest.raises(ValueError, match="inferred_thickness_ratio"):
        attach_occupancy(mapping, semantics, records, inferred_thickness_ratio=ratio)
