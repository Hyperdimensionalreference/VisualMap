"""Presentation refresh must preserve the live export's per-frame evidence contract."""

from __future__ import annotations

from copy import deepcopy

import numpy as np

from visualmap.detection.types import Detection, SemanticObject, SemanticResult, TrackedDetection
from visualmap.mapping.semantic_map import build_semantic_map, observations_from_records
from visualmap.slam.types import CameraPose, MapResult, Trajectory


def test_saved_tracking_join_preserves_exact_frame_evidence_and_classification():
    detection = Detection(1, 0.1, 39, "bottle", 0.9, (10, 20, 30, 40))
    tracked = TrackedDetection(7, detection)
    snapshot = {
        "frame_id": 1,
        "xyz": [1, 2, 3],
        "pose_valid": True,
        "position_relation_usable": True,
        "world_frame_id": "world-a",
        "support_evidence": {"point_ids": [11, 12, 13]},
    }
    obj = SemanticObject(
        7,
        39,
        "bottle",
        np.array([70.0, 80.0, 90.0]),
        90,
        1,
        8,
        8,
        0.9,
        (10, 20, 30, 40),
        8,
        metadata={"position_snapshots": [snapshot]},
    )
    result = MapResult(Trajectory([CameraPose.identity(1, 0.1)]))
    live = build_semantic_map(result, SemanticResult(objects=[obj], tracks=[tracked]))
    stored = [tracked.to_dict(), {**tracked.to_dict(), "track_id": 999}]
    original = deepcopy(stored)
    rebuilt = observations_from_records([obj], stored)
    assert rebuilt == live.object_observations
    assert rebuilt[0]["geometry"]["xyz"] == [1, 2, 3]
    assert rebuilt[0]["geometry"]["xyz"] != obj.xyz.tolist()
    assert stored == original
    rebuilt[0]["geometry"]["support_evidence"]["point_ids"].append(99)
    assert snapshot["support_evidence"]["point_ids"] == [11, 12, 13]


def test_saved_track_without_exact_snapshot_cannot_borrow_future_geometry():
    obj = SemanticObject(
        7,
        39,
        "bottle",
        np.array([70.0, 80.0, 90.0]),
        90,
        1,
        8,
        8,
        0.9,
        (10, 20, 30, 40),
        8,
        metadata={"position_snapshots": [{"frame_id": 8, "xyz": [70, 80, 90]}]},
    )
    records = [
        {"track_id": 7, "frame_id": 1, "class_name": "bottle", "classification_needs_review": True}
    ]
    row = observations_from_records([obj], records)[0]
    assert row["classification_needs_review"] is True
    assert row["geometry"]["xyz"] is None
    assert row["geometry"]["position_relation_usable"] is False
    assert row["geometry"]["pose_valid"] is False
