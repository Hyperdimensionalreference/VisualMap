"""Rebuilding a viewer must preserve rotation, evidence, and the video link."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from visualmap.detection.types import Detection, SemanticObject, SemanticResult, TrackedDetection
from visualmap.export.reconstruction import export_reconstruction
from visualmap.export.semantics_writers import export_semantics
from visualmap.export.writers import export_report, export_trajectory_csv
from visualmap.slam.types import CameraIntrinsics, CameraPose, MapPoint, MapResult, Trajectory


@pytest.fixture
def rebuild_module():
    path = Path(__file__).parents[2] / "scripts" / "rebuild_view.py"
    spec = importlib.util.spec_from_file_location("visualmap_rebuild_view_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def exported_run(tmp_path: Path):
    run_dir = tmp_path / "手机视频结果"
    run_dir.mkdir()
    rotation = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])
    intrinsics = CameraIntrinsics(100.0, 100.0, 50.0, 40.0, 100, 80)
    poses = Trajectory(
        [
            CameraPose.identity(frame_id=0, timestamp=0.0),
            CameraPose.from_rt(rotation, [2.0, 3.0, 4.0], frame_id=659, timestamp=21.966),
        ]
    )
    map_result = MapResult(trajectory=poses, intrinsics=intrinsics)
    export_trajectory_csv(map_result, run_dir / "trajectory.csv")
    export_report(map_result, run_dir / "run_report.json")
    (run_dir / "video_info.json").write_text(
        json.dumps({"camera": intrinsics.to_dict()}),
        encoding="utf-8",
    )
    metadata = {
        "position_status": "retained",
        "position_frame": 300,
        "position_history": [
            {
                "frame_id": 300,
                "timestamp": 10.0,
                "xyz": [1.0, 2.0, 5.0],
                "source": "sparse_observed",
                "uncertainty": 0.2,
                "status": "estimated",
            }
        ],
        "classification_consistency": 0.8,
        "classification_evidence": {
            "latest_raw_class": "cup",
            "latest_raw_confidence": 0.6,
            "accuracy_validated": False,
        },
    }
    obj = SemanticObject(
        track_id=1,
        class_id=39,
        class_name="bottle",
        xyz=np.array([1.0, 2.0, 5.0]),
        depth=5.0,
        first_frame=0,
        last_frame=659,
        observations=660,
        confidence=0.7,
        xyxy=(10.0, 10.0, 20.0, 30.0),
        frame_id=659,
        metadata=metadata,
    )
    tracks = [
        TrackedDetection(
            track_id=1,
            detection=Detection(i, i / 30.0, 39, "bottle", 0.7, (10.0, 10.0, 20.0, 30.0)),
            hits=i + 1,
        )
        for i in range(660)
    ]
    export_semantics(SemanticResult(objects=[obj], tracks=tracks), tmp_path, run_dir.name)
    (run_dir / "source_video.mp4").write_bytes(b"local-video-link-fixture")
    return run_dir, map_result, metadata


def test_load_run_restores_exported_quaternion_rotation(rebuild_module, exported_run):
    run_dir, original, _ = exported_run
    restored = rebuild_module.load_run(run_dir)
    assert [p.frame_id for p in restored.trajectory] == [0, 659]
    expected_pose = original.trajectory.poses[-1]
    restored_pose = restored.trajectory.poses[-1]
    np.testing.assert_allclose(restored_pose.matrix, expected_pose.matrix, atol=1e-8)
    np.testing.assert_allclose(restored_pose.transform_point([0.0, 0.0, 1.0]), [3.0, 3.0, 4.0])


def test_load_run_prefers_complete_reconstruction_for_independent_submaps(
    rebuild_module,
    exported_run,
):
    run_dir, original, _ = exported_run
    original.map_points = [
        MapPoint(8, [1, 2, 3], observations={0: 6}),
        MapPoint(9, [3, 2, 1], observations={659: 7}),
    ]
    original.metadata.update(
        submaps=[{"world_frame_id": "a"}, {"world_frame_id": "b"}],
        landmark_world_frame_ids={"8": "a", "9": "b"},
    )
    export_reconstruction(original, run_dir / "reconstruction.json")
    restored = rebuild_module.load_run(run_dir)
    assert [point.id for point in restored.map_points] == [8, 9]
    assert restored.map_points[1].observations == {659: 7}
    assert restored.metadata["landmark_world_frame_ids"] == {"8": "a", "9": "b"}


def test_load_semantics_preserves_history_classification_and_last_frame(
    rebuild_module,
    exported_run,
):
    run_dir, _, metadata = exported_run
    restored = rebuild_module.load_semantics(run_dir)
    assert restored.objects[0].metadata == metadata
    assert len(restored.tracks) == 660
    assert restored.tracks[-1].detection.frame_id == 659
    assert restored.objects[0].last_frame == 659


def test_rebuild_keeps_local_video_and_full_viewer_evidence(
    rebuild_module,
    exported_run,
    monkeypatch,
):
    run_dir, original, metadata = exported_run
    source_video = run_dir / "source_video.mp4"
    before = source_video.read_bytes()
    monkeypatch.setattr(sys, "argv", ["rebuild_view.py", str(run_dir)])
    rebuild_module.main()
    html = (run_dir / "semantic_map.html").read_text(encoding="utf-8")
    from visualmap.visualization.semantic_html import viewer_payload_from_html

    data = viewer_payload_from_html(html)
    assert data["video_filename"] == "source_video.mp4"
    assert source_video.read_bytes() == before
    assert data["trajectory_frame_ids"][-1] == 659
    assert data["object_observations"][-1]["frame_id"] == 659
    assert len(data["object_observations"]) == 660
    assert data["objects"][0]["position_history"] == metadata["position_history"]
    assert data["objects"][0]["classification_evidence"] == metadata["classification_evidence"]
    np.testing.assert_allclose(
        data["trajectory_rotations"][-1],
        original.trajectory.poses[-1].R,
        atol=1e-8,
    )
