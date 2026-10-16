"""Semantic map + HTML viewer tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from visualmap.detection.types import SemanticObject, SemanticResult
from visualmap.mapping import build_semantic_map
from visualmap.slam.types import CameraPose, MapPoint, MapResult, Trajectory
from visualmap.visualization import export_semantic_map_html


def _map_result() -> MapResult:
    traj = Trajectory()
    traj.append(CameraPose.identity(frame_id=0, timestamp=0.0))
    traj.append(CameraPose.from_rt(np.eye(3), [1, 0, 0], frame_id=1, timestamp=0.1))
    pts = [MapPoint(id=i, xyz=[i * 0.1, 0, 2.0]) for i in range(10)]
    return MapResult(trajectory=traj, map_points=pts)


def _sem() -> SemanticResult:
    obj = SemanticObject(
        track_id=1,
        class_id=0,
        class_name="person",
        xyz=np.array([1.0, 0.0, 3.0]),
        depth=3.0,
        first_frame=0,
        last_frame=1,
        observations=5,
        confidence=0.9,
        xyxy=(1, 1, 10, 10),
        frame_id=0,
    )
    return SemanticResult(objects=[obj])


def test_build_semantic_map():
    sm = build_semantic_map(_map_result(), _sem())
    assert len(sm.objects) == 1
    assert sm.objects[0].label == "person #001"
    assert len(sm.trajectory_xyz) == 2
    assert sm.trajectory_timestamps == [0.0, 0.1]
    assert sm.trajectory_frame_ids == [0, 1]
    assert sm.summary()["num_objects"] == 1


def test_export_html(tmp_path: Path):
    sm = build_semantic_map(_map_result(), _sem())
    path = export_semantic_map_html(sm, tmp_path, run_name="v")
    assert path.is_file()
    text = path.read_text(encoding="utf-8")
    assert "VisualMap" in text
    assert "person #001" in text or "person" in text
    assert "trajectory" in text
    assert 'id="plan"' in text
    assert 'id="objectList"' in text
    assert 'id="layerObjects"' in text
    assert '"observations": 5' in text
    assert "VisualMapOccupancy" in text
    assert 'id="layerVolumes"' in text
    import json

    occupied = json.loads((path.parent / "occupancy.json").read_text(encoding="utf-8"))
    assert occupied["volume_units"] == "map_units3"
    assert occupied["free_space_certified"] is False
    assert len(occupied["objects"]) == 1


def test_export_html_escapes_embedded_script_content(tmp_path: Path):
    sm = build_semantic_map(_map_result(), _sem())
    attack = "</script><script>window.injected=1</script>"
    sm.metadata["num_objects"] = attack
    path = export_semantic_map_html(sm, tmp_path, run_name="safe", title=attack)
    html = path.read_text(encoding="utf-8")
    assert attack not in html
    assert "\\u003c/script>" in html
    assert "value.textContent = String(v)" in html


def test_export_html_links_optional_video(tmp_path: Path):
    sm = build_semantic_map(_map_result(), _sem())
    path = export_semantic_map_html(
        sm, tmp_path, run_name="video", video_filename="source_video.mp4"
    )
    html = path.read_text(encoding="utf-8")
    assert 'id="sourceVideo"' in html
    assert '"video_filename": "source_video.mp4"' in html
    assert '"trajectory_timestamps": [0.0, 0.1]' in html


def test_export_html_rejects_video_path_escape(tmp_path: Path):
    sm = build_semantic_map(_map_result(), _sem())
    with pytest.raises(ValueError, match="run_name"):
        export_semantic_map_html(sm, tmp_path, run_name="video", video_filename="../outside.mp4")


def test_large_viewer_payload_is_lossless_and_self_contained(tmp_path):
    from visualmap.visualization.semantic_html import viewer_payload_from_html

    sm = build_semantic_map(_map_result(), _sem())
    evidence = [
        {
            "frame": frame,
            "xyz": [frame * 0.00123456789012345, 0.0, -1.23456789],
            "evidence": "原始证据，不能降采样" * 40,
        }
        for frame in range(1000)
    ]
    sm.metadata["lossless_evidence"] = evidence
    path = export_semantic_map_html(sm, tmp_path, "packed", video_filename="source_video.mp4")
    html = path.read_text(encoding="utf-8")
    assert "VISUALMAP_GZIP_DATA" in html
    assert "new DecompressionStream('gzip')" in html
    assert "__DATA__" not in html
    data = viewer_payload_from_html(html)
    assert data["summary"]["lossless_evidence"] == evidence
    assert data["run_id"] == "packed"
    assert data["video_filename"] == "source_video.mp4"
    assert len(html.encode("utf-8")) < len(str(evidence).encode("utf-8"))
