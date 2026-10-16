"""Integration: full VO pipeline writes trajectory artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from visualmap.app import main


@pytest.fixture
def pan_video(tmp_path: Path) -> Path:
    path = tmp_path / "pan.avi"
    w, h, n = 200, 150, 25
    scene = np.zeros((220, 320, 3), dtype=np.uint8)
    rng = np.random.default_rng(3)
    for _ in range(80):
        x = int(rng.integers(0, 300))
        y = int(rng.integers(0, 200))
        cv2.rectangle(scene, (x, y), (x + 18, y + 18), (int(rng.integers(0, 255)), 255, 120), -1)
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 15, (w, h))
    if not vw.isOpened():
        pytest.skip("VideoWriter unavailable")
    for i in range(n):
        ox = 5 + i * 2
        crop = scene[10 : 10 + h, ox : ox + w].copy()
        vw.write(crop)
    vw.release()
    return path


def test_cli_vo_writes_trajectory(pan_video: Path, tmp_path: Path):
    code = main(
        [
            "--video",
            str(pan_video),
            "--output",
            str(tmp_path / "out"),
            "--run-name",
            "vo",
            "--max-frames",
            "20",
        ]
    )
    assert code == 0
    run = tmp_path / "out" / "vo"
    assert (run / "trajectory.csv").is_file()
    assert (run / "trajectory.txt").is_file()
    assert (run / "map_points.ply").is_file()
    assert (run / "run_report.json").is_file()

    lines = (run / "trajectory.csv").read_text(encoding="utf-8").strip().splitlines()
    assert lines[0].startswith("frame_id")
    assert len(lines) == 21  # header + 20 poses

    txt_lines = (run / "trajectory.txt").read_text(encoding="utf-8").strip().splitlines()
    assert len(txt_lines) == 20
    report = json.loads((run / "run_report.json").read_text(encoding="utf-8"))
    assert report["video_ingestion"]["completeness"] == "requested_partial"
    assert report["video_ingestion"]["next_frame_index"] == 20
    assert report["video_ingestion"]["reported_frame_count"] == 25


def test_cli_real_video_full_decode_reports_observed_source_extent(pan_video: Path, tmp_path: Path):
    code = main(
        ["--video", str(pan_video), "--output", str(tmp_path / "out"), "--run-name", "full"]
    )
    assert code == 0
    report = json.loads((tmp_path / "out/full/run_report.json").read_text(encoding="utf-8"))
    assert report["video_ingestion"]["stop_reason"] == "normal_eof"
    assert report["video_ingestion"]["completeness"] == "matches_reported_length"
    assert report["video_ingestion"]["decoded_frames"] == 25
    assert report["video_ingestion"]["full_video_processed"] is True


def test_cli_premature_decoder_failure_is_nonzero_and_leaves_failure_report(
    monkeypatch, tmp_path: Path
):
    path = tmp_path / "declares_ten.avi"
    path.touch()

    class _PrematureCapture:
        position = 0

        def isOpened(self):
            return True

        def get(self, prop):
            return {
                cv2.CAP_PROP_FRAME_WIDTH: 64,
                cv2.CAP_PROP_FRAME_HEIGHT: 48,
                cv2.CAP_PROP_FPS: 10,
                cv2.CAP_PROP_FRAME_COUNT: 10,
                cv2.CAP_PROP_POS_FRAMES: self.position,
            }.get(prop, 0)

        def set(self, prop, value):
            self.position = int(value)
            return True

        def read(self):
            if self.position == 4:
                return False, None
            self.position += 1
            return True, np.zeros((48, 64, 3), np.uint8)

        def release(self):
            pass

    monkeypatch.setattr("visualmap.video.source.cv2.VideoCapture", lambda path: _PrematureCapture())
    code = main(["--video", str(path), "--output", str(tmp_path / "out"), "--run-name", "failed"])
    assert code == 1
    run = tmp_path / "out/failed"
    report = json.loads((run / "run_report.json").read_text(encoding="utf-8"))
    assert report["run_status"] == "failed_video_decode"
    assert report["frames_read"] == 4
    assert report["video_ingestion"]["next_frame_index"] == 4
    assert report["video_ingestion"]["completeness"] == "incomplete"
    assert not (run / "semantic_map.html").exists()
    assert not (run / "trajectory.csv").exists()
