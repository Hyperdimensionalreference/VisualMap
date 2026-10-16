"""CLI integration smoke test (requires OpenCV)."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from visualmap.app import main


@pytest.fixture
def tiny_video(tmp_path: Path) -> Path:
    path = tmp_path / "t.avi"
    w, h = 64, 48
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (w, h))
    if not vw.isOpened():
        pytest.skip("VideoWriter unavailable")
    for i in range(5):
        img = np.full((h, w, 3), i * 20, dtype=np.uint8)
        vw.write(img)
    vw.release()
    return path


def test_cli_probe_only(tiny_video: Path, tmp_path: Path, capsys):
    code = main(["--video", str(tiny_video), "--probe-only", "--output", str(tmp_path)])
    assert code == 0


def test_cli_writes_report(tiny_video: Path, tmp_path: Path):
    code = main(
        [
            "--video",
            str(tiny_video),
            "--output",
            str(tmp_path),
            "--run-name",
            "fixed",
            "--max-frames",
            "3",
        ]
    )
    assert code == 0
    report = tmp_path / "fixed" / "run_report.json"
    assert report.is_file()
    assert (tmp_path / "fixed" / "video_info.json").is_file()


def test_cli_missing_video(tmp_path: Path):
    code = main(["--video", str(tmp_path / "nope.mp4"), "--output", str(tmp_path)])
    assert code == 1


def test_cli_rejects_run_name_outside_output(tiny_video: Path, tmp_path: Path):
    output = tmp_path / "out"
    code = main(
        ["--video", str(tiny_video), "--output", str(output), "--run-name", "../escape"]
    )
    assert code == 2
    assert not output.exists()
    assert not (tmp_path / "escape").exists()


def test_cli_respects_output_switches(tiny_video: Path, tmp_path: Path):
    config_path = tmp_path / "settings.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "video": {"path": str(tiny_video)},
                "detection": {"enabled": False},
                "depth": {"enabled": False},
                "output": {
                    "dir": str(tmp_path / "out"),
                    "save_trajectory_csv": False,
                    "save_trajectory_txt": False,
                    "save_map_ply": False,
                    "save_report_json": False,
                },
            }
        ),
        encoding="utf-8",
    )
    assert main(["--config", str(config_path), "--run-name", "disabled"]) == 0
    run_dir = tmp_path / "out" / "disabled"
    assert (run_dir / "video_info.json").is_file()
    assert not (run_dir / "trajectory.csv").exists()
    assert not (run_dir / "trajectory.txt").exists()
    assert not (run_dir / "map_points.ply").exists()
    assert not (run_dir / "run_report.json").exists()
