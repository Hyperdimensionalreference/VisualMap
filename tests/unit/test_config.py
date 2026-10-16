from pathlib import Path

import pytest
import yaml

from visualmap.config import load_config
from visualmap.slam.types import CameraIntrinsics


def test_load_config_minimal(tmp_path: Path):
    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text(yaml.safe_dump({"video": {"path": "a.mp4"}}), encoding="utf-8")
    cfg = load_config(cfg_path)
    assert cfg.video.path.name == "a.mp4"
    assert cfg.slam.backend == "opencv_vo"
    assert cfg.output.save_trajectory_csv is True


def test_load_config_overrides(tmp_path: Path):
    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text(yaml.safe_dump({"video": {"path": "a.mp4"}}), encoding="utf-8")
    cfg = load_config(
        cfg_path,
        overrides={"video.path": str(tmp_path / "b.mp4"), "log_level": "DEBUG"},
    )
    assert cfg.video.path.name == "b.mp4"
    assert cfg.log_level == "DEBUG"


def test_load_config_requires_video_path(tmp_path: Path):
    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text("log_level: INFO\n", encoding="utf-8")
    with pytest.raises(ValueError, match="video.path"):
        load_config(cfg_path)


def test_occupancy_thickness_ratio_is_configurable_and_rejects_invalid_values(tmp_path: Path):
    cfg_path = tmp_path / "occupied.yaml"
    for value in (0.25, 0.8):
        cfg_path.write_text(
            yaml.safe_dump(
                {"video": {"path": "a.mp4"}, "detection": {"occupancy_thickness_ratio": value}}
            ),
            encoding="utf-8",
        )
        assert load_config(cfg_path).detection.occupancy_thickness_ratio == value
    for value in (0, -1, 5, float("nan"), float("inf")):
        cfg_path.write_text(
            yaml.safe_dump(
                {"video": {"path": "a.mp4"}, "detection": {"occupancy_thickness_ratio": value}}
            ),
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="occupancy_thickness_ratio"):
            load_config(cfg_path)


def test_camera_inline_from_config(tmp_path: Path):
    cfg_path = tmp_path / "c.yaml"
    payload = {
        "video": {"path": "a.mp4"},
        "camera": {"fx": 10, "fy": 10, "cx": 5, "cy": 5, "width": 20, "height": 10},
    }
    cfg_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    cfg = load_config(cfg_path)
    intr = cfg.camera.resolve(20, 10)
    assert isinstance(intr, CameraIntrinsics)
    assert intr.fx == 10


def test_camera_default_heuristic(tmp_path: Path):
    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text(yaml.safe_dump({"video": {"path": "a.mp4"}}), encoding="utf-8")
    cfg = load_config(cfg_path)
    intr = cfg.camera.resolve(1920, 1080)
    assert intr.fx == pytest.approx(1.2 * 1920)
    assert intr.cx == pytest.approx(960.0)


@pytest.mark.parametrize("run_name", ["../escape", r"..\escape", "/tmp/escape", "C:escape", "."])
def test_run_name_cannot_escape_output_directory(tmp_path: Path, run_name: str):
    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text(yaml.safe_dump({"video": {"path": "a.mp4"}}), encoding="utf-8")
    with pytest.raises(ValueError, match="run_name"):
        load_config(cfg_path, overrides={"run_name": run_name})


def test_camera_file_accepts_documented_opencv_format(tmp_path: Path):
    camera_path = tmp_path / "camera.yaml"
    camera_path.write_text(
        yaml.safe_dump(
            {
                "camera_matrix": [[100, 0, 50], [0, 110, 40], [0, 0, 1]],
                "dist_coeffs": [0, 0, 0, 0, 0],
                "image_size": [100, 80],
            }
        ),
        encoding="utf-8",
    )
    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {"video": {"path": "a.mp4"}, "camera": {"intrinsics_file": str(camera_path)}}
        ),
        encoding="utf-8",
    )
    intr = load_config(cfg_path).camera.resolve(100, 80)
    assert (intr.fx, intr.fy, intr.width, intr.height) == (100, 110, 100, 80)


def test_custom_detection_classes_and_input_size(tmp_path: Path):
    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "video": {"path": "a.mp4"},
                "detection": {
                    "detector": "yolo_onnx",
                    "input_width": 512,
                    "input_height": 512,
                    "min_track_hits": 3,
                    "class_names": ["person", "chair"],
                },
            }
        ),
        encoding="utf-8",
    )
    detection = load_config(cfg_path).detection
    assert detection.input_width == detection.input_height == 512
    assert detection.class_names == ("person", "chair")
    assert detection.min_track_hits == 3


def test_detection_rejects_class_names_string(tmp_path: Path):
    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text(
        yaml.safe_dump({"video": {"path": "a.mp4"}, "detection": {"class_names": "person"}}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="class_names"):
        load_config(cfg_path)
