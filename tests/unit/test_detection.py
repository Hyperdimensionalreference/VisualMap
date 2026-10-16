"""Tests for detection, tracking, lift, and semantics export."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from visualmap.depth.types import DepthMap, DepthResult
from visualmap.detection import (
    Detection,
    IoUTracker,
    LiftConfig,
    SemanticPipeline,
    SemanticPipelineConfig,
    TrackerConfig,
    create_detector,
    lift_detection,
)
from visualmap.detection.yolo_onnx import YOLOONNXConfig, YOLOONNXDetector
from visualmap.export import export_semantics
from visualmap.slam.types import (
    CameraIntrinsics,
    CameraPose,
    Frame,
    MapPoint,
    MapResult,
    Trajectory,
)


def _intr() -> CameraIntrinsics:
    return CameraIntrinsics(fx=100.0, fy=100.0, cx=50.0, cy=40.0, width=100, height=80)


def _det(fid: int, box, cls=0, name="person", conf=0.9) -> Detection:
    return Detection(
        frame_id=fid,
        timestamp=fid / 10.0,
        class_id=cls,
        class_name=name,
        confidence=conf,
        xyxy=box,
    )


def test_iou_tracker_assigns_ids():
    tr = IoUTracker(TrackerConfig(iou_threshold=0.3))
    d0 = [_det(0, (10, 10, 30, 30))]
    t0 = tr.update(d0)
    assert len(t0) == 1
    id0 = t0[0].track_id

    d1 = [_det(1, (11, 11, 31, 31))]
    t1 = tr.update(d1)
    assert len(t1) == 1
    assert t1[0].track_id == id0  # same object


def test_iou_tracker_new_object():
    tr = IoUTracker(TrackerConfig(iou_threshold=0.3))
    tr.update([_det(0, (0, 0, 10, 10))])
    t1 = tr.update(
        [
            _det(1, (0, 0, 10, 10)),
            _det(1, (50, 50, 70, 70), cls=1, name="chair"),
        ]
    )
    assert len(t1) == 2
    assert len({t.track_id for t in t1}) == 2


def test_tracker_expires_after_empty_frames():
    tracker = IoUTracker(TrackerConfig(max_age=1, min_hits=2))
    assert tracker.update([_det(0, (10, 10, 30, 30))]) == []
    tracker.update([])
    tracker.update([])
    assert tracker.update([_det(3, (10, 10, 30, 30))]) == []


def test_lift_with_depth():
    intr = _intr()
    pose = CameraPose.identity()
    depth = np.full((80, 100), 5.0, dtype=np.float32)
    dm = DepthMap(
        frame_id=0,
        timestamp=0.0,
        depth=depth,
        source_width=100,
        source_height=80,
    )
    det = _det(0, (40, 30, 60, 50))  # center ~ (50, 40) = principal point
    obj = lift_detection(det, pose, intr, depth_map=dm, config=LiftConfig())
    assert obj.depth == pytest.approx(5.0)
    assert obj.xyz is not None
    assert obj.xyz[2] == pytest.approx(5.0)


def test_pipeline_and_export(tmp_path: Path):
    traj = Trajectory()
    traj.append(CameraPose.identity(frame_id=0, timestamp=0.0))
    traj.append(CameraPose.from_rt(np.eye(3), [0.2, 0, 0], frame_id=1, timestamp=0.1))
    map_result = MapResult(
        trajectory=traj,
        map_points=[MapPoint(id=0, xyz=[0, 0, 5])],
        intrinsics=_intr(),
    )
    depth = DepthResult(
        maps=[
            DepthMap(
                frame_id=0,
                timestamp=0.0,
                depth=np.full((80, 100), 4.5, dtype=np.float32),
                source_width=100,
                source_height=80,
            )
        ]
    )
    pipe = SemanticPipeline(config=SemanticPipelineConfig())
    pipe.process_detections([_det(0, (20, 20, 40, 40))], map_result, depth, _intr())
    pipe.process_detections([_det(1, (22, 22, 42, 42))], map_result, depth, _intr())
    result = pipe.result()
    assert len(result.objects) >= 1
    assert result.objects[0].first_frame == 0
    assert result.objects[0].last_frame == 1
    written = export_semantics(result, tmp_path, run_name="s1")
    assert (tmp_path / "s1" / "semantics.json").is_file()
    assert (tmp_path / "s1" / "semantic_objects.csv").is_file()
    assert "semantics_json" in written


def test_create_detector_demo():
    det = create_detector("demo")
    assert det.name == "demo"


def test_create_detector_unknown():
    with pytest.raises(ValueError):
        create_detector("nope")


def test_create_detector_yolo_requires_model():
    with pytest.raises(ValueError):
        create_detector("yolo_onnx")


def test_yolo_letterbox_restores_portrait_boxes():
    class FakeNet:
        blob = None

        def setInput(self, blob):
            self.blob = blob

        def forward(self):
            # One prediction in a 64x64 letterboxed image, two classes.
            return np.array([[[32, 32, 16, 16, 0.9, 0.1]]], dtype=np.float32)

    detector = object.__new__(YOLOONNXDetector)
    detector.config = YOLOONNXConfig(
        model_path=Path("unused.onnx"),
        class_names=("person", "chair"),
        input_width=64,
        input_height=64,
    )
    detector._net = FakeNet()
    frame = Frame(index=0, timestamp=0, image=np.full((60, 30, 3), 255, dtype=np.uint8))
    detections = detector.detect(frame)
    assert len(detections) == 1
    assert detections[0].class_name == "person"
    assert detections[0].xyxy == pytest.approx((7.5, 22.5, 22.5, 37.5))
    assert detector._net.blob.shape == (1, 3, 64, 64)
    assert detector._net.blob[0, 0, 32, 0] == pytest.approx(114 / 255)


def test_yolo_nms_keeps_overlapping_different_classes():
    detector = object.__new__(YOLOONNXDetector)
    detector.config = YOLOONNXConfig(
        model_path=Path("unused.onnx"),
        class_names=("person", "chair"),
        input_width=64,
        input_height=64,
    )
    output = np.array(
        [[[32, 32, 20, 20, 0.9, 0.1], [32, 32, 20, 20, 0.1, 0.85]]],
        dtype=np.float32,
    )
    detections = detector._postprocess(output, 64, 64, 1.0, 0, 0)
    assert {class_id for class_id, _, _ in detections} == {0, 1}
