"""YOLOX adapter contract and portrait coordinate restoration."""

from __future__ import annotations

import numpy as np
import pytest

from visualmap.detection.yolo_onnx import YOLOONNXConfig, YOLOONNXDetector
from visualmap.detection.yolox_onnx import YOLOXONNXConfig, YOLOXONNXDetector
from visualmap.slam.types import Frame


class _FakeNet:
    def __init__(self, output: np.ndarray) -> None:
        self.output = output
        self.blob = None

    def setInput(self, blob: np.ndarray) -> None:  # noqa: N802 - OpenCV API spelling
        self.blob = blob

    def forward(self) -> np.ndarray:
        return self.output

    def setPreferableBackend(self, _value: int) -> None:  # noqa: N802
        pass

    def setPreferableTarget(self, _value: int) -> None:  # noqa: N802
        pass


def _detector(output: np.ndarray) -> YOLOXONNXDetector:
    detector = YOLOXONNXDetector.__new__(YOLOXONNXDetector)
    detector.config = YOLOXONNXConfig(model_path="unused.onnx")
    detector._grid, detector._strides = detector._make_grid(640, 640)
    detector._net = _FakeNet(output)
    return detector


def test_yolox_restores_portrait_box_and_class():
    output = np.zeros((1, 8400, 85), dtype=np.float32)
    output[0, 0, :2] = 2.0
    output[0, 0, 2:4] = np.log(4.0)
    output[0, 0, 4] = 0.9
    output[0, 0, 5] = 0.8
    detector = _detector(output)
    frame = Frame(index=13, timestamp=0.5, image=np.zeros((640, 320, 3), dtype=np.uint8))

    detections = detector.detect(frame)

    assert len(detections) == 1
    assert detections[0].class_name == "person"
    assert detections[0].frame_id == 13
    assert detections[0].confidence == pytest.approx(0.72)
    assert detections[0].xyxy == pytest.approx((0, 0, 32, 32))
    assert detector._net.blob.shape == (1, 3, 640, 640)
    assert np.all(detector._net.blob[:, :, :, 320:] == 114)


def test_yolox_rejects_incompatible_model_output():
    detector = _detector(np.zeros((1, 8400, 84), dtype=np.float32))
    frame = Frame(index=0, timestamp=0.0, image=np.zeros((640, 640, 3), dtype=np.uint8))
    with pytest.raises(ValueError, match="output shape"):
        detector.detect(frame)


def test_yolox_requires_stride_aligned_input():
    with pytest.raises(ValueError, match="multiples of 32"):
        YOLOXONNXConfig(model_path="unused.onnx", input_width=641)


@pytest.mark.parametrize("detector_class,config_class", [
    (YOLOXONNXDetector, YOLOXONNXConfig),
    (YOLOONNXDetector, YOLOONNXConfig),
])
def test_model_loading_accepts_unicode_path(
    tmp_path, monkeypatch, detector_class, config_class
):
    model = tmp_path / "中文模型.onnx"
    model.write_bytes(b"fake onnx bytes")
    captured = []

    def load(buffer):
        captured.append(buffer.tobytes())
        return _FakeNet(np.empty(0))

    monkeypatch.setattr("cv2.dnn.readNetFromONNX", load)
    detector_class(config_class(model_path=model))
    assert captured == [b"fake onnx bytes"]
