"""Decoder regressions that otherwise fabricate boxes or silently lose classes."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from visualmap.detection.yolo_onnx import YOLOONNXConfig, YOLOONNXDetector
from visualmap.detection.yolox_onnx import YOLOXONNXConfig, YOLOXONNXDetector


def _yolo() -> YOLOONNXDetector:
    detector = YOLOONNXDetector.__new__(YOLOONNXDetector)
    detector.config = YOLOONNXConfig(Path("unused"), class_names=("person", "chair"))
    return detector


@pytest.mark.parametrize("row", [
    (32, 32, 20, 20, np.nan, 0.8),
    (np.inf, 32, 20, 20, 0.9, 0.1),
    (32, 32, np.inf, 20, 0.9, 0.1),
    (32, 32, -20, 20, 0.9, 0.1),
    (32, 32, 20, 20, 1.2, 0.1),
    (32, 32, 20, 20, 0.9, np.inf),
])
def test_yolo_rejects_invalid_output_rows(row):
    output = np.asarray([[row]], dtype=np.float32)
    assert _yolo()._postprocess(output, 64, 64, 1.0, 0, 0) == []


def test_yolo_rejects_negative_objectness_with_negative_class_probabilities():
    output = np.array([[[32, 32, 20, 20, -0.9, -0.8, -0.9]]], dtype=np.float32)
    assert _yolo()._postprocess(output, 64, 64, 1.0, 0, 0) == []


@pytest.mark.parametrize("output", [
    np.zeros((2, 10, 6)),
    np.zeros((1, 10, 8)),
    np.zeros((1, 10, 6, 1)),
])
def test_yolo_incompatible_output_raises_instead_of_hiding_model_mismatch(output):
    with pytest.raises(ValueError, match="output shape"):
        _yolo()._postprocess(output, 64, 64, 1.0, 0, 0)


@pytest.mark.parametrize("objectness", [False, True])
def test_yolo_transposed_single_prediction_preserves_class_and_score(objectness):
    row = [32, 32, 20, 20]
    if objectness:
        row.append(0.8)
    row += [0.1, 0.9]
    output = np.asarray(row, dtype=np.float32).reshape(1, len(row), 1)
    result = _yolo()._postprocess(output, 64, 64, 1.0, 0, 0)
    assert len(result) == 1
    assert result[0][0] == 1
    assert result[0][1] == pytest.approx(0.72 if objectness else 0.9)
    assert result[0][2] == pytest.approx((22, 22, 42, 42))


def test_yolo_same_class_nms_and_deterministic_confidence_order():
    output = np.asarray([[
        [32, 32, 20, 20, 0.9, 0.1],
        [33, 32, 20, 20, 0.8, 0.2],
        [50, 50, 10, 10, 0.1, 0.95],
    ]], dtype=np.float32)
    result = _yolo()._postprocess(output, 64, 64, 1.0, 0, 0)
    assert [item[0] for item in result] == [1, 0]
    assert [item[1] for item in result] == pytest.approx([0.95, 0.9])


@pytest.mark.parametrize("column,value", [
    (0, np.inf),
    (2, np.inf),
    (3, np.nan),
    (4, np.nan),
    (5, np.inf),
    (4, -0.9),
])
def test_yolox_nonfinite_or_invalid_scores_never_become_large_clipped_boxes(column, value):
    detector = YOLOXONNXDetector.__new__(YOLOXONNXDetector)
    detector.config = YOLOXONNXConfig(Path("unused"), input_width=32, input_height=32)
    detector._grid, detector._strides = detector._make_grid(32, 32)
    output = np.zeros((1, len(detector._grid), 85), dtype=np.float32)
    output[0, 0, :2] = 2
    output[0, 0, 2:4] = np.log(4)
    output[0, 0, 4:6] = (0.9, 0.8)
    output[0, 0, column] = value
    assert detector._postprocess(output, 32, 32, 1.0) == []
