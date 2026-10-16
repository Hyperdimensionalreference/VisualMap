"""Meaningful detection evaluation cases, including wrong-class overlaps."""

from __future__ import annotations

import pytest

from visualmap.evaluation import evaluate_review_set


def _manifest():
    return {
        "dataset_id": "sample-1",
        "confidence_threshold": 0.4,
        "classes": ["bottle", "laptop", "tablet"],
        "frames": [
            {
                "frame_id": 0, "width": 100, "height": 100,
                "predictions": [
                    {"class_name": "laptop", "confidence": 0.9, "xyxy": [10, 10, 50, 50]},
                    {"class_name": "bottle", "confidence": 0.8, "xyxy": [60, 10, 80, 50]},
                ],
            },
            {"frame_id": 1, "width": 100, "height": 100, "predictions": []},
        ],
    }


def _annotations():
    return {
        "dataset_id": "sample-1",
        "reviewed_frame_ids": [0, 1],
        "frames": {
            "0": [
                {"class_name": "tablet", "xyxy": [11, 10, 51, 50]},
                {"class_name": "bottle", "xyxy": [60, 10, 80, 50]},
            ],
            "1": [],
        },
    }


def test_evaluation_counts_wrong_class_as_fp_and_fn():
    result = evaluate_review_set(_manifest(), _annotations())
    assert result["classes"]["bottle"]["tp"] == 1
    assert result["classes"]["laptop"]["fp"] == 1
    assert result["classes"]["tablet"]["fn"] == 1
    assert result["confusions"][0]["predicted"] == "laptop"
    assert result["confusions"][0]["actual"] == "tablet"
    assert result["partial"] is False


def test_evaluation_requires_reviewed_frames_by_default():
    annotations = _annotations()
    annotations["reviewed_frame_ids"] = [0]
    with pytest.raises(ValueError, match="review incomplete"):
        evaluate_review_set(_manifest(), annotations)
    result = evaluate_review_set(_manifest(), annotations, allow_partial=True)
    assert result["reviewed_frames"] == 1
    assert result["partial"] is True


def test_evaluation_rejects_wrong_set_and_out_of_frame_box():
    annotations = _annotations()
    annotations["dataset_id"] = "elsewhere"
    with pytest.raises(ValueError, match="different review set"):
        evaluate_review_set(_manifest(), annotations)
    annotations = _annotations()
    annotations["frames"]["0"][0]["xyxy"] = [11, 10, 101, 50]
    with pytest.raises(ValueError, match="inside the source frame"):
        evaluate_review_set(_manifest(), annotations)


def test_evaluation_cannot_recover_predictions_below_export_threshold():
    with pytest.raises(ValueError, match="lower than exported"):
        evaluate_review_set(_manifest(), _annotations(), confidence_threshold=0.3)
