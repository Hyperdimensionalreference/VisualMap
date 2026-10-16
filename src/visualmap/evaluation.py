"""Evaluate reviewed 2D detection boxes against manually drawn boxes."""

from __future__ import annotations

import math
from collections import defaultdict

__all__ = ["evaluate_review_set"]


def _box(box: list[float], width: int, height: int) -> tuple[float, float, float, float]:
    if len(box) != 4 or not all(isinstance(v, (int, float)) for v in box):
        raise ValueError("each box must contain four numbers")
    x1, y1, x2, y2 = (float(v) for v in box)
    if not all(math.isfinite(v) for v in (x1, y1, x2, y2)):
        raise ValueError("box coordinates must be finite")
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise ValueError("box coordinates must stay inside the source frame")
    return x1, y1, x2, y2


def _iou(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def evaluate_review_set(
    manifest: dict,
    annotations: dict,
    *,
    iou_threshold: float = 0.5,
    confidence_threshold: float | None = None,
    allow_partial: bool = False,
) -> dict:
    """Return per-class precision/recall and inspectable errors.

    Only explicitly reviewed frames count. A complete report is required unless
    ``allow_partial`` is set; this prevents unlabeled frames becoming false negatives.
    """
    if not 0 < iou_threshold <= 1:
        raise ValueError("iou_threshold must be in (0, 1]")
    if manifest.get("dataset_id") != annotations.get("dataset_id"):
        raise ValueError("annotations belong to a different review set")
    classes = manifest.get("classes")
    if not isinstance(classes, list) or not classes or len(classes) != len(set(classes)):
        raise ValueError("manifest classes must be a unique nonempty list")
    frame_list = manifest.get("frames")
    if not isinstance(frame_list, list) or not frame_list:
        raise ValueError("manifest has no frames")
    frame_by_id = {int(frame["frame_id"]): frame for frame in frame_list}
    if len(frame_by_id) != len(frame_list):
        raise ValueError("manifest frame ids must be unique")
    reviewed = {int(fid) for fid in annotations.get("reviewed_frame_ids", [])}
    if not reviewed <= frame_by_id.keys():
        raise ValueError("annotations contain unknown frame ids")
    if not allow_partial and reviewed != frame_by_id.keys():
        raise ValueError(
            f"review incomplete: {len(reviewed)}/{len(frame_by_id)} frames; "
            "pass --allow-partial for an interim report"
        )
    if not reviewed:
        raise ValueError("no reviewed frames")

    labels_by_frame = annotations.get("frames", {})
    source_threshold = float(manifest.get("confidence_threshold", 0.0))
    threshold = source_threshold if confidence_threshold is None else confidence_threshold
    if not 0 <= threshold <= 1:
        raise ValueError("confidence_threshold must be in [0, 1]")
    if threshold < source_threshold:
        raise ValueError("confidence_threshold cannot be lower than exported predictions")
    counts = {name: {"tp": 0, "fp": 0, "fn": 0} for name in classes}
    errors: list[dict] = []

    for frame_id in sorted(reviewed):
        frame = frame_by_id[frame_id]
        width, height = int(frame["width"]), int(frame["height"])
        truths = []
        for entry in labels_by_frame.get(str(frame_id), []):
            name = entry["class_name"]
            if name not in counts:
                raise ValueError(f"unknown annotated class: {name}")
            truths.append((name, _box(entry["xyxy"], width, height)))
        predictions = []
        for entry in frame.get("predictions", []):
            name = entry["class_name"]
            score = float(entry["confidence"])
            if name in counts and score >= threshold:
                predictions.append((name, score, _box(entry["xyxy"], width, height)))
        predictions.sort(key=lambda item: item[1], reverse=True)

        matched_truths: set[int] = set()
        for name, score, box in predictions:
            candidates = [
                (_iou(box, truth_box), index)
                for index, (truth_name, truth_box) in enumerate(truths)
                if truth_name == name and index not in matched_truths
            ]
            best_iou, best_index = max(candidates, default=(0.0, -1))
            if best_iou >= iou_threshold:
                matched_truths.add(best_index)
                counts[name]["tp"] += 1
            else:
                counts[name]["fp"] += 1
                errors.append({
                    "frame_id": frame_id, "kind": "false_positive", "class_name": name,
                    "xyxy": list(box), "confidence": score,
                })
        for index, (name, box) in enumerate(truths):
            if index not in matched_truths:
                counts[name]["fn"] += 1
                errors.append({
                    "frame_id": frame_id, "kind": "false_negative", "class_name": name,
                    "xyxy": list(box),
                })

    metrics = {}
    totals = defaultdict(int)
    for name, count in counts.items():
        tp, fp, fn = count["tp"], count["fp"], count["fn"]
        precision = tp / (tp + fp) if tp + fp else None
        recall = tp / (tp + fn) if tp + fn else None
        metrics[name] = {**count, "precision": precision, "recall": recall}
        for key in ("tp", "fp", "fn"):
            totals[key] += count[key]

    confusions = []
    by_frame: dict[int, list[dict]] = defaultdict(list)
    for error in errors:
        by_frame[error["frame_id"]].append(error)
    for frame_id, frame_errors in by_frame.items():
        for fp in (e for e in frame_errors if e["kind"] == "false_positive"):
            matches = [
                (_iou(fp["xyxy"], fn["xyxy"]), fn)
                for fn in frame_errors
                if fn["kind"] == "false_negative" and fn["class_name"] != fp["class_name"]
            ]
            overlap, fn = max(matches, default=(0.0, None), key=lambda item: item[0])
            if overlap >= iou_threshold and fn is not None:
                confusions.append({
                    "frame_id": frame_id, "predicted": fp["class_name"],
                    "actual": fn["class_name"], "iou": overlap,
                })

    return {
        "dataset_id": manifest["dataset_id"],
        "reviewed_frames": len(reviewed),
        "total_frames": len(frame_list),
        "partial": len(reviewed) != len(frame_list),
        "iou_threshold": iou_threshold,
        "confidence_threshold": threshold,
        "totals": dict(totals),
        "classes": metrics,
        "errors": errors,
        "confusions": confusions,
    }
