"""Audit exported recognition evidence without claiming ground-truth accuracy."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _iou(a: list[float], b: list[float]) -> float:
    intersection = max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(
        0.0, min(a[3], b[3]) - max(a[1], b[1])
    )
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def _overlaps(detections: list[dict], threshold: float = 0.6) -> list[dict]:
    by_frame: dict[int, list[dict]] = {}
    for detection in detections:
        by_frame.setdefault(detection["frame_id"], []).append(detection)
    overlaps = []
    for frame_id, predictions in sorted(by_frame.items()):
        for index, first in enumerate(predictions):
            for second in predictions[index + 1 :]:
                if first["class_name"] == second["class_name"]:
                    continue
                overlap = _iou(first["xyxy"], second["xyxy"])
                if overlap >= threshold:
                    overlaps.append(
                        {
                            "frame_id": frame_id,
                            "classes": [first["class_name"], second["class_name"]],
                            "confidence": [first["confidence"], second["confidence"]],
                            "iou": round(overlap, 4),
                        }
                    )
    return overlaps


def _position_consistency(run_dir: Path, objects: list[dict]) -> dict:
    """Check exported geometry against later boxes, without treating boxes as 3D truth."""
    trajectory_path = run_dir / "trajectory.csv"
    info_path = run_dir / "video_info.json"
    if not trajectory_path.is_file() or not info_path.is_file():
        return {"available": False}
    camera = _read_json(info_path).get("camera")
    if not camera:
        return {"available": False}
    poses = {}
    pose_updates = {}
    previous = None
    with trajectory_path.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            frame_id = int(float(row["frame_id"]))
            quaternion = np.array([float(row[key]) for key in ("qw", "qx", "qy", "qz")])
            w, x, y, z = quaternion / np.linalg.norm(quaternion)
            rotation = np.array([
                [1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)],
                [2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)],
                [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)],
            ])
            translation = np.array([float(row[key]) for key in ("tx", "ty", "tz")])
            matrix = np.eye(4)
            matrix[:3, :3], matrix[:3, 3] = rotation, translation
            poses[frame_id] = (rotation, translation)
            pose_updates[frame_id] = previous is not None and not np.allclose(matrix, previous)
            previous = matrix
    checks = []
    historical_update_checks = []
    for obj in objects:
        if obj.get("xyz") is None:
            continue
        historical_update_checks.extend(
            bool(pose_updates.get(entry["frame_id"], False))
            for entry in obj.get("position_history", [])
        )
        latest_frame = obj.get("frame_id", obj["last_frame"])
        if latest_frame not in poses:
            continue
        rotation, translation = poses[latest_frame]
        camera_xyz = rotation.T @ (np.array(obj["xyz"]) - translation)
        pixel = None
        compatible = False
        if camera_xyz[2] > 0:
            pixel = [
                float(camera["fx"] * camera_xyz[0] / camera_xyz[2] + camera["cx"]),
                float(camera["fy"] * camera_xyz[1] / camera_xyz[2] + camera["cy"]),
            ]
            box = obj["xyxy"]
            margin_x = 0.15 * (box[2] - box[0])
            margin_y = 0.15 * (box[3] - box[1])
            compatible = bool(
                box[0] - margin_x <= pixel[0] <= box[2] + margin_x
                and box[1] - margin_y <= pixel[1] <= box[3] + margin_y
            )
        checks.append({
            "label": obj["label"],
            "position_status": obj.get("position_status", "unspecified"),
            "position_frame": obj.get("position_frame"),
            "latest_frame": latest_frame,
            "projected_pixel": pixel,
            "compatible_with_latest_box_at_15_percent_margin": compatible,
            "latest_camera_pose_was_updated": bool(pose_updates.get(latest_frame, False)),
        })
    return {
        "available": True,
        "method": "world_centre_projection_into_latest_track_box_with_15_percent_margin",
        "validates_true_geometry_accuracy": False,
        "history_update_count": len(historical_update_checks),
        "history_updates_with_changed_camera_pose": sum(historical_update_checks),
        "latest_projection_outside_box_margin": sum(
            not check["compatible_with_latest_box_at_15_percent_margin"] for check in checks
        ),
        "objects": checks,
    }


def analyze_run(run_dir: Path, manifest_path: Path | None = None) -> dict:
    report = _read_json(run_dir / "run_report.json")
    semantics = _read_json(run_dir / "semantics.json")
    summary = report.get("summary", report)
    recognition = semantics.get("summary", {})
    objects = semantics.get("objects", [])
    detections = semantics.get("detections", [])
    tracks = semantics.get("tracks", [])
    overlaps = _overlaps(detections)
    display_overlaps = _overlaps(tracks)
    supported = [
        o for o in objects
        if o.get("xyz") is not None and o.get("depth_source") not in ("default", "none", None)
    ]
    retained = [
        o for o in supported
        if o.get("position_status") == "retained"
        or o.get("position_frame", o.get("frame_id")) != o.get("frame_id")
    ]
    audit = {
        "run_directory": str(run_dir.resolve()),
        "accuracy_status": "not_evaluated_against_ground_truth",
        "units": report.get("units"),
        "tracking": {
            key: summary.get(key)
            for key in ("frames_processed", "frames_tracked", "track_rate", "scale")
        },
        "recognition": {
            "objects": len(objects),
            "final_object_ids": len({o["track_id"] for o in objects}),
            "exported_observed_track_ids": len({t["track_id"] for t in tracks}),
            "exported_voted_class_changes": sum(
                t.get("raw_class_name", t["class_name"]) != t["class_name"] for t in tracks
            ),
            "by_class": dict(sorted(Counter(o["class_name"] for o in objects).items())),
            "reported_detections": recognition.get("num_detections"),
            "exported_detections": len(detections),
            "reported_track_observations": recognition.get("num_tracks"),
            "exported_track_observations": len(tracks),
            "detection_export_is_complete": len(detections)
            == recognition.get("num_detections"),
            "track_export_is_complete": len(tracks) == recognition.get("num_tracks"),
            "cross_class_overlap_count_in_export": len(overlaps),
            "cross_class_overlap_examples_in_export": overlaps[:30],
            "displayed_track_cross_class_overlap_count": len(display_overlaps),
            "displayed_track_cross_class_overlap_examples": display_overlaps[:30],
            "overlap_iou_threshold": 0.6,
            "overlap_candidates_are_not_ground_truth_errors": True,
        },
        "position_evidence": {
            "objects_with_xyz": sum(o.get("xyz") is not None for o in objects),
            "objects_with_depth_support": len(supported),
            "objects_with_direct_sparse_support": sum(
                o.get("xyz") is not None and o.get("position_source") == "sparse_observed"
                for o in objects
            ),
            "support_is_estimator_evidence_not_verified_spatial_accuracy": True,
            "retained_historical_positions": len(retained),
            "status_retained_positions": sum(
                o.get("position_status") == "retained" for o in objects
            ),
            "contradicted_positions": sum(
                o.get("position_status") in ("contradicted", "inconsistent")
                or o.get("position_temporal_consistency") == "contradicted"
                for o in objects
            ),
            "objects_eligible_for_current_spatial_relations": sum(
                o.get("position_temporal_consistency") == "consistent"
                and o.get("position_status") in ("estimated", "tentative", "retained")
                for o in supported
            ),
            "position_status_counts": dict(
                sorted(Counter(o.get("position_status", "unspecified") for o in objects).items())
            ),
            "depth_source_counts": dict(
                sorted(Counter(o.get("depth_source", "unspecified") for o in objects).items())
            ),
            "fallback_depth_objects": [
                o["label"] for o in objects if o.get("depth_source") == "default"
            ],
            "reprojection_consistency": _position_consistency(run_dir, objects),
        },
    }
    if manifest_path is not None:
        manifest = _read_json(manifest_path)
        frames = manifest.get("frames", [])
        reference_detections = [d for frame in frames for d in frame.get("predictions", [])]
        overlaps = _overlaps(reference_detections)
        audit["review_sample"] = {
            "prediction_source": "reference_manifest_snapshot",
            "dataset_id": manifest.get("dataset_id"),
            "frames": len(frames),
            "predictions_by_class": dict(
                sorted(
                    Counter(
                        detection["class_name"]
                        for frame in frames
                        for detection in frame.get("predictions", [])
                    ).items()
                )
            ),
            "cross_class_overlap_candidates": overlaps,
            "overlap_candidates_are_not_ground_truth_errors": True,
            "annotations_file_exists": (manifest_path.parent / "annotations.json").exists(),
        }
    return audit


def compare_runs(current: Path, baseline: Path, manifest: Path | None) -> dict:
    old = analyze_run(baseline)
    new = analyze_run(current)
    metrics = {
        "final_object_ids": ("recognition", "final_object_ids"),
        "reported_track_observations": ("recognition", "reported_track_observations"),
        "objects_with_xyz": ("position_evidence", "objects_with_xyz"),
        "objects_with_depth_support": ("position_evidence", "objects_with_depth_support"),
        "objects_with_direct_sparse_support": (
            "position_evidence", "objects_with_direct_sparse_support",
        ),
        "retained_historical_positions": ("position_evidence", "retained_historical_positions"),
        "contradicted_positions": ("position_evidence", "contradicted_positions"),
    }
    comparison = {}
    for name, (section, key) in metrics.items():
        before, after = old[section][key], new[section][key]
        comparison[name] = {
            "baseline": before,
            "current": after,
            "delta": after - before if before is not None and after is not None else None,
        }
    if manifest is not None:
        reference = _read_json(manifest)
        frame_ids = {frame["frame_id"] for frame in reference["frames"]}
        baseline_detections = [
            d for frame in reference["frames"] for d in frame.get("predictions", [])
        ]
        current_export = _read_json(current / "semantics.json")
        current_detections = [
            d for d in current_export.get("detections", []) if d["frame_id"] in frame_ids
        ]
        current_tracks = [
            t for t in current_export.get("tracks", []) if t["frame_id"] in frame_ids
        ]
        complete = new["recognition"]["detection_export_is_complete"]
        comparison["same_sample_cross_class_overlap"] = {
            "frames": len(frame_ids),
            "iou_threshold": 0.6,
            "baseline": len(_overlaps(baseline_detections)),
            "current": len(_overlaps(current_detections)) if complete else None,
            "current_displayed_track_conflicts": len(_overlaps(current_tracks))
            if new["recognition"]["track_export_is_complete"] else None,
            "displayed_baseline_unavailable_because_old_track_export_is_truncated":
                not old["recognition"]["track_export_is_complete"],
            "current_export_is_complete": complete,
            "baseline_source": "reference_manifest_snapshot",
            "current_source": "current_run_export_at_reference_frame_ids",
            "ground_truth_accuracy": "not_evaluated",
        }
    return {"baseline": old, "current": new, "comparison": comparison}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    audit = (
        compare_runs(args.run_dir, args.baseline, args.manifest)
        if args.baseline is not None else analyze_run(args.run_dir, args.manifest)
    )
    text = json.dumps(audit, ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
