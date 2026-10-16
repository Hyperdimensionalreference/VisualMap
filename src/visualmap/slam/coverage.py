"""Summarize observed pose coverage without equating it with video ingestion."""

from __future__ import annotations

from visualmap.slam.types import MapResult


def geometry_coverage(result: MapResult) -> dict:
    quality = result.metadata.get("frame_quality")
    if not isinstance(quality, dict):
        return {"status": "unverified", "total_frames": len(result.trajectory)}
    segments = []
    valid_frames = []
    for pose in result.trajectory:
        row = quality.get(str(pose.frame_id), quality.get(pose.frame_id, {}))
        valid = row.get("pose_valid") is True
        world = row.get("world_frame_id")
        if valid:
            valid_frames.append(pose.frame_id)
        if not segments or (segments[-1]["pose_valid"], segments[-1]["world_frame_id"]) != (
            valid, world
        ):
            segments.append(
                {"first_frame": pose.frame_id, "last_frame": pose.frame_id,
                 "start_seconds": pose.timestamp, "end_seconds": pose.timestamp,
                 "pose_valid": valid, "world_frame_id": world, "frames": 0}
            )
        segments[-1].update(last_frame=pose.frame_id, end_seconds=pose.timestamp)
        segments[-1]["frames"] += 1
    total = len(result.trajectory)
    return {
        "status": "complete" if len(valid_frames) == total else "partial",
        "total_frames": total,
        "valid_pose_frames": len(valid_frames),
        "invalid_pose_frames": total - len(valid_frames),
        "valid_pose_fraction": len(valid_frames) / max(1, total),
        "last_valid_frame": valid_frames[-1] if valid_frames else None,
        "segments": segments,
        "accuracy_validated": False,
    }


def landmark_world(result: MapResult, point) -> str | None:
    saved = result.metadata.get("landmark_world_frame_ids", {})
    if str(point.id) in saved:
        return saved[str(point.id)]
    quality = result.metadata.get("frame_quality", {})
    worlds = {
        quality.get(str(frame), quality.get(frame, {})).get("world_frame_id")
        for frame in point.observations
    } - {None}
    if len(worlds) == 1:
        return next(iter(worlds))
    if worlds or len(result.metadata.get("submaps", [])) > 1:
        return None
    return result.metadata.get("world_frame_id")
