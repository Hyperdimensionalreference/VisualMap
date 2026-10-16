"""Regenerate semantic_map.html from an existing run directory."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from visualmap.detection.types import Detection, SemanticObject, SemanticResult, TrackedDetection
from visualmap.export.reconstruction import load_reconstruction
from visualmap.mapping import build_semantic_map
from visualmap.slam.types import CameraIntrinsics, CameraPose, MapPoint, MapResult, Trajectory
from visualmap.storage_manager import copy_media_atomic
from visualmap.visualization import export_semantic_map_html


def load_run(run_dir: Path) -> MapResult:
    reconstruction = run_dir / "reconstruction.json"
    if reconstruction.is_file():
        return load_reconstruction(reconstruction)
    traj = Trajectory()
    csv_path = run_dir / "trajectory.csv"
    with csv_path.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            quat = np.array([float(row[k]) for k in ("qw", "qx", "qy", "qz")])
            norm = np.linalg.norm(quat)
            if not np.isfinite(quat).all() or norm < 1e-10:
                raise ValueError("invalid trajectory quaternion")
            w, x, y, z = quat / norm
            rotation = np.array([
                [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)],
            ])
            t = [float(row[k]) for k in ("tx", "ty", "tz")]
            traj.append(CameraPose.from_rt(
                rotation, t, frame_id=int(float(row["frame_id"])), timestamp=float(row["timestamp"])
            ))

    points: list[MapPoint] = []
    ply = run_dir / "map_points.ply"
    if ply.is_file():
        header_done = False
        pid = 0
        for raw in ply.read_text(encoding="utf-8").splitlines():
            if not header_done:
                if raw.strip() == "end_header":
                    header_done = True
                continue
            bits = raw.split()
            if len(bits) >= 3:
                xyz = [float(bits[0]), float(bits[1]), float(bits[2])]
                points.append(MapPoint(id=pid, xyz=xyz))
                pid += 1

    intrinsics = None
    info_path = run_dir / "video_info.json"
    if info_path.is_file():
        info = json.loads(info_path.read_text(encoding="utf-8"))
        if info.get("camera"):
            intrinsics = CameraIntrinsics.from_dict(info["camera"])
    report_path = run_dir / "run_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.is_file() else {}
    metadata = report.get("summary", report)
    if len(metadata.get("submaps", [])) > 1:
        raise ValueError("independent submaps require reconstruction.json to preserve all worlds")
    return MapResult(trajectory=traj, map_points=points, intrinsics=intrinsics, metadata=metadata)


def load_semantics(run_dir: Path) -> SemanticResult:
    path = run_dir / "semantics.json"
    if not path.is_file():
        return SemanticResult()
    data = json.loads(path.read_text(encoding="utf-8"))
    objects = []
    for o in data.get("objects", []):
        xyz = o.get("xyz")
        objects.append(
            SemanticObject(
                track_id=int(o.get("track_id", -1)),
                class_id=int(o.get("class_id", 0)),
                class_name=str(o.get("class_name", "object")),
                xyz=np.asarray(xyz, dtype=np.float64) if xyz else None,
                depth=o.get("depth"),
                first_frame=int(o.get("first_frame", 0)),
                last_frame=int(o.get("last_frame", 0)),
                observations=int(o.get("observations", 1)),
                confidence=float(o.get("confidence", 0.0)),
                xyxy=tuple(o.get("xyxy", (0, 0, 0, 0))),
                frame_id=int(o.get("frame_id", 0)),
                metadata={k: v for k, v in o.items() if k not in {
                    "track_id", "class_id", "class_name", "xyz", "depth", "first_frame",
                    "last_frame", "observations", "confidence", "xyxy", "frame_id", "label",
                }},
            )
        )
    summary = data.get("summary", {})
    tracks = []
    for row in data.get("tracks", []):
        detection = Detection(
            frame_id=int(row["frame_id"]), timestamp=float(row["timestamp"]),
            class_id=int(row["class_id"]), class_name=str(row["class_name"]),
            confidence=float(row["confidence"]), xyxy=tuple(row["xyxy"]),
        )
        tracks.append(TrackedDetection(
            track_id=int(row["track_id"]), detection=detection, hits=int(row.get("hits", 1)),
            class_consistency=float(row.get("class_consistency", 1)),
            class_observations=int(row.get("class_observations", 1)),
        ))
    return SemanticResult(
        objects=objects,
        tracks=tracks,
        metadata={"source": "semantics.json", **summary},
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--title", default=None)
    parser.add_argument("--detector", default=None, help="detector used by older runs")
    parser.add_argument("--video", type=Path, default=None, help="copy source video for playback")
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    map_result = load_run(run_dir)
    sem = load_semantics(run_dir)
    if args.detector:
        sem.metadata["detector"] = args.detector
    smap = build_semantic_map(map_result, sem)
    title = args.title or f"VisualMap 3D · {run_dir.name}"
    video_filename = next((p.name for p in run_dir.glob("source_video.*")
                           if p.suffix.lower() in (".mp4", ".webm", ".ogg")), None)
    if args.video is not None:
        video_path = args.video.resolve()
        if not video_path.is_file():
            parser.error(f"video not found: {video_path}")
        suffix = video_path.suffix.lower()
        if suffix not in (".mp4", ".webm", ".ogg"):
            parser.error(f"video format is not browser playback ready: {suffix}")
        video_filename = f"source_video{suffix}"
        destination = run_dir / video_filename
        if video_path != destination.resolve():
            copy_media_atomic(video_path, destination)
    path = export_semantic_map_html(
        smap,
        run_dir.parent,
        run_name=run_dir.name,
        title=title,
        video_filename=video_filename,
    )
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
