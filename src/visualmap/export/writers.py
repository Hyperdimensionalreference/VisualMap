"""Result export: trajectory / map / report files.

File formats are frozen in docs/adr/0004-output-format.md and docs/data_format.md.
Later stages (YOLO, depth, semantic map) must read these files — never recompute.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from visualmap.slam.coverage import landmark_world
from visualmap.slam.types import MapResult
from visualmap.utils.logging import get_logger
from visualmap.utils.paths import run_directory

logger = get_logger(__name__)

__all__ = [
    "export_all",
    "export_trajectory_csv",
    "export_trajectory_txt",
    "export_map_ply",
    "export_report",
]


def export_trajectory_csv(result: MapResult, path: str | Path) -> Path:
    """CSV: frame_id,timestamp,tx,ty,tz,qw,qx,qy,qz"""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        segmented = len(result.metadata.get("submaps", [])) > 1
        header = ["frame_id", "timestamp", "tx", "ty", "tz", "qw", "qx", "qy", "qz"]
        writer.writerow(header + (["world_frame_id", "pose_valid"] if segmented else []))
        quality = result.metadata.get("frame_quality", {})
        for pose in result.trajectory:
            t = pose.t
            q = _rot_to_quat(pose.R)
            row = [pose.frame_id, f"{pose.timestamp:.6f}", *t, *q]
            if segmented:
                evidence = quality.get(str(pose.frame_id), quality.get(pose.frame_id, {}))
                row += [evidence.get("world_frame_id"), evidence.get("pose_valid") is True]
            writer.writerow(row)
    logger.info("Wrote %s (%d poses)", p, len(result.trajectory))
    return p


def export_trajectory_txt(result: MapResult, path: str | Path) -> Path:
    """TUM-style: timestamp tx ty tz qx qy qz qw  (one pose per line)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    submaps = result.metadata.get("submaps", [])
    if len(submaps) > 1:
        quality = result.metadata.get("frame_quality", {})
        from visualmap.slam.types import Trajectory

        names = []
        for index, submap in enumerate(submaps):
            world = submap["world_frame_id"]
            poses = [
                pose for pose in result.trajectory
                if quality.get(str(pose.frame_id), {}).get("world_frame_id") == world
                and quality.get(str(pose.frame_id), {}).get("pose_valid") is True
            ]
            name = f"{p.stem}_submap_{index + 1}{p.suffix}"
            export_trajectory_txt(MapResult(Trajectory(poses)), p.with_name(name))
            names.append({"file": name, "world_frame_id": world})
        p.write_text(
            "# Independent submaps; this file is not a single TUM trajectory.\n"
            + "# Use trajectory.csv world_frame_id / pose_valid or these local files:\n"
            + "".join(f"# {row['file']} {row['world_frame_id']}\n" for row in names),
            encoding="utf-8",
        )
        return p
    with p.open("w", encoding="utf-8") as f:
        for pose in result.trajectory:
            t = pose.t
            q = _rot_to_quat(pose.R)  # w,x,y,z
            # TUM writes qx qy qz qw
            f.write(
                f"{pose.timestamp:.6f} "
                f"{t[0]:.8f} {t[1]:.8f} {t[2]:.8f} "
                f"{q[1]:.8f} {q[2]:.8f} {q[3]:.8f} {q[0]:.8f}\n"
            )
    logger.info("Wrote %s (%d poses)", p, len(result.trajectory))
    return p


def export_map_ply(result: MapResult, path: str | Path) -> Path:
    """ASCII PLY point cloud of map points."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    submaps = result.metadata.get("submaps", [])
    if len(submaps) > 1:
        from visualmap.slam.types import Trajectory

        manifest = {"coordinate_scope": "independent_unregistered_submaps", "maps": []}
        for index, submap in enumerate(submaps):
            world = submap["world_frame_id"]
            subset = [
                point for point in result.map_points if landmark_world(result, point) == world
            ]
            destination = p if index == 0 else p.with_name(f"{p.stem}_submap_{index + 1}{p.suffix}")
            export_map_ply(MapResult(Trajectory(), subset), destination)
            manifest["maps"].append(
                {"file": destination.name, "world_frame_id": world, "points": len(subset)}
            )
        p.with_suffix(".coordinate_frames.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
        return p
    points = [mp.xyz for mp in result.map_points]
    colors = [mp.color for mp in result.map_points]
    n = len(points)

    with p.open("w", encoding="utf-8") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {n}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for i in range(n):
            x, y, z = points[i]
            if colors[i] is not None:
                r, g, b = colors[i]
            else:
                r, g, b = 200, 200, 200
            f.write(f"{x:.6f} {y:.6f} {z:.6f} {int(r)} {int(g)} {int(b)}\n")
    logger.info("Wrote %s (%d points)", p, n)
    return p


def export_report(result: MapResult, path: str | Path, extra: dict | None = None) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "tool": "visualmap",
        "version": "0.1.0",
        "pose_convention": "T_world<-camera",
        "units": "map_units (monocular scale ambiguous)",
        "summary": result.summary(),
        **(extra or {}),
        **result.metadata,
    }
    p.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Wrote %s", p)
    return p


def export_all(result: MapResult, output_dir: str | Path, run_name: str = "run") -> dict[str, Path]:
    out = run_directory(output_dir, run_name)
    out.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    written["trajectory_csv"] = export_trajectory_csv(result, out / "trajectory.csv")
    written["trajectory_txt"] = export_trajectory_txt(result, out / "trajectory.txt")
    written["map_ply"] = export_map_ply(result, out / "map_points.ply")
    written["report"] = export_report(result, out / "run_report.json")
    return written


def _rot_to_quat(R: np.ndarray) -> np.ndarray:
    """Rotation matrix -> quaternion (w, x, y, z)."""
    R = np.asarray(R, dtype=np.float64)
    m00, m01, m02 = R[0]
    m10, m11, m12 = R[1]
    m20, m21, m22 = R[2]
    trace = m00 + m11 + m22
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (m21 - m12) * s
        y = (m02 - m20) * s
        z = (m10 - m01) * s
    elif m00 > m11 and m00 > m22:
        s = 2.0 * np.sqrt(1.0 + m00 - m11 - m22)
        w = (m21 - m12) / s
        x = 0.25 * s
        y = (m01 + m10) / s
        z = (m02 + m20) / s
    elif m11 > m22:
        s = 2.0 * np.sqrt(1.0 + m11 - m00 - m22)
        w = (m02 - m20) / s
        x = (m01 + m10) / s
        y = 0.25 * s
        z = (m12 + m21) / s
    else:
        s = 2.0 * np.sqrt(1.0 + m22 - m00 - m11)
        w = (m10 - m01) / s
        x = (m02 + m20) / s
        y = (m12 + m21) / s
        z = 0.25 * s
    q = np.array([w, x, y, z], dtype=np.float64)
    return q / np.linalg.norm(q)
