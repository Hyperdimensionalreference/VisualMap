"""Lossless geometry evidence for independent audits and semantic reprocessing."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from visualmap.slam.types import CameraIntrinsics, CameraPose, MapPoint, MapResult, Trajectory


def export_reconstruction(result: MapResult, path: str | Path) -> Path:
    """Keep landmark observations and pose quality that a PLY cannot store."""
    path = Path(path)
    payload = {
        "schema_version": 1,
        "pose_convention": "T_world<-camera",
        "units": "map_units",
        "intrinsics": result.intrinsics.to_dict() if result.intrinsics else None,
        "trajectory": result.trajectory.to_list(),
        "landmarks": [
            {"id": p.id, "xyz": p.xyz.tolist(), "observations": p.observations,
             "color": list(p.color) if p.color else None}
            for p in result.map_points
        ],
        "metadata": result.metadata,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    return path


def load_reconstruction(path: str | Path) -> MapResult:
    """Read versioned geometry without making up landmark observations."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("schema_version") != 1 or data.get("pose_convention") != "T_world<-camera":
        raise ValueError("unsupported reconstruction schema or pose convention")
    return MapResult(
        trajectory=Trajectory([CameraPose.from_dict(p) for p in data["trajectory"]]),
        map_points=[MapPoint(
            id=int(p["id"]), xyz=np.asarray(p["xyz"], dtype=float),
            observations={int(k): int(v) for k, v in p["observations"].items()},
            color=tuple(p["color"]) if p.get("color") else None,
        ) for p in data["landmarks"]],
        intrinsics=CameraIntrinsics.from_dict(data["intrinsics"]) if data["intrinsics"] else None,
        metadata=data["metadata"],
    )
