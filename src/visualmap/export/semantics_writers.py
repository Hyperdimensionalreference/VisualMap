"""Export semantic objects (V0.5)."""

from __future__ import annotations

import json
from pathlib import Path

from visualmap.detection.types import SemanticResult
from visualmap.utils.logging import get_logger
from visualmap.utils.paths import run_directory

logger = get_logger(__name__)

__all__ = ["export_semantics"]


def export_semantics(
    result: SemanticResult,
    output_dir: str | Path,
    run_name: str = "run",
) -> dict[str, Path]:
    out = run_directory(output_dir, run_name)
    out.mkdir(parents=True, exist_ok=True)

    path = out / "semantics.json"
    payload = {
        "summary": result.summary(),
        "objects": [o.to_dict() for o in result.objects],
        "detections": [d.to_dict() for d in result.detections],
        "tracks": [t.to_dict() for t in result.tracks],
    }
    path.write_text(
        json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(",", ":")),
        encoding="utf-8",
    )
    logger.info("Wrote %s (%d objects)", path, len(result.objects))

    # Compact object list for quick inspection
    list_path = out / "semantic_objects.csv"
    with list_path.open("w", encoding="utf-8") as f:
        f.write("label,class_name,track_id,x,y,z,depth,confidence,observations\n")
        for o in result.objects:
            xyz = o.xyz
            x = f"{xyz[0]:.6f}" if xyz is not None else ""
            y = f"{xyz[1]:.6f}" if xyz is not None else ""
            z = f"{xyz[2]:.6f}" if xyz is not None else ""
            depth = f"{o.depth:.6f}" if o.depth is not None else ""
            f.write(
                f"{o.label},{o.class_name},{o.track_id},{x},{y},{z},{depth},"
                f"{o.confidence:.4f},{o.observations}\n"
            )
    return {"semantics_json": path, "semantic_csv": list_path}
