"""Refresh a saved run's presentation while preserving recognition and geometry."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from visualmap.detection.types import SemanticObject, SemanticResult
from visualmap.export.reconstruction import load_reconstruction
from visualmap.mapping import build_semantic_map
from visualmap.visualization import export_semantic_map_html


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--thickness-ratio", type=float, help="未观测厚度与投影短边的估计比例")
    parser.add_argument(
        "--assets-only",
        action="store_true",
        help="只更新页面资源，保留存档的体积与几何，不重新估算",
    )
    args = parser.parse_args()
    if args.assets_only and args.thickness_ratio is not None:
        parser.error("--assets-only cannot change thickness assumptions")
    run = args.run.resolve()
    saved = json.loads((run / "semantics.json").read_text(encoding="utf-8"))
    objects = []
    core = {
        "track_id",
        "class_id",
        "class_name",
        "xyz",
        "depth",
        "first_frame",
        "last_frame",
        "observations",
        "confidence",
        "xyxy",
        "frame_id",
        "label",
    }
    for row in saved["objects"]:
        fields = {key: row[key] for key in core - {"label"}}
        fields["xyz"] = np.array(row["xyz"]) if row["xyz"] is not None else None
        fields["xyxy"] = tuple(row["xyxy"])
        fields["metadata"] = {key: value for key, value in row.items() if key not in core}
        objects.append(SemanticObject(**fields))
    semantics = SemanticResult(objects=objects, metadata=saved["summary"])
    reconstruction = load_reconstruction(run / "reconstruction.json")
    smap = build_semantic_map(
        reconstruction,
        semantics,
        observation_records=saved.get("tracks", []),
        occupancy_thickness_ratio=args.thickness_ratio,
        preserve_saved_occupancy=args.assets_only,
    )
    videos = list(run.glob("source_video.*"))
    result = export_semantic_map_html(
        smap,
        run.parent,
        run.name,
        title="VisualMap · 空间关系",
        video_filename=videos[0].name if videos else None,
    )
    (run / "semantic_map.json").write_text(
        json.dumps(smap.to_dict(), ensure_ascii=False, separators=(",", ":"), allow_nan=False),
        encoding="utf-8",
    )
    print(result)


if __name__ == "__main__":
    main()
