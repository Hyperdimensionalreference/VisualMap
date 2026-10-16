"""Reprocess the same recorded detections against a new, complete reconstruction."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from visualmap.config import load_config
from visualmap.detection import LiftConfig, SemanticPipeline, SemanticPipelineConfig, TrackerConfig
from visualmap.detection.types import Detection
from visualmap.export import export_report, export_semantics
from visualmap.export.reconstruction import load_reconstruction
from visualmap.mapping import build_semantic_map
from visualmap.storage_manager import copy_media_atomic
from visualmap.video.source import VideoSource
from visualmap.visualization import export_semantic_map_html


def main() -> None:
    parser = argparse.ArgumentParser(description="用同一批检测框验证新空间重建")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--geometry-run", type=Path, required=True)
    parser.add_argument("--recognition-run", type=Path, required=True)
    parser.add_argument("--output-run", type=Path, required=True)
    parser.add_argument(
        "--refine-pegboards",
        action="store_true",
        help="重新读取原视频，用孔阵列结构规则生成分类候选，保留原模型证据；不重跑相机建图",
    )
    args = parser.parse_args()
    cfg = load_config(args.config)
    result = load_reconstruction(args.geometry_run / "reconstruction.json")
    source = json.loads((args.recognition_run / "semantics.json").read_text(encoding="utf-8"))
    info = json.loads((args.recognition_run / "video_info.json").read_text(encoding="utf-8"))
    video = result.metadata.get("video") or {}
    if Path(video.get("path", "")).resolve() != Path(info["video"]["path"]).resolve() or (
        video.get("width"),
        video.get("height"),
    ) != (info["video"]["width"], info["video"]["height"]):
        parser.error("geometry and recorded detections must come from the same source video")
    detections = source.get("detections", [])
    if len(detections) != source.get("summary", {}).get("num_detections"):
        parser.error("recorded detections were truncated; a complete recognition run is required")
    options = cfg.detection
    pipeline = SemanticPipeline(
        config=SemanticPipelineConfig(
            tracker=TrackerConfig(**{"min_hits": options.min_track_hits, **options.tracker}),
            lift=LiftConfig(**options.lift),
            max_objects=options.max_objects,
            fusion_window=options.fusion_window,
            max_position_jump_ratio=options.max_position_jump_ratio,
            max_position_history=options.max_position_history,
            reprojection_box_margin=options.reprojection_box_margin,
            min_multiview_observations=options.min_multiview_observations,
            min_multiview_parallax_deg=options.min_multiview_parallax_deg,
            max_multiview_reprojection_px=options.max_multiview_reprojection_px,
            max_center_surface_depth_ratio=options.max_center_surface_depth_ratio,
        )
    )
    by_frame: dict[int, list[Detection]] = {}
    for row in detections:
        detection = Detection(
            frame_id=int(row["frame_id"]),
            timestamp=float(row["timestamp"]),
            class_id=int(row["class_id"]),
            class_name=row["class_name"],
            confidence=float(row["confidence"]),
            xyxy=tuple(row["xyxy"]),
            metadata=dict(row.get("detection_evidence") or {}),
        )
        by_frame.setdefault(detection.frame_id, []).append(detection)
    if args.refine_pegboards:
        from visualmap.detection.pegboard import PegboardConfig, PegboardRefiner

        source_video_path = Path(info["video"]["path"])
        if not source_video_path.is_file():
            parser.error("pixel-structure refinement requires the original source video")
        refiner = PegboardRefiner(PegboardConfig(**options.pegboard_config))
        wanted_frames = {pose.frame_id for pose in result.trajectory}
        refined_frames = set()
        with VideoSource(source_video_path) as source_video:
            for frame in source_video.frames():
                if frame.index in wanted_frames:
                    revised = refiner.refine(frame, by_frame.get(frame.index, []))
                    pipeline.process_detections(revised, result)
                    refined_frames.add(frame.index)
                    if frame.index % 120 == 0:
                        print(
                            f"frame {frame.index}: original pixels checked; camera geometry reused",
                            flush=True,
                        )
        if refined_frames != wanted_frames:
            parser.error("source video did not provide every reconstruction frame")
    else:
        for pose in result.trajectory:
            pipeline.process_detections(by_frame.get(pose.frame_id, []), result)
    semantics = pipeline.result()
    semantics.metadata.update(
        {
            "detector": source["summary"].get("detector", "unknown"),
            "detector_capabilities": source["summary"].get(
                "detector_capabilities", source["summary"].get("detection_capabilities", {})
            ),
            "supplement_inference_frames": source["summary"].get("supplement_inference_frames", []),
            "supplement_every_n_frames": source["summary"].get("supplement_every_n_frames"),
            "detections_reused": not args.refine_pegboards,
            "raw_model_predictions_reused": True,
            "pixel_structure_refinement": args.refine_pegboards,
            "classification_refinement_source": "image_structure_rule"
            if args.refine_pegboards
            else None,
            "trained_new_model": False,
            "camera_geometry_reused": True,
            "recognition_source": str(args.recognition_run),
        }
    )
    result.metadata["semantics"] = semantics.summary()
    output = args.output_run.resolve()
    output.mkdir(parents=True, exist_ok=True)
    for name in (
        "trajectory.csv",
        "trajectory.txt",
        "map_points.ply",
        "reconstruction.json",
        "video_info.json",
    ):
        src = args.geometry_run / name
        if src.is_file() and src.resolve() != (output / name).resolve():
            shutil.copy2(src, output / name)
    # A segmented reconstruction has separate PLY/TUM files. Copy their
    # coordinate manifest too; never silently turn it into one global map.
    for pattern in (
        "map_points_submap_*.ply",
        "map_points.coordinate_frames.json",
        "trajectory_submap_*.txt",
    ):
        for src in args.geometry_run.glob(pattern):
            if src.resolve() != (output / src.name).resolve():
                shutil.copy2(src, output / src.name)
    video_name = None
    for source_dir in (args.recognition_run, args.geometry_run):
        candidates = list(source_dir.glob("source_video.*"))
        if candidates:
            source_video = candidates[0]
            video_name = source_video.name
            if source_video.resolve() != (output / video_name).resolve():
                copy_media_atomic(source_video, output / video_name)
            break
    smap = build_semantic_map(
        result,
        semantics,
        occupancy_thickness_ratio=cfg.detection.occupancy_thickness_ratio,
        occupancy_multiview_options={
            "multiview_enabled": cfg.detection.occupancy_multiview_enabled,
            "multiview_max_views": cfg.detection.occupancy_multiview_max_views,
            "multiview_min_angle_deg": cfg.detection.occupancy_multiview_min_angle_deg,
            "multiview_bbox_margin": cfg.detection.occupancy_multiview_bbox_margin,
        },
    )
    result.metadata["semantics"] = semantics.summary()
    export_semantics(semantics, output.parent, output.name)
    export_report(result, output / "run_report.json")
    path = export_semantic_map_html(
        smap, output.parent, output.name, title="VisualMap · 空间关系", video_filename=video_name
    )
    print(json.dumps({"viewer": str(path), "semantics": semantics.summary()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
