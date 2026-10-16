"""VisualMap CLI entry point.

Orchestrates config → video → camera → SLAMBackend → export.
Does not implement CV math; never imports concrete backends directly.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from visualmap import __version__
from visualmap.camera import CameraModel
from visualmap.config import load_config
from visualmap.depth import create_depth_estimator
from visualmap.detection import (
    LiftConfig,
    SemanticPipeline,
    SemanticPipelineConfig,
    TrackerConfig,
    create_configured_detector,
)
from visualmap.export import (
    export_depth,
    export_map_ply,
    export_report,
    export_semantics,
    export_trajectory_csv,
    export_trajectory_txt,
)
from visualmap.export.reconstruction import export_reconstruction
from visualmap.mapping import build_semantic_map
from visualmap.runtime import data_root, is_frozen, resource_path_overrides, resource_root
from visualmap.slam.backends import create_backend
from visualmap.storage_manager import copy_media_atomic
from visualmap.utils.logging import get_logger, setup_logging
from visualmap.utils.paths import run_directory
from visualmap.video import VideoSource
from visualmap.video.source import VideoDecodeError
from visualmap.visualization import export_semantic_map_html

logger = get_logger("visualmap.app")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="visualmap",
        description="VisualMap — monocular video to camera trajectory + sparse 3D map",
    )
    parser.add_argument("--version", action="version", version=f"visualmap {__version__}")
    parser.add_argument("--config", type=Path, default=None, help="YAML config path")
    parser.add_argument("--data-dir", type=Path, default=None, help="workspace data directory")
    parser.add_argument("--resource-root", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument(
        "--video", type=Path, default=None, help="input video path (overrides config)"
    )
    parser.add_argument("--camera", type=Path, default=None, help="camera intrinsics YAML")
    parser.add_argument("--output", type=Path, default=None, help="output directory")
    parser.add_argument("--run-name", default=None, help="fixed run folder name (for regression)")
    parser.add_argument("--max-frames", type=int, default=None, help="limit number of frames read")
    parser.add_argument("--start-frame", type=int, default=None, help="first frame index")
    parser.add_argument(
        "--probe-only",
        action="store_true",
        help="only open video / print metadata (fast smoke test)",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="open Open3D window after run (requires visualmap[viz])",
    )
    parser.add_argument("--log-level", default=None, help="DEBUG/INFO/WARNING/ERROR")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    config_path = args.config
    resources = args.resource_root
    if is_frozen() and config_path is None:
        config_path = resource_root() / "configs/spatial_consistent.yaml"
        resources = resources or resource_root()
    overrides: dict = {}
    try:
        if resources is not None and config_path is not None:
            overrides.update(resource_path_overrides(config_path, resources.resolve()))
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if args.data_dir is not None or is_frozen():
        overrides["output.dir"] = str(data_root(args.data_dir) / "outputs")
    if args.video is not None:
        overrides["video.path"] = str(args.video)
    if args.camera is not None:
        overrides["camera.intrinsics_file"] = str(args.camera)
    if args.output is not None:
        overrides["output.dir"] = str(args.output)
    if args.run_name is not None:
        overrides["run_name"] = args.run_name
    if args.max_frames is not None:
        overrides["video.max_frames"] = args.max_frames
    if args.start_frame is not None:
        overrides["video.start_frame"] = args.start_frame
    if args.log_level is not None:
        overrides["log_level"] = args.log_level
    if args.visualize:
        overrides["output.visualize"] = True

    try:
        cfg = load_config(config_path, overrides=overrides or None)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    output_dir = cfg.output.dir
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(cfg.log_level, log_file=output_dir / "visualmap.log")

    logger.info("VisualMap %s starting", __version__)
    logger.info("Video: %s", cfg.video.path)

    try:
        source = VideoSource(
            cfg.video.path,
            start_frame=cfg.video.start_frame,
            max_frames=cfg.video.max_frames,
            every_n_frames=cfg.video.every_n_frames,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        logger.error("Cannot open video: %s", exc)
        return 1

    with source:
        info = source.info
        logger.info(
            "Video info: %dx%d @ %.3f fps, %d frames, %.2f s",
            info.width,
            info.height,
            info.fps,
            info.frame_count,
            info.duration_sec,
        )

        if cfg.camera.intrinsics_file is not None and not cfg.camera.intrinsics_file.is_file():
            logger.error("Camera intrinsics file not found: %s", cfg.camera.intrinsics_file)
            return 1

        intrinsics = cfg.camera.resolve(info.width, info.height)
        cam = CameraModel(intrinsics)
        if not cam.is_compatible_with(info.width, info.height):
            logger.warning(
                "Intrinsics size %sx%s != video %sx%s",
                intrinsics.width,
                intrinsics.height,
                info.width,
                info.height,
            )

        logger.info(
            "Camera: fx=%.2f fy=%.2f cx=%.2f cy=%.2f dist=%s",
            intrinsics.fx,
            intrinsics.fy,
            intrinsics.cx,
            intrinsics.cy,
            list(intrinsics.dist),
        )

        if args.probe_only:
            frame_count_read = 0
            try:
                for frame in source.frames():
                    frame_count_read += 1
                    logger.info(
                        "Probe frame %d shape=%s ts=%.3f",
                        frame.index,
                        frame.shape,
                        frame.timestamp,
                    )
                    break
            except VideoDecodeError as exc:
                logger.error("Video probe failed: %s", exc)
                return 1
            if frame_count_read == 0:
                logger.error("No frames decoded from video")
                return 1
            logger.info("Probe-only mode: skip SLAM and run artifacts")
            logger.info("Done")
            return 0

        # Prepare the run before decoding so a failure can leave an explicit
        # diagnostic, without exporting a viewer for an incomplete decode.
        run_name = cfg.run_name or datetime.now(timezone.utc).strftime("run_%Y%m%d_%H%M%S")
        run_dir = run_directory(output_dir, run_name)
        run_dir.mkdir(parents=True, exist_ok=True)

        try:
            backend = create_backend(cfg.slam.backend, **cfg.slam.vo)
        except (ValueError, NotImplementedError, TypeError) as exc:
            logger.error("Cannot create SLAM backend %r: %s", cfg.slam.backend, exc)
            return 1

        logger.info("SLAM backend: %s", getattr(backend, "name", cfg.slam.backend))
        backend.initialize(intrinsics)

        # Optional detector (V0.5) — same pass as SLAM, no second decode
        detector = None
        raw_detections = []
        if cfg.detection.enabled:
            try:
                detector = create_configured_detector(cfg.detection)
                detector.warmup()
                logger.info("Detector: %s", getattr(detector, "name", cfg.detection.detector))
            except Exception as exc:
                logger.error("Detector init failed: %s", exc)
                return 1

        frame_count_read = 0
        first_index = None
        last_index = None
        try:
            for frame in source.frames():
                frame_count_read += 1
                if first_index is None:
                    first_index = frame.index
                last_index = frame.index
                backend.process_frame(frame)
                if detector is not None:
                    try:
                        raw_detections.extend(detector.detect(frame))
                    except Exception as exc:
                        logger.error("Detection failed on frame %d: %s", frame.index, exc)
                        return 1
                if frame_count_read == 1 or frame_count_read % 30 == 0:
                    logger.info("Progress: frame %d / %d", frame.index + 1, info.frame_count)
        except VideoDecodeError as exc:
            logger.error("Video decode failed: %s", exc)
            result = backend.get_result()
            result.metadata.update(
                run_status="failed_video_decode",
                video=info.to_dict(),
                frames_read=frame_count_read,
                video_ingestion=source.read_diagnostics,
                error=str(exc),
            )
            backend.shutdown()
            # Failure diagnostics are retained even when successful reports are
            # disabled; no trajectory/map/semantic success artifacts follow.
            export_report(result, run_dir / "run_report.json", extra={"cli_version": __version__})
            return 1

        logger.info(
            "Processed %d frame(s): first=%s last=%s",
            frame_count_read,
            first_index,
            last_index,
        )

        if frame_count_read == 0:
            logger.error("No frames decoded from video")
            result = backend.get_result()
            result.metadata.update(
                run_status="failed_no_selected_frames",
                video=info.to_dict(),
                frames_read=0,
                video_ingestion=source.read_diagnostics,
                error="No frames decoded in the requested source range",
            )
            backend.shutdown()
            export_report(result, run_dir / "run_report.json")
            return 1

        result = backend.get_result()
        result.metadata["video"] = info.to_dict()
        result.metadata["frames_read"] = frame_count_read
        result.metadata["video_ingestion"] = source.read_diagnostics
        result.metadata["run_status"] = "completed"
        backend.shutdown()

        # V0.4 depth estimation (sparse-to-grid by default)
        depth_result = None
        depth_summary: dict = {"enabled": False}
        if cfg.depth.enabled:
            try:
                depth_est = create_depth_estimator(
                    cfg.depth.estimator,
                    downsample=cfg.depth.downsample,
                    max_frames=cfg.depth.max_frames,
                    fill_radius=cfg.depth.fill_radius,
                )
                depth_result = depth_est.estimate(result, intrinsics)
                depth_summary = depth_result.summary()
                result.metadata["depth"] = depth_summary
                logger.info(
                    "Depth maps: %d (mean valid %.1f%%)",
                    depth_summary.get("num_depth_maps", 0),
                    100.0 * float(depth_summary.get("mean_valid_ratio", 0.0)),
                )
            except Exception as exc:
                logger.warning("Depth estimation failed: %s", exc)
                depth_summary = {"enabled": True, "error": str(exc)}
                result.metadata["depth"] = depth_summary
        else:
            result.metadata["depth"] = depth_summary

        written: dict[str, Path] = {}
        if cfg.output.save_trajectory_csv:
            written["trajectory_csv"] = export_trajectory_csv(result, run_dir / "trajectory.csv")
        if cfg.output.save_trajectory_txt:
            written["trajectory_txt"] = export_trajectory_txt(result, run_dir / "trajectory.txt")
        if cfg.output.save_map_ply:
            written["map_ply"] = export_map_ply(result, run_dir / "map_points.ply")
        if cfg.output.save_report_json:
            written["report"] = export_report(result, run_dir / "run_report.json")
        (run_dir / "video_info.json").write_text(
            json.dumps({"video": info.to_dict(), "camera": intrinsics.to_dict()}, indent=2),
            encoding="utf-8",
        )
        if cfg.output.save_reconstruction:
            written["reconstruction"] = export_reconstruction(
                result, run_dir / "reconstruction.json"
            )

        if depth_result is not None and cfg.output.save_depth and len(depth_result) > 0:
            try:
                written.update(export_depth(depth_result, output_dir, run_name=run_name))
            except Exception as exc:
                logger.warning("Depth export failed: %s", exc)

        # V0.5 semantics: track + lift 2D detections to 3D world positions
        if detector is not None:
            try:
                sem = SemanticPipeline(
                    detector=detector,
                    config=SemanticPipelineConfig(
                        tracker=TrackerConfig(
                            **{"min_hits": cfg.detection.min_track_hits, **cfg.detection.tracker}
                        ),
                        lift=LiftConfig(**cfg.detection.lift),
                        max_objects=cfg.detection.max_objects,
                        fusion_window=cfg.detection.fusion_window,
                        max_position_jump_ratio=cfg.detection.max_position_jump_ratio,
                        max_position_history=cfg.detection.max_position_history,
                        reprojection_box_margin=cfg.detection.reprojection_box_margin,
                        min_multiview_observations=cfg.detection.min_multiview_observations,
                        min_multiview_parallax_deg=cfg.detection.min_multiview_parallax_deg,
                        max_multiview_reprojection_px=cfg.detection.max_multiview_reprojection_px,
                        max_center_surface_depth_ratio=cfg.detection.max_center_surface_depth_ratio,
                    ),
                )
                # Group detections by frame and run tracker sequentially
                by_frame: dict[int, list] = {}
                for d in raw_detections:
                    by_frame.setdefault(d.frame_id, []).append(d)
                for pose in result.trajectory:
                    sem.process_detections(
                        by_frame.get(pose.frame_id, []),
                        result,
                        depth_result=depth_result,
                        intrinsics=intrinsics,
                    )
                sem_result = sem.result()
                sem_result.metadata["detector_capabilities"] = getattr(detector, "capabilities", {})
                # Attach per-frame occupancy before persisting semantic objects.
                smap = build_semantic_map(
                    result,
                    sem_result,
                    occupancy_thickness_ratio=cfg.detection.occupancy_thickness_ratio,
                    occupancy_multiview_options={
                        "multiview_enabled": cfg.detection.occupancy_multiview_enabled,
                        "multiview_max_views": cfg.detection.occupancy_multiview_max_views,
                        "multiview_min_angle_deg": cfg.detection.occupancy_multiview_min_angle_deg,
                        "multiview_bbox_margin": cfg.detection.occupancy_multiview_bbox_margin,
                    },
                )
                result.metadata["semantics"] = sem_result.summary()
                written.update(export_semantics(sem_result, output_dir, run_name=run_name))

                # V0.6 semantic map viewer (HTML)
                video_filename = None
                if cfg.output.copy_source_video:
                    suffix = cfg.video.path.suffix.lower()
                    if suffix in (".mp4", ".webm", ".ogg"):
                        video_filename = f"source_video{suffix}"
                        destination = run_dir / video_filename
                        try:
                            if cfg.video.path.resolve() != destination.resolve():
                                copy_media_atomic(cfg.video.path, destination)
                            written["source_video"] = destination
                        except OSError as exc:
                            logger.warning("Video copy failed: %s", exc)
                            video_filename = None
                    else:
                        logger.warning("Video format %s is not browser playback ready", suffix)
                html_path = export_semantic_map_html(
                    smap,
                    output_dir,
                    run_name=run_name,
                    title=f"VisualMap — {run_name}",
                    video_filename=video_filename,
                )
                written["semantic_map_html"] = html_path
                written["occupancy_json"] = run_dir / "occupancy.json"
                result.metadata["semantic_map"] = smap.summary()
                # Refresh report so semantics/depth land in run_report.json
                if cfg.output.save_report_json:
                    export_report(
                        result,
                        run_dir / "run_report.json",
                        extra={"cli_version": __version__},
                    )

                logger.info(
                    "Semantic objects: %d (detections=%d)",
                    len(sem_result),
                    len(sem_result.detections),
                )
                for obj in sem_result.objects[:8]:
                    xyz = obj.xyz
                    pos = (
                        f"({xyz[0]:.2f}, {xyz[1]:.2f}, {xyz[2]:.2f})" if xyz is not None else "n/a"
                    )
                    logger.info(
                        "  %s conf=%.2f depth=%s xyz=%s",
                        obj.label,
                        obj.confidence,
                        f"{obj.depth:.2f}" if obj.depth is not None else "n/a",
                        pos,
                    )
            except Exception as exc:
                logger.error("Semantic pipeline failed: %s", exc)
                result.metadata["semantics"] = {"error": str(exc)}
                if cfg.output.save_report_json:
                    export_report(result, run_dir / "run_report.json")
                return 1

        logger.info("Trajectory poses: %d", len(result.trajectory))
        logger.info("Map points: %d", len(result.map_points))
        logger.info("Keyframes: %d", result.metadata.get("num_keyframes", 0))
        logger.info("Track rate: %.1f%%", 100.0 * float(result.metadata.get("track_rate", 0.0)))
        logger.info("Results written under %s", run_dir)
        for label, path in written.items():
            logger.info("  %s: %s", label, path)

        if cfg.output.visualize:
            try:
                from visualmap.visualization import open3d_available, show_map

                if open3d_available():
                    logger.info("Opening Open3D visualization...")
                    show_map(result, window_name=f"VisualMap — {run_name}")
                else:
                    logger.warning("Open3D not installed; skip visualization")
            except Exception as exc:  # visualization must not break exports
                logger.warning("Visualization failed: %s", exc)

    logger.info("Done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
