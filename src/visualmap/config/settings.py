"""Typed configuration loaded from YAML.

All tunable parameters live in config files. Code must not hardcode magic numbers
that a user might reasonably change (thresholds, paths, limits).
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from visualmap.camera import load_intrinsics
from visualmap.slam.types import CameraIntrinsics
from visualmap.utils.paths import validate_run_name

__all__ = ["AppConfig", "VideoConfig", "CameraConfig", "OutputConfig", "SLAMConfig", "load_config"]


def _expand_path(path: str | Path) -> Path:
    return Path(os.path.expandvars(str(path))).expanduser().resolve()


@dataclass
class VideoConfig:
    path: Path
    max_frames: int | None = None
    start_frame: int = 0
    every_n_frames: int = 1

    def __post_init__(self) -> None:
        self.path = _expand_path(self.path)
        if self.start_frame < 0:
            raise ValueError("start_frame must be >= 0")
        if self.every_n_frames < 1:
            raise ValueError("every_n_frames must be >= 1")
        if self.max_frames is not None and self.max_frames < 1:
            raise ValueError("max_frames must be >= 1 when set")


@dataclass
class CameraConfig:
    # Either load from a YAML file path, or inline intrinsics
    intrinsics_file: Path | None = None
    inline: CameraIntrinsics | None = None
    # Fallback when neither is set: assume square pixels + principal point at center
    default_focal_ratio: float = 1.2  # fx = fy = ratio * max(width, height)

    def resolve(self, video_width: int, video_height: int) -> CameraIntrinsics:
        if self.inline is not None:
            return self.inline
        if self.intrinsics_file is not None:
            return load_intrinsics(self.intrinsics_file)
        f = self.default_focal_ratio * max(video_width, video_height)
        return CameraIntrinsics(
            fx=f,
            fy=f,
            cx=video_width / 2.0,
            cy=video_height / 2.0,
            width=video_width,
            height=video_height,
        )


@dataclass
class SLAMConfig:
    backend: str = "opencv_vo"
    vo: dict[str, Any] = field(default_factory=dict)


@dataclass
class DetectionConfig:
    enabled: bool = True
    detector: str = "demo"
    model_path: str | None = None
    conf_threshold: float = 0.35
    nms_threshold: float = 0.45
    max_objects: int = 200
    min_track_hits: int = 1
    input_width: int = 640
    input_height: int = 640
    class_names: tuple[str, ...] | None = None
    pegboard_refinement: bool = False
    pegboard_config: dict[str, Any] = field(default_factory=dict)
    tiling: dict[str, Any] = field(default_factory=dict)
    open_vocabulary: dict[str, Any] = field(default_factory=dict)
    tracker: dict[str, Any] = field(default_factory=dict)
    lift: dict[str, Any] = field(default_factory=dict)
    fusion_window: int = 12
    max_position_jump_ratio: float = 0.5
    max_position_history: int = 500
    reprojection_box_margin: float = 0.15
    min_multiview_observations: int = 3
    min_multiview_parallax_deg: float = 1.5
    max_multiview_reprojection_px: float = 12.0
    max_center_surface_depth_ratio: float = 0.35
    occupancy_thickness_ratio: float = 0.5
    occupancy_multiview_enabled: bool = True
    occupancy_multiview_max_views: int = 6
    occupancy_multiview_min_angle_deg: float = 2.0
    occupancy_multiview_bbox_margin: float = 0.08

    def __post_init__(self) -> None:
        if self.input_width < 1 or self.input_height < 1:
            raise ValueError("detection input dimensions must be positive")
        if self.class_names is not None and not self.class_names:
            raise ValueError("detection.class_names must not be empty")
        if self.min_track_hits < 1:
            raise ValueError("detection.min_track_hits must be positive")
        if self.fusion_window < 1 or self.max_position_history < 2:
            raise ValueError("fusion_window must be positive and max_position_history >= 2")
        if not 0 < self.max_position_jump_ratio <= 1:
            raise ValueError("max_position_jump_ratio must be in (0, 1]")
        if not 0 <= self.reprojection_box_margin <= 1:
            raise ValueError("reprojection_box_margin must be in [0, 1]")
        if not 0 < self.occupancy_thickness_ratio <= 4:
            raise ValueError("occupancy_thickness_ratio must be finite and in (0, 4]")
        if not isinstance(self.occupancy_multiview_enabled, bool):
            raise ValueError("occupancy_multiview_enabled must be a boolean")
        if (
            isinstance(self.occupancy_multiview_max_views, bool)
            or not isinstance(self.occupancy_multiview_max_views, int)
            or not 2 <= self.occupancy_multiview_max_views <= 16
        ):
            raise ValueError("occupancy_multiview_max_views must be an integer in [2, 16]")
        if (
            not math.isfinite(self.occupancy_multiview_min_angle_deg)
            or not 0 < self.occupancy_multiview_min_angle_deg < 90
        ):
            raise ValueError("occupancy_multiview_min_angle_deg must be finite and in (0, 90)")
        if (
            not math.isfinite(self.occupancy_multiview_bbox_margin)
            or not 0 <= self.occupancy_multiview_bbox_margin <= 0.5
        ):
            raise ValueError("occupancy_multiview_bbox_margin must be finite and in [0, 0.5]")


@dataclass
class DepthConfig:
    enabled: bool = True
    estimator: str = "sparse_depth"
    downsample: int = 2
    max_frames: int = 12
    fill_radius: int = 8


@dataclass
class OutputConfig:
    dir: Path = field(default_factory=lambda: Path("outputs"))
    save_trajectory_csv: bool = True
    save_trajectory_txt: bool = True
    save_map_ply: bool = True
    save_report_json: bool = True
    save_depth: bool = True
    visualize: bool = False
    copy_source_video: bool = False
    save_reconstruction: bool = True

    def __post_init__(self) -> None:
        self.dir = _expand_path(self.dir)


@dataclass
class AppConfig:
    video: VideoConfig
    camera: CameraConfig = field(default_factory=CameraConfig)
    slam: SLAMConfig = field(default_factory=SLAMConfig)
    depth: DepthConfig = field(default_factory=DepthConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    log_level: str = "INFO"
    run_name: str | None = None

    def __post_init__(self) -> None:
        if self.run_name is not None:
            validate_run_name(self.run_name)


def _build_camera(raw: dict[str, Any] | None) -> CameraConfig:
    raw = raw or {}
    inline = None
    if "fx" in raw and "intrinsics_file" not in raw:
        inline = CameraIntrinsics.from_dict(raw)
    file_path = raw.get("intrinsics_file")
    return CameraConfig(
        intrinsics_file=_expand_path(file_path) if file_path else None,
        inline=inline,
        default_focal_ratio=float(raw.get("default_focal_ratio", 1.2)),
    )


def _class_names(raw: Any) -> tuple[str, ...] | None:
    if raw is None:
        return None
    if (
        not isinstance(raw, list)
        or not raw
        or any(not isinstance(name, str) or not name.strip() for name in raw)
    ):
        raise ValueError("detection.class_names must be a nonempty list of names")
    return tuple(raw)


def load_config(
    path: str | Path | None = None, overrides: dict[str, Any] | None = None
) -> AppConfig:
    """Load YAML config and apply flat CLI overrides.

    ``overrides`` keys use dotted paths like ``video.path`` or ``output.dir``.
    """
    data: dict[str, Any] = {}
    if path is not None:
        cfg_path = _expand_path(path)
        if not cfg_path.is_file():
            raise FileNotFoundError(f"config file not found: {cfg_path}")
        loaded = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
        if not isinstance(loaded, dict):
            raise ValueError(f"config root must be a mapping: {cfg_path}")
        data = loaded

    if overrides:
        data = _apply_overrides(data, overrides)

    video_raw = data.get("video") or {}
    if "path" not in video_raw:
        raise ValueError("config must set video.path")

    return AppConfig(
        video=VideoConfig(
            path=Path(video_raw["path"]),
            max_frames=video_raw.get("max_frames"),
            start_frame=int(video_raw.get("start_frame", 0)),
            every_n_frames=int(video_raw.get("every_n_frames", 1)),
        ),
        camera=_build_camera(data.get("camera")),
        slam=SLAMConfig(
            backend=str(data.get("slam", {}).get("backend", "opencv_vo")),
            vo=dict(data.get("slam", {}).get("vo") or {}),
        ),
        depth=DepthConfig(
            enabled=bool(data.get("depth", {}).get("enabled", True)),
            estimator=str(data.get("depth", {}).get("estimator", "sparse_depth")),
            downsample=int(data.get("depth", {}).get("downsample", 2)),
            max_frames=int(data.get("depth", {}).get("max_frames", 12)),
            fill_radius=int(data.get("depth", {}).get("fill_radius", 8)),
        ),
        detection=DetectionConfig(
            enabled=bool(data.get("detection", {}).get("enabled", True)),
            detector=str(data.get("detection", {}).get("detector", "demo")),
            model_path=data.get("detection", {}).get("model_path"),
            conf_threshold=float(data.get("detection", {}).get("conf_threshold", 0.35)),
            nms_threshold=float(data.get("detection", {}).get("nms_threshold", 0.45)),
            max_objects=int(data.get("detection", {}).get("max_objects", 200)),
            min_track_hits=int(data.get("detection", {}).get("min_track_hits", 1)),
            input_width=int(data.get("detection", {}).get("input_width", 640)),
            input_height=int(data.get("detection", {}).get("input_height", 640)),
            class_names=_class_names(data.get("detection", {}).get("class_names")),
            pegboard_refinement=bool(data.get("detection", {}).get("pegboard_refinement", False)),
            pegboard_config=dict(data.get("detection", {}).get("pegboard_config") or {}),
            tiling=dict(data.get("detection", {}).get("tiling") or {}),
            open_vocabulary=dict(data.get("detection", {}).get("open_vocabulary") or {}),
            tracker=dict(data.get("detection", {}).get("tracker") or {}),
            lift=dict(data.get("detection", {}).get("lift") or {}),
            fusion_window=int(data.get("detection", {}).get("fusion_window", 12)),
            max_position_jump_ratio=float(
                data.get("detection", {}).get("max_position_jump_ratio", 0.5)
            ),
            max_position_history=int(data.get("detection", {}).get("max_position_history", 500)),
            reprojection_box_margin=float(
                data.get("detection", {}).get("reprojection_box_margin", 0.15)
            ),
            min_multiview_observations=int(
                data.get("detection", {}).get("min_multiview_observations", 3)
            ),
            min_multiview_parallax_deg=float(
                data.get("detection", {}).get("min_multiview_parallax_deg", 1.5)
            ),
            max_multiview_reprojection_px=float(
                data.get("detection", {}).get("max_multiview_reprojection_px", 12.0)
            ),
            max_center_surface_depth_ratio=float(
                data.get("detection", {}).get("max_center_surface_depth_ratio", 0.35)
            ),
            occupancy_thickness_ratio=float(
                data.get("detection", {}).get("occupancy_thickness_ratio", 0.5)
            ),
            occupancy_multiview_enabled=data.get("detection", {}).get(
                "occupancy_multiview_enabled", True
            ),
            occupancy_multiview_max_views=data.get("detection", {}).get(
                "occupancy_multiview_max_views", 6
            ),
            occupancy_multiview_min_angle_deg=float(
                data.get("detection", {}).get("occupancy_multiview_min_angle_deg", 2.0)
            ),
            occupancy_multiview_bbox_margin=float(
                data.get("detection", {}).get("occupancy_multiview_bbox_margin", 0.08)
            ),
        ),
        output=OutputConfig(
            dir=Path(data.get("output", {}).get("dir", "outputs")),
            save_trajectory_csv=bool(data.get("output", {}).get("save_trajectory_csv", True)),
            save_trajectory_txt=bool(data.get("output", {}).get("save_trajectory_txt", True)),
            save_map_ply=bool(data.get("output", {}).get("save_map_ply", True)),
            save_report_json=bool(data.get("output", {}).get("save_report_json", True)),
            save_depth=bool(data.get("output", {}).get("save_depth", True)),
            visualize=bool(data.get("output", {}).get("visualize", False)),
            copy_source_video=bool(data.get("output", {}).get("copy_source_video", False)),
            save_reconstruction=bool(data.get("output", {}).get("save_reconstruction", True)),
        ),
        log_level=str(data.get("log_level", "INFO")),
        run_name=data.get("run_name"),
    )


def _apply_overrides(data: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    import copy

    result = copy.deepcopy(data)
    for key, value in overrides.items():
        if value is None:
            continue
        parts = key.split(".")
        node = result
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
    return result
