"""Lift detections using explicit geometric support, never invented depth.

Sparse-grid fill is a display approximation: it can copy a background depth
into a foreground box. It must not be used as object-position evidence.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral, Real

import cv2
import numpy as np

from visualmap.depth.types import DepthMap
from visualmap.detection.types import Detection, SemanticObject, TrackedDetection
from visualmap.slam.types import CameraIntrinsics, CameraPose, MapPoint

__all__ = ["LiftConfig", "lift_detection"]


@dataclass
class LiftConfig:
    # Fraction of each half-extent retained around the box centre.
    center_frac: float = 0.35
    min_valid_pixels: int = 4
    # Configuration compatibility only: this is no longer a measurement.
    default_depth: float | None = None
    min_sparse_points: int = 3
    max_depth_relative_spread: float = 0.35
    depth_cluster_gap_ratio: float = 0.2
    allow_projected_points: bool = False

    def __post_init__(self) -> None:
        for name in ("center_frac", "max_depth_relative_spread", "depth_cluster_gap_ratio"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not math.isfinite(value)
                or not 0 < value <= 1
            ):
                raise ValueError(f"{name} must be finite and in (0, 1]")
        for name in ("min_valid_pixels", "min_sparse_points"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
                raise ValueError(f"{name} must be an integer >= 1")
        if self.default_depth is not None and (
            isinstance(self.default_depth, bool)
            or not isinstance(self.default_depth, Real)
            or not math.isfinite(self.default_depth)
            or self.default_depth <= 0
        ):
            raise ValueError("default_depth must be None or finite and positive")
        if not isinstance(self.allow_projected_points, bool):
            raise ValueError("allow_projected_points must be a boolean")


def lift_detection(
    det: Detection,
    pose: CameraPose | None,
    intrinsics: CameraIntrinsics,
    depth_map: DepthMap | None = None,
    config: LiftConfig | None = None,
    track: TrackedDetection | None = None,
    map_points: list[MapPoint] | None = None,
    depth_method: str | None = None,
    object_observations: list[tuple[Detection, CameraPose]] | None = None,
    preferred_point_ids: set[int] | None = None,
    target_depth: float | None = None,
    object_observations_trusted: bool = False,
) -> SemanticObject:
    """Estimate a visible surface centre in scale-ambiguous map units.

    Uncertainty is support scatter, not a calibrated error interval. Camera
    pose error and monocular scale error are not estimated. Without usable
    geometry a detection always has ``xyz=None``.
    """
    cfg = config or LiftConfig()
    xyz = None
    depth_val = None
    source = "none"
    radius = None
    status = "unavailable"
    evidence: dict = {"reason": "no_depth_evidence", "sample_count": 0}

    if pose is None:
        evidence["reason"] = "missing_camera_pose"
    elif pose.frame_id not in (-1, det.frame_id):
        evidence["reason"] = "camera_pose_frame_mismatch"
    elif not np.isfinite(pose.matrix).all():
        evidence["reason"] = "invalid_camera_pose"
    else:
        xyz, depth_val, source, radius, evidence = _sparse_support(
            det,
            pose,
            intrinsics,
            map_points or [],
            cfg,
            object_observations or [],
            preferred_point_ids or set(),
            target_depth,
            object_observations_trusted,
        )
        if xyz is not None:
            status = (
                "estimated"
                if source in {"sparse_observed", "persistent_surface_multiview"}
                else "tentative"
            )
        elif depth_method in ("sparse_depth", "sparse", "triangulation"):
            if evidence.get("reason") == "insufficient_sparse_support":
                evidence["reason"] = "no_direct_sparse_support"
            evidence["grid_depth_rejected"] = "sparse_grid_fill_is_not_object_evidence"
        else:
            depth_val, dense_evidence = _sample_depth(det, depth_map, cfg)
            if depth_val is not None:
                if intrinsics.has_distortion:
                    pixel = np.array([[[det.cx, det.cy]]], dtype=np.float64)
                    x, y = cv2.undistortPoints(
                        pixel,
                        intrinsics.K,
                        intrinsics.dist_coeffs,
                    ).reshape(2)
                else:
                    x = (det.cx - intrinsics.cx) / intrinsics.fx
                    y = (det.cy - intrinsics.cy) / intrinsics.fy
                xyz = pose.transform_point(np.array([x, y, 1.0]) * depth_val)
                source = "depth_median"
                radius = float(dense_evidence["depth_spread"])
                evidence = dense_evidence
                status = "tentative"
            elif not map_points:
                evidence = dense_evidence

    track_id = track.track_id if track is not None else -1
    hits = track.hits if track is not None else 1
    return SemanticObject(
        track_id=track_id,
        class_id=det.class_id,
        class_name=det.class_name,
        xyz=xyz,
        depth=float(depth_val) if depth_val is not None else None,
        first_frame=det.frame_id,
        last_frame=det.frame_id,
        observations=hits,
        confidence=det.confidence,
        xyxy=det.xyxy,
        frame_id=det.frame_id,
        metadata={
            "depth_source": source,
            "position_status": status,
            "position_source": source,
            "position_frame": det.frame_id if xyz is not None else None,
            "position_uncertainty": radius,
            "position_observations": 1 if xyz is not None else 0,
            "position_estimate": "visible_surface_support_centre",
            "position_relation_kind": "surface_support"
            if source == "persistent_surface_multiview"
            else None,
            "center_bias_unknown": True,
            "uncertainty": {
                "radius": radius,
                "units": "map_units",
                "method": "robust_support_scatter",
                "includes_pose_error": False,
                "calibrated_error_bound": False,
            },
            "evidence": evidence,
            "units": "map_units",
        },
    )


def _inner_box(det: Detection, cfg: LiftConfig) -> tuple[float, float, float, float]:
    frac = float(cfg.center_frac)
    half_w = det.width * 0.5 * frac
    half_h = det.height * 0.5 * frac
    return det.cx - half_w, det.cy - half_h, det.cx + half_w, det.cy + half_h


def _sparse_support(
    det: Detection,
    pose: CameraPose,
    intr: CameraIntrinsics,
    points: list[MapPoint],
    cfg: LiftConfig,
    object_observations: list[tuple[Detection, CameraPose]] | None = None,
    preferred_point_ids: set[int] | None = None,
    target_depth: float | None = None,
    object_observations_trusted: bool = False,
) -> tuple[np.ndarray | None, float | None, str, float | None, dict]:
    evidence: dict = {"reason": "insufficient_sparse_support", "sample_count": 0}
    if not points:
        return None, None, "none", None, evidence
    xyz = np.stack([point.xyz for point in points])
    camera = pose.inverse().transform_point(xyz)
    z = camera[:, 2]
    valid = np.isfinite(camera).all(axis=1) & (z > 1e-6)
    if intr.has_distortion:
        # Boxes are measured on the original (distorted) source video frames.
        # Project only finite front-facing points; NaNs cannot enter OpenCV math.
        uv = np.full((len(points), 2), np.nan)
        if np.any(valid):
            projected, _ = cv2.projectPoints(
                camera[valid],
                np.zeros(3),
                np.zeros(3),
                intr.K,
                intr.dist_coeffs,
            )
            uv[valid] = projected.reshape(-1, 2)
        u, v = uv[:, 0], uv[:, 1]
    else:
        safe_z = np.where(valid, z, 1.0)
        u = intr.fx * camera[:, 0] / safe_z + intr.cx
        v = intr.fy * camera[:, 1] / safe_z + intr.cy
    x1, y1, x2, y2 = _inner_box(det, cfg)
    in_box = valid & (u >= x1) & (u <= x2) & (v >= y1) & (v <= y2)
    in_box &= (u >= 0) & (u < intr.width) & (v >= 0) & (v < intr.height)
    observed = np.array([det.frame_id in point.observations for point in points])
    triangulated = np.array([len(point.observations) >= 2 for point in points])
    direct = np.flatnonzero(in_box & observed & triangulated)
    minimum = cfg.min_sparse_points
    source = "sparse_observed"
    if direct.size < minimum and cfg.allow_projected_points:
        direct = np.flatnonzero(in_box)
        source = "sparse_projected"
    evidence.update(
        {
            "candidate_point_count": int(np.count_nonzero(in_box)),
            "observed_point_count": int(np.count_nonzero(in_box & observed & triangulated)),
        }
    )
    if direct.size < minimum:
        return None, None, "none", None, evidence

    # Membership in one frame's box cannot distinguish the object from a wall
    # behind it. Reuse landmark identity only when its observations and geometry
    # remain inside this *same object's* boxes in earlier valid camera frames.
    prior = [entry for entry in (object_observations or []) if entry[0].frame_id != det.frame_id]
    direct_centre_frames: dict[int, list[int]] = {int(index): [det.frame_id] for index in direct}
    support_poses = {entry[0].frame_id: entry[1] for entry in prior}
    support_poses[det.frame_id] = pose
    if prior and source == "sparse_observed":
        consistent = []
        for index in direct:
            point = points[index]
            rows = []
            directly_seen = 0
            for old_det, old_pose in prior:
                camera_point = old_pose.inverse().transform_point(point.xyz)
                if not np.isfinite(camera_point).all() or camera_point[2] <= 1e-6:
                    rows.append(False)
                    continue
                projected, _ = cv2.projectPoints(
                    camera_point.reshape(1, 3),
                    np.zeros(3),
                    np.zeros(3),
                    intr.K,
                    intr.dist_coeffs,
                )
                pu, pv = projected.reshape(2)
                bx1, by1, bx2, by2 = old_det.xyxy
                inside = bx1 <= pu <= bx2 and by1 <= pv <= by2
                rows.append(bool(inside))
                directly_seen += int(inside and old_det.frame_id in point.observations)
                cx1, cy1, cx2, cy2 = _inner_box(old_det, cfg)
                if cx1 <= pu <= cx2 and cy1 <= pv <= cy2 and old_det.frame_id in point.observations:
                    direct_centre_frames[int(index)].append(old_det.frame_id)
            if all(rows) and directly_seen >= 1:
                consistent.append(int(index))
        direct = np.asarray(consistent, dtype=int)
        evidence["temporally_consistent_point_count"] = int(direct.size)
        evidence["validated_object_frames"] = [entry[0].frame_id for entry in prior] + [
            det.frame_id
        ]
        if direct.size < minimum:
            evidence["reason"] = "landmarks_not_consistent_with_object_track"
            return None, None, "none", None, evidence

    order = direct[np.argsort(z[direct])]
    gaps = np.diff(z[order]) / np.maximum(z[order[:-1]], 1e-6)
    groups = np.split(order, np.flatnonzero(gaps > cfg.depth_cluster_gap_ratio) + 1)
    supported = [group for group in groups if group.size >= minimum]
    if not supported:
        evidence["reason"] = "no_supported_depth_layer"
        return None, None, "none", None, evidence
    evidence["supported_depth_layers"] = [
        {"depth": float(np.median(z[group])), "count": int(group.size)} for group in supported
    ]
    if target_depth is not None:
        compatible = [
            group
            for group in supported
            if abs(float(np.median(z[group])) - target_depth) / max(target_depth, 1e-6)
            <= cfg.max_depth_relative_spread
        ]
        if not compatible:
            evidence["reason"] = "object_centre_and_surface_depth_disagree"
            evidence["object_centre_depth"] = float(target_depth)
            return None, None, "none", None, evidence
        supported = compatible
    preferred = preferred_point_ids or set()
    persistent = [
        group
        for group in supported
        if sum(points[index].id in preferred for index in group) >= minimum
    ]
    if persistent:
        supported = persistent
        evidence["persistent_landmarks_reused"] = True
    if len(supported) != 1:
        # More texture must never grant a background layer ownership of an object.
        evidence["reason"] = "ambiguous_foreground_background_depth_layers"
        return None, None, "none", None, evidence
    support = supported[0]
    qualified = [
        int(index) for index in support if len(set(direct_centre_frames.get(int(index), []))) >= 3
    ]
    if (
        object_observations_trusted
        and source == "sparse_observed"
        and len(qualified) >= max(3, minimum)
    ):
        qualified_frames = sorted(
            {frame for index in qualified for frame in direct_centre_frames[index]}
        )
        centres = np.array([support_poses[frame].t for frame in qualified_frames])
        baseline = float(np.max(np.linalg.norm(centres[:, None] - centres[None, :], axis=2)))
        if baseline > 1e-6:
            support = np.asarray(qualified, dtype=int)
            source = "persistent_surface_multiview"
            evidence.update(
                {
                    "persistent_support_qualified": True,
                    "persistent_support_point_ids": [int(points[index].id) for index in support],
                    "persistent_support_frames": qualified_frames,
                    "persistent_support_point_frames": {
                        str(points[index].id): sorted(set(direct_centre_frames[int(index)]))
                        for index in support
                    },
                    "persistent_support_min_views": min(
                        len(set(direct_centre_frames[index])) for index in qualified
                    ),
                    "persistent_support_camera_baseline": baseline,
                    "persistent_support_poses_trusted": True,
                    "support_region": "object_centre_subbox",
                    "center_bias_unknown": True,
                    "physical_object_membership_verified": False,
                }
            )
    evidence.setdefault("persistent_support_qualified", False)
    median_z = float(np.median(z[support]))
    spread = float(1.4826 * np.median(np.abs(z[support] - median_z)))
    relative_spread = spread / median_z
    if relative_spread > cfg.max_depth_relative_spread:
        evidence.update(
            {
                "reason": "inconsistent_sparse_depth",
                "depth_relative_spread": relative_spread,
            }
        )
        return None, None, "none", None, evidence

    centre = np.median(xyz[support], axis=0)
    radius = float(np.quantile(np.linalg.norm(xyz[support] - centre, axis=1), 0.75))
    evidence.update(
        {
            "reason": "direct_geometric_support",
            "sample_count": int(support.size),
            "observed_point_count": int(
                np.count_nonzero(observed[support] & triangulated[support])
            ),
            "point_ids": [int(points[index].id) for index in support],
            "depth_relative_spread": relative_spread,
            "occlusion_verified": False,
            "object_segmentation_available": False,
        }
    )
    return centre, median_z, source, radius, evidence


def _sample_depth(
    det: Detection,
    depth_map: DepthMap | None,
    cfg: LiftConfig,
) -> tuple[float | None, dict]:
    evidence: dict = {"reason": "no_depth_evidence", "sample_count": 0}
    if depth_map is None:
        return None, evidence
    if depth_map.frame_id != det.frame_id:
        evidence["reason"] = "depth_frame_mismatch"
        return None, evidence
    h, w = depth_map.height, depth_map.width
    if depth_map.source_width > 0 and depth_map.source_height > 0:
        scale_x = w / float(depth_map.source_width)
        scale_y = h / float(depth_map.source_height)
    else:
        if det.xyxy[2] > w or det.xyxy[3] > h:
            evidence["reason"] = "depth_source_dimensions_unknown"
            return None, evidence
        scale_x = scale_y = 1.0
    x1, y1, x2, y2 = _inner_box(det, cfg)
    ux1, ux2 = max(0, int(np.floor(x1 * scale_x))), min(w, int(np.ceil(x2 * scale_x)))
    uy1, uy2 = max(0, int(np.floor(y1 * scale_y))), min(h, int(np.ceil(y2 * scale_y)))
    if ux2 <= ux1 or uy2 <= uy1:
        evidence["reason"] = "box_outside_depth_map"
        return None, evidence
    patch = depth_map.depth[uy1:uy2, ux1:ux2]
    valid = patch[np.isfinite(patch) & (patch > 0)]
    evidence["sample_count"] = int(valid.size)
    if valid.size < cfg.min_valid_pixels:
        evidence["reason"] = "insufficient_depth_pixels"
        return None, evidence
    median = float(np.median(valid))
    spread = float(1.4826 * np.median(np.abs(valid - median)))
    relative_spread = spread / median
    if relative_spread > cfg.max_depth_relative_spread:
        evidence.update(
            {
                "reason": "inconsistent_depth_pixels",
                "depth_relative_spread": relative_spread,
            }
        )
        return None, evidence
    evidence.update(
        {
            "reason": "depth_pixels_in_box_centre",
            "depth_spread": spread,
            "depth_relative_spread": relative_spread,
            "occlusion_verified": False,
            "object_segmentation_available": False,
        }
    )
    return median, evidence
