"""Same-frame visual occupancy proxies, never certified physical envelopes.

An image box and a surface depth constrain the visible width and height, but
cannot reveal an object's hidden thickness or prove its landmark membership.
This module makes that missing dimension explicit. It deliberately never marks
a proxy as safe for collision avoidance or an empty region as proven free.
"""

from __future__ import annotations

from copy import deepcopy
from numbers import Integral, Real

import cv2
import numpy as np

from visualmap.detection.types import Detection, SemanticResult
from visualmap.mapping.visual_hull import FRUSTUM_FACES, clip_faces, frustum_faces, serialize_shape
from visualmap.slam.types import CameraPose, MapResult

__all__ = ["attach_occupancy"]


def _vector(value, length: int) -> np.ndarray | None:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    return array if array.shape == (length,) and np.isfinite(array).all() else None


def _base(snapshot: dict) -> dict:
    return {
        "schema_version": 1,
        "frame_id": snapshot.get("frame_id"),
        "world_frame_id": snapshot.get("world_frame_id"),
        "units": "map_units",
        "status": "unavailable",
        "available": False,
        "method": "bbox_surface_depth_inferred_thickness_world_aabb",
        "dimensions_source": "same_frame_bbox_and_surface_depth_with_inferred_thickness",
        "inferred_dimensions": ["hidden_thickness"],
        "volume_verified": False,
        "volume_map_units3": None,
        "bounds_are_guaranteed": False,
        "collision_usable": False,
        "candidate_overlap_usable": False,
        "free_space_certified": False,
        "collision_clearance_certified": False,
        "false_free_policy": "absence_or_separation_of_proxies_does_not_prove_free_space",
        "observed_support_count": 0,
        "support_point_ids": [],
        "support_coordinates_used_for_dimensions": False,
        "position_error_included_in_extent": False,
        "historical_support_policy": "exact_snapshot_ids_observed_here_and_twice_by_this_frame",
        "min": None,
        "max": None,
        "center": None,
        "size": None,
        "corners": [],
        "reasons": [],
    }


def _unavailable(result: dict, reason: str) -> dict:
    result["reason"] = reason
    result["reasons"] = [reason]
    return result


def _record_evidence(record: dict) -> tuple[dict, dict]:
    classification = record.get("classification_evidence")
    detection = record.get("detection_evidence")
    return (
        classification if isinstance(classification, dict) else {},
        detection if isinstance(detection, dict) else {},
    )


def _structural_membership_unknown(record: dict) -> bool:
    classification, detection = _record_evidence(record)
    source = classification.get("source", detection.get("classification_source"))
    membership = classification.get(
        "physical_object_membership_verified",
        detection.get("physical_object_membership_verified"),
    )
    return source == "image_structure_rule" and membership is not True


def _has_class_conflict(record: dict) -> bool:
    classification, detection = _record_evidence(record)
    return bool(
        record.get("classification_needs_review")
        or classification.get("needs_review")
        or classification.get("temporal_class_disagreement")
        or detection.get("classification_structural_conflict")
        or detection.get("temporal_class_disagreement")
        or record.get("class_name") != record.get("raw_class_name", record.get("class_name"))
    )


def _dynamic(record: dict, frame: int) -> bool:
    # Reuse the pipeline's class-prior policy; this is not observed motion.
    from visualmap.detection.pipeline import _potentially_moving

    classification, detection = _record_evidence(record)
    raw_class = str(record.get("raw_class_name", record.get("class_name", "")))
    metadata = {**detection, "latest_raw_class": raw_class}
    if metadata.get("static_scene_assumed") is False:
        return True
    return _potentially_moving(
        Detection(
            frame, 0.0, -1, str(record.get("class_name", "")), 0.0, (0.0, 0.0, 1.0, 1.0), metadata
        )
    ) or bool(classification.get("motion_detected"))


def _support_ids(snapshot: dict, mapping: MapResult, frame: int, points: dict) -> list[int]:
    evidence = snapshot.get("support_evidence") or snapshot.get("surface_support_evidence") or {}
    candidates = evidence.get("point_ids", [])
    if not isinstance(candidates, (list, tuple)):
        return []
    quality = mapping.metadata.get("frame_quality", {})
    accepted = set()
    for point_id in candidates:
        if isinstance(point_id, bool) or not isinstance(point_id, Integral):
            continue
        point = points.get(point_id)
        if point is None or frame not in point.observations:
            continue
        # A landmark first triangulated in a later frame did not exist yet.
        observed_frames = [fid for fid in point.observations if fid <= frame]
        if len(observed_frames) < 2:
            continue
        worlds = [
            quality.get(str(fid), quality.get(fid, {})).get(
                "world_frame_id", mapping.metadata.get("world_frame_id")
            )
            for fid in observed_frames
        ]
        if any(world != snapshot.get("world_frame_id") for world in worlds):
            continue
        accepted.add(int(point_id))
    return sorted(accepted)


def _corners(low: np.ndarray, high: np.ndarray) -> list[list[float]]:
    return [
        [low[0], low[1], low[2]],
        [high[0], low[1], low[2]],
        [high[0], high[1], low[2]],
        [low[0], high[1], low[2]],
        [low[0], low[1], high[2]],
        [high[0], low[1], high[2]],
        [high[0], high[1], high[2]],
        [low[0], high[1], high[2]],
    ]


def _estimate(
    snapshot: dict,
    record: dict | None,
    pose: CameraPose | None,
    mapping: MapResult,
    points: dict,
    inferred_thickness_ratio: float,
) -> dict:
    result = _base(snapshot)
    frame = snapshot.get("frame_id")
    intr = mapping.intrinsics
    if isinstance(frame, bool) or not isinstance(frame, Integral) or frame < 0:
        return _unavailable(result, "invalid_snapshot_frame")
    if record is None:
        return _unavailable(result, "no_same_frame_detection")
    if record.get("_duplicate_frame_record"):
        return _unavailable(result, "ambiguous_same_frame_detection")
    if _structural_membership_unknown(record):
        return _unavailable(result, "structural_candidate_object_membership_unverified")
    if intr is None:
        return _unavailable(result, "missing_intrinsics")
    if (
        pose is None
        or snapshot.get("pose_valid") is not True
        or snapshot.get("pose_trusted") is not True
        or snapshot.get("position_frame") != frame
        or snapshot.get("status") == "retained"
    ):
        return _unavailable(result, "no_trusted_same_frame_position")
    if not np.allclose(pose.R.T @ pose.R, np.eye(3), atol=1e-6) or not np.isclose(
        np.linalg.det(pose.R), 1.0, atol=1e-6
    ):
        return _unavailable(result, "invalid_pose_rotation")
    quality = mapping.metadata.get("frame_quality", {})
    row = quality.get(str(frame), quality.get(frame, {}))
    world = row.get("world_frame_id", mapping.metadata.get("world_frame_id"))
    if (
        not snapshot.get("world_frame_id")
        or world != snapshot.get("world_frame_id")
        or mapping.metadata.get("scale_consistent") is not True
        or (
            row
            and (
                row.get("pose_valid") is not True
                or row.get("status") not in {"tracked", "bootstrap", "relocalized"}
            )
        )
    ):
        return _unavailable(result, "world_or_pose_quality_mismatch")
    xyz = _vector(snapshot.get("xyz"), 3)
    if xyz is None:
        return _unavailable(result, "no_finite_position")
    # Compute from this exact pose and snapshot, never obj.xyz or a later pose.
    camera_xyz = pose.inverse().transform_point(xyz)
    depth = float(camera_xyz[2])
    if not np.isfinite(depth) or depth <= 1e-6:
        return _unavailable(result, "position_not_in_front_of_camera")
    stored_camera = _vector(snapshot.get("camera_xyz"), 3)
    if stored_camera is not None and not np.allclose(
        stored_camera, camera_xyz, rtol=1e-5, atol=1e-7
    ):
        return _unavailable(result, "snapshot_camera_world_position_disagreement")
    bbox = _vector(record.get("xyxy"), 4)
    if bbox is None:
        return _unavailable(result, "no_finite_same_frame_bbox")
    x1, y1, x2, y2 = bbox
    if not (0 <= x1 < x2 <= intr.width and 0 <= y1 < y2 <= intr.height):
        return _unavailable(result, "invalid_or_outside_image_bbox")
    pixels = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float64)
    if intr.has_distortion:
        try:
            rays = cv2.undistortPoints(pixels.reshape(-1, 1, 2), intr.K, intr.dist_coeffs).reshape(
                -1, 2
            )
        except cv2.error:
            return _unavailable(result, "invalid_bbox_rays")
    else:
        rays = (pixels - [intr.cx, intr.cy]) / [intr.fx, intr.fy]
    if not np.isfinite(rays).all():
        return _unavailable(result, "invalid_bbox_rays")
    projected_size = np.ptp(rays, axis=0) * depth
    thickness = float(np.min(projected_size) * inferred_thickness_ratio)
    # Centre scatter is not observed object thickness. Keep it separate from
    # the solid proxy instead of turning uncertain positions into larger solids.
    scatter = snapshot.get("uncertainty")
    result["position_uncertainty"] = (
        float(scatter)
        if isinstance(scatter, Real)
        and not isinstance(scatter, bool)
        and np.isfinite(scatter)
        and scatter >= 0
        else None
    )
    if not np.isfinite(thickness) or thickness <= 1e-9:
        return _unavailable(result, "degenerate_extent")
    # Near-camera clipping is numerical geometry, not a bound on real thickness.
    near = max(depth * 0.05, depth - thickness * 0.5)
    far = depth + thickness * 0.5
    camera_corners = np.concatenate(
        [
            np.column_stack([rays * near, np.full(4, near)]),
            np.column_stack([rays * far, np.full(4, far)]),
        ]
    )
    world_corners = pose.transform_point(camera_corners)
    low, high = np.min(world_corners, axis=0), np.max(world_corners, axis=0)
    size = high - low
    volume = float(np.prod(size))
    if not np.isfinite(world_corners).all() or not np.isfinite(volume) or volume <= 0:
        return _unavailable(result, "non_finite_or_degenerate_extent")
    support = _support_ids(snapshot, mapping, frame, points)
    reasons = [
        "hidden_thickness_inferred",
        "full_object_boundary_unverified",
        "scale_is_not_metric",
        "pose_error_not_bounded",
    ]
    conflict = _has_class_conflict(record)
    dynamic = _dynamic(record, frame)
    clipped = x1 <= 1 or y1 <= 1 or x2 >= intr.width - 1 or y2 >= intr.height - 1
    if conflict:
        reasons.append("classification_conflict_or_review")
    if dynamic:
        reasons.append("potential_motion_class_prior")
    if clipped:
        reasons.append("image_boundary_clips_object")
    if snapshot.get("position_relation_usable") is not True:
        reasons.append("position_not_relation_usable")
    result.update(
        {
            "status": "inferred_unverified",
            "available": True,
            "reason": "hidden_thickness_and_full_extent_unverified",
            "reasons": reasons,
            "min": low.tolist(),
            "max": high.tolist(),
            "center": ((low + high) * 0.5).tolist(),
            "size": size.tolist(),
            "corners": _corners(low, high),
            "volume_map_units3": volume,
            "observed_support_count": len(support),
            "support_point_ids": support,
            "surface_depth": depth,
            "inferred_thickness": thickness,
            "inferred_thickness_ratio": inferred_thickness_ratio,
            "candidate_overlap_usable": bool(
                snapshot.get("position_relation_usable") is True
                and snapshot.get("temporal_consistency") == "consistent"
                and not conflict
                and not dynamic
                and not clipped
            ),
        }
    )
    _, _, proxy_volume = serialize_shape(frustum_faces(world_corners))
    result.update(
        {
            "shape_vertices": world_corners.tolist(),
            "shape_faces": [list(face) for face in FRUSTUM_FACES],
            "proxy_volume_map_units3": proxy_volume,
            "aabb_volume_map_units3": volume,
            "volume_basis": "bounded_camera_frustum_with_assumed_hidden_thickness",
        }
    )
    return result


def _view_planes(record, pose, intr, margin):
    """Bounding-box cone, widened for detection noise; only past trusted views."""
    box = np.asarray(record["xyxy"], dtype=float)
    width, height = box[2:] - box[:2]
    x1, y1 = box[:2] - np.array([width, height]) * margin
    x2, y2 = box[2:] + np.array([width, height]) * margin
    pixels = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=float)
    if intr.has_distortion:
        # A rectangular distorted outline has curved normalized edges. Avoid
        # claiming a four-plane enclosure for it without a proper silhouette.
        return []
    rays = (pixels - [intr.cx, intr.cy]) / [intr.fx, intr.fy]
    low, high = rays.min(axis=0), rays.max(axis=0)
    normals = ([1, 0, -low[0]], [-1, 0, high[0]], [0, 1, -low[1]], [0, -1, high[1]], [0, 0, 1])
    result = []
    for camera_normal in normals:
        normal = pose.R @ np.asarray(camera_normal, dtype=float)
        normal /= np.linalg.norm(normal)
        result.append((normal, -float(normal @ pose.t)))
    return result


def _refine_occupancy(result, snapshot, record, pose, history, intr, options):
    if result.get("available") is not True:
        return
    seed_vertices = result["shape_vertices"]
    faces = frustum_faces(seed_vertices)
    seed_volume = result["proxy_volume_map_units3"]
    current_ray = np.asarray(snapshot["xyz"]) - pose.t
    current_ray /= np.linalg.norm(current_ray)
    view_rays, view_ids, planes_used = [current_ray], [snapshot["frame_id"]], []
    contradictions = []
    enabled, max_views, min_angle, margin = options
    if enabled and result["candidate_overlap_usable"]:
        for previous in reversed(history):
            old_snapshot, old_record, old_pose, old_volume = previous
            if len(view_ids) >= max_views:
                break
            if (
                old_snapshot.get("world_frame_id") != snapshot.get("world_frame_id")
                or not old_volume.get("candidate_overlap_usable")
                or old_record.get("class_name") != record.get("class_name")
            ):
                continue
            ray = np.asarray(snapshot["xyz"]) - old_pose.t
            length = np.linalg.norm(ray)
            if length <= 1e-9:
                continue
            ray /= length
            angles = [np.degrees(np.arccos(np.clip(ray @ other, -1, 1))) for other in view_rays]
            if min(angles) < min_angle:
                continue
            planes = _view_planes(old_record, old_pose, intr, margin)
            if not planes:
                continue
            # Reject a wrong track association or a moving object instead of
            # shrinking it into an apparently precise, empty solid.
            if any(
                normal @ np.asarray(snapshot["xyz"]) + offset < -1e-7 for normal, offset in planes
            ):
                contradictions.append(old_snapshot["frame_id"])
                continue
            candidate = faces
            for normal, offset in planes:
                candidate = clip_faces(candidate, normal, offset)
            vertices, indices, volume = serialize_shape(candidate)
            if volume <= seed_volume * 0.05:
                contradictions.append(old_snapshot["frame_id"])
                continue
            faces = candidate
            planes_used.extend(planes)
            view_rays.append(ray)
            view_ids.append(old_snapshot["frame_id"])
    vertices, indices, proxy_volume = serialize_shape(faces)
    if len(view_ids) > 1:
        array = np.asarray(vertices)
        low, high = array.min(axis=0), array.max(axis=0)
        size = high - low
        result.update(
            {
                "method": "past_trusted_bbox_cones_refine_bounded_thickness_proxy",
                "min": low.tolist(),
                "max": high.tolist(),
                "size": size.tolist(),
                "center": ((low + high) / 2).tolist(),
                "corners": _corners(low, high),
                "shape_vertices": vertices,
                "shape_faces": indices,
                "proxy_volume_map_units3": proxy_volume,
                "aabb_volume_map_units3": float(np.prod(size)),
                # Legacy readers used this value as the AABB's volume. Preserve
                # that contract and expose the tighter shape's volume separately.
                "volume_map_units3": float(np.prod(size)),
                "volume_basis": "multi_view_constrained_proxy_with_assumed_hidden_thickness",
            }
        )
    result["multi_view"] = {
        "status": "constrained" if len(view_ids) > 1 else "single_view",
        "view_count": len(view_ids),
        "frame_ids": sorted(view_ids),
        "rejected_frame_ids": contradictions,
        "min_view_angle_deg": min_angle,
        "bbox_margin_fraction": margin,
        "future_views_used": False,
        "observed_silhouettes": False,
    }
    if contradictions:
        result["reasons"].append("some_past_views_conflict_with_current_geometry")
        result["candidate_overlap_usable"] = False
    # Report how much the answer changes when the unknown thickness changes.
    # This is assumption sensitivity, explicitly not a measured confidence band.
    camera_vertices = pose.inverse().transform_point(np.asarray(seed_vertices))
    rays = camera_vertices[:4, :2] / camera_vertices[:4, 2:3]
    depth, thickness = result["surface_depth"], result["inferred_thickness"]
    values = []
    for factor in (0.5, 1.0, 2.0):
        near, far = (
            max(depth * 0.05, depth - thickness * factor / 2),
            depth + thickness * factor / 2,
        )
        seed = pose.transform_point(
            np.concatenate(
                [
                    np.column_stack([rays * near, np.full(4, near)]),
                    np.column_stack([rays * far, np.full(4, far)]),
                ]
            )
        )
        candidate = frustum_faces(seed)
        for normal, offset in planes_used:
            candidate = clip_faces(candidate, normal, offset)
        values.append(serialize_shape(candidate)[2])
    result["volume_sensitivity"] = {
        "kind": "thickness_assumption_sensitivity_not_confidence_interval",
        "thickness_factors": [0.5, 1.0, 2.0],
        "proxy_volumes_map_units3": values,
        "min": min(values),
        "max": max(values),
    }


def attach_occupancy(
    map_result: MapResult,
    semantics: SemanticResult,
    observation_records: list[dict] | None = None,
    *,
    inferred_thickness_ratio: float = 0.5,
    multiview_enabled: bool = True,
    multiview_max_views: int = 6,
    multiview_min_angle_deg: float = 2.0,
    multiview_bbox_margin: float = 0.08,
) -> SemanticResult:
    """Attach grey-box display proxies without changing position evidence.

    ``observation_records`` is optional for live results; saved runs must pass
    their original tracking records. Every proxy joins an exact track/frame,
    pose and world. Missing joins produce an unavailable record, never a final
    or future coordinate/box. Thickness is an adjustable geometric heuristic
    with no class-specific dimensions and no guaranteed conservative bound.

    Returns the same result for convenience. ``candidate_overlap_usable`` only
    permits *possible proxy overlap* comparisons in the same frame/world. A
    negative comparison cannot certify collision clearance.
    """
    if (
        isinstance(inferred_thickness_ratio, bool)
        or not isinstance(inferred_thickness_ratio, Real)
        or not np.isfinite(inferred_thickness_ratio)
        or not 0 < inferred_thickness_ratio <= 4
    ):
        raise ValueError("inferred_thickness_ratio must be finite and in (0, 4]")
    if (
        not isinstance(multiview_enabled, bool)
        or isinstance(multiview_max_views, bool)
        or not isinstance(multiview_max_views, Integral)
        or not 1 <= multiview_max_views <= 16
        or isinstance(multiview_min_angle_deg, bool)
        or not isinstance(multiview_min_angle_deg, Real)
        or not np.isfinite(multiview_min_angle_deg)
        or not 0 < multiview_min_angle_deg <= 90
        or isinstance(multiview_bbox_margin, bool)
        or not isinstance(multiview_bbox_margin, Real)
        or not np.isfinite(multiview_bbox_margin)
        or not 0 <= multiview_bbox_margin <= 0.5
    ):
        raise ValueError("invalid multiview occupancy options")
    records = (
        observation_records
        if observation_records is not None
        else [track.to_dict() for track in semantics.tracks]
    )
    by_frame = {}
    for row in records:
        key = row.get("track_id"), row.get("frame_id")
        by_frame[key] = row if key not in by_frame else {"_duplicate_frame_record": True}
    poses = {pose.frame_id: pose for pose in map_result.trajectory}
    points = {point.id: point for point in map_result.map_points}
    generated, missing = 0, 0
    constrained = 0
    for obj in semantics.objects:
        current = None
        history = []
        for snapshot in sorted(
            obj.metadata.get("position_snapshots", []), key=lambda row: row.get("frame_id", -1)
        ):
            frame = snapshot.get("frame_id")
            occupancy = _estimate(
                snapshot,
                by_frame.get((obj.track_id, frame)),
                poses.get(frame),
                map_result,
                points,
                float(inferred_thickness_ratio),
            )
            if occupancy.get("available"):
                record, pose = by_frame[(obj.track_id, frame)], poses[frame]
                _refine_occupancy(
                    occupancy,
                    snapshot,
                    record,
                    pose,
                    history,
                    map_result.intrinsics,
                    (
                        multiview_enabled,
                        multiview_max_views,
                        multiview_min_angle_deg,
                        multiview_bbox_margin,
                    ),
                )
                constrained += occupancy["multi_view"]["view_count"] > 1
                history.append((snapshot, record, pose, occupancy))
            snapshot["occupancy"] = occupancy
            generated += occupancy["status"] == "inferred_unverified"
            missing += occupancy["status"] == "unavailable"
            if frame == obj.frame_id:
                current = occupancy
        obj.metadata["occupancy"] = (
            deepcopy(current)
            if current
            else _unavailable(
                _base({"frame_id": obj.frame_id}), "no_exact_current_position_snapshot"
            )
        )
    semantics.metadata["occupancy_policy"] = {
        "schema_version": 1,
        "method": "bounded_frustum_with_past_trusted_bbox_cones",
        "proxy_volume_field": "proxy_volume_map_units3",
        "broad_phase_volume_field": "aabb_volume_map_units3",
        "inferred_thickness_ratio": float(inferred_thickness_ratio),
        "units": "map_units",
        "estimated_snapshots": int(generated),
        "unavailable_snapshots": int(missing),
        "multiview_constrained_snapshots": int(constrained),
        "multiview_enabled": multiview_enabled,
        "multiview_max_views": multiview_max_views,
        "multiview_min_angle_deg": multiview_min_angle_deg,
        "multiview_bbox_margin": multiview_bbox_margin,
        "collision_usable_count": 0,
        "physical_volume_verified": False,
        "bounds_are_guaranteed": False,
        "free_space_certified": False,
        "false_free_policy": "absence_or_separation_of_proxies_does_not_prove_free_space",
    }
    return semantics
