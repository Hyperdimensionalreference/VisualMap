"""Semantic pipeline: detect → track → lift to 3D."""

from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import dataclass
from numbers import Integral, Real

import cv2
import numpy as np

from visualmap.depth.types import DepthMap, DepthResult
from visualmap.detection.base import ObjectDetector
from visualmap.detection.lift import LiftConfig, lift_detection
from visualmap.detection.tracker import IoUTracker, TrackerConfig
from visualmap.detection.types import Detection, SemanticObject, SemanticResult, TrackedDetection
from visualmap.slam.types import CameraIntrinsics, CameraPose, MapResult
from visualmap.utils.logging import get_logger

logger = get_logger(__name__)

__all__ = ["SemanticPipeline", "SemanticPipelineConfig", "create_detector"]


@dataclass
class SemanticPipelineConfig:
    tracker: TrackerConfig | None = None
    lift: LiftConfig | None = None
    max_objects: int = 200
    fusion_window: int = 12
    max_position_jump_ratio: float = 0.5
    max_position_history: int = 500
    reprojection_box_margin: float = 0.15
    min_multiview_observations: int = 3
    min_multiview_parallax_deg: float = 1.5
    max_multiview_reprojection_px: float = 12.0
    max_center_surface_depth_ratio: float = 0.35

    def __post_init__(self) -> None:
        for name, minimum in (
            ("max_objects", 1),
            ("fusion_window", 1),
            ("max_position_history", 2),
            ("min_multiview_observations", 3),
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        ratio = self.max_position_jump_ratio
        if (
            isinstance(ratio, bool)
            or not isinstance(ratio, Real)
            or not math.isfinite(ratio)
            or not 0 < ratio <= 1
        ):
            raise ValueError("max_position_jump_ratio must be finite and in (0, 1]")
        margin = self.reprojection_box_margin
        if (
            isinstance(margin, bool)
            or not isinstance(margin, Real)
            or not math.isfinite(margin)
            or not 0 <= margin <= 1
        ):
            raise ValueError("reprojection_box_margin must be finite and in [0, 1]")
        for name in (
            "min_multiview_parallax_deg",
            "max_multiview_reprojection_px",
            "max_center_surface_depth_ratio",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{name} must be finite and positive")
        if self.min_multiview_parallax_deg >= 90:
            raise ValueError("min_multiview_parallax_deg must be less than 90")
        if self.max_center_surface_depth_ratio > 1:
            raise ValueError("max_center_surface_depth_ratio must be at most 1")


class SemanticPipeline:
    """Run detector over frames (or precomputed detections) and lift to world."""

    def __init__(
        self,
        detector: ObjectDetector | None = None,
        config: SemanticPipelineConfig | None = None,
    ) -> None:
        self.detector = detector
        self.config = config or SemanticPipelineConfig()
        self._tracker = IoUTracker(self.config.tracker or TrackerConfig())
        self._all_detections: list[Detection] = []
        self._all_tracks: list[TrackedDetection] = []
        # Classification follows the current track; geometry has its own evidence.
        self._objects: dict[int, SemanticObject] = {}
        self._position_samples: dict[int, list[dict]] = {}
        self._position_history: dict[int, list[dict]] = {}
        self._geometry_observations: dict[int, list[dict]] = {}
        self._position_snapshots: dict[int, list[dict]] = {}

    def process_detections(
        self,
        detections: list[Detection],
        map_result: MapResult,
        depth_result: DepthResult | None = None,
        intrinsics: CameraIntrinsics | None = None,
    ) -> list[TrackedDetection]:
        intr = intrinsics or map_result.intrinsics
        if intr is None:
            raise ValueError("intrinsics required")

        tracked = self._tracker.update(detections)
        first_frames = {track.track_id: track.first_frame for track in self._tracker.tracks}
        self._all_detections.extend(detections)

        depth_by_frame: dict[int, DepthMap] = {}
        if depth_result is not None:
            for dm in depth_result.maps:
                depth_by_frame[dm.frame_id] = dm

        pose_by_frame = {p.frame_id: p for p in map_result.trajectory}

        for td in tracked:
            det = td.detection
            raw = getattr(td, "raw_detection", None) or det
            pose = pose_by_frame.get(det.frame_id)
            depth_map = depth_by_frame.get(det.frame_id)
            camera_evidence = self._camera_evidence(map_result, det.frame_id, pose)
            geometry_rows = self._geometry_observations.setdefault(td.track_id, [])
            # A new local map cannot inherit old world coordinates or camera rays.
            previous = self._objects.get(td.track_id)
            if (
                previous is not None
                and previous.metadata.get("world_frame_id") != camera_evidence["world_frame_id"]
            ):
                geometry_rows.clear()
                self._position_samples.pop(td.track_id, None)
                previous = None
                self._objects.pop(td.track_id, None)
            clipped = self._box_is_clipped(det, intr)
            # A delayed class vote must not let a current person/vehicle/animal
            # observation join a previously static triangulation or fusion prior.
            if _potentially_moving(raw):
                geometry_rows.clear()
                self._position_samples.pop(td.track_id, None)
            if camera_evidence["pose_trusted"] and pose is not None:
                geometry_rows[:] = [
                    row for row in geometry_rows if row["detection"].frame_id != det.frame_id
                ]
                geometry_rows.append({"detection": det, "pose": pose, **camera_evidence})
                del geometry_rows[
                    : -max(self.config.fusion_window, self.config.min_multiview_observations)
                ]
            multiview = self._triangulate_object_centre(geometry_rows, intr, det)
            multiview["centre_bias_unknown"] = True
            multiview["current_box_clipped"] = clipped
            if clipped:
                multiview["reason"] = "clipped_box_centre_is_not_a_stable_object_ray"
            earlier = [(row["detection"], row["pose"]) for row in geometry_rows]
            preferred_ids = (
                set(previous.metadata.get("evidence", {}).get("point_ids", []))
                if previous
                else set()
            )
            target_depth = (
                float(pose.inverse().transform_point(multiview["xyz"])[2])
                if multiview.get("xyz") is not None and pose
                else None
            )
            obj = lift_detection(
                det,
                pose if camera_evidence["pose_valid"] else None,
                intr,
                depth_map=depth_map,
                config=self.config.lift or LiftConfig(),
                track=td,
                map_points=self._points_in_world(map_result, camera_evidence["world_frame_id"]),
                depth_method=depth_result.metadata.get("method")
                if depth_result is not None
                else None,
                object_observations=earlier,
                preferred_point_ids=preferred_ids,
                target_depth=target_depth,
                object_observations_trusted=camera_evidence["pose_trusted"],
            )
            obj.metadata.update(camera_evidence)
            if not camera_evidence["pose_valid"]:
                obj.metadata["evidence"]["reason"] = "unreliable_camera_pose"
                obj.metadata["evidence"]["camera_status"] = camera_evidence["camera_status"]
            self._apply_multiview_position(obj, multiview, pose)
            obj.first_frame = first_frames[td.track_id]
            obj = self._update_position(obj, td, pose, intr)
            self._check_reprojection(obj, pose if camera_evidence["pose_valid"] else None, intr)
            obj.metadata.update(camera_evidence)
            obj.metadata["position_relation_usable"] = bool(
                camera_evidence["pose_trusted"]
                and obj.xyz is not None
                and obj.metadata.get("position_temporal_consistency") == "consistent"
                and (
                    obj.metadata.get("position_source") == "object_center_multiview"
                    or (
                        obj.metadata.get("position_source") == "persistent_surface_multiview"
                        and obj.metadata.get("evidence", {}).get("persistent_support_qualified")
                        is True
                    )
                )
                and obj.metadata.get("static_scene_assumed")
                and obj.metadata.get("position_status") != "retained"
                and not raw.metadata.get("classification_structural_conflict", False)
                and not det.metadata.get("classification_candidate_only", False)
                and not (
                    raw.metadata.get("classification_source") == "image_structure_rule"
                    and raw.metadata.get("physical_object_membership_verified") is not True
                )
            )
            self._record_snapshot(obj, td, pose, intr)
            obj.metadata["classification_consistency"] = getattr(td, "class_consistency", None)
            obj.metadata["classification_observations"] = getattr(td, "class_observations", td.hits)
            obj.metadata["classification_evidence"] = {
                "crop_verification": raw.metadata.get("crop_verification"),
                "candidate_only": det.metadata.get("classification_candidate_only", False),
                "latest_raw_class": raw.class_name,
                "latest_raw_confidence": float(raw.confidence),
                "alternatives": [
                    candidate.to_dict() for candidate in getattr(td, "class_alternatives", ())
                ],
                "accuracy_validated": False,
                "source": raw.metadata.get("classification_source", "model_inference"),
                "status": det.metadata.get("classification_status", "unverified_model_prediction"),
                "needs_review": det.metadata.get("classification_needs_review", False),
                "reason": det.metadata.get("classification_reason"),
                "raw_model_class": raw.metadata.get("raw_model_class", raw.class_name),
                "original_model_class": raw.metadata.get("original_model_class", raw.class_name),
                "structure_evidence": raw.metadata.get("structure_evidence"),
                "raw_model_detections": raw.metadata.get("raw_model_detections", []),
                "user_confirmed": raw.metadata.get("user_confirmed", False),
                "trained_pegboard_model": raw.metadata.get("trained_pegboard_model", False),
                "full_object_boundary_verified": raw.metadata.get("full_object_boundary_verified"),
                "physical_object_membership_verified": raw.metadata.get(
                    "physical_object_membership_verified"
                ),
                "class_semantics_validated": raw.metadata.get("class_semantics_validated", False),
                "possible_object_types": raw.metadata.get("possible_object_types", []),
                "score_is_probability": raw.metadata.get("rule_score_is_probability"),
                "stabilized_class": det.class_name,
                "temporal_class_disagreement": det.metadata.get(
                    "temporal_class_disagreement", False
                ),
                "temporal_reason": det.metadata.get("temporal_classification_reason"),
            }
            obj.metadata["detection_evidence"] = dict(raw.metadata)
            obj.metadata["classification_needs_review"] = det.metadata.get(
                "classification_needs_review", False
            )
            obj.metadata["classification_candidate_only"] = det.metadata.get(
                "classification_candidate_only", False
            )
            obj.metadata["classification_reason"] = det.metadata.get("classification_reason")
            self._objects[td.track_id] = obj

        self._all_tracks.extend(tracked)
        return tracked

    def result(self) -> SemanticResult:
        objects = list(self._objects.values())
        objects.sort(key=lambda o: (o.class_name, o.track_id))
        if len(objects) > self.config.max_objects:
            objects = objects[: self.config.max_objects]
        return SemanticResult(
            objects=objects,
            detections=list(self._all_detections),
            tracks=list(self._all_tracks),
            metadata={
                "pipeline": "detect_track_lift",
                "detector": getattr(self.detector, "name", "unknown"),
                "detection_capabilities": getattr(self.detector, "capabilities", {}),
                "position_policy": "persistent_landmarks_and_validated_multiview_centres",
                "position_units": "map_units",
                "position_accuracy_validated": False,
                "motion_detection_available": False,
                "motion_policy_basis": "class_prior_only",
                "motion_policy_limitation": "static_category_instances_can_also_move",
                "num_positioned_objects": sum(obj.xyz is not None for obj in objects),
                "num_unpositioned_objects": sum(obj.xyz is None for obj in objects),
                "num_retained_positions": sum(
                    obj.metadata.get("position_status") == "retained" for obj in objects
                ),
                "num_inconsistent_positions": sum(
                    obj.metadata.get("position_temporal_consistency") == "contradicted"
                    for obj in objects
                ),
                "num_relation_usable_positions": sum(
                    bool(obj.metadata.get("position_relation_usable")) for obj in objects
                ),
            },
        )

    @staticmethod
    def _box_is_clipped(det: Detection, intr: CameraIntrinsics) -> bool:
        x1, y1, x2, y2 = det.xyxy
        return x1 <= 1 or y1 <= 1 or x2 >= intr.width - 1 or y2 >= intr.height - 1

    @staticmethod
    def _camera_evidence(mapping: MapResult, frame_id: int, pose: CameraPose | None) -> dict:
        quality = mapping.metadata.get("frame_quality")
        row = (
            quality.get(str(frame_id), quality.get(frame_id, {}))
            if isinstance(quality, dict)
            else {}
        )
        known = isinstance(quality, dict)
        finite = pose is not None and np.isfinite(pose.matrix).all()
        valid = bool(finite and (row.get("pose_valid") is True if known else True))
        status = row.get("status", "unverified_legacy_pose")
        world_frame = row.get("world_frame_id", mapping.metadata.get("world_frame_id"))
        trusted = bool(
            valid
            and status in {"bootstrap", "tracked", "relocalized"}
            and world_frame is not None
            and mapping.metadata.get("scale_consistent") is True
        )
        return {
            "pose_valid": valid,
            "pose_trusted": trusted,
            "camera_status": status,
            "world_frame_id": world_frame,
            "scale_consistent": mapping.metadata.get("scale_consistent") is True,
            "global_scale_consistent": mapping.metadata.get("global_scale_consistent"),
            "scale_consistency_scope": mapping.metadata.get("scale_consistency_scope"),
        }

    @staticmethod
    def _points_in_world(mapping: MapResult, world_frame_id) -> list:
        """Never project a landmark from another independently initialized submap."""
        quality = mapping.metadata.get("frame_quality")
        labelled = mapping.metadata.get("landmark_world_frame_ids", {})
        if not isinstance(quality, dict):
            if (
                len(mapping.metadata.get("submaps", [])) <= 1
                and mapping.metadata.get("global_scale_consistent") is not False
                and not labelled
            ):
                return mapping.map_points  # Legacy single-map files lack world labels.
            quality = {}
        if world_frame_id is None:
            return []
        global_world = mapping.metadata.get("world_frame_id")
        known_worlds = {
            row.get("world_frame_id")
            for row in quality.values()
            if isinstance(row, dict) and row.get("world_frame_id") is not None
        }
        if global_world is not None:
            known_worlds.add(global_world)
        segmented = (
            len(known_worlds) > 1
            or len(mapping.metadata.get("submaps", [])) > 1
            or mapping.metadata.get("global_scale_consistent") is False
        )
        # Older single-world reconstructions put the world on MapResult only.
        # Dropping those points also drops contradicting background evidence,
        # which can wrongly promote an unsupported object-centre estimate.
        fallback_world = None if segmented else global_world
        selected = []
        for point in mapping.map_points:
            declared = labelled.get(str(point.id), labelled.get(point.id))
            if str(point.id) in labelled or point.id in labelled:
                if declared == world_frame_id:
                    selected.append(point)
                continue
            observed_worlds = [
                row.get("world_frame_id", fallback_world)
                for frame_id in point.observations
                if isinstance(row := quality.get(str(frame_id), quality.get(frame_id, {})), dict)
            ]
            worlds = set(observed_worlds)
            if not worlds and not point.observations and fallback_world is not None:
                worlds = {fallback_world}
            if worlds == {world_frame_id}:
                selected.append(point)
        return selected

    def _triangulate_object_centre(
        self,
        rows: list[dict],
        intr: CameraIntrinsics,
        current: Detection,
    ) -> dict:
        """Use actual camera baselines and a third-view check, never box size depth.

        Box centres approximate the visible object centre, not a rigid feature.
        Their residuals stay in the evidence and the estimate remains uncalibrated.
        A pair defines a hypothesis; independent views must validate it.
        """
        result: dict = {"xyz": None, "reason": "insufficient_reliable_object_views"}
        if _potentially_moving(current):
            result["reason"] = "dynamic_category_excluded_from_static_triangulation"
            return result
        rows = [row for row in rows if not self._box_is_clipped(row["detection"], intr)]
        if len(rows) < self.config.min_multiview_observations:
            return result
        if rows[-1]["detection"].frame_id != current.frame_id:
            result["reason"] = "current_camera_pose_not_trusted"
            return result
        pixels = np.array([[row["detection"].cx, row["detection"].cy] for row in rows])
        normal = cv2.undistortPoints(pixels.reshape(-1, 1, 2), intr.K, intr.dist_coeffs).reshape(
            -1, 2
        )
        poses = [row["pose"] for row in rows]
        centres = np.array([pose.t for pose in poses])
        rays = np.array(
            [pose.R @ np.r_[pixel, 1.0] for pose, pixel in zip(poses, normal, strict=True)]
        )
        rays /= np.linalg.norm(rays, axis=1, keepdims=True)
        thresholds = np.array(
            [
                min(
                    self.config.max_multiview_reprojection_px,
                    max(2.0, min(row["detection"].width, row["detection"].height) * 0.2),
                )
                for row in rows
            ]
        )
        baseline = float(np.max(np.linalg.norm(centres[:, None] - centres[None, :], axis=2)))
        result["camera_baseline"] = baseline
        if baseline <= 1e-6:
            result["reason"] = "no_translational_baseline"
            return result

        def evaluate(xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            camera = np.array([pose.inverse().transform_point(xyz) for pose in poses])
            depth = camera[:, 2]
            projected = np.full_like(pixels, np.inf)
            front = np.isfinite(camera).all(axis=1) & (depth > 1e-6)
            if front.any():
                projected[front] = cv2.projectPoints(
                    camera[front], np.zeros(3), np.zeros(3), intr.K, intr.dist_coeffs
                )[0].reshape(-1, 2)
            errors = np.linalg.norm(projected - pixels, axis=1)
            return errors <= thresholds, errors, depth

        best = None
        max_angle = 0.0
        for first in range(len(rows) - 1):
            for second in range(first + 1, len(rows)):
                angle = float(np.degrees(np.arccos(np.clip(rays[first] @ rays[second], -1, 1))))
                max_angle = max(max_angle, angle)
                if angle < self.config.min_multiview_parallax_deg:
                    continue
                if np.linalg.norm(centres[first] - centres[second]) <= 1e-6:
                    continue
                projections = [poses[index].inverse().matrix[:3] for index in (first, second)]
                homogeneous = cv2.triangulatePoints(
                    *projections, normal[first].reshape(2, 1), normal[second].reshape(2, 1)
                ).reshape(4)
                if abs(homogeneous[3]) <= 1e-10:
                    continue
                xyz = homogeneous[:3] / homogeneous[3]
                if not np.isfinite(xyz).all():
                    continue
                mask, errors, depth = evaluate(xyz)
                if not mask[-1] or int(mask.sum()) < self.config.min_multiview_observations:
                    continue
                if int(mask.sum()) < math.ceil(len(rows) * 0.8):
                    continue
                score = (int(mask.sum()), -float(np.median(errors[mask])))
                if best is None or score > best[0]:
                    best = (score, xyz, mask, errors, depth, [first, second])
        result["max_parallax_deg"] = max_angle
        if best is None:
            result["reason"] = (
                "insufficient_object_parallax"
                if max_angle < self.config.min_multiview_parallax_deg
                else "object_centre_reprojection_disagreement"
            )
            return result
        _, xyz, mask, errors, depth, pair = best
        # Refine all agreed ray intersections in one consistent world frame.
        projectors = np.eye(3)[None] - rays[mask, :, None] * rays[mask, None, :]
        lhs = projectors.sum(axis=0)
        if np.linalg.cond(lhs) > 1e8:
            result["reason"] = "ill_conditioned_object_rays"
            return result
        xyz = np.linalg.solve(lhs, np.einsum("nij,nj->i", projectors, centres[mask]))
        refined_mask, errors, depth = evaluate(xyz)
        if (
            not refined_mask[-1]
            or refined_mask.sum() < self.config.min_multiview_observations
            or refined_mask.sum() < math.ceil(len(rows) * 0.8)
        ):
            result["reason"] = "refined_centre_reprojection_disagreement"
            return result
        mask = refined_mask
        inlier_rays = rays[mask]
        inlier_angle = float(
            np.max(np.degrees(np.arccos(np.clip(inlier_rays @ inlier_rays.T, -1, 1))))
        )
        if inlier_angle < self.config.min_multiview_parallax_deg:
            result["reason"] = "insufficient_validated_object_parallax"
            return result
        # Near-parallel rays amplify even one pixel of centre jitter into a large
        # depth change. This explicit sensitivity floor is not a calibrated bound.
        pixel_scatter = max(1.0, float(np.quantile(errors[mask], 0.75)))
        lateral_sensitivity = float(np.median(depth[mask]) * pixel_scatter / min(intr.fx, intr.fy))
        depth_sensitivity = lateral_sensitivity / max(
            abs(math.sin(math.radians(inlier_angle))), 1e-6
        )
        radius = max(lateral_sensitivity, depth_sensitivity)
        result.update(
            {
                "xyz": xyz,
                "reason": "multiview_object_centre_validated",
                "radius": radius,
                "inlier_frames": [
                    rows[index]["detection"].frame_id for index in np.flatnonzero(mask)
                ],
                "hypothesis_frames": [rows[index]["detection"].frame_id for index in pair],
                "validation_frames": [
                    rows[index]["detection"].frame_id
                    for index in np.flatnonzero(mask)
                    if index not in pair
                ],
                "observation_count": int(mask.sum()),
                "median_reprojection_px": float(np.median(errors[mask])),
                "max_inlier_reprojection_px": float(np.max(errors[mask])),
                "validated_parallax_deg": inlier_angle,
                "assumed_pixel_jitter_floor": 1.0,
                "depth_sensitivity": depth_sensitivity,
                "centre_bias_included_in_uncertainty": False,
                "reprojection_errors_px": [
                    float(error) if math.isfinite(error) else None for error in errors
                ],
                "calibrated_error_bound": False,
                "object_center_is_rigid_feature": False,
                "world_frame_id": rows[-1]["world_frame_id"],
            }
        )
        return result

    def _apply_multiview_position(
        self,
        obj: SemanticObject,
        multiview: dict,
        pose: CameraPose | None,
    ) -> None:
        evidence = {key: value for key, value in multiview.items() if key != "xyz"}
        obj.metadata["multiview_evidence"] = evidence
        if multiview.get("xyz") is None or pose is None:
            return
        camera_depth = float(pose.inverse().transform_point(multiview["xyz"])[2])
        surface_depth = obj.depth
        if surface_depth is not None:
            ratio = abs(surface_depth - camera_depth) / max(camera_depth, 1e-6)
            evidence["surface_depth"] = surface_depth
            evidence["centre_surface_relative_difference"] = ratio
            if ratio > self.config.max_center_surface_depth_ratio:
                obj.xyz = None
                obj.depth = None
                obj.metadata["position_status"] = "unavailable"
                obj.metadata["evidence"]["reason"] = "object_centre_and_surface_depth_disagree"
                return
        elif (
            obj.metadata.get("evidence", {}).get("reason")
            == "object_centre_and_surface_depth_disagree"
        ):
            # A strong but incompatible background layer must not validate rays.
            obj.xyz = None
            obj.depth = None
            return
        obj.xyz = multiview["xyz"].copy()
        obj.depth = camera_depth
        obj.metadata.update(
            {
                "position_source": "object_center_multiview",
                "position_relation_kind": "object_center",
                "center_bias_unknown": True,
                "depth_source": "object_center_multiview",
                "position_status": "estimated",
                "position_estimate": "multiview_visible_object_centre",
                "position_frame": obj.frame_id,
                "position_uncertainty": multiview["radius"],
            }
        )
        obj.metadata["uncertainty"].update(
            {
                "radius": multiview["radius"],
                "method": "reprojection_and_parallax_sensitivity",
                "assumed_pixel_jitter_floor": 1.0,
                "centre_bias_included": False,
            }
        )
        obj.metadata["evidence"]["object_centre_depth"] = camera_depth

    def _record_snapshot(
        self,
        obj: SemanticObject,
        td: TrackedDetection,
        pose: CameraPose | None,
        intr: CameraIntrinsics,
    ) -> None:
        snapshots = self._position_snapshots.setdefault(obj.track_id, [])
        camera_xyz = (
            pose.inverse().transform_point(obj.xyz).tolist()
            if pose is not None and obj.xyz is not None and obj.metadata["pose_valid"]
            else None
        )
        snapshot = {
            "frame_id": obj.frame_id,
            "timestamp": float(td.detection.timestamp),
            "xyz": obj.xyz.tolist() if obj.xyz is not None else None,
            "camera_xyz": camera_xyz,
            "position_source": obj.metadata.get("position_source"),
            "position_frame": obj.metadata.get("position_frame"),
            "status": obj.metadata.get("position_status"),
            "uncertainty": obj.metadata.get("position_uncertainty"),
            "temporal_consistency": obj.metadata.get("position_temporal_consistency"),
            "position_relation_usable": obj.metadata.get("position_relation_usable", False),
            "position_relation_kind": obj.metadata.get("position_relation_kind"),
            "center_bias_unknown": obj.metadata.get("center_bias_unknown", True),
            "pose_valid": obj.metadata.get("pose_valid", False),
            "pose_trusted": obj.metadata.get("pose_trusted", False),
            "world_frame_id": obj.metadata.get("world_frame_id"),
            "reprojection": deepcopy(obj.metadata.get("position_reprojection")),
            "multiview_evidence": deepcopy(obj.metadata.get("multiview_evidence")),
            "support_evidence": {
                key: deepcopy(value)
                for key, value in obj.metadata.get("evidence", {}).items()
                if key
                in {
                    "sample_count",
                    "observed_point_count",
                    "point_ids",
                    "persistent_support_min_views",
                    "persistent_support_frames",
                    "persistent_support_qualified",
                    "reason",
                }
            },
            "surface_support_evidence": deepcopy(
                {
                    key: value
                    for key, value in obj.metadata.get("evidence", {}).items()
                    if key.startswith("persistent_support")
                    or key in {"point_ids", "sample_count", "depth_relative_spread"}
                }
            ),
            "evidence_reason": obj.metadata.get(
                "latest_position_evidence", obj.metadata.get("evidence", {})
            ).get("reason"),
        }
        snapshots.append(snapshot)
        limit = self.config.max_position_history
        if len(snapshots) > limit:
            indices = np.linspace(0, len(snapshots) - 1, limit, dtype=int)
            snapshots[:] = [snapshots[index] for index in indices]
        obj.metadata["position_snapshots"] = list(snapshots)

    def _update_position(
        self,
        obj: SemanticObject,
        td: TrackedDetection,
        pose: CameraPose | None,
        intr: CameraIntrinsics,
    ) -> SemanticObject:
        """Keep missing/outlier geometry from erasing a valid historical estimate."""
        previous = self._objects.get(obj.track_id)
        samples = self._position_samples.setdefault(obj.track_id, [])
        history = self._position_history.setdefault(obj.track_id, [])
        raw = td.raw_detection or td.detection
        dynamic = _potentially_moving(td.detection) or _potentially_moving(raw)
        if dynamic:
            samples.clear()
        obj.metadata["motion_policy"] = "latest_observation" if dynamic else "static_scene_fusion"
        obj.metadata["motion_policy_basis"] = "stable_and_latest_class_prior_only"
        obj.metadata["latest_raw_dynamic_class"] = (
            raw.class_name if _potentially_moving(raw) else None
        )
        obj.metadata["motion_detection_available"] = False
        obj.metadata["static_scene_assumed"] = not dynamic
        if obj.xyz is None:
            if previous is not None and previous.xyz is not None:
                return self._retain_position(
                    obj,
                    previous,
                    "missing_geometric_evidence",
                    td.detection.timestamp,
                )
            obj.metadata["position_history"] = list(history)
            return obj

        # A sudden static-object jump must not replace a previously stable position.
        # Dynamic categories use current evidence and are never averaged as static.
        previous_consistency = "unverified"
        if previous is not None and previous.xyz is not None:
            previous_consistency, _ = self._reprojection_details(
                previous.xyz,
                obj.frame_id,
                obj.xyxy,
                pose,
                intr,
            )
        if previous_consistency == "contradicted":
            # A new direct observation can recover after motion, pose drift, or
            # changed association. Keep the old history but reset its fusion prior.
            samples.clear()
            obj.metadata["position_reanchored"] = True
            obj.metadata["position_reanchor_reason"] = "historical_reprojection_contradiction"
        if previous is not None and previous.metadata.get("position_source") != obj.metadata.get(
            "position_source"
        ):
            # A visible surface support centre and a box-centre ray are distinct
            # estimates; averaging them would preserve the old surface bias.
            samples.clear()
        if not dynamic and previous is not None and previous.xyz is not None and len(samples) >= 2:
            jump = float(np.linalg.norm(obj.xyz - previous.xyz))
            old_radius = float(previous.metadata.get("position_uncertainty") or 0.0)
            new_radius = float(obj.metadata.get("position_uncertainty") or 0.0)
            depth_scale = min(obj.depth or 0.0, previous.depth or obj.depth or 0.0)
            gate = max(
                3.0 * (old_radius + new_radius), depth_scale * self.config.max_position_jump_ratio
            )
            if jump > max(gate, 1e-6):
                obj.metadata["evidence"]["rejected_jump_distance"] = jump
                obj.metadata["evidence"]["jump_gate"] = gate
                return self._retain_position(
                    obj,
                    previous,
                    "inconsistent_position_jump",
                    td.detection.timestamp,
                )

        sample = {
            "xyz": obj.xyz.copy(),
            "radius": float(obj.metadata.get("position_uncertainty") or 0.0),
            "source": obj.metadata.get("position_source", "none"),
            "frame_id": obj.frame_id,
            "timestamp": td.detection.timestamp,
        }
        samples.append(sample)
        limit = 1 if dynamic else self.config.fusion_window
        del samples[:-limit]
        multiview = obj.metadata.get("position_source") == "object_center_multiview"
        persistent_surface = obj.metadata.get("position_source") == "persistent_surface_multiview"
        if dynamic or multiview or persistent_surface:
            fused = sample["xyz"].copy()
            radius = sample["radius"]
        else:
            positions = np.stack([entry["xyz"] for entry in samples])
            fused = np.median(positions, axis=0)
            residuals = np.linalg.norm(positions - fused, axis=1)
            # Scatter is kept visible; averaging must not invent higher precision.
            radius = max(
                float(np.quantile(residuals, 0.75)),
                float(np.median([entry["radius"] for entry in samples])),
            )
        obj.xyz = fused
        valid_count = int(previous.metadata.get("position_observations", 0)) + 1 if previous else 1
        obj.metadata.update(
            {
                "position_uncertainty": radius,
                "position_observations": valid_count,
                "position_age_frames": 0,
                "position_age_seconds": 0.0,
                "position_fusion": "latest_dynamic_observation"
                if dynamic
                else (
                    "validated_multiview_ray_intersection"
                    if multiview
                    else (
                        "persistent_landmark_surface_support"
                        if persistent_surface
                        else "robust_temporal_median"
                    )
                ),
            }
        )
        obj.metadata["uncertainty"]["radius"] = radius
        obj.metadata["uncertainty"]["method"] = (
            "reprojection_and_parallax_sensitivity"
            if multiview
            else (
                "persistent_landmark_support_scatter"
                if persistent_surface
                else "support_and_temporal_scatter"
            )
        )
        obj.metadata["evidence"]["fusion_sample_count"] = len(samples)
        obj.metadata["evidence"]["fusion_sources"] = sorted({entry["source"] for entry in samples})
        history.append(
            {
                "frame_id": obj.frame_id,
                "timestamp": float(td.detection.timestamp),
                "xyz": fused.tolist(),
                "uncertainty": radius,
                "source": obj.metadata["position_source"],
                "status": obj.metadata["position_status"],
                "world_frame_id": obj.metadata.get("world_frame_id"),
            }
        )
        # Retain the entire temporal extent with bounded viewer payloads.
        history_limit = self.config.max_position_history
        if len(history) > history_limit:
            indices = np.linspace(0, len(history) - 1, history_limit, dtype=int)
            history[:] = [history[index] for index in indices]
        obj.metadata["position_history"] = list(history)
        return obj

    def _reprojection_details(
        self,
        xyz: np.ndarray | None,
        frame_id: int,
        xyxy: tuple[float, float, float, float],
        pose: CameraPose | None,
        intr: CameraIntrinsics,
    ) -> tuple[str, dict]:
        details: dict = {
            "frame_id": frame_id,
            "pixel": None,
            "inside_box": None,
            "inside_image": None,
            "box_margin": self.config.reprojection_box_margin,
        }
        if xyz is None:
            details["reason"] = "no_world_position"
            return "unverified", details
        if pose is None or pose.frame_id not in (-1, frame_id):
            details["reason"] = "missing_current_camera_pose"
            return "unverified", details
        if not np.isfinite(pose.matrix).all() or not np.isfinite(xyz).all():
            details["reason"] = "invalid_geometry"
            return "unverified", details
        camera = pose.inverse().transform_point(xyz)
        depth = float(camera[2])
        details["camera_depth"] = depth
        if depth <= 1e-6:
            details.update(
                {"reason": "behind_current_camera", "inside_box": False, "inside_image": False}
            )
            return "contradicted", details
        if intr.has_distortion:
            projected, _ = cv2.projectPoints(
                camera.reshape(1, 3),
                np.zeros(3),
                np.zeros(3),
                intr.K,
                intr.dist_coeffs,
            )
            u, v = projected.reshape(2)
        else:
            u = intr.fx * camera[0] / depth + intr.cx
            v = intr.fy * camera[1] / depth + intr.cy
        if not np.isfinite([u, v]).all():
            details["reason"] = "nonfinite_projection"
            return "contradicted", details
        x1, y1, x2, y2 = xyxy
        margin_x = max(0.0, x2 - x1) * self.config.reprojection_box_margin
        margin_y = max(0.0, y2 - y1) * self.config.reprojection_box_margin
        inside = x1 - margin_x <= u <= x2 + margin_x and y1 - margin_y <= v <= y2 + margin_y
        in_image = 0 <= u < intr.width and 0 <= v < intr.height
        details.update(
            {
                "pixel": [float(u), float(v)],
                "inside_box": bool(inside),
                "inside_image": bool(in_image),
                "expanded_box": [x1 - margin_x, y1 - margin_y, x2 + margin_x, y2 + margin_y],
                "reason": "consistent"
                if inside and in_image
                else ("outside_current_image" if not in_image else "outside_current_detection_box"),
            }
        )
        return ("consistent" if inside and in_image else "contradicted"), details

    def _check_reprojection(
        self,
        obj: SemanticObject,
        pose: CameraPose | None,
        intr: CameraIntrinsics,
    ) -> None:
        consistency, details = self._reprojection_details(
            obj.xyz,
            obj.frame_id,
            obj.xyxy,
            pose,
            intr,
        )
        obj.metadata["position_temporal_consistency"] = consistency
        obj.metadata["position_reprojection"] = details
        if consistency == "contradicted":
            obj.metadata["position_status"] = "inconsistent"
            latest_evidence = dict(obj.metadata.get("latest_position_evidence", {}))
            latest_evidence.update(
                {
                    "reason": "historical_position_reprojection_contradicted",
                    "possible_causes": [
                        "camera_pose_drift",
                        "object_motion",
                        "track_association_error",
                    ],
                    "cause_distinguishable": False,
                }
            )
            obj.metadata["latest_position_evidence"] = latest_evidence
        history = self._position_history.get(obj.track_id, [])
        if history and history[-1]["frame_id"] == obj.frame_id:
            history[-1]["position_temporal_consistency"] = consistency
            history[-1]["status"] = obj.metadata["position_status"]
            obj.metadata["position_history"] = list(history)

    def _retain_position(
        self,
        obj: SemanticObject,
        previous: SemanticObject,
        reason: str,
        timestamp: float,
    ) -> SemanticObject:
        latest_evidence = obj.metadata.get("evidence", {})
        latest_multiview = obj.metadata.get("multiview_evidence", {})
        motion_policy = obj.metadata["motion_policy"]
        motion_policy_basis = obj.metadata["motion_policy_basis"]
        latest_raw_dynamic_class = obj.metadata.get("latest_raw_dynamic_class")
        static_scene_assumed = obj.metadata["static_scene_assumed"]
        obj.xyz = previous.xyz.copy()
        obj.depth = previous.depth
        obj.metadata = {**previous.metadata}
        obj.metadata.update(
            {
                "position_status": "retained",
                "position_retained_reason": reason,
                "latest_position_evidence": latest_evidence,
                "multiview_evidence": latest_multiview,
                "motion_policy": motion_policy,
                "motion_policy_basis": motion_policy_basis,
                "latest_raw_dynamic_class": latest_raw_dynamic_class,
                "static_scene_assumed": static_scene_assumed,
                "position_age_frames": max(
                    0, obj.frame_id - int(previous.metadata["position_frame"])
                ),
            }
        )
        history = obj.metadata.get("position_history", [])
        if history:
            obj.metadata["position_age_seconds"] = max(
                0.0,
                timestamp - history[-1]["timestamp"],
            )
        if reason == "inconsistent_position_jump":
            obj.metadata["position_rejected_count"] = (
                int(previous.metadata.get("position_rejected_count", 0)) + 1
            )
        return obj


# These classes may move. Class membership is a conservative policy, not motion
# detection; stationary instances are still kept at their latest measured position.
_POTENTIALLY_MOVING_CLASSES = frozenset(
    {
        "person",
        "bicycle",
        "car",
        "motorcycle",
        "airplane",
        "bus",
        "train",
        "truck",
        "boat",
        "bird",
        "cat",
        "dog",
        "horse",
        "sheep",
        "cow",
        "elephant",
        "bear",
        "zebra",
        "giraffe",
    }
)


def _potentially_moving(detection: Detection) -> bool:
    """The latest raw class can veto a static temporal label, never vice versa."""
    return (
        detection.class_name.lower() in _POTENTIALLY_MOVING_CLASSES
        or str(detection.metadata.get("latest_raw_class", "")).lower()
        in _POTENTIALLY_MOVING_CLASSES
    )


def create_detector(name: str, **kwargs) -> ObjectDetector:
    """Factory — CLI never imports concrete detectors."""
    key = (name or "demo").strip().lower()
    if key in ("demo", "demo_detector", "blob"):
        from visualmap.detection.demo import DemoDetector, DemoDetectorConfig

        allowed = {"min_area", "conf_threshold", "class_id", "class_name"}
        return DemoDetector(DemoDetectorConfig(**{k: v for k, v in kwargs.items() if k in allowed}))
    if key in ("yolo_onnx", "yolo", "onnx"):
        from visualmap.detection.yolo_onnx import YOLOONNXConfig, YOLOONNXDetector

        if "model_path" not in kwargs:
            raise ValueError("yolo_onnx requires model_path=")
        allowed = {
            "model_path",
            "conf_threshold",
            "nms_threshold",
            "input_width",
            "input_height",
            "class_names",
        }
        return YOLOONNXDetector(YOLOONNXConfig(**{k: v for k, v in kwargs.items() if k in allowed}))
    if key in ("yolox_onnx", "yolox"):
        from visualmap.detection.yolox_onnx import YOLOXONNXConfig, YOLOXONNXDetector

        if "model_path" not in kwargs:
            raise ValueError("yolox_onnx requires model_path=")
        allowed = {
            "model_path",
            "conf_threshold",
            "nms_threshold",
            "input_width",
            "input_height",
            "class_names",
            "pegboard_refinement",
            "pegboard_config",
        }
        return YOLOXONNXDetector(
            YOLOXONNXConfig(**{k: v for k, v in kwargs.items() if k in allowed})
        )
    if key in ("none", "off", "disabled"):
        from visualmap.detection.demo import DemoDetector, DemoDetectorConfig

        # Empty detector: no detections
        class _NoneDetector(ObjectDetector):
            name = "none"

            def detect(self, frame):
                return []

        return _NoneDetector()
    raise ValueError(f"unknown detector: {name!r}")
