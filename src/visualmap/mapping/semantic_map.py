"""Semantic map: world-coordinate object list + map points (V0.6)."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field

import numpy as np

from visualmap.detection.types import SemanticObject, SemanticResult
from visualmap.mapping.occupancy import attach_occupancy
from visualmap.slam.coverage import geometry_coverage, landmark_world
from visualmap.slam.types import MapPoint, MapResult
from visualmap.utils.logging import get_logger

logger = get_logger(__name__)

__all__ = ["SemanticMap", "build_semantic_map", "observations_from_records"]


@dataclass
class SemanticMap:
    """Everything a downstream UI needs: trajectory, points, named objects."""

    objects: list[SemanticObject] = field(default_factory=list)
    map_points: list[MapPoint] = field(default_factory=list)
    trajectory_xyz: list[list[float]] = field(default_factory=list)
    trajectory_timestamps: list[float] = field(default_factory=list)
    trajectory_frame_ids: list[int] = field(default_factory=list)
    trajectory_rotations: list[list[list[float]]] = field(default_factory=list)
    trajectory_world_frame_ids: list[str | None] = field(default_factory=list)
    object_observations: list[dict] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    map_point_world_frame_ids: list[str | None] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.objects)

    def summary(self) -> dict:
        by_class: dict[str, int] = {}
        for obj in self.objects:
            by_class[obj.class_name] = by_class.get(obj.class_name, 0) + 1
        return {
            "num_objects": len(self.objects),
            "num_map_points": len(self.map_points),
            "num_trajectory_poses": len(self.trajectory_xyz),
            "num_positioned_objects": sum(obj.xyz is not None for obj in self.objects),
            "num_unpositioned_objects": sum(obj.xyz is None for obj in self.objects),
            "num_retained_positions": sum(
                obj.metadata.get("position_status") == "retained" for obj in self.objects
            ),
            "num_inconsistent_positions": sum(
                obj.metadata.get("position_temporal_consistency") == "contradicted"
                for obj in self.objects
            ),
            "num_relation_usable_positions": sum(
                bool(obj.metadata.get("position_relation_usable")) for obj in self.objects
            ),
            "num_object_observations": len(self.object_observations),
            "by_class": by_class,
            **self.metadata,
        }

    def to_dict(self) -> dict:
        return {
            "summary": self.summary(),
            "objects": [o.to_dict() for o in self.objects],
            "map_points": [{"id": mp.id, "xyz": mp.xyz.tolist()} for mp in self.map_points],
            "map_point_world_frame_ids": self.map_point_world_frame_ids,
            "trajectory": self.trajectory_xyz,
            "trajectory_timestamps": self.trajectory_timestamps,
            "trajectory_frame_ids": self.trajectory_frame_ids,
            "trajectory_rotations": self.trajectory_rotations,
            "trajectory_world_frame_ids": self.trajectory_world_frame_ids,
            "object_observations": self.object_observations,
            "metadata": self.metadata,
        }


def observations_from_records(objects: list[SemanticObject], records: list[dict]) -> list[dict]:
    """Reattach exact same-frame geometry to live or saved tracking records.

    Stored tracks contain classification, while position snapshots belong to
    their semantic object. Rebuilding presentation must join these two sources
    without substituting a final/future coordinate or dropping quality gates.
    """
    object_ids = {obj.track_id for obj in objects}
    observations = [deepcopy(row) for row in records if row.get("track_id") in object_ids]
    observations.sort(key=lambda row: (row["frame_id"], row["track_id"]))
    snapshots = {
        (obj.track_id, entry["frame_id"]): entry
        for obj in objects
        for entry in obj.metadata.get("position_snapshots", [])
    }
    for observation in observations:
        snapshot = snapshots.get((observation["track_id"], observation["frame_id"]))
        observation["geometry"] = (
            deepcopy(snapshot)
            if snapshot
            else {
                "xyz": None,
                "position_relation_usable": False,
                "pose_valid": False,
                "temporal_consistency": "unverified",
                "reason": "no_same_frame_geometry_snapshot",
                "world_frame_id": None,
            }
        )
    return observations


def build_semantic_map(
    map_result: MapResult,
    semantics: SemanticResult | None = None,
    max_points: int = 4000,
    observation_records: list[dict] | None = None,
    occupancy_thickness_ratio: float | None = None,
    occupancy_multiview_options: dict | None = None,
    preserve_saved_occupancy: bool = False,
) -> SemanticMap:
    if semantics is not None and not preserve_saved_occupancy:
        # Occupancy uses the complete reconstruction before viewer point sampling.
        ratio = occupancy_thickness_ratio
        if ratio is None:
            ratio = semantics.metadata.get("occupancy_policy", {}).get(
                "inferred_thickness_ratio", 0.5
            )
        previous_policy = semantics.metadata.get("occupancy_policy", {})
        multiview_options = {
            key: previous_policy[key]
            for key in (
                "multiview_enabled",
                "multiview_max_views",
                "multiview_min_angle_deg",
                "multiview_bbox_margin",
            )
            if key in previous_policy
        }
        multiview_options.update(occupancy_multiview_options or {})
        attach_occupancy(
            map_result,
            semantics,
            observation_records=observation_records,
            inferred_thickness_ratio=ratio,
            **multiview_options,
        )
    objects = list(semantics.objects) if semantics else []
    points = list(map_result.map_points)
    if len(points) > max_points:
        idx = np.linspace(0, len(points) - 1, max_points, dtype=int)
        points = [points[i] for i in idx]

    traj = map_result.trajectory.positions().tolist()
    object_ids = {obj.track_id for obj in objects}
    observations = (
        observation_records
        if observation_records is not None
        else (
            [tracked.to_dict() for tracked in semantics.tracks if tracked.track_id in object_ids]
            if semantics
            else []
        )
    )
    observations = observations_from_records(objects, observations)
    frame_quality = map_result.metadata.get("frame_quality", {})
    return SemanticMap(
        objects=objects,
        map_points=points,
        map_point_world_frame_ids=[landmark_world(map_result, point) for point in points],
        trajectory_xyz=traj,
        trajectory_timestamps=[pose.timestamp for pose in map_result.trajectory],
        trajectory_frame_ids=[pose.frame_id for pose in map_result.trajectory],
        trajectory_rotations=[pose.R.tolist() for pose in map_result.trajectory],
        trajectory_world_frame_ids=[
            frame_quality.get(str(pose.frame_id), frame_quality.get(pose.frame_id, {})).get(
                "world_frame_id", map_result.metadata.get("world_frame_id")
            )
            for pose in map_result.trajectory
        ],
        object_observations=observations,
        metadata={
            "units": "map_units",
            "pose_convention": "T_world<-camera",
            "stage": "V0.6-semantic-map",
            "detector": semantics.metadata.get("detector", "unknown") if semantics else "none",
            "intrinsics": map_result.intrinsics.to_dict() if map_result.intrinsics else None,
            "track_rate": map_result.metadata.get("track_rate"),
            "frame_quality": frame_quality,
            "slam_backend": map_result.metadata.get(
                "backend", map_result.metadata.get("slam_backend")
            ),
            "world_frame_id": map_result.metadata.get("world_frame_id"),
            "scale_consistent": map_result.metadata.get("scale_consistent") is True,
            "scale": map_result.metadata.get("scale", "monocular_arbitrary"),
            "scale_consistency_scope": map_result.metadata.get("scale_consistency_scope"),
            "global_scale_consistent": map_result.metadata.get("global_scale_consistent"),
            "submaps": map_result.metadata.get("submaps", []),
            "geometry_coverage": geometry_coverage(map_result),
            "video_ingestion": map_result.metadata.get("video_ingestion"),
            "world_axes": "opencv_y_down",
            "video": map_result.metadata.get("video"),
            "position_accuracy_validated": False,
            "position_policy": semantics.metadata.get("position_policy") if semantics else None,
            "motion_detection_available": False,
            "motion_policy_basis": "class_prior_only",
            "motion_policy_limitation": "static_category_instances_can_also_move",
            "occupancy_policy": "bounded_frustum_with_past_trusted_bbox_cones",
            "occupancy_volume_verified": False,
            "free_space_certified": False,
        },
    )
