"""Retain single-world counterevidence while isolating independent submaps."""

from visualmap.detection import SemanticPipeline
from visualmap.slam.types import MapPoint, MapResult, Trajectory


def test_legacy_global_world_keeps_observed_surface_and_background_evidence():
    points = [MapPoint(0, [0, 0, 6], observations={0: 0, 1: 0}),
              MapPoint(1, [0, 0, 30], observations={1: 1, 2: 1})]
    mapping = MapResult(
        Trajectory(), points,
        metadata={"world_frame_id": "single-world", "frame_quality": {
            str(frame): {"pose_valid": True, "status": "tracked"} for frame in range(3)
        }},
    )
    assert SemanticPipeline._points_in_world(mapping, "single-world") == points
    assert SemanticPipeline._points_in_world(mapping, "other-world") == []


def test_missing_world_in_segmented_observation_cannot_inherit_latest_global_world():
    mapping = MapResult(
        Trajectory(),
        [MapPoint(0, [0, 0, 6], observations={0: 0, 1: 0}),
         MapPoint(1, [0, 0, 6], observations={1: 1, 2: 1}),
         MapPoint(2, [0, 0, 6], observations={2: 2, 3: 2})],
        metadata={
            "world_frame_id": "latest", "global_scale_consistent": False,
            "frame_quality": {"0": {"world_frame_id": "earlier"}, "1": {},
                              "2": {"world_frame_id": "latest"},
                              "3": {"world_frame_id": "latest"}},
        },
    )
    assert [point.id for point in SemanticPipeline._points_in_world(mapping, "latest")] == [2]
    assert SemanticPipeline._points_in_world(mapping, "earlier") == []


def test_explicitly_unavailable_landmark_world_cannot_be_guessed_from_observations():
    mapping = MapResult(
        Trajectory(), [MapPoint(0, [0, 0, 6], observations={0: 0})],
        metadata={"world_frame_id": "world", "landmark_world_frame_ids": {"0": None},
                  "frame_quality": {"0": {"world_frame_id": "world"}}},
    )
    assert SemanticPipeline._points_in_world(mapping, "world") == []


def test_segmented_map_without_frame_quality_only_uses_explicit_landmark_world_labels():
    mapping = MapResult(
        Trajectory(), [MapPoint(0, [0, 0, 3]), MapPoint(1, [0, 0, 8]), MapPoint(2, [0, 0, 5])],
        metadata={"submaps": [{"world_frame_id": "a"}, {"world_frame_id": "b"}],
                  "landmark_world_frame_ids": {"0": "a", "1": "b"}},
    )
    assert [point.id for point in SemanticPipeline._points_in_world(mapping, "b")] == [1]
    assert SemanticPipeline._points_in_world(mapping, None) == []
