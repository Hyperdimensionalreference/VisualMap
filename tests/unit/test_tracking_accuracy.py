"""Video failure cases: class flicker, duplicate labels, motion and assignment."""

from __future__ import annotations

from itertools import permutations

import numpy as np
import pytest

from visualmap.detection.tracker import IoUTracker, TrackerConfig, _maximum_weight_matching
from visualmap.detection.types import Detection


def _detection(frame, box=(0, 0, 20, 20), class_id=39, name="bottle", confidence=0.9):
    return Detection(frame, frame / 30, class_id, name, confidence, box)


def test_one_frame_wrong_class_keeps_id_and_raw_evidence():
    tracker = IoUTracker()
    first = tracker.update([_detection(0)])[0]
    tracker.update([_detection(1)])
    wrong = _detection(2, class_id=41, name="cup", confidence=0.95)
    stable = tracker.update([wrong])[0]
    assert stable.track_id == first.track_id
    assert stable.class_name == "bottle"
    assert stable.raw_detection is wrong
    assert stable.detection.confidence == pytest.approx(0.9)
    assert stable.class_consistency == pytest.approx(1.8 / 2.75)
    assert stable.class_observations == 2
    assert stable.to_dict()["raw_class_name"] == "cup"
    assert stable.detection.metadata["classification_needs_review"] is True
    assert stable.to_dict()["classification_evidence"]["temporal_class_disagreement"] is True
    assert tracker.update([_detection(3)])[0].track_id == first.track_id


def test_persistent_corrected_class_can_replace_incorrect_first_frame():
    tracker = IoUTracker()
    first = tracker.update([_detection(0, class_id=41, name="cup", confidence=0.5)])[0]
    for frame in (1, 2):
        result = tracker.update([_detection(frame)])[0]
        assert result.class_name == "cup"  # require repeated supporting observations
    corrected = tracker.update([_detection(3)])[0]
    assert corrected.class_name == "bottle"
    assert corrected.track_id == first.track_id
    assert corrected.class_observations == 3


def test_window_can_adapt_after_a_long_wrong_class_history():
    tracker = IoUTracker(TrackerConfig(class_vote_window=4, class_switch_hits=3))
    for frame in range(20):
        tracker.update([_detection(frame, class_id=41, name="cup")])
    for frame in (20, 21, 22):
        tracked = tracker.update([_detection(frame)])[0]
    assert tracked.class_name == "bottle"
    assert tracked.hits == 23
    assert len(tracker.tracks[0].history) == 4


@pytest.mark.parametrize("first,second", [("bottle", "cup"), ("laptop", "tv")])
def test_nearly_identical_conflicting_labels_make_one_track_with_alternatives(first, second):
    tracker = IoUTracker()
    strongest = _detection(0, class_id=0, name=first, confidence=0.75)
    alternative = _detection(0, (0.5, 0.5, 20.5, 20.5), 1, second, 0.6)
    tracked = tracker.update([alternative, strongest])
    assert len(tracked) == 1
    assert tracked[0].raw_detection is strongest
    assert tracked[0].class_alternatives == (alternative,)
    assert tracked[0].to_dict()["class_alternatives"][0]["class_name"] == second


def test_duplicate_filter_preserves_real_overlapping_objects_and_can_be_disabled():
    bottle = _detection(0)
    cup = _detection(0, (5, 0, 25, 20), class_id=41, name="cup")
    assert len(IoUTracker().update([bottle, cup])) == 2
    person = _detection(0, class_id=0, name="person")
    chair = _detection(0, class_id=56, name="chair")
    assert len(IoUTracker().update([person, chair])) == 2
    cup_same_box = _detection(0, class_id=41, name="cup")
    tracker = IoUTracker(TrackerConfig(ambiguity_groups=()))
    assert len(tracker.update([bottle, cup_same_box])) == 2


def test_cross_class_match_requires_strong_geometric_agreement():
    tracker = IoUTracker()
    original = tracker.update([_detection(0)])[0]
    other = tracker.update([_detection(1, (8, 0, 28, 20), class_id=41, name="cup")])[0]
    assert other.track_id != original.track_id


def test_motion_prediction_reconnects_sampled_frames_after_missed_observation():
    tracker = IoUTracker(TrackerConfig(max_age=2))
    first = tracker.update([_detection(0)])[0]
    # Eight source frames between samples, but one tracker update each time.
    assert tracker.update([_detection(8, (8, 0, 28, 20))])[0].track_id == first.track_id
    assert tracker.update([]) == []
    reappeared = tracker.update([_detection(24, (24, 0, 44, 20))])[0]
    assert reappeared.track_id == first.track_id
    assert reappeared.hits == 3


def test_stationary_object_after_motion_keeps_its_id():
    tracker = IoUTracker()
    first = tracker.update([_detection(0)])[0]
    tracker.update([_detection(1, (8, 0, 28, 20))])
    stopped = tracker.update([_detection(2, (8, 0, 28, 20))])[0]
    assert stopped.track_id == first.track_id


def test_motion_keeps_two_same_class_ids_when_paths_cross():
    tracker = IoUTracker()
    first = tracker.update([
        _detection(0, (0, 0, 20, 20)),
        _detection(0, (28, 0, 48, 20)),
    ])
    left_id, right_id = first[0].track_id, first[1].track_id
    tracker.update([
        _detection(1, (8, 0, 28, 20)),
        _detection(1, (20, 0, 40, 20)),
    ])
    crossed = tracker.update([
        _detection(2, (12, 0, 32, 20)),
        _detection(2, (16, 0, 36, 20)),
    ])
    positions = {tracked.track_id: tracked.detection.xyxy[0] for tracked in crossed}
    assert positions[left_id] == 16
    assert positions[right_id] == 12


def test_global_assignment_recovers_pair_lost_by_greedy_association():
    tracker = IoUTracker(TrackerConfig(motion_prediction=False))
    original = tracker.update([
        _detection(0, (0, 0, 20, 20)),
        _detection(0, (10, 0, 30, 20)),
    ])
    # First track prefers x=1, but the second track can only reach x=1.
    # Assigning first→x=-8 and second→x=1 retains both identities.
    tracked = tracker.update([
        _detection(1, (1, 0, 21, 20)),
        _detection(1, (-8, 0, 12, 20)),
    ])
    assert len(tracked) == 2
    assert {item.track_id for item in tracked} == {item.track_id for item in original}
    assert tracked[0].detection.xyxy[0] == -8
    assert tracked[1].detection.xyxy[0] == 1


def test_assignment_matches_brute_force_for_small_random_rectangles():
    random = np.random.default_rng(2026)
    for rows, columns in ((2, 3), (3, 2), (3, 3)):
        for _ in range(10):
            weights = random.uniform(0, 1, (rows, columns))
            weights[random.random((rows, columns)) < 0.35] = -1
            padded = np.concatenate([weights, np.zeros((rows, rows))], axis=1)
            best = (-1, -1.0)
            for choices in permutations(range(columns + rows), rows):
                selected = [(row, col) for row, col in enumerate(choices) if col < columns]
                if any(padded[row, col] < 0 for row, col in selected):
                    continue
                score = (len(selected), sum(weights[row, col] for row, col in selected))
                best = max(best, score)
            actual = _maximum_weight_matching(weights)
            assert len(actual) == best[0]
            assert sum(weights[row, col] for row, col in actual) == pytest.approx(best[1])


@pytest.mark.parametrize("box,confidence", [
    ((0, 0, np.nan, 20), 0.9),
    ((0, 0, np.inf, 20), 0.9),
    ((0, 0, 0, 20), 0.9),
    ((0, 0, 20, 20), np.nan),
    ((0, 0, 20, 20), 1.1),
])
def test_invalid_model_detections_do_not_create_tracks(box, confidence):
    tracker = IoUTracker()
    assert tracker.update([_detection(0, box=box, confidence=confidence)]) == []
    assert tracker.tracks == []


def test_tracker_validates_temporal_evidence_configuration():
    with pytest.raises(ValueError, match="class_switch_hits"):
        TrackerConfig(class_vote_window=2, class_switch_hits=3)
    with pytest.raises(ValueError, match="duplicate_iou"):
        TrackerConfig(duplicate_iou=0)


def test_rule_and_model_never_share_identity_or_provenance_even_at_identical_boxes():
    tracker = IoUTracker()
    model = _detection(0)
    model_track = tracker.update([model])[0]
    structure = Detection(
        1, 1 / 30, 39, "bottle", 0.8, model.xyxy,
        metadata={
            "classification_source": "image_structure_rule", "classification_needs_review": True
        },
    )
    proposal = tracker.update([structure])[0]
    assert proposal.track_id != model_track.track_id
    assert proposal.detection.metadata["classification_source"] == "image_structure_rule"
    resumed = tracker.update([_detection(2)])[0]
    assert resumed.track_id == model_track.track_id
    assert "classification_source" not in resumed.detection.metadata


def test_duplicate_suppression_cannot_hide_a_model_behind_a_rule_score():
    bottle = _detection(0, confidence=0.6)
    structure = Detection(
        0, 0, 41, "cup", 0.9, bottle.xyxy,
        metadata={"classification_source": "image_structure_rule"},
    )
    results = IoUTracker().update([bottle, structure])
    assert len(results) == 2
    assert {row.class_name for row in results} == {"cup", "bottle"}
    assert all(not row.class_alternatives for row in results)
