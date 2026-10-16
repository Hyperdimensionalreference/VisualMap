"""Structure safety contracts using simulated patterns and one reviewed video.

Synthetic patterns verify abstention and provenance, not real recognition accuracy.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from visualmap.config.settings import load_config
from visualmap.detection.pegboard import PegboardRefiner
from visualmap.detection.pipeline import SemanticPipeline
from visualmap.detection.tracker import IoUTracker
from visualmap.detection.types import Detection
from visualmap.slam.types import CameraIntrinsics, CameraPose, Frame, MapResult, Trajectory


def _panel():
    image = np.full((480, 640, 3), 190, np.uint8)
    for row in range(14):
        for column in range(15):
            cv2.circle(image, (140 + column * 24, 65 + row * 24), 3, (20, 20, 20), -1)
    return image


def _chair(frame_id=0, xyxy=(125, 50, 495, 425)):
    return Detection(frame_id, frame_id / 30, 56, "chair", 0.55, xyxy)


def test_hole_lattice_keeps_model_object_and_adds_reviewable_structure():
    refiner = PegboardRefiner()
    detections = refiner.refine(Frame(0, 0, _panel()), [_chair()])
    assert len(detections) == 2
    original = next(d for d in detections if d.class_name == "chair")
    assert original.class_id == _chair().class_id
    assert original.xyxy == _chair().xyxy
    assert original.confidence == _chair().confidence
    assert original.metadata["classification_structural_conflict"] is True
    board = next(d for d in detections if d.class_name == "perforated_panel")
    assert board.class_id != 56
    assert board.xyxy != _chair().xyxy
    evidence = board.metadata
    assert evidence["classification_source"] == "image_structure_rule"
    assert evidence["raw_model_class"] is None
    assert evidence["raw_model_detections"][0]["xyxy"] == list(_chair().xyxy)
    assert evidence["structure_evidence"]["hole_count"] >= 150
    assert evidence["structure_evidence"]["rows"] >= 10
    assert evidence["structure_evidence"]["columns"] >= 10
    assert evidence["trained_pegboard_model"] is False
    assert evidence["user_confirmed"] is False
    assert evidence["full_object_boundary_verified"] is False
    assert evidence["rule_score_is_probability"] is False
    assert evidence["classification_needs_review"] is True
    assert evidence["class_semantics_validated"] is False
    assert "pegboard" in evidence["possible_object_types"]
    assert "displayed_pattern" in evidence["possible_object_types"]


def test_board_can_be_proposed_without_yolox_chair_or_any_other_model_detection():
    detections = PegboardRefiner().refine(Frame(7, 7 / 30, _panel()), [])
    assert [d.class_name for d in detections] == ["perforated_panel"]
    assert detections[0].metadata["raw_model_detections"] == []
    assert detections[0].frame_id == 7


def test_projective_perspective_grid_is_not_restricted_to_front_view():
    image = _panel()
    H = cv2.getPerspectiveTransform(
        np.float32([[0, 0], [640, 0], [640, 480], [0, 480]]),
        np.float32([[50, 0], [590, 40], [640, 465], [10, 430]]),
    )
    warped = cv2.warpPerspective(image, H, (640, 480), borderValue=(190, 190, 190))
    candidates = PegboardRefiner().candidates(warped)
    assert candidates
    assert candidates[0].evidence["hole_count"] >= 120
    assert candidates[0].evidence["median_lattice_error_pitch"] < 0.1


def test_ordinary_chair_slats_do_not_provide_round_hole_grid_evidence():
    image = np.full((480, 640, 3), 210, np.uint8)
    cv2.rectangle(image, (135, 80), (500, 340), (70, 70, 70), -1)
    for x in range(155, 490, 35):
        cv2.rectangle(image, (x, 100), (x + 15, 320), (200, 200, 200), -1)
    cv2.rectangle(image, (155, 340), (175, 465), (70, 70, 70), -1)
    cv2.rectangle(image, (460, 340), (480, 465), (70, 70, 70), -1)
    original = _chair()
    assert PegboardRefiner().refine(Frame(0, 0, image), [original]) == [original]


def test_enclosed_perforated_chair_back_is_ambiguous_and_does_not_replace_chair():
    image = _panel()
    # Same holes, but on a distinct foreground body with a closed outline.
    cv2.rectangle(image, (120, 45), (500, 410), (75, 75, 75), -1)
    for row in range(14):
        for column in range(15):
            cv2.circle(image, (140 + column * 24, 65 + row * 24), 3, (0, 0, 0), -1)
    original = _chair(xyxy=(110, 35, 510, 465))
    result = PegboardRefiner().refine(Frame(0, 0, image), [original])
    preserved = next(d for d in result if d.class_name == "chair")
    assert preserved.xyxy == original.xyxy
    assert preserved.class_id == original.class_id
    assert preserved.confidence == original.confidence
    assert all(d.metadata["classification_needs_review"] for d in result if d is not original)


def test_screen_box_in_front_of_a_valid_board_cannot_be_globally_relabelled():
    image = _panel()
    cv2.rectangle(image, (430, 285), (575, 460), (15, 15, 15), -1)
    screen_guess = _chair(xyxy=(435, 290, 570, 455))
    result = PegboardRefiner().refine(Frame(0, 0, image), [screen_guess])
    preserved = next(d for d in result if d.class_name == "chair")
    assert preserved.xyxy == screen_guess.xyxy
    assert preserved == screen_guess
    assert not preserved.metadata.get("classification_needs_review", False)
    assert any(d.class_name == "perforated_panel" for d in result)
    assert all(
        not d.metadata.get("raw_model_detections")
        for d in result if d.class_name == "perforated_panel"
    )


@pytest.mark.parametrize("kind", ["random_dots", "single_row", "screen_icons", "keyboard_text"])
def test_non_panel_texture_is_rejected(kind):
    image = np.full((480, 640, 3), 190, np.uint8)
    rng = np.random.default_rng(39)
    if kind == "random_dots":
        for x, y in rng.integers([100, 40], [540, 430], (200, 2)):
            cv2.circle(image, (int(x), int(y)), 3, (20, 20, 20), -1)
    elif kind == "single_row":
        for x in range(40, 620, 18):
            cv2.circle(image, (x, 200), 3, (20, 20, 20), -1)
    elif kind == "screen_icons":
        for y in range(90, 300, 50):
            for x in range(120, 510, 65):
                cv2.rectangle(image, (x, y), (x + 18, y + 18), (20, 20, 20), -1)
    else:
        for y in range(90, 360, 27):
            cv2.putText(
                image, "OOOOOOOOOOOO", (140, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (20, 20, 20), 2
            )
    assert not PegboardRefiner().candidates(image)


def test_structural_provenance_survives_tracking_and_semantic_output():
    detection = PegboardRefiner().refine(Frame(0, 0, _panel()), [_chair()])[1]
    track = IoUTracker().update([detection])[0]
    assert track.detection.metadata["classification_source"] == "image_structure_rule"
    assert track.to_dict()["detection_evidence"]["raw_model_class"] is None
    intr = CameraIntrinsics(500, 500, 320, 240, 640, 480)
    reconstruction = MapResult(Trajectory([CameraPose.identity(0)]), [], intr)
    pipeline = SemanticPipeline()
    pipeline.process_detections([detection], reconstruction)
    obj = pipeline.result().objects[0]
    assert obj.class_name == "perforated_panel"
    assert obj.metadata["classification_evidence"]["source"] == "image_structure_rule"
    assert obj.metadata["classification_evidence"]["raw_model_class"] is None
    assert obj.metadata["classification_evidence"]["accuracy_validated"] is False
    assert obj.xyz is None


def test_spatial_config_enables_separate_structural_refinement():
    cfg = load_config(Path("configs/spatial_consistent.yaml"))
    assert cfg.detection.pegboard_refinement is True
    assert cfg.detection.pegboard_config["min_seed_holes"] == 25




@pytest.mark.parametrize("confidence", [0.4, 0.95])
def test_a_different_surface_elsewhere_cannot_change_an_unrelated_chair(confidence):
    original = Detection(0, 0, 56, "chair", confidence, (10, 10, 80, 80))
    result = PegboardRefiner().refine(Frame(0, 0, _panel()), [original])
    assert result[0] is original


@pytest.mark.parametrize("kind", ["plain", "dark", "blur", "small", "occluded", "distorted"])
def test_degraded_simulated_patterns_preserve_predictions_and_never_claim_object_type(kind):
    image = _panel()
    if kind == "dark":
        image = np.uint8(image.astype(float) * 0.3)
    elif kind == "blur":
        image = cv2.GaussianBlur(image, (11, 11), 3)
    elif kind == "small":
        image = cv2.resize(image, (320, 240))
    elif kind == "occluded":
        cv2.rectangle(image, (245, 90), (380, 360), (50, 50, 50), -1)
    elif kind == "distorted":
        yy, xx = np.mgrid[:480, :640].astype(np.float32)
        rx, ry = (xx - 320) / 320, (yy - 240) / 240
        r2 = rx**2 + ry**2
        image = cv2.remap(image, xx + 10 * rx * r2, yy + 10 * ry * r2, cv2.INTER_LINEAR)
    height, width = image.shape[:2]
    original = _chair(xyxy=(0.1 * width, 0.1 * height, 0.9 * width, 0.9 * height))
    result = PegboardRefiner().refine(Frame(0, 0, image), [original])
    kept = next(d for d in result if d.class_id == original.class_id)
    assert kept.class_name == original.class_name
    assert kept.xyxy == original.xyxy
    assert kept.confidence == original.confidence
    for proposal in result:
        if proposal.metadata.get("classification_source") == "image_structure_rule":
            assert proposal.class_name == "perforated_panel"
            assert proposal.metadata["classification_needs_review"] is True
            assert proposal.metadata["class_semantics_validated"] is False
            assert proposal.metadata["physical_object_membership_verified"] is False
