"""Model-independent contracts for expanded vocabulary and tiled observations."""

from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

from visualmap.detection.base import ObjectDetector
from visualmap.detection.configured import CombinedDetector, TiledDetector, merge_detections
from visualmap.detection.crop_verifier import CLIPCropConfig, CLIPCropVerifier, clip_image_blob
from visualmap.detection.grounding_dino import GroundingDINOConfig, GroundingDINODetector
from visualmap.detection.open_vocabulary import OpenVocabularyConfig, load_vocabulary
from visualmap.detection.pipeline import SemanticPipeline
from visualmap.detection.tracker import IoUTracker, TrackerConfig
from visualmap.detection.types import Detection
from visualmap.slam.types import CameraPose, Frame, MapPoint, MapResult, Trajectory


class FakeDetector(ObjectDetector):
    name = "fake"

    def __init__(self, rows, names=("person", "mouse")):
        self.rows, self.calls = rows, []
        self.config = SimpleNamespace(class_names=names)
        self.capabilities = {"class_names": list(names)}

    def detect(self, frame):
        self.calls.append(frame)
        return self.rows(frame, len(self.calls))


def det(frame, name="mouse", cid=0, box=(10, 10, 30, 30), confidence=0.8):
    return Detection(frame.index, frame.timestamp, cid, name, confidence, box)


def test_combination_prevents_class_id_collision_and_reuses_identical_names():
    frame = Frame(0, 0, np.zeros((100, 200, 3), np.uint8))
    baseline = FakeDetector(lambda f, _: [det(f, "person", 0), det(f, "mouse", 1)])
    supplement = FakeDetector(
        lambda f, _: [det(f, "earbud case", 0), det(f, "mouse", 1, confidence=0.9)],
        ("earbud case", "mouse"),
    )
    combined = CombinedDetector(baseline, supplement)
    rows = combined.detect(frame)
    assert len(rows) == 3
    ids = {row.class_name: row.class_id for row in rows}
    assert ids == {"person": 0, "mouse": 1, "earbud case": 2}
    assert next(row for row in rows if row.class_name == "mouse").confidence == 0.9


def test_supplement_cadence_never_replays_old_box_as_current_observation():
    baseline = FakeDetector(lambda f, _: [det(f, "person")])
    supplement = FakeDetector(lambda f, _: [det(f, "watch")], ("watch",))
    combined = CombinedDetector(baseline, supplement, every_n_frames=3)
    assert len(combined.detect(Frame(0, 0, np.zeros((100, 200, 3), np.uint8)))) == 2
    rows = combined.detect(Frame(1, 0.1, np.zeros((100, 200, 3), np.uint8)))
    assert [row.class_name for row in rows] == ["person"]
    assert len(supplement.calls) == 1


def test_tiled_boxes_restore_source_coordinates_and_reject_internal_edge_fragment():
    def predict(frame, call):
        if call == 1:
            return []
        if call == 5:
            return [det(frame), det(frame, box=(0, 10, 5, 30))]
        return []

    fake = FakeDetector(predict)
    tiled = TiledDetector(fake, tile_fraction=0.65)
    rows = tiled.detect(Frame(6, 0.2, np.zeros((100, 200, 3), np.uint8)))
    assert len(rows) == 1
    assert rows[0].xyxy == (80, 45, 100, 65)
    assert rows[0].frame_id == 6 and rows[0].timestamp == 0.2
    assert rows[0].metadata["inference_crop_xyxy"] == [70, 35, 200, 100]


def test_tiles_preserve_whole_image_observation_and_class_aware_nms():
    frame = Frame(1, 0.1, np.zeros((100, 200, 3), np.uint8))
    first = det(frame, "watch", box=(10, 10, 30, 30), confidence=0.9)
    duplicate = det(frame, "watch", box=(11, 11, 31, 31), confidence=0.8)
    alternate = det(frame, "mouse", box=(10, 10, 30, 30))
    assert merge_detections([duplicate, alternate, first], 0.5) == [first, alternate]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"tile_fraction": 0.5},
        {"tile_fraction": float("nan")},
        {"every_n_frames": 0},
        {"every_n_frames": True},
        {"nms_threshold": float("inf")},
        {"edge_margin_px": -1},
    ],
)
def test_tile_options_reject_unsafe_geometry_or_cadence(kwargs):
    with pytest.raises(ValueError):
        TiledDetector(FakeDetector(lambda *_: []), **kwargs)


def test_vocabulary_hash_and_dimension_binding_rejects_wrong_or_modified_vectors(tmp_path):
    vectors = tmp_path / "v.npz"
    np.savez(vectors, embeddings=np.ones((1, 2, 512), np.float32))
    manifest = {
        "schema": "visualmap.yolo_world_vocabulary.v1",
        "embedding_file": "v.npz",
        "embedding_sha256": hashlib.sha256(vectors.read_bytes()).hexdigest(),
        "prompts": [{"prompt": "watch", "class_name": "watch"}],
    }
    path = tmp_path / "v.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="512"):
        load_vocabulary(path)
    np.savez(vectors, embeddings=np.ones((1, 1, 512), np.float32))
    with pytest.raises(ValueError, match="hash"):
        load_vocabulary(path)
    manifest["embedding_sha256"] = hashlib.sha256(vectors.read_bytes()).hexdigest()
    path.write_text(json.dumps(manifest), encoding="utf-8")
    _, encoded, _ = load_vocabulary(path)
    assert encoded.shape == (1, 1, 512)
    assert np.linalg.norm(encoded) == pytest.approx(1)


@pytest.mark.parametrize("kind", [OpenVocabularyConfig, GroundingDINOConfig])
@pytest.mark.parametrize("value", [0, 1, float("nan"), float("inf"), True])
def test_open_model_threshold_validation(kind, value):
    kwargs = {"model_path": "unused", "conf_threshold": value}
    if kind is OpenVocabularyConfig:
        kwargs["vocabulary_path"] = "unused"
    else:
        kwargs.update(tokenizer_path="unused", prompts_path="unused")
    with pytest.raises(ValueError):
        kind(**kwargs)


def test_submap_landmarks_cannot_leak_through_projected_point_fallback():
    mapping = MapResult(
        trajectory=Trajectory(),
        map_points=[
            MapPoint(0, [0, 0, 3], observations={0: 1, 1: 1}),
            MapPoint(1, [0, 0, 3], observations={10: 1, 11: 1}),
            MapPoint(2, [0, 0, 3], observations={0: 1, 11: 1}),
            MapPoint(3, [0, 0, 3]),
        ],
        metadata={
            "frame_quality": {
                "0": {"world_frame_id": "old"},
                "1": {"world_frame_id": "old"},
                "10": {"world_frame_id": "new"},
                "11": {"world_frame_id": "new"},
            }
        },
    )
    assert [point.id for point in SemanticPipeline._points_in_world(mapping, "new")] == [1]
    assert SemanticPipeline._points_in_world(mapping, None) == []


def test_relocalized_pose_has_trusted_geometry_only_with_world_and_scale():
    mapping = MapResult(
        trajectory=Trajectory(),
        metadata={
            "scale_consistent": True,
            "frame_quality": {
                "10": {"status": "relocalized", "pose_valid": True, "world_frame_id": "old"}
            },
        },
    )
    pose = CameraPose.identity(frame_id=10)
    assert SemanticPipeline._camera_evidence(mapping, 10, pose)["pose_trusted"] is True
    mapping.metadata["scale_consistent"] = False
    assert SemanticPipeline._camera_evidence(mapping, 10, pose)["pose_trusted"] is False


def test_single_real_open_vocabulary_candidate_is_visible_without_lowering_baseline_hits():
    tracker = IoUTracker(TrackerConfig(min_hits=3))
    frame = Frame(0, 0, np.zeros((100, 200, 3), np.uint8))
    baseline = det(frame, "mouse", cid=64)
    candidate = Detection(
        0,
        0,
        80,
        "earbud case",
        0.3,
        baseline.xyxy,
        metadata={"classification_source": "open_vocabulary_model"},
    )
    rows = tracker.update([baseline, candidate])
    assert len(rows) == 1 and rows[0].class_name == "earbud case"
    assert rows[0].detection.metadata["classification_candidate_only"] is True
    assert rows[0].detection.metadata["classification_needs_review"] is True
    assert len(tracker.tracks) == 2


def test_candidate_pipeline_disables_relations_and_preserves_review_evidence():
    from visualmap.detection.pipeline import SemanticPipelineConfig

    mapping = MapResult(trajectory=Trajectory())
    from visualmap.slam.types import CameraIntrinsics

    mapping.intrinsics = CameraIntrinsics(100, 100, 50, 50, 100, 100)
    pipeline = SemanticPipeline(config=SemanticPipelineConfig(tracker=TrackerConfig(min_hits=3)))
    candidate = Detection(
        0,
        0,
        80,
        "earbud case",
        0.3,
        (10, 10, 30, 30),
        metadata={"classification_source": "open_vocabulary_model"},
    )
    pipeline.process_detections([candidate], mapping)
    obj = pipeline.result().objects[0]
    assert obj.metadata["classification_candidate_only"] is True
    assert obj.metadata["classification_evidence"]["candidate_only"] is True
    assert obj.metadata["position_relation_usable"] is False


def test_grounding_letterbox_inverse_maps_exact_pixels_without_distortion():
    detector = GroundingDINODetector.__new__(GroundingDINODetector)
    detector.config = GroundingDINOConfig("unused", "unused", "unused")
    detector.prompts = [{"prompt": "watch", "class_name": "watch"}]
    detector._text_inputs = {"input_ids": np.array([[101, 1, 102]])}
    detector._token_ranges = [[1]]
    # Source 720x1280 -> 450x800 centered with pad_x=375, scale=.625.
    # A source box [100,200,300,600] becomes [437.5,125,562.5,375] in model input.
    boxes = np.array([[[500 / 1200, 250 / 800, 125 / 1200, 250 / 800]]])
    logits = np.array([[[-10, 3, -10]]])
    rows = detector._decode(
        logits, boxes, Frame(900, 30, np.zeros((1280, 720, 3), np.uint8)), (0.625, 375, 0)
    )
    assert len(rows) == 1
    assert rows[0].xyxy == pytest.approx((100, 200, 300, 600))


def _crop_verifier(vector):
    verifier = CLIPCropVerifier.__new__(CLIPCropVerifier)
    verifier.config = CLIPCropConfig(
        "unused",
        "unused",
        allowed_classes=("earbud case", "watch"),
        min_similarity=0.25,
        min_margin=0.003,
    )
    verifier._names = ["earbud case", "watch", "bottle"]
    verifier._vectors = np.eye(3, 512, dtype=np.float32)
    image_vector = np.zeros((1, 512), np.float32)
    image_vector[0, :3] = vector
    verifier._session = SimpleNamespace(run=lambda *_: [image_vector])
    verifier.rejected_predictions = []
    return verifier


def test_clip_crop_reorders_only_model_proposals_and_keeps_original_class_score_and_bounds():
    verifier = _crop_verifier([0.8, 0.4, 0.2])
    proposal = Detection(
        990,
        33,
        80,
        "watch",
        0.323,
        (10, 10, 40, 50),
        metadata={"classification_source": "open_vocabulary_model", "raw_model_class": "watch"},
    )
    frame = Frame(990, 33, np.full((100, 100, 3), 114, np.uint8))
    rows = verifier.verify(frame, [proposal])
    assert len(rows) == 1 and rows[0].class_name == "earbud case"
    assert rows[0].xyxy == proposal.xyxy and rows[0].confidence == proposal.confidence
    evidence = rows[0].metadata["crop_verification"]
    assert evidence["proposal"]["class_name"] == "watch"
    assert evidence["proposal"]["confidence"] == 0.323
    assert evidence["score_is_probability"] is False
    assert evidence["physical_boundary_verified"] is False
    assert rows[0].metadata["classification_candidate_only"] is True
    assert rows[0].to_dict()["classification_evidence"]["crop_verification"] == evidence


@pytest.mark.parametrize("vector", [[0.2, 0.1, 0.8], [0.7, 0.7, 0.1]])
def test_clip_crop_rejects_known_context_or_ties_and_retains_original_prediction(vector):
    verifier = _crop_verifier(vector)
    proposal = Detection(990, 33, 80, "watch", 0.3, (10, 10, 40, 50))
    assert verifier.verify(Frame(990, 33, np.zeros((100, 100, 3), np.uint8)), [proposal]) == []
    assert verifier.rejected_predictions[0]["proposal"]["class_name"] == "watch"
    assert verifier.rejected_predictions[0]["reason"] == "crop_not_supported"


def test_clip_preprocess_uses_rgb_and_documented_normalization():
    image = np.zeros((300, 100, 3), np.uint8)
    image[:, :, 2] = 255
    blob = clip_image_blob(image)
    assert blob.shape == (1, 3, 224, 224)
    assert blob[0, 0, 100, 100] == pytest.approx((1 - 0.48145466) / 0.26862954)
    assert blob[0, 2, 100, 100] == pytest.approx(-0.40821073 / 0.27577711)
