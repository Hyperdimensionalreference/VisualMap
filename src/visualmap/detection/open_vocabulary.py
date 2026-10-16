"""Local YOLO-World ONNX inference with a verified, pre-encoded vocabulary.

No runtime downloads, network calls, image uploads, PyTorch or model pickle loads.
The vocabulary is a model input, never a relabelling of a closed-vocabulary head.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from visualmap.detection.base import ObjectDetector
from visualmap.detection.types import Detection
from visualmap.detection.yolo_onnx import YOLOONNXConfig, YOLOONNXDetector
from visualmap.slam.types import Frame


@dataclass
class OpenVocabularyConfig:
    model_path: str | Path
    vocabulary_path: str | Path
    runtime_path: str | Path | None = None
    conf_threshold: float = 0.28
    nms_threshold: float = 0.45
    input_width: int = 640
    input_height: int = 640
    intra_op_threads: int = 2
    # Must follow the particular exported graph; never infer this from values.
    scores_are_logits: bool = False

    def __post_init__(self) -> None:
        for name in ("conf_threshold", "nms_threshold"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or not 0 < value < 1:
                raise ValueError(f"{name} must be finite and in (0, 1)")
        for name in ("input_width", "input_height", "intra_op_threads"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_vocabulary(path: str | Path) -> tuple[list[dict], np.ndarray, dict]:
    """Read data only; hashes bind vectors, prompt order and model provenance."""
    path = Path(path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema") not in {
        "visualmap.yolo_world_vocabulary.v1",
        "visualmap.clip_crop_vocabulary.v1",
    }:
        raise ValueError("unsupported YOLO-World vocabulary schema")
    prompts = manifest.get("prompts")
    if not isinstance(prompts, list) or not prompts or len(prompts) > 512:
        raise ValueError("vocabulary requires 1 to 512 prompts")
    for row in prompts:
        if not isinstance(row, dict) or any(
            not isinstance(row.get(key), str) or not row[key].strip()
            for key in ("prompt", "class_name")
        ):
            raise ValueError("vocabulary prompts require prompt and class_name")
    vector_name = manifest.get("embedding_file")
    if not isinstance(vector_name, str) or Path(vector_name).name != vector_name:
        raise ValueError("embedding_file must be a local filename")
    vector_path = path.parent / vector_name
    if _sha256(vector_path) != manifest.get("embedding_sha256"):
        raise ValueError("vocabulary embedding hash mismatch; prepare the vocabulary again")
    with np.load(vector_path, allow_pickle=False) as archive:
        vectors = np.asarray(archive["embeddings"], dtype=np.float32)
    if vectors.shape != (1, len(prompts), 512) or not np.isfinite(vectors).all():
        raise ValueError("vocabulary must contain finite [1, num_prompts, 512] CLIP embeddings")
    norm = np.linalg.norm(vectors, axis=-1, keepdims=True)
    if np.any(norm <= 1e-6):
        raise ValueError("vocabulary embeddings must not be zero")
    return prompts, np.ascontiguousarray(vectors / norm), manifest


class OpenVocabularyDetector(ObjectDetector):
    name = "yolo_world_onnx"

    def __init__(self, config: OpenVocabularyConfig) -> None:
        self.config = config
        model_path = Path(config.model_path)
        self._prompts, self._vectors, manifest = load_vocabulary(config.vocabulary_path)
        assets = {row["file"]: row for row in manifest.get("assets", [])}
        expected = assets.get(model_path.name, {}).get("sha256")
        if expected is None or _sha256(model_path) != expected:
            raise ValueError("YOLO-World model hash does not match the vocabulary manifest")
        self.vocabulary_hash = _sha256(Path(config.vocabulary_path))
        if config.runtime_path is not None:
            runtime_path = str(Path(config.runtime_path).resolve())
            if runtime_path not in sys.path:
                sys.path.insert(0, runtime_path)
        try:
            ort = importlib.import_module("onnxruntime")
        except ImportError as exc:
            raise ImportError(
                "YOLO-World needs local onnxruntime; run scripts/prepare_open_vocabulary.py"
            ) from exc
        options = ort.SessionOptions()
        options.intra_op_num_threads = config.intra_op_threads
        options.inter_op_num_threads = 1
        self._session = ort.InferenceSession(
            str(model_path), options, providers=["CPUExecutionProvider"]
        )
        inputs = {row.name for row in self._session.get_inputs()}
        if inputs != {"images", "txt_feats"}:
            raise ValueError(f"YOLO-World requires images and txt_feats inputs, got {inputs}")
        # Reuse the tested letterbox decoder, preserving the graph's prompt index.
        self._decoder = YOLOONNXDetector.__new__(YOLOONNXDetector)
        self._decoder.config = YOLOONNXConfig(
            model_path,
            class_names=tuple(row["prompt"] for row in self._prompts),
            conf_threshold=config.conf_threshold,
            nms_threshold=config.nms_threshold,
            input_width=config.input_width,
            input_height=config.input_height,
        )
        self.capabilities = {
            "backend": self.name,
            "vocabulary_kind": "encoded_text_prompts",
            "class_names": list(dict.fromkeys(row["class_name"] for row in self._prompts)),
            "prompts": [row["prompt"] for row in self._prompts],
            "vocabulary_sha256": self.vocabulary_hash,
            "runtime": "local_onnx_cpu",
            "offline": True,
            "accuracy_validated": False,
            "brand_recognition_supported": False,
            "limitation": (
                "text prompts expand candidates; tiny, blurred or occluded objects can fail"
            ),
        }

    def warmup(self) -> None:
        image = np.zeros((self.config.input_height, self.config.input_width, 3), np.uint8)
        self.detect(Frame(index=0, timestamp=0, image=image))

    def detect(self, frame: Frame) -> list[Detection]:
        image = frame.image
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        height, width = image.shape[:2]
        input_width, input_height = self.config.input_width, self.config.input_height
        scale = min(input_width / width, input_height / height)
        resized_width = min(input_width, max(1, round(width * scale)))
        resized_height = min(input_height, max(1, round(height * scale)))
        pad_x, pad_y = (input_width - resized_width) // 2, (input_height - resized_height) // 2
        padded = np.full((input_height, input_width, 3), 114, np.uint8)
        padded[pad_y : pad_y + resized_height, pad_x : pad_x + resized_width] = cv2.resize(
            image, (resized_width, resized_height)
        )
        blob = cv2.dnn.blobFromImage(padded, 1 / 255.0, swapRB=True)
        output = np.asarray(
            self._session.run(None, {"images": blob, "txt_feats": self._vectors})[0]
        )
        expected_channels = 4 + len(self._prompts)
        if output.ndim != 3 or output.shape[0] != 1 or output.shape[1] != expected_channels:
            raise ValueError(f"incompatible YOLO-World output shape: {output.shape}")
        if self.config.scores_are_logits:
            output = output.copy()
            output[:, 4:] = 1 / (1 + np.exp(-np.clip(output[:, 4:], -80, 80)))
        rows = self._decoder._postprocess(output, width, height, scale, pad_x, pad_y)
        return [
            Detection(
                frame_id=frame.index,
                timestamp=frame.timestamp,
                class_id=index,
                class_name=self._prompts[index]["class_name"],
                confidence=score,
                xyxy=box,
                metadata={
                    "classification_source": "open_vocabulary_model",
                    "classification_status": "unverified_text_prompt_prediction",
                    "classification_needs_review": True,
                    "classification_reason": "text_prompt_prediction_requires_review",
                    "raw_model_class": self._prompts[index]["prompt"],
                    "text_prompt": self._prompts[index]["prompt"],
                    "vocabulary_sha256": self.vocabulary_hash,
                    "brand_recognition_supported": False,
                    "score_is_probability": False,
                },
            )
            for index, score, box in rows
        ]
