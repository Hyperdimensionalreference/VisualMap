"""Data-only GroundingDINO ONNX adapter with local BERT phrase tokenization.

Implements the model's documented normalized-image and phrase-block attention
contracts; requires no annotation GUI, transformers, PyTorch or network access.
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
from visualmap.detection.configured import merge_detections
from visualmap.detection.types import Detection
from visualmap.slam.types import Frame


@dataclass
class GroundingDINOConfig:
    model_path: str | Path
    tokenizer_path: str | Path
    prompts_path: str | Path
    runtime_path: str | Path | None = None
    manifest_path: str | Path | None = None
    conf_threshold: float = 0.3
    text_threshold: float = 0.2
    nms_threshold: float = 0.45
    input_width: int = 1200
    input_height: int = 800
    intra_op_threads: int = 2

    def __post_init__(self) -> None:
        for name in ("conf_threshold", "text_threshold", "nms_threshold"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or not 0 < value < 1:
                raise ValueError(f"{name} must be finite and in (0, 1)")
        for name in ("input_width", "input_height", "intra_op_threads"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")


def phrase_inputs(tokenizer, prompts: list[dict]) -> tuple[dict, list[list[int]]]:
    """Bind each output label to exact token offsets; reject overlong captions."""
    caption = ""
    spans = []
    for row in prompts:
        text = row["prompt"].strip().lower().rstrip(".")
        start = len(caption)
        caption += text + " . "
        spans.append((start, start + len(text)))
    encoded = tokenizer.encode(caption.strip())
    ids = encoded.ids
    if len(ids) > 256:
        raise ValueError("GroundingDINO caption exceeds 256 tokens; reduce prompts")
    count = len(ids)
    attention = np.eye(count, dtype=bool)[None]
    positions = np.zeros((1, count), dtype=np.int64)
    # Period-delimited phrases attend internally, not to unrelated class text.
    previous = 0
    for index, token_id in enumerate(ids):
        if token_id in {101, 102, 1012, 1029}:
            if index not in (0, count - 1):
                attention[0, previous + 1 : index + 1, previous + 1 : index + 1] = True
                positions[0, previous + 1 : index + 1] = np.arange(index - previous)
            previous = index
    ranges = [
        [index for index, (a, b) in enumerate(encoded.offsets) if b > a and a < end and b > start]
        for start, end in spans
    ]
    if any(not indices for indices in ranges):
        raise ValueError("GroundingDINO empty phrase token range")
    return {
        "input_ids": np.asarray([ids], np.int64),
        "attention_mask": np.asarray([encoded.attention_mask], bool),
        "token_type_ids": np.asarray([encoded.type_ids], np.int64),
        "position_ids": positions,
        "text_token_mask": attention,
    }, ranges


class GroundingDINODetector(ObjectDetector):
    name = "grounding_dino_onnx"

    def __init__(self, config: GroundingDINOConfig) -> None:
        self.config = config
        if config.runtime_path is not None:
            runtime_path = str(Path(config.runtime_path).resolve())
            if runtime_path not in sys.path:
                sys.path.insert(0, runtime_path)
        ort = importlib.import_module("onnxruntime")
        tokenizer_type = importlib.import_module("tokenizers").Tokenizer
        if config.manifest_path is not None:
            manifest = json.loads(Path(config.manifest_path).read_text(encoding="utf-8"))
            assets = {row["file"]: row for row in manifest["assets"]}
            for path in (Path(config.model_path), Path(config.tokenizer_path)):
                if hashlib.sha256(path.read_bytes()).hexdigest() != assets.get(path.name, {}).get(
                    "sha256"
                ):
                    raise ValueError("GroundingDINO model/tokenizer hash mismatch")
        self.prompts = json.loads(Path(config.prompts_path).read_text(encoding="utf-8"))
        if (
            not isinstance(self.prompts, list)
            or not self.prompts
            or any(
                not isinstance(row, dict)
                or any(
                    not isinstance(row.get(key), str) or not row[key].strip()
                    for key in ("prompt", "class_name")
                )
                for row in self.prompts
            )
        ):
            raise ValueError("GroundingDINO prompts require prompt and class_name")
        tokenizer = tokenizer_type.from_file(str(config.tokenizer_path))
        self._text_inputs, self._token_ranges = phrase_inputs(tokenizer, self.prompts)
        options = ort.SessionOptions()
        options.intra_op_num_threads = config.intra_op_threads
        options.inter_op_num_threads = 1
        self._session = ort.InferenceSession(
            str(config.model_path), options, providers=["CPUExecutionProvider"]
        )
        expected = set(self._text_inputs) | {"img"}
        if {row.name for row in self._session.get_inputs()} != expected:
            raise ValueError("incompatible GroundingDINO input schema")
        self.capabilities = {
            "backend": self.name,
            "vocabulary_kind": "text_grounding",
            "class_names": list(dict.fromkeys(row["class_name"] for row in self.prompts)),
            "prompts": [row["prompt"] for row in self.prompts],
            "offline": True,
            "accuracy_validated": False,
            "brand_recognition_supported": False,
        }

    def warmup(self) -> None:
        self.detect(
            Frame(
                index=0,
                timestamp=0,
                image=np.zeros((self.config.input_height, self.config.input_width, 3), np.uint8),
            )
        )

    def detect(self, frame: Frame) -> list[Detection]:
        image = frame.image
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width = image.shape[:2]
        scale = min(self.config.input_width / width, self.config.input_height / height)
        resized_width, resized_height = max(1, round(width * scale)), max(1, round(height * scale))
        pad_x = (self.config.input_width - resized_width) // 2
        pad_y = (self.config.input_height - resized_height) // 2
        resized = np.full((self.config.input_height, self.config.input_width, 3), 114, np.uint8)
        resized[pad_y : pad_y + resized_height, pad_x : pad_x + resized_width] = cv2.resize(
            image, (resized_width, resized_height)
        )
        normalized = (
            resized.astype(np.float32) / 255.0 - np.asarray([0.485, 0.456, 0.406], np.float32)
        ) / np.asarray([0.229, 0.224, 0.225], np.float32)
        blob = np.ascontiguousarray(normalized.transpose(2, 0, 1)[None])
        outputs = self._session.run(None, {**self._text_inputs, "img": blob})
        return self._decode(outputs[0], outputs[1], frame, (scale, pad_x, pad_y))

    def _decode(
        self,
        logits: np.ndarray,
        boxes: np.ndarray,
        frame: Frame,
        preprocessing: tuple[float, float, float] | None = None,
    ) -> list[Detection]:
        if (
            logits.ndim != 3
            or boxes.ndim != 3
            or logits.shape[0] != 1
            or boxes.shape[0] != 1
            or boxes.shape[2] != 4
            or logits.shape[1] != boxes.shape[1]
            or logits.shape[2] < self._text_inputs["input_ids"].shape[1]
        ):
            raise ValueError("incompatible GroundingDINO output shape")
        scores = 1 / (1 + np.exp(-np.clip(logits[0], -80, 80)))
        height, width = frame.image.shape[:2]
        scale, pad_x, pad_y = preprocessing or (1.0, 0.0, 0.0)
        box_width = self.config.input_width if preprocessing else width
        box_height = self.config.input_height if preprocessing else height
        rows = []
        for anchor, box in enumerate(boxes[0]):
            if (
                not np.isfinite(box).all()
                or np.any(box[2:] <= 0)
                or not np.isfinite(
                    logits[0, anchor, : self._text_inputs["input_ids"].shape[1]]
                ).all()
            ):
                continue
            # A complete phrase must carry evidence. The max of arbitrary caption
            # tokens can relabel a box with an unrelated compound noun.
            phrase_scores = [
                float(np.mean(scores[anchor, indices])) for indices in self._token_ranges
            ]
            index = int(np.argmax(phrase_scores))
            score = phrase_scores[index]
            if score < self.config.conf_threshold:
                continue
            token_scores = scores[anchor, self._token_ranges[index]]
            if np.mean(token_scores >= self.config.text_threshold) < 0.5:
                continue
            cx, cy, bw, bh = box
            x1, x2 = np.clip(
                [
                    ((cx - bw / 2) * box_width - pad_x) / scale,
                    ((cx + bw / 2) * box_width - pad_x) / scale,
                ],
                0,
                width,
            )
            y1, y2 = np.clip(
                [
                    ((cy - bh / 2) * box_height - pad_y) / scale,
                    ((cy + bh / 2) * box_height - pad_y) / scale,
                ],
                0,
                height,
            )
            if x2 <= x1 or y2 <= y1:
                continue
            row = self.prompts[index]
            rows.append(
                Detection(
                    frame_id=frame.index,
                    timestamp=frame.timestamp,
                    class_id=index,
                    class_name=row["class_name"],
                    confidence=score,
                    xyxy=tuple(map(float, (x1, y1, x2, y2))),
                    metadata={
                        "classification_source": "open_vocabulary_model",
                        "classification_status": "unverified_text_prompt_prediction",
                        "classification_needs_review": True,
                        "classification_reason": "text_prompt_prediction_requires_review",
                        "raw_model_class": row["prompt"],
                        "text_prompt": row["prompt"],
                        "text_token_scores": list(map(float, token_scores)),
                        "score_aggregation": "mean_phrase_token_probability",
                        "brand_recognition_supported": False,
                        "score_is_probability": False,
                    },
                )
            )
        return merge_detections(rows, self.config.nms_threshold)
