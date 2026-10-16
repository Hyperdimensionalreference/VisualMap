"""OpenCV Zoo YOLOX-S ONNX adapter for COCO object detection."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from visualmap.detection.base import ObjectDetector
from visualmap.detection.types import Detection
from visualmap.detection.yolo_onnx import COCO80_NAMES
from visualmap.slam.types import Frame

__all__ = ["YOLOXONNXConfig", "YOLOXONNXDetector"]


@dataclass
class YOLOXONNXConfig:
    model_path: Path
    conf_threshold: float = 0.35
    nms_threshold: float = 0.45
    input_width: int = 640
    input_height: int = 640
    class_names: tuple[str, ...] = tuple(COCO80_NAMES)
    pegboard_refinement: bool = False
    pegboard_config: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.input_width < 32 or self.input_height < 32:
            raise ValueError("YOLOX input dimensions must be at least 32")
        if self.input_width % 32 or self.input_height % 32:
            raise ValueError("YOLOX input dimensions must be multiples of 32")
        if not self.class_names:
            raise ValueError("YOLOX class_names must not be empty")
        if not 0 < self.conf_threshold <= 1 or not 0 < self.nms_threshold < 1:
            raise ValueError("YOLOX thresholds must be in (0, 1)")


class YOLOXONNXDetector(ObjectDetector):
    """Decode the raw YOLOX-S head shipped in the OpenCV model zoo."""

    name = "yolox_onnx"

    def __init__(self, config: YOLOXONNXConfig) -> None:
        self.config = config
        if not Path(config.model_path).is_file():
            raise FileNotFoundError(f"YOLOX ONNX model not found: {config.model_path}")
        # OpenCV's Windows path overload can fail on non-ASCII directories.
        self._net = cv2.dnn.readNetFromONNX(
            np.frombuffer(Path(config.model_path).read_bytes(), dtype=np.uint8)
        )
        self._net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self._net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        self._grid, self._strides = self._make_grid(config.input_width, config.input_height)
        self._pegboard_refiner = None
        if config.pegboard_refinement:
            from visualmap.detection.pegboard import PegboardConfig, PegboardRefiner

            self._pegboard_refiner = PegboardRefiner(PegboardConfig(**config.pegboard_config))

    @staticmethod
    def _make_grid(width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
        grids = []
        strides = []
        for stride in (8, 16, 32):
            gx, gy = np.meshgrid(np.arange(width // stride), np.arange(height // stride))
            grids.append(np.stack((gx, gy), axis=-1).reshape(-1, 2))
            strides.append(np.full((grids[-1].shape[0], 1), stride))
        return np.concatenate(grids), np.concatenate(strides)

    def warmup(self) -> None:
        """Check that the loaded model runs and has the expected COCO head."""
        sample = np.zeros((self.config.input_height, self.config.input_width, 3), dtype=np.uint8)
        self.detect(Frame(index=0, timestamp=0.0, image=sample))

    def detect(self, frame: Frame) -> list[Detection]:
        image = frame.image
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        height, width = image.shape[:2]
        input_width, input_height = self.config.input_width, self.config.input_height
        scale = min(input_width / width, input_height / height)
        resized_width = max(1, min(input_width, int(width * scale)))
        resized_height = max(1, min(input_height, int(height * scale)))
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        padded = np.full((input_height, input_width, 3), 114, dtype=np.float32)
        padded[:resized_height, :resized_width] = cv2.resize(rgb, (resized_width, resized_height))
        blob = np.ascontiguousarray(padded.transpose(2, 0, 1)[None])
        self._net.setInput(blob)
        detections = [
            Detection(
                frame_id=frame.index,
                timestamp=frame.timestamp,
                class_id=class_id,
                class_name=self.config.class_names[class_id],
                confidence=score,
                xyxy=box,
            )
            for class_id, score, box in self._postprocess(self._net.forward(), width, height, scale)
        ]
        refiner = getattr(self, "_pegboard_refiner", None)
        return refiner.refine(frame, detections) if refiner is not None else detections

    def _postprocess(
        self, output: np.ndarray, width: int, height: int, scale: float
    ) -> list[tuple[int, float, tuple[float, float, float, float]]]:
        predictions = np.asarray(output)
        if predictions.ndim == 3 and predictions.shape[0] == 1:
            predictions = predictions[0]
        expected = (len(self._grid), 5 + len(self.config.class_names))
        if predictions.shape != expected:
            raise ValueError(f"YOLOX output shape {predictions.shape}, expected {expected}")

        centers = (predictions[:, :2] + self._grid) * self._strides
        sizes = np.exp(np.clip(predictions[:, 2:4], -20, 20)) * self._strides
        class_ids = np.argmax(predictions[:, 5:], axis=1)
        scores = predictions[:, 4] * predictions[np.arange(len(predictions)), 5 + class_ids]
        candidates = np.flatnonzero(
            np.all(np.isfinite(predictions), axis=1)
            & np.all((predictions[:, 4:] >= 0) & (predictions[:, 4:] <= 1), axis=1)
            & (scores >= self.config.conf_threshold)
            & (scores <= 1)
        )
        boxes = []
        for index in candidates:
            cx, cy = centers[index] / scale
            box_width, box_height = sizes[index] / scale
            x1 = float(np.clip(cx - box_width / 2, 0, width))
            y1 = float(np.clip(cy - box_height / 2, 0, height))
            x2 = float(np.clip(cx + box_width / 2, 0, width))
            y2 = float(np.clip(cy + box_height / 2, 0, height))
            boxes.append((x1, y1, x2, y2))

        results: list[tuple[int, float, tuple[float, float, float, float]]] = []
        for class_id in set(int(class_ids[i]) for i in candidates):
            positions = [
                pos
                for pos, index in enumerate(candidates)
                if class_ids[index] == class_id
                and boxes[pos][2] > boxes[pos][0]
                and boxes[pos][3] > boxes[pos][1]
            ]
            if not positions:
                continue
            nms_boxes = [
                [
                    boxes[pos][0],
                    boxes[pos][1],
                    boxes[pos][2] - boxes[pos][0],
                    boxes[pos][3] - boxes[pos][1],
                ]
                for pos in positions
            ]
            nms_scores = [float(scores[candidates[pos]]) for pos in positions]
            keep = cv2.dnn.NMSBoxes(
                nms_boxes, nms_scores, self.config.conf_threshold, self.config.nms_threshold
            )
            for local_index in np.asarray(keep).reshape(-1):
                pos = positions[int(local_index)]
                results.append((class_id, nms_scores[int(local_index)], boxes[pos]))
        return sorted(results, key=lambda result: (-result[1], result[0], result[2]))
