"""YOLO detector via OpenCV DNN + ONNX (no ultralytics/torch required)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from visualmap.detection.base import ObjectDetector
from visualmap.detection.types import Detection
from visualmap.slam.types import Frame
from visualmap.utils.logging import get_logger

logger = get_logger(__name__)

__all__ = ["YOLOONNXConfig", "YOLOONNXDetector", "COCO80_NAMES"]


COCO80_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
]


@dataclass
class YOLOONNXConfig:
    model_path: Path
    class_names: tuple[str, ...] = tuple(COCO80_NAMES)
    conf_threshold: float = 0.35
    nms_threshold: float = 0.45
    input_width: int = 640
    input_height: int = 640

    def __post_init__(self) -> None:
        if self.input_width < 1 or self.input_height < 1:
            raise ValueError("YOLO input dimensions must be positive")
        if not self.class_names:
            raise ValueError("YOLO class_names must not be empty")
        if not 0 < self.conf_threshold <= 1 or not 0 < self.nms_threshold < 1:
            raise ValueError("YOLO thresholds must be in (0, 1)")


class YOLOONNXDetector(ObjectDetector):
    name = "yolo_onnx"

    def __init__(self, config: YOLOONNXConfig) -> None:
        self.config = config
        model = Path(config.model_path)
        if not model.is_file():
            raise FileNotFoundError(f"YOLO ONNX model not found: {model}")
        # The buffer overload works when a Windows path contains non-ASCII text.
        self._net = cv2.dnn.readNetFromONNX(
            np.frombuffer(model.read_bytes(), dtype=np.uint8)
        )
        self._net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self._net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        logger.info("Loaded YOLO ONNX: %s", model)

    def detect(self, frame: Frame) -> list[Detection]:
        img = frame.image
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        h0, w0 = img.shape[:2]
        input_w, input_h = self.config.input_width, self.config.input_height
        scale = min(input_w / w0, input_h / h0)
        resized_w = min(input_w, max(1, round(w0 * scale)))
        resized_h = min(input_h, max(1, round(h0 * scale)))
        pad_x = (input_w - resized_w) // 2
        pad_y = (input_h - resized_h) // 2
        padded = np.full((input_h, input_w, 3), 114, dtype=np.uint8)
        padded[pad_y : pad_y + resized_h, pad_x : pad_x + resized_w] = cv2.resize(
            img, (resized_w, resized_h)
        )
        blob = cv2.dnn.blobFromImage(
            padded,
            scalefactor=1 / 255.0,
            size=(input_w, input_h),
            swapRB=True,
            crop=False,
        )
        self._net.setInput(blob)
        outputs = self._net.forward()
        preds = self._postprocess(outputs, w0, h0, scale, pad_x, pad_y)
        return [
            Detection(
                frame_id=frame.index,
                timestamp=frame.timestamp,
                class_id=cid,
                class_name=(
                    self.config.class_names[cid]
                    if cid < len(self.config.class_names)
                    else str(cid)
                ),
                confidence=conf,
                xyxy=box,
            )
            for cid, conf, box in preds
        ]

    def _postprocess(
        self, outputs, w0: int, h0: int, scale: float, pad_x: int, pad_y: int
    ) -> list[tuple[int, float, tuple]]:
        """Support YOLOv5/v8 ONNX layouts: (1, N, 5+C) or (1, 4+C, N)."""
        out = outputs[0] if isinstance(outputs, (list, tuple)) else outputs
        out = np.asarray(out)

        if out.ndim == 3:
            if out.shape[0] != 1:
                raise ValueError(f"YOLO expects batch size 1, got output shape {out.shape}")
            out = out[0]
        n_classes = len(self.config.class_names)
        widths = (4 + n_classes, 5 + n_classes)
        if out.ndim != 2:
            raise ValueError(f"Unsupported YOLO output shape {out.shape}")
        # A single prediction has N=1; detect orientation by channel count.
        if out.shape[1] not in widths and out.shape[0] in widths:
            out = out.T
        # columns: cx cy w h obj? scores...
        if out.shape[1] == 4 + n_classes:
            boxes_xywh = out[:, :4]
            scores = out[:, 4:]
            class_ids = np.argmax(scores, axis=1)
            confs = scores[np.arange(len(scores)), class_ids]
        elif out.shape[1] == 5 + n_classes:
            boxes_xywh = out[:, :4]
            obj = out[:, 4]
            scores = out[:, 5:]
            class_ids = np.argmax(scores, axis=1)
            confs = obj * scores[np.arange(len(scores)), class_ids]
        else:
            raise ValueError(f"Unsupported YOLO output shape {out.shape}")

        boxes: list[tuple[float, float, float, float]] = []
        keep_ids: list[int] = []
        for i in range(len(confs)):
            if (
                not np.isfinite(confs[i])
                or not self.config.conf_threshold <= confs[i] <= 1
                or not np.all(np.isfinite(boxes_xywh[i]))
                or np.any(boxes_xywh[i, 2:] <= 0)
                or not np.all(np.isfinite(scores[i]))
                or np.any((scores[i] < 0) | (scores[i] > 1))
                or (out.shape[1] == 5 + n_classes and not 0 <= obj[i] <= 1)
            ):
                continue
            cx, cy, bw, bh = boxes_xywh[i]
            x1 = np.clip((cx - bw / 2.0 - pad_x) / scale, 0, w0)
            y1 = np.clip((cy - bh / 2.0 - pad_y) / scale, 0, h0)
            x2 = np.clip((cx + bw / 2.0 - pad_x) / scale, 0, w0)
            y2 = np.clip((cy + bh / 2.0 - pad_y) / scale, 0, h0)
            if x2 <= x1 or y2 <= y1:
                continue
            boxes.append((float(x1), float(y1), float(x2), float(y2)))
            keep_ids.append(i)

        if not boxes:
            return []

        results: list[tuple[int, float, tuple]] = []
        for class_id in set(int(class_ids[i]) for i in keep_ids):
            positions = [j for j, i in enumerate(keep_ids) if int(class_ids[i]) == class_id]
            class_boxes = [boxes[j] for j in positions]
            idxs = cv2.dnn.NMSBoxes(
                [[b[0], b[1], b[2] - b[0], b[3] - b[1]] for b in class_boxes],
                [float(confs[keep_ids[j]]) for j in positions],
                self.config.conf_threshold,
                self.config.nms_threshold,
            )
            for local_index in np.asarray(idxs).reshape(-1):
                j = positions[int(local_index)]
                i = keep_ids[j]
                results.append((class_id, float(confs[i]), boxes[j]))
        return sorted(results, key=lambda result: (-result[1], result[0], result[2]))
