"""Demo detector: orange/bright moving blocks (works on sample.avi without weights)."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from visualmap.detection.base import ObjectDetector
from visualmap.detection.types import Detection
from visualmap.slam.types import Frame
from visualmap.utils.logging import get_logger

logger = get_logger(__name__)

__all__ = ["DemoDetector", "DemoDetectorConfig"]


@dataclass
class DemoDetectorConfig:
    min_area: int = 80
    conf_threshold: float = 0.4
    # Class presented by demo blobs
    class_id: int = 0
    class_name: str = "object"


class DemoDetector(ObjectDetector):
    """HSV color-blob detector for synthetic demos and tests (not for real scenes)."""

    name = "demo"

    def __init__(self, config: DemoDetectorConfig | None = None) -> None:
        self.config = config or DemoDetectorConfig()

    def detect(self, frame: Frame) -> list[Detection]:
        img = frame.image
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Broad "warm / bright saturated" mask — catches orange demo rectangles
        lower = np.array([5, 80, 80], dtype=np.uint8)
        upper = np.array([40, 255, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower, upper)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        dets: list[Detection] = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.config.min_area:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            conf = float(min(0.99, 0.5 + area / 2000.0))
            if conf < self.config.conf_threshold:
                continue
            dets.append(
                Detection(
                    frame_id=frame.index,
                    timestamp=frame.timestamp,
                    class_id=self.config.class_id,
                    class_name=self.config.class_name,
                    confidence=conf,
                    xyxy=(float(x), float(y), float(x + w), float(y + h)),
                )
            )
        return dets
