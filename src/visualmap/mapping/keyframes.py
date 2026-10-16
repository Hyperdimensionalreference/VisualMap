"""Keyframe selection for monocular VO / SLAM.

Keyframes are a subset of frames used for triangulation and (later) BA.
Criteria are config-driven: min matches, min translation, min time gap.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from visualmap.slam.types import CameraPose, Frame
from visualmap.utils.logging import get_logger

logger = get_logger(__name__)

__all__ = ["KeyframePolicy", "KeyframeSelector"]


@dataclass
class KeyframePolicy:
    min_frames_between: int = 3
    min_translation: float = 0.35
    min_rotation_deg: float = 8.0
    min_matches: int = 40
    # Always take the first frame as keyframe
    force_first: bool = True


class KeyframeSelector:
    """Decide whether a tracked frame should become a keyframe."""

    def __init__(self, policy: KeyframePolicy | None = None) -> None:
        self.policy = policy or KeyframePolicy()
        self._last_kf_index: int | None = None
        self._last_kf_pose: CameraPose | None = None
        self._count = 0

    @property
    def count(self) -> int:
        return self._count

    def reset(self) -> None:
        self._last_kf_index = None
        self._last_kf_pose = None
        self._count = 0

    def should_insert(
        self,
        frame: Frame,
        pose: CameraPose,
        n_matches: int,
        tracked: bool,
    ) -> bool:
        if not tracked and self._last_kf_index is not None:
            return False

        if self._last_kf_index is None:
            if self.policy.force_first or tracked:
                self._insert(frame, pose)
                return True
            return False

        if frame.index - self._last_kf_index < self.policy.min_frames_between:
            return False

        if n_matches < self.policy.min_matches:
            return False

        assert self._last_kf_pose is not None
        dt = float(np.linalg.norm(pose.t - self._last_kf_pose.t))
        drot = _rotation_angle_deg(self._last_kf_pose.R, pose.R)

        if dt >= self.policy.min_translation or drot >= self.policy.min_rotation_deg:
            self._insert(frame, pose)
            return True
        return False

    def _insert(self, frame: Frame, pose: CameraPose) -> None:
        self._last_kf_index = frame.index
        self._last_kf_pose = pose
        self._count += 1
        logger.debug("Keyframe #%d at frame %d", self._count, frame.index)


def _rotation_angle_deg(R1: np.ndarray, R2: np.ndarray) -> float:
    R = R1.T @ R2
    cos_theta = (np.trace(R) - 1.0) / 2.0
    cos_theta = float(np.clip(cos_theta, -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_theta)))
