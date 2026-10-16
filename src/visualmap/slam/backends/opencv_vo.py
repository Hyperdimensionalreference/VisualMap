"""Monocular visual odometry with OpenCV (ORB + Essential Matrix) + keyframes.

Pipeline (V0.3):
  Frame → ORB → match prev → Essential/recoverPose → T_world←cur
       → keyframe policy → triangulate (last KF ↔ cur) → sparse MapPoints

Pose convention (docs/adr/0003): stored poses are T_world←camera.
Monocular scale is arbitrary; each successful step normalizes ||t||.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from visualmap.mapping.keyframes import KeyframePolicy, KeyframeSelector
from visualmap.mapping.triangulation import TriangulationConfig, triangulate_two_view
from visualmap.slam.base import SLAMBackend
from visualmap.slam.types import (
    CameraIntrinsics,
    CameraPose,
    Frame,
    KeyFrame,
    MapPoint,
    MapResult,
    Trajectory,
)
from visualmap.utils.logging import get_logger

logger = get_logger(__name__)

__all__ = ["OpenCVVOBackend", "OpenCVVOConfig", "create_backend"]


@dataclass
class OpenCVVOConfig:
    """Tunable VO / mapping parameters (from config files)."""

    n_features: int = 1500
    match_ratio: float = 0.75
    min_matches: int = 30
    ransac_thresh_px: float = 1.0
    ransac_confidence: float = 0.999
    default_translation_norm: float = 1.0
    min_inliers: int = 20
    max_map_points: int = 8000
    kf_min_frames_between: int = 3
    kf_min_translation: float = 0.35
    kf_min_rotation_deg: float = 8.0
    kf_min_matches: int = 40
    tri_min_parallax_deg: float = 1.0
    tri_max_reproj_error_px: float = 3.0
    tri_min_depth: float = 0.1
    tri_max_depth: float = 200.0
    extra: dict = field(default_factory=dict)

    def keyframe_policy(self) -> KeyframePolicy:
        return KeyframePolicy(
            min_frames_between=self.kf_min_frames_between,
            min_translation=self.kf_min_translation,
            min_rotation_deg=self.kf_min_rotation_deg,
            min_matches=self.kf_min_matches,
        )

    def triangulation_config(self) -> TriangulationConfig:
        return TriangulationConfig(
            min_parallax_deg=self.tri_min_parallax_deg,
            max_reproj_error_px=self.tri_max_reproj_error_px,
            min_depth=self.tri_min_depth,
            max_depth=self.tri_max_depth,
        )


class OpenCVVOBackend(SLAMBackend):
    """Frame-to-frame monocular VO with keyframe-based sparse mapping."""

    name = "opencv_vo"

    def __init__(self, config: OpenCVVOConfig | None = None) -> None:
        self.config = config or OpenCVVOConfig()
        self._intrinsics: CameraIntrinsics | None = None
        self._detector = cv2.ORB_create(nfeatures=self.config.n_features)
        self._bf = cv2.BFMatcher(cv2.NORM_HAMMING)

        self._trajectory = Trajectory()
        self._map_points: list[MapPoint] = []
        self._keyframes: list[KeyFrame] = []

        self._prev_keypoints = None
        self._prev_descriptors = None
        self._pose = CameraPose.identity()

        self._kf_keypoints = None
        self._kf_descriptors = None
        self._kf_pose: CameraPose | None = None

        self._selector = KeyframeSelector(self.config.keyframe_policy())
        self._frames_processed = 0
        self._frames_tracked = 0
        self._initialized = False

    def initialize(self, intrinsics: CameraIntrinsics) -> None:
        self._intrinsics = intrinsics
        self._trajectory = Trajectory()
        self._map_points = []
        self._keyframes = []
        self._prev_keypoints = None
        self._prev_descriptors = None
        self._pose = CameraPose.identity()
        self._kf_keypoints = None
        self._kf_descriptors = None
        self._kf_pose = None
        self._selector = KeyframeSelector(self.config.keyframe_policy())
        self._frames_processed = 0
        self._frames_tracked = 0
        self._initialized = True
        logger.info(
            "OpenCVVO ready: %dx%d fx=%.2f fy=%.2f",
            intrinsics.width,
            intrinsics.height,
            intrinsics.fx,
            intrinsics.fy,
        )

    def process_frame(self, frame: Frame) -> None:
        if not self._initialized or self._intrinsics is None:
            raise RuntimeError("call initialize() before process_frame()")

        self._frames_processed += 1
        gray = self._to_gray(frame.image)
        keypoints, descriptors = self._detector.detectAndCompute(gray, None)
        n_kp = 0 if keypoints is None else len(keypoints)

        if descriptors is None or n_kp < self.config.min_matches:
            logger.debug("Frame %d: few features (%d)", frame.index, n_kp)
            self._append_current_pose(frame)
            self._set_prev(keypoints, descriptors)
            return

        if self._prev_descriptors is None:
            self._pose = CameraPose.identity(frame_id=frame.index, timestamp=frame.timestamp)
            self._append_current_pose(frame)
            self._maybe_keyframe(frame, keypoints, descriptors, n_matches=0, tracked=False)
            self._set_prev(keypoints, descriptors)
            return

        matches = self._match(self._prev_descriptors, descriptors)
        n_matches = len(matches)
        if n_matches < self.config.min_matches:
            logger.debug("Frame %d: few matches (%d)", frame.index, n_matches)
            self._append_current_pose(frame)
            self._maybe_keyframe(frame, keypoints, descriptors, n_matches, tracked=False)
            self._set_prev(keypoints, descriptors)
            return

        pts_prev = np.float64([self._prev_keypoints[m.queryIdx].pt for m in matches])
        pts_cur = np.float64([keypoints[m.trainIdx].pt for m in matches])
        K = self._intrinsics.K

        E, mask = cv2.findEssentialMat(
            pts_prev,
            pts_cur,
            K,
            method=cv2.RANSAC,
            prob=self.config.ransac_confidence,
            threshold=self.config.ransac_thresh_px,
        )
        if E is None or mask is None:
            logger.debug("Frame %d: findEssentialMat failed", frame.index)
            self._append_current_pose(frame)
            self._maybe_keyframe(frame, keypoints, descriptors, n_matches, tracked=False)
            self._set_prev(keypoints, descriptors)
            return

        mask_u8 = mask.reshape(-1, 1).astype(np.uint8)
        n_inliers, R, t, pose_mask = cv2.recoverPose(E, pts_prev, pts_cur, K, mask=mask_u8)
        if int(n_inliers) < self.config.min_inliers or R is None or t is None:
            logger.debug("Frame %d: recoverPose weak (%d)", frame.index, int(n_inliers))
            self._append_current_pose(frame)
            self._maybe_keyframe(frame, keypoints, descriptors, n_matches, tracked=False)
            self._set_prev(keypoints, descriptors)
            return

        T_cur_prev = np.eye(4, dtype=np.float64)
        T_cur_prev[:3, :3] = R
        T_cur_prev[:3, 3] = t.reshape(3)
        t_norm = float(np.linalg.norm(t))
        if t_norm > 1e-8:
            T_cur_prev[:3, 3] *= self.config.default_translation_norm / t_norm

        T_prev_cur = np.linalg.inv(T_cur_prev)
        T_world_cur = self._pose.matrix @ T_prev_cur
        self._pose = CameraPose(T_world_cur, frame_id=frame.index, timestamp=frame.timestamp)
        self._frames_tracked += 1
        self._append_current_pose(frame)

        was_kf = self._maybe_keyframe(frame, keypoints, descriptors, n_matches, tracked=True)
        if was_kf:
            self._triangulate_kf_to_cur(frame, keypoints, descriptors)

        self._set_prev(keypoints, descriptors)

    def get_result(self) -> MapResult:
        assert self._intrinsics is not None
        return MapResult(
            trajectory=self._trajectory,
            map_points=self._map_points,
            intrinsics=self._intrinsics,
            metadata={
                "stage": "V0.3-slam-mvp",
                "slam_backend": self.name,
                "slam_status": "ok",
                "frames_processed": self._frames_processed,
                "frames_tracked": self._frames_tracked,
                "num_keyframes": len(self._keyframes),
                "track_rate": (
                    self._frames_tracked / self._frames_processed
                    if self._frames_processed
                    else 0.0
                ),
                "scale": "monocular_arbitrary",
                "default_translation_norm": self.config.default_translation_norm,
            },
        )

    def shutdown(self) -> None:
        self._prev_keypoints = None
        self._prev_descriptors = None
        self._kf_keypoints = None
        self._kf_descriptors = None

    # ----- internals -----

    def _to_gray(self, image: np.ndarray) -> np.ndarray:
        if image.ndim == 2:
            return image
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def _match(self, desc_a: np.ndarray, desc_b: np.ndarray) -> list:
        knn = self._bf.knnMatch(desc_a, desc_b, k=2)
        good = []
        for pair in knn:
            if len(pair) < 2:
                continue
            m, n = pair
            if m.distance < self.config.match_ratio * n.distance:
                good.append(m)
        return good

    def _append_current_pose(self, frame: Frame) -> None:
        pose = CameraPose(
            self._pose.matrix.copy(),
            frame_id=frame.index,
            timestamp=frame.timestamp,
        )
        self._trajectory.append(pose)

    def _set_prev(self, keypoints, descriptors) -> None:
        self._prev_keypoints = keypoints
        self._prev_descriptors = descriptors

    def _maybe_keyframe(
        self,
        frame: Frame,
        keypoints,
        descriptors,
        n_matches: int,
        tracked: bool,
    ) -> bool:
        pose = CameraPose(self._pose.matrix.copy(), frame_id=frame.index, timestamp=frame.timestamp)
        if not self._selector.should_insert(frame, pose, n_matches, tracked):
            return False

        self._keyframes.append(
            KeyFrame(frame=frame, pose=pose, keypoints=keypoints, descriptors=descriptors)
        )
        # Keep KF features as triangulation reference (previous KF before overwrite)
        self._kf_keypoints = keypoints
        self._kf_descriptors = descriptors
        self._kf_pose = pose
        return True

    def _triangulate_kf_to_cur(self, frame: Frame, keypoints, descriptors) -> None:
        """Triangulate between the previous keyframe and the newly inserted one.

        Call only AFTER a new keyframe was inserted. We need the *previous*
        keyframe features — those are stored just before this KF was added,
        so we re-derive from the penultimate keyframe entry.
        """
        if self._intrinsics is None or len(self._keyframes) < 2:
            return
        if len(self._map_points) >= self.config.max_map_points:
            return

        prev_kf = self._keyframes[-2]
        cur_kf = self._keyframes[-1]
        if prev_kf.descriptors is None or cur_kf.descriptors is None:
            return
        if prev_kf.keypoints is None or cur_kf.keypoints is None:
            return

        matches = self._match(prev_kf.descriptors, cur_kf.descriptors)
        if len(matches) < self.config.min_matches:
            return

        pts_a = np.float64([prev_kf.keypoints[m.queryIdx].pt for m in matches])
        pts_b = np.float64([cur_kf.keypoints[m.trainIdx].pt for m in matches])
        ids_a = np.int64([m.queryIdx for m in matches])
        ids_b = np.int64([m.trainIdx for m in matches])

        points = triangulate_two_view(
            pts_a,
            pts_b,
            prev_kf.pose,
            cur_kf.pose,
            self._intrinsics,
            config=self.config.triangulation_config(),
            start_id=len(self._map_points),
            feature_ids_a=ids_a,
            feature_ids_b=ids_b,
            frame_a=prev_kf.frame.index,
            frame_b=cur_kf.frame.index,
        )
        room = self.config.max_map_points - len(self._map_points)
        self._map_points.extend(points[:room])
        logger.debug(
            "KF %d→%d triangulated %d points (total %d)",
            prev_kf.frame.index,
            cur_kf.frame.index,
            min(len(points), room),
            len(self._map_points),
        )


def create_backend(name: str, **kwargs) -> SLAMBackend:
    """Factory for app.py. Business code must not import concrete backends."""
    key = name.strip().lower()
    if key in ("opencv_sfm", "pnp_vo", "sfm"):
        from visualmap.slam.backends.opencv_sfm import OpenCVSFMBackend, OpenCVSFMConfig

        known = {f for f in OpenCVSFMConfig.__dataclass_fields__ if f != "extra"}
        cfg_kwargs = {k: v for k, v in kwargs.items() if k in known}
        extra = {k: v for k, v in kwargs.items() if k not in known}
        return OpenCVSFMBackend(OpenCVSFMConfig(**cfg_kwargs, extra=extra))
    if key in ("opencv_vo", "opencv", "vo"):
        # Accept OpenCVVOConfig field names; ignore unknown keys into `extra`
        known = {f for f in OpenCVVOConfig.__dataclass_fields__ if f != "extra"}
        cfg_kwargs = {k: v for k, v in kwargs.items() if k in known}
        extra = {k: v for k, v in kwargs.items() if k not in known}
        return OpenCVVOBackend(OpenCVVOConfig(**cfg_kwargs, extra=extra))
    if key in ("orb_slam3", "orbslam3"):
        raise NotImplementedError(
            "ORB-SLAM3 backend is planned for V0.3+ via WSL/Docker. "
            "Use slam.backend: opencv_vo for now."
        )
    raise ValueError(f"unknown slam backend: {name!r}")
