"""Single-map monocular odometry with persistent KLT landmarks and PnP.

Only the initial two-view bootstrap chooses an arbitrary scale. All later
camera translations are estimated against that same map, including after
tracking loss. No essential-matrix step is appended with a made-up length.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real

import cv2
import numpy as np

from visualmap.mapping.triangulation import triangulate_two_view
from visualmap.slam.backends.opencv_vo import OpenCVVOConfig
from visualmap.slam.base import SLAMBackend
from visualmap.slam.types import (
    CameraIntrinsics,
    CameraPose,
    Frame,
    MapPoint,
    MapResult,
    Trajectory,
)


@dataclass
class OpenCVSFMConfig(OpenCVVOConfig):
    """Legacy VO fields remain accepted; translation norm sets bootstrap only."""

    tracking_max_dimension: int = 960
    feature_min_distance: int = 9
    klt_fb_threshold_px: float = 1.0
    bootstrap_min_parallax_deg: float = 1.2
    bootstrap_min_rotation_compensated_px: float = 2.0
    bootstrap_min_landmarks: int = 60
    bootstrap_history_frames: int = 120
    pnp_reprojection_px: float = 3.0
    pnp_min_inlier_ratio: float = 0.45
    keyframe_max_gap: int = 15
    relocalization_interval: int = 5
    relocalization_keyframes: int = 8
    landmark_descriptor_history: int = 3
    recovery_submaps: bool = True
    submap_after_lost_frames: int = 30
    # Fractions of sensor area and sensor-normalized minor-axis span. A tiny
    # patch or near-line can fit PnP pixels well while its translation is unstable.
    pose_min_image_area_ratio: float = 0.001
    pose_min_image_minor_span_ratio: float = 0.01

    def __post_init__(self) -> None:
        for name in (
            "relocalization_keyframes", "landmark_descriptor_history", "submap_after_lost_frames"
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not isinstance(self.recovery_submaps, bool):
            raise ValueError("recovery_submaps must be a boolean")
        for name in ("pose_min_image_area_ratio", "pose_min_image_minor_span_ratio"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not np.isfinite(value)
                or not 0 <= value < 1
            ):
                raise ValueError(f"{name} must be finite and in [0, 1)")


@dataclass
class _Track:
    id: int
    pixel: np.ndarray
    anchor_pixel: np.ndarray
    anchor_pose: CameraPose
    anchor_frame: int
    landmark_id: int | None = None


class OpenCVSFMBackend(SLAMBackend):
    """CPU-only, one world map; failure holds the last verified camera pose."""

    name = "opencv_sfm"

    def __init__(self, config: OpenCVSFMConfig | None = None) -> None:
        self.config = config or OpenCVSFMConfig()
        self._intrinsics: CameraIntrinsics | None = None
        self._orb = cv2.ORB_create(nfeatures=self.config.n_features)
        self._bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        self._reset()

    def _reset(self) -> None:
        self._trajectory = Trajectory()
        self._pose = CameraPose.identity()
        self._tracks: dict[int, _Track] = {}
        self._landmarks: dict[int, MapPoint] = {}
        self._observation_pixels: dict[int, dict[int, np.ndarray]] = {}
        self._poses: dict[int, CameraPose] = {}
        self._quality: dict[str, dict] = {}
        self._next_track_id = 0
        self._next_landmark_id = 0
        self._prev_gray = None
        self._image_scale = 1.0
        self._bootstrapped = False
        self._bootstrap_anchor: CameraPose | None = None
        self._bootstrap_history: list[tuple[int, float, dict[int, np.ndarray]]] = []
        self._last_keyframe: CameraPose | None = None
        self._num_keyframes = 0
        self._frames_tracked = 0
        self._frames_processed = 0
        self._lost_frames = 0
        self._world_frame_id: str | None = None
        self._input_invalid_frames = 0
        self._descriptor_history: dict[int, list[np.ndarray]] = {}
        self._relocalization_keyframes: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        self._archived_landmarks: list[MapPoint] = []
        self._scale_initializations = 0
        self._submaps: list[dict] = []

    def initialize(self, intrinsics: CameraIntrinsics) -> None:
        self._intrinsics = intrinsics
        self._reset()

    def process_frame(self, frame: Frame) -> None:
        if self._intrinsics is None:
            raise RuntimeError("call initialize() before process_frame()")
        self._frames_processed += 1
        reason = self._frame_input_reason(frame)
        if reason is not None:
            # A resolution change must not rescale old KLT coordinates using the
            # new sensor size. Keep the existing world, but remove associations.
            self._tracks = {}
            self._prev_gray = None
            if not self._bootstrapped:
                self._bootstrap_anchor = None
                self._bootstrap_history = []
            self._input_invalid_frames += 1
            self._lost_frames += 1
            self._append_pose_quality(frame, False, "input_invalid", {"reason": reason})
            return
        gray = self._prepare_image(frame.image)
        if (
            self._bootstrapped
            and self.config.recovery_submaps
            and self._lost_frames >= max(1, self.config.submap_after_lost_frames)
        ):
            # A new view may share no verified landmarks with the old world.
            # Bootstrap an explicitly independent map instead of freezing for
            # the rest of the video or inventing a transform/scale between maps.
            self._start_independent_submap()
        if self._prev_gray is not None:
            self._track_features(gray)

        diagnostics: dict = {}
        valid = False
        if not self._bootstrapped:
            if self._bootstrap_anchor is None or len(self._tracks) < self.config.min_matches:
                self._tracks = {}
                self._bootstrap_anchor = CameraPose.identity(frame.index, frame.timestamp)
                self._bootstrap_history = []
                self._add_features(gray, self._bootstrap_anchor)
            self._bootstrap_history.append(
                (
                    frame.index,
                    frame.timestamp,
                    {fid: t.pixel.copy() for fid, t in self._tracks.items()},
                )
            )
            history_limit = max(2, self.config.bootstrap_history_frames)
            if len(self._bootstrap_history) > history_limit:
                self._bootstrap_history = [
                    self._bootstrap_history[0],
                    *self._bootstrap_history[-(history_limit - 1) :],
                ]
            valid, diagnostics = self._bootstrap(frame)
            status = "bootstrap" if valid else "bootstrap_pending"
            if valid:
                self._add_features(gray, self._pose)
                self._save_descriptors(gray)
        else:
            valid, diagnostics, inlier_ids = self._track_map_pose(frame)
            status = "tracked" if valid else "tracking_lost"
            if not valid and self._lost_frames % max(1, self.config.relocalization_interval) == 0:
                valid, diagnostics = self._relocalize_keyframes(gray, frame)
                if not valid:
                    valid, diagnostics = self._relocalize(gray, frame)
                if valid:
                    status = "relocalized"
                    inlier_ids = list(self._tracks)
            if valid:
                self._record_observations(frame.index, inlier_ids)
                if self._should_keyframe(frame, len(inlier_ids)):
                    self._triangulate_new_tracks(frame)
                    self._add_features(gray, self._pose)
                    self._save_descriptors(gray)
                    self._last_keyframe = self._pose
                    self._num_keyframes += 1

        if valid:
            self._lost_frames = 0
            self._frames_tracked += 1
        else:
            self._lost_frames += 1
        self._append_pose_quality(frame, valid, status, diagnostics)
        self._prev_gray = gray

    def _append_pose_quality(
        self, frame: Frame, valid: bool, status: str, diagnostics: dict
    ) -> None:
        pose = CameraPose(self._pose.matrix, frame.index, frame.timestamp)
        self._trajectory.append(pose)
        self._poses[frame.index] = pose
        self._quality[str(frame.index)] = {
            "status": status,
            "pose_valid": valid,
            "world_frame_id": self._world_frame_id,
            "tracked_features": len(self._tracks),
            **diagnostics,
        }

    def _frame_input_reason(self, frame: Frame) -> str | None:
        if (frame.width, frame.height) != (self._intrinsics.width, self._intrinsics.height):
            return "camera_intrinsics_frame_size_mismatch"
        # The generic Frame also accepts finite float images. This CPU KLT
        # backend requires uint8; guessing its range would alter the evidence.
        if frame.image.dtype != np.uint8:
            return "unsupported_sfm_image_dtype_requires_uint8"
        return None

    def get_result(self) -> MapResult:
        if self._intrinsics is None:
            raise RuntimeError("call initialize() before get_result()")
        status = "ok" if self._bootstrapped else "unobservable"
        if self._bootstrapped and self._lost_frames:
            status = "tracking_lost"
        initializations = max(self._scale_initializations, int(self._bootstrapped))
        result = MapResult(
            trajectory=self._trajectory,
            map_points=[*self._archived_landmarks, *self._landmarks.values()],
            intrinsics=self._intrinsics,
            metadata={
                "stage": "persistent-landmark-sfm",
                "slam_backend": self.name,
                "slam_status": status,
                "frames_processed": self._frames_processed,
                "frames_tracked": self._frames_tracked,
                "input_invalid_frames": self._input_invalid_frames,
                "num_keyframes": self._num_keyframes,
                "track_rate": self._frames_tracked / max(1, self._frames_processed),
                "scale": "monocular_arbitrary",
                "scale_consistent": initializations > 0,
                "scale_consistency_scope": "within_world_frame_id",
                "global_scale_consistent": initializations == 1,
                "coordinate_scope": "within_world_frame_id",
                "scale_initializations": initializations,
                "submaps": self._submaps,
                "submap_count": len(self._submaps),
                "submaps_registered": len(self._submaps) <= 1,
                "world_frame_id": self._world_frame_id,
                "pose_support_policy": {
                    "min_image_area_ratio": self.config.pose_min_image_area_ratio,
                    "min_image_minor_span_ratio": self.config.pose_min_image_minor_span_ratio,
                    "basis": "sensor_normalized_2d_support_not_3d_planarity",
                },
                "bootstrap_translation_norm": self.config.default_translation_norm,
                "frame_quality": self._quality,
                "map_point_observations": sum(
                    len(p.observations)
                    for p in [*self._archived_landmarks, *self._landmarks.values()]
                ),
                "geometry_note": (
                    "Each world_frame_id has an independent arbitrary scale. Submaps are not "
                    "registered and must not be compared or drawn as one world. "
                    "Failed poses are held and invalid."
                ),
            },
        )
        from visualmap.slam.coverage import geometry_coverage, landmark_world

        result.metadata["landmark_world_frame_ids"] = {
            str(point.id): landmark_world(result, point) for point in result.map_points
        }
        result.metadata["geometry_coverage"] = geometry_coverage(result)
        return result

    def shutdown(self) -> None:
        self._prev_gray = None

    def _start_independent_submap(self) -> None:
        self._archived_landmarks.extend(self._landmarks.values())
        self._landmarks = {}
        self._observation_pixels = {}
        self._descriptor_history = {}
        self._relocalization_keyframes = []
        self._tracks = {}
        self._prev_gray = None
        self._bootstrapped = False
        self._bootstrap_anchor = None
        self._bootstrap_history = []
        self._last_keyframe = None
        self._world_frame_id = None
        self._pose = CameraPose.identity()
        self._lost_frames = 0

    def _prepare_image(self, image: np.ndarray) -> np.ndarray:
        gray = (
            image.reshape(image.shape[:2])
            if image.ndim == 2 or image.shape[2] == 1
            else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        )
        limit = self.config.tracking_max_dimension
        self._image_scale = min(1.0, limit / max(gray.shape)) if limit > 0 else 1.0
        if self._image_scale < 1.0:
            gray = cv2.resize(gray, None, fx=self._image_scale, fy=self._image_scale)
        return np.ascontiguousarray(gray)

    def _add_features(self, gray: np.ndarray, pose: CameraPose) -> None:
        room = self.config.n_features - len(self._tracks)
        if room <= 0:
            return
        mask = np.full(gray.shape, 255, np.uint8)
        for track in self._tracks.values():
            xy = tuple(np.rint(track.pixel * self._image_scale).astype(int))
            cv2.circle(mask, xy, self.config.feature_min_distance, 0, -1)
        corners = cv2.goodFeaturesToTrack(
            gray,
            maxCorners=room,
            qualityLevel=0.01,
            minDistance=self.config.feature_min_distance,
            mask=mask,
            blockSize=7,
        )
        if corners is None:
            return
        for pixel in corners.reshape(-1, 2).astype(np.float64) / self._image_scale:
            fid = self._next_track_id
            self._next_track_id += 1
            self._tracks[fid] = _Track(fid, pixel.copy(), pixel.copy(), pose, pose.frame_id)

    def _track_features(self, gray: np.ndarray) -> None:
        if not self._tracks or self._prev_gray.shape != gray.shape:
            self._tracks = {}
            return
        ids = list(self._tracks)
        before = np.float32([self._tracks[fid].pixel * self._image_scale for fid in ids])
        before = before.reshape(-1, 1, 2)
        pyramid_level = max(0, min(4, int(np.log2(min(gray.shape) / 50))))
        params = dict(
            winSize=(31, 31),
            maxLevel=pyramid_level,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )
        after, status, _ = cv2.calcOpticalFlowPyrLK(self._prev_gray, gray, before, None, **params)
        if after is None:
            self._tracks = {}
            return
        back, back_status, _ = cv2.calcOpticalFlowPyrLK(
            gray, self._prev_gray, after, None, **params
        )
        if back is None:
            self._tracks = {}
            return
        fb_error = np.linalg.norm(back.reshape(-1, 2) - before.reshape(-1, 2), axis=1)
        xy = after.reshape(-1, 2)
        valid = (
            status.ravel().astype(bool)
            & back_status.ravel().astype(bool)
            & (fb_error <= self.config.klt_fb_threshold_px)
            & np.isfinite(xy).all(axis=1)
            & (xy[:, 0] >= 8)
            & (xy[:, 0] < gray.shape[1] - 8)
            & (xy[:, 1] >= 8)
            & (xy[:, 1] < gray.shape[0] - 8)
        )
        self._tracks = {fid: self._tracks[fid] for fid, ok in zip(ids, valid, strict=True) if ok}
        for fid, pixel in zip(ids, xy, strict=True):
            if fid in self._tracks:
                self._tracks[fid].pixel = pixel.astype(np.float64) / self._image_scale

    def _bearings(self, pixels: np.ndarray) -> np.ndarray:
        return cv2.undistortPoints(
            np.float64(pixels).reshape(-1, 1, 2),
            self._intrinsics.K,
            self._intrinsics.dist_coeffs,
        ).reshape(-1, 2)

    def _image_support(self, pixels: np.ndarray) -> tuple[bool, dict]:
        """Measure two-dimensional support, without rejecting planar 3D scenes."""
        if len(pixels) < 3:
            return False, {"image_support_area_ratio": 0.0, "image_support_minor_span_ratio": 0.0}
        normal = np.float64(pixels) / [self._intrinsics.width, self._intrinsics.height]
        hull = cv2.convexHull(np.float32(normal))
        area = float(abs(cv2.contourArea(hull)))
        singular = np.linalg.svd(normal - normal.mean(axis=0), compute_uv=False)
        # Four standard deviations describe the minor-axis spread. This is a
        # degeneracy guard, not a camera covariance or a calibrated error bound.
        minor_span = float(4 * singular[-1] / np.sqrt(len(normal)))
        diagnostics = {
            "image_support_area_ratio": area,
            "image_support_minor_span_ratio": minor_span,
        }
        valid = (
            area >= self.config.pose_min_image_area_ratio
            and minor_span >= self.config.pose_min_image_minor_span_ratio
        )
        return valid, diagnostics

    def _bootstrap(self, frame: Frame) -> tuple[bool, dict]:
        anchor = self._bootstrap_anchor
        candidates = [t for t in self._tracks.values() if t.anchor_frame == anchor.frame_id]
        if len(candidates) < self.config.min_matches or frame.index == anchor.frame_id:
            return False, {"reason": "waiting_for_baseline"}
        pixels_a = np.float64([t.anchor_pixel for t in candidates])
        pixels_b = np.float64([t.pixel for t in candidates])
        a, b = self._bearings(pixels_a), self._bearings(pixels_b)
        # A robust rotation-only fit guards pure rotation. It cannot supply depth.
        rays_a = np.column_stack((a, np.ones(len(a))))
        rays_b = np.column_stack((b, np.ones(len(b))))
        rays_a /= np.linalg.norm(rays_a, axis=1)[:, None]
        rays_b /= np.linalg.norm(rays_b, axis=1)[:, None]
        keep = np.ones(len(a), dtype=bool)
        residual = np.zeros(len(a))
        focal = float((self._intrinsics.fx + self._intrinsics.fy) / 2)
        for _ in range(3):
            U, _, Vt = np.linalg.svd(rays_b[keep].T @ rays_a[keep])
            R_rot = U @ np.diag([1.0, 1.0, np.linalg.det(U @ Vt)]) @ Vt
            predicted = (R_rot @ rays_a.T).T
            residual = np.linalg.norm(predicted[:, :2] / predicted[:, 2:3] - b, axis=1) * focal
            keep = residual <= max(1.0, float(np.quantile(residual, 0.8)))
        rotation_compensated = float(np.median(residual))
        diagnostics = {"rotation_compensated_motion_px": rotation_compensated}
        if rotation_compensated < self.config.bootstrap_min_rotation_compensated_px:
            return False, {**diagnostics, "reason": "rotation_or_insufficient_parallax"}
        threshold = self.config.ransac_thresh_px / focal
        E, mask = cv2.findEssentialMat(
            a,
            b,
            np.eye(3),
            method=cv2.RANSAC,
            prob=self.config.ransac_confidence,
            threshold=threshold,
        )
        if E is None or mask is None:
            return False, {**diagnostics, "reason": "essential_estimation_failed"}
        best = None
        for essential in np.asarray(E).reshape(-1, 3, 3):
            n, R, t, pose_mask = cv2.recoverPose(essential, a, b, np.eye(3), mask=mask.copy())
            if best is None or n > best[0]:
                best = n, R, t, pose_mask
        n, R, t, pose_mask = best
        if n < max(self.config.min_inliers, self.config.bootstrap_min_landmarks):
            return False, {**diagnostics, "reason": "too_few_cheirality_inliers", "inliers": int(n)}
        T_ca = np.eye(4)
        T_ca[:3, :3] = R
        T_ca[:3, 3] = t.ravel() * self.config.default_translation_norm
        pose = CameraPose(anchor.matrix @ np.linalg.inv(T_ca), frame.index, frame.timestamp)
        good = pose_mask.ravel() != 0
        ids = np.int64([track.id for track in candidates])[good]
        points = triangulate_two_view(
            pixels_a[good],
            pixels_b[good],
            anchor,
            pose,
            self._intrinsics,
            config=self.config.triangulation_config(),
            feature_ids_a=ids,
            feature_ids_b=ids,
            frame_a=anchor.frame_id,
            frame_b=frame.index,
        )
        if len(points) < self.config.bootstrap_min_landmarks:
            return False, {**diagnostics, "reason": "insufficient_triangulated_landmarks"}
        pixels_by_id = {track.id: track.pixel for track in candidates}
        support_valid, support_quality = self._image_support(
            np.float64([pixels_by_id[point.observations[frame.index]] for point in points])
        )
        diagnostics.update(support_quality)
        if not support_valid:
            return False, {**diagnostics, "reason": "bootstrap_insufficient_image_support"}
        angles = []
        for point in points:
            ra, rb = point.xyz - anchor.t, point.xyz - pose.t
            angles.append(
                np.degrees(
                    np.arccos(
                        np.clip(
                            np.dot(ra, rb) / (np.linalg.norm(ra) * np.linalg.norm(rb)),
                            -1,
                            1,
                        )
                    )
                )
            )
        median_parallax = float(np.median(angles))
        if median_parallax < self.config.bootstrap_min_parallax_deg:
            return False, {
                **diagnostics,
                "reason": "insufficient_parallax",
                "median_parallax_deg": median_parallax,
            }
        self._pose = pose
        self._bootstrapped = True
        self._world_frame_id = f"sfm_bootstrap_{anchor.frame_id}"
        self._scale_initializations += 1
        self._submaps.append(
            {
                "world_frame_id": self._world_frame_id,
                "anchor_frame": anchor.frame_id,
                "bootstrap_frame": frame.index,
                "scale": "independent_monocular_arbitrary",
                "registered_to_previous": False,
            }
        )
        self._poses[anchor.frame_id] = anchor
        self._poses[frame.index] = pose
        for point in points[: self.config.max_map_points]:
            fid = point.observations[frame.index]
            self._insert_landmark(point, self._tracks[fid])
        self._recover_bootstrap_history()
        self._last_keyframe = pose
        self._num_keyframes += 2
        return True, {
            **diagnostics,
            "inliers": len(points),
            "correspondences": len(candidates),
            "median_parallax_deg": median_parallax,
            "positive_depth_ratio": 1.0,
        }

    def _insert_landmark(self, point: MapPoint, track: _Track) -> None:
        point.id = self._next_landmark_id
        self._next_landmark_id += 1
        track.landmark_id = point.id
        self._landmarks[point.id] = point
        self._observation_pixels[point.id] = {
            track.anchor_frame: track.anchor_pixel.copy(),
            self._pose.frame_id: track.pixel.copy(),
        }

    def _recover_bootstrap_history(self) -> None:
        # Early poses are solved against the same bootstrap map, never by
        # distributing its baseline uniformly across video time.
        by_frame = {p.frame_id: i for i, p in enumerate(self._trajectory.poses)}
        for fid, timestamp, pixels in self._bootstrap_history:
            if fid == self._pose.frame_id:
                continue
            if fid == self._bootstrap_anchor.frame_id:
                pose = self._bootstrap_anchor
                inlier_tracks = list(pixels)
                diagnostics = {}
            else:
                tracks = [
                    t for t in self._tracks.values() if t.landmark_id is not None and t.id in pixels
                ]
                if len(tracks) < self.config.min_inliers:
                    continue
                xyz = np.float64([self._landmarks[t.landmark_id].xyz for t in tracks])
                xy = np.float64([pixels[t.id] for t in tracks])
                ok, pose, selected, diagnostics = self._estimate_pose(
                    xyz,
                    xy,
                    fid,
                    timestamp,
                    np.int64([t.landmark_id for t in tracks]),
                )
                if not ok:
                    continue
                inlier_tracks = [tracks[i].id for i in selected]
            self._poses[fid] = pose
            if fid in by_frame:
                self._trajectory.poses[by_frame[fid]] = pose
                self._quality[str(fid)] = {
                    "status": "bootstrap_anchor"
                    if fid == self._bootstrap_anchor.frame_id
                    else "tracked",
                    "pose_valid": True,
                    "world_frame_id": self._world_frame_id,
                    "retroactively_estimated": True,
                    **diagnostics,
                }
                self._frames_tracked += 1
            for tid in inlier_tracks:
                track = self._tracks.get(tid)
                if track is not None and track.landmark_id is not None:
                    point = self._landmarks[track.landmark_id]
                    point.observations[fid] = tid
                    self._observation_pixels[point.id][fid] = pixels[tid].copy()
        self._bootstrap_history = []

    def _estimate_pose(
        self,
        xyz: np.ndarray,
        pixels: np.ndarray,
        frame_id: int,
        timestamp: float,
        landmark_ids: np.ndarray | None = None,
    ):
        xyz = np.asarray(xyz, dtype=np.float64)
        pixels = np.asarray(pixels, dtype=np.float64)
        count = len(xyz)
        diagnostics = {"correspondences": count, "inliers": 0}
        if (
            xyz.shape != (count, 3)
            or pixels.shape != (count, 2)
            or not np.isfinite(xyz).all()
            or not np.isfinite(pixels).all()
        ):
            return (
                False,
                None,
                np.array([], dtype=int),
                {**diagnostics, "reason": "invalid_map_correspondences"},
            )
        if count < max(6, self.config.min_inliers):
            return (
                False,
                None,
                np.array([], dtype=int),
                {**diagnostics, "reason": "too_few_map_matches"},
            )
        support_valid, support_quality = self._image_support(pixels)
        diagnostics.update(support_quality)
        if not support_valid:
            return (
                False,
                None,
                np.array([], dtype=int),
                {**diagnostics, "reason": "pnp_insufficient_image_support"},
            )
        # Every fifth correspondence is withheld from pose fitting. Report its
        # error separately; fitting reprojection alone cannot measure prediction.
        ids = np.arange(count) if landmark_ids is None else np.asarray(landmark_ids)
        holdout = ids % 5 == 0 if count >= 50 else np.zeros(count, dtype=bool)
        if (~holdout).sum() < max(6, self.config.min_inliers):
            holdout[:] = False
        train_indices = np.flatnonzero(~holdout)
        ok, rvec, tvec, inliers = cv2.solvePnPRansac(
            xyz[~holdout],
            pixels[~holdout],
            self._intrinsics.K,
            self._intrinsics.dist_coeffs,
            iterationsCount=150,
            reprojectionError=self.config.pnp_reprojection_px,
            confidence=0.999,
            flags=cv2.SOLVEPNP_EPNP,
        )
        n_inliers = 0 if inliers is None else len(inliers)
        diagnostics["inliers"] = n_inliers
        if (
            not ok
            or n_inliers < self.config.min_inliers
            or (n_inliers / len(train_indices) < self.config.pnp_min_inlier_ratio)
        ):
            return (
                False,
                None,
                np.array([], dtype=int),
                {**diagnostics, "reason": "pnp_ransac_failed"},
            )
        fit = train_indices[inliers.ravel()]
        rvec, tvec = cv2.solvePnPRefineLM(
            xyz[fit],
            pixels[fit],
            self._intrinsics.K,
            self._intrinsics.dist_coeffs,
            rvec,
            tvec,
        )
        projected, _ = cv2.projectPoints(
            xyz,
            rvec,
            tvec,
            self._intrinsics.K,
            self._intrinsics.dist_coeffs,
        )
        error = np.linalg.norm(projected.reshape(-1, 2) - pixels, axis=1)
        R, _ = cv2.Rodrigues(rvec)
        depth = (R @ xyz.T).T[:, 2] + tvec.ravel()[2]
        accepted = (error <= self.config.pnp_reprojection_px) & (depth > self.config.tri_min_depth)
        holdout_count = int(holdout.sum())
        heldout_error = float(np.median(error[holdout])) if holdout_count else None
        diagnostics.update(
            {
                "inliers": int(accepted.sum()),
                "median_reprojection_px": float(np.median(error[fit])),
                "inlier_reprojection_rms_px": float(np.sqrt(np.mean(error[fit] ** 2))),
                "holdout_count": holdout_count,
                "holdout_median_reprojection_px": heldout_error,
                "heldout_count": holdout_count,
                "heldout_reprojection_median_px": heldout_error,
                "holdout_inlier_ratio": float(accepted[holdout].mean()) if holdout_count else None,
                "positive_depth_ratio": float((depth > 0).mean()),
            }
        )
        # Refinement can move RANSAC's boundary inliers beyond the threshold.
        # Check the final independently reprojected points, not the old mask.
        accepted_train = int(accepted[~holdout].sum())
        diagnostics["refined_train_inlier_ratio"] = accepted_train / len(train_indices)
        if (
            accepted_train < self.config.min_inliers
            or accepted_train / len(train_indices) < self.config.pnp_min_inlier_ratio
        ):
            return (
                False,
                None,
                np.array([], dtype=int),
                {**diagnostics, "reason": "pnp_refined_support_failed"},
            )
        support_valid, support_quality = self._image_support(pixels[accepted])
        diagnostics.update(support_quality)
        if not support_valid:
            return (
                False,
                None,
                np.array([], dtype=int),
                {**diagnostics, "reason": "pnp_insufficient_image_support"},
            )
        if (
            not np.isfinite(rvec).all()
            or not np.isfinite(tvec).all()
            or (depth[fit] > 0).mean() < 0.98
            or (holdout_count >= 8 and heldout_error > 2 * self.config.pnp_reprojection_px)
        ):
            return (
                False,
                None,
                np.array([], dtype=int),
                {**diagnostics, "reason": "pnp_validation_failed"},
            )
        T_cw = np.eye(4)
        T_cw[:3, :3] = R
        T_cw[:3, 3] = tvec.ravel()
        pose = CameraPose(np.linalg.inv(T_cw), frame_id, timestamp)
        return True, pose, np.flatnonzero(accepted), diagnostics

    def _track_map_pose(self, frame: Frame):
        tracks = [t for t in self._tracks.values() if t.landmark_id in self._landmarks]
        xyz = np.float64([self._landmarks[t.landmark_id].xyz for t in tracks]).reshape(-1, 3)
        pixels = np.float64([t.pixel for t in tracks]).reshape(-1, 2)
        ok, pose, inliers, diagnostics = self._estimate_pose(
            xyz,
            pixels,
            frame.index,
            frame.timestamp,
            np.int64([t.landmark_id for t in tracks]),
        )
        if ok:
            self._pose = pose
            self._poses[frame.index] = pose
            selected = set(inliers)
            # Reject flow tracks that fail the same-world geometric check. They
            # cannot keep contaminating PnP or become a second copy of a point.
            for index, track in enumerate(tracks):
                if index not in selected:
                    self._tracks.pop(track.id, None)
        return ok, diagnostics, [tracks[i].id for i in inliers]

    def _record_observations(self, frame_id: int, ids: list[int]) -> None:
        for fid in ids:
            track = self._tracks.get(fid)
            if track is not None and track.landmark_id in self._landmarks:
                point = self._landmarks[track.landmark_id]
                point.observations[frame_id] = fid
                self._observation_pixels[point.id][frame_id] = track.pixel.copy()

    def _should_keyframe(self, frame: Frame, n_inliers: int) -> bool:
        last = self._last_keyframe
        if last is None:
            return True
        gap = frame.index - last.frame_id
        if gap < self.config.kf_min_frames_between:
            return False
        angle = np.degrees(np.arccos(np.clip((np.trace(last.R.T @ self._pose.R) - 1) / 2, -1, 1)))
        return bool(
            gap >= self.config.keyframe_max_gap
            or np.linalg.norm(last.t - self._pose.t) >= self.config.kf_min_translation
            or angle >= self.config.kf_min_rotation_deg
            or n_inliers < self.config.kf_min_matches
        )

    def _triangulate_new_tracks(self, frame: Frame) -> None:
        room = self.config.max_map_points - len(self._landmarks)
        if room <= 0:
            return
        groups: dict[int, list[_Track]] = {}
        for track in self._tracks.values():
            if track.landmark_id is None and track.anchor_frame != frame.index:
                groups.setdefault(track.anchor_frame, []).append(track)
        for tracks in groups.values():
            ids = np.int64([t.id for t in tracks])
            points = triangulate_two_view(
                np.float64([t.anchor_pixel for t in tracks]),
                np.float64([t.pixel for t in tracks]),
                tracks[0].anchor_pose,
                self._pose,
                self._intrinsics,
                config=self.config.triangulation_config(),
                feature_ids_a=ids,
                feature_ids_b=ids,
                frame_a=tracks[0].anchor_frame,
                frame_b=frame.index,
            )
            for point in points[:room]:
                self._insert_landmark(point, self._tracks[point.observations[frame.index]])
            room = self.config.max_map_points - len(self._landmarks)
            if room <= 0:
                break

    def _save_descriptors(self, gray: np.ndarray) -> None:
        keypoints = []
        for track in self._tracks.values():
            if track.landmark_id is not None:
                x, y = track.pixel * self._image_scale
                keypoints.append(cv2.KeyPoint(float(x), float(y), 31, class_id=track.id))
        if not keypoints:
            return
        # Keep a bounded set of image/landmark correspondences. KLT directly
        # from a verified keyframe can recover points dropped by incremental
        # flow, while PnP still independently validates their world pose.
        ids = np.int64([self._tracks[k.class_id].landmark_id for k in keypoints])
        pixels = np.float64([self._tracks[k.class_id].pixel for k in keypoints])
        self._relocalization_keyframes.append((gray.copy(), ids, pixels))
        limit = max(1, self.config.relocalization_keyframes)
        self._relocalization_keyframes = self._relocalization_keyframes[-limit:]
        returned, descriptors = self._orb.compute(gray, keypoints)
        if descriptors is None:
            return
        for keypoint, descriptor in zip(returned, descriptors, strict=True):
            track = self._tracks.get(keypoint.class_id)
            if track is not None:
                self._landmarks[track.landmark_id].descriptor = descriptor.copy()
                history = self._descriptor_history.setdefault(track.landmark_id, [])
                history.append(descriptor.copy())
                del history[: -max(1, self.config.landmark_descriptor_history)]

    def _accept_relocalization(
        self, frame: Frame, pose: CameraPose, pixels: np.ndarray, landmark_ids: np.ndarray
    ) -> None:
        self._pose = pose
        self._poses[frame.index] = pose
        self._tracks = {}
        for pixel, landmark_id in zip(pixels, landmark_ids, strict=True):
            fid = self._next_track_id
            self._next_track_id += 1
            pixel = pixel.copy()
            self._tracks[fid] = _Track(
                fid, pixel, pixel.copy(), pose, frame.index, int(landmark_id)
            )

    def _relocalize_keyframes(self, gray: np.ndarray, frame: Frame) -> tuple[bool, dict]:
        for reference, ids, pixels in reversed(self._relocalization_keyframes):
            if reference.shape != gray.shape or len(ids) < self.config.min_inliers:
                continue
            before = np.float32(pixels * self._image_scale).reshape(-1, 1, 2)
            params = dict(
                winSize=(31, 31),
                maxLevel=max(0, min(4, int(np.log2(min(gray.shape) / 50)))),
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
            )
            after, status, _ = cv2.calcOpticalFlowPyrLK(reference, gray, before, None, **params)
            if after is None:
                continue
            back, reverse_status, _ = cv2.calcOpticalFlowPyrLK(
                gray, reference, after, None, **params
            )
            if back is None:
                continue
            xy = after.reshape(-1, 2)
            fb = np.linalg.norm(back.reshape(-1, 2) - before.reshape(-1, 2), axis=1)
            good = (
                status.ravel().astype(bool)
                & reverse_status.ravel().astype(bool)
                & np.isfinite(xy).all(axis=1)
                & (fb <= self.config.klt_fb_threshold_px)
                & (xy[:, 0] >= 8)
                & (xy[:, 0] < gray.shape[1] - 8)
                & (xy[:, 1] >= 8)
                & (xy[:, 1] < gray.shape[0] - 8)
            )
            candidate_ids, current = ids[good], xy[good] / self._image_scale
            xyz = np.float64([self._landmarks[int(i)].xyz for i in candidate_ids]).reshape(-1, 3)
            ok, pose, inliers, quality = self._estimate_pose(
                xyz, current, frame.index, frame.timestamp, candidate_ids
            )
            if ok:
                self._accept_relocalization(frame, pose, current[inliers], candidate_ids[inliers])
                return True, {
                    **quality,
                    "relocalization_method": "verified_keyframe_flow_pnp",
                    "reused_world_landmarks": len(inliers),
                }
        return False, {"reason": "keyframe_relocalization_failed"}

    def _current_descriptors(self, gray: np.ndarray):
        keypoints, descriptors = self._orb.detectAndCompute(gray, None)
        # Map descriptors are computed at Shi-Tomasi/KLT points with ORB's
        # supplied-keypoint angle. DetectAndCompute uses oriented FAST points;
        # these are not interchangeable. Also sample the same corners/angle as
        # the stored map, retaining oriented ORB for old saved descriptors.
        corners = cv2.goodFeaturesToTrack(
            gray,
            maxCorners=self.config.n_features,
            qualityLevel=0.01,
            minDistance=self.config.feature_min_distance,
            blockSize=7,
        )
        if corners is not None and hasattr(self._orb, "compute"):
            extra, extra_descriptors = self._orb.compute(
                gray, [cv2.KeyPoint(float(x), float(y), 31) for x, y in corners.reshape(-1, 2)]
            )
            if extra_descriptors is not None:
                if descriptors is None:
                    return list(extra), extra_descriptors
                keypoints = [*keypoints, *extra]
                descriptors = np.vstack((descriptors, extra_descriptors))
        return keypoints, descriptors

    def _relocalize(self, gray: np.ndarray, frame: Frame) -> tuple[bool, dict]:
        landmarks = [p for p in self._landmarks.values() if p.descriptor is not None]
        if len(landmarks) < self.config.min_matches:
            return False, {"reason": "tracking_lost_no_relocalization_descriptors"}
        keypoints, descriptors = self._current_descriptors(gray)
        if descriptors is None or len(descriptors) < self.config.min_matches:
            return False, {"reason": "tracking_lost_no_image_descriptors"}
        samples = [
            (point, descriptor)
            for point in landmarks
            for descriptor in self._descriptor_history.get(point.id, [point.descriptor])
        ]
        pairs = self._bf.knnMatch(np.uint8([row[1] for row in samples]), descriptors, k=2)
        matches = [
            pair[0]
            for pair in pairs
            if len(pair) == 2 and pair[0].distance < self.config.match_ratio * pair[1].distance
        ]
        # One sensor feature cannot represent two world landmarks.
        matches.sort(key=lambda m: m.distance)
        seen, seen_landmarks = set(), set()
        unique = []
        for match in matches:
            point_id = samples[match.queryIdx][0].id
            # ORB and corner sampling may describe the same sensor point.
            pixel = keypoints[match.trainIdx].pt
            sensor_key = (round(pixel[0], 1), round(pixel[1], 1))
            if sensor_key not in seen and point_id not in seen_landmarks:
                seen.add(sensor_key)
                seen_landmarks.add(point_id)
                unique.append(match)
        xyz = np.float64([samples[m.queryIdx][0].xyz for m in unique]).reshape(-1, 3)
        pixels = np.float64([keypoints[m.trainIdx].pt for m in unique]).reshape(-1, 2)
        pixels /= self._image_scale
        ok, pose, inliers, diagnostics = self._estimate_pose(
            xyz,
            pixels,
            frame.index,
            frame.timestamp,
            np.int64([samples[m.queryIdx][0].id for m in unique]),
        )
        if not ok:
            return False, {**diagnostics, "reason": "same_map_relocalization_failed"}
        ids = np.int64([samples[m.queryIdx][0].id for m in unique])
        self._accept_relocalization(frame, pose, pixels[inliers], ids[inliers])
        return True, {
            **diagnostics,
            "relocalization_method": "same_sampling_descriptor_pnp",
            "reused_world_landmarks": len(inliers),
        }
