"""Core data structures shared across VisualMap.

Conventions (see docs/data_format.md and docs/adr/0003-coordinate-system.md):

- World frame: right-handed, gravity-aligned when known; otherwise arbitrary but fixed.
- Camera frame: OpenCV convention — x right, y down, z forward.
- Pose matrix: ``CameraPose`` stores T_world←camera (4x4 SE(3)).
  A point in camera coords maps to world as: ``X_w = R @ X_c + t``.
- Units: translation is in "map units". Monocular scale is arbitrary until
  metric recovery (V0.4+). Do not label values as meters in this stage.
- Timestamps: float seconds from video start (frame index / fps is fine).
- Image coords: pixel (u, v) with origin at top-left, u right, v down.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from numbers import Integral, Real

import numpy as np

__all__ = [
    "CameraIntrinsics",
    "CameraPose",
    "Frame",
    "KeyFrame",
    "MapPoint",
    "Trajectory",
    "MapResult",
    "RotationRepresentation",
]


class RotationRepresentation(str, Enum):
    MATRIX = "matrix"
    QUATERNION = "quaternion"  # w, x, y, z
    SE3 = "se3"


@dataclass(frozen=True)
class CameraIntrinsics:
    """Pinhole intrinsics + optional Brown-Conrady distortion."""

    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    # k1, k2, p1, p2, k3 — zeros mean no distortion
    dist: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        for name in ("fx", "fy", "cx", "cy"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value):
                raise ValueError(f"{name} must be a finite number")
        if self.fx <= 0 or self.fy <= 0:
            raise ValueError("fx and fy must be positive")
        for name in ("width", "height"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if len(self.dist) not in (0, 4, 5):
            raise ValueError("dist must have length 0, 4, or 5")
        if any(
            isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value)
            for value in self.dist
        ):
            raise ValueError("dist coefficients must be finite numbers")

    @property
    def K(self) -> np.ndarray:
        """3x3 camera matrix."""
        return np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

    @property
    def dist_coeffs(self) -> np.ndarray:
        if not self.dist:
            return np.zeros(5, dtype=np.float64)
        return np.asarray(self.dist, dtype=np.float64)

    @property
    def has_distortion(self) -> bool:
        return bool(np.any(self.dist_coeffs != 0.0))

    def to_dict(self) -> dict:
        return {
            "fx": self.fx,
            "fy": self.fy,
            "cx": self.cx,
            "cy": self.cy,
            "width": self.width,
            "height": self.height,
            "dist": list(self.dist),
        }

    @classmethod
    def from_dict(cls, data: dict) -> CameraIntrinsics:
        return cls(
            fx=float(data["fx"]),
            fy=float(data["fy"]),
            cx=float(data["cx"]),
            cy=float(data["cy"]),
            width=int(data["width"]),
            height=int(data["height"]),
            dist=tuple(float(x) for x in data.get("dist", (0.0, 0.0, 0.0, 0.0, 0.0))),
        )


@dataclass(frozen=True)
class CameraPose:
    """T_world←camera as a 4x4 SE(3) matrix.

    Transform a camera-frame point to world:
        X_world = pose.transform_point(X_camera)
    """

    matrix: np.ndarray
    frame_id: int = -1
    timestamp: float = 0.0

    def __post_init__(self) -> None:
        mat = np.asarray(self.matrix, dtype=np.float64)
        if mat.shape != (4, 4):
            raise ValueError(f"pose matrix must be 4x4, got {mat.shape}")
        if not np.isfinite(mat).all():
            raise ValueError("pose matrix must contain finite values")
        if not np.allclose(mat[3], [0.0, 0.0, 0.0, 1.0], atol=1e-8):
            raise ValueError("pose matrix last row must be [0, 0, 0, 1]")
        # freeze a private copy so callers cannot mutate after construction
        object.__setattr__(self, "matrix", mat.copy())

    @classmethod
    def identity(cls, frame_id: int = -1, timestamp: float = 0.0) -> CameraPose:
        return cls(np.eye(4, dtype=np.float64), frame_id=frame_id, timestamp=timestamp)

    @classmethod
    def from_rt(
        cls,
        R: np.ndarray,
        t: np.ndarray,
        frame_id: int = -1,
        timestamp: float = 0.0,
    ) -> CameraPose:
        R = np.asarray(R, dtype=np.float64)
        t = np.asarray(t, dtype=np.float64).reshape(3)
        if R.shape != (3, 3):
            raise ValueError("R must be 3x3")
        mat = np.eye(4, dtype=np.float64)
        mat[:3, :3] = R
        mat[:3, 3] = t
        return cls(mat, frame_id=frame_id, timestamp=timestamp)

    @property
    def R(self) -> np.ndarray:
        """3x3 rotation (world←camera)."""
        return self.matrix[:3, :3].copy()

    @property
    def t(self) -> np.ndarray:
        """Translation (camera origin in world coordinates)."""
        return self.matrix[:3, 3].copy()

    @property
    def camera_center(self) -> np.ndarray:
        """Camera center in world coords (same as ``t`` for T_world←camera)."""
        return self.t

    def inverse(self) -> CameraPose:
        """Return T_camera←world."""
        R = self.R
        t = self.t
        mat = np.eye(4, dtype=np.float64)
        mat[:3, :3] = R.T
        mat[:3, 3] = -R.T @ t
        return CameraPose(mat, frame_id=self.frame_id, timestamp=self.timestamp)

    def transform_point(self, point: np.ndarray) -> np.ndarray:
        """Map 3D point(s) from camera coordinates to world coordinates.

        Accepts shape (3,) or (N, 3); returns the same layout.
        """
        p = np.asarray(point, dtype=np.float64)
        if p.shape == (3,):
            return self.R @ p + self.t
        if p.ndim == 2 and p.shape[1] == 3:
            return (self.R @ p.T).T + self.t
        raise ValueError(f"point must be (3,) or (N, 3), got {p.shape}")

    def compose(self, other: CameraPose) -> CameraPose:
        """self ∘ other = T_world←A @ T_A←B, i.e. apply other first, then self."""
        return CameraPose(self.matrix @ other.matrix)

    def to_dict(self) -> dict:
        return {
            "frame_id": self.frame_id,
            "timestamp": self.timestamp,
            "matrix": self.matrix.tolist(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> CameraPose:
        return cls(
            np.asarray(data["matrix"], dtype=np.float64),
            frame_id=int(data.get("frame_id", -1)),
            timestamp=float(data.get("timestamp", 0.0)),
        )


@dataclass(frozen=True)
class Frame:
    """One video frame with metadata. Image is BGR uint8 HxWx3 (OpenCV default)."""

    index: int
    timestamp: float
    image: np.ndarray

    def __post_init__(self) -> None:
        img = self.image
        if isinstance(self.index, bool) or not isinstance(self.index, Integral) or self.index < 0:
            raise ValueError("frame index must be an integer >= 0")
        if (
            isinstance(self.timestamp, bool)
            or not isinstance(self.timestamp, Real)
            or not np.isfinite(self.timestamp)
            or self.timestamp < 0
        ):
            raise ValueError("frame timestamp must be finite and >= 0")
        if not isinstance(img, np.ndarray):
            raise TypeError("image must be a numpy array")
        if img.ndim not in (2, 3):
            raise ValueError(f"image must be HxW or HxWxC, got shape {img.shape}")
        if img.shape[0] == 0 or img.shape[1] == 0:
            raise ValueError("image dimensions must be positive")
        if img.ndim == 3 and img.shape[2] not in (1, 3, 4):
            raise ValueError("image must have 1, 3, or 4 channels")
        if not np.issubdtype(img.dtype, np.number) or np.issubdtype(img.dtype, np.complexfloating):
            raise ValueError("image must contain real numeric values")
        if np.issubdtype(img.dtype, np.floating) and not np.isfinite(img).all():
            raise ValueError("image must contain finite values")
        object.__setattr__(self, "image", img)

    @property
    def height(self) -> int:
        return int(self.image.shape[0])

    @property
    def width(self) -> int:
        return int(self.image.shape[1])

    @property
    def shape(self) -> tuple[int, ...]:
        return self.image.shape


@dataclass
class MapPoint:
    """A sparse 3D landmark in world coordinates (map units, scale-ambiguous)."""

    id: int
    xyz: np.ndarray
    descriptor: np.ndarray | None = None
    observations: dict[int, int] = field(default_factory=dict)  # frame_id -> feature_idx
    color: tuple[int, int, int] | None = None

    def __post_init__(self) -> None:
        self.xyz = np.asarray(self.xyz, dtype=np.float64).reshape(3)


@dataclass
class KeyFrame:
    """A selected frame used for mapping."""

    frame: Frame
    pose: CameraPose
    keypoints: list | None = None
    descriptors: np.ndarray | None = None


@dataclass
class Trajectory:
    """Ordered camera poses. Must stay sorted by frame_id / timestamp."""

    poses: list[CameraPose] = field(default_factory=list)

    def append(self, pose: CameraPose) -> None:
        self.poses.append(pose)

    def __len__(self) -> int:
        return len(self.poses)

    def __iter__(self):
        return iter(self.poses)

    def positions(self) -> np.ndarray:
        """Nx3 array of camera centers in world coordinates."""
        if not self.poses:
            return np.zeros((0, 3), dtype=np.float64)
        return np.stack([p.camera_center for p in self.poses], axis=0)

    def to_list(self) -> list[dict]:
        return [p.to_dict() for p in self.poses]


@dataclass
class MapResult:
    """Full stage output consumed by export / visualization / later stages."""

    trajectory: Trajectory
    map_points: list[MapPoint] = field(default_factory=list)
    intrinsics: CameraIntrinsics | None = None
    metadata: dict = field(default_factory=dict)

    def summary(self) -> dict:
        return {
            "num_poses": len(self.trajectory),
            "num_map_points": len(self.map_points),
            "has_intrinsics": self.intrinsics is not None,
            **self.metadata,
        }
