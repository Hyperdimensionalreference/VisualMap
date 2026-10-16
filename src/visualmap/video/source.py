"""Video ingestion: open files, expose metadata, iterate frames.

OpenCV is used only inside this package and ``camera`` — higher layers depend
on our Frame type, not on cv2 objects.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from visualmap.slam.types import Frame
from visualmap.utils.logging import get_logger

logger = get_logger(__name__)

__all__ = ["VideoInfo", "VideoSource", "VideoDecodeError"]


class VideoDecodeError(RuntimeError):
    """The decoder stopped before its declared end, or returned an invalid frame."""

    def __init__(self, message: str, diagnostics: dict) -> None:
        super().__init__(message)
        self.diagnostics = dict(diagnostics)


@dataclass(frozen=True)
class VideoInfo:
    path: Path
    width: int
    height: int
    fps: float
    frame_count: int
    duration_sec: float

    def to_dict(self) -> dict:
        return {
            "path": str(self.path),
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "frame_count": self.frame_count,
            "duration_sec": self.duration_sec,
        }


class VideoSource:
    """Read frames from a video file (mp4/avi/...)."""

    def __init__(
        self,
        path: str | Path,
        start_frame: int = 0,
        max_frames: int | None = None,
        every_n_frames: int = 1,
    ) -> None:
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(f"video not found: {self.path}")
        if start_frame < 0:
            raise ValueError("start_frame must be >= 0")
        if every_n_frames < 1:
            raise ValueError("every_n_frames must be >= 1")
        if max_frames is not None and max_frames < 1:
            raise ValueError("max_frames must be >= 1 when set")

        self._start_frame = start_frame
        self._max_frames = max_frames
        self._every_n = every_n_frames

        self._cap = cv2.VideoCapture(str(self.path))
        if not self._cap.isOpened():
            raise RuntimeError(f"failed to open video: {self.path}")

        self._info = self._read_info()
        self._read_diagnostics = self._new_read_diagnostics()
        logger.debug("Opened video %s", self._info)

    def _new_read_diagnostics(self) -> dict:
        return {
            "status": "not_started",
            "stop_reason": None,
            "completeness": "unverified",
            "reported_frame_count": self._info.frame_count,
            "frame_count_basis": "decoder_reported_not_independently_verified",
            "reported_length_consistent": None,
            "start_frame": self._start_frame,
            "every_n_frames": self._every_n,
            "max_frames": self._max_frames,
            "decoded_frames": 0,
            "yielded_frames": 0,
            "first_decoded_frame_index": None,
            "last_decoded_frame_index": None,
            "first_yielded_frame_index": None,
            "last_yielded_frame_index": None,
            "next_frame_index": None,
            "decoder_position_at_stop": None,
            "full_video_decoded": False,
            "full_video_processed": False,
            "timestamp_basis": "absolute_frame_index_divided_by_reported_fps",
        }

    @property
    def read_diagnostics(self) -> dict:
        """Decode extent and stop cause; frame counts alone are not accuracy proof."""
        return dict(self._read_diagnostics)

    def _decoder_position(self) -> int | None:
        if self._cap is None:
            return None
        value = self._cap.get(cv2.CAP_PROP_POS_FRAMES)
        return max(0, int(round(value))) if np.isfinite(value) else None

    def _decode_failure(self, reason: str, current: int, message: str) -> None:
        self._read_diagnostics.update(
            status="failed",
            stop_reason=reason,
            completeness="incomplete",
            next_frame_index=current,
            decoder_position_at_stop=self._decoder_position(),
            error=message,
        )
        raise VideoDecodeError(message, self.read_diagnostics)

    def _read_info(self) -> VideoInfo:
        width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(self._cap.get(cv2.CAP_PROP_FPS))
        if fps <= 0 or not np.isfinite(fps):
            logger.warning("Video reported invalid fps=%s; defaulting to 30", fps)
            fps = 30.0
        frame_count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count < 0:
            frame_count = 0
        duration = frame_count / fps if fps > 0 else 0.0
        return VideoInfo(
            path=self.path.resolve(),
            width=width,
            height=height,
            fps=fps,
            frame_count=frame_count,
            duration_sec=duration,
        )

    @property
    def info(self) -> VideoInfo:
        return self._info

    @property
    def width(self) -> int:
        return self._info.width

    @property
    def height(self) -> int:
        return self._info.height

    @property
    def fps(self) -> float:
        return self._info.fps

    def frames(self) -> Iterator[Frame]:
        """Yield frames from ``start_frame`` with stride ``every_n_frames``.

        Frame.index is the absolute index in the source video (not 0-based
        within this iteration). Timestamp is index / fps seconds.
        """
        self._read_diagnostics = self._new_read_diagnostics()
        self._read_diagnostics["status"] = "reading"
        if not self._cap.set(cv2.CAP_PROP_POS_FRAMES, self._start_frame):
            # Some backends ignore seek; fall through and skip manually
            logger.debug("CAP_PROP_POS_FRAMES seek not honored; will skip frames")

        # The decoder may seek to a keyframe before the requested position.
        # Start counting from its reported position, not from zero after seeking.
        position = self._decoder_position()
        # Seeking past the requested frame or an unknown position cannot label
        # frames correctly. Rewind and decode forward rather than invent indices.
        if position is None or position > self._start_frame:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            position = self._decoder_position()
            if position != 0:
                self._decode_failure(
                    "seek_position_unverified", 0, "Cannot establish absolute source frame indices"
                )
        current = position
        self._read_diagnostics["initial_decoder_position"] = current
        self._read_diagnostics["next_frame_index"] = current

        try:
            while True:
                if (
                    self._max_frames is not None
                    and self._read_diagnostics["yielded_frames"] >= self._max_frames
                ):
                    self._read_diagnostics.update(
                        status="requested_partial",
                        stop_reason="max_frames_limit",
                        completeness="requested_partial",
                        next_frame_index=current,
                        decoder_position_at_stop=self._decoder_position(),
                    )
                    break

                try:
                    ok, image = self._cap.read()
                except cv2.error as exc:
                    self._decode_failure("decoder_exception", current, str(exc))
                if not ok or image is None:
                    expected = self._info.frame_count
                    if expected > 0 and current < expected:
                        self._read_diagnostics["reported_length_consistent"] = False
                        self._decode_failure(
                            "premature_decode_stop",
                            current,
                            f"Video decoding stopped at frame {current} "
                            f"before reported end {expected}; "
                            "the file may be truncated or the container length may be incorrect",
                        )
                    matches_count = expected > 0 and current == expected
                    self._read_diagnostics.update(
                        status="complete" if matches_count else "unverified",
                        stop_reason="normal_eof" if matches_count else "eof_length_unverified",
                        completeness="matches_reported_length" if matches_count else "unverified",
                        reported_length_consistent=current == expected if expected > 0 else None,
                        next_frame_index=current,
                        decoder_position_at_stop=self._decoder_position(),
                        full_video_decoded=matches_count
                        and self._read_diagnostics["initial_decoder_position"] == 0,
                        full_video_processed=matches_count
                        and self._start_frame == 0
                        and self._every_n == 1,
                    )
                    break

                self._read_diagnostics["decoded_frames"] += 1
                if self._read_diagnostics["first_decoded_frame_index"] is None:
                    self._read_diagnostics["first_decoded_frame_index"] = current
                self._read_diagnostics["last_decoded_frame_index"] = current
                index = current
                current += 1
                self._read_diagnostics["next_frame_index"] = current
                timestamp = index / self._info.fps
                try:
                    frame = Frame(index, timestamp, np.ascontiguousarray(image))
                except (TypeError, ValueError) as exc:
                    self._decode_failure("invalid_decoded_frame", index, str(exc))
                if index < self._start_frame or (index - self._start_frame) % self._every_n != 0:
                    continue
                self._read_diagnostics["yielded_frames"] += 1
                if self._read_diagnostics["first_yielded_frame_index"] is None:
                    self._read_diagnostics["first_yielded_frame_index"] = index
                self._read_diagnostics["last_yielded_frame_index"] = index
                yield frame
        finally:
            if self._read_diagnostics["status"] == "reading":
                self._read_diagnostics.update(
                    status="consumer_stopped",
                    stop_reason="consumer_stopped",
                    completeness="requested_partial",
                    next_frame_index=current,
                    decoder_position_at_stop=self._decoder_position(),
                )
            logger.debug(
                "Video iteration finished, yielded=%d stop=%s",
                self._read_diagnostics["yielded_frames"],
                self._read_diagnostics["stop_reason"],
            )

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self) -> VideoSource:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def __repr__(self) -> str:
        info = self._info
        return (
            f"VideoSource(path={info.path.name}, "
            f"{info.width}x{info.height}@{info.fps:.3f})"
        )
