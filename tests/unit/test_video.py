"""VideoSource tests. Uses a tiny synthetic AVI generated on the fly."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from visualmap.video import VideoSource
from visualmap.video.source import VideoDecodeError


@pytest.fixture
def tiny_video(tmp_path: Path) -> Path:
    path = tmp_path / "tiny.avi"
    w, h, n, fps = 64, 48, 8, 10
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (w, h))
    if not writer.isOpened():
        pytest.skip("OpenCV VideoWriter not available")
    for i in range(n):
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        frame[:, : i * 8] = 255
        writer.write(frame)
    writer.release()
    if not path.is_file() or path.stat().st_size == 0:
        pytest.skip("video file was not written")
    return path


def test_open_and_info(tiny_video: Path):
    with VideoSource(tiny_video) as src:
        info = src.info
        assert info.width == 64
        assert info.height == 48
        assert info.fps == pytest.approx(10.0)
        assert info.frame_count >= 1


def test_iterate_all_frames(tiny_video: Path):
    with VideoSource(tiny_video) as src:
        frames = list(src.frames())
        assert len(frames) >= 3
        assert frames[0].index == 0
        assert frames[0].image.shape == (48, 64, 3)
        assert frames[0].timestamp == pytest.approx(0.0)
        # timestamps should be non-decreasing
        ts = [f.timestamp for f in frames]
        assert ts == sorted(ts)
        diagnostics = src.read_diagnostics
        assert diagnostics["stop_reason"] == "normal_eof"
        assert diagnostics["completeness"] == "matches_reported_length"
        assert diagnostics["full_video_processed"] is True
        assert diagnostics["decoded_frames"] == len(frames)


def test_max_frames_and_stride(tiny_video: Path):
    with VideoSource(tiny_video, max_frames=2, every_n_frames=2) as src:
        frames = list(src.frames())
        assert len(frames) == 2
        assert frames[0].index == 0
        assert frames[1].index == 2
        diagnostics = src.read_diagnostics
        assert diagnostics["stop_reason"] == "max_frames_limit"
        assert diagnostics["completeness"] == "requested_partial"
        assert diagnostics["decoded_frames"] == 3
        assert diagnostics["yielded_frames"] == 2
        assert diagnostics["full_video_processed"] is False


def test_start_frame_keeps_source_index_and_timestamp(tiny_video: Path):
    with VideoSource(tiny_video, start_frame=3, max_frames=1) as src:
        frame = next(src.frames())
    assert frame.index == 3
    assert frame.timestamp == pytest.approx(0.3)
    assert frame.image[:, :24].mean() > 240
    assert frame.image[:, 32:].mean() < 15


def test_missing_file_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        VideoSource(tmp_path / "nope.mp4")


class _DeclaredLengthCapture:
    """Deterministic decoder with independently specified header and stop."""

    def __init__(self, images, declared_count, seek_honored=True):
        self.images = images
        self.declared_count = declared_count
        self.seek_honored = seek_honored
        self.position = 0

    def isOpened(self):
        return True

    def get(self, prop):
        return {
            cv2.CAP_PROP_FRAME_WIDTH: 64,
            cv2.CAP_PROP_FRAME_HEIGHT: 48,
            cv2.CAP_PROP_FPS: 10,
            cv2.CAP_PROP_FRAME_COUNT: self.declared_count,
            cv2.CAP_PROP_POS_FRAMES: self.position,
        }.get(prop, 0)

    def set(self, prop, value):
        if self.seek_honored:
            self.position = min(int(value), len(self.images))
        return self.seek_honored

    def read(self):
        if self.position >= len(self.images):
            return False, None
        image = self.images[self.position]
        self.position += 1
        return True, image

    def release(self):
        pass


def _mock_source(monkeypatch, tmp_path, *, count, actual=4, seek_honored=True, **options):
    path = tmp_path / "mock.avi"
    path.touch()
    images = [np.full((48, 64, 3), i, np.uint8) for i in range(actual)]
    capture = _DeclaredLengthCapture(images, count, seek_honored)
    monkeypatch.setattr("visualmap.video.source.cv2.VideoCapture", lambda path: capture)
    return VideoSource(path, **options)


def test_premature_decoder_stop_raises_with_actual_source_extent(monkeypatch, tmp_path):
    with _mock_source(monkeypatch, tmp_path, count=10, actual=4) as source:
        with pytest.raises(VideoDecodeError, match="before reported end 10"):
            list(source.frames())
        diagnostics = source.read_diagnostics
        assert diagnostics["stop_reason"] == "premature_decode_stop"
        assert diagnostics["completeness"] == "incomplete"
        assert diagnostics["last_decoded_frame_index"] == 3
        assert diagnostics["next_frame_index"] == 4
        assert diagnostics["decoded_frames"] == diagnostics["yielded_frames"] == 4


def test_sampling_and_start_do_not_masquerade_as_incomplete_decode(monkeypatch, tmp_path):
    with _mock_source(
        monkeypatch, tmp_path, count=8, actual=8, start_frame=2, every_n_frames=3
    ) as source:
        frames = list(source.frames())
        assert [frame.index for frame in frames] == [2, 5]
        diagnostics = source.read_diagnostics
        assert diagnostics["next_frame_index"] == 8
        assert diagnostics["decoded_frames"] == 6
        assert diagnostics["yielded_frames"] == 2
        assert diagnostics["completeness"] == "matches_reported_length"
        assert diagnostics["full_video_processed"] is False


def test_sampling_and_start_still_detect_premature_stop(monkeypatch, tmp_path):
    with _mock_source(monkeypatch, tmp_path, count=10, start_frame=2, every_n_frames=3) as source:
        with pytest.raises(VideoDecodeError):
            list(source.frames())
        assert source.read_diagnostics["yielded_frames"] == 1
        assert source.read_diagnostics["last_decoded_frame_index"] == 3
        assert source.read_diagnostics["stop_reason"] == "premature_decode_stop"


def test_explicit_limit_does_not_assert_unread_video_is_complete(monkeypatch, tmp_path):
    with _mock_source(monkeypatch, tmp_path, count=10, start_frame=2, max_frames=1) as source:
        assert [frame.index for frame in source.frames()] == [2]
        assert source.read_diagnostics["status"] == "requested_partial"
        assert source.read_diagnostics["next_frame_index"] == 3
        assert source.read_diagnostics["full_video_decoded"] is False


@pytest.mark.parametrize("declared_count", [0, 2])
def test_unknown_or_underreported_length_never_claims_complete(
    monkeypatch, tmp_path, declared_count
):
    with _mock_source(monkeypatch, tmp_path, count=declared_count) as source:
        assert len(list(source.frames())) == 4
        assert source.read_diagnostics["status"] == "unverified"
        assert source.read_diagnostics["completeness"] == "unverified"
        assert source.read_diagnostics["full_video_decoded"] is False


def test_failed_seek_skips_decoded_frames_without_relabeling(monkeypatch, tmp_path):
    with _mock_source(
        monkeypatch, tmp_path, count=8, actual=8, start_frame=3, seek_honored=False
    ) as source:
        frames = list(source.frames())
        assert frames[0].index == 3
        assert np.all(frames[0].image == 3)
        assert source.read_diagnostics["first_decoded_frame_index"] == 0
        assert source.read_diagnostics["full_video_decoded"] is True


def test_invalid_decoded_image_is_rejected_even_when_stride_skips_it(monkeypatch, tmp_path):
    source = _mock_source(monkeypatch, tmp_path, count=4, every_n_frames=2)
    source._cap.images[1] = np.zeros((0, 64, 3), np.uint8)
    with source, pytest.raises(VideoDecodeError):
        list(source.frames())
    assert source.read_diagnostics["stop_reason"] == "invalid_decoded_frame"
