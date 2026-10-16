"""Persistent catalogue tests use actual files, decoder metadata, and recovery state."""

from __future__ import annotations

import json
import os
from pathlib import Path

import cv2
import numpy as np
import pytest

from visualmap.video_library import VideoLibrary


def video(path: Path, content: bytes = b"original-video") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def run(root: Path, name: str, content: bytes, extent: dict | None = None) -> Path:
    directory = root / "outputs" / name
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "semantic_map.html").write_text("<html>real viewer</html>")
    video(directory / "source_video.mp4", content)
    (directory / "run_report.json").write_text(
        json.dumps(
            {"video": {"path": "original.mp4"}, "frames_read": 48, "video_ingestion": extent or {}}
        )
    )
    return directory


def test_top_level_discovery_associates_same_content_results_and_preserves_sources(tmp_path):
    source = video(tmp_path / "sample_data/original.mp4")
    run(tmp_path, "spatial_consistent_20261013", source.read_bytes())
    latest = run(tmp_path, "occupancy_upgrade_20261013", source.read_bytes())
    # Result ordering follows actual viewer update times, not lexicographic run names.
    os.utime(latest / "semantic_map.html", (2_000_000_000, 2_000_000_000))
    video(tmp_path / "different.mov", b"another-video")
    video(tmp_path / "sample_data/nested/ignored.mp4", b"not-a-top-level-source")
    run(tmp_path, "unit_fixture_test", b"fixture")
    run(tmp_path, "nested_tests/hidden", b"fixture")
    library = VideoLibrary(tmp_path)
    rows = library.listing()["items"]
    assert len(rows) == 2
    item = next(row for row in rows if row["name"] == "original.mp4")
    assert item["source"] == "sample"
    assert [row["id"] for row in item["results"]] == [
        "occupancy_upgrade_20261013",
        "spatial_consistent_20261013",
    ]
    assert item["status"] == "complete"
    assert not any(str(tmp_path) in json.dumps(row) for row in rows)
    assert source.read_bytes() == b"original-video"
    restored = VideoLibrary(tmp_path).listing()["items"]
    assert {row["id"] for row in restored} == {row["id"] for row in rows}
    assert next(row for row in restored if row["id"] == item["id"])["results"] == item["results"]


def test_metadata_and_thumbnail_are_read_from_a_real_local_video(tmp_path):
    path = tmp_path / "source.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (80, 60))
    assert writer.isOpened()
    try:
        for number in range(20):
            frame = np.zeros((60, 80, 3), dtype=np.uint8)
            frame[:, : 20 + number, 2] = 200
            writer.write(frame)
    finally:
        writer.release()
    library = VideoLibrary(tmp_path)
    item = library.listing()["items"][0]
    assert item["width"] == 80 and item["height"] == 60
    assert item["duration_seconds"] == pytest.approx(2)
    assert item["size_bytes"] == path.stat().st_size
    cover = library.thumbnail_path(item["id"])
    assert cover.read_bytes().startswith(b"\xff\xd8")
    assert cv2.imdecode(np.frombuffer(cover.read_bytes(), np.uint8), cv2.IMREAD_COLOR) is not None


def test_interrupted_job_never_becomes_complete_from_a_stale_html(tmp_path):
    path = video(tmp_path / "source.mp4")
    library = VideoLibrary(tmp_path)
    item = next(iter(library.items.values()))
    identity = "video_interrupted"
    item["last_job_id"] = identity
    library.jobs[identity] = {
        "id": identity,
        "library_id": item["id"],
        "status": "processing",
        "frame": 17,
        "total_frames": 40,
    }
    library.save(force=True)
    run(tmp_path, identity, path.read_bytes(), {"full_video_processed": True})
    restored = VideoLibrary(tmp_path)
    row = restored.listing()["items"][0]
    assert row["status"] == "interrupted"
    assert row["frame"] == 17
    assert row["results"] == []
    assert restored.active_job_id() is None
    assert restored.jobs[identity]["source_complete"] is False


def test_completed_partial_run_remains_partial_after_restart(tmp_path):
    video(tmp_path / "source.mp4")
    directory = run(
        tmp_path,
        "video_finished",
        b"original-video",
        {"full_video_processed": False, "completeness": "requested_partial"},
    )
    library = VideoLibrary(tmp_path)
    item = next(iter(library.items.values()))
    item["last_job_id"] = directory.name
    library.jobs[directory.name] = {
        "id": directory.name,
        "library_id": item["id"],
        "status": "complete",
        "stage": "所选片段处理完成",
        "frame": 48,
        "total_frames": 64,
        "requested_partial": True,
        "source_complete": False,
    }
    library.save(force=True)
    restored = VideoLibrary(tmp_path).listing()["items"][0]
    assert restored["status"] == "complete"
    assert restored["results"][0]["requested_partial"] is True
    assert restored["results"][0]["source_complete"] is False
    assert restored["results"][0]["processed_frames"] == 48
    assert restored["stage"] == "所选片段处理完成"


def test_upload_content_deduplicates_without_deleting_either_copy(tmp_path):
    library = VideoLibrary(tmp_path)
    first = video(tmp_path / "outputs/uploads/first.mp4")
    second = video(tmp_path / "outputs/uploads/second.mp4")
    original = library.add_video(first, "原始名称.mp4")
    duplicate = library.add_video(second, "复制名称.mp4")
    assert duplicate["id"] == original["id"]
    assert len(library.items) == 1
    assert duplicate["name"] == "原始名称.mp4"
    assert len(duplicate["sources"]) == 2
    assert first.exists() and second.exists()


def test_overwritten_source_cannot_play_new_content_with_old_result(tmp_path):
    source = video(tmp_path / "source.mp4")
    run(tmp_path, "occupancy_upgrade", source.read_bytes())
    library = VideoLibrary(tmp_path)
    old_id = next(iter(library.items))
    source.write_bytes(b"completely-new-video-content")
    library.refresh(force=True)
    assert len(library.items) == 2
    assert library.video_path(old_id) == tmp_path / "outputs/occupancy_upgrade/source_video.mp4"
    rows = library.listing()["items"]
    old = next(row for row in rows if row["id"] == old_id)
    assert old["results"] and old["status"] == "complete"
    new = next(row for row in rows if row["id"] != old_id)
    assert new["results"] == [] and new["status"] == "ready"


def test_overwritten_source_without_archive_is_unavailable(tmp_path):
    source = video(tmp_path / "source.mp4")
    library = VideoLibrary(tmp_path)
    old = next(iter(library.items.values()))
    source.write_bytes(b"replacement-with-different-length")
    assert library.video_path(old["id"]) is None
    assert library.public_item(old)["status"] == "unavailable"
    assert library.public_item(old)["video_url"] is None


@pytest.mark.parametrize(
    "relative",
    [
        "../outside.mp4",
        "private/movie.mp4",
        "outputs/library/video.mp4",
        "sample_data/deeper/movie.mp4",
    ],
)
def test_index_paths_do_not_authorize_arbitrary_files(tmp_path, relative):
    library = VideoLibrary(tmp_path)
    path = video(tmp_path / relative)
    assert library.authorized_video(path) is False
    with pytest.raises(ValueError):
        library.add_video(path)


def test_symlink_outside_workspace_is_not_imported_or_served(tmp_path):
    outside = tmp_path.parent / (tmp_path.name + "-outside.mp4")
    outside.write_bytes(b"private")
    link = tmp_path / "escape.mp4"
    try:
        link.symlink_to(outside)
    except OSError:
        pytest.skip("Creating symlinks requires OS permission on this Windows host")
    library = VideoLibrary(tmp_path)
    assert not library.items
    assert library.authorized_video(link) is False


def test_broken_catalogue_is_preserved_and_sources_rediscovered(tmp_path):
    video(tmp_path / "recover.mp4")
    storage = tmp_path / "outputs/library"
    storage.mkdir(parents=True)
    (storage / "index.json").write_text("broken catalogue")
    library = VideoLibrary(tmp_path)
    assert len(library.items) == 1
    assert len(list(storage.glob("index-recovery-*.json"))) == 1
    assert json.loads((storage / "index.json").read_text(encoding="utf-8"))["schema_version"] == 1


def test_atomic_write_failure_keeps_previous_index_and_media(tmp_path, monkeypatch):
    path = video(tmp_path / "source.mp4")
    library = VideoLibrary(tmp_path)
    before = library.index_path.read_bytes()
    monkeypatch.setattr(
        "visualmap.video_library.os.replace",
        lambda *_: (_ for _ in ()).throw(OSError("disk error")),
    )
    with pytest.raises(OSError):
        library.save(force=True)
    assert library.index_path.read_bytes() == before
    assert not list(library.storage.glob("*.tmp"))
    assert path.read_bytes() == b"original-video"


def test_failed_decode_report_never_appears_as_successful_history(tmp_path):
    video(tmp_path / "source.mp4")
    directory = run(tmp_path, "video_decode_failure", b"original-video")
    (directory / "run_report.json").write_text(json.dumps({"run_status": "failed_video_decode"}))
    row = VideoLibrary(tmp_path).listing()["items"][0]
    assert row["results"] == [] and row["status"] == "ready"


@pytest.mark.parametrize(
    "report",
    [
        [],
        {"video": [], "summary": "broken", "video_ingestion": ["broken"]},
        {"video": None, "summary": None, "video_ingestion": False},
    ],
)
def test_malformed_optional_report_fields_do_not_break_the_catalogue(tmp_path, report):
    video(tmp_path / "source.mp4")
    directory = run(tmp_path, "video_old_report", b"original-video")
    (directory / "run_report.json").write_text(json.dumps(report))
    row = VideoLibrary(tmp_path).listing()["items"][0]
    assert row["status"] == "complete"
    assert row["results"][0]["source_complete"] is False
    assert row["results"][0]["requested_partial"] is False


def test_recycle_is_persistent_and_preserves_all_video_sources_and_results(tmp_path):
    source = video(tmp_path / "sample_data/original.mp4")
    uploaded = video(tmp_path / "outputs/uploads/import.mp4", source.read_bytes())
    directory = run(tmp_path, "video_completed", source.read_bytes())
    library = VideoLibrary(tmp_path)
    original = library.listing()["items"][0]
    files = {path: path.read_bytes() for path in (source, uploaded, *directory.iterdir())}

    removed = library.set_trashed(original["id"], True)
    assert removed["status"] == "trashed" and removed["video_url"] is None
    assert library.video_path(original["id"]) is None
    library.refresh(force=True)
    assert library.listing()["items"] == []
    assert library.listing()["trashed_items"][0]["id"] == original["id"]

    reopened = VideoLibrary(tmp_path)
    assert reopened.listing()["items"] == []
    assert reopened.listing()["trashed_items"][0]["results"] == original["results"]
    restored = reopened.set_trashed(original["id"], False)
    assert restored["status"] == "complete" and restored["video_url"]
    assert reopened.listing()["trashed_items"] == []
    assert restored["results"] == original["results"]
    assert all(path.read_bytes() == contents for path, contents in files.items())


def test_running_video_cannot_be_recycled_even_if_last_job_is_not_running(tmp_path):
    video(tmp_path / "source.mp4")
    library = VideoLibrary(tmp_path)
    item = library.listing()["items"][0]
    library.jobs["live"] = {"library_id": item["id"], "status": "processing"}
    library.jobs["old"] = {"library_id": item["id"], "status": "complete"}
    library.items[item["id"]]["last_job_id"] = "old"
    with pytest.raises(ValueError, match="正在识别"):
        library.set_trashed(item["id"], True)
    assert not library.items[item["id"]].get("deleted_at")
    assert len(library.listing()["items"]) == 1


def test_explicit_reimport_restores_but_discovery_does_not(tmp_path):
    source = video(tmp_path / "outputs/uploads/original.mp4")
    library = VideoLibrary(tmp_path)
    item = library.listing()["items"][0]
    library.set_trashed(item["id"], True)
    library.add_video(source)
    assert library.listing()["items"] == []
    library.add_video(source, restore_deleted=True)
    assert library.listing()["items"][0]["id"] == item["id"]
    assert library.listing()["trashed_items"] == []


def test_recycle_save_failure_keeps_in_memory_and_persisted_state(tmp_path, monkeypatch):
    video(tmp_path / "source.mp4")
    library = VideoLibrary(tmp_path)
    item = library.listing()["items"][0]
    before = library.index_path.read_bytes()
    monkeypatch.setattr(
        "visualmap.video_library.os.replace",
        lambda *_: (_ for _ in ()).throw(OSError("disk error")),
    )
    with pytest.raises(OSError):
        library.set_trashed(item["id"], True)
    assert not library.items[item["id"]].get("deleted_at")
    assert library.index_path.read_bytes() == before
