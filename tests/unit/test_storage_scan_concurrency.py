"""Keep read-only storage scans responsive without weakening replacement checks."""

from __future__ import annotations

import os
import threading
import time
from collections import Counter
from pathlib import Path

import pytest

import visualmap.storage_manager as storage
from visualmap.storage_manager import StorageManager
from visualmap.video_library import VideoLibrary


def put(path: Path, content: bytes = b"identical-managed-video") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


@pytest.fixture
def managed_library(tmp_path, monkeypatch):
    # These tests concern catalogue/storage behavior, not decoding synthetic media.
    monkeypatch.setattr(VideoLibrary, "_probe", lambda self, path, item: None)
    first = put(tmp_path / "outputs/uploads/one.mp4")
    second = put(tmp_path / "outputs/uploads/two.mp4")
    return VideoLibrary(tmp_path), first, second


def background_call(callback):
    results, errors = [], []
    finished = threading.Event()

    def run():
        try:
            results.append(callback())
        except BaseException as error:
            errors.append(error)
        finally:
            finished.set()

    worker = threading.Thread(target=run, daemon=True)
    worker.start()
    return worker, finished, results, errors


@pytest.mark.parametrize("operation", ["stats", "plan"])
def test_library_listing_remains_available_during_blocked_storage_scan(
    managed_library, monkeypatch, operation
):
    library, _, _ = managed_library
    manager = StorageManager(library)
    entered, release = threading.Event(), threading.Event()

    if operation == "stats":
        original = storage._scan_stats

        def paused_scan(directory, root):
            if directory == library.root / "outputs":
                entered.set()
                if not release.wait(5):
                    raise TimeoutError("test did not release directory scan")
            yield from original(directory, root)

        monkeypatch.setattr(storage, "_scan_stats", paused_scan)
    else:
        original = storage._digest

        def paused_digest(path):
            if not entered.is_set():
                entered.set()
                if not release.wait(5):
                    raise TimeoutError("test did not release media hashing")
            return original(path)

        monkeypatch.setattr(storage, "_digest", paused_digest)

    scan_thread, _, scan_results, scan_errors = background_call(getattr(manager, operation))
    listing_thread = None
    try:
        assert entered.wait(2), "storage scan never reached the slow filesystem operation"
        # Exercise the real public listing, including its normal catalogue lock.
        library._last_scan = time.monotonic()
        listing_thread, finished, listings, listing_errors = background_call(library.listing)
        assert finished.wait(1), "storage scanning held the video catalogue lock"
        assert not listing_errors
        assert len(listings[0]["items"]) == 1
        assert not release.is_set(), "listing only returned after the storage scan resumed"
    finally:
        release.set()
        scan_thread.join(3)
        if listing_thread is not None:
            listing_thread.join(3)
    assert not scan_thread.is_alive()
    assert listing_thread is not None and not listing_thread.is_alive()
    assert not scan_errors
    assert len(scan_results) == 1


def test_plan_hashes_each_shared_allocation_only_once(managed_library, monkeypatch):
    library, first, _ = managed_library
    linked = first.with_name("one-linked.mp4")
    os.link(first, linked)
    original = storage._digest
    hashed = []

    def record_hash(path):
        value = path.stat()
        hashed.append((value.st_dev, value.st_ino))
        return original(path)

    monkeypatch.setattr(storage, "_digest", record_hash)
    plan = StorageManager(library).plan()
    assert plan["operation_count"] == 1
    assert len(hashed) == 2
    assert set(Counter(hashed).values()) == {1}


def test_fully_shared_media_group_needs_no_hashing(managed_library, monkeypatch):
    library, first, second = managed_library
    second.unlink()
    os.link(first, second)

    def unexpected_hash(path):
        pytest.fail(f"already-shared media was read again: {path.name}")

    monkeypatch.setattr(storage, "_digest", unexpected_hash)
    assert StorageManager(library).plan()["operations"] == []


def test_media_changed_during_hashing_cannot_issue_a_replacement_plan(
    managed_library, monkeypatch
):
    library, first, second = managed_library
    original_content = first.read_bytes()
    original = storage._digest
    changed = []

    def modify_after_read(path):
        digest = original(path)
        if not changed:
            changed.append(path)
            previous = path.stat()
            # Preserve the length so grouping by size cannot hide this race.
            path.write_bytes(b"x" * len(original_content))
            os.utime(path, ns=(previous.st_atime_ns, previous.st_mtime_ns + 1_000_000))
        return digest

    monkeypatch.setattr(storage, "_digest", modify_after_read)
    manager = StorageManager(library)
    with pytest.raises(ValueError, match="变化"):
        manager.plan()
    assert len(changed) == 1
    untouched = second if changed[0] == first else first
    assert untouched.read_bytes() == original_content
    assert changed[0].read_bytes() == b"x" * len(original_content)
    assert not os.path.samefile(first, second)
    assert not manager._plans
    assert not list(first.parent.glob(".storage-link-*"))


def test_change_to_previous_media_while_hashing_next_media_invalidates_preview(
    managed_library, monkeypatch
):
    library, first, second = managed_library
    original = storage._digest
    hashed = []
    original_content = first.read_bytes()

    def change_previous_media(path):
        digest = original(path)
        hashed.append(path)
        if len(hashed) == 2:
            # The first media has already passed its individual post-hash check.
            # A later read must not make its stale digest look current again.
            previous = hashed[0].stat()
            hashed[0].write_bytes(b"y" * len(original_content))
            os.utime(hashed[0], ns=(previous.st_atime_ns, previous.st_mtime_ns + 1_000_000))
        return digest

    monkeypatch.setattr(storage, "_digest", change_previous_media)
    manager = StorageManager(library)
    with pytest.raises(ValueError, match="变化"):
        manager.plan()
    assert first.read_bytes() == b"y" * len(original_content)
    assert second.read_bytes() == original_content
    assert not manager._plans
    assert not os.path.samefile(first, second)


@pytest.mark.parametrize("status", ["uploading", "processing", "reviewing"])
def test_task_started_after_preview_still_blocks_storage_replacement(managed_library, status):
    library, first, second = managed_library
    manager = StorageManager(library)
    contents = first.read_bytes(), second.read_bytes()
    plan = manager.plan()
    assert plan["operation_count"] == 1
    item_id = next(iter(library.items))
    with library.lock:
        library.jobs["new_live_task"] = {"library_id": item_id, "status": status}
    with pytest.raises(ValueError, match="变化"):
        manager.optimize(plan["plan_token"])
    assert (first.read_bytes(), second.read_bytes()) == contents
    assert not os.path.samefile(first, second)
    assert not list(first.parent.glob(".storage-link-*"))


def test_stats_tolerates_media_disappearing_after_managed_enumeration(
    managed_library, monkeypatch
):
    library, first, second = managed_library
    manager = StorageManager(library)
    original = manager._managed_videos
    deleted = False

    def remove_enumerated_media(*args, **kwargs):
        nonlocal deleted
        paths = original(*args, **kwargs)
        if not deleted:
            second.unlink()
            deleted = True
        return paths

    monkeypatch.setattr(manager, "_managed_videos", remove_enumerated_media)
    stats = manager.stats()
    assert deleted and not second.exists()
    assert stats["managed_video_bytes"] == first.stat().st_size
    assert stats["logical_bytes"] >= stats["unique_file_bytes"] >= 0
    assert first.read_bytes() == b"identical-managed-video"
