"""Storage operations are verified on isolated fixtures, never the real video library."""

from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path

import pytest

from visualmap.storage_manager import StorageManager, copy_media_atomic
from visualmap.video_library import VideoLibrary, inside_without_symlinks


def put(path: Path, content: bytes = b"identical-video-content") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def result(root: Path, name: str, content: bytes = b"identical-video-content") -> Path:
    directory = root / "outputs" / name
    put(directory / "semantic_map.html", b"<html>viewer</html>")
    put(directory / "source_video.mp4", content)
    put(directory / "depth/frame.npy", b"original-depth-evidence")
    put(directory / "semantics.json", b'{"original_model_evidence":true}')
    put(directory / "run_report.json", b'{"video":{"path":"original.mp4"}}')
    return directory


def trashed(root: Path):
    original = put(root / "sample_data/original.mp4")
    upload = put(root / "outputs/uploads/import.mp4")
    directory = result(root, "video_completed")
    library = VideoLibrary(root)
    item = next(iter(library.items.values()))
    thumbnail = put(library.thumbnails / (item["id"] + ".jpg"), b"thumbnail")
    library.set_trashed(item["id"], True)
    return library, item, original, upload, directory, thumbnail


def test_purge_requires_preview_and_trash_and_consumes_token_once(tmp_path):
    put(tmp_path / "outputs/uploads/import.mp4")
    library = VideoLibrary(tmp_path)
    item_id = next(iter(library.items))
    with pytest.raises(ValueError, match="预览"):
        library.purge(item_id, "made-up")
    with pytest.raises(ValueError, match="回收站"):
        library.plan_purge(item_id)
    library.set_trashed(item_id, True)
    plan = library.plan_purge(item_id)
    assert plan["irreversible"] and plan["confirmation_token"]
    assert plan["total_bytes"] > 0 and plan["file_count"] == 1
    outcome = library.purge(item_id, plan["confirmation_token"])
    assert outcome["purged"] and outcome["deleted_files"] == 1
    with pytest.raises(ValueError, match="过期"):
        library.purge(item_id, plan["confirmation_token"])


def test_purge_removes_only_owned_managed_copies_and_survives_rescan_restart(tmp_path):
    library, item, original, upload, directory, thumbnail = trashed(tmp_path)
    keep = {
        put(tmp_path / "models/model.onnx", b"model"): b"model",
        put(tmp_path / "src/visualmap/app.py", b"source-code"): b"source-code",
        put(tmp_path / "outputs/unrelated/report.json", b"unrelated"): b"unrelated",
        original: original.read_bytes(),
    }
    plan = library.plan_purge(item["id"])
    assert {row["kind"] for row in plan["files"]} == {"upload", "result", "thumbnail"}
    assert {row["path"] for row in plan["retained_sources"]} == {"sample_data/original.mp4"}
    for row in plan["files"]:
        assert row["path"].startswith(("outputs/uploads/", "outputs/video_completed/",
                                       "outputs/library/thumbnails/"))
    outcome = library.purge(item["id"], plan["confirmation_token"])
    assert outcome["removed_runs"] == ["video_completed"]
    assert not upload.exists() and not directory.exists() and not thumbnail.exists()
    assert all(path.read_bytes() == content for path, content in keep.items())
    library.refresh(force=True)
    assert library.listing()["items"] == library.listing()["trashed_items"] == []
    restarted = VideoLibrary(tmp_path)
    assert restarted.listing()["items"] == []
    assert restarted.tombstones[item["sha256"]]["status"] == "complete"
    # Only explicit reimport, not discovery, may recreate the purged catalogue item.
    restarted.add_video(original, restore_deleted=True)
    assert restarted.listing()["items"][0]["id"] == item["id"]


def test_shared_paths_and_shared_runs_are_retained_without_resurrection(tmp_path):
    library, item, original, upload, directory, _ = trashed(tmp_path)
    other = copy.deepcopy(item)
    other.update(id="other_video", sha256="b" * 64)
    other.pop("deleted_at")
    library.items[other["id"]] = other
    plan = library.plan_purge(item["id"])
    assert all(row["kind"] == "thumbnail" for row in plan["files"])
    assert any(row["path"] == "outputs/video_completed" for row in plan["retained_sources"])
    library.purge(item["id"], plan["confirmation_token"])
    assert original.exists() and upload.exists() and directory.exists()
    library.refresh(force=True)
    assert item["id"] not in library.items
    assert other["id"] in library.items


@pytest.mark.parametrize("status", ["uploading", "processing", "reviewing"])
def test_active_item_cannot_be_purged_even_if_it_is_already_trashed(tmp_path, status):
    library, item, *_ = trashed(tmp_path)
    library.jobs["live"] = {"library_id": item["id"], "status": status}
    with pytest.raises(ValueError, match="正在处理"):
        library.plan_purge(item["id"])


@pytest.mark.parametrize("change", ["file", "restore", "active", "reference"])
def test_preview_must_match_files_item_and_references_at_confirmation(tmp_path, change):
    library, item, _, upload, directory, _ = trashed(tmp_path)
    plan = library.plan_purge(item["id"])
    if change == "file":
        put(directory / "new_evidence.json", b"new")
    elif change == "restore":
        library.set_trashed(item["id"], False)
    elif change == "active":
        library.jobs["live"] = {"library_id": item["id"], "status": "processing"}
    else:
        library.items["shared"] = {"sources": [{"path": "outputs/uploads/import.mp4"}]}
    with pytest.raises(ValueError):
        library.purge(item["id"], plan["confirmation_token"])
    assert upload.exists() and directory.exists()


def test_expired_or_cross_item_purge_token_cannot_delete(tmp_path, monkeypatch):
    library, item, _, upload, *_ = trashed(tmp_path)
    plan = library.plan_purge(item["id"])
    monkeypatch.setattr("visualmap.storage_manager.time.monotonic", lambda: 10**20)
    with pytest.raises(ValueError, match="过期"):
        library.purge(item["id"], plan["confirmation_token"])
    assert upload.exists()


def test_index_crafted_original_paths_are_never_deleted(tmp_path):
    library, item, original, _, _, _ = trashed(tmp_path)
    private = put(tmp_path / "private/movie.mp4", original.read_bytes())
    item["sources"].extend([
        {"path": "private/movie.mp4"}, {"path": "outputs/uploads/../../sample_data/original.mp4"},
        {"path": str(private)},
    ])
    plan = library.plan_purge(item["id"])
    library.purge(item["id"], plan["confirmation_token"])
    assert private.exists() and original.exists()


def test_symlink_replacement_after_preview_is_refused(tmp_path):
    library, item, original, upload, *_ = trashed(tmp_path)
    plan = library.plan_purge(item["id"])
    upload.unlink()
    try:
        upload.symlink_to(original)
    except OSError:
        pytest.skip("Creating symlinks requires OS permission on this host")
    with pytest.raises(ValueError):
        library.purge(item["id"], plan["confirmation_token"])
    assert original.exists()


def test_reparse_point_attribute_is_rejected_even_for_an_internal_target(tmp_path, monkeypatch):
    directory = tmp_path / "outputs"
    directory.mkdir()
    original_lstat = Path.lstat

    # is_symlink also calls lstat on some platforms; supply the full delegated stat.
    class ProxyStat:
        st_file_attributes = 0x400

        def __init__(self, wrapped):
            self.wrapped = wrapped

        def __getattr__(self, key):
            return getattr(self.wrapped, key)

    monkeypatch.setattr(Path, "lstat", lambda path: (
        ProxyStat(original_lstat(path)) if path == directory else original_lstat(path)
    ))
    assert not inside_without_symlinks(directory, tmp_path)


def test_durable_tombstone_failure_prevents_any_file_deletion(tmp_path, monkeypatch):
    library, item, _, upload, directory, _ = trashed(tmp_path)
    plan = library.plan_purge(item["id"])
    monkeypatch.setattr(library, "save", lambda **_: (_ for _ in ()).throw(OSError("full")))
    with pytest.raises(OSError):
        library.purge(item["id"], plan["confirmation_token"])
    assert upload.exists() and directory.exists()
    assert item["sha256"] not in library.tombstones


def test_partial_deletion_keeps_trashed_record_and_can_be_retried(tmp_path, monkeypatch):
    library, item, _, _, directory, _ = trashed(tmp_path)
    plan = library.plan_purge(item["id"])
    original_unlink = Path.unlink
    block = directory / "semantics.json"

    def fail_one(path, *args, **kwargs):
        if path == block:
            raise OSError("busy file")
        return original_unlink(path, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "unlink", fail_one)
        with pytest.raises(OSError, match="busy"):
            library.purge(item["id"], plan["confirmation_token"])
    assert item["id"] in library.items and item["deleted_at"]
    assert library.tombstones[item["sha256"]]["status"] == "partial"
    restarted = VideoLibrary(tmp_path)
    second = restarted.plan_purge(item["id"])
    restarted.purge(item["id"], second["confirmation_token"])
    assert not directory.exists() and item["id"] not in restarted.items


def test_purge_of_hardlinked_copy_preserves_original_and_reports_logical_separately(tmp_path):
    library, item, original, upload, _, _ = trashed(tmp_path)
    upload.unlink()
    os.link(original, upload)
    plan = library.plan_purge(item["id"])
    outcome = library.purge(item["id"], plan["confirmation_token"])
    assert outcome["freed_logical_bytes"] > outcome["estimated_reclaimed_bytes"]
    assert original.read_bytes() == b"identical-video-content"


def test_storage_hardlinks_only_managed_equal_bytes_and_preserves_every_byte(tmp_path):
    original = put(tmp_path / "sample_data/original.mp4", b"same" * 2048)
    first = put(tmp_path / "outputs/uploads/one.mp4", original.read_bytes())
    second = put(tmp_path / "outputs/uploads/two.mp4", original.read_bytes())
    directory = result(tmp_path, "video_finished", original.read_bytes())
    different = put(tmp_path / "outputs/uploads/different.mp4", b"diff" * 2048)
    evidence = directory / "semantics.json"
    library = VideoLibrary(tmp_path)
    manager = StorageManager(library)
    contents = {path: path.read_bytes() for path in (original, first, second, different, evidence,
                                                   directory / "source_video.mp4")}
    plan = manager.plan()
    assert plan["operation_count"] == 2
    assert plan["potential_saved_bytes"] == 2 * first.stat().st_size
    assert all(not row["source"].startswith("sample_data") for row in plan["operations"])
    outcome = manager.optimize(plan["plan_token"])
    assert outcome["optimized_files"] == 2 and outcome["skipped"] == []
    assert outcome["saved_bytes"] == plan["potential_saved_bytes"]
    assert outcome["after"]["logical_bytes"] == outcome["before"]["logical_bytes"]
    assert (outcome["before"]["unique_file_bytes"] - outcome["after"]["unique_file_bytes"]
            == outcome["saved_bytes"])
    assert all(path.read_bytes() == content for path, content in contents.items())
    assert os.path.samefile(first, second)
    assert not os.path.samefile(first, original)
    assert manager.plan()["operation_count"] == 0


def test_active_media_and_nonmanaged_paths_do_not_enter_storage_plan(tmp_path):
    first = put(tmp_path / "outputs/uploads/one.mp4")
    second = put(tmp_path / "outputs/uploads/two.mp4")
    result(tmp_path, "video_complete")
    put(tmp_path / "models/backup.mp4")
    result(tmp_path, "benchmark_fixture")
    library = VideoLibrary(tmp_path)
    item = next(iter(library.items.values()))
    library.jobs["live"] = {"library_id": item["id"], "status": "processing"}
    plan = StorageManager(library).plan()
    assert plan["operations"] == []
    assert not os.path.samefile(first, second)


def test_storage_plan_rejects_mutation_and_token_reuse(tmp_path):
    put(tmp_path / "outputs/uploads/one.mp4")
    second = put(tmp_path / "outputs/uploads/two.mp4")
    manager = StorageManager(VideoLibrary(tmp_path))
    plan = manager.plan()
    second.write_bytes(b"changed")
    with pytest.raises(ValueError, match="变化"):
        manager.optimize(plan["plan_token"])
    with pytest.raises(ValueError, match="过期"):
        manager.optimize(plan["plan_token"])


def test_hardlink_failure_preserves_original_copies_and_quality(tmp_path, monkeypatch):
    first = put(tmp_path / "outputs/uploads/one.mp4")
    second = put(tmp_path / "outputs/uploads/two.mp4")
    manager = StorageManager(VideoLibrary(tmp_path))
    plan = manager.plan()
    digest = hashlib.sha256(first.read_bytes()).hexdigest()
    monkeypatch.setattr("visualmap.storage_manager.os.link", lambda *_: (
        _ for _ in ()
    ).throw(OSError("unsupported filesystem")))
    outcome = manager.optimize(plan["plan_token"])
    assert outcome["optimized_files"] == 0 and outcome["saved_bytes"] == 0
    assert len(outcome["skipped"]) == 1
    assert all(hashlib.sha256(path.read_bytes()).hexdigest() == digest for path in (first, second))
    assert not list(second.parent.glob(".storage-link-*"))


def test_stats_audits_cache_without_deleting_or_rewriting_it(tmp_path):
    put(tmp_path / "outputs/uploads/one.mp4")
    cache = put(tmp_path / ".pytest_cache/v/cache/nodeids", b"generated-cache")
    manager = StorageManager(VideoLibrary(tmp_path))
    before = cache.read_bytes()
    stats = manager.stats()
    assert stats["cache_bytes"] == len(before)
    assert stats["logical_bytes"] >= stats["unique_file_bytes"] > 0
    assert cache.read_bytes() == before
    # Public plans and stats contain only project-relative paths and JSON values.
    assert str(tmp_path) not in json.dumps({"stats": stats, "plan": manager.plan()})


def test_rewriting_a_linked_run_copy_never_changes_its_sibling_media(tmp_path):
    first = put(tmp_path / "outputs/one/source_video.mp4", b"old-media")
    second = tmp_path / "outputs/two/source_video.mp4"
    second.parent.mkdir(parents=True)
    os.link(first, second)
    replacement = put(tmp_path / "sample_data/new.mp4", b"new-media-content")
    copy_media_atomic(replacement, second)
    assert first.read_bytes() == b"old-media"
    assert second.read_bytes() == replacement.read_bytes()
    assert not os.path.samefile(first, second)
    assert not list(second.parent.glob(".media-copy-*"))


def test_atomic_media_copy_accepts_same_source_destination_without_truncation(tmp_path):
    source = put(tmp_path / "source.mp4", b"original-media")
    assert copy_media_atomic(source, source) == source
    assert source.read_bytes() == b"original-media"
    assert not list(tmp_path.glob(".media-copy-*"))


def test_atomic_media_copy_failure_preserves_previous_destination(tmp_path, monkeypatch):
    source = put(tmp_path / "source.mp4", b"new-media")
    destination = put(tmp_path / "outputs/run/source_video.mp4", b"previous-media")
    monkeypatch.setattr("visualmap.storage_manager.os.replace", lambda *_: (
        _ for _ in ()
    ).throw(OSError("locked destination")))
    with pytest.raises(OSError):
        copy_media_atomic(source, destination)
    assert destination.read_bytes() == b"previous-media"
    assert not list(destination.parent.glob(".media-copy-*"))


def test_storage_does_not_reuse_a_managed_file_linked_to_an_external_original(tmp_path):
    original = put(tmp_path / "sample_data/original.mp4")
    linked = tmp_path / "outputs/uploads/linked.mp4"
    linked.parent.mkdir(parents=True)
    os.link(original, linked)
    put(tmp_path / "outputs/uploads/independent.mp4")
    manager = StorageManager(VideoLibrary(tmp_path))
    assert manager.plan()["operation_count"] == 0
