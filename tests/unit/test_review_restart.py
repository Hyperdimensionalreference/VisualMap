"""Interrupted review recovery is durable before the service reports it idle."""

from __future__ import annotations

import json
import runpy
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from visualmap.workspace_extensions import WorkspaceExtensions, handle_extension


@pytest.fixture
def offline_server(tmp_path):
    lock = threading.RLock()
    return SimpleNamespace(
        root=tmp_path,
        lock=lock,
        library=SimpleNamespace(lock=lock, items={}, tombstones={}),
        extensions=None,
    )


def write_review(server, identifier="review_interrupted", **changes):
    directory = server.root / "work/llm/reviews"
    directory.mkdir(parents=True, exist_ok=True)
    row = {
        "id": identifier,
        "run_id": "scene01",
        "library_id": "video01",
        "track_id": 7,
        "status": "processing",
        "created_at": "synthetic-interrupted-time",
        "original_label": "remote",
        **changes,
    }
    path = directory / (identifier + ".json")
    path.write_text(json.dumps(row, ensure_ascii=False), encoding="utf-8")
    return path, row


def test_restart_persists_interruption_and_second_restart_is_identical(
    offline_server, monkeypatch
):
    path, original = write_review(offline_server)
    completed_path, completed = write_review(
        offline_server,
        "review_completed",
        status="complete",
        result={"candidate_name": "耳机盒", "evidence": ["盒盖接缝"]},
    )
    completed_bytes = completed_path.read_bytes()

    first = WorkspaceExtensions(offline_server)
    recovered = first.reviews[original["id"]]
    assert recovered == {
        **original,
        "status": "error",
        "error": "上次复核已中断，请手动重新提交。",
    }
    assert json.loads(path.read_text(encoding="utf-8")) == recovered
    assert first.reviews[completed["id"]] == completed
    assert completed_path.read_bytes() == completed_bytes
    first.ensure_not_reviewing()
    persisted_bytes = path.read_bytes()

    second = WorkspaceExtensions(offline_server)
    assert second.reviews == first.reviews
    assert path.read_bytes() == persisted_bytes
    second.ensure_not_reviewing()

    # Exercise the separate maintenance reader against persisted state without
    # ever contacting a running workbench or a remote provider.
    compact = runpy.run_path(
        str(Path(__file__).resolve().parents[2] / "scripts/compact_existing_results.py")
    )

    class OfflineConnection:
        def __init__(self, *args, **kwargs):
            pass

        def request(self, *args, **kwargs):
            raise ConnectionRefusedError("synthetic offline workbench")

        def close(self):
            pass

    monkeypatch.setitem(
        compact["check_service_idle"].__globals__, "HTTPConnection", OfflineConnection
    )
    assert compact["check_service_idle"](offline_server.root) == {
        "service_online": False,
        "idle_verified": True,
        "port": 8765,
    }


@pytest.mark.parametrize("failure_stage", ["temporary_write", "atomic_replace"])
def test_failed_recovery_preserves_original_and_next_request_retries(
    offline_server, monkeypatch, failure_stage
):
    path, original = write_review(offline_server)
    original_bytes = path.read_bytes()
    responses = []
    handler = SimpleNamespace(
        server=offline_server,
        json=lambda status, body: responses.append((status, body)),
    )
    real_write_text = Path.write_text
    real_replace = Path.replace

    def fail_temporary_write(temporary, *args, **kwargs):
        if temporary.parent == path.parent and temporary.name.startswith(".assist-"):
            # A partially written temporary file must never replace the original.
            real_write_text(temporary, "partial", encoding="utf-8")
            raise OSError("synthetic full disk")
        return real_write_text(temporary, *args, **kwargs)

    def fail_replace(temporary, target):
        if Path(target) == path:
            assert json.loads(temporary.read_text(encoding="utf-8"))["status"] == "error"
            assert path.read_bytes() == original_bytes
            raise PermissionError("synthetic locked review file")
        return real_replace(temporary, target)

    with monkeypatch.context() as failure:
        if failure_stage == "temporary_write":
            failure.setattr(Path, "write_text", fail_temporary_write)
        else:
            failure.setattr(Path, "replace", fail_replace)
        assert handle_extension(handler, "/api/llm/settings", {}, mutation=False)
        assert responses[-1][0] == 503
        assert responses[-1][1]["code"] == "storage_unavailable"
        assert offline_server.extensions is None
        assert path.read_bytes() == original_bytes
        assert json.loads(path.read_text(encoding="utf-8")) == original
        assert list(path.parent.glob(".assist-*.tmp")) == []

    assert handle_extension(handler, "/api/llm/settings", {}, mutation=False)
    assert responses[-1][0] == 200
    assert offline_server.extensions.reviews[original["id"]]["status"] == "error"
    assert json.loads(path.read_text(encoding="utf-8"))["status"] == "error"
    offline_server.extensions.ensure_not_reviewing()


@pytest.mark.parametrize(
    "identifier", [None, 123, "", "../escaped", "review_other", "r" * 121]
)
def test_invalid_or_mismatched_review_id_is_never_loaded_or_rewritten(
    offline_server, identifier
):
    path, original = write_review(offline_server)
    original["id"] = identifier
    path.write_text(json.dumps(original), encoding="utf-8")
    original_bytes = path.read_bytes()

    extension = WorkspaceExtensions(offline_server)

    assert extension.reviews == {}
    assert path.read_bytes() == original_bytes
    assert list(path.parent.iterdir()) == [path]


def test_linked_review_is_never_loaded_or_rewritten(offline_server, tmp_path):
    outside = tmp_path / "outside-review.json"
    original_bytes = json.dumps({"id": "review_linked", "status": "processing"}).encode()
    outside.write_bytes(original_bytes)
    directory = offline_server.root / "work/llm/reviews"
    directory.mkdir(parents=True)
    path = directory / "review_linked.json"
    try:
        path.symlink_to(outside)
    except OSError:
        pytest.skip("symbolic link creation is unavailable on this host")

    extension = WorkspaceExtensions(offline_server)

    assert extension.reviews == {}
    assert path.is_symlink()
    assert outside.read_bytes() == original_bytes
