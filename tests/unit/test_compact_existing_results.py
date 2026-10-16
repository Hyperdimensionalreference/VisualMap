"""Archive bytes, payloads, paths, and busy-state checks remain verifiable."""

from __future__ import annotations

import gzip
import importlib.util
import json
from pathlib import Path

import pytest

from visualmap.visualization.semantic_html import viewer_payload_from_html

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/compact_existing_results.py"
SPEC = importlib.util.spec_from_file_location("compact_existing_results", SCRIPT)
compact = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(compact)


@pytest.fixture
def archive(tmp_path, monkeypatch):
    run = tmp_path / "outputs/scene01"
    run.mkdir(parents=True)
    data = {
        "objects": [
            {
                "label": "耳机盒",
                "xyz": [1.23456789012345, -0.0, 4],
                "metadata": {"verified": False, "depth": None},
            }
        ]
        * 500
    }
    for name in compact.JSON_FILES:
        (run / name).write_text(json.dumps(data, ensure_ascii=False, indent=4), encoding="utf-8")
    payload = {
        "title": "旧版空间结果",
        "objects": [],
        "map_points": [[1.23456789012345, -0.0, 3]] * 1000,
        "trajectory": [],
    }
    html = (
        "<html><script>const DATA = "
        + json.dumps(payload, ensure_ascii=False)
        + ";\n</script></html>"
    )
    (run / "semantic_map.html").write_text(html, encoding="utf-8")
    (run / "source_video.mp4").write_bytes(b"synthetic-original-video-never-altered")
    monkeypatch.setattr(
        compact,
        "check_service_idle",
        lambda *_args, **_kwargs: {"service_online": True, "idle_verified": True, "port": 8765},
    )
    return tmp_path, run, data, payload


def test_default_plan_does_not_write_backups_or_alter_any_source(archive):
    root, run, _, _ = archive
    originals = {path: path.read_bytes() for path in run.iterdir()}
    report = compact.compact_existing_results(root)
    assert report["mode"] == "plan" and report["operation_count"] == 5
    assert report["backup_bytes"] > 0
    assert not (run / ".optimization-backups").exists()
    assert all(path.read_bytes() == original for path, original in originals.items())
    assert report["recognition_or_geometry_recomputed"] is False


def test_apply_preserves_all_json_and_html_payloads_and_exact_rollback_bytes(archive):
    root, run, expected, payload = archive
    originals = {path: path.read_bytes() for path in run.iterdir()}
    report = compact.compact_existing_results(root, apply=True)
    assert report["error_count"] == 0 and report["operation_count"] == 5
    for name in compact.JSON_FILES:
        assert json.loads((run / name).read_text(encoding="utf-8")) == expected
    assert viewer_payload_from_html((run / "semantic_map.html").read_text(encoding="utf-8")) == {
        **payload,
        "run_id": "scene01",
    }
    assert (run / "source_video.mp4").read_bytes() == originals[run / "source_video.mp4"]
    for row in report["files"]:
        if row["status"] != "applied":
            continue
        path = root / row["path"]
        rollback = gzip.decompress((root / row["backup"]).read_bytes())
        assert rollback == originals[path]
        assert compact._hash(rollback) == row["sha256_before"]
        assert compact._hash(path.read_bytes()) == row["sha256_after"]
        path.write_bytes(rollback)
        assert path.read_bytes() == originals[path]
    assert (
        report["net_saved_bytes"]
        == report["bytes_before"] - report["bytes_after"] - report["backup_bytes"]
    )


def test_existing_backup_is_never_replaced_even_after_source_changes(archive):
    root, run, _, _ = archive
    compact.compact_existing_results(root, apply=True)
    backup = run / ".optimization-backups/semantics.json.gz"
    original_backup = backup.read_bytes()
    later = b'{"later_user_update":true}'
    (run / "semantics.json").write_bytes(later)
    second = compact.compact_existing_results(root, apply=True)
    assert second["operation_count"] == 0
    assert (run / "semantics.json").read_bytes() == later
    assert backup.read_bytes() == original_backup
    assert all(row["reason"] == "existing_rollback_copy" for row in second["files"])


def test_only_whitelisted_files_in_direct_legal_run_directories_are_considered(archive):
    root, run, _, _ = archive
    (run / "private.json").write_text('{"never":"touch"}', encoding="utf-8")
    for name in ("library", "uploads", "nested/scene02", "fixture_test"):
        directory = root / "outputs" / name
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "semantics.json").write_text('{"private":true}', encoding="utf-8")
    report = compact.compact_existing_results(root)
    assert len(report["files"]) == 5
    assert all(row["path"].startswith("outputs/scene01/") for row in report["files"])


def test_payload_changed_by_renderer_fails_before_any_write(archive, monkeypatch):
    root, run, _, _ = archive
    original = (run / "semantic_map.html").read_bytes()

    def wrong_renderer(payload):
        payload["title"] = "changed"
        return "<script>const DATA = " + json.dumps(payload) + ";\n</script>"

    monkeypatch.setattr(compact, "render_viewer_payload", wrong_renderer)
    report = compact.compact_existing_results(root, apply=True)
    html = next(row for row in report["files"] if row["path"].endswith(".html"))
    assert html["status"] == "error" and report["error_count"] == 1
    assert (run / "semantic_map.html").read_bytes() == original
    assert not (run / ".optimization-backups/semantic_map.html.gz").exists()


@pytest.mark.parametrize("content", [b'{"x":1,"x":2}', b'{"x":NaN}', b'{"invalid":'])
def test_ambiguous_or_invalid_json_stays_unchanged(archive, content):
    root, run, _, _ = archive
    path = run / "semantics.json"
    path.write_bytes(content)
    report = compact.compact_existing_results(root, apply=True)
    row = next(row for row in report["files"] if row["path"].endswith("/semantics.json"))
    assert row["status"] == "error"
    assert path.read_bytes() == content
    assert not (run / ".optimization-backups/semantics.json.gz").exists()


def test_post_write_corruption_restores_original_from_verified_bytes(archive, monkeypatch):
    root, run, _, _ = archive
    path = run / "semantics.json"
    original = path.read_bytes()
    writer = compact._write_atomic
    calls = []

    def corrupt_once(target, data, base):
        if target == path and not calls:
            calls.append(True)
            return writer(target, b"not-json", base)
        return writer(target, data, base)

    monkeypatch.setattr(compact, "_write_atomic", corrupt_once)
    report = compact.compact_existing_results(root, apply=True)
    assert report["error_count"] == 1
    assert path.read_bytes() == original
    assert (
        gzip.decompress((run / ".optimization-backups/semantics.json.gz").read_bytes()) == original
    )
    assert report["backup_bytes"] == sum(
        path.stat().st_size for path in (run / ".optimization-backups").glob("*.gz")
    )
    assert (
        report["net_saved_bytes"]
        == report["bytes_before"] - report["bytes_after"] - report["backup_bytes"]
    )


def test_internal_junction_is_rejected_even_if_it_resolves_inside_project(archive, monkeypatch):
    root, run, _, _ = archive
    original = getattr(Path, "is_junction", lambda _self: False)
    monkeypatch.setattr(
        Path, "is_junction", lambda self: self == run or original(self), raising=False
    )
    report = compact.compact_existing_results(root, apply=True)
    assert report["operation_count"] == 0
    assert report["files"] == [
        {"path": "outputs/scene01", "status": "skipped", "reason": "linked_directory"}
    ]


def test_busy_service_blocks_every_write_and_report_cannot_overwrite_source(archive, monkeypatch):
    root, run, _, _ = archive

    def busy(*_args, **_kwargs):
        raise RuntimeError("processing")

    monkeypatch.setattr(compact, "check_service_idle", busy)
    with pytest.raises(RuntimeError):
        compact.compact_existing_results(root, apply=True)
    assert not (run / ".optimization-backups").exists()
    assert (
        compact.main(["--root", str(root), "--apply", "--output", str(run / "semantics.json")]) == 2
    )


def test_real_idle_check_detects_live_job_and_persisted_review(tmp_path, monkeypatch):
    class Connection:
        def __init__(self, *_args, **_kwargs):
            pass

        def request(self, *_args):
            pass

        def close(self):
            pass

        def getresponse(self):
            return self

        status = 200
        active = "job1"

        def read(self, *_args):
            return json.dumps({"active_job_id": self.active}).encode()

    monkeypatch.setattr(compact, "HTTPConnection", Connection)
    with pytest.raises(RuntimeError, match="正在识别"):
        compact.check_service_idle(tmp_path)
    Connection.active = None
    reviews = tmp_path / "work/llm/reviews"
    reviews.mkdir(parents=True)
    (reviews / "review_1.json").write_text('{"status":"processing"}')
    with pytest.raises(RuntimeError, match="复核进行中"):
        compact.check_service_idle(tmp_path)
    (reviews / "review_1.json").write_text('{"status":"complete"}')
    assert compact.check_service_idle(tmp_path)["idle_verified"] is True
