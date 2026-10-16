"""The local workspace must run real jobs without exposing arbitrary local files."""

from __future__ import annotations

import json
import threading
from http.client import HTTPConnection
from urllib.parse import quote

import pytest

from visualmap.web_app import WorkspaceServer, main


@pytest.fixture
def server(tmp_path, monkeypatch):
    instance = WorkspaceServer(tmp_path, port=0)
    monkeypatch.setattr(instance, "run_job", lambda *_: None)
    thread = threading.Thread(target=instance.serve_forever, daemon=True)
    thread.start()
    try:
        yield instance
    finally:
        instance.shutdown()
        instance.server_close()
        thread.join(timeout=2)


def request(server, method, url, body=None, headers=None):
    connection = HTTPConnection("127.0.0.1", server.server_port, timeout=3)
    connection.request(method, url, body=body, headers=headers or {})
    response = connection.getresponse()
    result = response.status, dict(response.getheaders()), response.read()
    connection.close()
    return result


def test_local_page_and_packaged_font(server):
    status, _, content = request(server, "GET", "/")
    assert status == 200
    assert b"__TOKEN__" not in content
    assert b"data:font/woff2;base64," in content


def test_existing_workspace_identifies_itself(server):
    status, _, content = request(server, "GET", "/api/info")
    assert status == 200
    assert json.loads(content) == {"application": "VisualMap", "protocol": 1}


def test_starting_again_reuses_the_existing_workspace(server, monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["workspace", "--port", str(server.server_port), "--no-open"])
    main()
    assert "已运行" in capsys.readouterr().out
    assert request(server, "GET", "/api/info")[0] == 200


@pytest.mark.parametrize(
    "headers",
    [
        {},
        {"X-VisualMap-Token": "wrong"},
        {"Origin": "https://external.example"},
        {"Host": "external.example"},
    ],
)
def test_upload_requires_local_host_and_session_token(server, headers):
    status, _, _ = request(server, "POST", "/api/jobs?filename=example.mp4", b"video", headers)
    assert status == 403
    assert not server.jobs


def test_upload_video_filename_never_becomes_disk_path(server):
    headers = {"X-VisualMap-Token": server.token}
    status, _, content = request(
        server, "POST", "/api/jobs?filename=..%2F..%2Fexample.mp4", b"video", headers
    )
    assert status == 202
    job_id = json.loads(content)["id"]
    uploaded = list((server.root / "outputs/uploads").glob("*.mp4"))
    assert len(uploaded) == 1
    assert uploaded[0].name == job_id + ".mp4"
    assert uploaded[0].read_bytes() == b"video"
    status, _, content = request(server, "GET", "/api/jobs/" + job_id)
    assert status == 200
    assert json.loads(content)["status"] == "processing"


def test_arbitrary_files_and_traversal_rejected(server):
    (server.root / "private.json").write_text("private")
    status, _, _ = request(server, "GET", "/results/../private.json")
    assert status == 404
    directory = server.root / "outputs/run"
    directory.mkdir(parents=True)
    (directory / "settings.py").write_text("private")
    status, _, _ = request(server, "GET", "/results/run/settings.py")
    assert status == 404


def test_video_ranges_allow_seeking(server):
    directory = server.root / "outputs/run"
    directory.mkdir(parents=True)
    (directory / "source_video.mp4").write_bytes(b"0123456789")
    status, headers, body = request(
        server, "GET", "/results/run/source_video.mp4", headers={"Range": "bytes=2-5"}
    )
    assert status == 206
    assert headers["Content-Range"] == "bytes 2-5/10"
    assert body == b"2345"


def test_unsupported_upload_rejected_without_job(server):
    status, _, _ = request(
        server,
        "POST",
        "/api/jobs?filename=anything.py",
        b"code",
        {"X-VisualMap-Token": server.token},
    )
    assert status == 400
    assert not server.jobs


@pytest.mark.parametrize(
    "extent, expected_stage, complete",
    [
        (
            {"full_video_processed": True, "completeness": "matches_reported_length"},
            "识别完成",
            True,
        ),
        (
            {"full_video_processed": False, "completeness": "requested_partial"},
            "所选片段处理完成",
            False,
        ),
        (
            {"full_video_processed": False, "completeness": "unverified"},
            "处理完成（读取范围未核实）",
            False,
        ),
    ],
)
def test_completion_preserves_decode_extent(server, extent, expected_stage, complete):
    result = server.root / "outputs/job/semantic_map.html"
    result.parent.mkdir(parents=True)
    result.write_text("viewer")
    (result.parent / "run_report.json").write_text(json.dumps({"video_ingestion": extent}))
    server.jobs["job"] = {"status": "processing"}
    server.finish_job("job", 0, result)
    assert server.jobs["job"]["stage"] == expected_stage
    assert server.jobs["job"]["source_complete"] is complete


def test_decode_failure_cannot_become_success_even_with_a_stale_viewer(server):
    result = server.root / "outputs/job/semantic_map.html"
    result.parent.mkdir(parents=True)
    result.write_text("stale viewer")
    (result.parent / "run_report.json").write_text(
        json.dumps(
            {
                "run_status": "failed_video_decode",
                "video_ingestion": {"next_frame_index": 12},
            }
        )
    )
    server.jobs["job"] = {"status": "processing", "last_log": "Wrote report.json"}
    server.finish_job("job", 1, result)
    assert server.jobs["job"]["status"] == "error"
    assert "第 12 帧" in server.jobs["job"]["error"]
    assert "result_url" not in server.jobs["job"]


def upload_library(server, content=b"0123456789", filename="库视频.mp4"):
    return request(
        server,
        "POST",
        "/api/library?filename=" + quote(filename),
        content,
        {"X-VisualMap-Token": server.token},
    )


def process_library(server, item_id):
    return request(
        server,
        "POST",
        f"/api/library/{item_id}/process",
        b"{}",
        {"X-VisualMap-Token": server.token, "Content-Type": "application/json"},
    )


def test_upload_only_persists_and_does_not_automatically_recognize(server):
    status, _, content = upload_library(server)
    item = json.loads(content)["item"]
    assert status == 201
    assert item["name"] == "库视频.mp4"
    assert item["status"] == "ready"
    assert item["results"] == [] and item["job_id"] is None
    assert not server.jobs
    status, _, content = request(server, "GET", "/api/library")
    listing = json.loads(content)
    assert status == 200 and listing["items"][0] == item
    assert listing["active_job_id"] is None
    assert str(server.root) not in content.decode()
    assert (server.library.index_path).is_file()


def test_legacy_upload_is_also_persisted_in_library(server):
    status, _, content = request(
        server,
        "POST",
        "/api/jobs?filename=original.mp4",
        b"actual-bytes",
        {"X-VisualMap-Token": server.token},
    )
    assert status == 202
    job_id = json.loads(content)["id"]
    listing = server.library.listing()
    assert len(listing["items"]) == 1
    item = listing["items"][0]
    assert item["name"] == "original.mp4" and item["job_id"] == job_id
    assert listing["active_job_id"] == job_id
    assert server.jobs[job_id]["library_id"] == item["id"]


def test_one_recognition_job_at_a_time_but_another_upload_can_be_saved(server):
    first = json.loads(upload_library(server, b"first")[2])["item"]
    code, _, response = process_library(server, first["id"])
    job_id = json.loads(response)["id"]
    assert code == 202
    second = json.loads(upload_library(server, b"second", "later.mov")[2])["item"]
    assert second["status"] == "ready"
    assert len(server.library.items) == 2
    for identity in (first["id"], second["id"]):
        code, _, response = process_library(server, identity)
        assert code == 409
        assert json.loads(response)["active_job_id"] == job_id
    assert len(server.jobs) == 1
    code, _, _ = request(
        server,
        "POST",
        "/api/jobs?filename=old.mp4",
        b"legacy",
        {"X-VisualMap-Token": server.token},
    )
    assert code == 409


def test_error_retry_uses_a_new_job_and_keeps_existing_history(server):
    item = json.loads(upload_library(server)[2])["item"]
    first_id = json.loads(process_library(server, item["id"])[2])["id"]
    with server.lock:
        server.jobs[first_id].update(status="error", error="视频处理失败")
        server.library.items[item["id"]]["results"].append(
            {
                "id": "older_run",
                "result_url": "/results/older_run/semantic_map.html",
                "created_at": "2026-10-01T00:00:00Z",
                "title": "已有结果",
            }
        )
        server.library.save(force=True)
    code, _, response = process_library(server, item["id"])
    second_id = json.loads(response)["id"]
    assert code == 202 and first_id != second_id
    assert server.jobs[first_id]["status"] == "error"
    assert server.library.items[item["id"]]["results"][0]["id"] == "older_run"
    assert server.library.public_item(server.library.items[item["id"]])["job_id"] == second_id


@pytest.mark.parametrize(
    "range_header, status, expected",
    [
        ("bytes=2-5", 206, b"2345"),
        ("bytes=7-", 206, b"789"),
        ("bytes=-3", 206, b"789"),
        ("bytes=0-999", 206, b"0123456789"),
        ("bytes=10-", 416, b""),
        ("bytes=8-4", 416, b""),
        ("bytes=-0", 416, b""),
        ("bytes=0-1,4-5", 416, b""),
        ("bytes=-", 416, b""),
        ("nonsense", 416, b""),
    ],
)
def test_library_media_supports_valid_ranges_and_rejects_invalid_ranges(
    server, range_header, status, expected
):
    item = json.loads(upload_library(server)[2])["item"]
    code, headers, body = request(server, "GET", item["video_url"], headers={"Range": range_header})
    assert code == status and body == expected
    assert headers["Content-Range"].endswith("/10")
    assert headers["Content-Length"] == str(len(expected))


def test_library_filename_paths_are_display_names_not_filesystem_authority(server):
    code, _, response = upload_library(server, filename="C:\\private\\..\\原始视频.mp4")
    item = json.loads(response)["item"]
    assert code == 201 and item["name"] == "原始视频.mp4"
    source = server.library.video_path(item["id"])
    assert source.parent == server.root / "outputs/uploads"
    assert "原始视频" not in source.name
    for url in (
        "/media/../../private.json",
        "/media/..%2F..%2Fprivate.mp4",
        "/thumbnails/../../private.jpg",
        "/results/library/index.json",
        "/results/uploads/" + source.name,
    ):
        assert request(server, "GET", url)[0] == 404


@pytest.mark.parametrize(
    "route",
    [
        "/api/library?filename=a.mp4",
        "/api/library/unknown/process",
        "/api/library/unknown/trash",
        "/api/library/unknown/restore",
    ],
)
def test_all_library_mutations_require_token_and_same_origin(server, route):
    for headers in (
        {},
        {"X-VisualMap-Token": "wrong"},
        {"X-VisualMap-Token": server.token, "Origin": "https://external.example"},
        {"X-VisualMap-Token": server.token, "Sec-Fetch-Site": "cross-site"},
    ):
        assert request(server, "POST", route, b"{}", headers)[0] == 403
    assert not server.jobs and not server.library.items


def test_malformed_process_body_cannot_start_a_task(server):
    item = json.loads(upload_library(server)[2])["item"]
    for body in (b"not-json", b"[]", b"null"):
        code, _, _ = request(
            server,
            "POST",
            f"/api/library/{item['id']}/process",
            body,
            {"X-VisualMap-Token": server.token},
        )
        assert code == 400
    assert not server.jobs
    assert process_library(server, "unknown")[0] == 404


def test_api_job_diagnostics_do_not_expose_private_paths(server):
    item = json.loads(upload_library(server)[2])["item"]
    job_id = json.loads(process_library(server, item["id"])[2])["id"]
    server.jobs[job_id]["last_log"] = str(server.root / "private/config.yaml")
    result = server.root / "outputs" / job_id / "semantic_map.html"
    result.parent.mkdir()
    (result.parent / "run_report.json").write_text(json.dumps({"error": str(server.root)}))
    server.finish_job(job_id, 1, result)
    _, _, content = request(server, "GET", "/api/jobs/" + job_id)
    row = json.loads(content)
    assert row["status"] == "error" and "last_log" not in row
    assert str(server.root) not in content.decode()
    assert str(server.root) not in json.dumps(server.library.listing(), ensure_ascii=False)


def test_duplicate_server_cannot_interrupt_a_current_persisted_job(server):
    item = json.loads(upload_library(server)[2])["item"]
    job_id = json.loads(process_library(server, item["id"])[2])["id"]
    before = server.library.index_path.read_bytes()
    with pytest.raises(OSError):
        WorkspaceServer(server.root, server.server_port)
    assert server.jobs[job_id]["status"] == "processing"
    assert server.library.index_path.read_bytes() == before


def test_completion_persists_result_on_the_same_library_item(server):
    item = json.loads(upload_library(server)[2])["item"]
    job_id = json.loads(process_library(server, item["id"])[2])["id"]
    directory = server.root / "outputs" / job_id
    directory.mkdir()
    (directory / "semantic_map.html").write_text("viewer")
    (directory / "source_video.mp4").write_bytes(b"0123456789")
    (directory / "run_report.json").write_text(
        json.dumps(
            {
                "frames_read": 48,
                "video_ingestion": {
                    "full_video_processed": False,
                    "completeness": "requested_partial",
                },
            }
        )
    )
    server.finish_job(job_id, 0, directory / "semantic_map.html")
    _, _, content = request(server, "GET", "/api/library")
    row = json.loads(content)["items"][0]
    assert row["id"] == item["id"] and row["status"] == "complete"
    assert len(server.library.items) == 1
    assert row["results"][0]["requested_partial"] is True
    assert row["results"][0]["source_complete"] is False
    assert row["results"][0]["processed_frames"] == 48


def test_closing_workspace_marks_unfinished_jobs_as_interrupted(server):
    item = json.loads(upload_library(server)[2])["item"]
    job_id = json.loads(process_library(server, item["id"])[2])["id"]
    server.shutdown()
    server.server_close()
    assert server.jobs[job_id]["status"] == "interrupted"
    assert server.library.active_job_id() is None
    assert server.start_job(item["id"])[0] == 503


def library_action(server, item_id, action, body=b"{}"):
    return request(
        server,
        "POST",
        f"/api/library/{item_id}/{action}",
        body,
        {"X-VisualMap-Token": server.token},
    )


def test_recycle_api_hides_media_and_restore_preserves_upload(server):
    item = json.loads(upload_library(server)[2])["item"]
    source = server.library.video_path(item["id"])
    before = source.read_bytes()
    code, _, content = library_action(server, item["id"], "trash")
    assert code == 200 and json.loads(content)["item"]["status"] == "trashed"
    listing = json.loads(request(server, "GET", "/api/library")[2])
    assert listing["items"] == [] and listing["trashed_items"][0]["id"] == item["id"]
    assert request(server, "GET", item["video_url"])[0] == 404
    assert process_library(server, item["id"])[0] == 409
    code, _, content = library_action(server, item["id"], "restore")
    assert code == 200 and json.loads(content)["item"]["status"] == "ready"
    assert request(server, "GET", item["video_url"])[2] == before
    assert source.read_bytes() == before


def test_recycle_api_refuses_running_video_and_invalid_requests(server):
    item = json.loads(upload_library(server)[2])["item"]
    assert process_library(server, item["id"])[0] == 202
    assert library_action(server, item["id"], "trash")[0] == 409
    assert library_action(server, "unknown", "trash")[0] == 404
    assert library_action(server, "unknown", "restore")[0] == 404
    for action in ("trash", "restore"):
        for body in (b"not-json", b"null", b"[]"):
            assert library_action(server, item["id"], action, body)[0] == 400
    assert not server.library.items[item["id"]].get("deleted_at")


def test_uploading_same_content_restores_trashed_video(server):
    item = json.loads(upload_library(server)[2])["item"]
    assert library_action(server, item["id"], "trash")[0] == 200
    restored = json.loads(upload_library(server)[2])["item"]
    assert restored["id"] == item["id"] and restored["status"] == "ready"
    assert len(server.library.items) == 1
    assert server.library.listing()["trashed_items"] == []
