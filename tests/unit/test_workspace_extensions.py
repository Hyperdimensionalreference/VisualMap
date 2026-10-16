"""Authorized routes with synthetic media and a controllable local review service."""

from __future__ import annotations

import hashlib
import json
import shutil
import threading
import time
from http.client import HTTPConnection

import cv2
import numpy as np
import pytest

from visualmap.web_app import WorkspaceServer

KEY = "synthetic-controller-api-secret"


def request(server, method, route, payload=None, headers=None, authenticated=True):
    values = {"Content-Type": "application/json"}
    if authenticated:
        values["X-VisualMap-Token"] = server.token
        values["Origin"] = server.origin
    values.update(headers or {})
    connection = HTTPConnection("127.0.0.1", server.server_port, timeout=3)
    connection.request(method, route, json.dumps(payload or {}).encode(), values)
    response = connection.getresponse()
    result = response.status, json.loads(response.read())
    connection.close()
    return result


@pytest.fixture
def server(tmp_path):
    source = tmp_path / "sample_data/synthetic.avi"
    source.parent.mkdir()
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"MJPG"), 10, (80, 60))
    assert writer.isOpened()
    for number in range(6):
        writer.write(np.full((60, 80, 3), 20 + number * 20, dtype=np.uint8))
    writer.release()
    run = tmp_path / "outputs/scene01"
    run.mkdir(parents=True)
    shutil.copy2(source, run / "source_video.avi")
    (run / "semantic_map.html").write_text("<html>Synthetic local result</html>", encoding="utf-8")
    (run / "semantics.json").write_text(
        json.dumps({"objects": [{"track_id": 7, "class_name": "remote", "position": [1, 2, 3]}]}),
        encoding="utf-8",
    )
    (run / "run_report.json").write_text(
        json.dumps(
            {
                "video": {"path": str(source)},
                "frames_read": 6,
                "video_ingestion": {"full_video_processed": True},
            }
        ),
        encoding="utf-8",
    )
    instance = WorkspaceServer(tmp_path, port=0)
    thread = threading.Thread(target=instance.serve_forever, daemon=True)
    thread.start()
    try:
        yield instance
    finally:
        instance.shutdown()
        instance.server_close()
        thread.join(2)


def configure(server):
    status, response = request(
        server,
        "POST",
        "/api/llm/settings",
        {
            "base_url": "https://example.invalid/v1",
            "model": "fake-vision",
            "api_key": KEY,
            "key_storage": "session",
        },
    )
    assert status == 200
    return response


def payload(server, track_id=7):
    status, context = request(server, "GET", "/api/llm/context?run_id=scene01")
    assert status == 200
    return {
        "run_id": "scene01",
        "library_id": context["library_id"],
        "track_id": track_id,
        "selections": [{"frame_id": number, "xyxy": [10, 10, 50, 40]} for number in (0, 2, 5)],
        "question": "识别此物体",
    }


def fake_service(server, monkeypatch, block=False):
    # Creating or reading settings never invokes this function. This fake avoids
    # all external network use, while the service-specific suite decodes crops.
    configure(server)
    calls, entered, release = [], threading.Event(), threading.Event()

    def review(video, selections, prompt="", cancel_event=None):
        calls.append((video, selections, prompt))
        entered.set()
        if block:
            release.wait(2)
        return {
            "id": "review_synthetic",
            "status": "candidate",
            "candidate_name": "耳机盒",
            "candidates": [{"name": "耳机盒", "evidence": ["盒盖接缝"], "uncertainty": "品牌未知"}],
            "evidence": ["三个视图盒体一致"],
            "uncertainty": "未确认品牌",
            "same_object": True,
            "provider": {"base_url": "https://example.invalid/v1", "model": "fake-vision"},
            "selections": [
                {
                    **row,
                    "image_sha256": hashlib.sha256(str(row).encode()).hexdigest(),
                    "width": 40,
                    "height": 30,
                    "source_width": 80,
                    "source_height": 60,
                    "timestamp_seconds": row["frame_id"] / 10,
                    "image_bytes": 100,
                }
                for row in selections
            ],
            "requires_confirmation": True,
            "model_training_performed": False,
        }

    monkeypatch.setattr(server.extensions.service, "review", review)
    return calls, entered, release


def complete(server, review_id):
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        status, row = request(server, "GET", "/api/llm/reviews/" + review_id)
        assert status == 200
        if row["status"] != "processing":
            assert row["status"] == "complete", row
            return row
        time.sleep(0.01)
    raise AssertionError("synthetic review did not finish")


@pytest.mark.parametrize(
    "route",
    [
        "/api/llm/settings",
        "/api/llm/reviews",
        "/api/llm/reviews/missing/confirm",
        "/api/llm/labels/missing/revert",
        "/api/storage/optimize",
        "/api/library/missing/purge",
    ],
)
@pytest.mark.parametrize(
    "headers",
    [
        {},
        {"X-VisualMap-Token": "wrong"},
        {"Origin": "https://external.example"},
    ],
)
def test_all_mutation_routes_require_token_and_trusted_origin(server, route, headers):
    status, _ = request(server, "POST", route, headers=headers, authenticated=False)
    assert status == 403
    assert server.extensions is None


def test_valid_token_does_not_authorize_cross_origin(server):
    status, _ = request(
        server, "POST", "/api/llm/settings", headers={"Origin": "https://evil.invalid"}
    )
    assert status == 403
    assert server.extensions is None


def test_public_context_and_settings_never_include_key_or_invoke_review(server, monkeypatch):
    calls, _, _ = fake_service(server, monkeypatch)
    for route in (
        "/api/llm/settings",
        "/api/llm/context?run_id=scene01",
        "/api/llm/labels?run_id=scene01",
    ):
        status, data = request(server, "GET", route)
        assert status == 200
        assert KEY not in json.dumps(data)
    assert calls == []
    stored = (server.root / "work/llm/settings/provider.json").read_text(encoding="utf-8")
    assert KEY not in stored
    assert request(server, "GET", "/results/../work/llm/settings/provider.json")[0] == 404


@pytest.mark.parametrize(
    "changes",
    [
        {"run_id": "../scene01"},
        {"run_id": "missing"},
        {"library_id": "wrong"},
        {"track_id": 999},
        {"track_id": True},
        {"selections": [{"frame_id": 1, "xyxy": [10, 10, 50, 40]}] * 3},
    ],
)
def test_run_track_and_three_distinct_frames_are_authorized(server, monkeypatch, changes):
    calls, _, _ = fake_service(server, monkeypatch)
    values = {**payload(server), **changes}
    status, _ = request(server, "POST", "/api/llm/reviews", values)
    assert status in {400, 404}
    assert calls == []


def test_async_confirm_persist_revoke_restart_preserves_original(server, monkeypatch):
    calls, entered, release = fake_service(server, monkeypatch, block=True)
    original = (server.root / "outputs/scene01/semantics.json").read_bytes()
    status, row = request(server, "POST", "/api/llm/reviews", payload(server))
    assert status == 202
    assert entered.wait(1)
    assert request(server, "GET", "/api/llm/reviews/" + row["id"])[1]["status"] == "processing"
    assert request(server, "GET", "/api/llm/labels?run_id=scene01")[1]["labels"] == []
    assert (
        request(server, "POST", "/api/llm/reviews/" + row["id"] + "/confirm", {"name": "耳机盒"})[0]
        == 400
    )
    release.set()
    result = complete(server, row["id"])
    assert result["result"]["requires_confirmation"] is True
    status, confirmed = request(
        server, "POST", "/api/llm/reviews/" + row["id"] + "/confirm", {"name": "耳机充电盒"}
    )
    assert status == 200
    label = confirmed["label"]
    assert label["track_id"] == 7 and label["original_label"] == "remote"
    assert label["model_training_performed"] is False
    assert KEY not in json.dumps(result) + json.dumps(confirmed)
    assert (server.root / "outputs/scene01/semantics.json").read_bytes() == original
    assert len(calls) == 1
    # A new server instance reads persisted structured evidence, never reissues
    # the provider request or treats the old session secret as persisted.
    restarted = WorkspaceServer(server.root, port=0)
    try:
        from visualmap.workspace_extensions import WorkspaceExtensions

        restarted.extensions = WorkspaceExtensions(restarted)
        labels = restarted.extensions.labels("scene01")["labels"]
        assert labels[0]["name"] == "耳机充电盒"
        assert len(labels[0]["selections"]) == 3
        assert restarted.extensions.settings()["has_api_key"] is False
        assert restarted.extensions.review(row["id"])["status"] == "complete"
    finally:
        restarted.server_close()
    status, reverted = request(server, "POST", "/api/llm/labels/" + label["id"] + "/revert")
    assert status == 200 and reverted["label"]["active"] is False
    assert request(server, "GET", "/api/llm/labels?run_id=scene01")[1]["labels"] == []
    persisted_review = request(server, "GET", "/api/llm/reviews/" + row["id"])[1]
    assert persisted_review["label"]["active"] is False
    assert (server.root / "outputs/scene01/semantics.json").read_bytes() == original


def test_missing_detection_is_a_manual_2d_annotation_without_geometry(server, monkeypatch):
    fake_service(server, monkeypatch)
    status, row = request(server, "POST", "/api/llm/reviews", payload(server, track_id=None))
    assert status == 202
    complete(server, row["id"])
    status, confirmed = request(
        server, "POST", "/api/llm/reviews/" + row["id"] + "/confirm", {"name": "手环"}
    )
    assert status == 200
    label = confirmed["label"]
    assert label["track_id"] is None
    assert label["object_id"].startswith("manual_")
    assert not any(field in label for field in ("position", "xyz", "volume", "world_frame_id"))
    assert len(label["selections"]) == 3


def test_busy_review_blocks_purge_preview_purge_and_storage_mutation(server, monkeypatch):
    _, entered, release = fake_service(server, monkeypatch, block=True)
    values = payload(server)
    status, row = request(server, "POST", "/api/llm/reviews", values)
    assert status == 202 and entered.wait(1)
    for method, route in (
        ("GET", f"/api/library/{values['library_id']}/purge-preview"),
        ("POST", f"/api/library/{values['library_id']}/purge"),
        ("POST", "/api/storage/optimize"),
    ):
        status, error = request(server, method, route, {"confirmation_token": "not-a-token"})
        assert status == 409 and error["code"] == "busy"
    release.set()
    complete(server, row["id"])


def test_review_worker_uses_video_returned_by_final_authorization(server, monkeypatch):
    calls, entered, _ = fake_service(server, monkeypatch)
    values = payload(server)
    extension = server.extensions
    authorize = extension.authorize_run
    replacement = server.root / "outputs/uploads/revalidated.avi"
    replacement.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(server.root / "sample_data/synthetic.avi", replacement)
    authorizations = []

    def change_video(run_id, library_id=None):
        directory, item, video = authorize(run_id, library_id)
        authorizations.append(video)
        return directory, item, replacement if len(authorizations) >= 2 else video

    monkeypatch.setattr(extension, "authorize_run", change_video)
    status, row = request(server, "POST", "/api/llm/reviews", values)
    assert status == 202 and entered.wait(1)
    complete(server, row["id"])
    assert calls[0][0] == replacement


def _confirmed_review(server, monkeypatch):
    fake_service(server, monkeypatch)
    values = payload(server)
    status, review = request(server, "POST", "/api/llm/reviews", values)
    assert status == 202
    complete(server, review["id"])
    status, confirmed = request(
        server, "POST", f"/api/llm/reviews/{review['id']}/confirm", {"name": "耳机盒"}
    )
    assert status == 200
    return values["library_id"], review, confirmed["label"]


def test_purge_cleanup_failure_is_durable_and_retried_only_after_completed_removal(
    server, monkeypatch
):
    item_id, review, label = _confirmed_review(server, monkeypatch)
    assert request(server, "POST", f"/api/library/{item_id}/trash")[0] == 200
    status, preview = request(server, "GET", f"/api/library/{item_id}/purge-preview")
    assert status == 200
    extension = server.extensions
    monkeypatch.setattr(extension, "forget_video", lambda *_: (
        _ for _ in ()
    ).throw(OSError("locked evidence record")))
    status, _ = request(
        server, "POST", f"/api/library/{item_id}/purge",
        {"confirmation_token": preview["confirmation_token"]},
    )
    assert status == 503 and item_id not in server.library.items
    pending = json.loads(extension.cleanup_path.read_text(encoding="utf-8"))
    assert pending["items"][0]["item_id"] == item_id
    assert extension.knowledge.get(label["id"])["label"] == "耳机盒"
    assert (extension.reviews_dir / (review["id"] + ".json")).is_file()
    restarted = WorkspaceServer(server.root, port=0)
    try:
        from visualmap.workspace_extensions import WorkspaceExtensions

        recovered = WorkspaceExtensions(restarted)
        assert recovered.knowledge.list_for_run("scene01", include_revoked=True) == []
        assert review["id"] not in recovered.reviews
        assert json.loads(recovered.cleanup_path.read_text(encoding="utf-8"))["items"] == []
        assert (server.root / "sample_data/synthetic.avi").is_file()
    finally:
        restarted.server_close()


def test_pending_intent_without_completed_tombstone_never_erases_existing_labels(
    server, monkeypatch
):
    item_id, review, label = _confirmed_review(server, monkeypatch)
    extension = server.extensions
    extension.pending_purges[item_id] = {
        "item_id": item_id, "sha256": server.library.items[item_id]["sha256"],
        "requested_at": "synthetic-interrupted-intent",
    }
    extension._save_purge_cleanup()
    restarted = WorkspaceServer(server.root, port=0)
    try:
        from visualmap.workspace_extensions import WorkspaceExtensions

        recovered = WorkspaceExtensions(restarted)
        assert recovered.knowledge.get(label["id"])["label"] == "耳机盒"
        assert review["id"] in recovered.reviews
        assert item_id in recovered.pending_purges
        assert item_id in restarted.library.items
    finally:
        restarted.server_close()


def test_invalid_purge_token_leaves_no_cleanup_intent_or_missing_video(server):
    configure(server)
    item_id = payload(server)["library_id"]
    assert request(server, "POST", f"/api/library/{item_id}/trash")[0] == 200
    status, _ = request(
        server, "POST", f"/api/library/{item_id}/purge", {"confirmation_token": "invalid"}
    )
    assert status == 409
    assert item_id in server.library.items
    assert (server.root / "outputs/scene01/source_video.avi").exists()
    assert server.extensions.pending_purges == {}
    assert json.loads(server.extensions.cleanup_path.read_text(encoding="utf-8"))["items"] == []
