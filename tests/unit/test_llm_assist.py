"""Synthetic video and loopback provider exercise privacy, persistence and undo."""

from __future__ import annotations

import base64
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np
import pytest

import visualmap.llm_assist as assist
from visualmap.llm_assist import AssistError, LLMAssistService, normalize_base_url
from visualmap.object_knowledge import ObjectKnowledgeStore

TEST_KEY = "synthetic-provider-secret"


@pytest.fixture
def movie(tmp_path):
    path = tmp_path / "source.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (80, 60))
    assert writer.isOpened()
    for number in range(6):
        frame = np.zeros((60, 80, 3), dtype=np.uint8)
        frame[10:40, 15:55] = (20 + number * 25, 50, 180)
        writer.write(frame)
    writer.release()
    return path


def views():
    return [{"frame_id": frame, "xyxy": [15, 10, 55, 40]} for frame in (1, 3, 5)]


@pytest.fixture
def provider():
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_POST(self):
            length = int(self.headers["Content-Length"])
            self.server.received.append((self.path, dict(self.headers), self.rfile.read(length)))
            self.server.started.set()
            if self.server.delay:
                self.server.release.wait(2)
            self.send_response(self.server.status)
            if self.server.status == 307:
                self.send_header("Location", self.server.redirect)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            result = {
                "candidate_name": "耳机充电盒",
                "candidates": [
                    {
                        "name": "耳机充电盒",
                        "evidence": ["可见盒盖接缝"],
                        "uncertainty": "无法确认品牌",
                    }
                ],
                "evidence": ["三个视图均有相同圆角盒体"],
                "uncertainty": "局部细节不足",
                "same_object": True,
            }
            body = {"choices": [{"message": {"content": json.dumps(result, ensure_ascii=False)}}]}
            if self.server.reply is not None:
                body = self.server.reply
            try:
                self.wfile.write(json.dumps(body).encode())
            except (OSError, BrokenPipeError):
                pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    server.received = []
    server.status = 200
    server.delay = False
    server.started = threading.Event()
    server.release = threading.Event()
    server.redirect = "http://127.0.0.1:1/stolen"
    server.reply = None
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield server, f"http://127.0.0.1:{server.server_port}/v1"
    server.release.set()
    server.shutdown()
    server.server_close()


def configured(tmp_path, base):
    service = LLMAssistService(tmp_path / "assist")
    service.save_settings(
        {"base_url": base, "model": "test-vision", "api_key": TEST_KEY, "persist_api_key": False}
    )
    return service


@pytest.mark.parametrize(
    "value",
    [
        "http://example.com/v1",
        "https://user:secret@example.com/v1",
        "https://example.com/v1?api_key=secret",
        "https://example.com/v1#secret",
        "file:///C:/secret",
        "https://example.com/../v1",
        "https://example.com/%76%31",
        "https://example.com/ bad",
        "https://example.com\\@evil.com",
    ],
)
def test_endpoint_rejects_credentials_unsafe_transport_and_ambiguous_paths(value):
    with pytest.raises(AssistError):
        normalize_base_url(value)


def test_endpoint_supports_https_explicit_loopback_and_full_completion_path():
    assert normalize_base_url("https://example.com/v1/") == (
        "https://example.com/v1",
        "https://example.com/v1/chat/completions",
    )
    assert normalize_base_url("http://localhost:11434/v1")[0].startswith("http://127.0.0.1:")
    assert normalize_base_url("http://[::1]:123/v1")[1] == "http://[::1]:123/v1/chat/completions"
    assert (
        normalize_base_url("https://example.com/v1/chat/completions")[1].count("chat/completions")
        == 1
    )


def test_dpapi_failure_keeps_session_only_and_never_writes_plaintext(tmp_path, monkeypatch):
    def fail(*_, **__):
        raise OSError("DPAPI unavailable")

    monkeypatch.setattr(assist, "_dpapi", fail)
    service = LLMAssistService(tmp_path / "assist")
    public = service.save_settings(
        {"base_url": "https://example.com/v1", "model": "vision", "api_key": TEST_KEY}
    )
    assert public["api_key_present"] is True
    assert public["key_storage"] == "session"
    assert TEST_KEY not in json.dumps(public)
    assert TEST_KEY not in service.settings_path.read_text()
    assert LLMAssistService(tmp_path / "assist").public_settings()["api_key_present"] is False


def test_encrypted_persistence_roundtrip_and_clear(tmp_path, monkeypatch):
    def crypto(data, *, decrypt=False):
        return data[::-1] if decrypt else data[::-1]

    monkeypatch.setattr(assist, "_dpapi", crypto)
    service = LLMAssistService(tmp_path / "assist")
    public = service.save_settings(
        {"base_url": "https://example.com/v1", "model": "vision", "api_key": TEST_KEY}
    )
    assert public["key_storage"] == "dpapi"
    assert TEST_KEY not in service.settings_path.read_text()
    restored = LLMAssistService(tmp_path / "assist")
    assert restored.public_settings()["api_key_present"] is True
    restored.save_settings({"clear_api_key": True})
    assert restored.public_settings()["api_key_present"] is False
    assert not json.loads(restored.settings_path.read_text())["key_encrypted"]


def test_switching_provider_never_reuses_previous_secret(tmp_path):
    service = configured(tmp_path, "https://example.com/v1")
    assert (
        service.save_settings({"base_url": "https://other.example/v1"})["api_key_present"] is False
    )


def test_only_explicit_review_sends_three_decoded_crops(movie, tmp_path, provider):
    server, base = provider
    service = configured(tmp_path, base)
    service.public_settings()
    assert server.received == []
    result = service.review(movie, views(), "查看物体的盒盖")
    assert result["candidate_name"] == "耳机充电盒"
    assert result["requires_confirmation"] is True
    assert result["model_training_performed"] is False
    assert len(server.received) == 1
    path, headers, sent = server.received[0]
    assert path == "/v1/chat/completions"
    assert headers["Authorization"] == "Bearer " + TEST_KEY
    payload = json.loads(sent)
    images = [
        row["image_url"]["url"]
        for row in payload["messages"][1]["content"]
        if row["type"] == "image_url"
    ]
    assert len(images) == 3
    decoded = []
    for data in images:
        pixels = cv2.imdecode(np.frombuffer(base64.b64decode(data.split(",", 1)[1]), np.uint8), 1)
        assert pixels.shape[:2] == (30, 40)
        decoded.append(float(pixels[:, :, 0].mean()))
    assert decoded[0] < decoded[1] < decoded[2]
    assert all(row["source_width"] == 80 for row in result["selections"])
    assert str(movie) not in sent.decode()
    assert TEST_KEY not in json.dumps(result)
    assert not list((tmp_path / "assist").glob("*.jpg"))


@pytest.mark.parametrize(
    "selected",
    [
        views()[:2],
        [views()[0]] * 3,
        [{"frame_id": frame, "xyxy": [-1, 0, 50, 40]} for frame in (1, 2, 3)],
        [{"frame_id": frame, "xyxy": [0, 0, 500, 40]} for frame in (1, 2, 3)],
        [{"frame_id": frame, "xyxy": [0, 0, float("nan"), 40]} for frame in (1, 2, 3)],
        [{"frame_id": frame, "xyxy": [0, 0, 50, 40]} for frame in (1, 2, 999)],
    ],
)
def test_invalid_views_never_send_any_images(movie, tmp_path, provider, selected):
    server, base = provider
    service = configured(tmp_path, base)
    with pytest.raises(AssistError):
        service.review(movie, selected)
    assert server.received == []


def test_redirect_does_not_forward_credentials_or_retry(movie, tmp_path, provider):
    server, base = provider
    server.status = 307
    service = configured(tmp_path, base)
    with pytest.raises(AssistError) as caught:
        service.review(movie, views())
    assert caught.value.code == "redirect_blocked"
    assert len(server.received) == 1
    assert TEST_KEY not in str(caught.value)


def test_cancel_inflight_closes_wait_promptly(movie, tmp_path, provider):
    server, base = provider
    server.delay = True
    service = configured(tmp_path, base)
    cancel = threading.Event()
    outcome = []

    def request():
        try:
            service.review(movie, views(), cancel_event=cancel)
        except AssistError as error:
            outcome.append(error.code)

    thread = threading.Thread(target=request)
    thread.start()
    assert server.started.wait(2)
    started = time.monotonic()
    cancel.set()
    thread.join(1)
    assert not thread.is_alive()
    assert time.monotonic() - started < 1
    assert outcome == ["cancelled"]


def test_provider_reflecting_a_key_is_rejected_without_echo(movie, tmp_path, provider):
    server, base = provider
    server.reply = {"choices": [{"message": {"content": TEST_KEY}}]}
    service = configured(tmp_path, base)
    with pytest.raises(AssistError) as caught:
        service.review(movie, views())
    assert caught.value.code == "invalid_response"
    assert TEST_KEY not in str(caught.value)


def test_pre_cancelled_review_does_not_contact_provider(movie, tmp_path, provider):
    server, base = provider
    service = configured(tmp_path, base)
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(AssistError) as caught:
        service.review(movie, views(), cancel_event=cancel)
    assert caught.value.code == "cancelled"
    assert server.received == []


def test_total_deadline_stops_a_waiting_provider_without_retry(tmp_path, provider):
    server, base = provider
    server.delay = True
    started = time.monotonic()
    with pytest.raises(AssistError) as caught:
        assist._post_chat(
            base + "/chat/completions",
            TEST_KEY,
            {"model": "test"},
            threading.Event(),
            started + 0.2,
        )
    assert caught.value.code == "timeout"
    assert time.monotonic() - started < 1
    assert len(server.received) == 1


@pytest.mark.parametrize(
    "reply",
    [
        {},
        {"choices": []},
        {"choices": [{"message": {"content": "not json"}}]},
        {"choices": [{"message": {"content": "[]"}}]},
    ],
)
def test_invalid_provider_output_does_not_become_a_result(movie, tmp_path, provider, reply):
    server, base = provider
    server.reply = reply
    with pytest.raises(AssistError) as caught:
        configured(tmp_path, base).review(movie, views())
    assert caught.value.code == "invalid_response"


def test_dpapi_must_roundtrip_before_ciphertext_is_persisted(tmp_path, monkeypatch):
    monkeypatch.setattr(assist, "_dpapi", lambda *_args, **_kwargs: b"not-a-roundtrip")
    service = LLMAssistService(tmp_path / "assist")
    result = service.save_settings(
        {"base_url": "https://example.com/v1", "model": "vision", "api_key": TEST_KEY}
    )
    assert result["key_storage"] == "session"
    assert json.loads(service.settings_path.read_text())["key_encrypted"] == ""


def test_unknown_name_cannot_be_saved_as_confirmed(movie, tmp_path, provider):
    _, base = provider
    review = configured(tmp_path, base).review(movie, views())
    store = ObjectKnowledgeStore(tmp_path / "knowledge")
    with pytest.raises(AssistError):
        store.confirm(
            run_id="run1",
            track_id=1,
            label="未知",
            review=review,
            original_label="remote",
            source_video_id="video1",
        )
    assert store.records == []


def test_confirm_restore_revoke_and_delete_evidence_without_changing_original(
    movie, tmp_path, provider
):
    _, base = provider
    service = configured(tmp_path, base)
    review = service.review(movie, views())
    store = ObjectKnowledgeStore(tmp_path / "knowledge")
    original = {"label": "remote", "xyz": [1, 2, 3]}
    row = store.confirm(
        run_id="run1",
        track_id=7,
        label="耳机盒",
        review=review,
        original_label=original["label"],
        source_video_id="video1",
    )
    assert original == {"label": "remote", "xyz": [1, 2, 3]}
    assert row["scope"] == "this_object_in_this_video"
    restored = ObjectKnowledgeStore(tmp_path / "knowledge")
    assert restored.get(row["id"])["label"] == "耳机盒"
    assert restored.list_for_run("run1")[0]["original_label"] == "remote"
    assert restored.get(row["id"])["selections"] == review["selections"]
    assert TEST_KEY not in store.path.read_text(encoding="utf-8")
    restored.revoke(row["id"])
    assert restored.list_for_run("run1") == []
    assert len(restored.list_for_run("run1", include_revoked=True)) == 1
    assert restored.delete_for_video("video1") == 1
    assert restored.records == []
    assert original["xyz"] == [1, 2, 3]
