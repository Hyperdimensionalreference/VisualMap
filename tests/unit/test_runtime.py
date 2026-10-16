"""Installed builds separate immutable assets, private data and worker processes."""

from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
from http.client import HTTPConnection
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import yaml

from visualmap import runtime
from visualmap.__main__ import main as desktop_main
from visualmap.web_app import WorkspaceServer
from visualmap.web_app import main as workspace_main


def test_frozen_defaults_never_use_install_directory(tmp_path, monkeypatch):
    bundle = tmp_path / "Program Files/VisualMap/_internal"
    local = tmp_path / "User/Local"
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "_MEIPASS", str(bundle), raising=False)
    monkeypatch.setenv("LOCALAPPDATA", str(local))
    assert runtime.resource_root() == bundle.resolve()
    assert runtime.data_root() == (local / "VisualMap").resolve()
    assert not local.exists()
    assert not bundle.exists()


def test_source_default_and_explicit_data_root(tmp_path):
    assert runtime.data_root() == runtime.resource_root()
    directory = tmp_path / "empty workspace"
    assert runtime.data_root(directory) == directory.resolve()
    assert not directory.exists()


def test_frozen_worker_restarts_same_exe_without_python(monkeypatch):
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", "C:/Installed App/VisualMap.exe")
    command = runtime.worker_command()
    assert command == [str(Path(sys.executable)), "--worker"]
    assert "-m" not in command


def test_resource_paths_resolve_nested_models_and_leave_user_paths(tmp_path):
    config = tmp_path / "settings.yaml"
    absolute_model = tmp_path / "external.onnx"
    config.write_text(
        yaml.safe_dump(
            {
                "video": {"path": "private/source.mp4"},
                "output": {"dir": "private/results"},
                "detection": {
                    "model_path": str(absolute_model),
                    "open_vocabulary": {
                        "model_path": "models/grounding.onnx",
                        "prompts_path": "configs/prompts.json",
                        "runtime_path": None,
                        "crop_verifier": {"vocabulary_path": "models/vocabulary.json"},
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    resources = tmp_path / "read-only-assets"
    result = runtime.resource_path_overrides(config, resources)
    assert result["detection.model_path"] == str(absolute_model)
    assert result["detection.open_vocabulary.model_path"] == str(
        (resources / "models/grounding.onnx").resolve()
    )
    assert result["detection.open_vocabulary.crop_verifier.vocabulary_path"] == str(
        (resources / "models/vocabulary.json").resolve()
    )
    assert "detection.open_vocabulary.runtime_path" not in result
    assert "video.path" not in result
    assert "output.dir" not in result
    assert not resources.exists()


def test_workspace_identity_separates_installations_and_data(tmp_path):
    assert runtime.workspace_identity(tmp_path / "one", tmp_path / "assets") != (
        runtime.workspace_identity(tmp_path / "two", tmp_path / "assets")
    )
    assert runtime.workspace_identity(tmp_path / "one", tmp_path / "assets") != (
        runtime.workspace_identity(tmp_path / "one", tmp_path / "other-assets")
    )


def test_worker_dispatch_preserves_cli_return_code(monkeypatch):
    seen = []
    monkeypatch.setattr("visualmap.app.main", lambda arguments: seen.append(arguments) or 7)
    assert desktop_main(["--worker", "--video", "user.mp4"]) == 7
    assert seen == [["--video", "user.mp4"]]


@pytest.mark.skipif(sys.platform != "win32", reason="Windows GUI stream ownership")
def test_windowed_worker_restores_shared_stdout_and_stderr_pipe(tmp_path):
    executable = Path(sys.executable).with_name("pythonw.exe")
    if not executable.is_file():
        pytest.skip("pythonw unavailable")
    code = (
        "import sys; from pathlib import Path; "
        "from visualmap.runtime import configure_windowed_streams; "
        "configure_windowed_streams(Path(sys.argv[1]), worker=True); "
        "print('stdout-ok', flush=True); print('stderr-ok', file=sys.stderr, flush=True)"
    )
    result = subprocess.run(
        [str(executable), "-c", code, str(tmp_path / "private-data")],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=runtime.worker_environment(),
        encoding="utf-8",
        errors="replace",
        timeout=15,
        check=True,
    )
    assert "stdout-ok" in result.stdout
    assert "stderr-ok" in result.stdout
    assert "Exception ignored" not in result.stdout
    assert not (tmp_path / "private-data").exists()


def test_self_check_json_does_not_open_or_create_data_directory(tmp_path, monkeypatch):
    directory = tmp_path / "private-data"
    monkeypatch.setattr(
        "visualmap.__main__.self_check",
        lambda root: {"ok": True, "data_root": str(root), "version": "test"},
    )
    report = tmp_path / "report.json"
    assert desktop_main(["--self-check", str(report), "--data-dir", str(directory)]) == 0
    assert json.loads(report.read_text(encoding="utf-8"))["data_root"] == str(directory)
    assert not directory.exists()


def test_read_only_windowed_commands_do_not_create_logs(tmp_path, monkeypatch):
    directory = tmp_path / "private-data"
    with monkeypatch.context() as patch:
        patch.setattr(sys, "stdout", None)
        patch.setattr(sys, "stderr", None)
        patch.setattr(runtime, "_inherited_stream", lambda _: None)
        runtime.configure_windowed_streams(directory, worker=True, write_log=False)
        stream = sys.stdout
    stream.close()
    assert not directory.exists()


@pytest.mark.parametrize("arguments", [["--port", "65536"], ["--max-frames", "0"]])
def test_invalid_workspace_parameters_do_not_create_data(tmp_path, arguments):
    data = tmp_path / "data"
    with pytest.raises(SystemExit) as error:
        workspace_main([*arguments, "--data-dir", str(data), "--no-open"])
    assert error.value.code == 2
    assert not data.exists()


@pytest.fixture
def running_workspace(tmp_path):
    server = WorkspaceServer(tmp_path / "private data", port=0)

    def serve():
        with server:
            server.serve_forever(poll_interval=0.05)

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        if thread.is_alive():
            server.shutdown()
            thread.join(timeout=3)
        server.server_close()


def post_shutdown(server, headers=None):
    connection = HTTPConnection("127.0.0.1", server.server_port, timeout=3)
    connection.request("POST", "/api/shutdown", body=b"{}", headers=headers or {})
    response = connection.getresponse()
    result = response.status, response.read()
    connection.close()
    return result


@pytest.mark.parametrize("headers", [{}, {"X-VisualMap-Token": "wrong"}])
def test_shutdown_requires_session_authorization(running_workspace, headers):
    server = running_workspace
    assert post_shutdown(server, headers)[0] == 403
    assert not server.closing.is_set()


def test_shutdown_requires_local_origin(running_workspace):
    server = running_workspace
    headers = {"X-VisualMap-Token": server.token, "Origin": "https://external.example"}
    assert post_shutdown(server, headers)[0] == 403
    assert not server.closing.is_set()


def test_shutdown_refuses_active_recognition(running_workspace):
    server = running_workspace
    with server.lock:
        server.jobs["active"] = {"status": "processing"}
    assert post_shutdown(server, {"X-VisualMap-Token": server.token})[0] == 409
    assert not server.closing.is_set()


def test_shutdown_refuses_active_review(running_workspace):
    server = running_workspace
    server.extensions = SimpleNamespace(
        lock=threading.RLock(), reviews={"active": {"status": "processing"}}, close=lambda: None
    )
    assert runtime.stop_workspace(server.root, server.server_port) == 4
    assert not server.closing.is_set()


def test_stop_leaves_another_data_workspace_running(running_workspace, tmp_path):
    server = running_workspace
    assert runtime.stop_workspace(tmp_path / "other user data", server.server_port) == 3
    assert not server.closing.is_set()


def test_stop_closes_own_idle_workspace_and_is_idempotent(running_workspace):
    server = running_workspace
    assert runtime.stop_workspace(server.root, server.server_port) == 0
    assert server.closing.is_set()
    assert runtime.stop_workspace(server.root, server.server_port) == 0


def test_separate_data_workspace_runs_real_worker_from_unrelated_cwd(tmp_path):
    resources = tmp_path / "installed resources"
    configs = resources / "configs"
    configs.mkdir(parents=True)
    (configs / "spatial_consistent.yaml").write_text(
        yaml.safe_dump(
            {
                "video": {"path": "private_video_must_not_be_used.mp4"},
                "depth": {"enabled": False},
                "detection": {"enabled": True, "detector": "demo"},
                "output": {"dir": "install_output_must_not_be_used", "copy_source_video": True},
            }
        ),
        encoding="utf-8",
    )
    data = tmp_path / "user data"
    uploads = data / "outputs/uploads"
    uploads.mkdir(parents=True)
    video = uploads / "selected.avi"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 10, (64, 48))
    if not writer.isOpened():
        pytest.skip("VideoWriter unavailable")
    for index in range(5):
        writer.write(np.full((48, 64, 3), index * 20, dtype=np.uint8))
    writer.release()
    server = WorkspaceServer(data, port=0, max_frames=3, resources=resources)
    try:
        item = server.library.add_video(video, source="upload")
        code, job = server.start_job(item["id"])
        assert code == 202
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            current = server.public_job(job["id"])
            if current["status"] not in {"uploading", "processing"}:
                break
            time.sleep(0.05)
        assert current["status"] == "complete", current
        run = data / "outputs" / job["id"]
        assert (run / "semantic_map.html").is_file()
        assert (run / "run_report.json").is_file()
        assert not (resources / "outputs").exists()
        assert not (data / "install_output_must_not_be_used").exists()
        assert not (resources / "work").exists()
    finally:
        server.server_close()
