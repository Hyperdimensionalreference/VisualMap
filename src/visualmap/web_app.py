"""Loopback-only local video workspace, backed by the actual VisualMap CLI."""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import re
import secrets
import subprocess
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlsplit
from urllib.request import urlopen

from visualmap import __version__
from visualmap.runtime import (
    data_root,
    is_frozen,
    resource_root,
    stop_workspace,
    worker_command,
    worker_environment,
    workspace_identity,
)
from visualmap.video_library import (
    VIDEO_EXTENSIONS,
    VideoLibrary,
    display_name,
    inside_without_symlinks,
    utc_now,
)

MAX_UPLOAD_BYTES = 2 * 1024**3


class WorkspaceServer(ThreadingHTTPServer):
    """Serve packaged assets and completed runs; uploads stay in the data directory."""

    daemon_threads = True
    # Windows' SO_REUSEADDR can let two local workspace instances bind the same
    # port. A second launch must reach the existing instance instead.
    allow_reuse_address = False

    def __init__(
        self,
        root: Path,
        port: int = 8765,
        max_frames: int | None = None,
        *,
        resources: Path | None = None,
    ):
        self.root = root.resolve()
        self.resources = (resources or resource_root()).resolve()
        self.token = secrets.token_urlsafe(32)
        self.max_frames = max_frames
        self.closing = threading.Event()
        self.processes: dict[str, subprocess.Popen] = {}
        self.extensions = None
        # Claim the port before loading/recovering persisted work. A duplicate launch
        # must never mark the already-running instance's task as interrupted.
        super().__init__(("127.0.0.1", port), WorkspaceHandler)
        try:
            self.library = VideoLibrary(self.root)
            self.jobs = self.library.jobs
            self.lock = self.library.lock
        except OSError:
            super().server_close()
            raise

    @property
    def origin(self) -> str:
        return f"http://127.0.0.1:{self.server_port}"

    def page(self) -> bytes:
        assets = Path(__file__).parent / "visualization"
        html = (assets / "upload.html").read_text(encoding="utf-8")
        font = base64.b64encode((assets / "fonts/Geist-Variable.woff2").read_bytes()).decode()
        if "__STYLE__" in html:
            html = html.replace("__STYLE__", (assets / "workspace.css").read_text(encoding="utf-8"))
        if "__SCRIPT__" in html:
            html = html.replace("__SCRIPT__", (assets / "workspace.js").read_text(encoding="utf-8"))
        return html.replace("__TOKEN__", self.token).replace("__FONT__", font).encode("utf-8")

    def public_job(self, job_id: str) -> dict:
        with self.lock:
            return {
                key: value for key, value in self.jobs.get(job_id, {}).items() if key != "last_log"
            }

    def start_job(self, item_id: str, job_id: str | None = None) -> tuple[int, dict]:
        with self.lock:
            if self.closing.is_set():
                return 503, {"error": "工作区正在关闭，请稍后重新打开"}
            if item_id not in self.library.items:
                return 404, {"error": "视频记录不存在"}
            if self.library.items[item_id].get("deleted_at"):
                return 409, {"error": "请先从回收站恢复这个视频"}
            busy = self.library.active_job_id()
            if busy is not None and busy != job_id:
                return 409, {"error": "正在识别另一段视频，请等待完成", "active_job_id": busy}
            video = self.library.video_path(item_id)
            if video is None:
                return 404, {"error": "源视频已不可用，请重新上传"}
            job_id = job_id or "video_" + secrets.token_hex(8)
            self.jobs[job_id] = {
                "id": job_id,
                "library_id": item_id,
                "status": "processing",
                "stage": "正在打开识别模型",
                "frame": 0,
                "total_frames": 0,
                "created_at": utc_now(),
            }
            self.library.items[item_id]["last_job_id"] = job_id
            self.library.save(force=True)
        threading.Thread(target=self.run_job, args=(job_id, video), daemon=True).start()
        return 202, {"id": job_id}

    def server_close(self) -> None:
        self.closing.set()
        if self.extensions is not None:
            self.extensions.close()
        if hasattr(self, "library"):
            with self.lock:
                processes = list(self.processes.values())
            for process in processes:
                if process.poll() is None:
                    if sys.platform == "win32":
                        # Terminate only this workspace's own CLI process tree.
                        subprocess.run(
                            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL,
                            creationflags=subprocess.CREATE_NO_WINDOW,
                            check=False,
                        )
                    else:
                        process.terminate()
            with self.lock:
                for job in self.jobs.values():
                    if job.get("status") in {"uploading", "processing"}:
                        job.update(
                            status="interrupted",
                            stage="处理已中断",
                            source_complete=False,
                            error="工作区已关闭，任务尚未完成，请重新识别。",
                        )
                self.library.save(force=True)
        super().server_close()

    def request_shutdown(self) -> bool:
        """Reserve an idle workspace for shutdown before accepting more work."""
        with self.lock:
            if self.library.active_job_id() is not None or any(
                process.poll() is None for process in self.processes.values()
            ):
                return False
            self.closing.set()
            extension = self.extensions
        # Review creation takes extension.lock before library.lock. Keep that
        # order, and set closing first so a new review cannot pass its recheck.
        if extension is not None:
            with extension.lock:
                if any(row.get("status") == "processing" for row in extension.reviews.values()):
                    self.closing.clear()
                    return False
        return True

    def finish_job(self, job_id: str, code: int, result: Path) -> None:
        report = {}
        try:
            report = json.loads((result.parent / "run_report.json").read_text(encoding="utf-8"))
            report = report if isinstance(report, dict) else {}
        except (OSError, ValueError):
            pass
        extent = report.get("video_ingestion") or {}
        extent = extent if isinstance(extent, dict) else {}
        with self.lock:
            job = self.jobs[job_id]
            if self.closing.is_set() or job.get("status") == "interrupted":
                job.update(status="interrupted", stage="处理已中断", source_complete=False)
                self.library.save(force=True)
                return
            if code == 0 and result.is_file() and report.get("run_status") != "failed_video_decode":
                complete = extent.get("full_video_processed") is True
                partial = extent.get("completeness") == "requested_partial"
                job.update(
                    status="complete",
                    result_url=f"/results/{job_id}/semantic_map.html",
                    source_complete=complete,
                    requested_partial=partial,
                    source_completeness=extent.get("completeness", "unverified"),
                    stage="识别完成"
                    if complete
                    else "所选片段处理完成"
                    if partial
                    else "处理完成（读取范围未核实）",
                )
            else:
                if report.get("run_status") == "failed_video_decode":
                    frame = extent.get("next_frame_index", report.get("frames_read", 0))
                    error = f"视频在第 {frame} 帧读取中断，未生成完整结果。请检查或重新导出视频。"
                else:
                    error = "视频处理未成功完成，请检查视频格式或重新导出后重试。"
                job.update(status="error", stage="处理未完成", error=error, source_complete=False)
            item = self.library.items.get(job.get("library_id"))
            if item and job["status"] == "complete":
                self.library.attach_result(item, result.parent, report)
                # Source copies created by the CLI are useful if an imported source is later moved.
                for copied in result.parent.glob("source_video.*"):
                    if self.library.authorized_video(copied):
                        self.library.add_video(copied, source="result")
            self.library.save(force=True)

    def run_job(self, job_id: str, video: Path) -> None:
        command = [
            *worker_command(),
            "--config",
            str(self.resources / "configs/spatial_consistent.yaml"),
            "--resource-root",
            str(self.resources),
            "--data-dir",
            str(self.root),
            "--video",
            str(video),
            "--output",
            str(self.root / "outputs"),
            "--run-name",
            job_id,
        ]
        if self.max_frames is not None:
            command.extend(["--max-frames", str(self.max_frames)])
        creationflags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        try:
            with subprocess.Popen(
                command,
                cwd=self.root,
                env=worker_environment(),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                creationflags=creationflags,
            ) as process:
                with self.lock:
                    self.processes[job_id] = process
                    if self.closing.is_set():
                        process.terminate()
                for line in process.stdout:
                    match = re.search(r"Progress: frame (\d+) / (\d+)", line)
                    with self.lock:
                        job = self.jobs[job_id]
                        if self.closing.is_set():
                            break
                        if match:
                            job.update(
                                stage="识别物体与重建空间",
                                frame=int(match[1]),
                                total_frames=int(match[2]),
                            )
                        elif "Processed " in line:
                            job["stage"] = "整理多帧位置与空间证据"
                        elif "Wrote spatial viewer" in line:
                            job["stage"] = "正在保存三维视图"
                        job["last_log"] = line.strip()[-800:]
                        self.library.save()
                code = process.wait()
            result = self.root / "outputs" / job_id / "semantic_map.html"
            self.finish_job(job_id, code, result)
        except (OSError, subprocess.SubprocessError):
            with self.lock:
                self.jobs[job_id].update(
                    status="interrupted" if self.closing.is_set() else "error",
                    stage="处理已中断" if self.closing.is_set() else "处理未完成",
                    error="无法启动或完成本机识别，请检查运行环境后重试。",
                )
                self.library.save(force=True)
        finally:
            with self.lock:
                self.processes.pop(job_id, None)


class WorkspaceHandler(BaseHTTPRequestHandler):
    server: WorkspaceServer
    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args) -> None:
        # Video filenames and local paths are not written to terminal access logs.
        pass

    def send_data(self, code: int, data: bytes, mime: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def json(self, code: int, data: dict) -> None:
        self.send_data(
            code, json.dumps(data, ensure_ascii=False).encode(), "application/json; charset=utf-8"
        )

    def trusted_request(self, mutation: bool = False) -> bool:
        if self.headers.get("Host") != f"127.0.0.1:{self.server.server_port}":
            return False
        origin = self.headers.get("Origin")
        if origin and origin != self.server.origin:
            return False
        if self.headers.get("Sec-Fetch-Site") == "cross-site":
            return False
        return not mutation or secrets.compare_digest(
            self.headers.get("X-VisualMap-Token", ""), self.server.token
        )

    def reject_upload(self, code: int, message: str) -> None:
        # Consume small rejected bodies so Windows can deliver the JSON response
        # before closing, rather than resetting a socket with unread input.
        try:
            size = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            size = 0
        if 0 < size <= 65536:
            self.connection.settimeout(2)
            try:
                self.rfile.read(size)
            except OSError:
                pass
        self.json(code, {"error": message})
        self.close_connection = True

    def send_file(self, file: Path) -> None:
        """Serve a previously authorized file, including normal and suffix byte ranges."""
        try:
            size = file.stat().st_size
            start, end, status = 0, size - 1, 200
            requested = self.headers.get("Range", "")
            if requested:
                match = re.fullmatch(r"bytes=(\d*)-(\d*)", requested)
                valid = match is not None and bool(match[1] or match[2]) and size > 0
                if valid and match[1]:
                    start = int(match[1])
                    end = min(size - 1, int(match[2])) if match[2] else size - 1
                    valid = 0 <= start <= end < size
                elif valid:
                    suffix = int(match[2])
                    valid = suffix > 0
                    start = max(0, size - suffix)
                if not valid:
                    self.send_response(416)
                    self.send_header("Content-Range", f"bytes */{size}")
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                status = 206
            with file.open("rb") as stream:
                self.send_response(status)
                self.send_header(
                    "Content-Type", mimetypes.guess_type(file)[0] or "application/octet-stream"
                )
                self.send_header("Content-Length", str(max(0, end - start + 1)))
                self.send_header("Accept-Ranges", "bytes")
                self.send_header("X-Content-Type-Options", "nosniff")
                if status == 206:
                    self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
                self.end_headers()
                stream.seek(start)
                remaining = max(0, end - start + 1)
                while remaining:
                    chunk = stream.read(min(remaining, 1024 * 1024))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except OSError:
            self.close_connection = True

    def do_GET(self) -> None:
        if not self.trusted_request():
            self.json(403, {"error": "仅支持本机工作区访问"})
            return
        route = unquote(urlsplit(self.path).path)
        from visualmap.workspace_extensions import handle_extension

        if handle_extension(self, route, parse_qs(urlsplit(self.path).query)):
            return
        if route == "/":
            self.send_data(200, self.server.page(), "text/html; charset=utf-8")
        elif route == "/assets/viewer_motion.js":
            file = Path(__file__).parent / "visualization/viewer_motion.js"
            self.send_data(200, file.read_bytes(), "text/javascript; charset=utf-8")
        elif route == "/api/info":
            info = {"application": "VisualMap", "protocol": 1}
            if parse_qs(urlsplit(self.path).query).get("runtime") == ["1"]:
                info.update(
                    version=__version__,
                    workspace=workspace_identity(self.server.root, self.server.resources),
                )
            self.json(200, info)
        elif route == "/api/library":
            self.json(200, self.server.library.listing())
        elif route.startswith("/api/jobs/"):
            job = self.server.public_job(route.rsplit("/", 1)[-1])
            self.json(200 if job else 404, job or {"error": "记录不存在"})
        elif route.startswith("/media/"):
            item_id = route[len("/media/") :]
            with self.server.lock:
                file = self.server.library.video_path(item_id)
            if file is None:
                self.json(404, {"error": "源视频不可用"})
            else:
                self.send_file(file)
        elif route.startswith("/thumbnails/") and route.endswith(".jpg"):
            item_id = route[len("/thumbnails/") : -4]
            with self.server.lock:
                file = self.server.library.thumbnail_path(item_id)
            if file is None:
                self.json(404, {"error": "视频封面不可用"})
            else:
                self.send_file(file)
        elif route.startswith("/results/"):
            output = (self.server.root / "outputs").resolve()
            file = (output / route[len("/results/") :]).resolve()
            # No traversal, directory listings, executable code, or arbitrary local files.
            if (
                not file.is_relative_to(output)
                or not inside_without_symlinks(output / route[len("/results/") :], output)
                or route[len("/results/") :].split("/", 1)[0] in {"library", "uploads"}
                or not file.is_file()
                or file.suffix.lower()
                not in {".html", ".mp4", ".mov", ".webm", ".json", ".png", ".jpg"}
            ):
                self.json(404, {"error": "结果不存在"})
                return
            self.send_file(file)
        else:
            self.json(404, {"error": "页面不存在"})

    def do_POST(self) -> None:
        if not self.trusted_request(mutation=True):
            self.reject_upload(403, "请从本机工作区提交视频")
            return
        route = urlsplit(self.path).path
        if route == "/api/shutdown":
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                length = -1
            if not 0 <= length <= 65536:
                self.reject_upload(400, "退出请求格式不正确")
                return
            if length:
                self.rfile.read(length)
            if not self.server.request_shutdown():
                self.json(409, {"error": "工作区正在识别或复核画面，请任务完成后再退出。"})
                return
            self.json(202, {"status": "closing"})
            threading.Thread(target=self.server.shutdown, daemon=True).start()
            return
        if self.server.closing.is_set():
            self.reject_upload(503, "工作区正在关闭，请稍后重新打开。")
            return
        from visualmap.workspace_extensions import handle_extension

        if handle_extension(self, route, parse_qs(urlsplit(self.path).query), mutation=True):
            return
        match = re.fullmatch(r"/api/library/([a-zA-Z0-9_-]{1,100})/(process|trash|restore)", route)
        if match:
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                length = -1
            if not 0 <= length <= 65536:
                self.reject_upload(400, "操作请求格式不正确")
                return
            try:
                body = json.loads(self.rfile.read(length)) if length else {}
                if not isinstance(body, dict):
                    raise ValueError
            except (ValueError, UnicodeDecodeError):
                self.json(400, {"error": "操作请求格式不正确"})
                return
            if match[2] == "process":
                code, response = self.server.start_job(match[1])
            else:
                try:
                    item = self.server.library.set_trashed(match[1], match[2] == "trash")
                    code, response = 200, {"item": item}
                except KeyError:
                    code, response = 404, {"error": "视频记录不存在"}
                except ValueError as error:
                    code, response = 409, {"error": str(error)}
                except OSError:
                    code, response = 503, {"error": "视频库保存未完成，请重试"}
            self.json(code, response)
            return
        if route not in {"/api/jobs", "/api/library"}:
            self.reject_upload(404, "未知操作")
            return
        query = parse_qs(urlsplit(self.path).query)
        name = query.get("filename", [""])[0]
        suffix = Path(display_name(name)).suffix.lower()
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = 0
        if suffix not in VIDEO_EXTENSIONS or not 0 < length <= MAX_UPLOAD_BYTES:
            self.reject_upload(400, "请选择 MP4、MOV、AVI、MKV 或 WebM 视频，最大 2 GB")
            return
        job_id = "video_" + secrets.token_hex(8)
        legacy = route == "/api/jobs"
        if legacy:
            with self.server.lock:
                if self.server.library.active_job_id() is not None:
                    self.reject_upload(409, "正在处理视频，请等待完成")
                    return
                self.server.jobs[job_id] = {
                    "id": job_id,
                    "status": "uploading",
                    "stage": "读取视频",
                    "frame": 0,
                    "total_frames": 0,
                    "created_at": utc_now(),
                }
                self.server.library.save(force=True)
        uploads = self.server.root / "outputs/uploads"
        if not inside_without_symlinks(uploads, self.server.root):
            self.reject_upload(400, "视频保存目录不可用")
            return
        uploads.mkdir(parents=True, exist_ok=True)
        video = uploads / (job_id + suffix)
        temporary = uploads / (job_id + ".part")
        remaining = length
        self.connection.settimeout(90)
        try:
            with temporary.open("xb") as stream:
                while remaining:
                    chunk = self.rfile.read(min(remaining, 1024 * 1024))
                    if not chunk:
                        raise OSError("视频传输中断")
                    stream.write(chunk)
                    remaining -= len(chunk)
            temporary.replace(video)
            item = self.server.library.add_video(
                video, name=name, source="upload", restore_deleted=True
            )
        except (OSError, TimeoutError, ValueError):
            temporary.unlink(missing_ok=True)
            if legacy:
                with self.server.lock:
                    self.server.jobs[job_id].update(
                        status="error", error="视频传输未完成，请重试。"
                    )
                    self.server.library.save(force=True)
            self.json(400, {"error": "视频传输未完成，请重新选择"})
            self.close_connection = True
            return
        if legacy:
            code, response = self.server.start_job(item["id"], job_id=job_id)
            self.json(code, response)
        else:
            with self.server.lock:
                public_item = self.server.library.public_item(item)
            self.json(201, {"item": public_item})


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="VisualMap 本机视频工作区")
    parser.add_argument("--version", action="version", version=f"VisualMap {__version__}")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-open", action="store_true")
    parser.add_argument("--stop", action="store_true", help="仅在空闲时安全退出本工作区")
    parser.add_argument("--max-frames", type=int, help="仅供短视频验证")
    parser.add_argument("--data-dir", type=Path, help="视频、结果与本机设置的保存目录")
    args = parser.parse_args(argv)
    if not 0 <= args.port <= 65535:
        parser.error("--port 必须在 0 到 65535 之间")
    if args.max_frames is not None and args.max_frames < 1:
        parser.error("--max-frames 必须为正整数")
    root = data_root(args.data_dir)
    if args.stop:
        raise SystemExit(stop_workspace(root, args.port))
    try:
        server = WorkspaceServer(root, args.port, args.max_frames)
    except OSError as exc:
        origin = f"http://127.0.0.1:{args.port}"
        info_path = "/api/info"
        expected = {"application": "VisualMap", "protocol": 1}
        if is_frozen() or args.data_dir is not None:
            info_path += "?runtime=1"
            expected.update(
                version=__version__, workspace=workspace_identity(root, resource_root())
            )
        try:
            with urlopen(origin + info_path, timeout=2) as response:
                existing = json.loads(response.read(1024))
        except (OSError, ValueError):
            raise SystemExit(f"工作区端口不可用：{args.port}") from exc
        if existing != expected:
            raise SystemExit(f"工作区端口不可用：{args.port}") from exc
        print(f"VisualMap 工作区已运行：{origin}", flush=True)
        if not args.no_open:
            webbrowser.open(origin)
        return
    with server:
        print(f"VisualMap 工作区：{server.origin}", flush=True)
        if not args.no_open:
            webbrowser.open(server.origin)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
