"""Persistent, offline video catalogue; paths in its index never authorize arbitrary files."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import secrets
import stat
import threading
import time
from datetime import datetime, timezone
from pathlib import Path, PureWindowsPath

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
_ID = re.compile(r"[a-zA-Z0-9_-]{1,100}\Z")
_SOURCE_LABELS = {
    "sample": "项目示例",
    "project": "项目视频",
    "upload": "上传视频",
    "result": "历史结果",
}
_STATUS_LABELS = {
    "ready": "待识别",
    "processing": "识别中",
    "complete": "已有结果",
    "error": "处理失败",
    "interrupted": "处理已中断",
    "unavailable": "源视频不可用",
    "trashed": "已移到回收站",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def display_name(value: str) -> str:
    """Keep only a display basename, including Windows names on non-Windows test hosts."""
    name = PureWindowsPath(str(value).replace("/", "\\")).name
    name = "".join(ch for ch in name if ch >= " " and ch != "\x7f").strip()
    return name[:240] or "未命名视频"


def real_run_name(name: str) -> bool:
    """Scan direct result runs only; never walk unit-test and verification workspaces."""
    lowered = name.lower()
    return (
        bool(_ID.fullmatch(name))
        and lowered not in {"uploads", "library"}
        and not any(
            word in lowered
            for word in ("test", "temp", "verification", "validation", "benchmark", "review")
        )
    )


def inside_without_symlinks(path: Path, base: Path) -> bool:
    """Reject symlink/junction escape, including a replaced parent directory."""
    try:
        relative = path.absolute().relative_to(base.absolute())
        if not path.resolve().is_relative_to(base.resolve()):
            return False
        current = base
        if current.is_symlink() or (
            current.exists()
            and getattr(current.lstat(), "st_file_attributes", 0)
            & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
        ):
            return False
        for part in relative.parts:
            current /= part
            if part in {".", ".."} or current.is_symlink() or (
                current.exists()
                and getattr(current.lstat(), "st_file_attributes", 0)
                & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
            ):
                return False
        return True
    except (OSError, ValueError, RuntimeError):
        return False


class VideoLibrary:
    """Catalogue content once, associate all same-content runs, and persist job state."""

    def __init__(self, root: Path):
        self.root = root.resolve()
        self.storage = self.root / "outputs/library"
        if not inside_without_symlinks(self.storage, self.root):
            raise OSError("视频库目录不可用")
        self.storage.mkdir(parents=True, exist_ok=True)
        self.thumbnails = self.storage / "thumbnails"
        if not inside_without_symlinks(self.thumbnails, self.root):
            raise OSError("视频封面目录不可用")
        self.thumbnails.mkdir(exist_ok=True)
        self.index_path = self.storage / "index.json"
        self.lock = threading.RLock()
        self.items: dict[str, dict] = {}
        self.jobs: dict[str, dict] = {}
        self.tombstones: dict[str, dict] = {}
        self._purge_manager = None
        self._fingerprints: dict[str, dict] = {}
        self._last_scan = 0.0
        self._last_save = 0.0
        self._load()
        # A dead process cannot finish a task. Existing files alone do not override this.
        for job in self.jobs.values():
            if job.get("status") in {"uploading", "processing"}:
                job.update(
                    status="interrupted",
                    stage="上次处理已中断",
                    source_complete=False,
                    error="工作区上次关闭时任务尚未完成，请重新识别。",
                )
        self.refresh(force=True)
        self.save(force=True)

    def _load(self) -> None:
        if not self.index_path.is_file() or not inside_without_symlinks(self.index_path, self.root):
            return
        try:
            data = json.loads(self.index_path.read_text(encoding="utf-8"))
            if data.get("schema_version") != 1:
                return
            for item in data.get("items", []):
                if (
                    isinstance(item, dict)
                    and _ID.fullmatch(str(item.get("id", "")))
                    and re.fullmatch(r"[a-f0-9]{64}", str(item.get("sha256", "")))
                ):
                    item["name"] = display_name(item.get("name", ""))
                    item["sources"] = [
                        row
                        for row in item.get("sources", [])
                        if isinstance(row, dict) and isinstance(row.get("path"), str)
                    ]
                    item["results"] = [
                        row
                        for row in item.get("results", [])
                        if isinstance(row, dict) and real_run_name(str(row.get("id", "")))
                    ]
                    self.items[item["id"]] = item
            for key, job in data.get("jobs", {}).items():
                if _ID.fullmatch(key) and isinstance(job, dict):
                    self.jobs[key] = job
            fingerprints = data.get("fingerprints", {})
            self._fingerprints = {
                key: value
                for key, value in fingerprints.items()
                if isinstance(key, str) and isinstance(value, dict)
            }
            self.tombstones = {
                key: value
                for key, value in data.get("tombstones", {}).items()
                if re.fullmatch(r"[a-f0-9]{64}", str(key)) and isinstance(value, dict)
            }
        except (OSError, ValueError, TypeError, AttributeError):
            # Preserve a malformed catalogue for recovery; source media is never modified.
            try:
                backup = self.storage / f"index-recovery-{secrets.token_hex(4)}.json"
                backup.write_bytes(self.index_path.read_bytes())
            except OSError:
                pass

    def save(self, force: bool = False) -> None:
        with self.lock:
            if not force and time.monotonic() - self._last_save < 1:
                return
            if not inside_without_symlinks(self.index_path, self.root):
                raise OSError("视频库索引目录不可用")
            temporary = self.storage / f"index-{secrets.token_hex(5)}.tmp"
            data = {
                "schema_version": 1,
                "items": list(self.items.values()),
                "jobs": self.jobs,
                "fingerprints": self._fingerprints,
                "tombstones": self.tombstones,
            }
            try:
                with temporary.open("x", encoding="utf-8") as stream:
                    json.dump(data, stream, ensure_ascii=False, allow_nan=False)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary, self.index_path)
            finally:
                temporary.unlink(missing_ok=True)
            self._last_save = time.monotonic()

    def authorized_video(self, path: Path) -> bool:
        if (
            not inside_without_symlinks(path, self.root)
            or path.suffix.lower() not in VIDEO_EXTENSIONS
        ):
            return False
        try:
            relative = path.absolute().relative_to(self.root)
        except ValueError:
            return False
        parts = relative.parts
        approved = (
            len(parts) == 1
            or (len(parts) == 2 and parts[0] == "sample_data")
            or (len(parts) == 3 and parts[:2] == ("outputs", "uploads"))
            or (
                len(parts) == 3
                and parts[0] == "outputs"
                and real_run_name(parts[1])
                and path.stem == "source_video"
                and (path.parent / "semantic_map.html").is_file()
            )
        )
        return approved and path.is_file()

    def _relative(self, path: Path) -> str:
        return path.absolute().relative_to(self.root).as_posix()

    def _fingerprint(self, path: Path) -> str:
        stat = path.stat()
        relative = self._relative(path)
        cached = self._fingerprints.get(relative, {})
        stamp = [stat.st_size, stat.st_mtime_ns]
        if cached.get("stamp") == stamp and re.fullmatch(
            r"[a-f0-9]{64}", str(cached.get("sha256", ""))
        ):
            return cached["sha256"]
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
        value = digest.hexdigest()
        self._fingerprints[relative] = {"stamp": stamp, "sha256": value}
        return value

    def _probe(self, path: Path, item: dict) -> None:
        """Read actual local decoder metadata and one frame; use no remote services."""
        import cv2

        item.update(duration_seconds=None, width=None, height=None, thumbnail=False)
        capture = cv2.VideoCapture(str(path))
        try:
            if not capture.isOpened():
                return
            fps = capture.get(cv2.CAP_PROP_FPS)
            count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
            fps = fps if math.isfinite(fps) and fps > 0 else 0
            if math.isfinite(count) and math.isfinite(fps) and count > 0 and fps > 0:
                item["duration_seconds"] = round(count / fps, 3)
            if count > 1 and math.isfinite(count):
                capture.set(
                    cv2.CAP_PROP_POS_FRAMES, min(int(count * 0.1), int(fps * 5) if fps > 0 else 0)
                )
            ok, frame = capture.read()
            if not ok:
                capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame = capture.read()
            if not ok or frame is None:
                return
            height, width = frame.shape[:2]
            item.update(width=width, height=height)
            ratio = min(1.0, 480 / max(width, height))
            frame = cv2.resize(frame, (max(1, round(width * ratio)), max(1, round(height * ratio))))
            ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 82])
            if ok:
                target = self.thumbnails / (item["id"] + ".jpg")
                if not inside_without_symlinks(target, self.root):
                    return
                temporary = target.with_suffix(".tmp")
                temporary.write_bytes(encoded.tobytes())
                os.replace(temporary, target)
                item["thumbnail"] = True
        except cv2.error:
            item.update(duration_seconds=None, width=None, height=None, thumbnail=False)
        finally:
            capture.release()

    def add_video(
        self,
        path: Path,
        name: str | None = None,
        source: str = "upload",
        *,
        restore_deleted: bool = False,
    ) -> dict:
        with self.lock:
            if not self.authorized_video(path):
                raise ValueError("视频文件不在授权视频库目录内")
            digest = self._fingerprint(path)
            if digest in self.tombstones:
                if not restore_deleted:
                    raise ValueError("这个视频已彻底移除，仅明确重新导入才能加入视频库")
                self.tombstones.pop(digest, None)
            item = next((it for it in self.items.values() if it.get("sha256") == digest), None)
            if item is None:
                identity = "clip_" + digest[:24]
                item = {
                    "id": identity,
                    "sha256": digest,
                    "name": display_name(name or path.name),
                    "source": source,
                    "size_bytes": path.stat().st_size,
                    "created_at": datetime.fromtimestamp(
                        path.stat().st_mtime, timezone.utc
                    ).isoformat(timespec="seconds"),
                    "sources": [],
                    "results": [],
                    "last_job_id": None,
                }
                self.items[identity] = item
                self._probe(path, item)
            elif name and item.get("source") == "result" and source != "result":
                item.update(name=display_name(name), source=source)
            relative = self._relative(path)
            if not any(s.get("path") == relative for s in item["sources"]):
                item["sources"].append({"path": relative, "source": source})
            # Discovery must not undo a user's removal. Only an explicit new import does.
            if restore_deleted:
                item.pop("deleted_at", None)
            self.save(force=True)
            return item

    def set_trashed(self, item_id: str, trashed: bool) -> dict:
        """Persist a recoverable catalogue removal; never move or unlink media/results."""
        with self.lock:
            item = self.items.get(item_id)
            if item is None:
                raise KeyError(item_id)
            if trashed and any(
                job.get("library_id") == item_id
                and job.get("status") in {"uploading", "processing"}
                for job in self.jobs.values()
            ):
                raise ValueError("这个视频正在识别，完成后才能移到回收站")
            previous = item.get("deleted_at")
            if trashed:
                item["deleted_at"] = previous or utc_now()
            else:
                item.pop("deleted_at", None)
            try:
                self.save(force=True)
            except OSError:
                if previous is None:
                    item.pop("deleted_at", None)
                else:
                    item["deleted_at"] = previous
                raise
            return self.public_item(item)

    def video_path(self, item_id: str) -> Path | None:
        item = self.items.get(item_id)
        if not item or item.get("deleted_at"):
            return None
        for source in item.get("sources", []):
            relative = source.get("path", "")
            if not isinstance(relative, str):
                continue
            path = self.root / relative
            if self.authorized_video(path):
                try:
                    if self._fingerprint(path) == item.get("sha256"):
                        return path
                except OSError:
                    continue
        return None

    def thumbnail_path(self, item_id: str) -> Path | None:
        if (
            item_id not in self.items
            or self.items[item_id].get("deleted_at")
            or not _ID.fullmatch(item_id)
        ):
            return None
        path = self.thumbnails / (item_id + ".jpg")
        return path if inside_without_symlinks(path, self.root) and path.is_file() else None

    def _report(self, directory: Path) -> dict:
        path = directory / "run_report.json"
        if not inside_without_symlinks(path, self.root):
            return {}
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
            return report if isinstance(report, dict) else {}
        except (OSError, ValueError):
            return {}

    def attach_result(self, item: dict, directory: Path, report: dict | None = None) -> None:
        if not real_run_name(directory.name) or not inside_without_symlinks(directory, self.root):
            return
        viewer = directory / "semantic_map.html"
        if not viewer.is_file() or not inside_without_symlinks(viewer, self.root):
            return
        report = self._report(directory) if report is None else report
        if report.get("run_status") == "failed_video_decode":
            return
        extent = report.get("video_ingestion") or {}
        extent = extent if isinstance(extent, dict) else {}
        title = "灰色占用体积视图" if directory.name.startswith("occupancy_") else "三维识别结果"
        summary = report.get("summary") or {}
        summary = summary if isinstance(summary, dict) else {}
        processed = report.get("frames_read", summary.get("frames_processed"))
        result = {
            "id": directory.name,
            "title": title,
            "created_at": datetime.fromtimestamp(viewer.stat().st_mtime, timezone.utc).isoformat(
                timespec="seconds"
            ),
            "result_url": f"/results/{directory.name}/semantic_map.html",
            "processed_frames": processed
            if isinstance(processed, int) and processed >= 0
            else None,
            "source_complete": extent.get("full_video_processed") is True,
            "requested_partial": extent.get("completeness") == "requested_partial",
            "tracked_frames": summary.get("frames_tracked")
            if isinstance(summary.get("frames_tracked"), int)
            else None,
            "submap_count": summary.get("submap_count")
            if isinstance(summary.get("submap_count"), int)
            else None,
        }
        results = item.setdefault("results", [])
        results[:] = [row for row in results if row.get("id") != directory.name]
        results.append(result)
        results.sort(key=lambda row: (row.get("created_at", ""), row.get("id", "")), reverse=True)

    def refresh(self, force: bool = False) -> None:
        with self.lock:
            if not force and time.monotonic() - self._last_scan < 5:
                return
            self._last_scan = time.monotonic()
            candidates = [
                (self.root, "project"),
                (self.root / "sample_data", "sample"),
                (self.root / "outputs/uploads", "upload"),
            ]
            for base, label in candidates:
                if not base.is_dir() or not inside_without_symlinks(base, self.root):
                    continue
                for path in sorted(base.iterdir()):
                    if self.authorized_video(path):
                        try:
                            self.add_video(path, source=label)
                        except (OSError, ValueError):
                            continue
            output = self.root / "outputs"
            if output.is_dir() and inside_without_symlinks(output, self.root):
                for directory in sorted(output.iterdir()):
                    if not directory.is_dir() or not real_run_name(directory.name):
                        continue
                    # Do not reinterpret an interrupted/failed live run as a finished run.
                    known_job = self.jobs.get(directory.name)
                    if known_job and known_job.get("status") != "complete":
                        continue
                    if not (directory / "semantic_map.html").is_file():
                        continue
                    videos = [
                        p for p in directory.glob("source_video.*") if self.authorized_video(p)
                    ]
                    if not videos:
                        continue
                    report = self._report(directory)
                    if report.get("run_status") == "failed_video_decode":
                        continue
                    try:
                        metadata = report.get("video")
                        metadata = metadata if isinstance(metadata, dict) else {}
                        original_name = display_name(metadata.get("path", videos[0].name))
                        item = self.add_video(videos[0], original_name, "result")
                        self.attach_result(item, directory, report)
                    except (OSError, ValueError):
                        continue
            self.save(force=True)

    def active_job_id(self) -> str | None:
        return next(
            (
                key
                for key, job in self.jobs.items()
                if job.get("status") in {"processing", "uploading"}
            ),
            None,
        )

    def plan_purge(self, item_id: str) -> dict:
        """Preview the exact managed files; only a later explicit token can remove them."""
        if self._purge_manager is None:
            from visualmap.storage_manager import PurgeManager

            self._purge_manager = PurgeManager(self)
        return self._purge_manager.plan(item_id)

    def purge(self, item_id: str, confirmation_token: str) -> dict:
        if self._purge_manager is None:
            raise ValueError("请先预览彻底删除内容并确认")
        return self._purge_manager.purge(item_id, confirmation_token)

    def public_item(self, item: dict) -> dict:
        job_id = item.get("last_job_id")
        job = self.jobs.get(job_id, {})
        available = self.video_path(item["id"]) is not None
        if item.get("deleted_at"):
            status = "trashed"
        elif not available:
            status = "unavailable"
        elif job.get("status") in {"processing", "uploading"}:
            status = "processing"
        elif job.get("status") in {"error", "interrupted"}:
            status = job["status"]
        else:
            status = "complete" if item.get("results") else "ready"
        source = item.get("source", "upload")
        result = {
            key: item.get(key)
            for key in (
                "id",
                "name",
                "size_bytes",
                "duration_seconds",
                "width",
                "height",
                "created_at",
                "deleted_at",
            )
        }
        result.update(
            source=source,
            source_label=_SOURCE_LABELS.get(source, "本机视频"),
            status=status,
            status_label=_STATUS_LABELS[status],
            video_url=f"/media/{item['id']}" if available else None,
            thumbnail_url=f"/thumbnails/{item['id']}.jpg"
            if self.thumbnail_path(item["id"])
            else None,
            results=item.get("results", []),
            job_id=job_id,
            stage=job.get("stage", ""),
            frame=job.get("frame", 0),
            total_frames=job.get("total_frames", 0),
            error=job.get("error"),
        )
        return result

    def listing(self) -> dict:
        self.refresh()
        with self.lock:
            items = [
                self.public_item(item) for item in self.items.values() if not item.get("deleted_at")
            ]
            items.sort(key=lambda item: (item.get("created_at", ""), item["id"]), reverse=True)
            trashed = [
                self.public_item(item) for item in self.items.values() if item.get("deleted_at")
            ]
            trashed.sort(key=lambda item: (item.get("deleted_at", ""), item["id"]), reverse=True)
            return {"items": items, "trashed_items": trashed, "active_job_id": self.active_job_id()}
