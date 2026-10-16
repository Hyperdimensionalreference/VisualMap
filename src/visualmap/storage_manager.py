"""Explicit, bounded removal and byte-preserving reuse of managed workspace media.

An index entry is never permission to delete an arbitrary path. Every operation is
planned from actual files, bound to a short-lived token, and checked again while
holding the catalogue lock. Original project/sample videos and models are excluded.
"""

from __future__ import annotations

import copy
import hashlib
import os
import secrets
import shutil
import stat
import time
from pathlib import Path
from typing import TYPE_CHECKING

from visualmap.video_library import (
    VIDEO_EXTENSIONS,
    inside_without_symlinks,
    real_run_name,
    utc_now,
)

if TYPE_CHECKING:
    from visualmap.video_library import VideoLibrary

TOKEN_SECONDS = 120
_ACTIVE = {"uploading", "processing", "reviewing"}


def copy_media_atomic(source: Path, destination: Path) -> Path:
    """Copy media by replacement, so rewriting one run never mutates sibling links.

    Callers retain their normal source/destination authorization. This helper also
    works with project originals as input, but always creates an independent copy.
    """
    source, destination = Path(source), Path(destination)
    if source.exists() and destination.exists() and os.path.samefile(source, destination):
        return destination
    temporary = destination.parent / (".media-copy-" + secrets.token_hex(12))
    try:
        # Reserve exclusively; copy2 itself would otherwise follow an attacker-made
        # temporary symlink. The random path is never supplied by an API client.
        with temporary.open("xb") as output, source.open("rb") as incoming:
            shutil.copyfileobj(incoming, output, length=1024 * 1024)
            output.flush()
            os.fsync(output.fileno())
        shutil.copystat(source, temporary, follow_symlinks=False)
        os.replace(temporary, destination)
        return destination
    finally:
        temporary.unlink(missing_ok=True)


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _stamp(path: Path) -> tuple[int, int, int, int]:
    value = path.stat()
    return value.st_size, value.st_mtime_ns, value.st_dev, value.st_ino


def _scan_stats(directory: Path, root: Path):
    """Read cached directory metadata for statistics, never authorize a mutation.

    Check the base chain once, then reject every reparse entry before descending.
    This avoids resolving thousands of ancestor chains for a read-only byte count.
    The delete/replace paths retain their full immediate-before-mutation checks.
    """
    if not directory.is_dir() or not inside_without_symlinks(directory, root):
        return
    pending = [directory]
    reparse = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    while pending:
        current = pending.pop()
        try:
            current_stat = current.lstat()
            if not stat.S_ISDIR(current_stat.st_mode) or (
                getattr(current_stat, "st_file_attributes", 0) & reparse
            ):
                continue
            with os.scandir(current) as entries:
                for entry in entries:
                    try:
                        value = entry.stat(follow_symlinks=False)
                    except OSError:
                        continue
                    if getattr(value, "st_file_attributes", 0) & reparse:
                        continue
                    if stat.S_ISDIR(value.st_mode):
                        pending.append(Path(entry.path))
                    elif stat.S_ISREG(value.st_mode):
                        # Windows DirEntry.stat caches zero inode/device fields;
                        # a direct stat is required to count shared allocations.
                        if not value.st_ino:
                            value = os.stat(entry.path, follow_symlinks=False)
                        yield Path(entry.path), value
        except OSError:
            continue


def _walk_files(directory: Path, root: Path) -> tuple[list[Path], list[Path]]:
    """Enumerate without following reparse points; an unsafe child rejects a deletion."""
    if not directory.is_dir() or not inside_without_symlinks(directory, root):
        raise ValueError("目录包含链接或已超出项目管理范围")
    files, directories = [], [directory]
    for parent, folder_names, file_names in os.walk(directory, followlinks=False):
        for name in folder_names:
            path = Path(parent) / name
            if not inside_without_symlinks(path, root):
                raise ValueError("目录包含链接，无法安全操作")
            directories.append(path)
        for name in file_names:
            path = Path(parent) / name
            if not inside_without_symlinks(path, root) or not stat.S_ISREG(path.lstat().st_mode):
                raise ValueError("目录包含链接或特殊文件，无法安全操作")
            files.append(path)
    return sorted(files), sorted(directories, key=lambda path: len(path.parts), reverse=True)


def _relative(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


class _TokenPlans:
    def __init__(self) -> None:
        self._plans: dict[str, tuple[float, dict]] = {}

    def _issue(self, plan: dict) -> str:
        now = time.monotonic()
        self._plans = {
            key: value for key, value in self._plans.items() if value[0] > now
        }
        # Repeated previews cannot grow memory without bound.
        if len(self._plans) >= 64:
            self._plans.pop(next(iter(self._plans)))
        token = secrets.token_urlsafe(32)
        self._plans[token] = (now + TOKEN_SECONDS, plan)
        return token

    def _consume(self, token: str) -> dict:
        row = self._plans.pop(token, None) if isinstance(token, str) else None
        if row is None or row[0] <= time.monotonic():
            raise ValueError("确认已过期或不可用，请重新预览并确认")
        return row[1]


class PurgeManager(_TokenPlans):
    """Permanent deletion only of exclusive uploads, thumbnails, and verified runs."""

    def __init__(self, library: VideoLibrary):
        super().__init__()
        self.library = library
        self.root = library.root

    def _busy(self, item_id: str, run_name: str | None = None) -> bool:
        return any(
            job.get("status") in _ACTIVE
            and (job.get("library_id") == item_id or key == run_name)
            for key, job in self.library.jobs.items()
        )

    def _path_shared(self, item_id: str, path: Path, directory: bool = False) -> bool:
        relative = _relative(path, self.root)
        if directory and any(
            key == path.name and job.get("library_id") not in {None, item_id}
            for key, job in self.library.jobs.items()
        ):
            return True
        for key, other in self.library.items.items():
            if key == item_id:
                continue
            if directory and any(
                row.get("id") == path.name for row in other.get("results", [])
            ):
                return True
            for row in other.get("sources", []):
                source = row.get("path", "")
                if source == relative or (
                    directory and isinstance(source, str) and source.startswith(relative + "/")
                ):
                    return True
        return False

    def _collect(self, item_id: str) -> dict:
        item = self.library.items.get(item_id)
        if item is None:
            raise KeyError(item_id)
        if not item.get("deleted_at"):
            raise ValueError("请先把视频移到回收站，再彻底删除")
        if self._busy(item_id):
            raise ValueError("这个视频正在处理，完成后才能彻底删除")
        files: dict[str, dict] = {}
        directories: list[str] = []
        retained: dict[str, dict] = {}

        def retain(relative: str, reason: str) -> None:
            retained[relative] = {"path": relative, "reason": reason}

        def add_file(path: Path, kind: str) -> None:
            if not inside_without_symlinks(path, self.root):
                raise ValueError("待删除路径包含链接或超出项目范围")
            if not path.is_file() or not stat.S_ISREG(path.lstat().st_mode):
                raise ValueError("待删除文件发生变化，请重新预览")
            relative = _relative(path, self.root)
            files[relative] = {
                "path": relative, "bytes": path.stat().st_size, "kind": kind,
                "stamp": _stamp(path),
            }

        uploads = self.root / "outputs/uploads"
        for row in item.get("sources", []):
            relative = row.get("path", "")
            if not isinstance(relative, str):
                continue
            path = self.root / relative
            # Require a canonical relative spelling, not an index-crafted traversal.
            if (
                path.is_absolute() and inside_without_symlinks(path, self.root)
                and _relative(path, self.root) == relative
                and path.parent == uploads and path.suffix.lower() in VIDEO_EXTENSIONS
            ):
                if not path.exists():
                    continue
                if self._path_shared(item_id, path):
                    retain(relative, "仍被其他视频记录引用")
                elif _digest(path) != item["sha256"]:
                    retain(relative, "文件内容已变化，不属于当前视频")
                else:
                    add_file(path, "upload")
            else:
                retain(relative, "保留项目原视频、示例或其他非上传来源")

        run_names = {
            row.get("id") for row in item.get("results", [])
            if real_run_name(str(row.get("id", "")))
        }
        # Failed/interrupted jobs may also have sizeable managed partial output.
        run_names.update(
            key for key, job in self.library.jobs.items()
            if job.get("library_id") == item_id and real_run_name(key)
        )
        for name in sorted(run_names):
            directory = self.root / "outputs" / name
            relative = _relative(directory, self.root)
            if not directory.exists():
                continue
            if not inside_without_symlinks(directory, self.root):
                raise ValueError("识别结果目录包含链接，无法彻底删除")
            if self._busy(item_id, name) or self._path_shared(item_id, directory, True):
                retain(relative, "结果仍在处理或被其他视频记录引用")
                continue
            source_files = sorted(directory.glob("source_video.*"))
            tombstone = self.library.tombstones.get(item["sha256"], {})
            prior_manifest = tombstone.get("manifest", {})
            resume_partial = (
                tombstone.get("status") in {"partial", "pending"}
                and relative in tombstone.get("owned_runs", [])
            )
            if (not source_files and not resume_partial) or any(
                not (
                    self.library.authorized_video(path)
                    or (resume_partial and path.is_file()
                        and inside_without_symlinks(path, self.root)
                        and path.suffix.lower() in VIDEO_EXTENSIONS
                        and list(_stamp(path)) == prior_manifest.get(_relative(path, self.root)))
                )
                or _digest(path) != item["sha256"] for path in source_files
            ):
                retain(relative, "未能核实此结果目录专属于当前视频")
                continue
            run_files, run_directories = _walk_files(directory, self.root)
            if not source_files and any(
                list(_stamp(path)) != prior_manifest.get(_relative(path, self.root))
                for path in run_files
            ):
                retain(relative, "部分删除后的文件发生变化，请检查剩余结果")
                continue
            for path in run_files:
                if self._path_shared(item_id, path):
                    retain(relative, "目录内文件仍被其他视频记录引用")
                    break
            else:
                for path in run_files:
                    add_file(path, "result")
                directories.extend(_relative(path, self.root) for path in run_directories)

        thumbnail = self.library.thumbnails / (item_id + ".jpg")
        if thumbnail.exists():
            add_file(thumbnail, "thumbnail")
        retained = {key: row for key, row in retained.items() if key not in files}
        return {
            "item_id": item_id, "name": item.get("name", ""),
            "sha256": item["sha256"], "deleted_at": item["deleted_at"],
            "files": sorted(files.values(), key=lambda row: row["path"]),
            "directories": sorted(set(directories)),
            "retained_sources": sorted(retained.values(), key=lambda row: row["path"]),
        }

    def plan(self, item_id: str) -> dict:
        with self.library.lock:
            plan = self._collect(item_id)
            token = self._issue(copy.deepcopy(plan))
            return {
                "item_id": item_id, "name": plan["name"],
                "confirmation_token": token, "expires_in_seconds": TOKEN_SECONDS,
                "files": [{k: row[k] for k in ("path", "bytes", "kind")}
                          for row in plan["files"]],
                "file_count": len(plan["files"]),
                "total_bytes": sum(row["bytes"] for row in plan["files"]),
                "estimated_reclaimed_bytes": sum(
                    row["bytes"] for row in plan["files"]
                    if (self.root / row["path"]).stat().st_nlink == 1
                ),
                "retained_sources": plan["retained_sources"], "irreversible": True,
            }

    def purge(self, item_id: str, confirmation_token: str) -> dict:
        with self.library.lock:
            plan = self._consume(confirmation_token)
            if plan["item_id"] != item_id or self._collect(item_id) != plan:
                raise ValueError("视频、文件或引用已变化，请重新预览并确认")
            item = self.library.items[item_id]
            digest = item["sha256"]
            previous = self.library.tombstones.get(digest)
            # Persist scan suppression before unlinking anything. On partial failure
            # the trashed item remains available for a new preview/retry.
            self.library.tombstones[digest] = {
                "item_id": item_id, "purged_at": utc_now(), "status": "pending",
                "manifest": {row["path"]: list(row["stamp"]) for row in plan["files"]},
                "owned_runs": [value for value in plan["directories"]
                               if len(Path(value).parts) == 2],
            }
            try:
                self.library.save(force=True)
            except OSError:
                if previous is None:
                    self.library.tombstones.pop(digest, None)
                else:
                    self.library.tombstones[digest] = previous
                raise
            removed, freed, reclaimed = [], 0, 0
            try:
                for row in sorted(plan["files"], key=lambda value: (
                    Path(value["path"]).stem == "source_video", value["path"]
                )):
                    path = self.root / row["path"]
                    # Each absolute target is revalidated immediately before unlink.
                    if not inside_without_symlinks(path, self.root) or _stamp(path) != row["stamp"]:
                        raise ValueError("文件发生变化，已停止删除，请重新预览")
                    if path.stat().st_nlink == 1:
                        reclaimed += row["bytes"]
                    path.unlink()
                    removed.append(row["path"])
                    freed += row["bytes"]
                for relative in sorted(
                    plan["directories"], key=lambda value: len(Path(value).parts), reverse=True
                ):
                    directory = self.root / relative
                    if not inside_without_symlinks(directory, self.root):
                        raise ValueError("目录发生变化，已停止删除")
                    # rmdir requires an empty directory and cannot recursively follow
                    # a newly introduced junction or delete an unexpected file.
                    directory.rmdir()
            except (OSError, ValueError):
                self.library.tombstones[digest].update(status="partial", deleted_files=removed)
                self.library.save(force=True)
                raise
            self.library.tombstones[digest].update(status="complete", deleted_files=removed)
            self.library.tombstones[digest].pop("manifest", None)
            self.library.tombstones[digest].pop("owned_runs", None)
            self.library.items.pop(item_id)
            for key in list(self.library.jobs):
                if self.library.jobs[key].get("library_id") == item_id:
                    self.library.jobs.pop(key)
            for relative in removed:
                self.library._fingerprints.pop(relative, None)
            self.library.save(force=True)
            return {
                "item_id": item_id, "purged": True, "deleted_files": len(removed),
                "freed_logical_bytes": freed, "retained_sources": plan["retained_sources"],
                "estimated_reclaimed_bytes": reclaimed,
                "removed_runs": sorted({
                    Path(row["path"]).parts[1] for row in plan["files"]
                    if row["kind"] == "result"
                }),
            }


class StorageManager(_TokenPlans):
    """Keep every media byte and result; reuse identical managed video allocations."""

    def __init__(self, library: VideoLibrary):
        super().__init__()
        self.library = library
        self.root = library.root

    def _managed_videos(self) -> list[Path]:
        # Only copy catalogue state while holding its lock. Directory traversal,
        # authorization stat calls and later hashing must not block video browsing.
        with self.library.lock:
            active_jobs = {
                key for key, job in self.library.jobs.items() if job.get("status") in _ACTIVE
            }
            active_items = {
                job.get("library_id") for job in self.library.jobs.values()
                if job.get("status") in _ACTIVE
            }
            blocked = {
                row.get("path") for key, item in self.library.items.items() if key in active_items
                for row in item.get("sources", [])
            }
        paths = []
        output = self.root / "outputs"
        if not output.is_dir() or not inside_without_symlinks(output, self.root):
            return paths
        for directory in sorted(output.iterdir()):
            try:
                if not directory.is_dir() or not inside_without_symlinks(directory, self.root):
                    continue
                if directory.name == "uploads":
                    candidates = list(directory.iterdir())
                elif real_run_name(directory.name) and (directory / "semantic_map.html").is_file():
                    candidates = list(directory.glob("source_video.*"))
                else:
                    continue
                paths.extend(path for path in candidates if self.library.authorized_video(path))
            except FileNotFoundError:
                # A read-only snapshot can overlap a completed deletion. Mutation
                # still reauthorizes every path and file stamp under the lock.
                continue
        return [path for path in sorted(paths) if _relative(path, self.root) not in blocked
                and path.parent.name not in active_jobs]

    def stats(self) -> dict:
        # Statistics are a read-only snapshot, not deletion authorization. Keep
        # slow filesystem I/O outside the catalogue lock used by /api/library.
        output = self.root / "outputs"
        logical_bytes, file_count, unique = 0, 0, {}
        for _, value in _scan_stats(output, self.root):
            logical_bytes += value.st_size
            file_count += 1
            unique[(value.st_dev, value.st_ino)] = value.st_size
        cache_bytes = 0
        for name in (".pytest_cache", ".ruff_cache", "src", "tests"):
            directory = self.root / name
            cache_bytes += sum(
                value.st_size for path, value in _scan_stats(directory, self.root)
                if name.startswith(".") or path.suffix == ".pyc"
            )
        managed_bytes = 0
        for path in self._managed_videos():
            try:
                managed_bytes += path.stat().st_size
            except FileNotFoundError:
                continue
        return {
            "logical_bytes": logical_bytes, "unique_file_bytes": sum(unique.values()),
            "allocated_bytes_estimate": sum(unique.values()), "file_count": file_count,
            "managed_video_bytes": managed_bytes, "cache_bytes": cache_bytes,
            "scope": "outputs", "snapshot": True,
            "notes": [
                "统计范围为视频库与识别结果（outputs），不含模型、运行环境或其他项目文件。",
                "逻辑大小按每个路径计数；独占文件大小按文件标识去重，反映硬链接节省。",
                "磁盘占用为读取时快照的文件长度估计，不含簇舍入或文件系统压缩差异。",
                "仅复用项目管理的相同字节视频，不转码、不删除识别数据、模型或原始示例。",
                "缓存只审计，不自动清理正在使用的文件。",
            ],
        }

    def _collect(self) -> list[dict]:
        # Group by length before hashing; unlike name/mtime, SHA-256 proves identical
        # media contents before a copy may share storage with another managed copy.
        by_size: dict[int, list[Path]] = {}
        for path in self._managed_videos():
            try:
                by_size.setdefault(path.stat().st_size, []).append(path)
            except FileNotFoundError:
                continue
        operations = []
        sampled_stamps: dict[Path, tuple[int, int, int, int]] = {}
        for size, paths in sorted(by_size.items()):
            if len(paths) < 2 or size == 0:
                continue
            identities = {(path.stat().st_dev, path.stat().st_ino) for path in paths}
            if len(identities) == 1:
                # These paths already share one allocation: no bytes to reclaim,
                # and no reason to hash that same large video over and over.
                continue
            groups: dict[str, list[Path]] = {}
            digests: dict[tuple[int, int, int, int], str] = {}
            for path in paths:
                before = _stamp(path)
                sampled_stamps[path] = before
                if before not in digests:
                    digest = _digest(path)
                    if _stamp(path) != before:
                        raise ValueError("扫描期间视频发生变化，请重新扫描")
                    digests[before] = digest
                groups.setdefault(digests[before], []).append(path)
            for digest, copies in groups.items():
                if len(copies) < 2:
                    continue
                known_links: dict[tuple[int, int], int] = {}
                for path in copies:
                    value = path.stat()
                    identity = (value.st_dev, value.st_ino)
                    known_links[identity] = known_links.get(identity, 0) + 1
                copies = [path for path in copies if (
                    (value := path.stat()).st_nlink
                    <= known_links[(value.st_dev, value.st_ino)]
                )]
                if len(copies) < 2:
                    continue
                canonical = copies[0]
                canonical_stat = canonical.stat()
                seen_inodes = {(canonical_stat.st_dev, canonical_stat.st_ino)}
                for duplicate in copies[1:]:
                    duplicate_stat = duplicate.stat()
                    identity = (duplicate_stat.st_dev, duplicate_stat.st_ino)
                    if identity in seen_inodes or duplicate_stat.st_dev != canonical_stat.st_dev:
                        continue
                    # External hardlinks mean replacing this one path may free no
                    # physical allocation; leave them alone to report honest savings.
                    if duplicate_stat.st_nlink != 1:
                        continue
                    operations.append({
                        "source": _relative(canonical, self.root),
                        "target": _relative(duplicate, self.root), "bytes": size,
                        "sha256": digest, "source_stamp": sampled_stamps[canonical],
                        "target_stamp": sampled_stamps[duplicate],
                    })
                    seen_inodes.add(identity)
        # A previous file can change while a later file is being hashed. Bind the
        # entire preview to the sampled versions, not stamps obtained after I/O.
        if any(_stamp(path) != stamp for path, stamp in sampled_stamps.items()):
            raise ValueError("扫描期间视频发生变化，请重新扫描")
        return operations

    def plan(self) -> dict:
        operations = self._collect()
        with self.library.lock:
            token = self._issue(copy.deepcopy({"operations": operations}))
        return {
            "plan_token": token, "expires_in_seconds": TOKEN_SECONDS,
            "operations": [{key: row[key] for key in ("source", "target", "bytes", "sha256")}
                           for row in operations],
            "operation_count": len(operations),
            "potential_saved_bytes": sum(row["bytes"] for row in operations),
            "quality_preserved": True, "method": "hardlink_identical_managed_videos",
        }

    def optimize(self, plan_token: str) -> dict:
        with self.library.lock:
            plan = self._consume(plan_token)
            if self._collect() != plan["operations"]:
                raise ValueError("视频文件或任务状态已变化，请重新查看存储优化方案")
            before = self.stats()
            optimized, skipped, saved = 0, [], 0
            managed = set(self._managed_videos())
            for row in plan["operations"]:
                source, target = self.root / row["source"], self.root / row["target"]
                temporary = target.parent / (".storage-link-" + secrets.token_hex(8))
                try:
                    if (
                        source not in managed or target not in managed
                        or not inside_without_symlinks(source, self.root)
                        or not inside_without_symlinks(target, self.root)
                        or _stamp(source) != row["source_stamp"]
                        or _stamp(target) != row["target_stamp"]
                        or _digest(source) != row["sha256"]
                        or _digest(target) != row["sha256"]
                        or not inside_without_symlinks(temporary, self.root)
                    ):
                        raise ValueError("文件或任务发生变化，未修改此副本")
                    os.link(source, temporary)
                    if (
                        not os.path.samefile(source, temporary)
                        or _digest(temporary) != row["sha256"]
                    ):
                        raise OSError("无法验证无损副本")
                    os.replace(temporary, target)
                    optimized += 1
                    saved += row["bytes"]
                except (OSError, ValueError) as error:
                    skipped.append({"path": row["target"], "reason": str(error)})
                finally:
                    if temporary.exists() and inside_without_symlinks(temporary, self.root):
                        temporary.unlink(missing_ok=True)
            return {
                "optimized_files": optimized, "saved_bytes": saved, "before": before,
                "after": self.stats(), "quality_preserved": True, "skipped": skipped,
            }
