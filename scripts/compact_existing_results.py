"""Plan lossless archive compaction; --apply writes verified gzip rollback copies.

Only four named JSON files and semantic_map.html in direct, legal output runs
are considered. Original videos, geometry values and recognition are untouched.
For rollback, decompress <run>/.optimization-backups/<filename>.gz and verify
its SHA-256 against sha256_before in the generated report before restoring it.
"""

from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import json
import os
import secrets
from http.client import HTTPConnection, HTTPException
from pathlib import Path

from visualmap.video_library import real_run_name
from visualmap.visualization.semantic_html import (
    render_viewer_payload,
    viewer_payload_from_html,
)

JSON_FILES = ("semantics.json", "semantic_map.json", "reconstruction.json", "occupancy.json")
FILES = (*JSON_FILES, "semantic_map.html")
_ACTIVE = {"uploading", "processing", "reviewing"}


def _safe(path: Path, root: Path) -> bool:
    """Reject every symlink/junction, even when it points back inside the root."""
    try:
        relative = path.absolute().relative_to(root.absolute())
        if not path.resolve().is_relative_to(root.resolve()):
            return False
        current = root
        for part in (None, *relative.parts):
            if part is not None:
                current /= part
            if current.is_symlink() or (hasattr(current, "is_junction") and current.is_junction()):
                return False
        return True
    except (OSError, ValueError, RuntimeError):
        return False


def _hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _unique(pairs):
    result = {}
    for name, value in pairs:
        if name in result:
            raise ValueError("duplicate JSON field")
        result[name] = value
    return result


def _invalid_constant(_):
    raise ValueError("non-finite JSON number")


def _json(data: bytes):
    return json.loads(
        data.decode("utf-8-sig"), object_pairs_hook=_unique, parse_constant=_invalid_constant
    )


def check_service_idle(root: Path, port: int = 8765) -> dict:
    """Fail closed on ambiguous connectivity; never send keys or mutate service."""
    connection = HTTPConnection("127.0.0.1", port, timeout=3)
    online = False
    try:
        connection.request("GET", "/api/library")
        response = connection.getresponse()
        if response.status != 200:
            raise RuntimeError("无法确认本机工作台任务状态，请先停止服务或等待任务结束")
        data = json.loads(response.read(2_000_000))
        online = True
        if not isinstance(data, dict) or "active_job_id" not in data:
            raise RuntimeError("本机端口未返回可验证的工作台状态")
        if data["active_job_id"]:
            raise RuntimeError("工作台正在识别视频，暂不能整理已有结果")
    except ConnectionRefusedError:
        pass
    except (OSError, HTTPException, ValueError) as error:
        raise RuntimeError("无法确认本机服务空闲；请检查访问权限与服务状态") from error
    finally:
        connection.close()
    # Persisted processing records are also checked when offline: an interrupted
    # run is not automatically permission to overwrite a possibly live process.
    index = root / "outputs/library/index.json"
    if index.is_file():
        if not _safe(index, root):
            raise RuntimeError("视频库索引路径不安全")
        jobs = _json(index.read_bytes()).get("jobs", {})
        if any(row.get("status") in _ACTIVE for row in jobs.values()):
            raise RuntimeError("视频库仍有进行中的任务，请先在工作台处理完毕")
    reviews = root / "work/llm/reviews"
    if reviews.exists():
        if not _safe(reviews, root):
            raise RuntimeError("复核记录路径不安全")
        for path in reviews.glob("review_*.json"):
            if not _safe(path, root):
                raise RuntimeError("复核记录包含链接，无法确认空闲")
            if _json(path.read_bytes()).get("status") == "processing":
                raise RuntimeError("仍有大模型复核进行中，请先结束复核再整理结果")
    return {"service_online": online, "idle_verified": True, "port": port}


def _replacement(path: Path, original: bytes) -> tuple[bytes, object, object]:
    if path.name in JSON_FILES:
        before = _json(original)
        after_bytes = json.dumps(
            before, ensure_ascii=False, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        after = _json(after_bytes)
        if after != before:
            raise ValueError("JSON verification failed")
        return after_bytes, before, after
    before = viewer_payload_from_html(original.decode("utf-8-sig").replace("\r\n", "\n"))
    if not isinstance(before, dict):
        raise ValueError("viewer payload must be an object")
    expected = copy.deepcopy(before)
    if "run_id" not in expected:
        expected["run_id"] = path.parent.name
    rendered = render_viewer_payload(copy.deepcopy(expected)).encode("utf-8")
    after = viewer_payload_from_html(rendered.decode("utf-8"))
    if after != expected:
        raise ValueError("viewer payload verification failed")
    # Apart from the explicit missing run_id addition, every existing field is
    # preserved; no rebuilding SemanticMap, tracking or volume calculations.
    return rendered, expected, after


def _write_atomic(path: Path, data: bytes, root: Path) -> None:
    temporary = path.parent / (".compact-" + secrets.token_hex(8) + ".tmp")
    if not _safe(path, root) or not _safe(temporary, root):
        raise ValueError("unsafe replacement path")
    try:
        with temporary.open("xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        if temporary.read_bytes() != data:
            raise ValueError("temporary byte verification failed")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _apply(path: Path, original: bytes, replacement: bytes, packed: bytes, expected, root: Path):
    backup = path.parent / ".optimization-backups" / (path.name + ".gz")
    if not _safe(backup, root):
        raise ValueError("unsafe rollback path")
    backup.parent.mkdir(exist_ok=True)
    if backup.exists():
        raise ValueError("rollback copy already exists; never overwriting it")
    # First write a same-directory temporary backup and verify decompression.
    temporary = backup.parent / (".rollback-" + secrets.token_hex(8) + ".tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(packed)
            stream.flush()
            os.fsync(stream.fileno())
        if gzip.decompress(temporary.read_bytes()) != original:
            raise ValueError("rollback byte verification failed")
        # A hard link installs the already complete rollback atomically, without
        # replacing an existing backup created by a concurrent invocation.
        os.link(temporary, backup)
    finally:
        temporary.unlink(missing_ok=True)
    if not _safe(path, root) or path.read_bytes() != original:
        raise ValueError("source changed before replacement; rollback preserved")
    _write_atomic(path, replacement, root)
    written = path.read_bytes()
    try:
        actual = (
            _json(written)
            if path.name in JSON_FILES
            else viewer_payload_from_html(written.decode("utf-8"))
        )
        if actual != expected or written != replacement:
            raise ValueError("post-write verification failed")
    except (ValueError, TypeError, KeyError, IndexError):
        # The original rollback remains available even if the automatic restore
        # itself is interrupted by an I/O failure.
        _write_atomic(path, original, root)
        raise ValueError("post-write verification failed; restored original") from None


def compact_existing_results(root: Path, *, apply: bool = False, port: int = 8765) -> dict:
    root = Path(root).absolute()
    if root.resolve() != root or not _safe(root, root):
        raise ValueError("project root must be a regular, canonical directory")
    service = check_service_idle(root, port)
    output = root / "outputs"
    if not output.is_dir() or not _safe(output, root):
        raise ValueError("outputs directory is unavailable or linked")
    rows = []
    for run in sorted(output.iterdir()):
        if not run.is_dir() or not real_run_name(run.name):
            continue
        if not _safe(run, root):
            rows.append(
                {
                    "path": run.relative_to(root).as_posix(),
                    "status": "skipped",
                    "reason": "linked_directory",
                }
            )
            continue
        if apply:
            check_service_idle(root, port)
        for name in FILES:
            path = run / name
            if not path.exists():
                continue
            row = {"path": path.relative_to(root).as_posix(), "status": "skipped"}
            rows.append(row)
            backup = run / ".optimization-backups" / (name + ".gz")
            if not path.is_file() or not _safe(path, root) or not _safe(backup, root):
                row["reason"] = "unsafe_path"
                continue
            if backup.exists():
                row["reason"] = "existing_rollback_copy"
                continue
            try:
                original = path.read_bytes()
                replacement, expected, _ = _replacement(path, original)
                packed = gzip.compress(original, compresslevel=9, mtime=0)
                row.update(
                    {
                        "bytes_before": len(original),
                        "bytes_after": len(replacement),
                        "backup_bytes": len(packed),
                        "net_saved_bytes": len(original) - len(replacement) - len(packed),
                        "sha256_before": _hash(original),
                        "sha256_after": _hash(replacement),
                        "backup": backup.relative_to(root).as_posix(),
                        "method": "refresh_viewer" if name.endswith(".html") else "compact_json",
                    }
                )
                if replacement == original:
                    row["reason"] = "already_current"
                    continue
                if name in JSON_FILES and row["net_saved_bytes"] <= 0:
                    row["reason"] = "no_net_saving_including_rollback"
                    continue
                if gzip.decompress(packed) != original:
                    raise ValueError("rollback verification failed")
                row["status"] = "planned"
                if apply:
                    _apply(path, original, replacement, packed, expected, root)
                    row["status"] = "applied"
                    row["backup_created"] = True
            except (OSError, ValueError, TypeError, KeyError, IndexError) as error:
                row.update(status="error", reason=str(error)[:240])
                if backup.is_file() and "backup_bytes" in row:
                    row["backup_created"] = True
                    row["actual_bytes_after"] = path.stat().st_size
    selected = [row for row in rows if row["status"] in {"planned", "applied"}]
    accounted = selected + [
        row for row in rows if row["status"] == "error" and row.get("backup_created")
    ]
    before_bytes = sum(row["bytes_before"] for row in accounted)
    after_bytes = sum(row.get("actual_bytes_after", row["bytes_after"]) for row in accounted)
    backup_bytes = sum(row["backup_bytes"] for row in accounted)
    return {
        "schema_version": 1,
        "mode": "apply" if apply else "plan",
        **service,
        "quality_preserved": True,
        "recognition_or_geometry_recomputed": False,
        "files": rows,
        "operation_count": len(selected),
        "bytes_before": before_bytes,
        "bytes_after": after_bytes,
        "backup_bytes": backup_bytes,
        "net_saved_bytes": before_bytes - after_bytes - backup_bytes,
        "error_count": sum(row["status"] == "error" for row in rows),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--apply", action="store_true", help="实施已验证的整理；默认只输出方案")
    parser.add_argument("--port", type=int, default=8765, help="本机工作台端口")
    parser.add_argument("--output", type=Path, help="保存JSON报告")
    args = parser.parse_args(argv)
    try:
        destination = args.output.absolute() if args.output else None
        if destination and (
            not _safe(destination, args.root.absolute())
            or destination.suffix != ".json"
            or destination.name in FILES
            or ".optimization-backups" in destination.parts
        ):
            raise ValueError("report must be a separate JSON file inside the project without links")
        report = compact_existing_results(args.root, apply=args.apply, port=args.port)
        serialized = json.dumps(report, ensure_ascii=False, indent=2)
        if destination:
            root = args.root.absolute()
            destination.parent.mkdir(parents=True, exist_ok=True)
            _write_atomic(destination, serialized.encode("utf-8"), root)
        print(serialized)
        return 1 if report["error_count"] else 0
    except (OSError, ValueError, RuntimeError) as error:
        print(json.dumps({"error": str(error)}, ensure_ascii=False))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
