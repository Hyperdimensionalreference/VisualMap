"""Authorized manual review jobs and management routes for the local workbench."""

from __future__ import annotations

import copy
import json
import re
import secrets
import threading

from visualmap.llm_assist import AssistError, LLMAssistService, _atomic_json, _now
from visualmap.object_knowledge import ObjectKnowledgeStore
from visualmap.video_library import inside_without_symlinks, real_run_name

_ID = re.compile(r"[A-Za-z0-9_-]{1,120}\Z")


class WorkspaceExtensions:
    """Keep provider credentials out of page data and ordinary job history."""

    def __init__(self, server):
        self.server = server
        self.directory = server.root / "work/llm"
        if not inside_without_symlinks(self.directory, server.root):
            raise AssistError("大模型本机目录不可用", 503, "storage_unavailable")
        self.directory.mkdir(parents=True, exist_ok=True)
        self.service = LLMAssistService(self.directory / "settings")
        self.knowledge = ObjectKnowledgeStore(self.directory / "knowledge")
        self.reviews_dir = self.directory / "reviews"
        if not inside_without_symlinks(self.reviews_dir, self.directory):
            raise AssistError("复核记录目录不可用", 503, "storage_unavailable")
        self.reviews_dir.mkdir(exist_ok=True)
        self.lock = threading.RLock()
        self.reviews: dict[str, dict] = {}
        self.cancellations: dict[str, threading.Event] = {}
        self._storage = None
        self.cleanup_path = self.directory / "purge_cleanup.json"
        self.pending_purges: dict[str, dict] = {}
        for path in self.reviews_dir.glob("review_*.json"):
            try:
                if (
                    not inside_without_symlinks(path, self.reviews_dir)
                    or path.stat().st_size > 300_000
                ):
                    continue
                row = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if (
                not isinstance(row, dict)
                or not isinstance(row.get("id"), str)
                or not _ID.fullmatch(row["id"])
                or path.name != row["id"] + ".json"
            ):
                continue
            if row.get("status") == "processing":
                recovered = {**row, "status": "error", "error": "上次复核已中断，请手动重新提交。"}
                try:
                    # Publish recovery only after the on-disk idle check sees it,
                    # preserving the original record if atomic replacement fails.
                    _atomic_json(path, recovered, self.reviews_dir)
                except (OSError, AssistError) as error:
                    raise AssistError(
                        "中断复核记录无法保存，请检查本机存储后重试。",
                        503,
                        "storage_unavailable",
                    ) from error
                row = recovered
            self.reviews[row["id"]] = row
        self._recover_purge_cleanup()

    @property
    def storage(self):
        if self._storage is None:
            from visualmap.storage_manager import StorageManager

            self._storage = StorageManager(self.server.library)
        return self._storage

    def close(self):
        with self.lock:
            for event in self.cancellations.values():
                event.set()

    def settings(self):
        settings = self.service.public_settings()
        return {
            **settings,
            "has_api_key": settings["api_key_present"],
            "key_storage": "protected"
            if settings["key_storage"] == "dpapi"
            else settings["key_storage"],
        }

    def save_settings(self, payload):
        values = dict(payload)
        if "key_storage" in values:
            if values["key_storage"] not in {"session", "protected"}:
                raise AssistError("密钥保存选项不正确")
            values["persist_api_key"] = values.pop("key_storage") == "protected"
        with self.lock:
            self.ensure_not_reviewing()
            self.service.save_settings(values)
        return self.settings()

    def authorize_run(self, run_id, library_id=None):
        if not isinstance(run_id, str) or not real_run_name(run_id):
            raise AssistError("识别结果编号不正确", 404, "not_found")
        directory = self.server.root / "outputs" / run_id
        if (
            not inside_without_symlinks(directory, self.server.root / "outputs")
            or not (directory / "semantic_map.html").is_file()
        ):
            raise AssistError("识别结果不可用，请从视频库重新打开", 404, "not_found")
        self.server.library.refresh()
        with self.server.lock:
            for item in self.server.library.items.values():
                if library_id and item["id"] != library_id:
                    continue
                if item.get("deleted_at"):
                    continue
                if not any(row.get("id") == run_id for row in item.get("results", [])):
                    continue
                video = self.server.library.video_path(item["id"])
                if video is not None:
                    return directory, copy.deepcopy(item), video
        raise AssistError("源视频不可用或已移入回收站，请先恢复", 404, "not_found")

    def context(self, run_id):
        _, item, _ = self.authorize_run(run_id)
        return {
            "token": self.server.token,
            "run_id": run_id,
            "library_id": item["id"],
            "settings": self.settings(),
            "manual_only": True,
        }

    def _original_label(self, directory, track_id):
        if track_id is None:
            return ""
        if isinstance(track_id, bool) or not isinstance(track_id, int) or track_id < 0:
            raise AssistError("物体编号不正确")
        path = directory / "semantics.json"
        if not inside_without_symlinks(path, directory) or not path.is_file():
            raise AssistError("原识别记录不可用")
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            for row in data.get("objects", []):
                if row.get("track_id") == track_id:
                    return str(row.get("class_name", ""))[:160]
        except (OSError, ValueError) as error:
            raise AssistError("原识别记录无法读取") from error
        raise AssistError("所选物体不属于这个识别结果")

    def _save(self, row):
        _atomic_json(self.reviews_dir / (row["id"] + ".json"), row, self.reviews_dir)

    @staticmethod
    def public_review(row):
        result = row.get("result")
        return {
            "id": row["id"],
            "status": row["status"],
            **(
                {
                    "result": {
                        **result,
                        "name": result.get("candidate_name", "未知"),
                        "reason": result.get("evidence", []),
                    }
                }
                if result
                else {}
            ),
            **({"error": row["error"]} if row.get("error") else {}),
            **({"label": row["label"]} if row.get("label") else {}),
        }

    def start_review(self, payload):
        run_id, track_id = payload.get("run_id"), payload.get("track_id")
        directory, item, video = self.authorize_run(run_id, payload.get("library_id"))
        original = self._original_label(directory, track_id)
        selections = payload.get("selections")
        if not isinstance(selections, list) or len(selections) != 3:
            raise AssistError("请为同一个物体框选三个不同时间的画面")
        frames = [row.get("frame_id") for row in selections if isinstance(row, dict)]
        if (
            len(frames) != 3
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value < 0
                for value in frames
            )
            or len(set(frames)) != 3
        ):
            raise AssistError("需要三个不同的源视频帧")
        question = payload.get("question", "")
        if not isinstance(question, str) or len(question) > 1200:
            raise AssistError("补充说明过长或格式不正确")
        if not self.service.public_settings()["model"]:
            raise AssistError("请先在大模型设置中填写视觉模型与API地址")
        with self.lock:
            # Recheck after acquiring the same lock used by purge. An item
            # could have moved to the bin while the initial track was read.
            directory, item, video = self.authorize_run(run_id, item["id"])
            if self.server.closing.is_set():
                raise AssistError("工作区正在关闭", 503, "closing")
            if any(row["status"] == "processing" for row in self.reviews.values()):
                raise AssistError("正在复核另一组画面，请等待完成", 409, "busy")
            identifier = "review_" + secrets.token_hex(12)
            row = {
                "id": identifier,
                "run_id": run_id,
                "library_id": item["id"],
                "track_id": track_id,
                "original_label": original,
                "created_at": _now(),
                "status": "processing",
            }
            self._save(row)
            self.reviews[identifier] = row
            cancel = self.cancellations[identifier] = threading.Event()
        threading.Thread(
            target=self._review_worker,
            args=(identifier, video, selections, question, cancel),
            daemon=True,
        ).start()
        return self.public_review(row)

    def _review_worker(self, identifier, video, selections, question, cancel):
        try:
            result = self.service.review(video, selections, prompt=question, cancel_event=cancel)
            with self.lock:
                row = self.reviews[identifier]
                if cancel.is_set() or self.server.closing.is_set():
                    row.update(status="error", error="复核已中断，请手动重试。")
                else:
                    row.update(status="complete", result={**result, "id": identifier})
                self._save(row)
        except Exception as error:
            with self.lock:
                row = self.reviews.get(identifier)
                if row is not None:
                    row.update(
                        status="error",
                        error=str(error)
                        if isinstance(error, AssistError)
                        else "本次复核未完成，请检查设置后手动重试。",
                    )
                    self._save(row)
        finally:
            with self.lock:
                self.cancellations.pop(identifier, None)

    def review(self, identifier):
        with self.lock:
            row = self.reviews.get(identifier)
            if row is None:
                raise AssistError("复核记录不存在", 404, "not_found")
            self.authorize_run(row["run_id"], row["library_id"])
            return self.public_review(copy.deepcopy(row))

    @staticmethod
    def label(row):
        return {
            **row,
            "name": row["label"],
            "active": not bool(row.get("revoked_at")),
            "object_id": str(row["track_id"])
            if row["track_id"] is not None
            else "manual_" + row["id"],
            "source": "manual_llm_assisted",
        }

    def labels(self, run_id):
        self.authorize_run(run_id)
        return {"labels": [self.label(row) for row in self.knowledge.list_for_run(run_id)]}

    def confirm(self, identifier, name):
        with self.lock:
            row = self.reviews.get(identifier)
            if row is None or row.get("status") != "complete":
                raise AssistError("请先等三视图复核完成")
            self.authorize_run(row["run_id"], row["library_id"])
            if row.get("label"):
                existing = self.knowledge.get(row["label"]["id"])
                if not existing.get("revoked_at"):
                    row["label"] = self.label(existing)
                    return {"label": row["label"]}
            record = self.knowledge.confirm(
                run_id=row["run_id"],
                track_id=row["track_id"],
                label=name,
                review=row["result"],
                original_label=row["original_label"],
                source_video_id=row["library_id"],
            )
            row["label"] = self.label(record)
            self._save(row)
            return {"label": row["label"]}

    def revert(self, identifier):
        row = self.knowledge.get(identifier)
        self.authorize_run(row["run_id"], row["source_video_id"])
        label = self.label(self.knowledge.revoke(identifier))
        with self.lock:
            for review in self.reviews.values():
                if review.get("label", {}).get("id") == identifier:
                    review["label"] = label
                    self._save(review)
        return {"label": label}

    def ensure_not_reviewing(self, item_id=None):
        with self.lock:
            if any(
                row["status"] == "processing" and (item_id is None or row["library_id"] == item_id)
                for row in self.reviews.values()
            ):
                raise AssistError("大模型正在读取/复核画面，请完成后再执行此操作", 409, "busy")

    def forget_video(self, item_id):
        self.ensure_not_reviewing(item_id)
        self.knowledge.delete_for_video(item_id)
        with self.lock:
            for identifier, row in list(self.reviews.items()):
                if row["library_id"] != item_id:
                    continue
                path = self.reviews_dir / (identifier + ".json")
                if not inside_without_symlinks(path, self.reviews_dir):
                    raise AssistError("复核记录路径不可用", 503, "storage_unavailable")
                path.unlink(missing_ok=True)
                del self.reviews[identifier]

    def _save_purge_cleanup(self):
        _atomic_json(
            self.cleanup_path,
            {"schema_version": 1, "items": list(self.pending_purges.values())},
            self.directory,
        )

    def _purge_completed(self, item_id, intent):
        """A missing item alone is never authorization to erase learned records."""
        with self.server.lock:
            tombstone = self.server.library.tombstones.get(intent.get("sha256"), {})
            return (
                item_id not in self.server.library.items
                and tombstone.get("item_id") == item_id
                and tombstone.get("status") == "complete"
            )

    def _recover_purge_cleanup(self):
        """Retry only previously confirmed, durably completed video removals."""
        if self.cleanup_path.exists():
            if (
                not inside_without_symlinks(self.cleanup_path, self.directory)
                or self.cleanup_path.stat().st_size > 300_000
            ):
                raise AssistError("待完成删除记录不可用", 503, "storage_unavailable")
            try:
                data = json.loads(self.cleanup_path.read_text(encoding="utf-8"))
                if not isinstance(data, dict) or data.get("schema_version") != 1:
                    raise ValueError
                self.pending_purges = {
                    row["item_id"]: row for row in data.get("items", [])
                    if isinstance(row, dict)
                    and _ID.fullmatch(str(row.get("item_id", "")))
                    and re.fullmatch(r"[a-f0-9]{64}", str(row.get("sha256", "")))
                }
            except (OSError, ValueError, TypeError):
                # Preserve unreadable evidence and make no deletion assumption.
                raise AssistError("待完成删除记录无法读取", 503, "storage_unavailable") from None
        changed = False
        for item_id, intent in list(self.pending_purges.items()):
            if not self._purge_completed(item_id, intent):
                continue
            try:
                self.forget_video(item_id)
            except (OSError, AssistError):
                continue
            self.pending_purges.pop(item_id)
            changed = True
        if changed:
            try:
                self._save_purge_cleanup()
            except (OSError, AssistError):
                # The persisted intent remains and an idempotent retry is safe.
                pass

    def purge(self, item_id, confirmation_token):
        with self.lock:
            self.ensure_not_reviewing(item_id)
            with self.server.lock:
                item = self.server.library.items.get(item_id)
                if item is None:
                    raise KeyError(item_id)
                previous = self.pending_purges.get(item_id)
                self.pending_purges[item_id] = {
                    "item_id": item_id, "sha256": item["sha256"], "requested_at": _now(),
                }
                try:
                    self._save_purge_cleanup()
                except (OSError, AssistError):
                    if previous is None:
                        self.pending_purges.pop(item_id, None)
                    else:
                        self.pending_purges[item_id] = previous
                    raise
                try:
                    result = self.server.library.purge(item_id, confirmation_token)
                except (OSError, ValueError, KeyError):
                    # An invalid preview/token did not authorize deletion. Partial
                    # removals keep the intent, but recovery waits for completion.
                    tombstone = self.server.library.tombstones.get(item["sha256"], {})
                    if tombstone.get("item_id") != item_id:
                        self.pending_purges.pop(item_id, None)
                        self._save_purge_cleanup()
                    raise
            self.forget_video(item_id)
            self.pending_purges.pop(item_id, None)
            self._save_purge_cleanup()
            return result


def _body(handler):
    try:
        length = int(handler.headers.get("Content-Length", "0"))
    except ValueError:
        length = -1
    if not 0 <= length <= 65536:
        handler.reject_upload(400, "设置或操作请求过大")
        return None
    try:
        value = json.loads(handler.rfile.read(length)) if length else {}
        if not isinstance(value, dict):
            raise ValueError
        return value
    except (ValueError, UnicodeDecodeError):
        handler.json(400, {"error": "请求格式不正确"})
        return None


def handle_extension(handler, route, query, mutation=False):
    """Return False for ordinary workspace routes, keeping their behavior intact."""
    recognized = (
        route.startswith("/api/llm/")
        or route in {"/api/storage", "/api/storage/optimize"}
        or bool(re.fullmatch(r"/api/library/[^/]+/(purge-preview|purge)", route))
    )
    if not recognized:
        return False
    try:
        if mutation:
            payload = _body(handler)
            if payload is None:
                return True
        else:
            payload = {}
        server = handler.server
        with server.lock:
            if server.extensions is None:
                server.extensions = WorkspaceExtensions(server)
            extension = server.extensions
        run_id = query.get("run_id", [""])[0]
        if route == "/api/llm/settings":
            result = extension.save_settings(payload) if mutation else extension.settings()
        elif route == "/api/llm/context" and not mutation:
            result = extension.context(run_id)
        elif route == "/api/llm/labels" and not mutation:
            result = extension.labels(run_id)
        elif route == "/api/llm/reviews" and mutation:
            handler.json(202, extension.start_review(payload))
            return True
        elif match := re.fullmatch(r"/api/llm/reviews/([A-Za-z0-9_-]+)(/confirm)?", route):
            if match[2] == "/confirm" and mutation:
                result = extension.confirm(match[1], payload.get("name"))
            elif not match[2] and not mutation:
                result = extension.review(match[1])
            else:
                raise AssistError("未知复核操作", 404, "not_found")
        elif match := re.fullmatch(r"/api/llm/labels/([A-Za-z0-9_-]+)/revert", route):
            if not mutation:
                raise AssistError("请显式提交撤销操作", 405, "method_not_allowed")
            result = extension.revert(match[1])
        elif route == "/api/storage" and not mutation:
            result = {"stats": extension.storage.stats(), "plan": extension.storage.plan()}
        elif route == "/api/storage/optimize" and mutation:
            with extension.lock:
                extension.ensure_not_reviewing()
                result = extension.storage.optimize(payload.get("plan_token"))
        elif match := re.fullmatch(
            r"/api/library/([A-Za-z0-9_-]{1,100})/(purge-preview|purge)", route
        ):
            with extension.lock:
                extension.ensure_not_reviewing(match[1])
                if match[2] == "purge-preview" and not mutation:
                    result = server.library.plan_purge(match[1])
                elif match[2] == "purge" and mutation:
                    result = extension.purge(match[1], payload.get("confirmation_token"))
                else:
                    raise AssistError("未知删除操作", 405, "method_not_allowed")
        else:
            raise AssistError("未知操作", 404, "not_found")
        handler.json(200, result)
    except AssistError as error:
        handler.json(error.status, {"error": str(error), "code": error.code})
    except KeyError:
        handler.json(404, {"error": "视频或记录不存在"})
    except (ValueError, TypeError) as error:
        handler.json(409, {"error": str(error)[:240]})
    except OSError:
        handler.json(503, {"error": "本机操作未完成，请稍后重试"})
    return True
