"""Confirmed, reversible per-object naming evidence; never model training.

Records reference three original source frames instead of duplicating images.
They overlay only the explicitly confirmed object's label. Original detection
labels, coordinates, geometry and future videos are never modified here.
"""

from __future__ import annotations

import copy
import json
import re
import secrets
import threading
from pathlib import Path

from visualmap.llm_assist import AssistError, _atomic_json, _now, _text, normalize_base_url
from visualmap.video_library import inside_without_symlinks

_ID = re.compile(r"[A-Za-z0-9_-]{1,120}\Z")


def _identifier(value: object, field: str) -> str:
    if not isinstance(value, str) or not _ID.fullmatch(value):
        raise AssistError(f"{field}不正确")
    return value


class ObjectKnowledgeStore:
    """Persist only confirmed names and bounded source references, with undo."""

    def __init__(self, state_dir: Path):
        self.directory = Path(state_dir).absolute()
        if self.directory.is_symlink() or self.directory.resolve() != self.directory:
            raise AssistError("物体记忆目录不可用", 500, "storage_unavailable")
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / "objects.json"
        self.lock = threading.RLock()
        self.records: list[dict] = []
        if self.path.is_file():
            if not inside_without_symlinks(self.path, self.directory):
                raise AssistError("物体记忆文件不可用", 500, "storage_unavailable")
            try:
                if self.path.stat().st_size > 8_000_000:
                    raise ValueError("oversized knowledge")
                data = json.loads(self.path.read_text(encoding="utf-8"))
                if data.get("schema_version") != 1 or not isinstance(data.get("records"), list):
                    raise ValueError("invalid knowledge")
                for row in data["records"]:
                    if not isinstance(row, dict) or not _ID.fullmatch(str(row.get("id", ""))):
                        raise ValueError("invalid record")
                self.records = data["records"]
            except (ValueError, TypeError, AttributeError, OSError) as error:
                # Preserve recoverable evidence in place. Do not silently replace
                # malformed records with an empty store during a new confirmation.
                raise AssistError(
                    "物体记忆无法读取，已保留原文件，请先修复", 500, "storage_invalid"
                ) from error

    def _save(self) -> None:
        _atomic_json(self.path, {"schema_version": 1, "records": self.records}, self.directory)

    def confirm(
        self,
        *,
        run_id: str,
        track_id: int | None,
        label: str,
        review: dict,
        original_label: str,
        source_video_id: str,
    ) -> dict:
        """Call only after the human has explicitly accepted/edited the name.

        A null track identifies an untracked manual object: a 2D annotation only,
        with no inferred geometry. Re-confirming the same track overlays previous
        confirmations; revoking it reveals the preceding active correction.
        """
        run_id = _identifier(run_id, "结果编号")
        source_video_id = _identifier(source_video_id, "源视频编号")
        if track_id is not None and (
            isinstance(track_id, bool) or not isinstance(track_id, int) or track_id < 0
        ):
            raise AssistError("物体轨迹编号不正确")
        label = _text(label, 100, "物体名称")
        original_label = _text(original_label, 160, "原始名称", empty=True)
        if label.lower() in {"unknown", "未知", "不确定", "无法判断"}:
            raise AssistError("物体仍不确定，不能作为已确认名称保存")
        if not isinstance(review, dict) or review.get("requires_confirmation") is not True:
            raise AssistError("请先完成三视图复核")
        review_id = _identifier(review.get("id"), "复核编号")
        selections = review.get("selections")
        if not isinstance(selections, list) or len(selections) != 3:
            raise AssistError("缺少三视图依据")
        evidence = []
        for selection in selections:
            if not isinstance(selection, dict):
                raise AssistError("三视图依据不正确")
            frame = selection.get("frame_id")
            box = selection.get("xyxy")
            digest = selection.get("image_sha256")
            if (
                isinstance(frame, bool)
                or not isinstance(frame, int)
                or frame < 0
                or not isinstance(box, list)
                or len(box) != 4
                or any(isinstance(v, bool) or not isinstance(v, int) for v in box)
                or box[0] < 0
                or box[1] < 0
                or box[2] <= box[0]
                or box[3] <= box[1]
                or not isinstance(digest, str)
                or not re.fullmatch(r"[a-f0-9]{64}", digest)
            ):
                raise AssistError("三视图依据不正确")
            evidence.append(
                {
                    "frame_id": frame,
                    "xyxy": list(box),
                    "image_sha256": digest,
                    **{
                        name: selection[name]
                        for name in (
                            "timestamp_seconds",
                            "source_width",
                            "source_height",
                            "width",
                            "height",
                            "image_bytes",
                        )
                        if name in selection
                    },
                }
            )
        if len({row["frame_id"] for row in evidence}) != 3:
            raise AssistError("需要三个不同源帧作为依据")
        provider = review.get("provider", {})
        base = normalize_base_url(provider.get("base_url"))[0]
        model = _text(provider.get("model"), 160, "模型名称")
        record = {
            "id": "object_" + secrets.token_hex(12),
            "created_at": _now(),
            "run_id": run_id,
            "track_id": track_id,
            "source_video_id": source_video_id,
            "label": label,
            "original_label": original_label,
            "review_id": review_id,
            "selections": evidence,
            "provider": {"base_url": base, "model": model},
            "source": "human_confirmed_llm_assistance",
            "revoked_at": None,
            "scope": "this_object_in_this_video",
            "model_training_performed": False,
        }
        with self.lock:
            if len(self.records) >= 5000:
                raise AssistError("物体记忆记录已达上限，请先导出整理", 409, "storage_limit")
            self.records.append(record)
            try:
                self._save()
            except BaseException:
                self.records.pop()
                raise
        return copy.deepcopy(record)

    def get(self, correction_id: str) -> dict:
        correction_id = _identifier(correction_id, "修正编号")
        with self.lock:
            for row in self.records:
                if row["id"] == correction_id:
                    return copy.deepcopy(row)
        raise AssistError("物体修正记录不存在", 404, "not_found")

    def list_for_run(self, run_id: str, *, include_revoked: bool = False) -> list[dict]:
        run_id = _identifier(run_id, "结果编号")
        with self.lock:
            return copy.deepcopy(
                [
                    row
                    for row in self.records
                    if row.get("run_id") == run_id
                    and (include_revoked or not row.get("revoked_at"))
                ]
            )

    def revoke(self, correction_id: str) -> dict:
        correction_id = _identifier(correction_id, "修正编号")
        with self.lock:
            for row in self.records:
                if row["id"] != correction_id:
                    continue
                previous = row.get("revoked_at")
                row["revoked_at"] = previous or _now()
                try:
                    self._save()
                except BaseException:
                    row["revoked_at"] = previous
                    raise
                return copy.deepcopy(row)
        raise AssistError("物体修正记录不存在", 404, "not_found")

    def delete_for_video(self, source_video_id: str) -> int:
        """Forget evidence only as part of an explicitly confirmed video purge."""
        source_video_id = _identifier(source_video_id, "源视频编号")
        with self.lock:
            previous = self.records
            self.records = [
                row for row in previous if row.get("source_video_id") != source_video_id
            ]
            removed = len(previous) - len(self.records)
            if removed:
                try:
                    self._save()
                except BaseException:
                    self.records = previous
                    raise
            return removed
