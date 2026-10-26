"""Stage an explicit, private-data-free set of assets for the Windows build."""

from __future__ import annotations

import hashlib
import json
import runpy
import shutil
import struct
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
RESOURCES = ROOT / "work/windows/resources"
MODELS = ["object_detection_yolox_2022nov.onnx"]


def validate_inputs():
    """Reject changed weights and leftover assets before modifying the staging tree."""
    download = runpy.run_path(str(ROOT / "scripts/download_yolox.py"))
    source = ROOT / "models" / MODELS[0]
    if (
        source.is_symlink()
        or not source.is_file()
        or source.stat().st_size != download["SIZE"]
        or hashlib.sha256(source.read_bytes()).hexdigest() != download["SHA256"]
    ):
        raise ValueError("YOLOX model failed the published size/SHA-256 check")
    allowed = {"configs/spatial_consistent.yaml", "configs/grounding_prompts.json"}
    allowed.update("models/" + name for name in MODELS)
    if RESOURCES.is_symlink() or not RESOURCES.resolve().is_relative_to(
        (ROOT / "work/windows").resolve()
    ):
        raise ValueError("Resource staging must remain inside the build directory")
    for path in RESOURCES.rglob("*"):
        if path.is_symlink() or (
            path.is_file() and path.relative_to(RESOURCES).as_posix() not in allowed
        ):
            raise ValueError("Unexpected staged asset; use a fresh build directory")


def icon():
    """Original cube icon, drawn as vectors into native Windows bitmap entries."""
    vertices = [
        (0.5, 0.18),
        (0.78, 0.34),
        (0.78, 0.66),
        (0.5, 0.82),
        (0.22, 0.66),
        (0.22, 0.34),
        (0.5, 0.50),
    ]
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0), (1, 6), (3, 6), (5, 6)]

    def distance(x, y, a, b):
        dx, dy = b[0] - a[0], b[1] - a[1]
        t = max(0, min(1, ((x - a[0]) * dx + (y - a[1]) * dy) / (dx * dx + dy * dy)))
        return ((x - a[0] - t * dx) ** 2 + (y - a[1] - t * dy) ** 2) ** 0.5

    images = []
    for size in (16, 32, 48, 256):
        pixels = bytearray()
        for yy in range(size - 1, -1, -1):
            for xx in range(size):
                x, y = (xx + 0.5) / size, (yy + 0.5) / size
                color = (37, 43, 40, 255)
                if min(x, 1 - x) < 0.035 and min(y, 1 - y) < 0.035:
                    color = (0, 0, 0, 0)
                elif any(
                    distance(x, y, vertices[a], vertices[b]) < max(0.012, 0.6 / size)
                    for a, b in edges
                ):
                    color = (241, 236, 221, 255)
                if ((x - 0.5) ** 2 + (y - 0.5) ** 2) ** 0.5 < 0.035:
                    color = (129, 185, 159, 255)
                r, g, b, a = color
                pixels.extend((b, g, r, a))
        mask = bytes(((size + 31) // 32) * 4 * size)
        header = struct.pack(
            "<IIIHHIIIIII", 40, size, size * 2, 1, 32, 0, len(pixels) + len(mask), 0, 0, 0, 0
        )
        images.append((size, header + pixels + mask))
    offset = 6 + 16 * len(images)
    data = bytearray(struct.pack("<HHH", 0, 1, len(images)))
    for size, blob in images:
        data.extend(
            struct.pack("<BBBBHHII", size % 256, size % 256, 0, 0, 1, 32, len(blob), offset)
        )
        offset += len(blob)
    for _, blob in images:
        data.extend(blob)
    (ROOT / "work/windows/VisualMap.ico").write_bytes(data)


def main():
    validate_inputs()
    (ROOT / "packaging/DISTRIBUTION_LICENSE.txt").write_text(
        "VisualMap 与第三方组件的许可范围\n\n"
        "VisualMap 自有程序源代码采用下方 MIT 许可。本安装包还包含独立的第三方组件，"
        "这些组件继续遵循各自许可，并非整包统一采用 MIT。\n"
        "请阅读安装目录 THIRD_PARTY_NOTICES.txt 和 licenses 目录中的版权、许可、"
        "源码与修改说明；开源组件授予的权利不受其他附加限制。"
        "微软运行库保持微软许可，不属于开源软件。\n\n" + (ROOT / "LICENSE").read_text("utf-8"),
        encoding="utf-8-sig",
    )
    (RESOURCES / "configs").mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load((ROOT / "configs/spatial_consistent.yaml").read_text("utf-8"))
    config["video"]["path"] = "input.mp4"
    config["run_name"] = "visualmap"
    config["detection"]["open_vocabulary"] = {"enabled": False, "crop_verifier": {"enabled": False}}
    (RESOURCES / "configs/spatial_consistent.yaml").write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    shutil.copy2(
        ROOT / "configs/grounding_prompts.json", RESOURCES / "configs/grounding_prompts.json"
    )
    records = []
    for name in MODELS:
        source = ROOT / "models" / name
        target = RESOURCES / "models" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        assert hashlib.sha256(target.read_bytes()).hexdigest() == digest
        records.append({"file": "models/" + name, "bytes": source.stat().st_size, "sha256": digest})
    out = ROOT / "outputs/windows"
    out.mkdir(parents=True, exist_ok=True)
    (out / "model_inventory.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    icon()
    print(
        json.dumps(
            {"required_model_bytes": sum(row["bytes"] for row in records), "files": len(records)}
        )
    )


if __name__ == "__main__":
    main()
