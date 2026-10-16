"""Generate a textured sample video with strong corners (better for ORB)."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def main() -> None:
    out_dir = Path(__file__).resolve().parents[1] / "sample_data"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "sample.avi"
    w, h, n, fps = 320, 240, 60, 15

    scene_w, scene_h = 720, 400
    scene = np.full((scene_h, scene_w, 3), 50, dtype=np.uint8)
    rng = np.random.default_rng(7)

    # Dense random blocks + circles = ORB-friendly texture
    for _ in range(220):
        x = int(rng.integers(0, scene_w - 36))
        y = int(rng.integers(0, scene_h - 36))
        color = tuple(int(c) for c in rng.integers(0, 255, size=3))
        x2 = x + int(rng.integers(12, 36))
        y2 = y + int(rng.integers(12, 36))
        cv2.rectangle(scene, (x, y), (x2, y2), color, -1)
        cv2.circle(scene, (x + 18, y + 18), int(rng.integers(4, 12)), (255, 255, 255), -1)
        cv2.line(scene, (x, y), (x + 30, y + 30), (0, 0, 0), 2)

    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    vw = cv2.VideoWriter(str(path), fourcc, fps, (w, h))
    if not vw.isOpened():
        raise SystemExit("VideoWriter failed to open")

    for i in range(n):
        ox = int(16 + i * 3.5)
        oy = int(12 + i * 1.2)
        ox = min(ox, scene_w - w - 1)
        oy = min(oy, scene_h - h - 1)
        crop = scene[oy : oy + h, ox : ox + w].copy()
        cv2.putText(crop, f"F{i}", (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        vw.write(crop)
    vw.release()
    print(f"wrote {path} ({path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
