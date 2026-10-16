"""Export depth maps: PNG (visual + 16-bit), NPY, and JSON summary."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from visualmap.depth.types import DepthResult
from visualmap.utils.logging import get_logger
from visualmap.utils.paths import run_directory

logger = get_logger(__name__)

__all__ = ["export_depth"]


def export_depth(
    result: DepthResult, output_dir: str | Path, run_name: str = "run"
) -> dict[str, Path]:
    """Write depth artifacts under outputs/<run_name>/depth/."""
    out = run_directory(output_dir, run_name) / "depth"
    out.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}

    index = []
    for i, dm in enumerate(result.maps):
        stem = f"depth_f{dm.frame_id:05d}"
        npy_path = out / f"{stem}.npy"
        png_path = out / f"{stem}.png"
        np.save(npy_path, dm.depth)
        _write_png_gray16(png_path, dm.depth)
        written[f"depth_npy_{i}"] = npy_path
        written[f"depth_png_{i}"] = png_path
        index.append(dm.to_dict())

    summary_path = out / "depth_summary.json"
    payload = {
        "summary": result.summary(),
        "frames": index,
        "point_depths_sample": {
            str(k): v for k, v in list(result.point_depths.items())[:50]
        },
    }
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    written["depth_summary"] = summary_path
    logger.info("Wrote %d depth maps under %s", len(result.maps), out)
    return written


def _write_png_gray16(path: Path, depth: np.ndarray) -> None:
    """16-bit PNG: normalized depth for visualization / inspection."""
    # Prefer imageio-free path via OpenCV
    import cv2

    valid = np.isfinite(depth) & (depth > 0)
    if not np.any(valid):
        img = np.zeros(depth.shape, dtype=np.uint16)
    else:
        vals = depth.copy()
        lo = float(vals[valid].min())
        hi = float(vals[valid].max())
        span = max(hi - lo, 1e-6)
        norm = np.zeros_like(vals, dtype=np.float32)
        norm[valid] = (vals[valid] - lo) / span
        # Near = bright for readability
        img = (norm * 65535.0).astype(np.uint16)
        img[~valid] = 0

    # OpenCV's filename APIs may silently fail on Unicode paths on Windows.
    # Encode in memory, then use Python's Unicode-aware filesystem handling.
    encoded_ok, encoded = cv2.imencode(".png", img)
    if not encoded_ok:
        raise OSError(f"Could not encode depth PNG: {path}")
    path.write_bytes(encoded.tobytes())
