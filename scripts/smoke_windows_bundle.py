"""Exercise the real frozen Windows worker using only generated video and public weights.

Detailed reports stay in the ignored build directory; stdout contains no local paths.
This does not build, upload or certify an installer.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def run(executable: Path, arguments: list[str], directory: Path, label: str, expected=0):
    result = subprocess.run(
        [str(executable), *arguments],
        cwd=directory,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=180,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    (directory / (label + ".log")).write_bytes(result.stdout)
    if result.returncode != expected:
        raise RuntimeError(f"{label}: unexpected exit code {result.returncode}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=ROOT / "work/windows/dist/VisualMap")
    args = parser.parse_args()
    if sys.platform != "win32":
        parser.error("the frozen Windows smoke test requires Windows")
    bundle = args.bundle.resolve()
    executable = bundle / "VisualMap.exe"
    if not executable.is_file():
        parser.error("build the frozen application first")
    work = ROOT / "work/windows"
    work.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix="smoke-", dir=work))
    data = directory / "isolated-data"
    report = directory / "self-check.json"
    run(executable, ["--self-check", str(report), "--data-dir", str(data)], directory, "check")
    check = json.loads(report.read_text(encoding="utf-8"))
    if not check.get("ok") or not check.get("frozen") or data.exists():
        raise RuntimeError("Frozen self-check failed or created a user data directory")

    video = directory / "synthetic.avi"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 10, (320, 240))
    if not writer.isOpened():
        raise RuntimeError("Synthetic video encoder is unavailable")
    rng = np.random.default_rng(7)
    texture = rng.integers(0, 256, size=(240, 360, 3), dtype=np.uint8)
    try:
        for index in range(8):
            writer.write(texture[:, index * 3 : index * 3 + 320].copy())
    finally:
        writer.release()
    output = directory / "results"
    run(
        executable,
        [
            "--worker",
            "--video",
            str(video),
            "--output",
            str(output),
            "--data-dir",
            str(data),
            "--max-frames",
            "8",
            "--run-name",
            "synthetic",
        ],
        directory,
        "worker",
    )
    result_dir = output / "synthetic"
    result = json.loads((result_dir / "run_report.json").read_text(encoding="utf-8"))
    if result.get("frames_read") != 8:
        raise RuntimeError("Frozen worker did not process all eight generated frames")
    for filename in ("semantic_map.html", "semantics.json", "trajectory.csv", "map_points.ply"):
        if not (result_dir / filename).is_file():
            raise RuntimeError("Frozen worker is missing a required output: " + filename)

    invalid = directory / "invalid.avi"
    invalid.write_bytes(b"not a video")
    run(
        executable,
        [
            "--worker",
            "--video",
            str(invalid),
            "--output",
            str(output),
            "--data-dir",
            str(data),
            "--run-name",
            "invalid",
        ],
        directory,
        "invalid",
        expected=1,
    )
    print(
        json.dumps(
            {
                "frozen_self_check": True,
                "self_check_preserves_data_directory": True,
                "synthetic_frames_processed": 8,
                "required_outputs_present": True,
                "invalid_media_rejected": True,
                "runtime_versions": check["runtime_versions"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
