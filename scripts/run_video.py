"""Choose an input video, process it locally, and open the spatial result."""

from __future__ import annotations

import argparse
import os
import webbrowser
from datetime import datetime
from pathlib import Path

from visualmap.app import main as process_video


def main() -> int:
    parser = argparse.ArgumentParser(description="选择视频并生成三维空间视图")
    parser.add_argument("--video", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--no-open", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    video = args.video
    if video is None:
        try:
            import tkinter as tk
            from tkinter import filedialog

            dialog = tk.Tk()
            dialog.withdraw()
            selection = filedialog.askopenfilename(
                title="选择要识别的视频",
                filetypes=[("视频文件", "*.mp4 *.avi *.mov *.mkv *.webm"), ("所有文件", "*.*")],
            )
            dialog.destroy()
        except ImportError:
            parser.error("当前 Python 缺少文件选择器，请用 --video 指定视频路径")
        if not selection:
            return 0
        video = Path(selection)
    video = video.resolve()
    run_name = datetime.now().strftime("video_%Y%m%d_%H%M%S_%f")
    argv = [
        "--config", str(args.config or root / "configs/spatial_consistent.yaml"),
        "--video", str(video), "--output", str(root / "outputs"), "--run-name", run_name,
    ]
    if args.max_frames is not None:
        argv.extend(["--max-frames", str(args.max_frames)])
    code = process_video(argv)
    result = root / "outputs" / run_name / "semantic_map.html"
    if code == 0 and result.is_file():
        print(f"空间视图已生成：{result}")
        if not args.no_open:
            webbrowser.open(result.as_uri())
    return code


if __name__ == "__main__":
    raise SystemExit(main())
