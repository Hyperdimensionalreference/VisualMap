"""Open the local workspace without leaving a console window on the desktop."""

from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path


def launch() -> None:
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    logs = root / "work" / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    with (logs / "workspace_startup.log").open("a", encoding="utf-8") as log:
        sys.stdout = sys.stderr = log
        try:
            from visualmap.web_app import main

            main()
        except (Exception, SystemExit) as exc:
            if isinstance(exc, SystemExit) and exc.code in (None, 0):
                return
            traceback.print_exc(file=log)
            log.flush()
            import tkinter as tk
            from tkinter import messagebox

            dialog = tk.Tk()
            dialog.withdraw()
            messagebox.showerror("VisualMap", "工作台未能启动。请确认本机运行环境可用后重试。")
            dialog.destroy()


if __name__ == "__main__":
    launch()
