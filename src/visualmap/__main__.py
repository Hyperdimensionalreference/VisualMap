"""Desktop executable entry point; ``--worker`` runs the real CLI in isolation."""

from __future__ import annotations

import multiprocessing
import sys
import traceback

from visualmap.runtime import configure_windowed_streams, data_root, is_frozen, self_check


def main(argv: list[str] | None = None) -> int:
    multiprocessing.freeze_support()
    arguments = list(sys.argv[1:] if argv is None else argv)
    worker = bool(arguments and arguments[0] == "--worker")
    informational = bool(
        arguments and arguments[0] in {"--self-check", "--version", "--help", "-h"}
    )
    # Select the log directory before importing modules which may log on startup.
    directory = None
    for index, argument in enumerate(arguments):
        if argument == "--data-dir" and index + 1 < len(arguments):
            directory = arguments[index + 1]
        elif argument.startswith("--data-dir="):
            directory = argument.partition("=")[2]
    try:
        configure_windowed_streams(
            data_root(directory), worker=worker or informational, write_log=not informational
        )
        if arguments and arguments[0] == "--self-check":
            import argparse
            import json
            from pathlib import Path

            parser = argparse.ArgumentParser(description="VisualMap 安装资源检测")
            parser.add_argument("--self-check", nargs="?", const="-", metavar="REPORT.json")
            parser.add_argument("--data-dir", type=Path)
            args = parser.parse_args(arguments)
            report = self_check(data_root(args.data_dir))
            output = json.dumps(report, ensure_ascii=False, indent=2)
            if args.self_check == "-":
                print(output, flush=True)
            else:
                Path(args.self_check).write_text(output + "\n", encoding="utf-8")
            return 0 if report["ok"] else 1
        if worker:
            from visualmap.app import main as worker_main

            return worker_main(arguments[1:])
        from visualmap.web_app import main as workspace_main

        workspace_main(arguments)
        return 0
    except (Exception, SystemExit) as exc:
        if isinstance(exc, SystemExit) and (exc.code is None or isinstance(exc.code, int)):
            return exc.code or 0
        traceback.print_exc()
        if is_frozen() and not worker and sys.platform == "win32":
            import ctypes

            ctypes.windll.user32.MessageBoxW(
                None,
                "VisualMap 启动未完成。请检查数据目录权限或端口是否被占用。\n"
                f"诊断日志：{data_root(directory) / 'logs' / 'workspace.log'}",
                "VisualMap",
                0x10,
            )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
