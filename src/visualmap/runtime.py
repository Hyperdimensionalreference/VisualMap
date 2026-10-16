"""Locations and process dispatch shared by source and installed Windows builds.

Packaged files are immutable resources. Videos, results and credentials belong to
the user's data directory and are never discovered in the installation directory.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import TextIO


def is_frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


def resource_root() -> Path:
    """PyInstaller places explicitly bundled configs/models under ``_MEIPASS``."""
    if is_frozen():
        return Path(sys._MEIPASS).resolve()
    return Path(__file__).resolve().parents[2]


def data_root(directory: str | Path | None = None) -> Path:
    if directory is not None:
        return Path(directory).expanduser().resolve()
    if not is_frozen():
        return resource_root()
    local = os.environ.get("LOCALAPPDATA")
    base = Path(local) if local else Path.home() / ".local" / "share"
    return (base / "VisualMap").resolve()


def worker_command() -> list[str]:
    """A frozen executable must dispatch workers itself, never invoke Python."""
    executable = Path(sys.executable)
    if is_frozen():
        return [str(executable), "--worker"]
    console_python = executable.with_name("python.exe")
    if executable.name.lower() == "pythonw.exe" and console_python.is_file():
        executable = console_python
    return [str(executable), "-u", "-X", "utf8", "-m", "visualmap.app"]


def worker_environment() -> dict[str, str]:
    environment = dict(os.environ)
    environment["PYTHONUTF8"] = "1"
    environment["PYTHONUNBUFFERED"] = "1"
    if not is_frozen():
        source = str(resource_root() / "src")
        existing = environment.get("PYTHONPATH")
        environment["PYTHONPATH"] = source + (os.pathsep + existing if existing else "")
    return environment


def resource_path_overrides(config_path: Path, resources: Path) -> dict[str, str]:
    """Resolve only declared resource inputs, leaving video/output paths alone."""
    import yaml

    loaded = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"config root must be a mapping: {config_path}")
    keys = (
        "camera.intrinsics_file",
        "detection.model_path",
        "detection.open_vocabulary.model_path",
        "detection.open_vocabulary.vocabulary_path",
        "detection.open_vocabulary.tokenizer_path",
        "detection.open_vocabulary.prompts_path",
        "detection.open_vocabulary.manifest_path",
        "detection.open_vocabulary.runtime_path",
        "detection.open_vocabulary.crop_verifier.model_path",
        "detection.open_vocabulary.crop_verifier.vocabulary_path",
        "detection.open_vocabulary.crop_verifier.runtime_path",
    )
    overrides = {}
    for key in keys:
        value = loaded
        for part in key.split("."):
            value = value.get(part) if isinstance(value, dict) else None
        if value:
            path = Path(os.path.expandvars(str(value))).expanduser()
            overrides[key] = str(path if path.is_absolute() else (resources / path).resolve())
    return overrides


def workspace_identity(directory: Path, resources: Path) -> str:
    """Distinguish installations/data stores without publishing a local path."""
    import hashlib

    paths = [os.path.normcase(str(path.resolve())) for path in (directory, resources)]
    return hashlib.sha256("\0".join(paths).encode("utf-8")).hexdigest()


def _loopback_listener_present(port: int) -> bool | None:
    """Read Windows' TCP table: a closed loopback port may silently time out."""
    if sys.platform != "win32":
        return None
    import ctypes
    import socket
    from ctypes import wintypes

    class TCPRow(ctypes.Structure):
        _fields_ = [
            (name, wintypes.DWORD)
            for name in ("state", "local_address", "local_port", "remote_address", "remote_port")
        ]

    get_table = ctypes.WinDLL("iphlpapi", use_last_error=True).GetTcpTable
    get_table.argtypes = [ctypes.c_void_p, ctypes.POINTER(wintypes.DWORD), wintypes.BOOL]
    get_table.restype = wintypes.DWORD
    size = wintypes.DWORD()
    if get_table(None, ctypes.byref(size), False) != 122:
        return None
    buffer = ctypes.create_string_buffer(size.value)
    if get_table(buffer, ctypes.byref(size), False) != 0:
        return None
    count = wintypes.DWORD.from_buffer(buffer).value
    offset, row_size = ctypes.sizeof(wintypes.DWORD), ctypes.sizeof(TCPRow)
    if offset + count * row_size > size.value:
        return None
    loopback = int.from_bytes(socket.inet_aton("127.0.0.1"), sys.byteorder)
    for index in range(count):
        row = TCPRow.from_buffer(buffer, offset + index * row_size)
        if (
            row.state == 2
            and row.local_address in (0, loopback)
            and socket.ntohs(row.local_port & 0xFFFF) == port
        ):
            return True
    return False


def stop_workspace(directory: Path, port: int) -> int:
    """Gracefully stop only the matching idle installation for upgrade/uninstall."""
    import json
    import re
    import time
    from urllib.error import HTTPError, URLError
    from urllib.request import Request, urlopen

    from visualmap import __version__

    origin = f"http://127.0.0.1:{port}"
    if _loopback_listener_present(port) is False:
        return 0
    try:
        with urlopen(origin + "/api/info?runtime=1", timeout=2) as response:
            info = json.loads(response.read(4096))
    except URLError as exc:
        if isinstance(exc.reason, ConnectionRefusedError):
            return 0
        print("无法确认 VisualMap 服务状态，请稍后重试。", file=sys.stderr)
        return 1
    except (OSError, ValueError):
        print("无法确认 VisualMap 服务状态，请稍后重试。", file=sys.stderr)
        return 1
    expected = {
        "application": "VisualMap",
        "protocol": 1,
        "version": __version__,
        "workspace": workspace_identity(directory, resource_root()),
    }
    if not isinstance(info, dict) or any(info.get(key) != value for key, value in expected.items()):
        print("该端口属于另一个工作区，已保留其运行状态。", file=sys.stderr)
        return 3
    try:
        with urlopen(origin, timeout=2) as response:
            page = response.read(2 * 1024**2).decode("utf-8")
        token = re.search(r"const VM_TOKEN='([A-Za-z0-9_-]{20,100})';", page)
        if token is None:
            print("无法验证本机工作区控制凭据。", file=sys.stderr)
            return 1
        request = Request(
            origin + "/api/shutdown",
            data=b"{}",
            method="POST",
            headers={
                "X-VisualMap-Token": token[1],
                "Origin": origin,
                "Content-Type": "application/json",
            },
        )
        with urlopen(request, timeout=3) as response:
            if response.status != 202:
                return 1
    except HTTPError as exc:
        if exc.code == 409:
            print("工作区正在识别或复核画面，请任务完成后再退出或卸载。", file=sys.stderr)
            return 4
        print("工作区未接受安全退出请求。", file=sys.stderr)
        return 1
    except (OSError, ValueError):
        print("工作区退出请求未完成，请稍后重试。", file=sys.stderr)
        return 1
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if _loopback_listener_present(port) is False:
            return 0
        try:
            with urlopen(origin + "/api/info?runtime=1", timeout=0.5) as response:
                response.read(4096)
        except URLError as exc:
            if isinstance(exc.reason, (ConnectionRefusedError, ConnectionResetError)):
                return 0
        except ConnectionResetError:
            return 0
        except OSError:
            pass
        time.sleep(0.1)
    print("工作区仍在关闭，请稍后重试。", file=sys.stderr)
    return 1


def self_check(directory: Path | None = None) -> dict:
    """Inspect bundled resources and native imports; never open a user's library."""
    import importlib

    from visualmap import __version__

    resources = resource_root()
    config = resources / "configs/spatial_consistent.yaml"
    errors = []
    files = {}
    try:
        inputs = resource_path_overrides(config, resources)
        for name, filename in inputs.items():
            if not name.endswith("runtime_path"):
                files[name] = Path(filename).is_file()
        # Source mode can use the explicitly configured isolated runtime as well.
        runtime = inputs.get("detection.open_vocabulary.runtime_path")
        if runtime and Path(runtime).is_dir() and runtime not in sys.path:
            sys.path.insert(0, runtime)
    except (OSError, ValueError) as exc:
        errors.append(type(exc).__name__ + ": bundled config is unavailable or invalid")
    assets = Path(__file__).parent / "visualization"
    for name in (
        "upload.html",
        "workspace.css",
        "workspace.js",
        "viewer.html",
        "viewer.css",
        "viewer.js",
        "viewer_motion.js",
        "viewer_occupancy.js",
        "fonts/Geist-Variable.woff2",
    ):
        files["visualization/" + name] = (assets / name).is_file()
    versions = {}
    for name in ("numpy", "cv2", "yaml", "onnxruntime", "tokenizers"):
        try:
            module = importlib.import_module(name)
            versions[name] = getattr(module, "__version__", "available")
        except (ImportError, OSError) as exc:
            errors.append(name + ": " + type(exc).__name__)
    return {
        "application": "VisualMap",
        "version": __version__,
        "frozen": is_frozen(),
        "resource_root": str(resources),
        "data_root": str(directory or data_root()),
        "resources": files,
        "runtime_versions": versions,
        "errors": errors,
        "ok": bool(files) and all(files.values()) and not errors,
    }


def _inherited_stream(handle_number: int) -> TextIO | None:
    """Restore a worker pipe in a PyInstaller ``--windowed`` executable."""
    if sys.platform != "win32":
        return None
    import ctypes
    import msvcrt
    from ctypes import wintypes

    kernel = ctypes.windll.kernel32
    get_handle = kernel.GetStdHandle
    get_handle.argtypes = [wintypes.DWORD]
    get_handle.restype = wintypes.HANDLE
    handle = get_handle(handle_number & 0xFFFFFFFF)
    if handle in (None, 0, ctypes.c_void_p(-1).value):
        return None
    current_process = kernel.GetCurrentProcess
    current_process.restype = wintypes.HANDLE
    duplicate_handle = kernel.DuplicateHandle
    duplicate_handle.argtypes = [
        wintypes.HANDLE,
        wintypes.HANDLE,
        wintypes.HANDLE,
        ctypes.POINTER(wintypes.HANDLE),
        wintypes.DWORD,
        wintypes.BOOL,
        wintypes.DWORD,
    ]
    duplicate_handle.restype = wintypes.BOOL
    duplicate = wintypes.HANDLE()
    process = current_process()
    if not duplicate_handle(process, handle, process, ctypes.byref(duplicate), 0, False, 2):
        return None
    try:
        descriptor = msvcrt.open_osfhandle(duplicate.value, os.O_WRONLY | os.O_BINARY)
        return open(descriptor, "w", buffering=1, encoding="utf-8", errors="replace", closefd=True)
    except OSError:
        close_handle = kernel.CloseHandle
        close_handle.argtypes = [wintypes.HANDLE]
        close_handle(duplicate)
        return None


def configure_windowed_streams(
    directory: Path, *, worker: bool = False, write_log: bool = True
) -> None:
    """Keep GUI diagnostics on disk and preserve streamed worker progress."""
    if sys.stdout is not None and sys.stderr is not None:
        return
    fallback = None
    for name, handle in (("stdout", -11), ("stderr", -12)):
        if getattr(sys, name) is not None:
            continue
        stream = _inherited_stream(handle) if worker else None
        if stream is None:
            if fallback is None:
                if write_log:
                    logs = directory / "logs"
                    logs.mkdir(parents=True, exist_ok=True)
                    fallback = (logs / "workspace.log").open(
                        "a", encoding="utf-8", buffering=1, errors="replace"
                    )
                else:
                    fallback = open(os.devnull, "w", encoding="utf-8")
            stream = fallback
        setattr(sys, name, stream)
