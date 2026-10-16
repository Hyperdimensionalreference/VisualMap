"""Safe output paths shared by the CLI and artifact exporters."""

from __future__ import annotations

from pathlib import Path


def validate_run_name(name: str) -> str:
    """Require one portable directory component for an output run."""
    if (
        not isinstance(name, str)
        or name in ("", ".", "..")
        or any(char in name for char in ("/", "\\", ":", "\0"))
    ):
        raise ValueError("run_name must be a single directory name")
    return name


def run_directory(output_dir: str | Path, run_name: str) -> Path:
    """Return a run path confined to output_dir, including existing links."""
    name = validate_run_name(run_name)
    root = Path(output_dir)
    run = root / name
    if not run.resolve().is_relative_to(root.resolve()):
        raise ValueError("run_name must stay within output directory")
    return run
