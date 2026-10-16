"""Local publication checks. Findings never print the suspected secret value."""

from __future__ import annotations

import argparse
import getpass
import hashlib
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PRIVATE_DIRS = {
    ".git",
    ".venv",
    "venv",
    "outputs",
    "work",
    "release",
    "vendor",
    ".codex",
    ".agents",
    ".aws",
    ".ssh",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    "build",
    "dist",
}
BINARY_SUFFIXES = {
    ".exe",
    ".dll",
    ".pyd",
    ".pyc",
    ".zip",
    ".gz",
    ".xz",
    ".7z",
    ".onnx",
    ".pt",
    ".pth",
    ".npz",
    ".npy",
    ".db",
    ".sqlite",
    ".log",
    ".mp4",
    ".mov",
    ".avi",
    ".mkv",
    ".webm",
    ".mp3",
    ".wav",
    ".jpg",
    ".jpeg",
    ".png",
    ".pem",
    ".key",
    ".p12",
    ".pfx",
}
FONT = "src/visualmap/visualization/fonts/Geist-Variable.woff2"
FONT_SHA256 = "2ffebe993e969069a9789d15164b7715d42491b5835516c5e3b935d5f81b05f1"
RULES = {
    "private_key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----"),
    "github_token": re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{36,}|github_pat_[A-Za-z0-9_]{40,})\b"),
    "service_key": re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_-]{24,}\b"),
    "aws_key": re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
    "slack_token": re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{20,}\b"),
    "personal_windows_path": re.compile(r"(?i)\b[A-Z]:[\\/]+Users[\\/]+[A-Za-z0-9_.-]+"),
    "private_phone_filename": re.compile(r"\bVID_\d{8}_\d{6}\b"),
}
EMAIL = re.compile(r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b")


def inspect(name: str, data: bytes) -> list[dict]:
    """Inspect a relative path and its exact staged bytes, without network calls."""
    path = Path(name)
    findings = []
    if path.is_absolute() or ".." in path.parts or set(path.parts) & PRIVATE_DIRS:
        findings.append({"file": name, "rule": "private_directory_or_path"})
    if path.parts[0] in {"models", "sample_data"} and path.name != "README.md":
        findings.append({"file": name, "rule": "local_asset"})
    if path.suffix.lower() in BINARY_SUFFIXES or path.name.startswith(".env"):
        findings.append({"file": name, "rule": "private_or_binary_artifact"})
    if name == FONT:
        if hashlib.sha256(data).hexdigest() != FONT_SHA256:
            findings.append({"file": name, "rule": "invalid_font"})
        return findings
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError:
        return findings + [{"file": name, "rule": "unexpected_binary"}]
    for rule, pattern in RULES.items():
        for match in pattern.finditer(text):
            findings.append(
                {"file": name, "line": text.count("\n", 0, match.start()) + 1, "rule": rule}
            )
    for match in EMAIL.finditer(text):
        domain = match[0].split("@", 1)[1].lower()
        if domain in {"example.com", "example.org", "example.net", "example.test"} or (
            domain.endswith(".example.com")
            or domain.endswith(".users.noreply.github.com")
            or domain == "users.noreply.github.com"
        ):
            continue
        findings.append(
            {
                "file": name,
                "line": text.count("\n", 0, match.start()) + 1,
                "rule": "email_requires_publication_review",
            }
        )
    username = getpass.getuser()
    if len(username) >= 5 and re.search(
        r"(?i)(?<![\w-])" + re.escape(username) + r"(?![\w-])", text
    ):
        findings.append({"file": name, "rule": "current_machine_username"})
    return findings


def git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staged", action="store_true", help="Inspect exact Git index bytes")
    args = parser.parse_args()
    findings = []
    if (ROOT / ".git").exists():
        names = git("ls-files", "--cached", "-z").stdout.split(b"\0")
        if not args.staged:
            names += git("ls-files", "--others", "--exclude-standard", "-z").stdout.split(b"\0")
        files = sorted({name.decode("utf-8") for name in names if name})
    else:
        if args.staged:
            parser.error("--staged requires a Git repository")
        files = [
            p.relative_to(ROOT).as_posix()
            for p in ROOT.rglob("*")
            if p.is_file() and not set(p.relative_to(ROOT).parts) & PRIVATE_DIRS
        ]
    for name in files:
        path = ROOT / name
        if path.is_symlink() or not path.resolve().is_relative_to(ROOT):
            findings.append({"file": name, "rule": "symbolic_link"})
            continue
        data = git("show", ":" + name).stdout if args.staged else path.read_bytes()
        findings += inspect(name, data)
    print(
        json.dumps(
            {"checked_files": len(files), "findings": findings, "all_passed": not findings},
            ensure_ascii=False,
            indent=2,
        )
    )
    return int(bool(findings))


if __name__ == "__main__":
    raise SystemExit(main())
