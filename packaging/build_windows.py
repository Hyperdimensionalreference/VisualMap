"""Reproducible local Windows build; no personal workspace folders are packaged."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "work/windows"
DIST = WORK / "dist/VisualMap"
OUT = ROOT / "outputs/windows"
RELEASE = ROOT / "release"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--freeze-only", action="store_true")
    mode.add_argument("--installer-only", action="store_true")
    parser.add_argument("--makensis", type=Path)
    parser.add_argument("--legal-dir", type=Path)
    args = parser.parse_args()
    if not args.freeze_only:
        if args.makensis is None or args.legal_dir is None:
            parser.error("installer builds require --makensis and --legal-dir")
        args.makensis = args.makensis.resolve()
        args.legal_dir = args.legal_dir.resolve()
    OUT.mkdir(parents=True, exist_ok=True)
    RELEASE.mkdir(exist_ok=True)
    if not args.installer_only:
        subprocess.run([sys.executable, str(ROOT / "packaging/prepare_bundle.py")], check=True)
        with (OUT / "pyinstaller.log").open("w", encoding="utf-8") as log:
            subprocess.run(
                [
                    sys.executable,
                    "-X",
                    "utf8",
                    "-m",
                    "PyInstaller",
                    "--noconfirm",
                    "--clean",
                    "--distpath",
                    str(WORK / "dist"),
                    "--workpath",
                    str(WORK / "pyi-build"),
                    str(ROOT / "packaging/VisualMap.spec"),
                ],
                cwd=ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
    if args.freeze_only:
        return
    for source, target in [
        (args.legal_dir / "licenses", DIST / "licenses"),
        (ROOT / "LICENSE", DIST / "LICENSE.txt"),
        (ROOT / "packaging/DISTRIBUTION_LICENSE.txt", DIST / "DISTRIBUTION_LICENSE.txt"),
        (ROOT / "packaging/README_FIRST.txt", DIST / "README_FIRST.txt"),
        (ROOT / "docs/WINDOWS_GUIDE.md", DIST / "WINDOWS_GUIDE.md"),
        (ROOT / "packaging/THIRD_PARTY_NOTICES.md", DIST / "THIRD_PARTY_NOTICES.txt"),
        (WORK / "VisualMap.ico", DIST / "VisualMap.ico"),
    ]:
        if source.is_dir():
            shutil.copytree(source, target, dirs_exist_ok=True)
        else:
            shutil.copy2(source, target)
    legal = json.loads((args.legal_dir / "legal_sources.json").read_text("utf-8"))
    for row in legal["license_and_source_files"]:
        assert (
            hashlib.sha256((DIST / "licenses" / row["path"]).read_bytes()).hexdigest()
            == row["sha256"]
        )
    legal["final_distribution_files"] = [
        {
            "path": p.relative_to(DIST).as_posix(),
            "bytes": p.stat().st_size,
            "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
        }
        for p in sorted(DIST.rglob("*"))
        if p.is_file() and p.name not in {"license_manifest.json", "bundle_manifest.json"}
    ]
    (DIST / "license_manifest.json").write_text(json.dumps(legal, indent=2), encoding="utf-8")
    manifest_path = DIST / "bundle_manifest.json"
    files = sorted(path for path in DIST.rglob("*") if path.is_file() and path != manifest_path)
    records = []
    for path in files:
        relative = path.relative_to(DIST).as_posix()
        assert ".." not in relative.split("/") and not path.is_symlink()
        records.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    manifest_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    files.append(manifest_path)
    # Delete only known application files on uninstall, preserving extra user files
    # and always preserving LocalAppData\VisualMap personal data.
    deletes = []
    for path in files:
        relative = str(path.relative_to(DIST)).replace("$", "$$").replace('"', '$\\"')
        deletes.append(f'Delete "$INSTDIR\\{relative}"')
    directories = sorted(
        (path for path in DIST.rglob("*") if path.is_dir()),
        key=lambda path: len(path.parts),
        reverse=True,
    )
    deletes.extend(f'RMDir "$INSTDIR\\{path.relative_to(DIST)}"' for path in directories)
    delete_manifest = WORK / "installer_delete.nsh"
    delete_manifest.write_text("\n".join(deletes) + "\n", encoding="utf-8-sig")
    target = RELEASE / "VisualMap-0.1.0-Windows-x64-Setup.exe"
    compiler = args.makensis
    with (OUT / "installer_build.log").open("w", encoding="utf-8") as log:
        subprocess.run(
            [
                str(compiler),
                "/INPUTCHARSET",
                "UTF8",
                "/V3",
                "/DBUNDLE=" + str(DIST),
                "/DOUTPUT=" + str(target),
                "/DDELETE_MANIFEST=" + str(delete_manifest),
                str(ROOT / "packaging/installer.nsi"),
            ],
            cwd=ROOT / "packaging",
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
            env={**os.environ, "NSISDIR": str(compiler.parent.parent)},
        )
    result = {
        "installer": str(target.relative_to(ROOT)),
        "bytes": target.stat().st_size,
        "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        "installed_files": len(files),
        "installed_bytes": sum(path.stat().st_size for path in files),
    }
    (OUT / "build_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
