"""Build staging rejects unreviewed assets before touching the output directory."""

from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "packaging" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def staging(tmp_path):
    module = load_script("prepare_bundle")
    module.ROOT = tmp_path
    module.RESOURCES = tmp_path / "work/windows/resources"
    model = tmp_path / "models" / module.MODELS[0]
    model.parent.mkdir()
    model.write_bytes(b"model")
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    digest = hashlib.sha256(model.read_bytes()).hexdigest()
    (scripts / "download_yolox.py").write_text(f"SIZE = 5\nSHA256 = {digest!r}\n", encoding="utf-8")
    return module, model


def test_staging_accepts_verified_model_and_only_whitelisted_assets(staging):
    module, _ = staging
    module.validate_inputs()
    target = module.RESOURCES / "configs/spatial_consistent.yaml"
    target.parent.mkdir(parents=True)
    target.write_text("{}", encoding="utf-8")
    module.validate_inputs()


@pytest.mark.parametrize("content", [b"wrong", b"shorter-or-longer"])
def test_changed_weights_are_rejected_before_staging_or_license_writes(staging, content):
    module, model = staging
    model.write_bytes(content)
    with pytest.raises(ValueError, match="size/SHA-256"):
        module.main()
    assert not module.RESOURCES.exists()
    assert not (module.ROOT / "packaging").exists()
    assert model.read_bytes() == content


def test_leftover_extension_model_is_rejected_without_deleting_it(staging):
    module, _ = staging
    leftover = module.RESOURCES / "models/unreviewed.onnx"
    leftover.parent.mkdir(parents=True)
    leftover.write_bytes(b"unreviewed asset")
    with pytest.raises(ValueError, match="Unexpected staged asset"):
        module.main()
    assert leftover.read_bytes() == b"unreviewed asset"


def test_freeze_only_does_not_require_installer_tools(tmp_path, monkeypatch):
    module = load_script("build_windows")
    for name, path in {
        "ROOT": tmp_path,
        "OUT": tmp_path / "outputs",
        "RELEASE": tmp_path / "release",
    }.items():
        monkeypatch.setattr(module, name, path)
    calls = []
    monkeypatch.setattr(module.subprocess, "run", lambda command, **kw: calls.append(command))
    monkeypatch.setattr(sys, "argv", ["build_windows.py", "--freeze-only"])
    module.main()
    assert len(calls) == 2
    assert calls[0][-1].endswith("prepare_bundle.py")
    assert "PyInstaller" in calls[1]


@pytest.mark.parametrize(
    "arguments", [[], ["--installer-only"], ["--freeze-only", "--installer-only"]]
)
def test_invalid_build_modes_fail_before_writing(monkeypatch, arguments):
    module = load_script("build_windows")
    monkeypatch.setattr(sys, "argv", ["build_windows.py", *arguments])
    with pytest.raises(SystemExit) as error:
        module.main()
    assert error.value.code == 2


def test_frozen_smoke_rejects_worker_failure(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "smoke_windows_bundle", ROOT / "scripts/smoke_windows_bundle.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda *a, **kw: SimpleNamespace(returncode=1, stdout=b"worker failed"),
    )
    with pytest.raises(RuntimeError, match="unexpected exit code"):
        module.run(tmp_path / "app.exe", ["--worker"], tmp_path, "worker")
    assert (tmp_path / "worker.log").read_bytes() == b"worker failed"
