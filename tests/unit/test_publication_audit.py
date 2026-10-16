"""The publication guard must reject sensitive artifacts without echoing values."""

import runpy
from pathlib import Path

AUDIT = runpy.run_path(str(Path(__file__).resolve().parents[2] / "scripts/audit_public_tree.py"))


def test_guard_rejects_sensitive_paths_and_binary_artifacts():
    inspect = AUDIT["inspect"]
    assert inspect("outputs/history.json", b"{}")
    assert inspect("scripts/probe.exe", b"MZ")
    assert inspect("models/private.onnx", b"model")
    assert inspect(".env", b"placeholder")


def test_guard_rejects_token_and_private_path_without_echoing_value():
    inspect = AUDIT["inspect"]
    token = "gh" + "p_" + "A" * 36
    rows = inspect("example.txt", token.encode())
    assert any(row["rule"] == "github_token" for row in rows)
    assert token not in str(rows)
    private_path = "C:" + "\\" + "Users" + "\\" + "someone" + "\\" + "record.txt"
    assert inspect("example.txt", private_path.encode())
    email = "private.person" + "@" + "personal-mail.invalid"
    assert inspect("example.txt", email.encode())


def test_guard_accepts_project_text_and_published_font_header():
    inspect = AUDIT["inspect"]
    assert not inspect("README.md", b"A local-first project with no credentials.")
    font = AUDIT["ROOT"] / AUDIT["FONT"]
    assert not inspect(AUDIT["FONT"], font.read_bytes())
    assert inspect(AUDIT["FONT"], b"wOF2unexpected-font-content")
