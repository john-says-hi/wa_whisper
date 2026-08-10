"""Tests for the Wayland text-injection path."""

import importlib
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

main_mod = importlib.import_module("wa_whisper.main")

InjectionMode = main_mod.InjectionMode


def test_wtype_is_a_recognised_mode():
    assert main_mod.parse_injection_mode("wtype") is InjectionMode.WTYPE


def test_typing_invokes_wtype_with_end_of_options(monkeypatch, tmp_path):
    """``--`` is load-bearing: wtype parses a leading dash as its own option.

    A transcript starting with "- " would otherwise be swallowed with
    "Missing argument to -...", losing the dictation silently.
    """
    calls = []

    monkeypatch.setattr(main_mod.shutil, "which", lambda name: "/usr/bin/wtype")
    monkeypatch.setattr(
        main_mod.subprocess,
        "run",
        lambda cmd, **kw: calls.append(cmd) or subprocess.CompletedProcess(cmd, 0),
    )

    assert main_mod.type_text_with_wtype("- dashed transcript", tmp_path / "log.txt") is True
    assert calls == [["/usr/bin/wtype", "--", "- dashed transcript"]]


def test_missing_wtype_binary_reports_rather_than_raises(monkeypatch, tmp_path):
    monkeypatch.setattr(main_mod.shutil, "which", lambda name: None)
    log = tmp_path / "log.txt"

    assert main_mod.type_text_with_wtype("hello", log) is False
    assert "wtype not found" in log.read_text()


def test_wtype_failure_is_logged_with_stderr(monkeypatch, tmp_path):
    monkeypatch.setattr(main_mod.shutil, "which", lambda name: "/usr/bin/wtype")

    def boom(cmd, **kw):
        raise subprocess.CalledProcessError(1, cmd, stderr=b"no virtual keyboard protocol")

    monkeypatch.setattr(main_mod.subprocess, "run", boom)
    log = tmp_path / "log.txt"

    assert main_mod.type_text_with_wtype("hello", log) is False
    assert "no virtual keyboard protocol" in log.read_text()


def test_inject_text_routes_wtype_mode_without_touching_xdotool(monkeypatch, tmp_path):
    """WTYPE must not fall through to xdotool, which cannot reach Wayland clients."""
    used = []

    def fail_xdotool(*_args, **_kwargs):
        pytest.fail("xdotool must not be used in wtype mode")

    monkeypatch.setattr(
        main_mod, "type_text_with_wtype", lambda text, log_path: used.append("wtype") or True
    )
    monkeypatch.setattr(main_mod, "type_text_with_xdotool", fail_xdotool)
    monkeypatch.setattr(main_mod, "play_completion_beep", lambda *_args: used.append("beep"))

    delivered = main_mod.inject_text(
        "hello",
        None,
        tmp_path / "log.txt",
        enable_beep=True,
        injection_mode=InjectionMode.WTYPE,
    )

    assert delivered is True
    assert used == ["wtype", "beep"]
