import importlib
import json
import queue
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from wa_whisper import focused_window, resolve_protection as protection
from wa_whisper.dictation_archive import DictationArchive
from wa_whisper.hotkeys import CaptureEndReason, CaptureResult
from wa_whisper.recovery_queue import RecoveryQueueResult

main = importlib.import_module("wa_whisper.main")


def window(app_id, activated, **overrides):
    return {"id": app_id + "-window", "app_id": app_id, "activated": activated,
            "state_known": True, **overrides}


@pytest.mark.parametrize("rows, expected", [
    ([window("dev.warp.Warp", True), window("resolve", False)], None),
    ([window("dev.warp.Warp", False), window("resolve", True)], protection.RESOLVE_FOCUSED),
    ([window("resolve", False), window("resolve", True, id="settings", title="Project Settings")],
     protection.RESOLVE_FOCUSED),
    ([window("dev.warp.Warp", True, title="DaVinci Resolve"), window("resolve", False)], None),
    ([window("google-chrome", True), window("resolve", False, state_known=False)], None),
    ([window("dev.warp.Warp", False), window("resolve", False)], protection.DETECTION_UNAVAILABLE),
    ([window("dev.warp.Warp", True), window("resolve", True)], protection.DETECTION_UNAVAILABLE),
    ([window("dev.warp.Warp", True, state_known=False)], protection.DETECTION_UNAVAILABLE),
    ([window("", True)], protection.DETECTION_UNAVAILABLE),
    ([], protection.DETECTION_UNAVAILABLE),
])
def test_only_keyboard_focused_application_controls_insertion(monkeypatch, rows, expected):
    monkeypatch.setattr(protection, "read_focused_window",
                        lambda: focused_window.focused_window_from_snapshot({"toplevels": rows}))
    assert protection.insertion_block_reason() == expected


@pytest.mark.parametrize("payload", [None, [], {}, {"toplevels": "invalid"}, {"toplevels": [None]}])
def test_malformed_inventory_cannot_allow_typing(payload):
    with pytest.raises(focused_window.FocusUnavailable):
        focused_window.focused_window_from_snapshot(payload)


def test_helper_reads_a_fresh_snapshot_with_a_timeout(monkeypatch, tmp_path):
    calls = []
    snapshot = {"toplevels": [window("dev.warp.Warp", True), window("resolve", False)]}

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(stdout=json.dumps(snapshot))

    monkeypatch.setattr(focused_window.subprocess, "run", run)
    assert focused_window.read_focused_window(tmp_path / "wa_toplevels").app_id == "dev.warp.Warp"
    assert calls[0][0] == [str(tmp_path / "wa_toplevels"), "list"]
    assert calls[0][1]["timeout"] == 1.0
    assert calls[0][1]["check"] is True


@pytest.mark.parametrize("error", [FileNotFoundError(), subprocess.TimeoutExpired("helper", 1),
                                   subprocess.CalledProcessError(1, "helper")])
def test_failed_helper_blocks_insertion(monkeypatch, error):
    def fail(*_args, **_kwargs):
        raise error

    monkeypatch.setattr(focused_window.subprocess, "run", fail)
    assert protection.insertion_block_reason() == protection.DETECTION_UNAVAILABLE


@pytest.mark.parametrize("mode", list(main.InjectionMode))
@pytest.mark.parametrize("reason", [protection.RESOLVE_FOCUSED, protection.DETECTION_UNAVAILABLE])
def test_every_delivery_mode_blocks_before_any_backend(tmp_path, monkeypatch, mode, reason):
    monkeypatch.setattr(protection, "insertion_block_reason", lambda: reason)

    def forbidden(*_args, **_kwargs):
        pytest.fail("No input backend may run when Resolve has focus or focus is unknown")

    for name in ("type_text_with_wtype", "type_text_with_xdotool", "inject_text_auto", "inject_text_orca_daemon"):
        monkeypatch.setattr(main, name, forbidden)
    with pytest.raises(protection.InsertionBlocked, match=reason):
        main.inject_text("delete this", Path("xdotool"), tmp_path / "log", enable_beep=True, injection_mode=mode)


@pytest.mark.parametrize("initial_app, final_app", [
    ("resolve", "dev.warp.Warp"),
    ("dev.warp.Warp", "resolve"),
    ("dev.warp.Warp", "dev.warp.Warp"),
    ("resolve", "resolve"),
    ("dev.warp.Warp", None),
])
@pytest.mark.parametrize("copyq_available", [True, False])
def test_delivery_uses_final_focus_and_keeps_blocked_transcript(
    tmp_path, monkeypatch, initial_app, final_app, copyq_available,
):
    focus = [initial_app]
    queries, history, typed, notices = [], [], [], []

    def read_focus():
        queries.append(focus[0])
        if focus[0] is None:
            raise focused_window.FocusUnavailable("unknown focus")
        return focused_window.FocusedWindow("active", focus[0])

    def transcribe(_path):
        focus[0] = final_app
        return SimpleNamespace(text="saved words", info={})

    monkeypatch.setattr(protection, "read_focused_window", read_focus)
    monkeypatch.setattr(main, "insert_transcript_into_recovery_queue", lambda text, _log:
                        history.append(text) or RecoveryQueueResult("copyq", copyq_available, 1))
    monkeypatch.setattr(main, "type_text_with_wtype", lambda text, _log: typed.append(text) or True)
    monkeypatch.setattr(protection, "notify_saved_transcript", lambda *args: notices.append(args))
    archive = DictationArchive(root=tmp_path / "archive")
    audio = tmp_path / "speech.wav"
    audio.write_bytes(b"recorded audio")
    capture = CaptureResult(path=audio, stats=None, created_at=datetime.now(timezone.utc),
                            end_reason=CaptureEndReason.USER_COMPLETED)
    backend = SimpleNamespace(transcribe=transcribe)
    tasks = queue.Queue()
    main.handle_capture_result(capture, task_queue=tasks, archive=archive, backend=backend, log_path=tmp_path / "log")
    assert queries == []
    task = tasks.get_nowait()
    main.process_capture(
        audio_path=task.audio_path, stats=task.stats, created_at=task.created_at,
        backend=backend, voice_isolation=None, log_path=tmp_path / "log",
        append_space=False, normalize_numbers=False, normalize_acronyms=False, ensure_punct=False,
        xdotool_bin=Path("xdotool"), enable_beep=False, injection_mode=main.InjectionMode.WTYPE,
        orca_daemon_dir=None, orca_session_id=None, archive=archive,
    )

    metadata = json.loads((archive.latest_dir / "metadata.json").read_text())
    assert queries == [final_app]
    assert (archive.latest_dir / "transcript.txt").read_text() == "saved words"
    assert (archive.latest_dir / "audio.wav").read_bytes() == b"recorded audio"
    assert metadata["clipboard"] == {"active_clipboard_modified": False}
    assert metadata["recovery_queue"]["inserted"] is copyq_available
    assert history == ["saved words"]
    assert tasks.empty()
    if final_app == "resolve" or final_app is None:
        assert typed == []
        assert metadata["status"] == "saved_without_injection"
        assert metadata["injection"]["blocked_reason"] == (
            protection.RESOLVE_FOCUSED if final_app else protection.DETECTION_UNAVAILABLE)
        if copyq_available:
            assert notices[0][1] == "CopyQ history"
        else:
            assert Path(notices[0][1]).read_text() == "saved words"
    else:
        assert typed == ["saved words"]
        assert metadata["status"] == "injected"
        assert notices == []


def test_notification_failure_preserves_blocked_result(tmp_path, monkeypatch):
    monkeypatch.setattr(protection.shutil, "which", lambda _name: "/usr/bin/gdbus")

    def timeout(*_args, **_kwargs):
        raise subprocess.TimeoutExpired("gdbus", 2)

    monkeypatch.setattr(protection.subprocess, "run", timeout)
    protection.notify_saved_transcript(protection.RESOLVE_FOCUSED, "CopyQ history", tmp_path / "log")
    assert "notification failed" in (tmp_path / "log").read_text()
