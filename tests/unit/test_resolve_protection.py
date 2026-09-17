import importlib
import json
import queue
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from wa_whisper import resolve_protection as protection
from wa_whisper.dictation_archive import DictationArchive
from wa_whisper.hotkeys import CaptureEndReason, CaptureResult
from wa_whisper.recovery_queue import RecoveryQueueResult

main = importlib.import_module("wa_whisper.main")


@pytest.mark.parametrize("suffix", ["", " (deleted)"])
def test_detects_resolve_executable_despite_gui_thread_name(tmp_path, suffix):
    process = tmp_path / "123"
    process.mkdir()
    (process / "comm").write_text("GUI Thread\n")
    (process / "exe").symlink_to(protection.RESOLVE_EXECUTABLE + suffix)

    assert protection.insertion_block_reason(tmp_path) == protection.RESOLVE_RUNNING


def test_unrelated_process_named_resolve_does_not_block(tmp_path):
    process = tmp_path / "123"
    process.mkdir()
    (process / "comm").write_text("resolve\n")
    (process / "exe").symlink_to("/usr/bin/editor")

    assert protection.insertion_block_reason(tmp_path) is None


def test_missing_process_directory_blocks_insertion(tmp_path):
    assert protection.insertion_block_reason(tmp_path / "missing") == protection.DETECTION_UNAVAILABLE


def test_unreadable_user_process_blocks_insertion(tmp_path, monkeypatch):
    process = tmp_path / "123"
    process.mkdir()
    (process / "cmdline").write_bytes(b"")

    def denied(_path):
        raise PermissionError("process identity unavailable")

    monkeypatch.setattr(Path, "readlink", denied)
    assert protection.insertion_block_reason(tmp_path) == protection.DETECTION_UNAVAILABLE


@pytest.mark.parametrize("command, expected", [
    [b"/usr/lib/systemd/systemd\0--user\0", None],
    [b"/opt/resolve/bin/resolve\0", protection.RESOLVE_RUNNING],
    [b"./resolve\0", protection.RESOLVE_RUNNING],
])
def test_restricted_process_uses_launch_command(tmp_path, monkeypatch, command, expected):
    process = tmp_path / "123"
    process.mkdir()
    (process / "cmdline").write_bytes(command)

    def denied(_path):
        raise PermissionError("restricted executable link")

    monkeypatch.setattr(Path, "readlink", denied)
    assert protection.insertion_block_reason(tmp_path) == expected


def test_exited_process_does_not_disable_dictation(tmp_path):
    (tmp_path / "123").mkdir()
    assert protection.insertion_block_reason(tmp_path) is None


@pytest.mark.parametrize("mode", list(main.InjectionMode))
@pytest.mark.parametrize("reason", [protection.RESOLVE_RUNNING, protection.DETECTION_UNAVAILABLE])
def test_every_delivery_mode_blocks_before_any_backend(tmp_path, monkeypatch, mode, reason):
    monkeypatch.setattr(protection, "insertion_block_reason", lambda: reason)

    def forbidden(*_args, **_kwargs):
        pytest.fail("No injection or paste backend may run when Resolve protection blocks delivery")

    for name in ("type_text_with_wtype", "type_text_with_xdotool", "inject_text_auto", "inject_text_orca_daemon"):
        monkeypatch.setattr(main, name, forbidden)
    with pytest.raises(protection.InsertionBlocked, match=reason):
        main.inject_text("delete this", Path("xdotool"), tmp_path / "log", enable_beep=True, injection_mode=mode)


@pytest.mark.parametrize("checks", [
    [protection.RESOLVE_RUNNING, None, None],
    [None, protection.RESOLVE_RUNNING, None],
    [None, None, protection.RESOLVE_RUNNING],
    [None, None, protection.DETECTION_UNAVAILABLE],
    [None, None, None],
])
@pytest.mark.parametrize("copyq_available", [True, False])
def test_capture_saves_text_and_retains_block_across_queue_and_transcription(
    tmp_path, monkeypatch, checks, copyq_available,
):
    states = iter(checks)
    monkeypatch.setattr(protection, "insertion_block_reason", lambda: next(states))
    history, typed, notices = [], [], []
    monkeypatch.setattr(main, "insert_transcript_into_recovery_queue", lambda text, _log:
                        history.append(text) or RecoveryQueueResult("copyq", copyq_available, 1))
    monkeypatch.setattr(main, "type_text_with_wtype", lambda text, _log: typed.append(text) or True)
    monkeypatch.setattr(protection, "notify_saved_transcript", lambda *args: notices.append(args))
    archive = DictationArchive(root=tmp_path / "archive")
    audio = tmp_path / "speech.wav"
    audio.write_bytes(b"recorded audio")
    capture = CaptureResult(path=audio, stats=None, created_at=datetime.now(timezone.utc),
                            end_reason=CaptureEndReason.USER_COMPLETED)
    backend = SimpleNamespace(transcribe=lambda _path: SimpleNamespace(text="saved words", info={}))
    tasks = queue.Queue()
    main.handle_capture_result(capture, task_queue=tasks, archive=archive, backend=backend, log_path=tmp_path / "log")
    task = tasks.get_nowait()
    main.process_capture(
        audio_path=task.audio_path, stats=task.stats, created_at=task.created_at,
        backend=backend, voice_isolation=None, log_path=tmp_path / "log",
        append_space=False, normalize_numbers=False, normalize_acronyms=False, ensure_punct=False,
        xdotool_bin=Path("xdotool"), enable_beep=False, injection_mode=main.InjectionMode.WTYPE,
        orca_daemon_dir=None, orca_session_id=None, archive=archive,
        insertion_block_reason=task.insertion_block_reason,
    )

    metadata = json.loads((archive.latest_dir / "metadata.json").read_text())
    assert (archive.latest_dir / "transcript.txt").read_text() == "saved words"
    assert (archive.latest_dir / "audio.wav").read_bytes() == b"recorded audio"
    assert metadata["clipboard"] == {"active_clipboard_modified": False}
    assert metadata["recovery_queue"]["inserted"] is copyq_available
    assert history == ["saved words"]
    assert tasks.empty()
    blocked = next((reason for reason in checks if reason), None)
    if blocked:
        assert typed == []
        assert metadata["status"] == "saved_without_injection"
        assert metadata["injection"]["blocked_reason"] == blocked
        assert notices[0][0] == blocked
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
    protection.notify_saved_transcript(protection.RESOLVE_RUNNING, "CopyQ history", tmp_path / "log")
    assert "notification failed" in (tmp_path / "log").read_text()
