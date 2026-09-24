"""Per-recording protected-app permission across real hotkey completion and queued delivery."""

import importlib
import json
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from wa_whisper import app_protection, focused_window, hotkeys
from wa_whisper.dictation_archive import DictationArchive
from wa_whisper.processing_queue import CaptureQueue
from wa_whisper.recording_mode import RecordingMode
from wa_whisper.recovery_queue import RecoveryQueueResult

main = importlib.import_module("wa_whisper.main")


class Recorder:
    def __init__(self, root):
        self.root = root
        self.count = 0

    def start(self):
        self.count += 1
        self.path = self.root / f"capture_{self.count}.wav"
        self.path.write_bytes(b"saved audio")
        return self.path

    def stop(self, _timeout):
        return self.path

    def last_capture_stats(self):
        return None


def toggle_listen_mode(key):
    key._handle_press(hotkeys.keyboard.Key.ctrl_l)
    key._handle_press(hotkeys.keyboard.Key.alt_r)
    key._handle_release(hotkeys.keyboard.Key.alt_r)
    key._handle_release(hotkeys.keyboard.Key.ctrl_l)


def wait_for_finalization(key):
    deadline = time.monotonic() + 2
    while key.capture_state()["finalizing"]:
        assert time.monotonic() < deadline
        time.sleep(0.005)


@pytest.mark.parametrize("app_id, allowed_mode", [
    ("resolve", RecordingMode.HANDS_FREE),
    ("net-runelite-launcher-Launcher", RecordingMode.PUSH_TO_TALK),
])
@pytest.mark.parametrize("first_mode", [RecordingMode.PUSH_TO_TALK, RecordingMode.HANDS_FREE])
def test_mixed_hotkeys_keep_their_own_permission_while_transcription_is_queued(
    tmp_path, monkeypatch, first_mode, app_id, allowed_mode,
):
    monkeypatch.setattr(hotkeys, "play_system_bell", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(app_protection, "read_focused_window",
                        lambda: focused_window.FocusedWindow("active", app_id))
    typed, history, notices = [], [], []
    monkeypatch.setattr(main, "type_text_with_wtype", lambda text, _log: typed.append(text) or True)
    monkeypatch.setattr(main, "insert_transcript_into_recovery_queue",
                        lambda text, _log: history.append(text) or RecoveryQueueResult("copyq", True, 1))
    monkeypatch.setattr(app_protection, "notify_saved_transcript", lambda *args: notices.append(args))
    queue = CaptureQueue()
    archive = DictationArchive(root=tmp_path / "archive")
    backend = SimpleNamespace(transcribe=lambda _path: SimpleNamespace(text="marker words", info={}))

    def capture_finished(result):
        main.handle_capture_result(result, task_queue=queue, archive=archive,
                                   backend=backend, log_path=tmp_path / "log")

    key = hotkeys.PushToTalkHotkey(
        Recorder(tmp_path), on_capture_finished=capture_finished, silence_timeout=0,
        log_path=tmp_path / "log", enable_audio_mute=False, enable_hotkey_shield=False,
        hotkey_repress_grace_seconds=0,
    )
    modes = [first_mode, RecordingMode.HANDS_FREE if first_mode is RecordingMode.PUSH_TO_TALK
             else RecordingMode.PUSH_TO_TALK, first_mode]
    try:
        for mode in modes:
            if mode is RecordingMode.HANDS_FREE:
                toggle_listen_mode(key)
                toggle_listen_mode(key)
            else:
                key._handle_press(hotkeys.keyboard.Key.alt_r)
                key._handle_release(hotkeys.keyboard.Key.alt_r)
            wait_for_finalization(key)
        assert queue.waiting_recordings() == len(modes)
        assert key.capture_state()["recording"] is False
        assert not typed

        for mode in modes:
            task = queue.get_nowait()
            assert task.recording_mode is mode
            main.process_capture(
                audio_path=task.audio_path, stats=task.stats, created_at=task.created_at,
                recording_mode=task.recording_mode, backend=backend, voice_isolation=None,
                log_path=tmp_path / "log", append_space=False, normalize_numbers=False,
                normalize_acronyms=False, ensure_punct=False, xdotool_bin=Path("xdotool"),
                enable_beep=False, injection_mode=main.InjectionMode.WTYPE,
                orca_daemon_dir=None, orca_session_id=None, archive=archive,
            )
            metadata = json.loads((archive.latest_dir / "metadata.json").read_text())
            assert metadata["status"] == (
                "injected" if mode is allowed_mode else "saved_without_injection")
            assert (archive.latest_dir / "audio.wav").read_bytes() == b"saved audio"
            assert (archive.latest_dir / "transcript.txt").read_text() == "marker words"
            queue.task_done()
        assert queue.empty()
        assert typed == ["marker words"] * modes.count(allowed_mode)
        assert history == ["marker words"] * len(modes)
        assert len(notices) == len(modes) - modes.count(allowed_mode)
    finally:
        key.stop()
