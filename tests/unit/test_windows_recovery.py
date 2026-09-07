"""Windows-specific recovery when inference or power-off interrupts delivery."""

import json
import threading
from datetime import datetime, timezone
from types import SimpleNamespace

from wa_whisper import windows_worker as worker
from wa_whisper.hotkeys import CaptureEndReason, CaptureResult


def test_failed_inference_keeps_audio_and_records_error(tmp_path, monkeypatch):
    monkeypatch.setattr(worker, "ARCHIVE_ROOT", tmp_path / "archive")
    monkeypatch.setattr(worker, "report", lambda message: None)
    source = tmp_path / "source.wav"
    source.write_bytes(b"recoverable audio")

    def fail(path):
        raise RuntimeError("GPU unavailable")

    backend = SimpleNamespace(archive_metadata=lambda: {}, transcribe=fail)
    capture = CaptureResult(source, None, datetime.now(timezone.utc), CaptureEndReason.USER_COMPLETED)
    worker.process_capture(capture, backend, threading.Event())
    record = next((tmp_path / "archive").iterdir())
    assert (record / "audio.wav").read_bytes() == b"recoverable audio"
    assert (record / "transcript.txt").read_text() == ""
    assert json.loads((record / "metadata.json").read_text())["error"] == "GPU unavailable"


def test_power_off_during_inference_saves_text_without_typing(tmp_path, monkeypatch):
    monkeypatch.setattr(worker, "ARCHIVE_ROOT", tmp_path / "archive")
    monkeypatch.setattr(worker, "report", lambda message: None)
    source = tmp_path / "source.wav"
    source.write_bytes(b"recoverable audio")
    stopping = threading.Event()

    def transcribe(path):
        stopping.set()
        return SimpleNamespace(text="Saved dictation.")

    def disallow_typing():
        raise AssertionError("Typing must not occur after power-off")

    from wa_whisper import windows_input

    monkeypatch.setattr(windows_input, "type_text", lambda text: disallow_typing())
    backend = SimpleNamespace(archive_metadata=lambda: {}, transcribe=transcribe)
    capture = CaptureResult(source, None, datetime.now(timezone.utc), CaptureEndReason.USER_COMPLETED)
    worker.process_capture(capture, backend, stopping)
    record = next((tmp_path / "archive").iterdir())
    assert "saved dictation" in (record / "transcript.txt").read_text().lower()
    assert json.loads((record / "metadata.json").read_text())["status"] == "saved_without_injection"


def test_blocked_windows_input_preserves_transcript(tmp_path, monkeypatch):
    from wa_whisper import windows_input

    monkeypatch.setattr(worker, "ARCHIVE_ROOT", tmp_path / "archive")
    monkeypatch.setattr(worker, "report", lambda message: None)
    source = tmp_path / "source.wav"
    source.write_bytes(b"recoverable audio")

    def blocked(text):
        raise RuntimeError("Windows blocked input")

    monkeypatch.setattr(windows_input, "type_text", blocked)
    backend = SimpleNamespace(archive_metadata=lambda: {}, transcribe=lambda path: SimpleNamespace(text="Keep this text."))
    capture = CaptureResult(source, None, datetime.now(timezone.utc), CaptureEndReason.USER_COMPLETED)
    worker.process_capture(capture, backend, threading.Event())
    record = next((tmp_path / "archive").iterdir())
    assert "keep this text" in (record / "transcript.txt").read_text().lower()
    assert json.loads((record / "metadata.json").read_text())["error"] == "Windows blocked input"
