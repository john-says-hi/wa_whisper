"""Interactive Windows dictation using the shared capture and Whisper engine."""

from __future__ import annotations

import json
import queue
import shutil
import sys
import threading
import uuid
from pathlib import Path

from .hotkeys import CaptureEndReason, CaptureResult
from .log_utils import DEFAULT_LOG_PATH, write_log
from .recorder import Recorder
from .text_postprocess import postprocess_text
from .whisper_backend import WhisperBackend, WhisperConfig

STATE_PATH = Path.home() / ".cache" / "wa_whisper" / "windows_status.txt"
ARCHIVE_ROOT = Path.home() / "Music" / "wa_whisper_recordings"


def report(message: str) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(message, encoding="utf-8")
    write_log(message)


def archive_capture(capture: CaptureResult) -> Path:
    """Preserve the audio before inference, without requiring Windows symlinks."""
    stamp = capture.created_at.astimezone().strftime("%Y-%m-%d_%H-%M-%S")
    record = ARCHIVE_ROOT / f"{stamp}_{uuid.uuid4().hex[:8]}"
    record.mkdir(parents=True)
    if capture.path is None:
        raise ValueError("Capture has no audio")
    shutil.move(str(capture.path), str(record / "audio.wav"))
    (record / "transcript.txt").write_text("", encoding="utf-8")
    save_metadata(record, {"status": "captured", "end_reason": capture.end_reason.value})
    return record


def save_metadata(record: Path, metadata: dict) -> None:
    (record / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def process_capture(capture: CaptureResult, backend: WhisperBackend, stopping: threading.Event) -> None:
    if capture.path is None:
        return
    record = archive_capture(capture)
    metadata = backend.archive_metadata()
    if stopping.is_set() or capture.end_reason != CaptureEndReason.USER_COMPLETED:
        save_metadata(record, {**metadata, "status": "interrupted"})
        return
    try:
        report("Transcribing")
        result = backend.transcribe(record / "audio.wav")
        text = postprocess_text(result.text)
        (record / "transcript.txt").write_text(text, encoding="utf-8")
        if text and not stopping.is_set():
            from .windows_input import type_text

            type_text(text + " ")
            status = "injected"
        else:
            status = "saved_without_injection"
        save_metadata(record, {**metadata, "status": status})
        report("Ready — hold Right Alt to dictate")
    except Exception as exc:
        save_metadata(record, {**metadata, "status": "failed", "error": str(exc)})
        report(f"Dictation failed; audio saved: {exc}")


def main() -> None:
    from .windows_hotkeys import WindowsPushToTalkHotkey

    stopping = threading.Event()
    captures: queue.Queue[CaptureResult] = queue.Queue()
    report("Loading Whisper large-v3 on GPU")
    backend = WhisperBackend(WhisperConfig(device="cuda", compute_mode="gpu", fp16=True), DEFAULT_LOG_PATH)
    recorder = Recorder(sample_rate=16000, device_index=None, log_path=DEFAULT_LOG_PATH, rms_threshold=0.01)
    hotkey = WindowsPushToTalkHotkey(
        recorder,
        silence_timeout=0.5,
        on_capture_finished=captures.put,
        log_path=DEFAULT_LOG_PATH,
        enable_audio_mute=True,
        enable_hotkey_shield=False,
        exit_on_esc=False,
    )

    def stop_on_input() -> None:
        sys.stdin.readline()
        stopping.set()
        hotkey.stop(end_reason=CaptureEndReason.SERVICE_SHUTDOWN)

    threading.Thread(target=stop_on_input, daemon=True).start()
    try:
        backend.load()
        if not stopping.is_set():
            hotkey.start()
            report("Ready — hold Right Alt to dictate")
        while not stopping.is_set() or not captures.empty():
            try:
                capture = captures.get(timeout=0.2)
            except queue.Empty:
                continue
            process_capture(capture, backend, stopping)
    finally:
        hotkey.stop(end_reason=CaptureEndReason.SERVICE_SHUTDOWN)
        while not captures.empty():
            capture = captures.get_nowait()
            if capture.path:
                archive_capture(capture)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        report(f"Startup failed: {error}")
        raise
