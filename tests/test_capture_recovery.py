import importlib
import json
from pathlib import Path

from wa_whisper.dictation_archive import DictationArchive
from wa_whisper.recovery_queue import RecoveryQueueResult
from wa_whisper.whisper_backend import WhisperResult

main_mod = importlib.import_module("wa_whisper.main")


class FakeBackend:
    def __init__(self, text: str = "hello world", *, error: Exception | None = None) -> None:
        self.text = text
        self.error = error
        self.transcribed_path: Path | None = None

    def archive_metadata(self) -> dict:
        return {
            "model_name": "large-v3",
            "device": "cpu",
            "compute_mode": "ram",
            "fp16": False,
        }

    def transcribe(self, audio_path: Path) -> WhisperResult:
        self.transcribed_path = audio_path
        if self.error:
            raise self.error
        return WhisperResult(text=self.text, segments=[], info={"duration": 1.5})


def test_process_capture_archives_audio_transcript_and_queues_copyq_recovery(tmp_path, monkeypatch):
    audio_path = write_audio(tmp_path)
    archive = DictationArchive(root=tmp_path / "archive")
    recovery_queue_calls = []
    injection_calls = []

    monkeypatch.setattr(
        main_mod,
        "insert_transcript_into_recovery_queue",
        lambda text, log_path: recovery_queue_calls.append((text, log_path))
        or RecoveryQueueResult(provider="copyq", inserted=True, row=1),
    )
    monkeypatch.setattr(
        main_mod,
        "inject_text",
        lambda text, *_args, **_kwargs: injection_calls.append(text) or True,
    )

    main_mod.process_capture(
        audio_path=audio_path,
        stats=None,
        backend=FakeBackend(),
        voice_isolation=None,
        log_path=tmp_path / "log.txt",
        append_space=False,
        normalize_numbers=False,
        normalize_acronyms=False,
        ensure_punct=False,
        xdotool_bin=Path("/usr/bin/xdotool"),
        enable_beep=False,
        injection_mode=main_mod.InjectionMode.AUTO,
        orca_daemon_dir=None,
        orca_session_id=None,
        archive=archive,
    )

    metadata = only_archived_metadata(tmp_path / "archive")
    record_dir = metadata.parent
    assert not audio_path.exists()
    assert (record_dir / "audio.wav").read_bytes() == b"audio bytes"
    assert (record_dir / "transcript.txt").read_text(encoding="utf-8") == "hello world"
    assert (archive.latest_dir / "audio.wav").read_bytes() == b"audio bytes"
    assert (archive.latest_dir / "transcript.txt").read_text(encoding="utf-8") == "hello world"
    assert recovery_queue_calls == [("hello world", tmp_path / "log.txt")]
    assert injection_calls == ["hello world"]

    data = read_json(metadata)
    assert data["status"] == "injected"
    assert data["clipboard"] == {"active_clipboard_modified": False}
    assert data["recovery_queue"] == {"provider": "copyq", "inserted": True, "row": 1}
    assert data["injection"] == {"mode": "auto", "succeeded": True}
    assert data["backend"]["compute_mode"] == "ram"


def test_process_capture_preserves_transcript_when_injection_reports_failure(tmp_path, monkeypatch):
    audio_path = write_audio(tmp_path)
    archive = DictationArchive(root=tmp_path / "archive")

    monkeypatch.setattr(
        main_mod,
        "insert_transcript_into_recovery_queue",
        lambda *_args: RecoveryQueueResult(provider="copyq", inserted=True, row=1),
    )
    monkeypatch.setattr(main_mod, "inject_text", lambda *_args, **_kwargs: False)

    main_mod.process_capture(
        audio_path=audio_path,
        stats=None,
        backend=FakeBackend("important dictation"),
        voice_isolation=None,
        log_path=tmp_path / "log.txt",
        append_space=False,
        normalize_numbers=False,
        normalize_acronyms=False,
        ensure_punct=False,
        xdotool_bin=Path("/usr/bin/xdotool"),
        enable_beep=False,
        injection_mode=main_mod.InjectionMode.XDOTOOL_TYPE,
        orca_daemon_dir=None,
        orca_session_id=None,
        archive=archive,
    )

    metadata = read_json(only_archived_metadata(tmp_path / "archive"))
    assert metadata["status"] == "injection_failed"
    assert metadata["clipboard"] == {"active_clipboard_modified": False}
    assert metadata["recovery_queue"] == {"provider": "copyq", "inserted": True, "row": 1}
    assert metadata["injection"] == {"mode": "xdotool-type", "succeeded": False}
    assert (archive.latest_dir / "transcript.txt").read_text(encoding="utf-8") == "important dictation"


def test_process_capture_archives_audio_when_transcription_fails(tmp_path, monkeypatch):
    audio_path = write_audio(tmp_path)
    archive = DictationArchive(root=tmp_path / "archive")

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("recovery queue and injection should not run after transcription failure")

    monkeypatch.setattr(main_mod, "insert_transcript_into_recovery_queue", fail_if_called)
    monkeypatch.setattr(main_mod, "inject_text", fail_if_called)

    main_mod.process_capture(
        audio_path=audio_path,
        stats=None,
        backend=FakeBackend(error=RuntimeError("boom")),
        voice_isolation=None,
        log_path=tmp_path / "log.txt",
        append_space=False,
        normalize_numbers=False,
        normalize_acronyms=False,
        ensure_punct=False,
        xdotool_bin=Path("/usr/bin/xdotool"),
        enable_beep=False,
        injection_mode=main_mod.InjectionMode.AUTO,
        orca_daemon_dir=None,
        orca_session_id=None,
        archive=archive,
    )

    metadata_path = only_archived_metadata(tmp_path / "archive")
    data = read_json(metadata_path)
    assert not audio_path.exists()
    assert (metadata_path.parent / "audio.wav").read_bytes() == b"audio bytes"
    assert not (metadata_path.parent / "transcript.txt").exists()
    assert not (archive.latest_dir / "transcript.txt").exists()
    assert data["status"] == "processing_failed"
    assert data["error"] == "boom"


def write_audio(tmp_path: Path) -> Path:
    audio_path = tmp_path / "capture.wav"
    audio_path.write_bytes(b"audio bytes")
    return audio_path


def only_archived_metadata(root: Path) -> Path:
    metadata_files = [path for path in root.rglob("metadata.json") if "latest" not in path.parts]
    assert len(metadata_files) == 1
    return metadata_files[0]


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))
