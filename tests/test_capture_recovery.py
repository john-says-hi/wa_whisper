import importlib
import json
import queue
import threading
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from wa_whisper import hotkeys as hotkeys_mod
from wa_whisper.dictation_archive import DictationArchive
from wa_whisper.hotkeys import CaptureEndReason, CaptureResult, PushToTalkHotkey
from wa_whisper.recorder import RecorderStats
from wa_whisper.recovery_queue import RecoveryQueueResult
from wa_whisper.whisper_backend import WhisperResult

main_mod = importlib.import_module("wa_whisper.main")
CAPTURE_CREATED_AT = datetime(2026, 7, 28, 22, 30, 45, tzinfo=timezone.utc)
CAPTURE_INTERRUPTED_AT = datetime(2026, 7, 28, 22, 31, 5, tzinfo=timezone.utc)


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


class TrackingArchive(DictationArchive):
    def __init__(self, root: Path) -> None:
        super().__init__(root=root)
        self.prune_calls = 0

    def prune_expired(self) -> int:
        self.prune_calls += 1
        return 0


class FailingArchive(DictationArchive):
    def __init__(self, root: Path, *, failure_stage: str) -> None:
        super().__init__(root=root)
        self.failure_stage = failure_stage

    def start_record(self, *args, **kwargs):
        if self.failure_stage == "start_record":
            raise OSError("start record failed")
        return super().start_record(*args, **kwargs)

    def update_record(self, *args, **kwargs) -> None:
        if self.failure_stage == "update_record":
            raise OSError("update record failed")
        super().update_record(*args, **kwargs)


class RecordingQueue:
    def __init__(self, events: list[object]) -> None:
        self.events = events

    def put(self, item) -> None:
        self.events.append(("queue", item))


class RecordingStopEvent:
    def __init__(self, events: list[object]) -> None:
        self.events = events

    def set(self) -> None:
        self.events.append("stop_event")


class FinalizingHotkey:
    def __init__(self, events: list[object]) -> None:
        self.events = events

    def stop(self, *, end_reason: CaptureEndReason) -> None:
        self.events.append(("hotkey_stop", end_reason))
        self.events.append("capture_archived")


def test_main_does_not_require_writable_log_path_before_compute_setup(
    tmp_path,
    monkeypatch,
):
    blocking_file = tmp_path / "not_a_directory"
    blocking_file.write_text("blocked", encoding="utf-8")
    unusable_log_path = blocking_file / "runtime.log"

    class FakeParser:
        @staticmethod
        def parse_args(_argv):
            return SimpleNamespace(
                log_path=unusable_log_path,
                compute_mode=None,
                device=None,
            )

    def reach_compute_setup(**_kwargs):
        raise RuntimeError("compute setup reached")

    monkeypatch.setattr(main_mod, "build_arg_parser", FakeParser)
    monkeypatch.setattr(main_mod, "resolve_compute_device", reach_compute_setup)

    with pytest.raises(RuntimeError, match="compute setup reached"):
        main_mod.main([])

    assert blocking_file.read_text(encoding="utf-8") == "blocked"


class FailingHotkey:
    def __init__(self, events: list[object]) -> None:
        self.events = events

    def stop(self, *, end_reason: CaptureEndReason) -> None:
        self.events.append(("hotkey_stop", end_reason))
        raise RuntimeError("stop failed")


def test_user_completed_capture_maps_to_current_capture_task(tmp_path):
    audio_path = write_audio(tmp_path)
    stats = capture_stats()
    task_queue = queue.Queue()

    main_mod.handle_capture_result(
        CaptureResult(
            path=audio_path,
            stats=stats,
            created_at=CAPTURE_CREATED_AT,
            end_reason=CaptureEndReason.USER_COMPLETED,
        ),
        task_queue=task_queue,
        archive=DictationArchive(root=tmp_path / "archive"),
        backend=FakeBackend(),
        log_path=tmp_path / "log.txt",
    )

    assert task_queue.get_nowait() == main_mod.CaptureTask(
        audio_path=audio_path,
        stats=stats,
        created_at=CAPTURE_CREATED_AT,
    )
    assert task_queue.empty()
    assert audio_path.exists()
    assert not (tmp_path / "archive").exists()


def test_unusable_log_path_does_not_block_normal_capture_task_archive(
    tmp_path,
    monkeypatch,
):
    audio_path = write_audio(tmp_path)
    blocking_file = tmp_path / "not_a_directory"
    blocking_file.write_text("blocked", encoding="utf-8")
    unusable_log_path = blocking_file / "runtime.log"
    task_queue = queue.Queue()
    archive = DictationArchive(root=tmp_path / "archive")
    monkeypatch.setattr(
        main_mod,
        "insert_transcript_into_recovery_queue",
        lambda *_args: RecoveryQueueResult(provider="copyq", inserted=True, row=1),
    )
    monkeypatch.setattr(main_mod, "inject_text", lambda *_args, **_kwargs: True)

    main_mod.handle_capture_result(
        CaptureResult(
            path=audio_path,
            stats=None,
            created_at=CAPTURE_CREATED_AT,
            end_reason=CaptureEndReason.USER_COMPLETED,
        ),
        task_queue=task_queue,
        archive=archive,
        backend=FakeBackend(),
        log_path=unusable_log_path,
    )
    task = task_queue.get_nowait()
    main_mod.process_capture(
        audio_path=task.audio_path,
        stats=task.stats,
        created_at=task.created_at,
        backend=FakeBackend(),
        voice_isolation=None,
        log_path=unusable_log_path,
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

    metadata = read_json(only_archived_metadata(archive.root))
    assert metadata["status"] == "injected"
    assert (archive.latest_dir / "transcript.txt").read_text(encoding="utf-8") == "hello world"
    assert blocking_file.read_text(encoding="utf-8") == "blocked"


@pytest.mark.parametrize(
    ("end_reason", "expected_reason"),
    [
        (CaptureEndReason.SERVICE_SHUTDOWN, "service_shutdown"),
        (CaptureEndReason.ESCAPE, "escape"),
    ],
)
def test_interrupted_capture_is_archived_without_processing(
    tmp_path,
    monkeypatch,
    end_reason,
    expected_reason,
):
    audio_path = write_audio(tmp_path)
    archive = DictationArchive(root=tmp_path / "archive")
    backend = FakeBackend()
    task_queue = queue.Queue()
    stats = capture_stats()
    monkeypatch.setattr(main_mod, "utc_now", lambda: CAPTURE_INTERRUPTED_AT)

    main_mod.handle_capture_result(
        CaptureResult(
            path=audio_path,
            stats=stats,
            created_at=CAPTURE_CREATED_AT,
            end_reason=end_reason,
        ),
        task_queue=task_queue,
        archive=archive,
        backend=backend,
        log_path=tmp_path / "log.txt",
    )

    metadata_path = only_archived_metadata(archive.root)
    metadata = read_json(metadata_path)
    assert task_queue.empty()
    assert backend.transcribed_path is None
    assert not audio_path.exists()
    assert (metadata_path.parent / "audio.wav").read_bytes() == b"audio bytes"
    assert (metadata_path.parent / "transcript.txt").read_text(encoding="utf-8") == ""
    assert (archive.latest_dir / "transcript.txt").read_text(encoding="utf-8") == ""
    assert read_json(archive.latest_dir / "metadata.json") == metadata
    assert metadata["created_at"] == "2026-07-28T22:30:45Z"
    assert metadata["capture_stats"]["total_ms"] == 1250.0
    assert metadata["status"] == "interrupted"
    assert metadata["interrupted_at"] == "2026-07-28T22:31:05Z"
    assert metadata["interruption_reason"] == expected_reason


@pytest.mark.parametrize("failure_stage", ["start_record", "update_record"])
def test_interrupted_archive_failure_retains_temp_audio_and_logs_exact_path(
    tmp_path,
    failure_stage,
):
    audio_path = write_audio(tmp_path)
    log_path = tmp_path / "log.txt"

    main_mod.handle_capture_result(
        CaptureResult(
            path=audio_path,
            stats=None,
            created_at=CAPTURE_CREATED_AT,
            end_reason=CaptureEndReason.SERVICE_SHUTDOWN,
        ),
        task_queue=queue.Queue(),
        archive=FailingArchive(root=tmp_path / "archive", failure_stage=failure_stage),
        backend=FakeBackend(),
        log_path=log_path,
    )

    assert audio_path.read_bytes() == b"audio bytes"
    log_text = log_path.read_text(encoding="utf-8")
    assert "retained temporary audio" in log_text
    assert str(audio_path) in log_text


def test_shutdown_finalizes_before_sentinel_and_stop_event_and_is_idempotent(tmp_path):
    events = []
    coordinator = main_mod.ShutdownCoordinator(
        task_queue=RecordingQueue(events),
        stop_event=RecordingStopEvent(events),
        log_path=tmp_path / "log.txt",
    )
    coordinator.bind_hotkey(FinalizingHotkey(events))

    assert coordinator.request(15, end_reason=CaptureEndReason.SERVICE_SHUTDOWN) is True
    assert coordinator.request(15, end_reason=CaptureEndReason.SERVICE_SHUTDOWN) is False

    assert events == [
        ("hotkey_stop", CaptureEndReason.SERVICE_SHUTDOWN),
        "capture_archived",
        ("queue", None),
        "stop_event",
    ]


def test_shutdown_reentrant_request_is_rejected_before_once_state_is_published(tmp_path):
    events = []
    nested_results = []
    coordinator = main_mod.ShutdownCoordinator(
        task_queue=RecordingQueue(events),
        stop_event=RecordingStopEvent(events),
        log_path=tmp_path / "log.txt",
    )
    coordinator.bind_hotkey(FinalizingHotkey(events))

    class ReentrantRequestGate:
        def __init__(self) -> None:
            self._lock = threading.Lock()
            self._reentered = False

        def acquire(self, *, blocking: bool) -> bool:
            acquired = self._lock.acquire(blocking=blocking)
            if acquired and not self._reentered:
                self._reentered = True
                nested_results.append(
                    coordinator.request(15, end_reason=CaptureEndReason.SERVICE_SHUTDOWN)
                )
            return acquired

        def release(self) -> None:
            self._lock.release()

    coordinator._request_gate = ReentrantRequestGate()

    assert coordinator.request(15, end_reason=CaptureEndReason.SERVICE_SHUTDOWN) is True
    assert nested_results == [False]
    assert events == [
        ("hotkey_stop", CaptureEndReason.SERVICE_SHUTDOWN),
        "capture_archived",
        ("queue", None),
        "stop_event",
    ]


def test_concurrent_shutdown_requests_publish_one_sentinel(tmp_path):
    request_count = 8
    events = []
    results = [None] * request_count
    barrier = threading.Barrier(request_count)
    coordinator = main_mod.ShutdownCoordinator(
        task_queue=RecordingQueue(events),
        stop_event=RecordingStopEvent(events),
        log_path=tmp_path / "log.txt",
    )
    coordinator.bind_hotkey(FinalizingHotkey(events))

    def request_shutdown(index: int) -> None:
        barrier.wait(timeout=1.0)
        results[index] = coordinator.request(
            15,
            end_reason=CaptureEndReason.SERVICE_SHUTDOWN,
        )

    threads = [threading.Thread(target=request_shutdown, args=(index,)) for index in range(request_count)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=1.0)

    assert all(thread.is_alive() is False for thread in threads)
    assert results.count(True) == 1
    assert results.count(False) == request_count - 1
    assert events.count(("hotkey_stop", CaptureEndReason.SERVICE_SHUTDOWN)) == 1
    assert events.count(("queue", None)) == 1
    assert events.count("stop_event") == 1


def test_shutdown_archives_real_hotkey_capture_before_single_sentinel(tmp_path, monkeypatch):
    events = []
    audio_path = tmp_path / "active-capture.wav"
    archive = DictationArchive(root=tmp_path / "archive")
    worker_queue = queue.Queue()

    class ActiveCaptureRecorder:
        def start(self) -> Path:
            audio_path.write_bytes(b"active audio")
            events.append("recorder_start")
            return audio_path

        def stop(self, silence_timeout: float) -> Path:
            events.append(("recorder_stop", silence_timeout))
            return audio_path

        @staticmethod
        def last_capture_stats():
            return None

    def archive_capture(result: CaptureResult) -> None:
        main_mod.handle_capture_result(
            result,
            task_queue=worker_queue,
            archive=archive,
            backend=FakeBackend(),
            log_path=tmp_path / "log.txt",
        )
        events.append("capture_archived")

    monkeypatch.setattr(hotkeys_mod, "utc_now", lambda: CAPTURE_CREATED_AT)
    monkeypatch.setattr(main_mod, "utc_now", lambda: CAPTURE_INTERRUPTED_AT)
    hotkey = PushToTalkHotkey(
        recorder=ActiveCaptureRecorder(),
        silence_timeout=0.0,
        on_capture_finished=archive_capture,
        log_path=tmp_path / "log.txt",
        enable_audio_mute=False,
        exit_on_esc=False,
        enable_hotkey_shield=False,
    )
    coordinator = main_mod.ShutdownCoordinator(
        task_queue=RecordingQueue(events),
        stop_event=RecordingStopEvent(events),
        log_path=tmp_path / "log.txt",
    )
    coordinator.bind_hotkey(hotkey)
    hotkey._handle_press(hotkeys_mod.keyboard.Key.alt_r)
    original_write_log = main_mod.write_log
    shutdown_log_attempts = []

    def fail_initial_shutdown_log(message, log_path):
        if message.startswith("Received shutdown signal"):
            shutdown_log_attempts.append(message)
            raise OSError("shutdown log unavailable")
        original_write_log(message, log_path)

    monkeypatch.setattr(main_mod, "write_log", fail_initial_shutdown_log)

    assert coordinator.request(15, end_reason=CaptureEndReason.SERVICE_SHUTDOWN) is True
    assert coordinator.request(15, end_reason=CaptureEndReason.SERVICE_SHUTDOWN) is False

    metadata = read_json(only_archived_metadata(archive.root))
    assert shutdown_log_attempts == ["Received shutdown signal 15 (service_shutdown)"]
    assert metadata["status"] == "interrupted"
    assert metadata["interruption_reason"] == "service_shutdown"
    assert worker_queue.empty()
    assert events == [
        "recorder_start",
        ("recorder_stop", 0.0),
        "capture_archived",
        ("queue", None),
        "stop_event",
    ]


def test_shutdown_log_and_failure_log_errors_cannot_escape_or_suppress_sentinel(
    tmp_path,
    monkeypatch,
):
    events = []
    coordinator = main_mod.ShutdownCoordinator(
        task_queue=RecordingQueue(events),
        stop_event=RecordingStopEvent(events),
        log_path=tmp_path / "log.txt",
    )
    coordinator.bind_hotkey(FailingHotkey(events))
    log_attempts = []

    def fail_every_log(message, _log_path):
        log_attempts.append(message)
        raise OSError("log unavailable")

    monkeypatch.setattr(main_mod, "write_log", fail_every_log)

    assert coordinator.request(15, end_reason=CaptureEndReason.SERVICE_SHUTDOWN) is True

    assert log_attempts == [
        "Received shutdown signal 15 (service_shutdown)",
        "Hotkey shutdown failed: stop failed",
    ]
    assert events == [
        ("hotkey_stop", CaptureEndReason.SERVICE_SHUTDOWN),
        ("queue", None),
        "stop_event",
    ]


def test_escape_shutdown_log_is_not_labeled_as_a_signal(tmp_path):
    events = []
    log_path = tmp_path / "log.txt"
    coordinator = main_mod.ShutdownCoordinator(
        task_queue=RecordingQueue(events),
        stop_event=RecordingStopEvent(events),
        log_path=log_path,
    )
    coordinator.bind_hotkey(FinalizingHotkey(events))

    assert coordinator.request(None, end_reason=CaptureEndReason.ESCAPE) is True

    log_text = log_path.read_text(encoding="utf-8")
    assert "Received shutdown request (escape)" in log_text
    assert "Received shutdown signal" not in log_text


def test_completion_beep_uses_shared_system_bell(tmp_path, monkeypatch):
    calls = []
    log_path = tmp_path / "log.txt"
    monkeypatch.setattr(
        main_mod,
        "play_system_bell",
        lambda path, *, purpose: calls.append((path, purpose)),
    )

    main_mod.play_completion_beep(log_path)

    assert calls == [(log_path, "completion")]


def test_process_capture_archives_audio_transcript_and_queues_copyq_recovery(tmp_path, monkeypatch):
    audio_path = write_audio(tmp_path)
    archive = TrackingArchive(root=tmp_path / "archive")
    existing_file = archive.root / "existing_record" / "keep.txt"
    existing_file.parent.mkdir(parents=True)
    existing_file.write_text("keep", encoding="utf-8")
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
        created_at=CAPTURE_CREATED_AT,
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
    assert existing_file.read_text(encoding="utf-8") == "keep"
    assert archive.prune_calls == 0

    data = read_json(metadata)
    assert data["created_at"] == "2026-07-28T22:30:45Z"
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
        created_at=CAPTURE_CREATED_AT,
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
        created_at=CAPTURE_CREATED_AT,
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
    assert (metadata_path.parent / "transcript.txt").read_text(encoding="utf-8") == ""
    assert (archive.latest_dir / "transcript.txt").read_text(encoding="utf-8") == ""
    assert data["status"] == "processing_failed"
    assert data["error"] == "boom"


def test_process_capture_keeps_empty_transcript_when_transcription_has_no_text(tmp_path, monkeypatch):
    audio_path = write_audio(tmp_path)
    archive = DictationArchive(root=tmp_path / "archive")

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("recovery queue and injection should not run without text")

    monkeypatch.setattr(main_mod, "insert_transcript_into_recovery_queue", fail_if_called)
    monkeypatch.setattr(main_mod, "inject_text", fail_if_called)

    main_mod.process_capture(
        audio_path=audio_path,
        stats=None,
        created_at=CAPTURE_CREATED_AT,
        backend=FakeBackend("  "),
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
    assert (metadata_path.parent / "transcript.txt").read_text(encoding="utf-8") == ""
    assert (archive.latest_dir / "transcript.txt").read_text(encoding="utf-8") == ""
    assert read_json(metadata_path)["status"] == "no_text"


def capture_stats() -> RecorderStats:
    return RecorderStats(
        total_ms=1250.0,
        speech_ms=1000.0,
        silence_ms=250.0,
        max_rms=0.5,
        avg_silence_rms=0.01,
        speech_blocks=8,
        silence_blocks=2,
        total_blocks=10,
        max_speech_streak_ms=750.0,
    )


def write_audio(tmp_path: Path) -> Path:
    audio_path = tmp_path / "capture.wav"
    audio_path.write_bytes(b"audio bytes")
    return audio_path


def only_archived_metadata(root: Path) -> Path:
    metadata_files = [
        path
        for path in root.rglob("metadata.json")
        if "latest" not in path.parts and ".latest_snapshots" not in path.parts
    ]
    assert len(metadata_files) == 1
    return metadata_files[0]


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))
