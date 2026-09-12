"""CLI entrypoint for wa_whisper."""

from __future__ import annotations

import argparse
import json
import queue
import re
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Optional, Tuple

from .audio_cues import play_system_bell
from .admission import enabled as shared_admission_enabled
from .compute_mode import ComputeModeError, compute_mode_values, resolve_compute_device
from .control_server import ControlServer
from .control_state import HandoffController
from .dictation_archive import DictationArchive, DictationRecord, format_timestamp, utc_now
from .hotkeys import CaptureEndReason, CaptureResult, PushToTalkHotkey
from .log_utils import DEFAULT_LOG_PATH, write_log
from .processing_queue import CaptureQueue, CaptureWorker, DrainBarrier
from .recorder import Recorder, RecorderStats
from .recording_admission import RecordingAdmission
from .recovery_queue import insert_transcript_into_recovery_queue, skipped_recovery_queue_result
from .text_postprocess import postprocess_text
from .voice_isolation import VoiceIsolationPipeline
from .whisper_backend import DEFAULT_MODEL_CACHE, WhisperBackend, WhisperConfig
from .device_routing import RoutedBackend


def positive_int_argument(value: str) -> int:
    """Parse a positive integer CLI argument."""
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Push-to-talk dictation powered by OpenAI Whisper.")
    parser.add_argument("--sample-rate", type=int, default=16_000, help="Microphone sample rate (Hz).")
    parser.add_argument("--rms-threshold", type=float, default=0.01, help="Voice activity RMS threshold.")
    parser.add_argument("--preamp", type=float, default=1.0, help="Signal gain applied before encoding.")
    parser.add_argument("--silence-timeout", type=float, default=0.5, help="Seconds of silence before stop.")
    parser.add_argument("--device-index", type=int, default=None, help="SoundDevice input index.")
    parser.add_argument(
        "--input-channel",
        type=positive_int_argument,
        default=None,
        metavar="N",
        help="One-based input channel to record. Omit to select automatically.",
    )
    parser.add_argument("--log-path", type=Path, default=DEFAULT_LOG_PATH, help="Log file path.")
    parser.add_argument("--model", default="large-v3", help="Whisper model name.")
    parser.add_argument("--beam-size", type=int, default=5, help="Beam search width.")
    parser.add_argument("--best-of", type=int, default=5, help="Number of candidate samples.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature.")
    parser.add_argument("--initial-prompt", default=None, help="Optional prompt bias string.")
    parser.add_argument("--no-audio-mute", action="store_true", help="Disable automatic desktop audio mute.")
    parser.add_argument("--append-space", action="store_true", help="Add a trailing space to output.")
    parser.add_argument("--disable-number-normalization", action="store_true", help="Skip number conversions.")
    parser.add_argument("--disable-acronym-normalization", action="store_true", help="Skip acronym conversions.")
    parser.add_argument("--disable-punctuation", action="store_true", help="Do not enforce sentence punctuation.")
    compute_group = parser.add_mutually_exclusive_group()
    compute_group.add_argument("--device", default=None, help="Force Whisper device (cuda/cpu).")
    compute_group.add_argument(
        "--compute-mode",
        choices=compute_mode_values(),
        default=None,
        help="Use a named compute mode for this run. Missing mode config defaults to gpu.",
    )
    parser.add_argument("--model-cache", type=Path, default=None, help="Override Whisper model cache dir.")
    parser.add_argument("--exit-on-esc", action="store_true", default=False, help="Stop listener on ESC.")
    parser.add_argument("--no-voice-isolation", action="store_true", help="Disable placeholder voice isolation.")
    parser.add_argument("--xdotool-path", type=Path, default=None, help="Override xdotool binary path.")
    parser.add_argument(
        "--injection-mode",
        choices=["xdotool-type", "orca-daemon", "auto", "wtype"],
        default="xdotool-type",
        help=(
            "Text delivery strategy. The default preserves xdotool typing. "
            "Use 'wtype' under Wayland, where xdotool can only reach XWayland clients."
        ),
    )
    parser.add_argument("--orca-daemon-dir", type=Path, default=None, help="Override Orca daemon directory.")
    parser.add_argument("--orca-session-id", default=None, help="Target one Orca daemon terminal session.")
    parser.add_argument("--disable-complete-beep", action="store_true", help="Disable post-paste completion beep.")
    parser.add_argument(
        "--no-hotkey-shield",
        action="store_true",
        help="Do not grab the hotkey; focused apps will also see Right Alt (Electron apps open their menu).",
    )
    return parser


@dataclass(frozen=True, slots=True)
class CaptureTask:
    audio_path: Path
    stats: Optional[RecorderStats]
    created_at: datetime


TaskItem = Optional[CaptureTask | DrainBarrier]
INTERRUPTION_REASONS = {
    CaptureEndReason.SERVICE_SHUTDOWN: "service_shutdown",
    CaptureEndReason.ESCAPE: "escape",
}
ORCA_BRACKETED_PASTE_END = "\x1b[201~"
ORCA_BRACKETED_PASTE_START = "\x1b[200~"
ORCA_DAEMON_PROTOCOL_VERSION = 10
ORCA_DAEMON_TIMEOUT_SECONDS = 2.0
DEFAULT_ORCA_STATE_PATH = Path.home() / ".config" / "orca" / "orca-data.json"
# Electron may read the clipboard well after the paste keystroke arrives;
# restoring the previous clipboard too early pastes stale content instead.
CLIPBOARD_PASTE_SETTLE_SECONDS = 1.0


class InjectionMode(str, Enum):
    XDOTOOL_TYPE = "xdotool-type"
    ORCA_DAEMON = "orca-daemon"
    AUTO = "auto"
    # Wayland compositors expose no XTEST, so xdotool silently types into
    # nothing. wtype drives zwp_virtual_keyboard_manager_v1 instead, which
    # reaches native Wayland windows and XWayland ones alike.
    WTYPE = "wtype"


@dataclass(frozen=True)
class ActiveWindowInfo:
    window_id: str
    wm_classes: tuple[str, ...]
    name: str
    pid: Optional[int]
    process_args: str


class ShutdownCoordinator:
    """Finalize capture and close the worker queue exactly once."""

    def __init__(
        self,
        *,
        task_queue: "queue.Queue[TaskItem]",
        stop_event: threading.Event,
        log_path: Path,
    ) -> None:
        self._task_queue = task_queue
        self._stop_event = stop_event
        self._log_path = log_path
        self._hotkey: Optional[PushToTalkHotkey] = None
        self._requested = False
        self._lock = threading.RLock()
        self._request_gate = threading.Lock()

    def bind_hotkey(self, hotkey: PushToTalkHotkey) -> None:
        """Bind the hotkey after constructing its exit callback."""
        with self._lock:
            if self._hotkey is not None:
                raise RuntimeError("Shutdown hotkey is already bound")
            if self._requested:
                raise RuntimeError("Cannot bind shutdown hotkey after shutdown starts")
            self._hotkey = hotkey

    def request(self, signum: int | None, *, end_reason: CaptureEndReason) -> bool:
        """Stop capture before publishing the sentinel and stop event."""
        if not self._request_gate.acquire(blocking=False):
            return False
        try:
            with self._lock:
                if self._requested:
                    return False
                if self._hotkey is None:
                    raise RuntimeError("Shutdown hotkey has not been bound")
                self._requested = True
                hotkey = self._hotkey
        finally:
            self._request_gate.release()

        if end_reason is CaptureEndReason.ESCAPE:
            self._write_log_best_effort("Received shutdown request (escape)")
        else:
            self._write_log_best_effort(
                f"Received shutdown signal {signum} ({end_reason.value})"
            )

        try:
            hotkey.stop(end_reason=end_reason)
        except Exception as exc:
            self._write_log_best_effort(f"Hotkey shutdown failed: {exc}")
        finally:
            self._task_queue.put(None)
            self._stop_event.set()
        return True

    def _write_log_best_effort(self, message: str) -> None:
        try:
            write_log(message, self._log_path)
        except Exception:
            return


def ensure_xdotool(path_override: Optional[Path]) -> Path:
    if path_override:
        return path_override
    resolved = shutil.which("xdotool")
    if not resolved:
        raise RuntimeError("xdotool not found; install it to enable text injection.")
    return Path(resolved)


def ensure_wtype() -> str:
    resolved = shutil.which("wtype")
    if not resolved:
        raise RuntimeError("wtype not found; install it (apt install wtype) to type under Wayland.")
    return resolved


def parse_injection_mode(value: str) -> InjectionMode:
    try:
        return InjectionMode(value)
    except ValueError as exc:
        raise RuntimeError(f"Unsupported injection mode: {value}") from exc


def main(argv: Optional[list[str]] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    log_path = args.log_path
    write_log("wa_whisper starting", log_path)

    try:
        compute_settings = resolve_compute_device(
            compute_mode_override=args.compute_mode,
            device_override=args.device,
        )
    except ComputeModeError as exc:
        parser.error(str(exc))

    config = WhisperConfig(
        model_name=args.model,
        beam_size=args.beam_size,
        best_of=args.best_of,
        temperature=args.temperature,
        initial_prompt=args.initial_prompt,
        cache_dir=args.model_cache or DEFAULT_MODEL_CACHE,
        device=compute_settings.device,
        compute_mode=compute_settings.compute_mode.value if compute_settings.compute_mode else None,
        fp16=compute_settings.fp16,
    )

    backend = RoutedBackend(config, log_path)
    recorder = Recorder(
        sample_rate=args.sample_rate,
        device_index=args.device_index,
        input_channel=args.input_channel,
        log_path=log_path,
        rms_threshold=args.rms_threshold,
        preamp=args.preamp,
    )
    injection_mode = parse_injection_mode(args.injection_mode)
    voice_isolation = None if args.no_voice_isolation else VoiceIsolationPipeline(log_path)
    xdotool_bin = (
        ensure_xdotool(args.xdotool_path)
        if injection_mode in (InjectionMode.XDOTOOL_TYPE, InjectionMode.AUTO)
        else args.xdotool_path or Path("xdotool")
    )
    archive = DictationArchive()

    task_queue: "queue.Queue[TaskItem]" = CaptureQueue()
    stop_event = threading.Event()
    shutdown_coordinator = ShutdownCoordinator(
        task_queue=task_queue,
        stop_event=stop_event,
        log_path=log_path,
    )

    def process_task(item: CaptureTask) -> None:
        process_capture(
            audio_path=item.audio_path,
            stats=item.stats,
            created_at=item.created_at,
            backend=backend,
            voice_isolation=voice_isolation,
            log_path=log_path,
            append_space=True,
            normalize_numbers=not args.disable_number_normalization,
            normalize_acronyms=not args.disable_acronym_normalization,
            ensure_punct=not args.disable_punctuation,
            xdotool_bin=xdotool_bin,
            enable_beep=not args.disable_complete_beep,
            injection_mode=injection_mode,
            orca_daemon_dir=args.orca_daemon_dir,
            orca_session_id=args.orca_session_id,
            archive=archive,
        )

    worker = CaptureWorker(task_queue, process_task, log_path)
    worker.start()

    def handle_capture(result: CaptureResult) -> None:
        handle_capture_result(
            result,
            task_queue=task_queue,
            archive=archive,
            backend=backend,
            log_path=log_path,
        )

    def handle_exit(end_reason: CaptureEndReason) -> None:
        backend.begin_shutdown()
        shutdown_coordinator.request(
            None,
            end_reason=end_reason,
        )

    hotkey = PushToTalkHotkey(
        recorder,
        silence_timeout=args.silence_timeout,
        on_capture_finished=handle_capture,
        log_path=log_path,
        enable_audio_mute=not args.no_audio_mute,
        exit_on_esc=args.exit_on_esc,
        on_exit=handle_exit,
        enable_hotkey_shield=not args.no_hotkey_shield,
        capture_admission=RecordingAdmission(task_queue, backend),
        on_device_switch=backend.request_switch,
    )
    shutdown_coordinator.bind_hotkey(hotkey)

    def shutdown(signum: int, _frame) -> None:
        backend.begin_shutdown()
        shutdown_coordinator.request(
            signum,
            end_reason=CaptureEndReason.SERVICE_SHUTDOWN,
        )

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    backend.bind(hotkey, worker, archive)
    control = ControlServer(HandoffController(
        hotkey,
        worker,
        lambda: shutdown_coordinator.request(None, end_reason=CaptureEndReason.SERVICE_SHUTDOWN),
        devices=backend,
    ))
    try:
        start_capture_service(control, hotkey, backend, log_path)
        stop_event.wait()
    finally:
        backend.begin_shutdown()
        shutdown_coordinator.request(
            signal.SIGTERM,
            end_reason=CaptureEndReason.SERVICE_SHUTDOWN,
        )
        task_queue.join()
        worker.join(timeout=2.0)
        control.close()
        backend.close()
        write_log("wa_whisper stopped", log_path)


def start_capture_service(control, hotkey, backend, log_path: Path) -> None:
    """Start the CPU control path before admitting any microphone capture."""
    try:
        control.start()
        backend.enable_cooperative_capture()
        write_log(f"Cooperative handoff control ready: {control.path}", log_path)
    except OSError as exc:
        write_log(f"Cooperative handoff control unavailable: {exc}", log_path)
        if shared_admission_enabled():
            raise RuntimeError("Shared GPU admission requires the Whisper handoff endpoint") from exc
    if hotkey.start():
        write_log(
            "Ready for dictation (Right Alt push-to-talk; "
            "Left Ctrl + Right Alt hands-free)",
            log_path,
        )


def handle_capture_result(
    result: CaptureResult,
    *,
    task_queue: "queue.Queue[TaskItem]",
    archive: DictationArchive,
    backend: WhisperBackend,
    log_path: Path,
) -> None:
    """Queue completed captures and synchronously archive interrupted ones."""
    if result.path is None:
        write_log("Capture finished with no audio file", log_path)
        return

    if result.end_reason == CaptureEndReason.USER_COMPLETED:
        task_queue.put(
            CaptureTask(
                audio_path=result.path,
                stats=result.stats,
                created_at=result.created_at,
            )
        )
        return

    archive_interrupted_capture(
        audio_path=result.path,
        stats=result.stats,
        created_at=result.created_at,
        end_reason=result.end_reason,
        archive=archive,
        backend=backend,
        log_path=log_path,
    )


def archive_interrupted_capture(
    *,
    audio_path: Path,
    stats: Optional[RecorderStats],
    created_at: datetime,
    end_reason: CaptureEndReason,
    archive: DictationArchive,
    backend: WhisperBackend,
    log_path: Path,
) -> bool:
    """Archive an interrupted WAV without sending it through transcription."""
    interruption_reason = INTERRUPTION_REASONS.get(end_reason)
    if interruption_reason is None:
        write_log(
            f"Unsupported capture end reason {end_reason}; retained temporary audio at {audio_path}",
            log_path,
        )
        return False

    try:
        interrupted_at = format_timestamp(utc_now())
        record = archive.start_record(
            audio_path,
            stats=stats,
            backend=backend_archive_metadata(backend),
            created_at=created_at,
        )
        archive.update_record(
            record,
            status="interrupted",
            interrupted_at=interrupted_at,
            interruption_reason=interruption_reason,
        )
    except Exception as exc:
        write_log(
            f"Interrupted capture archive failed; retained temporary audio at {audio_path}: {exc}",
            log_path,
        )
        return False

    try:
        audio_path.unlink()
    except OSError as exc:
        write_log(f"Archived interrupted capture but could not delete {audio_path}: {exc}", log_path)
        return False

    write_log(f"Archived interrupted capture audio: {record.audio_path}", log_path)
    return True


def process_capture(
    *,
    audio_path: Path,
    stats: Optional[RecorderStats],
    created_at: datetime,
    backend: WhisperBackend,
    voice_isolation: Optional[VoiceIsolationPipeline],
    log_path: Path,
    append_space: bool,
    normalize_numbers: bool,
    normalize_acronyms: bool,
    ensure_punct: bool,
    xdotool_bin: Path,
    enable_beep: bool,
    injection_mode: InjectionMode,
    orca_daemon_dir: Optional[Path],
    orca_session_id: Optional[str],
    archive: Optional[DictationArchive] = None,
) -> None:
    write_log(f"Processing capture {audio_path}", log_path)
    if stats:
        ratio = f"{stats.speech_ratio:.2f}" if stats.speech_ratio is not None else "n/a"
        max_db = f"{stats.speech_max_db:.1f}dB" if stats.speech_max_db is not None else "n/a"
        write_log(
            "Capture stats "
            f"total={stats.total_ms:.1f}ms speech={stats.speech_ms:.1f}ms "
            f"silence={stats.silence_ms:.1f}ms ratio={ratio} max_db={max_db}",
            log_path,
        )
    archive_record = start_dictation_archive_record(
        archive,
        audio_path,
        stats=stats,
        created_at=created_at,
        backend=backend_archive_metadata(backend),
        log_path=log_path,
    )
    enhanced_path = audio_path
    recovery = getattr(backend, "recovery", None)
    persisted = False
    try:
        if recovery:
            if archive_record is None:
                raise RuntimeError("Cannot transcribe until the recording is safely archived")
            enhanced_path = archive_record.audio_path
            recovery.begin(archive_record, {
                "normalize_numbers_enabled": normalize_numbers,
                "normalize_acronyms_enabled": normalize_acronyms,
                "ensure_punctuation": ensure_punct, "append_space": append_space,
            })
        if voice_isolation:
            enhanced_path = voice_isolation.enhance(enhanced_path)
        result = backend.transcribe(enhanced_path)
        text = postprocess_text(
            result.text,
            normalize_numbers_enabled=normalize_numbers,
            normalize_acronyms_enabled=normalize_acronyms,
            ensure_punctuation=ensure_punct,
            append_space=append_space,
        )
        if recovery:
            archive.save_transcript(archive_record, text, whisper_info=result.info)
            persisted = True
            backend.acknowledge(archive_record.audio_path)
        if not text.strip():
            recovery_queue_result = skipped_recovery_queue_result()
            update_dictation_archive_record(
                archive,
                archive_record,
                log_path,
                status="no_text",
                transcription={
                    "text_chars": 0,
                    "whisper_info": result.info,
                },
                clipboard=recovery_queue_result.clipboard_metadata(),
                recovery_queue=recovery_queue_result.recovery_queue_metadata(),
                injection={"mode": injection_mode.value, "succeeded": None},
            )
            write_log("No text produced from transcription", log_path)
            return
        save_dictation_archive_transcript(
            archive,
            archive_record,
            text,
            whisper_info=result.info,
            log_path=log_path,
        )
        recovery_queue_result = insert_transcript_into_recovery_queue(text, log_path)
        if recovery and backend._closed.is_set():
            archive.update_record(archive_record, status="saved_without_injection")
            return
        injected = inject_text(
            text,
            xdotool_bin,
            log_path,
            enable_beep=enable_beep,
            injection_mode=injection_mode,
            orca_daemon_dir=orca_daemon_dir,
            orca_session_id=orca_session_id,
        )
        update_dictation_archive_record(
            archive,
            archive_record,
            log_path,
            status="injected" if injected else "injection_failed",
            clipboard=recovery_queue_result.clipboard_metadata(),
            recovery_queue=recovery_queue_result.recovery_queue_metadata(),
            injection={"mode": injection_mode.value, "succeeded": injected},
        )
        if injected:
            write_log(f"Injected text: {text}", log_path)
        else:
            write_log(f"Text injection failed: {text}", log_path)
    except Exception as exc:  # pragma: no cover - defensive log
        update_dictation_archive_record(
            archive,
            archive_record,
            log_path,
            status="processing_failed",
            error=str(exc),
        )
        write_log(f"Capture processing failed: {exc}", log_path)
    finally:
        if recovery and archive_record:
            recovery.finish(archive_record, persisted)
        if (enhanced_path != audio_path and enhanced_path.exists()
                and (archive_record is None or enhanced_path != archive_record.audio_path)):
            enhanced_path.unlink(missing_ok=True)
        # A failed archive must never delete the only surviving audio.
        if not recovery or archive_record is not None:
            audio_path.unlink(missing_ok=True)


def start_dictation_archive_record(
    archive: Optional[DictationArchive],
    audio_path: Path,
    *,
    stats: Optional[RecorderStats],
    created_at: datetime,
    backend: Mapping[str, Any],
    log_path: Path,
) -> Optional[DictationRecord]:
    if archive is None:
        return None
    try:
        record = archive.start_record(
            audio_path,
            stats=stats,
            backend=backend,
            created_at=created_at,
        )
    except Exception as exc:
        write_log(f"Dictation archive audio save failed: {exc}", log_path)
        return None
    write_log(f"Archived capture audio: {record.audio_path}", log_path)
    return record


def save_dictation_archive_transcript(
    archive: Optional[DictationArchive],
    record: Optional[DictationRecord],
    text: str,
    *,
    whisper_info: Mapping[str, Any],
    log_path: Path,
) -> None:
    if archive is None or record is None:
        return
    try:
        archive.save_transcript(record, text, whisper_info=whisper_info)
    except Exception as exc:
        write_log(f"Dictation archive transcript save failed: {exc}", log_path)
        return
    write_log(f"Archived transcript: {record.transcript_path}", log_path)


def update_dictation_archive_record(
    archive: Optional[DictationArchive],
    record: Optional[DictationRecord],
    log_path: Path,
    **updates: Any,
) -> None:
    if archive is None or record is None:
        return
    try:
        archive.update_record(record, **updates)
    except Exception as exc:
        write_log(f"Dictation archive metadata update failed: {exc}", log_path)


def backend_archive_metadata(backend: WhisperBackend) -> dict[str, Any]:
    metadata = getattr(backend, "archive_metadata", None)
    if not callable(metadata):
        return {}
    try:
        return dict(metadata())
    except Exception:
        return {}


def inject_text(
    text: str,
    xdotool_bin: Path,
    log_path: Path,
    *,
    enable_beep: bool,
    injection_mode: InjectionMode = InjectionMode.XDOTOOL_TYPE,
    orca_daemon_dir: Optional[Path] = None,
    orca_session_id: Optional[str] = None,
) -> bool:
    if injection_mode == InjectionMode.ORCA_DAEMON:
        delivered = inject_text_orca_daemon(
            text,
            log_path,
            enable_beep=enable_beep,
            daemon_dir=orca_daemon_dir,
            preferred_session_id=orca_session_id,
        )
        if not delivered:
            write_log("Orca daemon injection failed; skipped xdotool fallback", log_path)
        return delivered

    if injection_mode == InjectionMode.WTYPE:
        delivered = type_text_with_wtype(text, log_path)
        if delivered and enable_beep:
            play_completion_beep(log_path)
        return delivered

    if injection_mode == InjectionMode.AUTO:
        return inject_text_auto(
            text,
            xdotool_bin,
            log_path,
            enable_beep=enable_beep,
            daemon_dir=orca_daemon_dir,
            preferred_session_id=orca_session_id,
        )

    delivered = type_text_with_xdotool(text, xdotool_bin, log_path)
    if delivered and enable_beep:
        play_completion_beep(log_path)
    return delivered


def inject_text_orca_daemon(
    text: str,
    log_path: Path,
    *,
    enable_beep: bool,
    daemon_dir: Optional[Path] = None,
    preferred_session_id: Optional[str] = None,
) -> bool:
    delivered = send_text_to_orca_daemon(
        text,
        log_path,
        daemon_dir=daemon_dir,
        preferred_session_id=preferred_session_id,
    )
    if delivered and enable_beep:
        play_completion_beep(log_path)
    return delivered


def inject_text_auto(
    text: str,
    xdotool_bin: Path,
    log_path: Path,
    *,
    enable_beep: bool,
    daemon_dir: Optional[Path] = None,
    preferred_session_id: Optional[str] = None,
) -> bool:
    active_window = get_active_window_info(xdotool_bin, log_path)
    if active_window and is_orca_window(active_window):
        write_log(
            f"Auto injection selected orca-clipboard-paste for {describe_active_window(active_window)}",
            log_path,
        )
        delivered = paste_text_with_clipboard_shortcut(text, xdotool_bin, log_path)
        if delivered:
            if enable_beep:
                play_completion_beep(log_path)
            return True

        write_log("Orca focused clipboard paste failed; falling back to daemon", log_path)
        delivered = inject_text_orca_daemon(
            text,
            log_path,
            enable_beep=enable_beep,
            daemon_dir=daemon_dir,
            preferred_session_id=preferred_session_id,
        )
        if not delivered:
            write_log("Orca daemon injection failed in auto mode; skipped xdotool typing fallback", log_path)
        return delivered

    if active_window and is_warp_window(active_window):
        write_log(f"Auto injection selected xdotool-type for Warp {describe_active_window(active_window)}", log_path)
    elif active_window:
        write_log(f"Auto injection selected xdotool-type for {describe_active_window(active_window)}", log_path)
    else:
        write_log("Auto injection could not identify active window; selected xdotool-type", log_path)

    delivered = type_text_with_xdotool(text, xdotool_bin, log_path)
    if delivered and enable_beep:
        play_completion_beep(log_path)
    return delivered


def type_text_with_wtype(text: str, log_path: Path) -> bool:
    """Type text into the focused window through the Wayland virtual keyboard.

    ``--`` matters: a transcript beginning with a dash would otherwise be parsed
    as wtype's own options and silently dropped.
    """
    try:
        wtype_bin = ensure_wtype()
    except RuntimeError as exc:
        write_log(str(exc), log_path)
        return False
    try:
        subprocess.run(
            [wtype_bin, "--", text],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or b"").decode(errors="replace").strip()
        write_log(f"wtype failed: {exc}{f' ({detail})' if detail else ''}", log_path)
        return False
    return True


def type_text_with_xdotool(text: str, xdotool_bin: Path, log_path: Path) -> bool:
    try:
        subprocess.run(
            [str(xdotool_bin), "type", "--clearmodifiers", text],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except subprocess.CalledProcessError as exc:
        write_log(f"xdotool failed: {exc}", log_path)
        return False
    return True


def paste_text_with_clipboard_shortcut(
    text: str,
    xdotool_bin: Path,
    log_path: Path,
    *,
    clipboard_bin: Optional[Path] = None,
) -> bool:
    xclip_bin = clipboard_bin or resolve_xclip_path()
    if xclip_bin is None:
        write_log("Clipboard paste failed: xclip not found", log_path)
        return False

    previous_clipboard = read_xclip_clipboard(xclip_bin, log_path)
    if previous_clipboard is None:
        return False

    if not write_xclip_clipboard(xclip_bin, text.encode("utf-8"), log_path):
        return False

    delivered = send_clipboard_paste_shortcut(xdotool_bin, log_path)
    time.sleep(CLIPBOARD_PASTE_SETTLE_SECONDS)
    restored = write_xclip_clipboard(xclip_bin, previous_clipboard, log_path)
    if not restored:
        write_log("Clipboard paste warning: previous clipboard restore failed", log_path)
    return delivered


def resolve_xclip_path() -> Optional[Path]:
    resolved = shutil.which("xclip")
    return Path(resolved) if resolved else None


def read_xclip_clipboard(xclip_bin: Path, log_path: Path) -> Optional[bytes]:
    try:
        completed = subprocess.run(
            [str(xclip_bin), "-selection", "clipboard", "-out"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=1.0,
        )
    except subprocess.CalledProcessError:
        return b""
    except (OSError, subprocess.TimeoutExpired) as exc:
        write_log(f"Clipboard read failed: {exc}", log_path)
        return None
    return completed.stdout


def write_xclip_clipboard(xclip_bin: Path, data: bytes, log_path: Path) -> bool:
    try:
        subprocess.run(
            [str(xclip_bin), "-selection", "clipboard", "-in"],
            input=data,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=1.0,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        write_log(f"Clipboard write failed: {exc}", log_path)
        return False
    return True


def send_clipboard_paste_shortcut(xdotool_bin: Path, log_path: Path) -> bool:
    try:
        subprocess.run(
            [str(xdotool_bin), "key", "--clearmodifiers", "ctrl+shift+v"],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        write_log(f"Clipboard paste shortcut failed: {exc}", log_path)
        return False
    return True


def get_active_window_info(xdotool_bin: Path, log_path: Path) -> Optional[ActiveWindowInfo]:
    try:
        window_id = subprocess.check_output(
            [str(xdotool_bin), "getactivewindow"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=1.0,
        ).strip()
        if not window_id:
            write_log("Active window detection failed: xdotool returned no window id", log_path)
            return None
        return inspect_x11_window(window_id, log_path)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError) as exc:
        write_log(f"Active window detection failed: {exc}", log_path)
    return None


def inspect_x11_window(window_id: str, log_path: Path) -> Optional[ActiveWindowInfo]:
    try:
        output = subprocess.check_output(
            ["xprop", "-id", window_id, "WM_CLASS", "WM_NAME", "_NET_WM_PID"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=1.0,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError) as exc:
        write_log(f"Active window xprop inspection failed for {window_id}: {exc}", log_path)
        return None

    wm_classes = parse_xprop_quoted_values(output, "WM_CLASS")
    name_values = parse_xprop_quoted_values(output, "WM_NAME")
    pid = parse_xprop_pid(output)
    return ActiveWindowInfo(
        window_id=window_id,
        wm_classes=tuple(wm_classes),
        name=name_values[0] if name_values else "",
        pid=pid,
        process_args=read_process_args(pid),
    )


def parse_xprop_quoted_values(output: str, property_name: str) -> list[str]:
    for line in output.splitlines():
        if line.startswith(f"{property_name}("):
            return re.findall(r'"([^"]*)"', line)
    return []


def parse_xprop_pid(output: str) -> Optional[int]:
    for line in output.splitlines():
        if line.startswith("_NET_WM_PID"):
            _, _, value = line.partition("=")
            return parse_positive_int(value.strip())
    return None


def read_process_args(pid: Optional[int]) -> str:
    if pid is None:
        return ""
    try:
        return subprocess.check_output(
            ["ps", "-p", str(pid), "-o", "args="],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=1.0,
        ).strip()
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        return ""


def is_orca_window(window: ActiveWindowInfo) -> bool:
    return "orca" in app_identity_haystack(window)


def is_warp_window(window: ActiveWindowInfo) -> bool:
    return "warp" in app_identity_haystack(window)


def app_identity_haystack(window: ActiveWindowInfo) -> str:
    # Window titles carry arbitrary user content (tab names, document text),
    # so app detection must rely on WM_CLASS and the owning process only.
    return " ".join([*window.wm_classes, window.process_args]).lower()


def describe_active_window(window: ActiveWindowInfo) -> str:
    classes = ",".join(window.wm_classes) or "unknown-class"
    name = window.name or "unknown-name"
    pid = window.pid if window.pid is not None else "unknown-pid"
    return f"window id={window.window_id} name={name!r} class={classes!r} pid={pid}"


def send_text_to_orca_daemon(
    text: str,
    log_path: Path,
    *,
    daemon_dir: Optional[Path] = None,
    preferred_session_id: Optional[str] = None,
) -> bool:
    daemon_paths = resolve_orca_daemon_paths(daemon_dir)
    if daemon_paths is None:
        write_log("Orca daemon send skipped: daemon socket/token not found", log_path)
        return False

    socket_path, token_path, protocol_version = daemon_paths
    try:
        token = token_path.read_text(encoding="utf-8").strip()
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
            client.settimeout(ORCA_DAEMON_TIMEOUT_SECONDS)
            client.connect(str(socket_path))
            stream = client.makefile("rwb")

            write_orca_daemon_frame(
                stream,
                {
                    "type": "hello",
                    "version": protocol_version,
                    "token": token,
                    "role": "control",
                    "clientId": str(uuid.uuid4()),
                },
            )
            hello_response = read_orca_daemon_frame(stream)
            if hello_response.get("ok") is not True:
                write_log(f"Orca daemon hello rejected: {hello_response}", log_path)
                return False

            write_orca_daemon_frame(
                stream,
                {"id": str(uuid.uuid4()), "type": "listSessions", "payload": {}},
            )
            sessions_response = read_orca_daemon_frame(stream)
            if sessions_response.get("ok") is not True:
                write_log(f"Orca daemon listSessions failed: {sessions_response}", log_path)
                return False

            sessions = extract_orca_sessions(sessions_response)
            session = select_orca_daemon_session(
                sessions,
                preferred_session_id=preferred_session_id,
                active_state_path=DEFAULT_ORCA_STATE_PATH,
            )
            if session is None:
                candidates = format_orca_session_candidates(sessions)
                write_log(
                    "Orca daemon send skipped: no unambiguous live terminal session found"
                    f"; candidates={candidates}",
                    log_path,
                )
                return False

            session_id = get_orca_session_id(session)
            if not session_id:
                write_log("Orca daemon send skipped: selected session has no id", log_path)
                return False

            write_orca_daemon_frame(
                stream,
                {
                    "id": str(uuid.uuid4()),
                    "type": "write",
                    "payload": {
                        "sessionId": session_id,
                        "data": build_orca_bracketed_paste(text),
                    },
                },
            )
            write_response = read_orca_daemon_frame(stream)
            if write_response.get("ok") is not True:
                write_log(f"Orca daemon write failed: {write_response}", log_path)
                return False

            cwd = session.get("cwd") or "unknown cwd"
            write_log(f"Orca daemon write succeeded for session {session_id} cwd={cwd}", log_path)
            return True
    except (OSError, TimeoutError, json.JSONDecodeError, RuntimeError) as exc:
        write_log(f"Orca daemon send failed: {exc}", log_path)
    return False


def resolve_orca_daemon_paths(daemon_dir: Optional[Path] = None) -> Optional[Tuple[Path, Path, int]]:
    search_dir = daemon_dir or Path.home() / ".config" / "orca" / "daemon"
    if not search_dir.exists():
        return None

    sockets = sorted(
        search_dir.glob("daemon-v*.sock"),
        key=parse_orca_daemon_version,
        reverse=True,
    )
    for socket_path in sockets:
        token_path = socket_path.with_suffix(".token")
        if token_path.exists():
            return socket_path, token_path, parse_orca_daemon_version(socket_path)
    return None


def parse_orca_daemon_version(path: Path) -> int:
    try:
        return int(path.stem.rsplit("v", maxsplit=1)[1])
    except (IndexError, ValueError):
        return ORCA_DAEMON_PROTOCOL_VERSION


def write_orca_daemon_frame(stream, message: dict[str, object]) -> None:
    stream.write(json.dumps(message).encode("utf-8") + b"\n")
    stream.flush()


def read_orca_daemon_frame(stream) -> dict[str, object]:
    line = stream.readline()
    if not line:
        raise RuntimeError("Orca daemon closed the connection")
    response = json.loads(line.decode("utf-8"))
    if not isinstance(response, dict):
        raise RuntimeError(f"Orca daemon returned a non-object frame: {response!r}")
    return response


def extract_orca_sessions(response: dict[str, object]) -> list[dict[str, object]]:
    payload = response.get("payload")
    if not isinstance(payload, dict):
        return []
    sessions = payload.get("sessions")
    if not isinstance(sessions, list):
        return []
    return [session for session in sessions if isinstance(session, dict)]


def select_orca_daemon_session(
    sessions: list[dict[str, object]],
    *,
    preferred_session_id: Optional[str] = None,
    active_state_path: Optional[Path] = None,
) -> Optional[dict[str, object]]:
    if preferred_session_id:
        return find_live_orca_session_by_id(sessions, preferred_session_id)

    if active_state_path is not None:
        active_session_id = read_active_orca_terminal_session_id(active_state_path)
        if active_session_id:
            active_session = find_live_orca_session_by_id(sessions, active_session_id)
            if active_session is not None:
                return active_session

    live_sessions = [session for session in sessions if is_live_orca_session(session)]
    if len(live_sessions) == 1:
        return live_sessions[0]

    codex_sessions = [
        session
        for session in live_sessions
        if (pid := parse_positive_int(session.get("pid"))) is not None
        and process_tree_contains(pid, "codex")
    ]
    if len(codex_sessions) == 1:
        return codex_sessions[0]
    return None


def find_live_orca_session_by_id(
    sessions: list[dict[str, object]],
    session_id: str,
) -> Optional[dict[str, object]]:
    return next(
        (
            session
            for session in sessions
            if is_live_orca_session(session) and get_orca_session_id(session) == session_id
        ),
        None,
    )


def read_active_orca_terminal_session_id(state_path: Path) -> Optional[str]:
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None

    if not isinstance(state, dict):
        return None
    workspace_session = state.get("workspaceSession")
    if not isinstance(workspace_session, dict):
        return None
    return resolve_active_orca_terminal_session_id(workspace_session)


def resolve_active_orca_terminal_session_id(
    workspace_session: dict[str, object],
) -> Optional[str]:
    active_worktree_id = coerce_nonempty_string(workspace_session.get("activeWorktreeId"))
    active_tab_ids = collect_active_orca_tab_ids(workspace_session, active_worktree_id)

    for active_tab_id in active_tab_ids:
        session_id = resolve_orca_tab_terminal_session_id(
            workspace_session,
            active_tab_id,
            active_worktree_id,
        )
        if session_id:
            return session_id
    return None


def collect_active_orca_tab_ids(
    workspace_session: dict[str, object],
    active_worktree_id: Optional[str],
) -> list[str]:
    active_tab_ids: list[str] = []

    active_tab_id_by_worktree = workspace_session.get("activeTabIdByWorktree")
    if active_worktree_id and isinstance(active_tab_id_by_worktree, dict):
        append_unique_nonempty_string(
            active_tab_ids,
            active_tab_id_by_worktree.get(active_worktree_id),
        )

    active_group_id = read_active_orca_group_id(workspace_session, active_worktree_id)
    tab_groups = workspace_session.get("tabGroups")
    if active_worktree_id and active_group_id and isinstance(tab_groups, dict):
        groups = tab_groups.get(active_worktree_id)
        if isinstance(groups, list):
            for group in groups:
                if not isinstance(group, dict) or group.get("id") != active_group_id:
                    continue
                append_unique_nonempty_string(active_tab_ids, group.get("activeTabId"))
                break

    append_unique_nonempty_string(active_tab_ids, workspace_session.get("activeTabId"))
    return active_tab_ids


def read_active_orca_group_id(
    workspace_session: dict[str, object],
    active_worktree_id: Optional[str],
) -> Optional[str]:
    if not active_worktree_id:
        return None
    active_group_id_by_worktree = workspace_session.get("activeGroupIdByWorktree")
    if not isinstance(active_group_id_by_worktree, dict):
        return None
    return coerce_nonempty_string(active_group_id_by_worktree.get(active_worktree_id))


def resolve_orca_tab_terminal_session_id(
    workspace_session: dict[str, object],
    active_tab_id: str,
    active_worktree_id: Optional[str],
) -> Optional[str]:
    return resolve_orca_layout_terminal_session_id(
        workspace_session,
        active_tab_id,
    ) or resolve_orca_tab_list_terminal_session_id(
        workspace_session,
        active_tab_id,
        active_worktree_id,
    )


def resolve_orca_layout_terminal_session_id(
    workspace_session: dict[str, object],
    active_tab_id: str,
) -> Optional[str]:
    terminal_layouts_by_tab_id = workspace_session.get("terminalLayoutsByTabId")
    if not isinstance(terminal_layouts_by_tab_id, dict):
        return None

    layout = terminal_layouts_by_tab_id.get(active_tab_id)
    if not isinstance(layout, dict):
        return None

    pty_ids_by_leaf_id = layout.get("ptyIdsByLeafId")
    if not isinstance(pty_ids_by_leaf_id, dict):
        return None

    active_leaf_id = coerce_nonempty_string(layout.get("activeLeafId"))
    if active_leaf_id:
        active_session_id = coerce_nonempty_string(pty_ids_by_leaf_id.get(active_leaf_id))
        if active_session_id:
            return active_session_id

    session_ids = [
        session_id
        for session_id in (
            coerce_nonempty_string(value) for value in pty_ids_by_leaf_id.values()
        )
        if session_id
    ]
    return session_ids[0] if len(session_ids) == 1 else None


def resolve_orca_tab_list_terminal_session_id(
    workspace_session: dict[str, object],
    active_tab_id: str,
    active_worktree_id: Optional[str],
) -> Optional[str]:
    tabs_by_worktree = workspace_session.get("tabsByWorktree")
    if not isinstance(tabs_by_worktree, dict):
        return None

    worktree_ids = list(tabs_by_worktree)
    if active_worktree_id in worktree_ids:
        worktree_ids.remove(active_worktree_id)
        worktree_ids.insert(0, active_worktree_id)

    for worktree_id in worktree_ids:
        tabs = tabs_by_worktree.get(worktree_id)
        if not isinstance(tabs, list):
            continue
        for tab in tabs:
            if not isinstance(tab, dict) or tab.get("id") != active_tab_id:
                continue
            return coerce_nonempty_string(tab.get("ptyId"))
    return None


def append_unique_nonempty_string(values: list[str], candidate: object) -> None:
    value = coerce_nonempty_string(candidate)
    if value and value not in values:
        values.append(value)


def coerce_nonempty_string(value: object) -> Optional[str]:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


def is_live_orca_session(session: dict[str, object]) -> bool:
    state = session.get("state")
    is_alive = session.get("isAlive")
    return is_alive is not False and state in (None, "", "running")


def get_orca_session_id(session: dict[str, object]) -> str:
    return str(session.get("sessionId") or session.get("id") or "")


def format_orca_session_candidates(sessions: list[dict[str, object]]) -> str:
    live_sessions = [session for session in sessions if is_live_orca_session(session)]
    if not live_sessions:
        return "none"
    formatted = []
    for session in live_sessions:
        formatted.append(
            "id="
            f"{get_orca_session_id(session) or 'unknown'} "
            f"pid={session.get('pid') or 'unknown'} "
            f"cwd={session.get('cwd') or 'unknown'}"
        )
    return "; ".join(formatted)


def parse_positive_int(value: object) -> Optional[int]:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def process_tree_contains(root_pid: int, needle: str) -> bool:
    try:
        output = subprocess.check_output(
            ["ps", "-eo", "pid=,ppid=,args="],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=1.0,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        return False

    rows: list[tuple[int, int, str]] = []
    for line in output.splitlines():
        parts = line.strip().split(maxsplit=2)
        if len(parts) < 2:
            continue
        pid = parse_positive_int(parts[0])
        parent_pid = parse_positive_int(parts[1])
        if pid is None or parent_pid is None:
            continue
        rows.append((pid, parent_pid, parts[2] if len(parts) == 3 else ""))

    needle_lower = needle.lower()
    descendant_pids = {root_pid}
    changed = True
    while changed:
        changed = False
        for pid, parent_pid, _args in rows:
            if parent_pid in descendant_pids and pid not in descendant_pids:
                descendant_pids.add(pid)
                changed = True

    return any(pid in descendant_pids and needle_lower in args.lower() for pid, _parent, args in rows)


def build_orca_bracketed_paste(text: str) -> str:
    return f"{ORCA_BRACKETED_PASTE_START}{text}{ORCA_BRACKETED_PASTE_END}"


def play_completion_beep(log_path: Path) -> None:
    play_system_bell(log_path, purpose="completion")


if __name__ == "__main__":
    main(sys.argv[1:])
