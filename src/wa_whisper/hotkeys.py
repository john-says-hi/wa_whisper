"""Hotkey management for wa_whisper push-to-talk."""

from __future__ import annotations

import shutil
import subprocess
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Callable, Optional

from pynput import keyboard

from .audio_cues import play_system_bell
from .evdev_listener import EvdevKeyListener, wayland_session
from .log_utils import write_log
from .recorder import Recorder, RecorderStartError, RecorderStats
from .x11_key_shield import X11KeyShield


DEFAULT_HOTKEY_REPRESS_GRACE_SECONDS = 1.0
HOTKEY_RELEASE_TIMEOUT_SECONDS = 5.0
AUDIO_CONTROL_TIMEOUT_SECONDS = 1.0


class RecordingMode(Enum):
    """Current hotkey-controlled recording mode."""

    IDLE = "idle"
    PUSH_TO_TALK = "push_to_talk"
    HANDS_FREE = "hands_free"


class CaptureEndReason(Enum):
    """Reason a capture ended."""

    USER_COMPLETED = "user_completed"
    SERVICE_SHUTDOWN = "service_shutdown"
    ESCAPE = "escape"


@dataclass(frozen=True, slots=True)
class CaptureResult:
    """Completed capture passed from hotkey handling to processing."""

    path: Optional[Path]
    stats: Optional[RecorderStats]
    created_at: datetime
    end_reason: CaptureEndReason


@dataclass(frozen=True, slots=True)
class _FinalizationRequest:
    created_at: datetime
    silence_timeout: float
    end_reason: CaptureEndReason
    play_off_bell: bool
    wait_for_hotkey_release: bool


CaptureFinishedCallback = Callable[[CaptureResult], None]


def utc_now() -> datetime:
    """Return an aware UTC timestamp for a newly started capture."""
    return datetime.now(timezone.utc)


class AudioMuteError(Exception):
    """Raised when system audio mute operations fail."""


class AudioMuteController:
    """Mute desktop audio while recording to avoid feedback loops."""

    def __init__(self, log_path: Path) -> None:
        self._log_path = log_path
        self._strategy: _MuteStrategy | None = None
        try:
            self._strategy = self._detect_strategy()
        except Exception as exc:
            write_log(f"Auto-mute disabled: initialization failed: {exc}", self._log_path)

    def _detect_strategy(self) -> "_MuteStrategy | None":
        if shutil.which("wpctl"):
            return _WpctlStrategy(self._log_path)
        if shutil.which("pactl"):
            return _PactlStrategy(self._log_path)
        write_log("Auto-mute disabled: wpctl/pactl not found", self._log_path)
        return None

    def mute(self) -> None:
        if not self._strategy:
            return
        try:
            self._strategy.mute()
        except Exception as exc:
            write_log(f"Failed to mute audio: {exc}", self._log_path)

    def restore(self) -> None:
        if not self._strategy:
            return
        try:
            self._strategy.restore()
        except Exception as exc:
            write_log(f"Failed to restore audio: {exc}", self._log_path)


class _MuteStrategy:
    def __init__(self, log_path: Path) -> None:
        self._log_path = log_path
        self._active = False
        self._previously_muted: Optional[bool] = None

    def mute(self) -> None:
        raise NotImplementedError

    def restore(self) -> None:
        raise NotImplementedError


class _WpctlStrategy(_MuteStrategy):
    _TARGET = "@DEFAULT_AUDIO_SINK@"

    def mute(self) -> None:
        if self._active:
            return
        self._previously_muted = self._read_muted()
        try:
            subprocess.run(
                ["wpctl", "set-mute", self._TARGET, "1"],
                check=True,
                timeout=AUDIO_CONTROL_TIMEOUT_SECONDS,
            )
        except (subprocess.SubprocessError, OSError) as exc:
            raise AudioMuteError(f"wpctl mute failed: {exc}") from exc
        self._active = True
        write_log("Muted system audio (wpctl)", self._log_path)

    def restore(self) -> None:
        if not self._active:
            return
        try:
            if self._previously_muted is False:
                subprocess.run(
                    ["wpctl", "set-mute", self._TARGET, "0"],
                    check=True,
                    timeout=AUDIO_CONTROL_TIMEOUT_SECONDS,
                )
                write_log("Restored system audio (wpctl)", self._log_path)
            else:
                write_log("Audio was muted before capture; left muted (wpctl)", self._log_path)
        except (subprocess.SubprocessError, OSError) as exc:
            raise AudioMuteError(f"wpctl restore failed: {exc}") from exc
        finally:
            self._active = False
            self._previously_muted = None

    def _read_muted(self) -> Optional[bool]:
        try:
            output = subprocess.check_output(
                ["wpctl", "get-volume", self._TARGET],
                text=True,
                timeout=AUDIO_CONTROL_TIMEOUT_SECONDS,
            )
        except (subprocess.SubprocessError, OSError) as exc:
            raise AudioMuteError(f"wpctl get-volume failed: {exc}") from exc
        normalized = output.strip().lower()
        if "muted:" in normalized:
            return "muted: yes" in normalized
        if "[muted]" in normalized:
            return True
        return False


class _PactlStrategy(_MuteStrategy):
    def __init__(self, log_path: Path) -> None:
        super().__init__(log_path)
        self._sink = self._detect_sink()

    def mute(self) -> None:
        if self._active:
            return
        self._previously_muted = self._read_muted()
        try:
            subprocess.run(
                ["pactl", "set-sink-mute", self._sink, "1"],
                check=True,
                timeout=AUDIO_CONTROL_TIMEOUT_SECONDS,
            )
        except (subprocess.SubprocessError, OSError) as exc:
            raise AudioMuteError(f"pactl mute failed: {exc}") from exc
        self._active = True
        write_log(f"Muted system audio (pactl sink {self._sink})", self._log_path)

    def restore(self) -> None:
        if not self._active:
            return
        try:
            if self._previously_muted is False:
                subprocess.run(
                    ["pactl", "set-sink-mute", self._sink, "0"],
                    check=True,
                    timeout=AUDIO_CONTROL_TIMEOUT_SECONDS,
                )
                write_log(f"Restored system audio (pactl sink {self._sink})", self._log_path)
            else:
                write_log("Audio was muted before capture; left muted (pactl)", self._log_path)
        except (subprocess.SubprocessError, OSError) as exc:
            raise AudioMuteError(f"pactl restore failed: {exc}") from exc
        finally:
            self._active = False
            self._previously_muted = None

    def _detect_sink(self) -> str:
        try:
            output = subprocess.check_output(
                ["pactl", "get-default-sink"],
                text=True,
                timeout=AUDIO_CONTROL_TIMEOUT_SECONDS,
            )
        except (subprocess.SubprocessError, OSError) as exc:
            raise AudioMuteError(f"pactl get-default-sink failed: {exc}") from exc
        return output.strip()

    def _read_muted(self) -> Optional[bool]:
        try:
            output = subprocess.check_output(
                ["pactl", "get-sink-mute", self._sink],
                text=True,
                timeout=AUDIO_CONTROL_TIMEOUT_SECONDS,
            )
        except (subprocess.SubprocessError, OSError) as exc:
            raise AudioMuteError(f"pactl get-sink-mute failed: {exc}") from exc
        normalized = output.strip().lower()
        if "yes" in normalized:
            return True
        if "no" in normalized:
            return False
        return None


class PushToTalkHotkey:
    """Handle push-to-talk and hands-free recording lifecycles."""

    def __init__(
        self,
        recorder: Recorder,
        *,
        silence_timeout: float,
        on_capture_finished: CaptureFinishedCallback,
        log_path: Path,
        enable_audio_mute: bool = True,
        exit_on_esc: bool = True,
        on_exit: Callable[[CaptureEndReason], None] | None = None,
        enable_hotkey_shield: bool = True,
        hotkey_repress_grace_seconds: float = DEFAULT_HOTKEY_REPRESS_GRACE_SECONDS,
    ) -> None:
        self._recorder = recorder
        self._silence_timeout = silence_timeout
        self._on_capture_finished = on_capture_finished
        self._log_path = log_path
        self._exit_on_esc = exit_on_esc
        self._on_exit = on_exit
        self._mute_controller = AudioMuteController(log_path) if enable_audio_mute else None
        # The shield is an X11 root-window grab. Under Wayland there is no root
        # window to grab and the Xlib call fails, so it is skipped entirely --
        # which means a bare Right Alt still reaches the focused window there.
        self._use_evdev = wayland_session()
        self._hotkey_shield = (
            X11KeyShield("Alt_R", log_path)
            if enable_hotkey_shield and not self._use_evdev
            else None
        )
        self._hotkey_repress_grace_seconds = max(0.0, hotkey_repress_grace_seconds)

        self._listener: keyboard.Listener | EvdevKeyListener | None = None
        self._mode = RecordingMode.IDLE
        self._left_ctrl_down = False
        self._right_alt_down = False
        self._finalizing_capture = False
        self._finalization_owner_ident: int | None = None
        self._ignore_presses_until = 0.0
        self._capture_started_at: datetime | None = None
        self._events_enabled = True
        self._exit_requested = False
        self._lifecycle_closed = False
        self._listening_resources_active = False
        self._lock = threading.RLock()
        self._state_changed = threading.Condition(self._lock)

    def _create_listener(self) -> "keyboard.Listener | EvdevKeyListener":
        """Pick a hotkey backend that can actually see keys in this session.

        pynput reads the keyboard through X11, which a Wayland compositor will
        not expose. It starts without error and then never fires, so the choice
        has to be made up front rather than discovered at the first keypress.
        Both backends deliver the same ``pynput`` key objects, so everything
        downstream of here is identical.
        """
        if self._use_evdev:
            return EvdevKeyListener(
                on_press=self._handle_press,
                on_release=self._handle_release,
                log_path=self._log_path,
            )
        return keyboard.Listener(
            on_press=self._handle_press,
            on_release=self._handle_release,
            suppress=False,
        )

    def start(self) -> bool:
        """Begin listening for hotkey events."""
        with self._state_changed:
            if self._lifecycle_closed:
                write_log("Hotkey listener start skipped: lifecycle is closed", self._log_path)
                return False
            if self._listener:
                return True
            self._events_enabled = False
            self._exit_requested = False
            listener: keyboard.Listener | EvdevKeyListener | None = None
            try:
                if self._hotkey_shield:
                    self._hotkey_shield.start()
                if self._lifecycle_closed:
                    self._stop_listening_resources(None)
                    return False

                listener = self._create_listener()
                listener.start()
            except Exception:
                self._stop_listening_resources(listener)
                self._events_enabled = False
                self._state_changed.notify_all()
                raise

            if self._lifecycle_closed:
                self._stop_listening_resources(listener)
                return False

            self._listener = listener
            self._listening_resources_active = True
            self._events_enabled = True
            self._state_changed.notify_all()

            if self._lifecycle_closed or self._listener is not listener:
                owns_resources = self._listener is listener
                if owns_resources:
                    self._listener = None
                self._listening_resources_active = False
                self._events_enabled = False
                self._state_changed.notify_all()
                if owns_resources:
                    self._stop_listening_resources(listener)
                return False

            write_log(
                "Hotkey listener started (Right Alt push-to-talk; "
                "Left Ctrl + Right Alt hands-free)",
                self._log_path,
            )

            # Python signal handlers can re-enter this RLock on the main thread.
            # If shutdown ran during startup, it owns resource cleanup and the
            # terminal lifecycle guard prevents this method from reviving it.
            return not self._lifecycle_closed and self._listener is listener

    def stop(
        self,
        end_reason: CaptureEndReason = CaptureEndReason.SERVICE_SHUTDOWN,
    ) -> None:
        """Stop listening and synchronously interrupt any active capture."""
        with self._state_changed:
            self._lifecycle_closed = True
            self._events_enabled = False
            listener = self._listener
            self._listener = None
            should_stop_resources = self._listening_resources_active or listener is not None
            self._listening_resources_active = False
            self._state_changed.notify_all()
            finalization = self._claim_finalization_locked(
                silence_timeout=0.0,
                end_reason=end_reason,
                play_off_bell=False,
                wait_for_hotkey_release=False,
                apply_repress_grace=False,
            )

        if should_stop_resources:
            self._stop_listening_resources(listener)

        if finalization:
            self._finalize_capture(finalization)
        else:
            self._wait_for_in_progress_finalization()

        self._restore_audio()
        write_log("Hotkey listener stopped", self._log_path)

    # Internal event handling ---------------------------------------------------------

    def _handle_press(
        self,
        key: keyboard.Key | keyboard.KeyCode,
        injected: bool = False,
    ) -> None:
        if injected:
            return

        if key == keyboard.Key.ctrl_l:
            with self._state_changed:
                if self._events_enabled and not self._left_ctrl_down:
                    self._left_ctrl_down = True
                    self._state_changed.notify_all()
            return

        if key == keyboard.Key.alt_r:
            self._handle_right_alt_press()
            return

        if self._exit_on_esc and key == keyboard.Key.esc:
            with self._lock:
                if not self._events_enabled or self._exit_requested:
                    return
                self._exit_requested = True
            write_log("ESC pressed; requesting shutdown", self._log_path)
            if self._on_exit:
                self._on_exit(CaptureEndReason.ESCAPE)
            else:
                self.stop(CaptureEndReason.ESCAPE)

    def _handle_release(
        self,
        key: keyboard.Key | keyboard.KeyCode,
        injected: bool = False,
    ) -> None:
        if injected:
            return

        if key == keyboard.Key.ctrl_l:
            with self._state_changed:
                if self._left_ctrl_down:
                    self._left_ctrl_down = False
                    self._state_changed.notify_all()
            return

        if key != keyboard.Key.alt_r:
            return

        with self._state_changed:
            if self._right_alt_down:
                self._right_alt_down = False
                self._state_changed.notify_all()
            if self._mode is not RecordingMode.PUSH_TO_TALK:
                return
            finalization = self._claim_finalization_locked(
                silence_timeout=self._silence_timeout,
                end_reason=CaptureEndReason.USER_COMPLETED,
                play_off_bell=False,
                wait_for_hotkey_release=False,
                apply_repress_grace=True,
            )
            if finalization:
                self._launch_finalization_locked(finalization)

    def _handle_right_alt_press(self) -> None:
        with self._state_changed:
            if not self._events_enabled or self._right_alt_down:
                return
            self._right_alt_down = True
            self._state_changed.notify_all()

            if self._mode is RecordingMode.HANDS_FREE:
                if self._left_ctrl_down:
                    finalization = self._claim_finalization_locked(
                        silence_timeout=self._silence_timeout,
                        end_reason=CaptureEndReason.USER_COMPLETED,
                        play_off_bell=True,
                        wait_for_hotkey_release=True,
                        apply_repress_grace=True,
                    )
                    if finalization:
                        self._launch_finalization_locked(finalization)
                else:
                    write_log("Right Alt ignored while hands-free recording is active", self._log_path)
                return

            if self._mode is not RecordingMode.IDLE:
                return

            ignore_reason = self._ignored_press_reason(time.monotonic())
            if ignore_reason:
                write_log(f"Right Alt press ignored {ignore_reason}", self._log_path)
                return

            requested_mode = (
                RecordingMode.HANDS_FREE if self._left_ctrl_down else RecordingMode.PUSH_TO_TALK
            )
            self._start_capture_locked(requested_mode)

    def _start_capture_locked(self, requested_mode: RecordingMode) -> None:
        self._mode = requested_mode
        self._capture_started_at = None
        if requested_mode is RecordingMode.HANDS_FREE:
            self._play_toggle_bell("hands-free on")
        self._mute_audio()

        try:
            path = self._recorder.start()
        except RecorderStartError as exc:
            self._rollback_failed_start(
                requested_mode,
                f"Recorder could not start: {exc}",
            )
            return
        except Exception as exc:  # pragma: no cover - defensive guard
            self._rollback_failed_start(
                requested_mode,
                f"Unexpected recorder failure: {exc}",
            )
            return

        self._capture_started_at = utc_now()
        write_log(f"Recording started in {requested_mode.value} mode -> {path}", self._log_path)

    def _rollback_failed_start(self, requested_mode: RecordingMode, message: str) -> None:
        self._mode = RecordingMode.IDLE
        self._restore_audio()
        if requested_mode is RecordingMode.HANDS_FREE:
            self._play_toggle_bell("hands-free start failed")
        write_log(message, self._log_path)

    def _claim_finalization_locked(
        self,
        *,
        silence_timeout: float,
        end_reason: CaptureEndReason,
        play_off_bell: bool,
        wait_for_hotkey_release: bool,
        apply_repress_grace: bool,
    ) -> _FinalizationRequest | None:
        if self._mode is RecordingMode.IDLE or self._finalizing_capture:
            return None

        created_at = self._capture_started_at or utc_now()
        self._mode = RecordingMode.IDLE
        self._capture_started_at = None
        self._finalizing_capture = True
        if apply_repress_grace:
            self._ignore_presses_until = time.monotonic() + self._hotkey_repress_grace_seconds
        return _FinalizationRequest(
            created_at=created_at,
            silence_timeout=silence_timeout,
            end_reason=end_reason,
            play_off_bell=play_off_bell,
            wait_for_hotkey_release=wait_for_hotkey_release,
        )

    def _launch_finalization_locked(self, finalization: _FinalizationRequest) -> None:
        threading.Thread(
            target=self._finalize_capture,
            args=(finalization,),
            daemon=True,
        ).start()

    def _finalize_capture(self, finalization: _FinalizationRequest) -> None:
        with self._state_changed:
            self._finalization_owner_ident = threading.get_ident()

        result_path: Path | None = None
        stats: RecorderStats | None = None
        try:
            try:
                result_path = self._recorder.stop(finalization.silence_timeout)
                stats = self._recorder.last_capture_stats()
            except Exception as exc:  # pragma: no cover - exercised by focused test doubles
                write_log(f"Recorder failed while finalizing capture: {exc}", self._log_path)
            finally:
                self._restore_audio()

            if finalization.play_off_bell:
                self._play_toggle_bell("hands-free off")
            if finalization.wait_for_hotkey_release:
                self._wait_for_hotkey_release()

            write_log(
                f"Recording finalized ({finalization.end_reason.value})",
                self._log_path,
            )
            result = CaptureResult(
                path=result_path,
                stats=stats,
                created_at=finalization.created_at,
                end_reason=finalization.end_reason,
            )
            try:
                self._on_capture_finished(result)
            except Exception as exc:  # pragma: no cover - callback boundary guard
                write_log(f"Capture callback failed: {exc}", self._log_path)
        finally:
            with self._state_changed:
                self._finalizing_capture = False
                self._finalization_owner_ident = None
                self._state_changed.notify_all()

    def _wait_for_hotkey_release(self) -> None:
        deadline = time.monotonic() + HOTKEY_RELEASE_TIMEOUT_SECONDS
        with self._state_changed:
            while self._events_enabled and (self._left_ctrl_down or self._right_alt_down):
                remaining_seconds = deadline - time.monotonic()
                if remaining_seconds <= 0:
                    write_log(
                        f"Timed out after {HOTKEY_RELEASE_TIMEOUT_SECONDS:.2f}s "
                        "waiting for hands-free hotkeys to be released",
                        self._log_path,
                    )
                    return
                self._state_changed.wait(timeout=remaining_seconds)

    def _wait_for_in_progress_finalization(self) -> None:
        with self._state_changed:
            while self._finalizing_capture:
                if self._finalization_owner_ident == threading.get_ident():
                    return
                self._state_changed.wait()

    def _ignored_press_reason(self, now: float) -> str | None:
        if self._finalizing_capture:
            return "during capture finalization"
        remaining_grace_seconds = self._ignore_presses_until - now
        if remaining_grace_seconds > 0:
            return f"for {remaining_grace_seconds:.2f}s grace period"
        return None

    def _mute_audio(self) -> None:
        if self._mute_controller:
            try:
                self._mute_controller.mute()
            except Exception as exc:  # pragma: no cover - defensive boundary guard
                write_log(f"Unexpected audio mute failure: {exc}", self._log_path)

    def _restore_audio(self) -> None:
        if self._mute_controller:
            try:
                self._mute_controller.restore()
            except Exception as exc:  # pragma: no cover - defensive boundary guard
                write_log(f"Unexpected audio restore failure: {exc}", self._log_path)

    def _play_toggle_bell(self, purpose: str) -> None:
        try:
            play_system_bell(self._log_path, purpose=purpose)
        except Exception as exc:  # pragma: no cover - defensive boundary guard
            write_log(f"System bell failed for {purpose}: {exc}", self._log_path)

    def _stop_listening_resources(self, listener: keyboard.Listener | None) -> None:
        if listener:
            try:
                listener.stop()
            except Exception as exc:  # pragma: no cover - defensive cleanup guard
                write_log(f"Hotkey listener cleanup failed: {exc}", self._log_path)
        if self._hotkey_shield:
            try:
                self._hotkey_shield.stop()
            except Exception as exc:  # pragma: no cover - defensive cleanup guard
                write_log(f"Hotkey shield cleanup failed: {exc}", self._log_path)
