"""Hotkey management for wa_whisper push-to-talk."""

from __future__ import annotations

import shutil
import subprocess
import threading
from pathlib import Path
from typing import Callable, Optional

from pynput import keyboard

from .log_utils import write_log
from .recorder import Recorder, RecorderStartError, RecorderStats
from .x11_key_shield import X11KeyShield


class AudioMuteError(Exception):
    """Raised when system audio mute operations fail."""


class AudioMuteController:
    """Mute desktop audio while recording to avoid feedback loops."""

    def __init__(self, log_path: Path) -> None:
        self._log_path = log_path
        self._strategy: _MuteStrategy | None = self._detect_strategy()

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
        except AudioMuteError as exc:
            write_log(f"Failed to mute audio: {exc}", self._log_path)
            self._strategy = None

    def restore(self) -> None:
        if not self._strategy:
            return
        try:
            self._strategy.restore()
        except AudioMuteError as exc:
            write_log(f"Failed to restore audio: {exc}", self._log_path)
            self._strategy = None


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
            subprocess.run(["wpctl", "set-mute", self._TARGET, "1"], check=True)
        except subprocess.CalledProcessError as exc:
            raise AudioMuteError(f"wpctl mute failed: {exc}") from exc
        self._active = True
        write_log("Muted system audio (wpctl)", self._log_path)

    def restore(self) -> None:
        if not self._active:
            return
        try:
            if self._previously_muted is False:
                subprocess.run(["wpctl", "set-mute", self._TARGET, "0"], check=True)
                write_log("Restored system audio (wpctl)", self._log_path)
            else:
                write_log("Audio was muted before capture; left muted (wpctl)", self._log_path)
        except subprocess.CalledProcessError as exc:
            raise AudioMuteError(f"wpctl restore failed: {exc}") from exc
        finally:
            self._active = False
            self._previously_muted = None

    def _read_muted(self) -> Optional[bool]:
        try:
            output = subprocess.check_output(["wpctl", "get-volume", self._TARGET], text=True)
        except subprocess.CalledProcessError as exc:
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
            subprocess.run(["pactl", "set-sink-mute", self._sink, "1"], check=True)
        except subprocess.CalledProcessError as exc:
            raise AudioMuteError(f"pactl mute failed: {exc}") from exc
        self._active = True
        write_log(f"Muted system audio (pactl sink {self._sink})", self._log_path)

    def restore(self) -> None:
        if not self._active:
            return
        try:
            if self._previously_muted is False:
                subprocess.run(["pactl", "set-sink-mute", self._sink, "0"], check=True)
                write_log(f"Restored system audio (pactl sink {self._sink})", self._log_path)
            else:
                write_log("Audio was muted before capture; left muted (pactl)", self._log_path)
        except subprocess.CalledProcessError as exc:
            raise AudioMuteError(f"pactl restore failed: {exc}") from exc
        finally:
            self._active = False
            self._previously_muted = None

    def _detect_sink(self) -> str:
        try:
            output = subprocess.check_output(["pactl", "get-default-sink"], text=True)
        except subprocess.CalledProcessError as exc:
            raise AudioMuteError(f"pactl get-default-sink failed: {exc}") from exc
        return output.strip()

    def _read_muted(self) -> Optional[bool]:
        try:
            output = subprocess.check_output(["pactl", "get-sink-mute", self._sink], text=True)
        except subprocess.CalledProcessError as exc:
            raise AudioMuteError(f"pactl get-sink-mute failed: {exc}") from exc
        normalized = output.strip().lower()
        if "yes" in normalized:
            return True
        if "no" in normalized:
            return False
        return None


class PushToTalkHotkey:
    """Handle Right-Alt push-to-talk recording lifecycle."""

    def __init__(
        self,
        recorder: Recorder,
        *,
        silence_timeout: float,
        on_capture_finished: Callable[[Optional[Path], Optional[RecorderStats]], None],
        log_path: Path,
        enable_audio_mute: bool = True,
        exit_on_esc: bool = True,
        on_exit: Callable[[], None] | None = None,
        enable_hotkey_shield: bool = True,
    ) -> None:
        self._recorder = recorder
        self._silence_timeout = silence_timeout
        self._on_capture_finished = on_capture_finished
        self._log_path = log_path
        self._exit_on_esc = exit_on_esc
        self._on_exit = on_exit
        self._mute_controller = AudioMuteController(log_path) if enable_audio_mute else None
        self._hotkey_shield = X11KeyShield("Alt_R", log_path) if enable_hotkey_shield else None

        self._listener: keyboard.Listener | None = None
        self._active = False
        self._lock = threading.RLock()

    def start(self) -> None:
        """Begin listening for hotkey events."""
        if self._listener:
            return
        if self._hotkey_shield:
            self._hotkey_shield.start()
        self._listener = keyboard.Listener(
            on_press=self._handle_press,
            on_release=self._handle_release,
            suppress=False,
        )
        self._listener.start()
        write_log("Hotkey listener started (Right Alt)", self._log_path)

    def stop(self) -> None:
        """Stop listening and restore state."""
        with self._lock:
            if self._listener:
                self._listener.stop()
                self._listener = None
        if self._hotkey_shield:
            self._hotkey_shield.stop()
        self._restore_audio()
        write_log("Hotkey listener stopped", self._log_path)

    # Internal event handling ---------------------------------------------------------

    def _handle_press(self, key: keyboard.Key | keyboard.KeyCode) -> None:
        if key == keyboard.Key.alt_r:
            with self._lock:
                if self._active:
                    return
                self._active = True
            self._mute_audio()
            try:
                path = self._recorder.start()
            except RecorderStartError as exc:
                with self._lock:
                    self._active = False
                self._restore_audio()
                write_log(f"Recorder could not start: {exc}", self._log_path)
                return
            except Exception as exc:  # pragma: no cover - defensive guard
                with self._lock:
                    self._active = False
                self._restore_audio()
                write_log(f"Unexpected recorder failure: {exc}", self._log_path)
                return
            write_log(f"Right Alt pressed; recording -> {path}", self._log_path)
        elif self._exit_on_esc and key == keyboard.Key.esc:
            write_log("ESC pressed; terminating listener", self._log_path)
            self.stop()
            if self._on_exit:
                self._on_exit()

    def _handle_release(self, key: keyboard.Key | keyboard.KeyCode) -> None:
        if key != keyboard.Key.alt_r:
            return
        with self._lock:
            if not self._active:
                return
            self._active = False
        threading.Thread(target=self._finalize_capture, daemon=True).start()

    def _finalize_capture(self) -> None:
        result_path = self._recorder.stop(self._silence_timeout)
        stats = self._recorder.last_capture_stats()
        self._restore_audio()
        write_log("Recording finalized", self._log_path)
        self._on_capture_finished(result_path, stats)

    def _mute_audio(self) -> None:
        if self._mute_controller:
            self._mute_controller.mute()

    def _restore_audio(self) -> None:
        if self._mute_controller:
            self._mute_controller.restore()
