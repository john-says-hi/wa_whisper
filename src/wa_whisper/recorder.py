"""Audio capture utilities for wa_whisper."""

from __future__ import annotations

import contextlib
import math
import os
import queue
import re
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import sounddevice as sd
import soundfile as sf

from .log_utils import write_log


class RecorderStartError(Exception):
    """Raised when the recorder fails to start capturing audio."""

    def __init__(self, message: str, *, attempts: int) -> None:
        super().__init__(message)
        self.attempts = attempts


class RecorderInputChannelError(ValueError):
    """Raised when a configured input channel cannot be captured."""


@dataclass(slots=True)
class RecorderStats:
    """Summary statistics for the most recent capture."""

    total_ms: float
    speech_ms: float
    silence_ms: float
    max_rms: float
    avg_silence_rms: Optional[float]
    speech_blocks: int
    silence_blocks: int
    total_blocks: int
    max_speech_streak_ms: float

    @property
    def speech_ratio(self) -> Optional[float]:
        """Ratio of speech time to total time."""
        if self.total_ms <= 0:
            return None
        return self.speech_ms / self.total_ms

    @property
    def speech_max_db(self) -> Optional[float]:
        """Maximum speech RMS expressed in dBFS."""
        if self.max_rms <= 0:
            return None
        return 20 * math.log10(self.max_rms)

    @property
    def silence_avg_db(self) -> Optional[float]:
        """Average silence RMS expressed in dBFS."""
        if not self.avg_silence_rms or self.avg_silence_rms <= 0:
            return None
        return 20 * math.log10(self.avg_silence_rms)


class Recorder:
    """Stream audio from the default microphone into a temporary WAV file."""

    def __init__(
        self,
        sample_rate: int,
        device_index: Optional[int],
        log_path: Path,
        rms_threshold: float,
        preamp: float = 1.0,
        start_retry_attempts: int = 3,
        start_retry_delay: float = 0.2,
        input_channel: Optional[int] = None,
    ) -> None:
        if input_channel is not None and input_channel < 1:
            raise RecorderInputChannelError(
                f"input_channel must be a positive one-based channel number; got {input_channel}",
            )

        self.sample_rate = sample_rate
        self.device_index = device_index
        self.log_path = log_path
        self.rms_threshold = rms_threshold
        self.preamp = preamp
        self.input_channel = input_channel
        self._start_retry_attempts = max(1, start_retry_attempts)
        self._start_retry_delay = max(0.0, start_retry_delay)

        self._queue: queue.Queue[np.ndarray] = queue.Queue()
        self._stream: sd.InputStream | None = None
        self._writer: sf.SoundFile | None = None
        self._file: Path | None = None
        self._running = False
        self._worker: threading.Thread | None = None
        self._last_audio_time = 0.0

        # Capture statistics that feed silence gating.
        self._speech_duration_ms = 0.0
        self._silence_duration_ms = 0.0
        self._total_duration_ms = 0.0
        self._max_rms = 0.0
        self._silence_rms_sum = 0.0
        self._silence_block_count = 0
        self._speech_block_count = 0
        self._total_block_count = 0
        self._max_speech_streak_ms = 0.0
        self._current_speech_streak_ms = 0.0

        self._lock = threading.Lock()
        self._last_stats: RecorderStats | None = None
        self._resolved_input_device: int | None = None
        self._resolved_input_channels = 1
        self._resolved_input_name = "default"
        self._resolved_input_sample_rate = float(sample_rate)
        self._channel_selection_logged = False

    def start(self) -> Path:
        """Begin recording and return the output WAV path."""
        with self._lock:
            if self._running and self._file:
                return self._file

        last_error: Exception | None = None

        def audio_callback(indata, _frames, _time_info, status):
            if status:
                write_log(f"Audio status: {status}", self.log_path)
            self._queue.put(indata.copy())

        for attempt in range(1, self._start_retry_attempts + 1):
            with self._lock:
                # Reset state so we always begin from a clean queue.
                self._queue = queue.Queue()
                self._prepare_writer()

            stream: sd.InputStream | None = None
            try:
                (
                    self._resolved_input_device,
                    self._resolved_input_channels,
                    self._resolved_input_name,
                    self._resolved_input_sample_rate,
                ) = self._resolve_input_stream()
                self._validate_input_channel(self._resolved_input_channels)
                with self._lock:
                    # Reset state so we always begin from a clean queue.
                    self._queue = queue.Queue()
                    self._prepare_writer()
                stream = sd.InputStream(
                    samplerate=self._resolved_input_sample_rate,
                    device=self._resolved_input_device,
                    channels=self._resolved_input_channels,
                    dtype="float32",
                    callback=audio_callback,
                )
                stream.start()
            except RecorderInputChannelError:
                if stream is not None:
                    with contextlib.suppress(Exception):
                        stream.close()
                with self._lock:
                    self._running = False
                    self._teardown_stream()
                    self._close_writer(remove_file=True)
                raise
            except Exception as exc:  # pragma: no cover - exercised in tests via stub
                last_error = exc
                if stream is not None:
                    with contextlib.suppress(Exception):
                        stream.close()
                self._handle_failed_start(exc=exc, attempt=attempt)
                continue

            with self._lock:
                self._stream = stream
                self._running = True
                self._last_audio_time = time.time()
                self._reset_stats()

                self._worker = threading.Thread(target=self._drain_queue, daemon=True)
                self._worker.start()
                write_log(
                    "Recorder using input "
                    f"{self._resolved_input_name} "
                    "(device="
                    f"{self._resolved_input_device}, channels={self._resolved_input_channels}, "
                    f"sample_rate={self._resolved_input_sample_rate:.0f})",
                    self.log_path,
                )
                write_log(f"Recorder started -> {self._file}", self.log_path)
                return self._file

            # Success path shouldn't reach here, but guard anyway.
            break

        attempts = self._start_retry_attempts
        message = "Input stream failed to start"
        if last_error:
            message = f"{message}: {last_error}"
        raise RecorderStartError(message, attempts=attempts)

    def stop(self, timeout: float) -> Path | None:
        """Stop recording after `timeout` seconds of silence."""
        with self._lock:
            if not self._running:
                return None
            deadline = self._last_audio_time + timeout

        while time.time() < deadline:
            time.sleep(0.05)

        with self._lock:
            self._running = False
            self._teardown_stream()

        if self._worker:
            self._worker.join()
            self._worker = None

        self._close_writer(remove_file=False)

        stats = self._finalize_stats()
        self._last_stats = stats
        target = self._file
        write_log(f"Recorder stopped (stats={stats})", self.log_path)
        self._file = None
        return target

    def last_capture_stats(self) -> RecorderStats | None:
        """Return statistics for the most recent capture."""
        return self._last_stats

    # Internal helpers -----------------------------------------------------------------

    def _reset_stats(self) -> None:
        self._speech_duration_ms = 0.0
        self._silence_duration_ms = 0.0
        self._total_duration_ms = 0.0
        self._max_rms = 0.0
        self._silence_rms_sum = 0.0
        self._silence_block_count = 0
        self._speech_block_count = 0
        self._total_block_count = 0
        self._max_speech_streak_ms = 0.0
        self._current_speech_streak_ms = 0.0
        self._last_stats = None
        self._channel_selection_logged = False

    def _drain_queue(self) -> None:
        while self._running or not self._queue.empty():
            try:
                block = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue
            processed = self._prepare_block(block)
            if self._writer:
                self._writer.write(processed)
            self._accumulate_stats(processed)

    def _prepare_block(self, block: np.ndarray) -> np.ndarray:
        block = self._collapse_to_mono(block)
        if self.preamp != 1.0:
            block = np.clip(block * self.preamp, -1.0, 1.0)
        return block

    def _collapse_to_mono(self, block: np.ndarray) -> np.ndarray:
        channel_count = self._block_channel_count(block)
        if channel_count <= 0:
            return block

        if self.input_channel is not None:
            self._validate_input_channel(channel_count)
            selected_channel = self.input_channel - 1
            self._log_channel_selection(
                channel_count=channel_count,
                selected_channel=selected_channel,
                configured=True,
            )
            if hasattr(block, "shape"):
                return block[:, selected_channel : selected_channel + 1]
            if isinstance(block[0], (list, tuple)):
                return [[frame[selected_channel]] for frame in block]  # pragma: no cover - test shim
            return [[sample] for sample in block]  # pragma: no cover - test shim

        if channel_count <= 1:
            if getattr(block, "ndim", 1) == 1 and hasattr(block, "reshape"):
                return block.reshape(-1, 1)
            if getattr(block, "ndim", 1) == 1:
                return [[sample] for sample in block]  # pragma: no cover - test shim
            return block

        selected_channel = 0
        selected_rms = -1.0
        for channel_index in range(channel_count):
            rms = self._channel_rms(block, channel_index)
            if rms > selected_rms:
                selected_channel = channel_index
                selected_rms = rms

        self._log_channel_selection(
            channel_count=channel_count,
            selected_channel=selected_channel,
            configured=False,
        )

        if hasattr(block, "shape"):
            return block[:, selected_channel : selected_channel + 1]
        return [[frame[selected_channel]] for frame in block]  # pragma: no cover - test shim

    def _validate_input_channel(self, available_channel_count: int) -> None:
        if self.input_channel is None or self.input_channel <= available_channel_count:
            return
        raise RecorderInputChannelError(
            f"Configured input channel {self.input_channel} is unavailable; "
            f"the resolved input exposes {available_channel_count} channel(s)",
        )

    def _log_channel_selection(
        self,
        *,
        channel_count: int,
        selected_channel: int,
        configured: bool,
    ) -> None:
        if self._channel_selection_logged:
            return
        selection_kind = "configured channel" if configured else "channel"
        write_log(
            f"Recorder collapsing {channel_count} channels to mono via "
            f"{selection_kind} {selected_channel + 1}",
            self.log_path,
        )
        self._channel_selection_logged = True

    def _block_channel_count(self, block: np.ndarray) -> int:
        shape = getattr(block, "shape", None)
        if shape is not None:
            if len(shape) < 2:
                return 1
            return int(shape[1])
        if not block:
            return 0
        first_frame = block[0]
        if isinstance(first_frame, (list, tuple)):
            return len(first_frame)
        return 1

    def _channel_rms(self, block: np.ndarray, channel_index: int) -> float:
        total = 0.0
        frame_count = 0
        for frame in block:
            sample = frame[channel_index]
            total += float(sample) * float(sample)
            frame_count += 1
        if frame_count <= 0:
            return 0.0
        return math.sqrt(total / frame_count)

    def _resolve_input_stream(self) -> tuple[int | None, int, str, float]:
        resolved_device = self.device_index
        if resolved_device is None:
            resolved_device = self._resolve_linux_default_input_device()

        device_info = self._query_device_info(resolved_device)
        if not device_info:
            return resolved_device, 1, "default", float(self.sample_rate)

        channel_count = int(device_info.get("max_input_channels", 1) or 1)
        channel_count = max(1, channel_count)
        device_name = str(device_info.get("name") or resolved_device or "default")
        sample_rate = self._resolve_supported_sample_rate(
            resolved_device=resolved_device,
            device_name=device_name,
            channel_count=channel_count,
            device_info=device_info,
        )
        return resolved_device, channel_count, device_name, sample_rate

    def _resolve_supported_sample_rate(
        self,
        *,
        resolved_device: int | None,
        device_name: str,
        channel_count: int,
        device_info: dict,
    ) -> float:
        requested_sample_rate = float(self.sample_rate)
        if self._supports_input_settings(
            resolved_device=resolved_device,
            sample_rate=requested_sample_rate,
            channel_count=channel_count,
        ):
            return requested_sample_rate

        for fallback_sample_rate in self._candidate_sample_rates(
            requested_sample_rate=requested_sample_rate,
            device_info=device_info,
        ):
            if self._supports_input_settings(
                resolved_device=resolved_device,
                sample_rate=fallback_sample_rate,
                channel_count=channel_count,
            ):
                write_log(
                    f"Recorder sample rate fallback for {device_name}: "
                    f"{requested_sample_rate:.0f} -> {fallback_sample_rate:.0f}",
                    self.log_path,
                )
                return fallback_sample_rate

        return requested_sample_rate

    def _candidate_sample_rates(
        self,
        *,
        requested_sample_rate: float,
        device_info: dict,
    ) -> list[float]:
        device_default = float(
            device_info.get("default_samplerate", requested_sample_rate) or requested_sample_rate,
        )
        candidates = [
            device_default,
            44_100.0,
            48_000.0,
            32_000.0,
            22_050.0,
            16_000.0,
            96_000.0,
        ]
        return [
            candidate
            for index, candidate in enumerate(candidates)
            if candidate != requested_sample_rate and candidate not in candidates[:index]
        ]

    def _supports_input_settings(
        self,
        *,
        resolved_device: int | None,
        sample_rate: float,
        channel_count: int,
    ) -> bool:
        try:
            sd.check_input_settings(
                device=resolved_device,
                samplerate=sample_rate,
                channels=channel_count,
                dtype="float32",
            )
        except Exception:
            return False
        return True

    def _query_device_info(self, device_index: int | None) -> dict | None:
        try:
            if device_index is None:
                return sd.query_devices(None, "input")
            return sd.query_devices(device_index)
        except Exception:  # pragma: no cover - defensive guard for unavailable hosts
            return None

    def _resolve_linux_default_input_device(self) -> int | None:
        default_source_description = self._read_default_source_description()
        if not default_source_description:
            return None

        normalized_target = self._normalize_device_label(default_source_description)
        try:
            devices = sd.query_devices()
        except Exception:  # pragma: no cover - defensive guard for unavailable hosts
            return None

        for device_index, device in enumerate(devices):
            max_input_channels = int(device.get("max_input_channels", 0) or 0)
            if max_input_channels <= 0:
                continue
            device_name = str(device.get("name") or "")
            normalized_name = self._normalize_device_label(device_name)
            if not normalized_name:
                continue
            if normalized_name == normalized_target or normalized_target in normalized_name:
                return device_index
        return None

    def _read_default_source_description(self) -> str | None:
        try:
            pactl_info = subprocess.check_output(["pactl", "info"], text=True)
            pactl_sources = subprocess.check_output(["pactl", "list", "sources"], text=True)
        except (FileNotFoundError, subprocess.CalledProcessError):
            return None

        default_source_name: str | None = None
        for line in pactl_info.splitlines():
            if line.startswith("Default Source:"):
                default_source_name = line.split(":", 1)[1].strip()
                break

        if not default_source_name:
            return None

        descriptions = self._parse_source_descriptions(pactl_sources)
        return descriptions.get(default_source_name)

    def _parse_source_descriptions(self, pactl_sources: str) -> dict[str, str]:
        descriptions: dict[str, str] = {}
        current_name: str | None = None
        for line in pactl_sources.splitlines():
            stripped = line.strip()
            if stripped.startswith("Name:"):
                current_name = stripped.split(":", 1)[1].strip()
                continue
            if stripped.startswith("Description:") and current_name:
                descriptions[current_name] = stripped.split(":", 1)[1].strip()
                current_name = None
        return descriptions

    def _normalize_device_label(self, value: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", value.lower())

    def _accumulate_stats(self, block: np.ndarray) -> None:
        frames = int(block.shape[0]) if block.ndim > 0 else 0
        if frames <= 0:
            return
        block_ms = (frames / self._resolved_input_sample_rate) * 1000.0
        rms = float(np.sqrt(np.mean(np.square(block), dtype=np.float64)))

        with self._lock:
            self._total_duration_ms += block_ms
            self._total_block_count += 1
            self._max_rms = max(self._max_rms, rms)

            if rms > self.rms_threshold:
                self._last_audio_time = time.time()
                self._speech_duration_ms += block_ms
                self._speech_block_count += 1
                self._current_speech_streak_ms += block_ms
                if self._current_speech_streak_ms > self._max_speech_streak_ms:
                    self._max_speech_streak_ms = self._current_speech_streak_ms
            else:
                self._silence_duration_ms += block_ms
                self._current_speech_streak_ms = 0.0
                if rms > 0:
                    self._silence_rms_sum += rms
                    self._silence_block_count += 1

    def _finalize_stats(self) -> RecorderStats:
        avg_silence_rms: Optional[float] = None
        if self._silence_block_count > 0:
            avg_silence_rms = self._silence_rms_sum / self._silence_block_count
        return RecorderStats(
            total_ms=self._total_duration_ms,
            speech_ms=self._speech_duration_ms,
            silence_ms=self._silence_duration_ms,
            max_rms=self._max_rms,
            avg_silence_rms=avg_silence_rms,
            speech_blocks=self._speech_block_count,
            silence_blocks=self._silence_block_count,
            total_blocks=self._total_block_count,
            max_speech_streak_ms=self._max_speech_streak_ms,
        )

    # Internal setup/teardown helpers ------------------------------------------------

    def _prepare_writer(self) -> None:
        self._close_writer(remove_file=True)
        tmp_dir = Path(tempfile.gettempdir()) / "wa_whisper"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        fd, filename = tempfile.mkstemp(suffix=".wav", dir=tmp_dir)
        os.close(fd)
        self._file = Path(filename)
        self._writer = sf.SoundFile(
            str(self._file),
            mode="w",
            samplerate=int(self._resolved_input_sample_rate),
            channels=1,
            subtype="PCM_16",
        )

    def _handle_failed_start(self, *, exc: Exception, attempt: int) -> None:
        attempts = self._start_retry_attempts
        write_log(
            f"Recorder start failed (attempt {attempt}/{attempts}): {exc}",
            self.log_path,
        )
        with self._lock:
            self._running = False
            self._teardown_stream()
            self._close_writer(remove_file=True)
        if attempt < attempts and self._start_retry_delay > 0:
            time.sleep(self._start_retry_delay)

    def _teardown_stream(self) -> None:
        if not self._stream:
            return
        with contextlib.suppress(Exception):
            self._stream.stop()
        with contextlib.suppress(Exception):
            self._stream.close()
        self._stream = None

    def _close_writer(self, *, remove_file: bool) -> None:
        if self._writer:
            with contextlib.suppress(Exception):
                self._writer.close()
            self._writer = None
        if remove_file and self._file:
            with contextlib.suppress(FileNotFoundError):
                self._file.unlink()
            self._file = None
