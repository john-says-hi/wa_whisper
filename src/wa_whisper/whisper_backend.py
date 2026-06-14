"""Wrapper around OpenAI Whisper for wa_whisper."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import whisper

from .log_utils import write_log

DEFAULT_MODEL_NAME = "large-v3"
DEFAULT_MODEL_CACHE = Path.home() / ".cache" / "huggingface" / "hub"


@dataclass(slots=True)
class WhisperConfig:
    """Configuration parameters for Whisper decoding."""

    model_name: str = DEFAULT_MODEL_NAME
    temperature: float = 0.0
    beam_size: int = 5
    best_of: int = 5
    patience: Optional[float] = None
    compression_ratio_threshold: float = 2.4
    logprob_threshold: Optional[float] = None
    no_speech_threshold: Optional[float] = None
    initial_prompt: Optional[str] = None
    suppress_tokens: Optional[str] = "-1"
    condition_on_previous_text: bool = False
    cache_dir: Path = DEFAULT_MODEL_CACHE
    device: Optional[str] = None
    compute_mode: Optional[str] = None
    fp16: Optional[bool] = None


@dataclass(slots=True)
class WhisperSegment:
    """Individual segment output from Whisper."""

    text: str
    start: float
    end: float
    avg_logprob: Optional[float] = None
    compression_ratio: Optional[float] = None
    no_speech_prob: Optional[float] = None


@dataclass(slots=True)
class WhisperResult:
    """Structured transcription result."""

    text: str
    segments: List[WhisperSegment] = field(default_factory=list)
    info: Dict[str, Any] = field(default_factory=dict)


class WhisperBackend:
    """Lazy-loading wrapper that enforces English transcription."""

    def __init__(self, config: WhisperConfig, log_path: Path) -> None:
        self._config = config
        self._log_path = log_path
        self._model: whisper.Whisper | None = None
        self._device = self._resolve_device(config.device)
        self._fp16 = config.fp16 if config.fp16 is not None else self._device != "cpu"
        self._lock = threading.Lock()
        self._log_compute_selection()

    def _resolve_device(self, requested: Optional[str]) -> str:
        if requested:
            if self._config.compute_mode == "gpu" and requested == "cuda" and not torch.cuda.is_available():
                raise RuntimeError(
                    "gpu compute mode requires CUDA, but torch reports CUDA is unavailable. "
                    "Run `wa-whisper-mode ram` to switch to CPU/system RAM mode.",
                )
            return requested
        if torch.cuda.is_available():
            return "cuda"
        write_log("CUDA unavailable; falling back to CPU", self._log_path)
        return "cpu"

    def _log_compute_selection(self) -> None:
        if self._config.compute_mode:
            write_log(
                f"Whisper compute mode {self._config.compute_mode} -> device={self._device} fp16={self._fp16}",
                self._log_path,
            )
            return
        write_log(f"Whisper device {self._device} fp16={self._fp16}", self._log_path)

    def load(self) -> None:
        """Load the Whisper model if it has not been loaded yet."""
        with self._lock:
            if self._model is not None:
                return
            self._config.cache_dir.mkdir(parents=True, exist_ok=True)
            write_log(
                f"Loading Whisper model {self._config.model_name} on {self._device}",
                self._log_path,
            )
            self._model = whisper.load_model(
                self._config.model_name,
                device=self._device,
                download_root=str(self._config.cache_dir),
            )
            write_log("Whisper model loaded", self._log_path)

    def transcribe(self, audio_path: Path) -> WhisperResult:
        """Transcribe `audio_path` and return a structured result."""
        self.load()
        assert self._model is not None  # Guard for type checkers

        kwargs: Dict[str, Any] = {
            "language": "en",
            "task": "transcribe",
            "beam_size": self._config.beam_size,
            "best_of": self._config.best_of,
            "temperature": self._config.temperature,
            "compression_ratio_threshold": self._config.compression_ratio_threshold,
            "condition_on_previous_text": self._config.condition_on_previous_text,
            "fp16": self._fp16,
        }
        if self._config.patience is not None:
            kwargs["patience"] = self._config.patience
        if self._config.logprob_threshold is not None:
            kwargs["logprob_threshold"] = self._config.logprob_threshold
        if self._config.no_speech_threshold is not None:
            kwargs["no_speech_threshold"] = self._config.no_speech_threshold
        if self._config.initial_prompt:
            kwargs["initial_prompt"] = self._config.initial_prompt
        if self._config.suppress_tokens is not None:
            kwargs["suppress_tokens"] = self._config.suppress_tokens

        write_log(f"Transcribing {audio_path}", self._log_path)
        result = self._model.transcribe(str(audio_path), **kwargs)

        segments = [
            WhisperSegment(
                text=seg.get("text", "").strip(),
                start=float(seg.get("start", 0.0)),
                end=float(seg.get("end", 0.0)),
                avg_logprob=seg.get("avg_logprob"),
                compression_ratio=seg.get("compression_ratio"),
                no_speech_prob=seg.get("no_speech_prob"),
            )
            for seg in result.get("segments", [])
        ]
        text = result.get("text", "").strip()
        info = {
            "language": result.get("language"),
            "duration": result.get("duration"),
        }
        return WhisperResult(text=text, segments=segments, info=info)
