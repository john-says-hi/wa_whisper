"""CTranslate2 inference with the same full Whisper model and result contract."""
from __future__ import annotations

from pathlib import Path

from .admission import ensure_process_admission
from .log_utils import write_log
from .whisper_backend import WhisperBackend, WhisperResult, WhisperSegment


class FasterWhisperBackend(WhisperBackend):
    ENGINE_NAME = "faster-whisper"

    def load(self) -> None:
        with self._lock:
            if self._model is not None:
                return
            from faster_whisper import WhisperModel

            if self._device.startswith("cuda") and not self._cooperative_capture:
                ensure_process_admission()
            compute_type = "float16" if self._fp16 else "float32"
            device, _, index = self._device.partition(":")
            write_log(
                f"Loading faster-whisper {self._config.model_name} on {self._device} {compute_type}",
                self._log_path,
            )
            self._model = WhisperModel(
                self._config.model_name, device=device, device_index=int(index or 0),
                compute_type=compute_type, download_root=str(self._config.cache_dir),
            )
            write_log("faster-whisper model loaded", self._log_path)

    def _decode_options(self) -> dict:
        options = super()._decode_options()
        if "logprob_threshold" in options:
            options["log_prob_threshold"] = options.pop("logprob_threshold")
        tokens = options.get("suppress_tokens")
        if isinstance(tokens, str):
            options["suppress_tokens"] = [int(token.strip()) for token in tokens.split(",") if token.strip()]
        # Preserve existing audio segmentation and timestamp behavior.
        options["vad_filter"] = False
        options["without_timestamps"] = False
        return options

    def warmup(self) -> None:
        import numpy as np

        self.load()
        segments, _ = self._model.transcribe(np.zeros(16000, dtype=np.float32), **self._decode_options())
        # CTranslate2 inference is lazy: readiness requires consuming the generator.
        for _ in segments:
            pass

    def transcribe(self, audio_path: Path) -> WhisperResult:
        self.load()
        write_log(f"Transcribing with faster-whisper {audio_path}", self._log_path)
        output, info = self._model.transcribe(str(audio_path), **self._decode_options())
        text_parts = []
        segments = []
        for segment in output:
            text_parts.append(segment.text)
            segments.append(WhisperSegment(
                text=segment.text.strip(), start=float(segment.start), end=float(segment.end),
                avg_logprob=segment.avg_logprob, compression_ratio=segment.compression_ratio,
                no_speech_prob=segment.no_speech_prob,
            ))
        return WhisperResult(
            text="".join(text_parts).strip(), segments=segments,
            info={"language": info.language, "duration": info.duration, "engine": "faster-whisper"},
        )
