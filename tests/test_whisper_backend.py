from pathlib import Path

import pytest

from wa_whisper import whisper_backend
from wa_whisper.whisper_backend import WhisperBackend, WhisperConfig


def test_backend_uses_ram_mode_cpu_and_disables_fp16(tmp_path, monkeypatch):
    seen = {}

    class RecordingModel:
        def transcribe(self, audio_path, **kwargs):
            seen["audio_path"] = audio_path
            seen["transcribe_kwargs"] = kwargs
            return {"text": "hello", "segments": [], "language": "en", "duration": 1.0}

    def fake_load_model(model_name, *, device, download_root):
        seen["load_model"] = {
            "model_name": model_name,
            "device": device,
            "download_root": download_root,
        }
        return RecordingModel()

    monkeypatch.setattr(whisper_backend.whisper, "load_model", fake_load_model)

    config = WhisperConfig(
        model_name="large-v3",
        cache_dir=tmp_path / "models",
        device="cpu",
        compute_mode="ram",
        fp16=False,
    )
    backend = WhisperBackend(config, tmp_path / "log.txt")

    result = backend.transcribe(Path("sample.wav"))

    assert result.text == "hello"
    assert seen["load_model"]["device"] == "cpu"
    assert seen["transcribe_kwargs"]["fp16"] is False
    assert "Whisper compute mode ram -> device=cpu fp16=False" in (tmp_path / "log.txt").read_text(
        encoding="utf-8",
    )


def test_backend_fails_clearly_when_gpu_mode_has_no_cuda(tmp_path, monkeypatch):
    monkeypatch.setattr(whisper_backend.torch.cuda, "is_available", lambda: False)

    config = WhisperConfig(
        device="cuda",
        compute_mode="gpu",
        fp16=True,
    )

    with pytest.raises(RuntimeError, match="gpu compute mode requires CUDA"):
        WhisperBackend(config, tmp_path / "log.txt")
