"""Verify engine adaptation without loading GPU models."""
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from wa_whisper.faster_whisper_backend import FasterWhisperBackend
from wa_whisper.inference_worker import create_backend, inference_error_code
from wa_whisper.whisper_backend import WhisperBackend, WhisperConfig


def fake_backend(tmp_path, monkeypatch):
    seen = {"consumed": 0, "loads": 0}

    class Model:
        def __init__(self, name, **options):
            seen.update(name=name, load_options=options, loads=seen["loads"] + 1)

        def transcribe(self, audio, **options):
            seen.update(audio=audio, options=options)

            def generate():
                for text, start in [(" Hello", 0), (" again.", 1)]:
                    seen["consumed"] += 1
                    yield SimpleNamespace(text=text, start=start, end=start + 1,
                                          avg_logprob=-0.2, compression_ratio=1.1, no_speech_prob=0.01)

            return generate(), SimpleNamespace(language="en", duration=2.0)

    monkeypatch.setitem(sys.modules, "faster_whisper", SimpleNamespace(WhisperModel=Model))
    config = WhisperConfig(device="cpu", fp16=False, engine="faster-whisper", cache_dir=tmp_path)
    return FasterWhisperBackend(config, tmp_path / "inference.log"), seen


def test_preserves_full_model_decode_settings_and_result_contract(tmp_path, monkeypatch):
    backend, seen = fake_backend(tmp_path, monkeypatch)
    backend._config = replace(backend._config, beam_size=5, patience=1.5, logprob_threshold=-0.8,
                              no_speech_threshold=0.4, initial_prompt="Names", suppress_tokens="-1, 20")
    result = backend.transcribe(Path("voice.wav"))
    assert seen["name"] == "large-v3"
    assert seen["load_options"] == dict(device="cpu", device_index=0, compute_type="float32",
                                       download_root=str(tmp_path))
    assert seen["options"] == dict(language="en", task="transcribe", beam_size=5, best_of=5,
                                   temperature=0.0, compression_ratio_threshold=2.4,
                                   condition_on_previous_text=False, patience=1.5,
                                   log_prob_threshold=-0.8, no_speech_threshold=0.4,
                                   initial_prompt="Names", suppress_tokens=[-1, 20],
                                   vad_filter=False, without_timestamps=False)
    assert result.text == "Hello again."
    assert [segment.text for segment in result.segments] == ["Hello", "again."]
    assert result.segments[1].end == 2.0
    assert result.segments[0].no_speech_prob == 0.01
    assert result.info == dict(language="en", duration=2.0, engine="faster-whisper")
    assert seen["consumed"] == 2


def test_warmup_consumes_lazy_inference_and_reuses_model(tmp_path, monkeypatch):
    backend, seen = fake_backend(tmp_path, monkeypatch)
    monkeypatch.setitem(sys.modules, "numpy", SimpleNamespace(zeros=lambda *args, **kwargs: [], float32=float))
    backend._device = "cuda:0"
    backend._fp16 = True
    backend.enable_cooperative_capture()
    backend.warmup()
    assert seen["consumed"] == 2
    assert seen["load_options"]["compute_type"] == "float16"
    backend._config = replace(backend._config, beam_size=2)
    backend.transcribe(Path("next.wav"))
    assert seen["loads"] == 1
    assert seen["options"]["beam_size"] == 2
    assert backend.archive_metadata()["engine"] == "faster-whisper"


def test_engine_selection_defaults_to_faster_whisper_with_explicit_rollback(tmp_path, monkeypatch):
    fake_backend(tmp_path, monkeypatch)
    config = WhisperConfig(device="cpu")
    assert isinstance(create_backend(config, tmp_path / "log"), FasterWhisperBackend)
    assert type(create_backend(replace(config, engine="openai"), tmp_path / "log")) is WhisperBackend
    with pytest.raises(ValueError, match="Unknown Whisper engine"):
        create_backend(replace(config, engine="unknown"), tmp_path / "log")


def test_desktop_launch_defaults_to_full_large_v3_and_allows_engine_rollback():
    from wa_whisper.main import build_arg_parser

    parser = build_arg_parser()
    args = parser.parse_args([])
    assert (args.engine, args.model, args.beam_size, args.temperature) == ("faster-whisper", "large-v3", 5, 0.0)
    assert parser.parse_args(["--engine", "openai"]).engine == "openai"


def test_ctranslate_memory_errors_keep_existing_broker_error_contract(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(OutOfMemoryError=MemoryError)))
    assert inference_error_code(RuntimeError("CUDA failed with error out of memory")) == "memory_full"
    assert inference_error_code(RuntimeError("Library not found")) == "inference_failed"
