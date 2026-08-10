import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

try:  # pragma: no cover - fallback for test environments without sounddevice
    import sounddevice as sd  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    class _PortAudioError(Exception):
        pass

    class _SDModule:
        PortAudioError = _PortAudioError

    sd = _SDModule()  # type: ignore

from wa_whisper.recorder import Recorder, RecorderInputChannelError, RecorderStartError


def _make_recorder(
    tmp_path: Path,
    monkeypatch,
    *,
    fail_attempts: int,
    sleep: float = 0.0,
    log_path: Path | None = None,
) -> Recorder:
    counter = {"remaining": fail_attempts}

    class SequencedStream:
        def __init__(self, *, callback, **_kwargs):
            self.callback = callback
            self.stopped = False

        def start(self):
            if counter["remaining"] > 0:
                counter["remaining"] -= 1
                raise sd.PortAudioError("Wait timed out", -9987)

        def stop(self):
            self.stopped = True

        def close(self):
            self.stopped = True

    def stream_factory(**kwargs):
        if sleep:
            time.sleep(sleep)
        return SequencedStream(**kwargs)

    monkeypatch.setattr("wa_whisper.recorder.sd.InputStream", stream_factory)
    monkeypatch.setattr("wa_whisper.recorder.tempfile.gettempdir", lambda: str(tmp_path))
    return Recorder(
        sample_rate=16_000,
        device_index=None,
        log_path=log_path or tmp_path / "log.txt",
        rms_threshold=0.01,
        preamp=1.0,
    )


def test_recorder_start_raises_after_retries(tmp_path, monkeypatch):
    recorder = _make_recorder(tmp_path, monkeypatch, fail_attempts=3)

    with pytest.raises(RecorderStartError):
        recorder.start()

    temp_dir = tmp_path / "wa_whisper"
    assert not list(temp_dir.glob("*.wav"))
    assert recorder.last_capture_stats() is None


def test_recorder_start_recovers_after_transient_failure(tmp_path, monkeypatch):
    recorder = _make_recorder(tmp_path, monkeypatch, fail_attempts=1)

    path = recorder.start()

    assert path.exists()
    assert recorder.last_capture_stats() is None


def test_recorder_start_and_stop_ignore_unusable_log_path(tmp_path, monkeypatch):
    blocking_file = tmp_path / "not_a_directory"
    blocking_file.write_text("blocked", encoding="utf-8")
    recorder = _make_recorder(
        tmp_path,
        monkeypatch,
        fail_attempts=0,
        log_path=blocking_file / "runtime.log",
    )

    capture_path = recorder.start()
    stopped_path = recorder.stop(timeout=0.0)

    assert capture_path.exists()
    assert stopped_path == capture_path
    assert recorder.last_capture_stats() is not None
    assert blocking_file.read_text(encoding="utf-8") == "blocked"


def test_recorder_resolves_linux_default_source_to_concrete_device(tmp_path, monkeypatch):
    seen = {}

    class RecordingStream:
        def __init__(self, *, callback, device, channels, **_kwargs):
            self.callback = callback
            seen["device"] = device
            seen["channels"] = channels

        def start(self):
            return None

        def stop(self):
            return None

        def close(self):
            return None

    def query_devices(device=None, *_args):
        devices = [
            {"name": "default", "max_input_channels": 32},
            {"name": "SSL 2 Analog Surround 4.0", "max_input_channels": 4},
            {"name": "USB Audio Microphone", "max_input_channels": 2},
        ]
        if device is None:
            return devices
        return devices[device]

    def check_output(cmd, text=True):
        if cmd == ["pactl", "info"]:
            return "Default Source: alsa_input.usb-Solid_State_Logic_SSL_2-00.analog-surround-40\n"
        if cmd == ["pactl", "list", "sources"]:
            return (
                "Source #1\n"
                "Name: alsa_input.usb-Solid_State_Logic_SSL_2-00.analog-surround-40\n"
                "Description: SSL 2 Analog Surround 4.0\n"
            )
        raise AssertionError(cmd)

    monkeypatch.setattr("wa_whisper.recorder.sd.InputStream", RecordingStream)
    monkeypatch.setattr("wa_whisper.recorder.sd.query_devices", query_devices)
    monkeypatch.setattr("wa_whisper.recorder.sd.check_input_settings", lambda **kwargs: None)
    monkeypatch.setattr("wa_whisper.recorder.subprocess.check_output", check_output)
    monkeypatch.setattr("wa_whisper.recorder.tempfile.gettempdir", lambda: str(tmp_path))

    recorder = Recorder(
        sample_rate=16_000,
        device_index=None,
        log_path=tmp_path / "log.txt",
        rms_threshold=0.01,
        preamp=1.0,
    )

    recorder.start()
    recorder.stop(timeout=0.0)

    assert seen["device"] == 1
    assert seen["channels"] == 4


def test_recorder_falls_back_to_device_default_sample_rate(tmp_path, monkeypatch):
    seen = {}

    class RecordingStream:
        def __init__(self, *, callback, device, channels, samplerate, **_kwargs):
            self.callback = callback
            seen["device"] = device
            seen["channels"] = channels
            seen["samplerate"] = samplerate

        def start(self):
            return None

        def stop(self):
            return None

        def close(self):
            return None

    def query_devices(device=None, *_args):
        devices = [
            {
                "name": "SSL 2 Analog Surround 4.0",
                "max_input_channels": 4,
                "default_samplerate": 48000.0,
            },
        ]
        if device is None:
            return devices
        return devices[device]

    def check_input_settings(*, samplerate, **_kwargs):
        if samplerate != 48000.0:
            raise sd.PortAudioError("Invalid sample rate", -9997)

    monkeypatch.setattr("wa_whisper.recorder.sd.InputStream", RecordingStream)
    monkeypatch.setattr("wa_whisper.recorder.sd.query_devices", query_devices)
    monkeypatch.setattr("wa_whisper.recorder.sd.check_input_settings", check_input_settings)
    monkeypatch.setattr("wa_whisper.recorder.tempfile.gettempdir", lambda: str(tmp_path))

    recorder = Recorder(
        sample_rate=16_000,
        device_index=0,
        log_path=tmp_path / "log.txt",
        rms_threshold=0.01,
        preamp=1.0,
    )

    path = recorder.start()
    recorder.stop(timeout=0.0)

    assert path.exists()
    assert seen["device"] == 0
    assert seen["channels"] == 4
    assert seen["samplerate"] == 48000.0


def test_recorder_falls_back_when_device_default_sample_rate_is_rejected(tmp_path, monkeypatch):
    seen = {}

    class RecordingStream:
        def __init__(self, *, callback, samplerate, **_kwargs):
            self.callback = callback
            seen["samplerate"] = samplerate

        def start(self):
            return None

        def stop(self):
            return None

        def close(self):
            return None

    def query_devices(device=None, *_args):
        devices = [
            {
                "name": "SSL 2 Analog Surround 4.0",
                "max_input_channels": 4,
                "default_samplerate": 48000.0,
            },
        ]
        if device is None:
            return devices
        return devices[device]

    def check_input_settings(*, samplerate, **_kwargs):
        if samplerate != 44100.0:
            raise sd.PortAudioError("Invalid sample rate", -9997)

    monkeypatch.setattr("wa_whisper.recorder.sd.InputStream", RecordingStream)
    monkeypatch.setattr("wa_whisper.recorder.sd.query_devices", query_devices)
    monkeypatch.setattr("wa_whisper.recorder.sd.check_input_settings", check_input_settings)
    monkeypatch.setattr("wa_whisper.recorder.tempfile.gettempdir", lambda: str(tmp_path))

    recorder = Recorder(
        sample_rate=16_000,
        device_index=0,
        log_path=tmp_path / "log.txt",
        rms_threshold=0.01,
        preamp=1.0,
    )

    path = recorder.start()
    recorder.stop(timeout=0.0)

    assert path.exists()
    assert seen["samplerate"] == 44100.0


def test_prepare_block_collapses_to_loudest_channel(tmp_path):
    recorder = Recorder(
        sample_rate=16_000,
        device_index=None,
        log_path=tmp_path / "log.txt",
        rms_threshold=0.01,
        preamp=1.0,
    )

    block = [
        [0.0, 0.10, 0.0, 0.0],
        [0.0, 0.25, 0.0, 0.0],
        [0.0, 0.40, 0.0, 0.0],
    ]

    assert recorder._prepare_block(block) == [[0.10], [0.25], [0.40]]


def test_configured_input_channel_overrides_louder_channels(tmp_path):
    recorder = Recorder(
        sample_rate=16_000,
        device_index=None,
        log_path=tmp_path / "log.txt",
        rms_threshold=0.01,
        preamp=1.0,
        input_channel=1,
    )

    block = [
        [0.10, 0.90, 0.80, 0.70],
        [0.25, 0.95, 0.85, 0.75],
        [0.40, 1.00, 0.90, 0.80],
    ]

    assert recorder._prepare_block(block) == [[0.10], [0.25], [0.40]]


def test_configured_input_channel_does_not_switch_between_blocks(tmp_path, monkeypatch):
    log_messages = []
    monkeypatch.setattr(
        "wa_whisper.recorder.write_log",
        lambda message, _path: log_messages.append(message),
    )
    recorder = Recorder(
        sample_rate=16_000,
        device_index=None,
        log_path=tmp_path / "log.txt",
        rms_threshold=0.01,
        preamp=1.0,
        input_channel=1,
    )

    first_block = [[0.10, 0.90], [0.20, 0.80]]
    second_block = [[0.30, 0.01], [0.40, 0.02]]

    assert recorder._prepare_block(first_block) == [[0.10], [0.20]]
    assert recorder._prepare_block(second_block) == [[0.30], [0.40]]
    assert log_messages == ["Recorder collapsing 2 channels to mono via configured channel 1"]


@pytest.mark.parametrize("input_channel", [0, -1])
def test_configured_input_channel_must_be_positive(tmp_path, input_channel):
    with pytest.raises(RecorderInputChannelError, match="positive one-based channel number"):
        Recorder(
            sample_rate=16_000,
            device_index=None,
            log_path=tmp_path / "log.txt",
            rms_threshold=0.01,
            preamp=1.0,
            input_channel=input_channel,
        )


def test_configured_input_channel_must_exist_on_resolved_device(tmp_path, monkeypatch):
    stream_opened = False

    def stream_factory(**_kwargs):
        nonlocal stream_opened
        stream_opened = True
        raise AssertionError("invalid channel should be rejected before opening the stream")

    monkeypatch.setattr("wa_whisper.recorder.sd.InputStream", stream_factory)
    monkeypatch.setattr(
        "wa_whisper.recorder.sd.query_devices",
        lambda _device=None, *_args: {
            "name": "SSL 2 Analog Surround 4.0",
            "max_input_channels": 4,
            "default_samplerate": 48_000.0,
        },
    )
    monkeypatch.setattr("wa_whisper.recorder.sd.check_input_settings", lambda **_kwargs: None)
    monkeypatch.setattr("wa_whisper.recorder.tempfile.gettempdir", lambda: str(tmp_path))
    recorder = Recorder(
        sample_rate=16_000,
        device_index=0,
        log_path=tmp_path / "log.txt",
        rms_threshold=0.01,
        preamp=1.0,
        input_channel=5,
    )

    with pytest.raises(
        RecorderInputChannelError,
        match=r"Configured input channel 5 is unavailable.*4 channel\(s\)",
    ):
        recorder.start()

    assert stream_opened is False
    assert not list((tmp_path / "wa_whisper").glob("*.wav"))


def test_writer_remains_mono_pcm_16_with_multichannel_input(tmp_path, monkeypatch):
    class RecordingStream:
        def __init__(self, **_kwargs):
            pass

        def start(self):
            return None

        def stop(self):
            return None

        def close(self):
            return None

    monkeypatch.setattr("wa_whisper.recorder.sd.InputStream", RecordingStream)
    monkeypatch.setattr(
        "wa_whisper.recorder.sd.query_devices",
        lambda _device=None, *_args: {
            "name": "SSL 2 Analog Surround 4.0",
            "max_input_channels": 4,
            "default_samplerate": 48_000.0,
        },
    )
    monkeypatch.setattr("wa_whisper.recorder.sd.check_input_settings", lambda **_kwargs: None)
    monkeypatch.setattr("wa_whisper.recorder.tempfile.gettempdir", lambda: str(tmp_path))
    recorder = Recorder(
        sample_rate=16_000,
        device_index=0,
        log_path=tmp_path / "log.txt",
        rms_threshold=0.01,
        preamp=1.0,
        input_channel=1,
    )

    recorder.start()
    writer = recorder._writer

    assert writer is not None
    assert writer.channels == 1
    assert writer.subtype == "PCM_16"

    recorder.stop(timeout=0.0)
