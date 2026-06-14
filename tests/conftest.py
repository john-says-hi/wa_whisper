import math
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

pynput_module = types.ModuleType("pynput")
keyboard_module = types.ModuleType("pynput.keyboard")


class _Key:
    alt_r = "alt_r"
    esc = "esc"


class _Listener:
    def __init__(self, *_, **__):
        pass

    def start(self):
        return self

    def stop(self):
        return None


def _suppress(*_, **__):
    return False


keyboard_module.Key = _Key
keyboard_module.Listener = _Listener
keyboard_module.Events = types.SimpleNamespace()  # for compatibility if needed
keyboard_module.Listener.suppress = False

pynput_module.keyboard = keyboard_module

sys.modules.setdefault("pynput", pynput_module)
sys.modules.setdefault("pynput.keyboard", keyboard_module)


np_module = types.ModuleType("numpy")
np_module.float64 = float
np_module.ndarray = list


def _sqrt(value):  # pragma: no cover - simple shim
    return math.sqrt(value)


def _mean(values, dtype=None):  # pragma: no cover - simple shim
    return sum(values) / len(values)


def _square(values, dtype=None):  # pragma: no cover - simple shim
    return [v * v for v in values]


np_module.sqrt = _sqrt
np_module.mean = _mean
np_module.square = _square

sys.modules.setdefault("numpy", np_module)


sd_module = types.ModuleType("sounddevice")


class _PortAudioError(Exception):
    pass


class _DummyStream:
    def __init__(self, *_, **__):
        pass

    def start(self):  # pragma: no cover - tests replace this stub
        raise _PortAudioError("not patched")

    def stop(self):
        return None

    def close(self):
        return None


sd_module.PortAudioError = _PortAudioError
sd_module.InputStream = _DummyStream
sd_module.default = types.SimpleNamespace(device=[0, 0])


def _query_devices(device=None, *args, **_kwargs):
    devices = [{"name": "default", "max_input_channels": 1}]
    if args:
        return devices[0]
    if device is None:
        return devices
    return devices[device]


sd_module.query_devices = _query_devices
sd_module.check_input_settings = lambda *args, **kwargs: None

sys.modules.setdefault("sounddevice", sd_module)


sf_module = types.ModuleType("soundfile" )


class _SoundFile:
    def __init__(self, path, mode="w", samplerate=16_000, channels=1, subtype="PCM_16"):
        self.path = Path(path)
        self.mode = mode
        self.samplerate = samplerate
        self.channels = channels
        self.subtype = subtype
        self._buffer = []
        self.path.touch(exist_ok=True)

    def write(self, data):  # pragma: no cover - queue is empty for these tests
        self._buffer.append(data)

    def close(self):
        return None


sf_module.SoundFile = _SoundFile

sys.modules.setdefault("soundfile", sf_module)


w2n_module = types.ModuleType("word2number")
w2n_sub = types.ModuleType("word2number.w2n")


def _word_to_num(text):  # pragma: no cover - not used in these tests
    raise ValueError("word2number stub")


w2n_sub.word_to_num = _word_to_num
w2n_module.w2n = w2n_sub

sys.modules.setdefault("word2number", w2n_module)
sys.modules.setdefault("word2number.w2n", w2n_sub)


torch_module = types.ModuleType("torch")


class _DummyTorchModule(types.ModuleType):
    def __init__(self):
        super().__init__("torch")

    def device(self, *_):  # pragma: no cover - not used
        return "cpu"


torch_module.cuda = types.SimpleNamespace(is_available=lambda: False)
torch_module.device = lambda *_args, **_kwargs: "cpu"
torch_module.dtype = types.SimpleNamespace()
torch_module.Tensor = list

sys.modules.setdefault("torch", torch_module)


whisper_module = types.ModuleType("whisper")


class _DummyModel:
    def transcribe(self, *_args, **_kwargs):  # pragma: no cover
        return type("Result", (), {"text": ""})


def _load_model(*_args, **_kwargs):  # pragma: no cover
    return _DummyModel()


whisper_module.load_model = _load_model

sys.modules.setdefault("whisper", whisper_module)
