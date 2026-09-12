"""Tests for the Wayland (evdev) hotkey backend and backend selection."""

import sys
import types

import pytest


def _install_evdev_stub(monkeypatch, devices):
    """Install a fake evdev module exposing ``devices``.

    ``evdev_listener`` imports evdev lazily inside ``load_evdev()`` so X11
    sessions never pay for it, which also means the stub only has to exist by
    the time ``start()`` runs.
    """
    ecodes = types.SimpleNamespace(
        EV_KEY=1,
        KEY_RIGHTALT=100,
        KEY_LEFTCTRL=29,
        KEY_ESC=1,
    )

    module = types.ModuleType("evdev")
    module.ecodes = ecodes
    module.InputDevice = lambda path: devices[path]
    module.list_devices = lambda: list(devices)
    monkeypatch.setitem(sys.modules, "evdev", module)
    return ecodes


class FakeDevice:
    def __init__(self, path, name, key_caps, events=()):
        self.path = path
        self.name = name
        self._key_caps = list(key_caps)
        self._events = list(events)
        self.closed = False

    def capabilities(self):
        return {1: self._key_caps}

    def fileno(self):
        return 0

    def read(self):
        events, self._events = self._events, []
        return events

    def close(self):
        self.closed = True


def _event(code, value):
    return types.SimpleNamespace(type=1, code=code, value=value)


@pytest.fixture
def listener_module():
    from wa_whisper import evdev_listener

    return evdev_listener


def test_wayland_session_reads_env(monkeypatch, listener_module):
    monkeypatch.setenv("XDG_SESSION_TYPE", "wayland")
    assert listener_module.wayland_session() is True
    monkeypatch.setenv("XDG_SESSION_TYPE", "x11")
    assert listener_module.wayland_session() is False
    monkeypatch.delenv("XDG_SESSION_TYPE", raising=False)
    assert listener_module.wayland_session() is False


def test_only_keyboards_are_watched(monkeypatch, tmp_path, listener_module):
    keyboard = FakeDevice("/dev/input/event0", "AT keyboard", [100, 29, 1])
    mouse = FakeDevice("/dev/input/event1", "Logitech mouse", [272, 273])
    _install_evdev_stub(monkeypatch, {d.path: d for d in (keyboard, mouse)})

    listener = listener_module.EvdevKeyListener(
        on_press=lambda key: None,
        on_release=lambda key: None,
        log_path=tmp_path / "log.txt",
    )
    monkeypatch.setattr(listener_module.threading, "Thread", lambda **kw: types.SimpleNamespace(start=lambda: None, is_alive=lambda: False, join=lambda timeout=None: None))
    listener.start()

    assert "/dev/input/event0" in listener._devices
    assert "/dev/input/event1" not in listener._devices
    assert mouse.closed is True


def test_start_fails_loudly_when_no_keyboard_readable(monkeypatch, tmp_path, listener_module):
    """A silent no-op is the exact failure this backend exists to prevent."""
    _install_evdev_stub(monkeypatch, {})

    listener = listener_module.EvdevKeyListener(
        on_press=lambda key: None,
        on_release=lambda key: None,
        log_path=tmp_path / "log.txt",
    )
    with pytest.raises(RuntimeError, match="input"):
        listener.start()


def test_press_release_dispatch_and_autorepeat_ignored(monkeypatch, tmp_path, listener_module):
    from pynput import keyboard

    ecodes = _install_evdev_stub(monkeypatch, {})
    device = FakeDevice(
        "/dev/input/event0",
        "kbd",
        [100, 29, 1],
        events=[
            _event(ecodes.KEY_RIGHTALT, 1),
            _event(ecodes.KEY_RIGHTALT, 2),  # autorepeat -> must not re-fire
            _event(ecodes.KEY_RIGHTALT, 0),
            _event(ecodes.KEY_LEFTCTRL, 1),
            _event(999, 1),  # unmapped key -> dropped
        ],
    )

    pressed, released = [], []
    listener = listener_module.EvdevKeyListener(
        on_press=pressed.append,
        on_release=released.append,
        log_path=tmp_path / "log.txt",
    )
    listener._ecodes = ecodes
    listener._key_codes = listener_module._key_map(ecodes)

    listener._drain_device(device.path, device)

    assert pressed == [keyboard.Key.alt_r, keyboard.Key.ctrl_l]
    assert released == [keyboard.Key.alt_r]


def test_handler_exception_does_not_kill_the_listener(monkeypatch, tmp_path, listener_module):
    ecodes = _install_evdev_stub(monkeypatch, {})
    device = FakeDevice(
        "/dev/input/event0",
        "kbd",
        [100],
        events=[_event(ecodes.KEY_RIGHTALT, 1), _event(ecodes.KEY_RIGHTALT, 0)],
    )

    released = []

    def explode(_key):
        raise RuntimeError("handler blew up")

    listener = listener_module.EvdevKeyListener(
        on_press=explode,
        on_release=released.append,
        log_path=tmp_path / "log.txt",
    )
    listener._ecodes = ecodes
    listener._key_codes = listener_module._key_map(ecodes)

    from pynput import keyboard

    listener._drain_device(device.path, device)

    # The release still lands, so a raising press handler cannot strand
    # dictation in a permanently recording state.
    assert released == [keyboard.Key.alt_r]


def test_unplugged_device_is_dropped(monkeypatch, tmp_path, listener_module):
    import errno

    ecodes = _install_evdev_stub(monkeypatch, {})

    class Vanishing(FakeDevice):
        def read(self):
            raise OSError(errno.ENODEV, "No such device")

    device = Vanishing("/dev/input/event0", "kbd", [100])
    listener = listener_module.EvdevKeyListener(
        on_press=lambda k: None,
        on_release=lambda k: None,
        log_path=tmp_path / "log.txt",
    )
    listener._ecodes = ecodes
    listener._key_codes = listener_module._key_map(ecodes)
    listener._devices[device.path] = device

    listener._drain_device(device.path, device)

    assert device.path not in listener._devices
