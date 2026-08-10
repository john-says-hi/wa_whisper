import importlib
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

shield_mod = importlib.import_module("wa_whisper.x11_key_shield")
hotkeys_mod = importlib.import_module("wa_whisper.hotkeys")


class FakeRoot:
    def __init__(self):
        self.grab_calls = []
        self.ungrab_calls = []

    def grab_key(self, keycode, modifiers, owner_events, pointer_mode, keyboard_mode, onerror=None):
        self.grab_calls.append((keycode, modifiers, owner_events, pointer_mode, keyboard_mode))

    def ungrab_key(self, keycode, modifiers):
        self.ungrab_calls.append((keycode, modifiers))


class FakeDisplay:
    def __init__(self, keycode=108, events=None):
        self.root = FakeRoot()
        self.closed = False
        self._keycode = keycode
        self.events = list(events or [])

    def keysym_to_keycode(self, _keysym):
        return self._keycode

    def screen(self):
        return SimpleNamespace(root=self.root)

    def sync(self):
        return None

    def pending_events(self):
        return len(self.events)

    def next_event(self):
        return self.events.pop(0)

    def close(self):
        self.closed = True


class FakeCatchError:
    def __init__(self, *_errors):
        pass

    def get_error(self):
        return None


class FakeCatchErrorWithFailure(FakeCatchError):
    def get_error(self):
        return "BadAccess"


def make_fake_xlib(display, catch_error_cls=FakeCatchError):
    return SimpleNamespace(
        X=SimpleNamespace(AnyModifier=1 << 15, GrabModeAsync=1),
        XK=SimpleNamespace(string_to_keysym=lambda name: 0xFFEA if name == "Alt_R" else 0),
        display=SimpleNamespace(Display=lambda: display),
        error=SimpleNamespace(CatchError=catch_error_cls, BadAccess=RuntimeError),
    )


def wait_until(predicate, timeout_seconds=1.0):
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


def test_shield_grabs_key_and_drains_events(monkeypatch, tmp_path):
    display = FakeDisplay(events=["press", "release"])
    monkeypatch.setattr(shield_mod, "load_xlib", lambda: make_fake_xlib(display))
    shield = shield_mod.X11KeyShield("Alt_R", tmp_path / "log.txt")

    assert shield.start() is True
    assert shield.active is True
    assert display.root.grab_calls == [(108, 1 << 15, False, 1, 1)]
    assert wait_until(lambda: not display.events)

    shield.stop()
    assert shield.active is False
    assert display.root.ungrab_calls == [(108, 1 << 15)]
    assert display.closed is True


def test_shield_start_is_idempotent_while_active(monkeypatch, tmp_path):
    display = FakeDisplay()
    monkeypatch.setattr(shield_mod, "load_xlib", lambda: make_fake_xlib(display))
    shield = shield_mod.X11KeyShield("Alt_R", tmp_path / "log.txt")

    assert shield.start() is True
    assert shield.start() is True
    assert len(display.root.grab_calls) == 1

    shield.stop()


def test_shield_start_and_cleanup_ignore_unusable_log_path(monkeypatch, tmp_path):
    display = FakeDisplay()
    blocking_file = tmp_path / "not_a_directory"
    blocking_file.write_text("blocked", encoding="utf-8")
    monkeypatch.setattr(shield_mod, "load_xlib", lambda: make_fake_xlib(display))
    shield = shield_mod.X11KeyShield("Alt_R", blocking_file / "runtime.log")

    assert shield.start() is True
    shield.stop()

    assert shield.active is False
    assert display.root.ungrab_calls == [(108, 1 << 15)]
    assert display.closed is True
    assert blocking_file.read_text(encoding="utf-8") == "blocked"


def test_shield_disabled_when_keycode_missing(monkeypatch, tmp_path):
    display = FakeDisplay(keycode=0)
    monkeypatch.setattr(shield_mod, "load_xlib", lambda: make_fake_xlib(display))
    shield = shield_mod.X11KeyShield("Alt_R", tmp_path / "log.txt")

    assert shield.start() is False
    assert shield.active is False
    assert display.closed is True
    assert display.root.grab_calls == []


def test_shield_disabled_when_grab_rejected(monkeypatch, tmp_path):
    display = FakeDisplay()
    monkeypatch.setattr(
        shield_mod,
        "load_xlib",
        lambda: make_fake_xlib(display, catch_error_cls=FakeCatchErrorWithFailure),
    )
    shield = shield_mod.X11KeyShield("Alt_R", tmp_path / "log.txt")

    assert shield.start() is False
    assert shield.active is False
    assert display.closed is True


def test_shield_disabled_when_xlib_unavailable(monkeypatch, tmp_path):
    def raise_import_error():
        raise ImportError("no Xlib in test environment")

    monkeypatch.setattr(shield_mod, "load_xlib", raise_import_error)
    shield = shield_mod.X11KeyShield("Alt_R", tmp_path / "log.txt")

    assert shield.start() is False
    assert shield.active is False


class FakeShield:
    def __init__(self, keysym_name, log_path):
        self.keysym_name = keysym_name
        self.log_path = log_path
        self.started = False
        self.stopped = False

    def start(self):
        self.started = True
        return True

    def stop(self):
        self.stopped = True


def build_hotkey(tmp_path, enable_hotkey_shield):
    return hotkeys_mod.PushToTalkHotkey(
        recorder=object(),
        silence_timeout=0.5,
        on_capture_finished=lambda *_: None,
        log_path=tmp_path / "log.txt",
        enable_audio_mute=False,
        enable_hotkey_shield=enable_hotkey_shield,
    )


def test_push_to_talk_starts_and_stops_shield(monkeypatch, tmp_path):
    monkeypatch.setattr(hotkeys_mod, "X11KeyShield", FakeShield)
    hotkey = build_hotkey(tmp_path, enable_hotkey_shield=True)
    shield = hotkey._hotkey_shield

    assert isinstance(shield, FakeShield)
    assert shield.keysym_name == "Alt_R"

    hotkey.start()
    assert shield.started is True

    hotkey.stop()
    assert shield.stopped is True


def test_push_to_talk_can_disable_shield(monkeypatch, tmp_path):
    monkeypatch.setattr(hotkeys_mod, "X11KeyShield", FakeShield)
    hotkey = build_hotkey(tmp_path, enable_hotkey_shield=False)

    assert hotkey._hotkey_shield is None

    hotkey.start()
    hotkey.stop()


def test_hotkey_stop_before_start_permanently_prevents_resource_start(monkeypatch, tmp_path):
    listener_constructions = []
    monkeypatch.setattr(hotkeys_mod, "X11KeyShield", FakeShield)
    monkeypatch.setattr(
        hotkeys_mod.keyboard,
        "Listener",
        lambda **kwargs: listener_constructions.append(kwargs),
    )
    hotkey = build_hotkey(tmp_path, enable_hotkey_shield=True)
    shield = hotkey._hotkey_shield

    hotkey.stop()

    assert hotkey.start() is False
    assert listener_constructions == []
    assert shield.started is False
    assert hotkey._listener is None


def test_hotkey_reentrant_stop_during_start_cleans_late_shield(monkeypatch, tmp_path):
    listener_constructions = []
    hotkey = build_hotkey(tmp_path, enable_hotkey_shield=False)

    class ReentrantStopShield(FakeShield):
        def start(self):
            self.started = True
            hotkey.stop()
            return True

    shield = ReentrantStopShield("Alt_R", tmp_path / "log.txt")
    hotkey._hotkey_shield = shield
    monkeypatch.setattr(
        hotkeys_mod.keyboard,
        "Listener",
        lambda **kwargs: listener_constructions.append(kwargs),
    )

    assert hotkey.start() is False
    assert listener_constructions == []
    assert shield.started is True
    assert shield.stopped is True
    assert hotkey._listener is None


def test_hotkey_reentrant_stop_after_start_publication_cleans_resources(monkeypatch, tmp_path):
    class TrackingListener:
        def __init__(self, **_kwargs):
            self.stopped = False

        def start(self):
            return self

        def stop(self):
            self.stopped = True

    listeners = []

    def listener_factory(**kwargs):
        listener = TrackingListener(**kwargs)
        listeners.append(listener)
        return listener

    monkeypatch.setattr(hotkeys_mod, "X11KeyShield", FakeShield)
    monkeypatch.setattr(hotkeys_mod.keyboard, "Listener", listener_factory)
    hotkey = build_hotkey(tmp_path, enable_hotkey_shield=True)
    shield = hotkey._hotkey_shield
    original_write_log = hotkeys_mod.write_log
    shutdown_requested = False

    def request_shutdown_from_start_log(message, log_path):
        nonlocal shutdown_requested
        if message.startswith("Hotkey listener started") and not shutdown_requested:
            shutdown_requested = True
            hotkey.stop()
        original_write_log(message, log_path)

    monkeypatch.setattr(hotkeys_mod, "write_log", request_shutdown_from_start_log)

    assert hotkey.start() is False
    assert shutdown_requested is True
    assert listeners[0].stopped is True
    assert shield.stopped is True
    assert hotkey._listener is None
    assert hotkey._events_enabled is False


def test_hotkey_concurrent_start_and_stop_leave_resources_stopped(monkeypatch, tmp_path):
    class BlockingShield(FakeShield):
        def __init__(self, keysym_name, log_path):
            super().__init__(keysym_name, log_path)
            self.start_entered = threading.Event()
            self.allow_start = threading.Event()

        def start(self):
            self.start_entered.set()
            assert self.allow_start.wait(timeout=1.0)
            self.started = True
            return True

    class TrackingListener:
        def __init__(self, **_kwargs):
            self.started = False
            self.stopped = False

        def start(self):
            self.started = True
            return self

        def stop(self):
            self.stopped = True

    listeners = []

    def listener_factory(**kwargs):
        listener = TrackingListener(**kwargs)
        listeners.append(listener)
        return listener

    monkeypatch.setattr(hotkeys_mod, "X11KeyShield", BlockingShield)
    monkeypatch.setattr(hotkeys_mod.keyboard, "Listener", listener_factory)
    hotkey = build_hotkey(tmp_path, enable_hotkey_shield=True)
    shield = hotkey._hotkey_shield
    start_results = []
    stop_started = threading.Event()
    stop_finished = threading.Event()

    start_thread = threading.Thread(target=lambda: start_results.append(hotkey.start()))

    def stop_hotkey():
        stop_started.set()
        hotkey.stop()
        stop_finished.set()

    stop_thread = threading.Thread(target=stop_hotkey)
    start_thread.start()
    assert shield.start_entered.wait(timeout=1.0)
    stop_thread.start()
    assert stop_started.wait(timeout=1.0)
    assert stop_finished.is_set() is False

    shield.allow_start.set()
    start_thread.join(timeout=1.0)
    stop_thread.join(timeout=1.0)

    assert start_thread.is_alive() is False
    assert stop_thread.is_alive() is False
    assert start_results == [True]
    assert len(listeners) == 1
    assert listeners[0].started is True
    assert listeners[0].stopped is True
    assert shield.started is True
    assert shield.stopped is True
    assert hotkey._listener is None
    assert hotkey._lifecycle_closed is True


def test_hotkey_listener_start_failure_cleans_listener_and_shield(monkeypatch, tmp_path):
    class FailingListener:
        def __init__(self, **_kwargs):
            self.stopped = False

        def start(self):
            raise RuntimeError("listener failed")

        def stop(self):
            self.stopped = True

    listeners = []

    def listener_factory(**kwargs):
        listener = FailingListener(**kwargs)
        listeners.append(listener)
        return listener

    monkeypatch.setattr(hotkeys_mod, "X11KeyShield", FakeShield)
    monkeypatch.setattr(hotkeys_mod.keyboard, "Listener", listener_factory)
    hotkey = build_hotkey(tmp_path, enable_hotkey_shield=True)
    shield = hotkey._hotkey_shield

    with pytest.raises(RuntimeError, match="listener failed"):
        hotkey.start()

    assert listeners[0].stopped is True
    assert shield.started is True
    assert shield.stopped is True
    assert hotkey._listener is None
    assert hotkey._listening_resources_active is False
