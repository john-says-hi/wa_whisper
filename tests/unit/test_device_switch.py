"""Destination transactions, exact modifier keys, and local control availability."""
from types import SimpleNamespace

import pytest
from pynput import keyboard

from wa_whisper import device_routing as routing
from wa_whisper.hotkeys import PushToTalkHotkey
from wa_whisper.model_process import BackendError


class Backend:
    def __init__(self):
        self.ready = True
        self.pid = 123
        self.closed = False
        self.failure = None

    def load(self, cancelled):
        if self.failure:
            raise self.failure
        if cancelled():
            raise BackendError("cancelled", "cancelled")
        self.ready = True

    def close(self):
        self.closed, self.ready, self.pid = True, False, None


@pytest.fixture
def device(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
    monkeypatch.setattr(routing, "read_destination", lambda: "desktop")
    notices = []
    monkeypatch.setattr(routing, "DeviceNotices", lambda log: SimpleNamespace(say=lambda *args: notices.append(args), close=lambda: None))
    monkeypatch.setattr(routing, "ModelProcess", lambda *_: Backend())
    monkeypatch.setattr(routing, "new_capture_reason", lambda: None)
    stored = []
    monkeypatch.setattr(routing, "save_destination", stored.append)
    value = routing.RoutedBackend(SimpleNamespace(device="cuda", model_name="large-v3"), tmp_path / "log")
    value.capture = SimpleNamespace(begin_quiesce=lambda token: None, wait_capture_idle=lambda *args: None,
                                    cancel_quiesce=lambda token: None)
    value.worker = SimpleNamespace(drain=lambda *args: None)
    yield value, stored, notices
    value.close()


def switch(device):
    assert device.request_switch()["accepted"]
    device._switch_thread.join(3)
    assert not device._switch_thread.is_alive()


def test_failed_laptop_load_keeps_desktop_model_and_preference(device, monkeypatch):
    value, stored, notices = device
    remote = Backend()
    remote.failure = BackendError("memory_full", "full")
    monkeypatch.setattr(routing, "BrokerClient", lambda: remote)
    switch(value)
    assert value.destination == "desktop" and not value.local.closed and not stored
    assert remote.closed and notices[-1][0] == "laptop_memory_full"
    assert [notice[0] for notice in notices] == ["transferring_voice", "laptop_memory_full"]


def test_destination_ready_before_desktop_unloads(device, monkeypatch):
    value, stored, notices = device
    remote = Backend()
    def load(cancelled):
        assert not value.local.closed
        assert notices == [("transferring_voice",)]
    remote.load = load
    monkeypatch.setattr(routing, "BrokerClient", lambda: remote)
    switch(value)
    assert value.destination == "laptop" and value.local.closed
    assert stored == ["laptop"] and notices[-1][0] == "voice_ready"


def test_offline_laptop_does_not_block_switching_back(device):
    value, stored, _ = device
    value.destination = "laptop"
    value.remote = Backend()
    value.remote.ready = False
    switch(value)
    assert value.destination == "desktop" and stored == ["desktop"]


def test_busy_desktop_keeps_laptop_selected(device, monkeypatch):
    value, stored, notices = device
    value.destination = "laptop"
    value.remote = Backend()
    monkeypatch.setattr(routing, "new_capture_reason", lambda: "busy")
    switch(value)
    assert value.destination == "laptop" and not stored and not value.remote.closed
    assert notices[-1][0] == "desktop_busy"


def test_failed_persistence_keeps_source_model(device, monkeypatch):
    value, _stored, _ = device
    monkeypatch.setattr(routing, "BrokerClient", Backend)
    def fail(_):
        raise OSError("Disk full")
    monkeypatch.setattr(routing, "save_destination", fail)
    switch(value)
    assert value.destination == "desktop" and not value.local.closed


def test_exact_left_chord_fires_once_and_never_after_stop(tmp_path):
    calls = []
    hotkey = PushToTalkHotkey(SimpleNamespace(), on_capture_finished=lambda _: None,
                             silence_timeout=0.5, log_path=tmp_path / "log", enable_audio_mute=False,
                             enable_hotkey_shield=False, on_device_switch=lambda: calls.append(True))
    for key in (keyboard.Key.ctrl_l, keyboard.Key.shift_l, keyboard.Key.alt_l, keyboard.Key.f1, keyboard.Key.f1):
        hotkey._handle_press(key)
    assert calls == [True]
    hotkey._handle_release(keyboard.Key.f1)
    hotkey._handle_release(keyboard.Key.shift_l)
    hotkey._handle_press(keyboard.Key.shift_r)
    hotkey._handle_press(keyboard.Key.f1)
    assert calls == [True]
    hotkey._events_enabled = False
    hotkey._handle_release(keyboard.Key.f1)
    hotkey._handle_release(keyboard.Key.shift_r)
    hotkey._handle_press(keyboard.Key.shift_l)
    hotkey._handle_press(keyboard.Key.f1)
    assert calls == [True]


def test_each_switch_gets_both_announcements_even_within_cooldown(tmp_path, monkeypatch):
    from wa_whisper import device_notices
    monkeypatch.setattr(device_notices.threading.Thread, "start", lambda self: None)
    notices = device_notices.DeviceNotices(tmp_path / "log")
    for _ in range(2):
        notices.say("transferring_voice")
        notices.say("voice_ready")
    assert [notices._queue.get_nowait()[0] for _ in range(4)] == [
        "transferring_voice", "voice_ready", "transferring_voice", "voice_ready",
    ]
    notices.close()
