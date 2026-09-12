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


def test_failed_laptop_load_keeps_desktop_free_and_laptop_selected(device, monkeypatch):
    value, stored, notices = device
    remote = Backend()
    remote.failure = BackendError("memory_full", "full")
    monkeypatch.setattr(routing, "BrokerClient", lambda: remote)
    switch(value)
    assert value.destination == "laptop" and value.local.closed and stored == ["laptop"]
    assert remote.closed and notices[-1][0] == "laptop_memory_full"
    assert [notice[0] for notice in notices] == ["transferring_voice", "laptop_memory_full"]


def test_desktop_unloads_before_destination_loading(device, monkeypatch):
    value, stored, notices = device
    remote = Backend()
    def load(cancelled):
        assert value.local.closed and value.local.pid is None
        assert stored == ["laptop"]
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


def test_missing_notifications_cannot_suppress_spoken_feedback(tmp_path, monkeypatch):
    from wa_whisper import device_notices
    monkeypatch.setattr(device_notices.threading.Thread, "start", lambda self: None)
    monkeypatch.setattr(device_notices.Path, "home", lambda: tmp_path)
    sound = tmp_path / ".local/share/wa_whisper/power_phrases/voice_ready.wav"
    sound.parent.mkdir(parents=True)
    sound.write_bytes(b"cached audio")
    notices = device_notices.DeviceNotices(tmp_path / "log")
    played = []
    def run(command, **kwargs):
        if command[0] == "notify-send":
            raise FileNotFoundError("notify-send unavailable")
        played.append(command)
        notices.close()
    monkeypatch.setattr(device_notices.subprocess, "run", run)
    notices.say("voice_ready")
    notices._run()
    assert played == [["paplay", str(sound)]]


@pytest.mark.parametrize("source", ["desktop", "laptop"])
@pytest.mark.parametrize("fail_load", [False, True])
def test_recordings_queue_during_transfer_and_resume_in_order(device, tmp_path, monkeypatch, source, fail_load):
    import threading
    import time
    from wa_whisper.processing_queue import CaptureQueue, CaptureWorker
    value, _, _ = device
    value.destination = source
    old = Backend()
    target = Backend()
    if source == "desktop":
        value.local = old
        monkeypatch.setattr(routing, "BrokerClient", lambda: target)
    else:
        value.remote, value.local = old, target
    for key in routing.DECODE_FIELDS:
        setattr(value.config, key, None)
    loading, finish_load = threading.Event(), threading.Event()
    calls, results = [], []
    def load(cancelled):
        loading.set()
        assert finish_load.wait(3)
        if fail_load:
            raise BackendError("memory_full", "full")
    target.load = load
    for name, backend in (("source", old), ("target", target)):
        def transcribe(path, cancelled, name=name):
            calls.append((name, path.name))
            return {"text": path.name, "segments": []}
        backend.transcribe = transcribe
    tasks = CaptureQueue()
    worker = CaptureWorker(tasks, lambda result: results.append(value.transcribe(result.path).text), tmp_path / "queue.log")
    worker.start()
    class Recorder:
        count = 0
        def start(self):
            self.count += 1
            self.path = tmp_path / f"recording{self.count}.wav"
            self.path.write_bytes(b"recorded audio")
            return self.path
        def stop(self, timeout):
            return self.path
        def last_capture_stats(self):
            return None
    accepted = []
    def accept(result):
        accepted.append(result)
        tasks.put(result)
    hotkey = PushToTalkHotkey(Recorder(), on_capture_finished=accept,
                             silence_timeout=0.5, log_path=tmp_path / "hotkey.log",
                             enable_audio_mute=False, enable_hotkey_shield=False,
                             hotkey_repress_grace_seconds=0, capture_admission=value.capture_admission_reason)
    value.capture, value.worker = hotkey, worker
    try:
        assert value.request_switch()["accepted"]
        assert loading.wait(1)
        for index in range(2):
            hotkey._handle_press(keyboard.Key.alt_r)
            assert hotkey.capture_state()["recording"]
            hotkey._handle_release(keyboard.Key.alt_r)
            deadline = time.monotonic() + 1
            while hotkey.capture_state()["finalizing"]:
                assert time.monotonic() < deadline
                time.sleep(0.005)
            assert len(accepted) == index + 1
        assert not calls and not results
        finish_load.set()
        value._switch_thread.join(1)
        if source == "desktop" and fail_load:
            assert value.destination == "laptop" and old.closed
            assert not calls and not results
            switch(value)
            assert value.destination == "desktop"
        worker.drain(time.monotonic() + 2, lambda: False)
        assert results == ["recording1.wav", "recording2.wav"]
        assert calls == [("source" if fail_load else "target", name) for name in results]
    finally:
        finish_load.set()
        hotkey.stop()
        tasks.put(None)
        worker.join(2)


def test_shutdown_releases_transcription_waiting_for_transfer(device, tmp_path):
    import threading
    value, _, _ = device
    value._switching = True
    value._switch_finished.clear()
    errors = []
    def transcribe():
        try:
            value.transcribe(tmp_path / "saved.wav")
        except BackendError as exc:
            errors.append(exc.code)
    waiting = threading.Thread(target=transcribe)
    waiting.start()
    value.begin_shutdown()
    waiting.join(1)
    assert not waiting.is_alive()
    assert errors == ["cancelled"]


def test_running_desktop_process_exits_before_laptop_load_and_same_wav_retries(device, tmp_path, monkeypatch):
    import subprocess
    import sys
    import threading
    import time
    from wa_whisper.model_process import ModelProcess
    from wa_whisper.processing_queue import CaptureQueue, CaptureWorker
    value, _, notices = device
    for key in routing.DECODE_FIELDS:
        setattr(value.config, key, None)
    local = ModelProcess(value.config, tmp_path / "model.log")
    process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"],
                               stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    local.process, local.ready = process, True
    value.local = local
    entered, loading, finish = threading.Event(), threading.Event(), threading.Event()
    transcribe = local.transcribe
    def start_local(path, cancelled):
        entered.set()
        return transcribe(path, cancelled)
    local.transcribe = start_local
    remote = Backend()
    def load(cancelled):
        assert process.poll() is not None
        assert local.pid is None
        loading.set()
        assert finish.wait(3)
    remote.load = load
    received, completed = [], []
    def remote_transcribe(path, cancelled):
        received.append(path)
        return {"text": path.name, "segments": []}
    remote.transcribe = remote_transcribe
    monkeypatch.setattr(routing, "BrokerClient", lambda: remote)
    first, second = tmp_path / "first.wav", tmp_path / "second.wav"
    first.write_bytes(b"first preserved WAV")
    second.write_bytes(b"second preserved WAV")
    tasks = CaptureQueue()
    worker = CaptureWorker(tasks, lambda path: completed.append(value.transcribe(path).text), tmp_path / "queue.log")
    worker.start()
    try:
        tasks.put(first)
        assert entered.wait(1)
        tasks.put(second)
        assert value.request_switch()["accepted"]
        assert loading.wait(2)
        assert not completed and not received
        assert process.poll() is not None
        assert value.destination == "laptop"
        finish.set()
        worker.drain(time.monotonic() + 2, lambda: False)
        assert received == [first, second]
        assert completed == ["first.wav", "second.wav"]
        assert first.read_bytes() == b"first preserved WAV"
        assert second.read_bytes() == b"second preserved WAV"
        assert [item[0] for item in notices].count("voice_ready") == 1
    finally:
        finish.set()
        value.begin_shutdown()
        tasks.put(None)
        worker.join(2)
        local.close()


def test_failed_laptop_load_recovers_without_reloading_desktop(device, tmp_path, monkeypatch):
    value, _, notices = device
    for key in routing.DECODE_FIELDS:
        setattr(value.config, key, None)
    remote = Backend()
    remote.failure = BackendError("memory_full", "full")
    monkeypatch.setattr(routing, "BrokerClient", lambda: remote)
    switch(value)
    assert value.destination == "laptop" and value.local.pid is None
    value.local.load = lambda *args: pytest.fail("Desktop must not reload automatically")
    remote.failure = None
    remote.transcribe = lambda path, cancelled: {"text": "Recovered pending recording", "segments": []}
    assert value.transcribe(tmp_path / "pending.wav").text == "Recovered pending recording"
    assert value.destination == "laptop" and value.local.pid is None
    assert notices[-1][0] == "voice_ready"


def test_shutdown_after_laptop_failure_releases_pending_recording(device, tmp_path, monkeypatch):
    import threading
    value, _, _ = device
    remote = Backend()
    remote.failure = BackendError("offline", "offline")
    monkeypatch.setattr(routing, "BrokerClient", lambda: remote)
    switch(value)
    errors = []
    def transcribe():
        try:
            value.transcribe(tmp_path / "preserved.wav")
        except BackendError as exc:
            errors.append(exc.code)
    thread = threading.Thread(target=transcribe)
    thread.start()
    value.begin_shutdown()
    thread.join(2)
    assert not thread.is_alive()
    assert errors == ["cancelled"]
    assert value.local.pid is None
