import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wa_whisper import hotkeys as hotkeys_mod


REAL_MONOTONIC = time.monotonic


class FakeClock:
    def __init__(self, initial_time: float = 100.0) -> None:
        self._time = initial_time

    def __call__(self) -> float:
        return self._time

    def advance(self, seconds: float) -> None:
        self._time += seconds


class BlockingRecorder:
    def __init__(self, tmp_path: Path) -> None:
        self._tmp_path = tmp_path
        self.start_count = 0
        self.stop_started = threading.Event()
        self.allow_stop = threading.Event()

    def start(self) -> Path:
        self.start_count += 1
        path = self._tmp_path / f"capture-{self.start_count}.wav"
        path.touch()
        return path

    def stop(self, _silence_timeout: float) -> Path:
        self.stop_started.set()
        self.allow_stop.wait(timeout=1.0)
        return self._tmp_path / f"capture-{self.start_count}.wav"

    def last_capture_stats(self):
        return None


class ImmediateRecorder:
    def __init__(self, tmp_path: Path) -> None:
        self._tmp_path = tmp_path
        self.start_count = 0

    def start(self) -> Path:
        self.start_count += 1
        path = self._tmp_path / f"capture-{self.start_count}.wav"
        path.touch()
        return path

    def stop(self, _silence_timeout: float) -> Path:
        return self._tmp_path / f"capture-{self.start_count}.wav"

    def last_capture_stats(self):
        return None


def wait_until(predicate, timeout_seconds: float = 1.0) -> bool:
    deadline = REAL_MONOTONIC() + timeout_seconds
    while REAL_MONOTONIC() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


def build_hotkey(tmp_path, recorder, on_capture_finished):
    return hotkeys_mod.PushToTalkHotkey(
        recorder=recorder,
        silence_timeout=0.0,
        on_capture_finished=on_capture_finished,
        log_path=tmp_path / "log.txt",
        enable_audio_mute=False,
        enable_hotkey_shield=False,
        hotkey_repress_grace_seconds=1.0,
    )


def press(hotkey):
    hotkey._handle_press(hotkeys_mod.keyboard.Key.alt_r)


def release(hotkey):
    hotkey._handle_release(hotkeys_mod.keyboard.Key.alt_r)


def test_press_is_ignored_while_capture_finalizes(monkeypatch, tmp_path):
    clock = FakeClock()
    monkeypatch.setattr(hotkeys_mod.time, "monotonic", clock)
    recorder = BlockingRecorder(tmp_path)
    captures = []
    hotkey = build_hotkey(tmp_path, recorder, lambda *args: captures.append(args))

    press(hotkey)
    release(hotkey)
    assert recorder.stop_started.wait(timeout=1.0)

    clock.advance(2.0)
    press(hotkey)

    assert recorder.start_count == 1
    assert "ignored during capture finalization" in (tmp_path / "log.txt").read_text(encoding="utf-8")

    recorder.allow_stop.set()
    assert wait_until(lambda: len(captures) == 1)

    press(hotkey)

    assert recorder.start_count == 2


def test_press_is_ignored_during_post_release_grace_period(monkeypatch, tmp_path):
    clock = FakeClock()
    monkeypatch.setattr(hotkeys_mod.time, "monotonic", clock)
    recorder = ImmediateRecorder(tmp_path)
    captures = []
    hotkey = build_hotkey(tmp_path, recorder, lambda *args: captures.append(args))

    press(hotkey)
    release(hotkey)
    assert wait_until(lambda: len(captures) == 1)

    press(hotkey)
    assert recorder.start_count == 1

    clock.advance(1.01)
    press(hotkey)

    assert recorder.start_count == 2
    assert "ignored for 1.00s grace period" in (tmp_path / "log.txt").read_text(encoding="utf-8")
