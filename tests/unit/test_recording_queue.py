"""Microphone queue capacity, ordered processing and accidental repress protection."""
import threading
import time
from types import SimpleNamespace

import pytest

from wa_whisper import hotkeys
from wa_whisper.processing_queue import CaptureQueue, CaptureWorker, DrainBarrier
from wa_whisper.recording_admission import RecordingAdmission


class Recorder:
    def __init__(self, root):
        self.root = root
        self.count = 0

    def start(self):
        self.count += 1
        self.path = self.root / f"capture{self.count}.wav"
        self.path.write_bytes(b"retained audio")
        return self.path

    def stop(self, timeout):
        return self.path

    def last_capture_stats(self):
        return None


def wait_until(predicate):
    deadline = time.monotonic() + 2
    while not predicate():
        assert time.monotonic() < deadline
        time.sleep(0.005)


def start_recording(key, hands_free):
    if hands_free:
        key._handle_press(hotkeys.keyboard.Key.ctrl_l)
    key._handle_press(hotkeys.keyboard.Key.alt_r)
    if hands_free:
        key._handle_release(hotkeys.keyboard.Key.alt_r)
        key._handle_release(hotkeys.keyboard.Key.ctrl_l)


def finish_recording(key, hands_free):
    if hands_free:
        start_recording(key, hands_free)
    else:
        key._handle_release(hotkeys.keyboard.Key.alt_r)
    wait_until(lambda: not key.capture_state()["finalizing"])


@pytest.mark.parametrize("hands_free", [False, True])
def test_busy_transcription_accepts_seven_waiting_and_rejects_next_before_recording(tmp_path, monkeypatch, hands_free):
    clock = [100.0]
    monkeypatch.setattr(hotkeys, "time", SimpleNamespace(monotonic=lambda: clock[0], sleep=time.sleep))
    monkeypatch.setattr(hotkeys, "play_system_bell", lambda *args, **kwargs: None)
    tasks = CaptureQueue()
    notices, completed = [], []
    backend = SimpleNamespace(notices=SimpleNamespace(say=notices.append), capture_admission_reason=lambda: None)
    admission = RecordingAdmission(tasks, backend)
    recorder = Recorder(tmp_path)
    key = hotkeys.PushToTalkHotkey(recorder, on_capture_finished=tasks.put,
                                  silence_timeout=0.5, log_path=tmp_path / "hotkey.log",
                                  enable_audio_mute=False, enable_hotkey_shield=False,
                                  capture_admission=admission)
    processing, release = threading.Event(), threading.Event()
    def process(result):
        if not completed:
            processing.set()
            assert release.wait(3)
        assert result.path.read_bytes() == b"retained audio"
        completed.append(result.path.name)
    worker = CaptureWorker(tasks, process, tmp_path / "worker.log")
    worker.start()
    try:
        start_recording(key, hands_free)
        assert key.capture_state()["recording"]
        finish_recording(key, hands_free)
        assert processing.wait(1)
        start_recording(key, hands_free)
        assert recorder.count == 1
        key._handle_release(hotkeys.keyboard.Key.alt_r)
        for expected in range(2, 9):
            clock[0] += 2
            start_recording(key, hands_free)
            assert key.capture_state()["recording"]
            finish_recording(key, hands_free)
            assert recorder.count == expected
        assert worker.status()["processing"] and worker.status()["queued_items"] == 7
        assert not completed
        clock[0] += 2
        start_recording(key, hands_free)
        assert not key.capture_state()["recording"] and recorder.count == 8
        key._handle_release(hotkeys.keyboard.Key.alt_r)
        assert notices == ["recording_queue_full"]
        release.set()
        worker.drain(time.monotonic() + 2, lambda: False)
        assert completed == [f"capture{i}.wav" for i in range(1, 9)]
        clock[0] += 2
        start_recording(key, hands_free)
        assert key.capture_state()["recording"]
        finish_recording(key, hands_free)
        worker.drain(time.monotonic() + 2, lambda: False)
        assert completed[-1] == "capture9.wav"
    finally:
        release.set()
        key.stop()
        tasks.put(None)
        worker.join(2)


def test_control_markers_do_not_take_recording_slots():
    tasks = CaptureQueue()
    tasks.put("recording")
    tasks.put(DrainBarrier())
    tasks.put(None)
    assert tasks.waiting_recordings() == 1


def test_available_queue_preserves_gpu_admission():
    notices = []
    backend = SimpleNamespace(notices=SimpleNamespace(say=notices.append),
                              capture_admission_reason=lambda: "GPU reserved")
    assert RecordingAdmission(CaptureQueue(), backend)() == "GPU reserved"
    assert not notices
