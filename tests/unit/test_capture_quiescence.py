import json
import threading
import time
from pathlib import Path

import pytest

from wa_whisper import hotkeys


class FakeRecorder:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.start_count = 0
        self.stop_count = 0

    def start(self) -> Path:
        self.start_count += 1
        return self.path

    def stop(self, _silence_timeout: float) -> Path:
        self.stop_count += 1
        return self.path

    def last_capture_stats(self):
        return None


@pytest.fixture
def capture(tmp_path, monkeypatch):
    monkeypatch.setattr(hotkeys, "play_system_bell", lambda *_args, **_kwargs: None)
    recorder = FakeRecorder(tmp_path / "capture.wav")
    results = []
    controller = hotkeys.PushToTalkHotkey(
        recorder,
        silence_timeout=0.0,
        on_capture_finished=results.append,
        log_path=tmp_path / "hotkeys.log",
        enable_audio_mute=False,
        enable_hotkey_shield=False,
        hotkey_repress_grace_seconds=0.0,
    )
    yield controller, recorder, results
    controller.stop()


def press_alt(controller):
    controller._handle_press(hotkeys.keyboard.Key.alt_r)


def release_alt(controller):
    controller._handle_release(hotkeys.keyboard.Key.alt_r)


def toggle_hands_free(controller):
    controller._handle_press(hotkeys.keyboard.Key.ctrl_l)
    press_alt(controller)
    release_alt(controller)
    controller._handle_release(hotkeys.keyboard.Key.ctrl_l)


def wait_idle(controller):
    controller.wait_capture_idle("owner", time.monotonic() + 2.0, lambda: False)


def test_idle_gate_blocks_both_modes_and_only_owner_can_reopen(capture):
    controller, recorder, _ = capture
    controller.begin_quiesce("owner")
    controller.begin_quiesce("owner")
    with pytest.raises(RuntimeError, match="Another operation"):
        controller.begin_quiesce("other")
    controller.cancel_quiesce("other")

    press_alt(controller)
    release_alt(controller)
    toggle_hands_free(controller)
    wait_idle(controller)
    assert recorder.start_count == 0
    assert controller.capture_state() == {
        "mode": "idle",
        "recording": False,
        "finalizing": False,
        "quiescing": True,
        "closed": False,
    }
    assert "owner" not in json.dumps(controller.capture_state())

    controller.cancel_quiesce("owner")
    press_alt(controller)
    assert recorder.start_count == 1


@pytest.mark.parametrize("hands_free", [False, True])
def test_active_capture_completes_normally_while_gate_is_closed(capture, hands_free):
    controller, recorder, results = capture
    if hands_free:
        toggle_hands_free(controller)
    else:
        press_alt(controller)
    controller.begin_quiesce("owner")

    assert controller.capture_state()["recording"] is True
    assert recorder.stop_count == 0
    if hands_free:
        toggle_hands_free(controller)
    else:
        release_alt(controller)
    wait_idle(controller)

    assert len(results) == 1
    assert results[0].end_reason is hotkeys.CaptureEndReason.USER_COMPLETED
    assert recorder.stop_count == 1
    press_alt(controller)
    release_alt(controller)
    assert recorder.start_count == 1


def test_idle_wait_includes_capture_callback_completion(capture):
    controller, _, results = capture
    callback_entered = threading.Event()
    callback_release = threading.Event()
    wait_started = threading.Event()
    wait_finished = threading.Event()
    failures = []

    def enqueue_capture(result):
        callback_entered.set()
        callback_release.wait(timeout=2.0)
        results.append(result)

    def await_capture():
        wait_started.set()
        try:
            wait_idle(controller)
        except (InterruptedError, RuntimeError, TimeoutError) as exc:
            failures.append(exc)
        finally:
            wait_finished.set()

    controller._on_capture_finished = enqueue_capture
    press_alt(controller)
    release_alt(controller)
    assert callback_entered.wait(timeout=1.0)
    controller.begin_quiesce("owner")
    waiter = threading.Thread(target=await_capture)
    waiter.start()
    try:
        assert wait_started.wait(timeout=1.0)
        assert controller.capture_state()["recording"] is False
        assert controller.capture_state()["finalizing"] is True
        assert not wait_finished.wait(timeout=0.05)
    finally:
        callback_release.set()
        waiter.join(timeout=2.0)

    assert not waiter.is_alive()
    assert not failures
    assert wait_finished.is_set()
    assert len(results) == 1
    assert controller.capture_state()["finalizing"] is False


@pytest.mark.parametrize("cancelled", [False, True])
def test_timeout_or_cancellation_leaves_active_capture_untouched(capture, cancelled):
    controller, recorder, _ = capture
    press_alt(controller)
    controller.begin_quiesce("owner")
    expected_error = InterruptedError if cancelled else TimeoutError
    with pytest.raises(expected_error):
        controller.wait_capture_idle(
            "owner", time.monotonic() - 0.01, lambda: cancelled
        )

    assert recorder.stop_count == 0
    assert controller.capture_state()["recording"] is True
    assert controller.capture_state()["quiescing"] is True
    controller.cancel_quiesce("owner")
    assert controller.capture_state()["quiescing"] is False
    release_alt(controller)


def test_releasing_gate_interrupts_waiter_without_stopping_capture(capture):
    controller, recorder, _ = capture
    checked_cancel = threading.Event()
    failures = []

    def cancelled():
        checked_cancel.set()
        return False

    def await_capture():
        try:
            controller.wait_capture_idle("owner", time.monotonic() + 2.0, cancelled)
        except (InterruptedError, RuntimeError, TimeoutError) as exc:
            failures.append(exc)

    press_alt(controller)
    controller.begin_quiesce("owner")
    waiter = threading.Thread(target=await_capture)
    waiter.start()
    try:
        assert checked_cancel.wait(timeout=1.0)
        controller.cancel_quiesce("owner")
    finally:
        waiter.join(timeout=2.0)

    assert not waiter.is_alive()
    assert len(failures) == 1
    assert isinstance(failures[0], InterruptedError)
    assert recorder.stop_count == 0
    assert controller.capture_state()["recording"] is True


def test_cancel_callback_is_polled_while_capture_is_active(capture):
    controller, recorder, _ = capture
    checked_cancel = threading.Event()
    cancelled_event = threading.Event()
    failures = []

    def cancelled():
        checked_cancel.set()
        return cancelled_event.is_set()

    def await_capture():
        try:
            controller.wait_capture_idle("owner", time.monotonic() + 2.0, cancelled)
        except (InterruptedError, RuntimeError, TimeoutError) as exc:
            failures.append(exc)

    press_alt(controller)
    controller.begin_quiesce("owner")
    waiter = threading.Thread(target=await_capture)
    waiter.start()
    try:
        assert checked_cancel.wait(timeout=1.0)
        cancelled_event.set()
    finally:
        waiter.join(timeout=1.0)

    assert not waiter.is_alive()
    assert len(failures) == 1
    assert isinstance(failures[0], InterruptedError)
    assert recorder.stop_count == 0
    assert controller.capture_state()["quiescing"] is True


def test_gate_races_with_capture_start_without_interrupting_it(capture):
    controller, recorder, results = capture
    start_entered = threading.Event()
    start_release = threading.Event()
    quiesce_attempted = threading.Event()
    quiesce_finished = threading.Event()
    real_start = recorder.start

    def blocking_start():
        start_entered.set()
        start_release.wait(timeout=2.0)
        return real_start()

    def quiesce():
        quiesce_attempted.set()
        controller.begin_quiesce("owner")
        quiesce_finished.set()

    recorder.start = blocking_start
    capture_thread = threading.Thread(target=lambda: press_alt(controller))
    quiesce_thread = threading.Thread(target=quiesce)
    capture_thread.start()
    try:
        assert start_entered.wait(timeout=1.0)
        quiesce_thread.start()
        assert quiesce_attempted.wait(timeout=1.0)
        assert not quiesce_finished.wait(timeout=0.05)
    finally:
        start_release.set()
        capture_thread.join(timeout=2.0)
        if quiesce_thread.ident is not None:
            quiesce_thread.join(timeout=2.0)

    assert not capture_thread.is_alive()
    assert not quiesce_thread.is_alive()
    assert quiesce_finished.is_set()
    assert controller.capture_state()["recording"] is True
    assert recorder.stop_count == 0
    release_alt(controller)
    wait_idle(controller)
    assert results[0].end_reason is hotkeys.CaptureEndReason.USER_COMPLETED


def test_closed_lifecycle_cannot_acquire_quiescence(capture):
    controller, recorder, _ = capture
    controller.stop()
    with pytest.raises(RuntimeError, match="closed"):
        controller.begin_quiesce("owner")
    controller.cancel_quiesce("owner")
    press_alt(controller)
    assert recorder.start_count == 0
    assert controller.capture_state()["closed"] is True


def test_empty_token_cannot_acquire_quiescence(capture):
    controller, _, _ = capture
    with pytest.raises(ValueError, match="nonempty"):
        controller.begin_quiesce("")
    assert controller.capture_state()["quiescing"] is False
