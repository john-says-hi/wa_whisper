import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import pytest

from wa_whisper import hotkeys as hotkeys_mod


REAL_MONOTONIC = time.monotonic


class FakeClock:
    def __init__(self, initial_time: float = 100.0) -> None:
        self._time = initial_time

    def __call__(self) -> float:
        return self._time

    def advance(self, seconds: float) -> None:
        self._time += seconds


class ImmediateRecorder:
    def __init__(
        self,
        tmp_path: Path,
        *,
        events: list[object] | None = None,
        start_error: Exception | None = None,
        stop_error: Exception | None = None,
    ) -> None:
        self._tmp_path = tmp_path
        self._events = events
        self._start_error = start_error
        self._stop_error = stop_error
        self.start_count = 0
        self.stop_timeouts: list[float] = []

    def start(self) -> Path:
        self.start_count += 1
        if self._events is not None:
            self._events.append("recorder_start")
        if self._start_error:
            raise self._start_error
        path = self._tmp_path / f"capture-{self.start_count}.wav"
        path.touch()
        return path

    def stop(self, silence_timeout: float) -> Path:
        self.stop_timeouts.append(silence_timeout)
        if self._events is not None:
            self._events.append(("recorder_stop", silence_timeout))
        if self._stop_error:
            raise self._stop_error
        return self._tmp_path / f"capture-{self.start_count}.wav"

    def last_capture_stats(self):
        return None


class BlockingRecorder(ImmediateRecorder):
    def __init__(self, tmp_path: Path) -> None:
        super().__init__(tmp_path)
        self.stop_started = threading.Event()
        self.allow_stop = threading.Event()

    def stop(self, silence_timeout: float) -> Path:
        self.stop_timeouts.append(silence_timeout)
        self.stop_started.set()
        self.allow_stop.wait(timeout=1.0)
        return self._tmp_path / f"capture-{self.start_count}.wav"


class RecordingMuteController:
    def __init__(self, events: list[object]) -> None:
        self._events = events

    def mute(self) -> None:
        self._events.append("mute")

    def restore(self) -> None:
        self._events.append("restore")


class FailingMuteController(RecordingMuteController):
    def __init__(
        self,
        events: list[object],
        *,
        mute_error: Exception | None = None,
        restore_error: Exception | None = None,
    ) -> None:
        super().__init__(events)
        self._mute_error = mute_error
        self._restore_error = restore_error

    def mute(self) -> None:
        super().mute()
        if self._mute_error:
            raise self._mute_error

    def restore(self) -> None:
        super().restore()
        if self._restore_error:
            raise self._restore_error


@pytest.fixture(autouse=True)
def disable_real_bells(monkeypatch):
    monkeypatch.setattr(hotkeys_mod, "play_system_bell", lambda *_args, **_kwargs: None)


def wait_until(predicate, timeout_seconds: float = 1.0) -> bool:
    deadline = REAL_MONOTONIC() + timeout_seconds
    while REAL_MONOTONIC() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


def build_hotkey(
    tmp_path: Path,
    recorder,
    captures: list[hotkeys_mod.CaptureResult],
    *,
    on_exit=None,
    log_path: Path | None = None,
) -> hotkeys_mod.PushToTalkHotkey:
    return hotkeys_mod.PushToTalkHotkey(
        recorder=recorder,
        silence_timeout=0.0,
        on_capture_finished=captures.append,
        log_path=log_path or tmp_path / "log.txt",
        enable_audio_mute=False,
        exit_on_esc=True,
        on_exit=on_exit,
        enable_hotkey_shield=False,
        hotkey_repress_grace_seconds=1.0,
    )


def press_left_ctrl(hotkey, *, injected: bool = False) -> None:
    hotkey._handle_press(hotkeys_mod.keyboard.Key.ctrl_l, injected)


def release_left_ctrl(hotkey, *, injected: bool = False) -> None:
    hotkey._handle_release(hotkeys_mod.keyboard.Key.ctrl_l, injected)


def press_right_ctrl(hotkey) -> None:
    hotkey._handle_press(hotkeys_mod.keyboard.Key.ctrl_r)


def release_right_ctrl(hotkey) -> None:
    hotkey._handle_release(hotkeys_mod.keyboard.Key.ctrl_r)


def press_right_alt(hotkey, *, injected: bool = False) -> None:
    hotkey._handle_press(hotkeys_mod.keyboard.Key.alt_r, injected)


def release_right_alt(hotkey, *, injected: bool = False) -> None:
    hotkey._handle_release(hotkeys_mod.keyboard.Key.alt_r, injected)


def activate_hands_free(hotkey) -> None:
    press_left_ctrl(hotkey)
    press_right_alt(hotkey)
    release_right_alt(hotkey)
    release_left_ctrl(hotkey)


def stop_hands_free(hotkey, captures) -> None:
    press_left_ctrl(hotkey)
    press_right_alt(hotkey)
    release_right_alt(hotkey)
    release_left_ctrl(hotkey)
    assert wait_until(lambda: len(captures) == 1)


def test_normal_right_alt_push_to_talk_is_unchanged(tmp_path):
    recorder = ImmediateRecorder(tmp_path)
    captures = []
    hotkey = build_hotkey(tmp_path, recorder, captures)

    press_right_alt(hotkey)
    assert hotkey._mode is hotkeys_mod.RecordingMode.PUSH_TO_TALK
    release_right_alt(hotkey)
    assert wait_until(lambda: len(captures) == 1)

    assert recorder.start_count == 1
    assert recorder.stop_timeouts == [0.0]
    assert captures[0].path == tmp_path / "capture-1.wav"
    assert captures[0].end_reason is hotkeys_mod.CaptureEndReason.USER_COMPLETED


def test_left_ctrl_then_right_alt_toggles_hands_free(tmp_path):
    recorder = ImmediateRecorder(tmp_path)
    captures = []
    hotkey = build_hotkey(tmp_path, recorder, captures)

    activate_hands_free(hotkey)

    assert hotkey._mode is hotkeys_mod.RecordingMode.HANDS_FREE
    assert recorder.start_count == 1
    assert recorder.stop_timeouts == []
    assert captures == []

    stop_hands_free(hotkey, captures)

    assert hotkey._mode is hotkeys_mod.RecordingMode.IDLE
    assert recorder.stop_timeouts == [0.0]
    assert captures[0].end_reason is hotkeys_mod.CaptureEndReason.USER_COMPLETED


def test_plain_right_alt_is_ignored_while_hands_free_is_active(tmp_path):
    recorder = ImmediateRecorder(tmp_path)
    captures = []
    hotkey = build_hotkey(tmp_path, recorder, captures)
    activate_hands_free(hotkey)

    press_right_alt(hotkey)
    release_right_alt(hotkey)

    assert hotkey._mode is hotkeys_mod.RecordingMode.HANDS_FREE
    assert recorder.stop_timeouts == []
    assert "Right Alt ignored" in (tmp_path / "log.txt").read_text(encoding="utf-8")

    stop_hands_free(hotkey, captures)


def test_right_alt_then_left_ctrl_remains_push_to_talk(tmp_path):
    recorder = ImmediateRecorder(tmp_path)
    captures = []
    hotkey = build_hotkey(tmp_path, recorder, captures)

    press_right_alt(hotkey)
    press_left_ctrl(hotkey)
    release_right_alt(hotkey)
    release_left_ctrl(hotkey)
    assert wait_until(lambda: len(captures) == 1)

    assert recorder.start_count == 1
    assert captures[0].end_reason is hotkeys_mod.CaptureEndReason.USER_COMPLETED


def test_right_ctrl_does_not_activate_hands_free(tmp_path):
    recorder = ImmediateRecorder(tmp_path)
    captures = []
    hotkey = build_hotkey(tmp_path, recorder, captures)

    press_right_ctrl(hotkey)
    press_right_alt(hotkey)
    release_right_alt(hotkey)
    release_right_ctrl(hotkey)
    assert wait_until(lambda: len(captures) == 1)

    assert recorder.start_count == 1
    assert recorder.stop_timeouts == [0.0]


def test_right_alt_repeat_cannot_double_toggle(tmp_path):
    recorder = ImmediateRecorder(tmp_path)
    captures = []
    hotkey = build_hotkey(tmp_path, recorder, captures)

    press_left_ctrl(hotkey)
    press_right_alt(hotkey)
    press_right_alt(hotkey)
    release_right_alt(hotkey)

    assert hotkey._mode is hotkeys_mod.RecordingMode.HANDS_FREE
    assert recorder.start_count == 1
    assert recorder.stop_timeouts == []

    press_right_alt(hotkey)
    release_right_alt(hotkey)
    release_left_ctrl(hotkey)
    assert wait_until(lambda: len(captures) == 1)

    assert recorder.start_count == 1
    assert recorder.stop_timeouts == [0.0]


def test_ctrl_pressed_while_plain_alt_is_held_requires_fresh_alt_press(tmp_path):
    recorder = ImmediateRecorder(tmp_path)
    captures = []
    hotkey = build_hotkey(tmp_path, recorder, captures)
    activate_hands_free(hotkey)

    press_right_alt(hotkey)
    press_left_ctrl(hotkey)
    assert hotkey._mode is hotkeys_mod.RecordingMode.HANDS_FREE
    assert recorder.stop_timeouts == []

    release_right_alt(hotkey)
    press_right_alt(hotkey)
    release_right_alt(hotkey)
    release_left_ctrl(hotkey)
    assert wait_until(lambda: len(captures) == 1)


def test_hands_free_bell_mute_and_callback_order(monkeypatch, tmp_path):
    events: list[object] = []
    recorder = ImmediateRecorder(tmp_path, events=events)
    captures = []
    hotkey = build_hotkey(tmp_path, recorder, captures)
    hotkey._mute_controller = RecordingMuteController(events)
    monkeypatch.setattr(
        hotkeys_mod,
        "play_system_bell",
        lambda _path, *, purpose: events.append(("bell", purpose)),
    )
    hotkey._on_capture_finished = lambda result: events.append(("callback", result.end_reason))

    press_left_ctrl(hotkey)
    press_right_alt(hotkey)
    release_right_alt(hotkey)
    release_left_ctrl(hotkey)

    assert events == [("bell", "hands-free on"), "mute", "recorder_start"]

    press_left_ctrl(hotkey)
    press_right_alt(hotkey)
    assert wait_until(lambda: ("bell", "hands-free off") in events)
    assert not any(event[0] == "callback" for event in events if isinstance(event, tuple))

    release_right_alt(hotkey)
    release_left_ctrl(hotkey)
    assert wait_until(lambda: any(event[0] == "callback" for event in events if isinstance(event, tuple)))

    assert events == [
        ("bell", "hands-free on"),
        "mute",
        "recorder_start",
        ("recorder_stop", 0.0),
        "restore",
        ("bell", "hands-free off"),
        ("callback", hotkeys_mod.CaptureEndReason.USER_COMPLETED),
    ]


def test_wpctl_probe_mute_and_restore_use_audio_control_timeout(monkeypatch, tmp_path):
    check_output_calls = []
    run_calls = []

    def fake_check_output(command, **kwargs):
        check_output_calls.append((command, kwargs))
        return "Volume: 1.00"

    def fake_run(command, **kwargs):
        run_calls.append((command, kwargs))

    monkeypatch.setattr(hotkeys_mod.subprocess, "check_output", fake_check_output)
    monkeypatch.setattr(hotkeys_mod.subprocess, "run", fake_run)
    strategy = hotkeys_mod._WpctlStrategy(tmp_path / "log.txt")

    strategy.mute()
    strategy.restore()

    expected_timeout = hotkeys_mod.AUDIO_CONTROL_TIMEOUT_SECONDS
    assert check_output_calls == [
        (
            ["wpctl", "get-volume", "@DEFAULT_AUDIO_SINK@"],
            {"text": True, "timeout": expected_timeout},
        )
    ]
    assert run_calls == [
        (
            ["wpctl", "set-mute", "@DEFAULT_AUDIO_SINK@", "1"],
            {"check": True, "timeout": expected_timeout},
        ),
        (
            ["wpctl", "set-mute", "@DEFAULT_AUDIO_SINK@", "0"],
            {"check": True, "timeout": expected_timeout},
        ),
    ]


def test_pactl_probes_mute_and_restore_use_audio_control_timeout(monkeypatch, tmp_path):
    check_output_calls = []
    run_calls = []

    def fake_check_output(command, **kwargs):
        check_output_calls.append((command, kwargs))
        if command == ["pactl", "get-default-sink"]:
            return "test-sink\n"
        return "Mute: no\n"

    def fake_run(command, **kwargs):
        run_calls.append((command, kwargs))

    monkeypatch.setattr(hotkeys_mod.subprocess, "check_output", fake_check_output)
    monkeypatch.setattr(hotkeys_mod.subprocess, "run", fake_run)
    strategy = hotkeys_mod._PactlStrategy(tmp_path / "log.txt")

    strategy.mute()
    strategy.restore()

    expected_timeout = hotkeys_mod.AUDIO_CONTROL_TIMEOUT_SECONDS
    assert check_output_calls == [
        (
            ["pactl", "get-default-sink"],
            {"text": True, "timeout": expected_timeout},
        ),
        (
            ["pactl", "get-sink-mute", "test-sink"],
            {"text": True, "timeout": expected_timeout},
        ),
    ]
    assert run_calls == [
        (
            ["pactl", "set-sink-mute", "test-sink", "1"],
            {"check": True, "timeout": expected_timeout},
        ),
        (
            ["pactl", "set-sink-mute", "test-sink", "0"],
            {"check": True, "timeout": expected_timeout},
        ),
    ]


def test_unexpected_mute_failure_does_not_prevent_capture_start(monkeypatch, tmp_path):
    events: list[object] = []
    recorder = ImmediateRecorder(tmp_path, events=events)
    captures = []
    hotkey = build_hotkey(tmp_path, recorder, captures)
    hotkey._mute_controller = FailingMuteController(
        events,
        mute_error=OSError("mute mechanism disappeared"),
    )
    monkeypatch.setattr(
        hotkeys_mod,
        "play_system_bell",
        lambda _path, *, purpose: events.append(("bell", purpose)),
    )

    activate_hands_free(hotkey)

    assert hotkey._mode is hotkeys_mod.RecordingMode.HANDS_FREE
    assert recorder.start_count == 1
    assert events == [
        ("bell", "hands-free on"),
        "mute",
        "recorder_start",
    ]
    assert "Unexpected audio mute failure" in (tmp_path / "log.txt").read_text(encoding="utf-8")

    stop_hands_free(hotkey, captures)
    assert captures[0].path == tmp_path / "capture-1.wav"


def test_unexpected_restore_failure_does_not_prevent_interrupted_callback(tmp_path):
    events: list[object] = []
    recorder = ImmediateRecorder(tmp_path, events=events)
    captures = []
    hotkey = build_hotkey(tmp_path, recorder, captures)
    hotkey._mute_controller = FailingMuteController(
        events,
        restore_error=OSError("restore mechanism disappeared"),
    )

    def record_callback(result):
        captures.append(result)
        events.append("callback")

    hotkey._on_capture_finished = record_callback
    activate_hands_free(hotkey)

    hotkey.stop(end_reason=hotkeys_mod.CaptureEndReason.SERVICE_SHUTDOWN)

    assert hotkey._mode is hotkeys_mod.RecordingMode.IDLE
    assert hotkey._finalizing_capture is False
    assert len(captures) == 1
    assert captures[0].path == tmp_path / "capture-1.wav"
    assert captures[0].end_reason is hotkeys_mod.CaptureEndReason.SERVICE_SHUTDOWN
    assert events.index("restore") < events.index("callback")
    assert "Unexpected audio restore failure" in (tmp_path / "log.txt").read_text(encoding="utf-8")


def test_timed_out_restore_still_reaches_interrupted_capture_callback(monkeypatch, tmp_path):
    recorder = ImmediateRecorder(tmp_path)
    captures = []
    hotkey = build_hotkey(tmp_path, recorder, captures)
    activate_hands_free(hotkey)
    strategy = hotkeys_mod._WpctlStrategy(tmp_path / "log.txt")
    strategy._active = True
    strategy._previously_muted = False
    monkeypatch.setattr(
        hotkeys_mod.AudioMuteController,
        "_detect_strategy",
        lambda _self: strategy,
    )
    hotkey._mute_controller = hotkeys_mod.AudioMuteController(tmp_path / "log.txt")
    run_calls = []

    def time_out_restore(command, **kwargs):
        run_calls.append((command, kwargs))
        raise hotkeys_mod.subprocess.TimeoutExpired(command, kwargs["timeout"])

    monkeypatch.setattr(hotkeys_mod.subprocess, "run", time_out_restore)

    hotkey.stop(end_reason=hotkeys_mod.CaptureEndReason.SERVICE_SHUTDOWN)

    assert len(captures) == 1
    assert captures[0].path == tmp_path / "capture-1.wav"
    assert captures[0].end_reason is hotkeys_mod.CaptureEndReason.SERVICE_SHUTDOWN
    assert run_calls == [
        (
            ["wpctl", "set-mute", "@DEFAULT_AUDIO_SINK@", "0"],
            {
                "check": True,
                "timeout": hotkeys_mod.AUDIO_CONTROL_TIMEOUT_SECONDS,
            },
        )
    ]
    assert "Failed to restore audio: wpctl restore failed" in (
        tmp_path / "log.txt"
    ).read_text(encoding="utf-8")


def test_hands_free_release_wait_has_safety_timeout(monkeypatch, tmp_path):
    recorder = ImmediateRecorder(tmp_path)
    captures = []
    hotkey = build_hotkey(tmp_path, recorder, captures)
    monkeypatch.setattr(hotkeys_mod, "HOTKEY_RELEASE_TIMEOUT_SECONDS", 0.0)
    activate_hands_free(hotkey)

    press_left_ctrl(hotkey)
    press_right_alt(hotkey)
    assert wait_until(lambda: len(captures) == 1)

    assert "Timed out after 0.00s" in (tmp_path / "log.txt").read_text(encoding="utf-8")
    release_right_alt(hotkey)
    release_left_ctrl(hotkey)


def test_press_is_ignored_while_capture_finalizes(monkeypatch, tmp_path):
    clock = FakeClock()
    monkeypatch.setattr(hotkeys_mod.time, "monotonic", clock)
    recorder = BlockingRecorder(tmp_path)
    captures = []
    hotkey = build_hotkey(tmp_path, recorder, captures)

    press_right_alt(hotkey)
    release_right_alt(hotkey)
    assert recorder.stop_started.wait(timeout=1.0)

    clock.advance(2.0)
    press_right_alt(hotkey)
    release_right_alt(hotkey)
    assert recorder.start_count == 1
    assert "ignored during capture finalization" in (tmp_path / "log.txt").read_text(encoding="utf-8")

    recorder.allow_stop.set()
    assert wait_until(lambda: len(captures) == 1)

    press_right_alt(hotkey)
    assert recorder.start_count == 2


def test_press_is_ignored_during_post_release_grace_period(monkeypatch, tmp_path):
    clock = FakeClock()
    monkeypatch.setattr(hotkeys_mod.time, "monotonic", clock)
    recorder = ImmediateRecorder(tmp_path)
    captures = []
    hotkey = build_hotkey(tmp_path, recorder, captures)

    press_right_alt(hotkey)
    release_right_alt(hotkey)
    assert wait_until(lambda: len(captures) == 1)

    press_right_alt(hotkey)
    release_right_alt(hotkey)
    assert recorder.start_count == 1

    clock.advance(1.01)
    press_right_alt(hotkey)

    assert recorder.start_count == 2
    assert "ignored for 1.00s grace period" in (tmp_path / "log.txt").read_text(encoding="utf-8")


def test_capture_callback_keeps_aware_utc_recording_start_time(monkeypatch, tmp_path):
    capture_started_at = datetime(2026, 7, 28, 22, 30, 45, tzinfo=timezone.utc)
    recorder = ImmediateRecorder(tmp_path)
    timestamp_calls = []

    def capture_utc_now():
        assert recorder.start_count == 1
        timestamp_calls.append(capture_started_at)
        return capture_started_at

    monkeypatch.setattr(hotkeys_mod, "utc_now", capture_utc_now)
    captures = []
    hotkey = build_hotkey(tmp_path, recorder, captures)

    press_right_alt(hotkey)
    release_right_alt(hotkey)
    assert wait_until(lambda: len(captures) == 1)

    assert captures[0].created_at == capture_started_at
    assert timestamp_calls == [capture_started_at]
    assert captures[0].created_at.utcoffset().total_seconds() == 0


def test_start_failure_rolls_back_with_second_bell_and_no_callback(monkeypatch, tmp_path):
    error = hotkeys_mod.RecorderStartError("microphone unavailable", attempts=1)
    recorder = ImmediateRecorder(tmp_path, start_error=error)
    captures = []
    events: list[object] = []
    hotkey = build_hotkey(tmp_path, recorder, captures)
    hotkey._mute_controller = RecordingMuteController(events)
    monkeypatch.setattr(
        hotkeys_mod,
        "play_system_bell",
        lambda _path, *, purpose: events.append(("bell", purpose)),
    )

    press_left_ctrl(hotkey)
    press_right_alt(hotkey)
    release_right_alt(hotkey)
    release_left_ctrl(hotkey)

    assert hotkey._mode is hotkeys_mod.RecordingMode.IDLE
    assert recorder.stop_timeouts == []
    assert captures == []
    assert events == [
        ("bell", "hands-free on"),
        "mute",
        "restore",
        ("bell", "hands-free start failed"),
    ]


def test_injected_modifier_events_are_ignored(tmp_path):
    recorder = ImmediateRecorder(tmp_path)
    captures = []
    hotkey = build_hotkey(tmp_path, recorder, captures)

    press_left_ctrl(hotkey, injected=True)
    press_right_alt(hotkey, injected=True)
    release_right_alt(hotkey, injected=True)
    release_left_ctrl(hotkey, injected=True)

    assert recorder.start_count == 0
    assert hotkey._mode is hotkeys_mod.RecordingMode.IDLE


@pytest.mark.parametrize(
    "end_reason",
    [hotkeys_mod.CaptureEndReason.SERVICE_SHUTDOWN, hotkeys_mod.CaptureEndReason.ESCAPE],
)
def test_stop_synchronously_archives_active_capture_without_boundary_bell(
    monkeypatch,
    tmp_path,
    end_reason,
):
    bells = []
    monkeypatch.setattr(
        hotkeys_mod,
        "play_system_bell",
        lambda _path, *, purpose: bells.append(purpose),
    )
    recorder = ImmediateRecorder(tmp_path)
    captures = []
    hotkey = build_hotkey(tmp_path, recorder, captures)
    activate_hands_free(hotkey)
    bells.clear()

    hotkey.stop(end_reason=end_reason)

    assert recorder.stop_timeouts == [0.0]
    assert len(captures) == 1
    assert captures[0].end_reason is end_reason
    assert bells == []


def test_finalizer_failure_restores_audio_and_reports_empty_path(tmp_path):
    events: list[object] = []
    recorder = ImmediateRecorder(tmp_path, events=events, stop_error=RuntimeError("writer failed"))
    captures = []
    hotkey = build_hotkey(tmp_path, recorder, captures)
    hotkey._mute_controller = RecordingMuteController(events)

    activate_hands_free(hotkey)
    hotkey.stop(end_reason=hotkeys_mod.CaptureEndReason.SERVICE_SHUTDOWN)

    assert "restore" in events
    assert len(captures) == 1
    assert captures[0].path is None
    assert captures[0].end_reason is hotkeys_mod.CaptureEndReason.SERVICE_SHUTDOWN


def test_escape_delegates_central_shutdown_reason(tmp_path):
    exit_reasons = []
    hotkey = build_hotkey(
        tmp_path,
        ImmediateRecorder(tmp_path),
        [],
        on_exit=exit_reasons.append,
    )

    hotkey._handle_press(hotkeys_mod.keyboard.Key.esc)
    hotkey._handle_press(hotkeys_mod.keyboard.Key.esc)

    assert exit_reasons == [hotkeys_mod.CaptureEndReason.ESCAPE]


def test_unusable_log_path_does_not_block_escape_or_capture_lifecycle(tmp_path):
    blocking_file = tmp_path / "not_a_directory"
    blocking_file.write_text("blocked", encoding="utf-8")
    unusable_log_path = blocking_file / "runtime.log"
    recorder = ImmediateRecorder(tmp_path)
    captures = []
    exit_reasons = []
    hotkey = build_hotkey(
        tmp_path,
        recorder,
        captures,
        on_exit=exit_reasons.append,
        log_path=unusable_log_path,
    )

    press_right_alt(hotkey)
    release_right_alt(hotkey)
    assert wait_until(lambda: len(captures) == 1)

    hotkey._handle_press(hotkeys_mod.keyboard.Key.esc)

    assert recorder.start_count == 1
    assert recorder.stop_timeouts == [0.0]
    assert captures[0].path == tmp_path / "capture-1.wav"
    assert exit_reasons == [hotkeys_mod.CaptureEndReason.ESCAPE]
    assert blocking_file.read_text(encoding="utf-8") == "blocked"
