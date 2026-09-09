import time
from unittest.mock import Mock

import pytest

from wa_whisper import admission, hotkeys, whisper_backend
import wa_whisper.main as main_module


def test_absent_configuration_keeps_optional_dependency_absent(monkeypatch, tmp_path):
    monkeypatch.setattr(admission.Path, "home", lambda: tmp_path)
    load = Mock(side_effect=AssertionError("optional runtime imported"))
    monkeypatch.setattr(admission.importlib, "import_module", load)
    assert admission.new_capture_reason() is None
    load.assert_not_called()


def test_enabled_missing_runtime_denies_new_capture(monkeypatch, tmp_path):
    monkeypatch.setattr(admission.Path, "home", lambda: tmp_path)
    config = tmp_path / ".config/local_media/admission.json"
    config.parent.mkdir(parents=True)
    config.write_text('{"enabled":true,"module_root":"/missing/runtime"}')
    assert "unavailable" in admission.new_capture_reason()


def test_capture_gate_checks_queue_without_claiming_or_waiting(monkeypatch):
    runtime = Mock()
    lease = runtime.configured_lease.return_value
    lease.queue.snapshot.return_value = [
        {"state": "waiting", "owner_alive": True, "owner": "media"},
    ]
    monkeypatch.setattr(admission, "_runtime", lambda: runtime)
    assert admission.new_capture_reason() == "while shared GPU work is waiting (media)"
    lease.acquire.assert_not_called()
    lease.queue.enqueue.assert_not_called()


def test_recoverable_dead_utility_does_not_permanently_block_dictation(monkeypatch):
    runtime = Mock()
    runtime.configured_lease.return_value.queue.snapshot.return_value = [{
        "state": "active", "owner_alive": False, "owner": "preview",
        "metadata": {"recover_when_process_exits": True}, "children": [],
    }]
    monkeypatch.setattr(admission, "_runtime", lambda: runtime)
    assert admission.new_capture_reason() is None


def test_live_child_keeps_dead_utility_blocking_dictation(monkeypatch):
    runtime = Mock()
    runtime.process_identity.return_value = "same-start"
    runtime.configured_lease.return_value.queue.snapshot.return_value = [{
        "state": "active", "owner_alive": False, "owner": "preview",
        "metadata": {"recover_when_process_exits": True},
        "children": [{"pid": 22, "start": "same-start"}],
    }]
    monkeypatch.setattr(admission, "_runtime", lambda: runtime)
    assert "preview" in admission.new_capture_reason()


@pytest.mark.parametrize("hands_free", [False, True])
def test_pending_gpu_work_blocks_next_capture_but_current_audio_finishes(monkeypatch, tmp_path, hands_free):
    monkeypatch.setattr(hotkeys, "play_system_bell", lambda *_args, **_kwargs: None)
    blocked = False
    recorder = Mock()
    recorder.start.return_value = tmp_path / "capture.wav"
    recorder.stop.return_value = tmp_path / "capture.wav"
    recorder.last_capture_stats.return_value = None
    results = []
    controller = hotkeys.PushToTalkHotkey(
        recorder, silence_timeout=0, on_capture_finished=results.append,
        log_path=tmp_path / "hotkeys.log", enable_audio_mute=False,
        enable_hotkey_shield=False, hotkey_repress_grace_seconds=0,
        capture_admission=lambda: "while media owns GPU" if blocked else None,
    )

    def press():
        if hands_free:
            controller._handle_press(hotkeys.keyboard.Key.ctrl_l)
        controller._handle_press(hotkeys.keyboard.Key.alt_r)

    def release():
        controller._handle_release(hotkeys.keyboard.Key.alt_r)
        if hands_free:
            controller._handle_release(hotkeys.keyboard.Key.ctrl_l)

    try:
        press()
        if hands_free:
            release()
        blocked = True
        if hands_free:
            press()
        release()
        # This gate only waits for the existing capture's callback boundary.
        controller.begin_quiesce("test-drain")
        controller.wait_capture_idle("test-drain", time.monotonic() + 2, lambda: False)
        controller.cancel_quiesce("test-drain")
        assert len(results) == 1
        assert results[0].end_reason is hotkeys.CaptureEndReason.USER_COMPLETED
        press()
        release()
        recorder.start.assert_called_once()
        assert "while media owns GPU" in (tmp_path / "hotkeys.log").read_text()
    finally:
        controller.stop()


def make_backend(tmp_path, device="cuda"):
    return whisper_backend.WhisperBackend(
        whisper_backend.WhisperConfig(device=device, cache_dir=tmp_path / "models"),
        tmp_path / "backend.log",
    )


def test_standalone_gpu_backend_claims_before_loading(monkeypatch, tmp_path):
    events = []
    monkeypatch.setattr(whisper_backend, "ensure_process_admission", lambda: events.append("claim"))
    monkeypatch.setattr(whisper_backend.whisper, "load_model", lambda *_args, **_kwargs: events.append("load") or object())
    backend = make_backend(tmp_path)
    backend.load()
    assert events == ["claim", "load"]


def test_managed_accepted_capture_load_does_not_wait_behind_its_own_quiesce(monkeypatch, tmp_path):
    denied = Mock(side_effect=AssertionError("accepted audio joined foreground queue"))
    monkeypatch.setattr(whisper_backend, "ensure_process_admission", denied)
    monkeypatch.setattr(whisper_backend.whisper, "load_model", lambda *_args, **_kwargs: object())
    backend = make_backend(tmp_path)
    backend.enable_cooperative_capture()
    backend.load()
    assert backend._model is not None
    denied.assert_not_called()


def test_ram_capture_does_not_depend_on_gpu_admission(monkeypatch, tmp_path):
    forbidden = Mock(side_effect=AssertionError("RAM mode consulted GPU queue"))
    monkeypatch.setattr(whisper_backend, "new_capture_reason", forbidden)
    monkeypatch.setattr(whisper_backend, "ensure_process_admission", forbidden)
    monkeypatch.setattr(whisper_backend.whisper, "load_model", lambda *_args, **_kwargs: object())
    backend = make_backend(tmp_path, device="cpu")
    assert backend.capture_admission_reason() is None
    backend.load()
    forbidden.assert_not_called()


def test_utility_claim_lasts_until_process_exit(monkeypatch):
    monkeypatch.setattr(admission, "_process_lease", None)
    runtime = Mock()
    monkeypatch.setattr(admission, "_runtime", lambda: runtime)
    admission.ensure_process_admission()
    admission.ensure_process_admission()
    lease = runtime.configured_lease.return_value
    lease.acquire.assert_called_once_with(timeout=3600)
    lease.recover_when_process_exits.assert_called_once()
    lease.close.assert_not_called()


def test_restoration_starts_control_before_hotkeys_without_queue_wait(monkeypatch, tmp_path):
    events = []
    control, hotkey, backend = Mock(), Mock(), Mock()
    control.start.side_effect = lambda: events.append("control")
    backend.enable_cooperative_capture.side_effect = lambda: events.append("managed")
    hotkey.start.side_effect = lambda: events.append("hotkeys")
    monkeypatch.setattr(main_module, "shared_admission_enabled", lambda: True)
    main_module.start_capture_service(control, hotkey, backend, tmp_path / "startup.log")
    assert events == ["control", "managed", "hotkeys"]
    backend.load.assert_not_called()
    backend.capture_admission_reason.assert_not_called()


def test_enabled_admission_requires_control_endpoint_before_hotkeys(monkeypatch, tmp_path):
    control, hotkey, backend = Mock(), Mock(), Mock()
    control.start.side_effect = OSError("socket in use")
    monkeypatch.setattr(main_module, "shared_admission_enabled", lambda: True)
    with pytest.raises(RuntimeError, match="requires the Whisper handoff endpoint"):
        main_module.start_capture_service(control, hotkey, backend, tmp_path / "startup.log")
    hotkey.start.assert_not_called()
    backend.enable_cooperative_capture.assert_not_called()
