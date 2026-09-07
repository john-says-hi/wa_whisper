"""Regression coverage for off-state reporting and idempotent activation."""

import io
import queue
from types import SimpleNamespace

from wa_whisper import windows_controller as controller


def make_controller(tmp_path, monkeypatch, process):
    monkeypatch.setattr(controller, "STATE_DIR", tmp_path)
    (tmp_path / "windows_status.txt").write_text("Ready", encoding="utf-8")
    instance = object.__new__(controller.DictationController)
    instance.process = process
    instance.stopping = False
    instance.start_pending = False
    instance.events = queue.Queue()
    instance.status = SimpleNamespace(set=lambda value: None)
    instance.root = SimpleNamespace(title=lambda value: None, after=lambda *args: None,
                                    deiconify=lambda: None, lift=lambda: None)
    return instance


def test_clean_worker_exit_replaces_stale_ready_status(tmp_path, monkeypatch):
    monkeypatch.setattr(controller, "take_request", lambda: None)
    process = SimpleNamespace(poll=lambda: 0, stdin=io.StringIO())
    instance = make_controller(tmp_path, monkeypatch, process)
    instance.poll()
    assert instance.process is None
    assert (tmp_path / "windows_status.txt").read_text(encoding="utf-8").startswith("OFF")


def test_start_request_activates_an_off_controller(tmp_path, monkeypatch):
    monkeypatch.setattr(controller, "take_request", lambda: "start")
    instance = make_controller(tmp_path, monkeypatch, None)
    starts = []
    instance.start = lambda: starts.append(True)
    instance.poll()
    assert starts == [True]


def test_start_request_does_not_toggle_a_running_worker_off(tmp_path, monkeypatch):
    monkeypatch.setattr(controller, "take_request", lambda: "start")
    process = SimpleNamespace(poll=lambda: None)
    instance = make_controller(tmp_path, monkeypatch, process)
    instance.start = lambda: (_ for _ in ()).throw(AssertionError("Must not restart"))
    instance.toggle = lambda: (_ for _ in ()).throw(AssertionError("Must not stop"))
    instance.poll()
    assert instance.process is process
