import threading
import time

import pytest

from wa_whisper.processing_queue import CaptureQueue, CaptureWorker


def test_empty_queue_with_active_transcription_does_not_drain(tmp_path):
    tasks = CaptureQueue()
    entered = threading.Event()
    finish = threading.Event()
    events = []

    def process(value):
        entered.set()
        assert finish.wait(2)
        events.append(value)

    worker = CaptureWorker(tasks, process, tmp_path / "worker.log")
    worker.start()
    tasks.put("capture")
    assert entered.wait(1)
    assert tasks.empty()
    drained = threading.Event()

    def drain():
        worker.drain(time.monotonic() + 2, lambda: False)
        drained.set()

    waiter = threading.Thread(target=drain)
    waiter.start()
    try:
        assert not drained.wait(0.05)
        finish.set()
        assert drained.wait(1)
        assert events == ["capture"]
    finally:
        finish.set()
        tasks.put(None)
        worker.join(1)
        waiter.join(1)


def test_barrier_waits_for_all_queued_captures(tmp_path):
    tasks = CaptureQueue()
    events = []
    worker = CaptureWorker(tasks, events.append, tmp_path / "worker.log")
    worker.start()
    for item in range(4):
        tasks.put(item)
    try:
        worker.drain(time.monotonic() + 1, lambda: False)
        assert events == [0, 1, 2, 3]
    finally:
        tasks.put(None)
        worker.join(1)


def test_failed_processing_cannot_report_success_or_leave_join_stuck(tmp_path):
    tasks = CaptureQueue()

    def fail(_):
        raise OSError("synthetic archive cleanup failure")

    worker = CaptureWorker(tasks, fail, tmp_path / "worker.log")
    worker.start()
    tasks.put("capture")
    try:
        with pytest.raises(RuntimeError, match="failed"):
            worker.drain(time.monotonic() + 1, lambda: False)
    finally:
        tasks.put(None)
        worker.join(1)
    assert tasks.unfinished_tasks == 0


def test_shutdown_cannot_accept_a_barrier_after_its_sentinel(tmp_path):
    tasks = CaptureQueue()
    worker = CaptureWorker(tasks, lambda _: None, tmp_path / "worker.log")
    worker.start()
    tasks.put(None)
    with pytest.raises(RuntimeError, match="shutdown"):
        worker.drain(time.monotonic() + 1, lambda: False)
    worker.join(1)
    assert tasks.unfinished_tasks == 0
