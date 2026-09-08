"""Cooperate with dictation before committing to a GPU handoff shutdown."""

from __future__ import annotations

import os
import threading
import time
import uuid
from collections.abc import Callable

from .processing_queue import CaptureWorker


class HandoffBusyError(RuntimeError):
    pass


class HandoffController:
    def __init__(self, capture, worker: CaptureWorker, shutdown: Callable[[], None]) -> None:
        self._capture = capture
        self._worker = worker
        self._shutdown = shutdown
        self._owner_lock = threading.Lock()
        self._committed = threading.Event()

    def status(self) -> dict:
        return {
            "pid": os.getpid(),
            **self._capture.capture_state(),
            **self._worker.status(),
            "stopping": self._committed.is_set(),
        }

    def quiesce_stop(self, wait_seconds: float, cancelled: Callable[[], bool]) -> dict:
        if not self._owner_lock.acquire(blocking=False):
            raise HandoffBusyError("Another GPU handoff is already waiting")
        token = uuid.uuid4().hex
        committed = False
        try:
            if self._committed.is_set():
                raise HandoffBusyError("Whisper is already stopping")
            self._capture.begin_quiesce(token)
            deadline = time.monotonic() + wait_seconds
            self._capture.wait_capture_idle(token, deadline, cancelled)
            self._worker.drain(deadline, cancelled)
            if cancelled():
                raise InterruptedError("Handoff requester disconnected before shutdown")
            if time.monotonic() >= deadline:
                raise TimeoutError("Handoff deadline elapsed before shutdown")
            self._shutdown()
            self._committed.set()
            committed = True
            return {"pid": os.getpid(), "operation_id": token, "state": "stopping"}
        finally:
            if not committed:
                self._capture.cancel_quiesce(token)
            self._owner_lock.release()
