"""Track transcription completion without treating an empty queue as idle."""

from __future__ import annotations

import queue
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from .log_utils import write_log


@dataclass
class DrainBarrier:
    completed: threading.Event = field(default_factory=threading.Event)


class CaptureQueue(queue.Queue):
    """Never admit a drain barrier after the shutdown sentinel."""

    def __init__(self) -> None:
        super().__init__()
        self._admission = threading.Lock()
        self._closing = False

    def put(self, item, block: bool = True, timeout: float | None = None) -> None:
        with self._admission:
            if isinstance(item, DrainBarrier) and self._closing:
                raise RuntimeError("Whisper shutdown already began")
            if item is None:
                self._closing = True
            super().put(item, block, timeout)


class CaptureWorker:
    def __init__(self, tasks: queue.Queue, process: Callable, log_path: Path) -> None:
        self._tasks = tasks
        self._process = process
        self._log_path = log_path
        self._active = threading.Event()
        self._failed = threading.Event()
        self._thread = threading.Thread(target=self._run, name="whisper-captures", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def join(self, timeout: float | None = None) -> None:
        self._thread.join(timeout)

    def status(self) -> dict[str, bool | int]:
        return {
            "processing": self._active.is_set(),
            "queued_items": self._tasks.qsize(),
            "worker_alive": self._thread.is_alive(),
            "worker_failed": self._failed.is_set(),
        }

    def drain(self, deadline: float, cancelled: Callable[[], bool]) -> None:
        barrier = DrainBarrier()
        self._tasks.put(barrier)
        while True:
            if cancelled():
                raise InterruptedError("Handoff requester disconnected or shutdown began")
            if self._failed.is_set() or not self._thread.is_alive():
                raise RuntimeError("Capture worker failed or stopped; cannot confirm completed transcription")
            if barrier.completed.is_set():
                return
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("Timed out waiting for pending transcription to finish")
            barrier.completed.wait(min(0.05, remaining))

    def _run(self) -> None:
        while True:
            item = self._tasks.get()
            try:
                if item is None:
                    return
                if isinstance(item, DrainBarrier):
                    item.completed.set()
                    continue
                self._active.set()
                self._process(item)
            except Exception as exc:  # noqa: BLE001 - preserve queue accounting at the worker boundary.
                self._failed.set()
                write_log(f"Capture worker failed ({type(exc).__name__}); handoff unavailable", self._log_path)
            finally:
                self._active.clear()
                self._tasks.task_done()
