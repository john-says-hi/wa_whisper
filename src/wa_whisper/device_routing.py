"""Desktop destination lifecycle. Microphone and archives never leave the desktop."""
from __future__ import annotations

import fcntl
import os
import threading
import time
import uuid
from pathlib import Path

from .admission import new_capture_reason
from .broker_client import BrokerClient
from .broker_state import DECODE_FIELDS
from .device_config import read_destination, save_destination
from .device_notices import DeviceNotices
from .model_process import BackendError, ModelProcess
from .whisper_backend import WhisperResult, WhisperSegment


class RoutedBackend:
    def __init__(self, config, log_path):
        self.config = config
        self.log_path = log_path
        self.destination = read_destination()
        self.local = ModelProcess(config, log_path)
        self.remote = None
        self.notices = DeviceNotices(log_path)
        self.capture = None
        self.worker = None
        self.recovery = None
        self._closed = threading.Event()
        self._cancel_switch = threading.Event()
        self._switching = False
        self._switch_lock = threading.Lock()
        self._inference_lock = threading.Lock()
        self._switch_thread = None

    def bind(self, capture, worker, archive):
        from .device_recovery import DeviceRecovery
        self.capture, self.worker = capture, worker
        self.notices.capture = capture
        self.recovery = DeviceRecovery(self, archive)
        self.recovery.start()

    def enable_cooperative_capture(self):
        pass

    def capture_admission_reason(self):
        if self.destination == "laptop":
            return None
        return new_capture_reason() if self.config.device != "cpu" else None

    def archive_metadata(self):
        return {"model_name": self.config.model_name, "destination": self.destination,
                "device": "cuda" if self.destination == "laptop" else self.config.device,
                "compute_mode": self.config.compute_mode, "fp16": self.config.fp16,
                "beam_size": self.config.beam_size, "temperature": self.config.temperature}

    def status(self):
        selected = self.remote if self.destination == "laptop" else self.local
        return {"destination": self.destination, "switching": self._switching,
                "ready": bool(selected and selected.ready), "local_model_pid": self.local.pid}

    def _remote(self):
        if self.remote is None:
            self.remote = BrokerClient()
        return self.remote

    def transcribe(self, path, *, recovery=False):
        with self._inference_lock:
            if recovery and (self._switching or self.capture_admission_reason()):
                raise BackendError("busy", "Recovery waits for the selected GPU")
            cancelled = lambda: (self._closed.is_set() or (recovery and self._switching)
                                  or (self._switching and self.destination == "laptop"
                                      and (self.remote is None or not self.remote.ready)))
            selected = self._remote() if self.destination == "laptop" else self.local
            try:
                if self.destination == "laptop":
                    if self.config.model_name != "large-v3":
                        raise BackendError("configuration", "Laptop broker requires the large-v3 model")
                    selected.decode_options = {key: getattr(self.config, key) for key in DECODE_FIELDS}
                result = selected.transcribe(path, cancelled)
                return WhisperResult(text=result["text"], info=result.get("info", {}),
                                     segments=[WhisperSegment(**segment) for segment in result.get("segments", [])])
            except BackendError as exc:
                if exc.code == "memory_full":
                    self.notices.say(self.destination + "_memory_full")
                elif self.destination == "laptop" and exc.code != "cancelled":
                    self.notices.say("laptop_offline", str(exc))
                raise

    def acknowledge(self, path):
        if self.destination == "laptop" and self.remote:
            self.remote.acknowledge(path)

    def request_switch(self):
        if self._closed.is_set() or not self._switch_lock.acquire(blocking=False):
            return {"accepted": False, **self.status()}
        self._cancel_switch.clear()
        self._switching = True
        self._switch_thread = threading.Thread(target=self._switch, daemon=True, name="whisper-device-switch")
        self.notices.say("transferring_voice")
        self._switch_thread.start()
        return {"accepted": True, **self.status()}

    def cancel_switch(self):
        self._cancel_switch.set()
        return {"cancelled": True}

    def _switch(self):
        target = "laptop" if self.destination == "desktop" else "desktop"
        token = uuid.uuid4().hex
        candidate = None
        committed = False
        cancelled = lambda: self._closed.is_set() or self._cancel_switch.is_set()
        lock = None
        try:
            runtime = Path(os.environ.get("XDG_RUNTIME_DIR", f"/run/user/{os.getuid()}"))
            lock = (runtime / "wa-whisper-power-toggle.lock").open("a")
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise BackendError("busy", "Another power or desktop GPU operation is active") from exc
            self.capture.begin_quiesce(token)
            deadline = time.monotonic() + 300
            self.capture.wait_capture_idle(token, deadline, cancelled)
            self.worker.drain(deadline, cancelled)
            with self._inference_lock:
                if target == "desktop" and self.capture_admission_reason_for_desktop():
                    raise BackendError("busy", "Desktop GPU is reserved by another job")
                if target == "laptop" and self.config.model_name != "large-v3":
                    raise BackendError("configuration", "Laptop broker requires the large-v3 model")
                candidate = BrokerClient() if target == "laptop" else self.local
                candidate.load(lambda: cancelled() or time.monotonic() >= deadline)
                if cancelled():
                    raise BackendError("cancelled", "Device switch cancelled")
                save_destination(target)
                previous = self.remote if self.destination == "laptop" else self.local
                if target == "laptop":
                    self.remote = candidate
                else:
                    self.remote = None
                self.destination = target
                committed = True
                if previous:
                    previous.close()
        except (OSError, ValueError, RuntimeError) as exc:
            if not committed and candidate is not None:
                candidate.close()
            code = getattr(exc, "code", "switch_failed")
            if not cancelled():
                name = (target + "_memory_full" if code == "memory_full" else
                        "desktop_busy" if code == "busy" else
                        "laptop_offline" if target == "laptop" and code == "offline" else "switch_failed")
                self.notices.say(name, str(exc))
        finally:
            if self.capture:
                self.capture.cancel_quiesce(token)
            if lock:
                lock.close()
            self._switching = False
            if committed and not self._closed.is_set():
                self.notices.say("voice_ready")
            self._switch_lock.release()

    def capture_admission_reason_for_desktop(self):
        return new_capture_reason() if self.config.device != "cpu" else None

    def begin_shutdown(self):
        self._closed.set()
        self._cancel_switch.set()

    def close(self):
        self._closed.set()
        self._cancel_switch.set()
        if self._switch_thread:
            self._switch_thread.join(timeout=15)
        if self.recovery:
            self.recovery.close()
        with self._inference_lock:
            self.local.close()
            if self.remote:
                self.remote.close()
        self.notices.close()
