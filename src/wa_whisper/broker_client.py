"""Lease-backed laptop inference, with SSH transport and independent heartbeats."""
from __future__ import annotations

import base64
import hashlib
import json
import subprocess
import threading
import time
import urllib.error
import urllib.request

from .device_config import broker_settings, broker_token
from .model_process import BackendError


class BrokerClient:
    def __init__(self, remote=True):
        self.settings = broker_settings()
        self.token = broker_token(self.settings)
        self.remote = remote
        self.client = "desktop" if remote else "laptop"
        self.session = None
        self.decode_options = {}
        self.ready = False
        self._tunnel = None
        self._closed = threading.Event()
        self._connection_lock = threading.Lock()
        self._heartbeat = None

    def _rpc(self, action, **values):
        port = self.settings["local_port" if self.remote else "port"]
        body = json.dumps({"version": 1, "action": action, "session": self.session, **values}).encode()
        request = urllib.request.Request(f"http://127.0.0.1:{port}/rpc", data=body,
                                         headers={"Authorization": "Bearer " + self.token,
                                                  "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=5) as response:
                result = json.load(response)
        except (OSError, ValueError, urllib.error.URLError) as exc:
            self.ready = False
            raise BackendError("offline", "Laptop offline") from exc
        if result.get("version") != 1 or not isinstance(result.get("ok"), bool):
            raise BackendError("unavailable", "Laptop broker protocol mismatch")
        if not result["ok"]:
            raise BackendError(**result["error"])
        return result["result"]

    def _ensure_connection(self):
        with self._connection_lock:
            if self._closed.is_set():
                raise BackendError("cancelled", "Laptop connection is closing")
            if self.remote and (self._tunnel is None or self._tunnel.poll() is not None):
                self._tunnel = subprocess.Popen(
                    ["ssh", "-N", "-T", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes",
                     "-o", "ConnectTimeout=5", "-o", "ExitOnForwardFailure=yes", "-o", "ServerAliveInterval=5",
                     "-o", "ServerAliveCountMax=2", "-L",
                     f"127.0.0.1:{self.settings['local_port']}:127.0.0.1:{self.settings['port']}",
                     "--", self.settings["host"]], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                self.session = None
                deadline = time.monotonic() + 5
                while time.monotonic() < deadline and not self._closed.wait(0.1):
                    if self._tunnel.poll() is not None:
                        raise BackendError("offline", "Cannot connect to laptop")
                    try:
                        self.session = self._rpc("acquire", client=self.client)["session"]
                        break
                    except BackendError:
                        continue
            if not self.session:
                self.session = self._rpc("acquire", client=self.client)["session"]

    def _start_heartbeat(self):
        if self._heartbeat is not None:
            return
        def heartbeat():
            while not self._closed.wait(5):
                try:
                    self._ensure_connection()
                    status = self._rpc("heartbeat")
                    self.ready = bool(status["ready"])
                except BackendError as exc:
                    self.ready = False
                    if exc.code == "session_expired":
                        self.session = None
        self._heartbeat = threading.Thread(target=heartbeat, daemon=True, name="whisper-laptop-heartbeat")
        self._heartbeat.start()

    def load(self, cancelled=lambda: False, warmup=True):
        self._ensure_connection()
        self._start_heartbeat()
        deadline = time.monotonic() + 300
        while not self._closed.is_set() and not cancelled():
            try:
                status = self._rpc("status")
            except BackendError as exc:
                if exc.code != "session_expired":
                    raise
                self.session = None
                self._ensure_connection()
                continue
            if status["error"]:
                raise BackendError(**status["error"])
            if status["ready"]:
                self.ready = True
                return
            if time.monotonic() >= deadline:
                raise BackendError("timeout", "Laptop model loading timed out")
            self._closed.wait(0.2)
        raise BackendError("cancelled", "Laptop model loading cancelled")

    def transcribe(self, path, cancelled=lambda: False):
        self.load(cancelled)
        audio = path.read_bytes()
        # A stable recording path and contents identify retries across sessions.
        job_id = hashlib.sha256(str(path).encode() + audio).hexdigest()
        self._rpc("submit", job_id=job_id, audio=base64.b64encode(audio).decode("ascii"), decode_options=self.decode_options)
        deadline = time.monotonic() + 1800
        while not self._closed.is_set() and not cancelled():
            result = self._rpc("result", job_id=job_id)
            if result["state"] == "completed":
                return result["result"]
            if result["state"] == "failed":
                self._rpc("acknowledge", job_id=job_id)
                raise BackendError(**result["error"])
            if time.monotonic() >= deadline:
                raise BackendError("timeout", "Laptop transcription timed out; audio saved")
            self._closed.wait(0.2)
        raise BackendError("cancelled", "Laptop transcription cancelled; audio saved")

    def acknowledge(self, path):
        audio = path.read_bytes()
        job_id = hashlib.sha256(str(path).encode() + audio).hexdigest()
        try:
            self._rpc("acknowledge", job_id=job_id)
        except BackendError:
            pass

    def close(self):
        self._closed.set()
        self.ready = False
        if self.session:
            try:
                self._rpc("release")
            except BackendError:
                pass
            self.session = None
        if self._tunnel is not None:
            self._tunnel.terminate()
            try:
                self._tunnel.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self._tunnel.kill()
                self._tunnel.wait(timeout=2)
            self._tunnel = None
        if self._heartbeat and self._heartbeat is not threading.current_thread():
            self._heartbeat.join(timeout=6)


class LaptopBackend:
    """The Windows microphone is a client, never a second model owner."""
    def __init__(self, cancelled=lambda: False):
        self.connection = BrokerClient(remote=False)
        self.cancelled = cancelled

    def load(self):
        self.connection.load(self.cancelled)

    def transcribe(self, path):
        from .whisper_backend import WhisperResult, WhisperSegment
        result = self.connection.transcribe(path, self.cancelled)
        return WhisperResult(text=result["text"], info=result.get("info", {}),
                             segments=[WhisperSegment(**segment) for segment in result.get("segments", [])])

    def acknowledge(self, path):
        self.connection.acknowledge(path)

    def archive_metadata(self):
        return {"model_name": "large-v3", "device": "cuda", "compute_mode": "gpu",
                "fp16": True, "destination": "laptop", "shared_model": True}

    def close(self):
        self.connection.close()
