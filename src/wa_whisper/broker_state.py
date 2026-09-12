"""Single-model scheduler with client leases and idempotent transcription jobs."""
from __future__ import annotations

import base64
import hashlib
import io
import threading
import time
import uuid
import wave
from pathlib import Path

from .model_process import BackendError

LEASE_SECONDS = 20
RESULT_SECONDS = 3600
MAX_AUDIO_BYTES = 64 * 1024 * 1024
MAX_PENDING_JOBS = 64
DECODE_FIELDS = {"temperature", "beam_size", "best_of", "patience", "compression_ratio_threshold",
                 "logprob_threshold", "no_speech_threshold", "initial_prompt", "suppress_tokens",
                 "condition_on_previous_text"}


class BrokerState:
    def __init__(self, model, spool: Path):
        self.model = model
        self.spool = spool
        spool.mkdir(parents=True, exist_ok=True)
        for path in spool.glob("*.wav"):
            path.unlink()
        self.sessions = {}
        self.jobs = {}
        self.error = None
        self.loading = False
        self._condition = threading.Condition(threading.RLock())
        self._closed = threading.Event()
        self._last_client = None
        self._worker = threading.Thread(target=self._run, daemon=True, name="whisper-broker-inference")
        self._worker.start()

    def request(self, request):
        if request.get("version") != 1:
            raise ValueError("Unsupported broker protocol version")
        action = request.get("action")
        with self._condition:
            self._expire()
            if action == "acquire":
                client = request.get("client")
                if client not in ("desktop", "laptop"):
                    raise ValueError("Unknown client")
                session = uuid.uuid4().hex
                self.sessions[session] = {"client": client, "expires": time.monotonic() + LEASE_SECONDS}
                self.error = None
                self._condition.notify_all()
                return {"session": session}
            session = request.get("session")
            if session not in self.sessions:
                raise BackendError("session_expired", "Model-use session expired")
            client = self.sessions[session]["client"]
            self.sessions[session]["expires"] = time.monotonic() + LEASE_SECONDS
            if action in ("status", "heartbeat"):
                return {"ready": self.model.ready and bool(self.model.pid), "loading": self.loading,
                        "model_pid": self.model.pid, "clients": len(self.sessions), "error": self.error}
            if action == "release":
                del self.sessions[session]
                self._condition.notify_all()
                return {}
            job_id = request.get("job_id")
            if not isinstance(job_id, str) or not 1 <= len(job_id) <= 128 or not job_id.isalnum():
                raise ValueError("Invalid recording identifier")
            key = (client, job_id)
            if action == "submit":
                return self._submit(key, request)
            job = self.jobs.get(key)
            if job is None:
                raise BackendError("not_found", "Recording is not queued")
            if action == "result":
                return {name: job[name] for name in ("state", "result", "error") if name in job}
            if action == "acknowledge":
                if job["state"] not in ("completed", "failed"):
                    raise BackendError("busy", "Recording is still processing")
                del self.jobs[key]
                return {}
            raise ValueError("Unsupported broker action")

    def _submit(self, key, request):
        if not self.model.ready or self.error:
            raise BackendError("unavailable", "Laptop model is not ready")
        options = request.get("decode_options", {})
        if not isinstance(options, dict) or not set(options).issubset(DECODE_FIELDS):
            raise ValueError("Invalid decoding options")
        encoded = request.get("audio")
        if not isinstance(encoded, str) or len(encoded) > (MAX_AUDIO_BYTES * 4 // 3 + 4):
            raise ValueError("Recording exceeds transfer limit")
        audio = base64.b64decode(encoded, validate=True)
        digest = hashlib.sha256(audio).hexdigest()
        if key in self.jobs:
            if self.jobs[key]["digest"] != digest:
                raise ValueError("Recording identifier was reused for different audio")
            return {"state": self.jobs[key]["state"]}
        if len(self.jobs) >= MAX_PENDING_JOBS:
            raise BackendError("busy", "Laptop recording queue is full; audio remains on its source computer")
        with wave.open(io.BytesIO(audio), "rb") as recording:
            if recording.getnchannels() not in (1, 2) or recording.getsampwidth() not in (2, 3, 4):
                raise ValueError("Unsupported WAV format")
            if not 8000 <= recording.getframerate() <= 192000 or recording.getnframes() == 0:
                raise ValueError("Invalid WAV recording")
        path = self.spool / (uuid.uuid4().hex + ".wav")
        path.write_bytes(audio)
        self.jobs[key] = {"state": "queued", "path": path, "digest": digest, "created": time.monotonic(), "decode_options": options}
        self._condition.notify_all()
        return {"state": "queued"}

    def _expire(self):
        now = time.monotonic()
        self.sessions = {key: value for key, value in self.sessions.items() if value["expires"] > now}
        clients = {value["client"] for value in self.sessions.values()}
        for key, job in list(self.jobs.items()):
            if job["state"] == "queued" and key[0] not in clients:
                job.update(state="failed", error={"code": "session_expired", "message": "Client disconnected"})
                job["path"].unlink(missing_ok=True)
            if job["state"] in ("completed", "failed") and now - job["created"] > RESULT_SECONDS:
                job["path"].unlink(missing_ok=True)
                del self.jobs[key]

    def _next_job(self):
        queued = [key for key, job in self.jobs.items() if job["state"] == "queued"]
        return next((key for key in queued if key[0] != self._last_client), queued[0] if queued else None)

    def _run(self):
        while not self._closed.is_set():
            with self._condition:
                self._expire()
                if not self.sessions:
                    # No request can acquire a lease during this bounded unload.
                    self.model.close()
                    self._condition.wait(0.2)
                    continue
                if self.model.ready and not self.model.pid:
                    self.model.ready = False
                load = not self.model.ready and self.error is None
                key = self._next_job() if self.model.ready else None
                if not load and key is None:
                    self._condition.wait(0.2)
                    continue
                self.loading = load
                if key is not None:
                    self.jobs[key]["state"] = "running"
                    job = self.jobs[key]
                    self._last_client = key[0]
            try:
                if load:
                    self.model.load(self._closed.is_set)
                else:
                    result = self.model.transcribe(job["path"], self._closed.is_set, job["decode_options"])
                    with self._condition:
                        job.update(state="completed", result=result)
            except Exception as exc:  # noqa: BLE001 - isolate model-library failures from the broker
                error = {"code": getattr(exc, "code", "inference_failed"), "message": str(exc)}
                with self._condition:
                    if load:
                        self.error = error
                    else:
                        job.update(state="failed", error=error)
                if load or error["code"] in ("memory_full", "unavailable"):
                    self.model.close()
            finally:
                with self._condition:
                    self.loading = False
                    if key is not None:
                        job["path"].unlink(missing_ok=True)
                    self._condition.notify_all()
        self.model.close()

    def close(self):
        self._closed.set()
        with self._condition:
            self._condition.notify_all()
        self._worker.join(timeout=10)
        self.model.close()
        for path in self.spool.glob("*.wav"):
            path.unlink(missing_ok=True)
