"""Supervise one inference process so unloading releases its CUDA context."""
from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
import time
from dataclasses import asdict
from pathlib import Path


class BackendError(RuntimeError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


class ModelProcess:
    def __init__(self, config, log_path):
        self.config = config
        self.log_path = log_path
        self.process = None
        self.ready = False
        self._lock = threading.Lock()
        self._responses = queue.Queue()

    @property
    def pid(self):
        process = self.process
        return process.pid if process is not None and process.poll() is None else None

    def load(self, cancelled=lambda: False, warmup=True):
        with self._lock:
            if self.ready and self.pid:
                return
            self.close()
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            self._responses = queue.Queue()
            with self.log_path.open("a", encoding="utf-8") as log:
                self.process = subprocess.Popen(
                    [str(Path(sys.executable).with_name("python.exe")) if os.name == "nt" else sys.executable,
                     "-u", "-m", "wa_whisper.inference_worker", "--parent-pid", str(os.getpid())],
                    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log, text=True,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
            process = self.process
            responses = self._responses

            def read_responses():
                try:
                    for line in process.stdout:
                        responses.put(json.loads(line))
                except (OSError, ValueError):
                    pass
                finally:
                    responses.put(None)

            threading.Thread(target=read_responses, daemon=True, name="whisper-inference-output").start()
            config = asdict(self.config)
            config["cache_dir"] = str(config["cache_dir"])
            try:
                self._call({"command": "load", "config": config, "log_path": str(self.log_path),
                            "warmup": warmup}, cancelled, 300)
                self.ready = True
            except BaseException:
                self.close()
                raise

    def transcribe(self, path, cancelled=lambda: False, options=None):
        self.load(cancelled)
        with self._lock:
            return self._call({"command": "transcribe", "path": str(path), "decode_options": options or {}}, cancelled, 1800)

    def _call(self, request, cancelled, timeout):
        try:
            self.process.stdin.write(json.dumps(request) + "\n")
            self.process.stdin.flush()
        except (OSError, ValueError) as exc:
            raise BackendError("unavailable", "Inference worker disconnected") from exc
        deadline = time.monotonic() + timeout
        while True:
            if cancelled() or time.monotonic() >= deadline:
                self.close()
                raise BackendError("cancelled", "Inference cancelled or timed out; recording preserved")
            try:
                response = self._responses.get(timeout=0.1)
            except queue.Empty:
                continue
            if response is None:
                self.ready = False
                raise BackendError("unavailable", "Inference worker exited")
            if not response["ok"]:
                raise BackendError(**response["error"])
            return response["result"]

    def close(self):
        self.ready = False
        process, self.process = self.process, None
        if process is None:
            return
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        for stream in (process.stdin, process.stdout):
            if stream:
                stream.close()
