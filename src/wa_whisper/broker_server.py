"""Authenticated loopback HTTP endpoint for the Windows model broker."""
from __future__ import annotations

import hmac
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .broker_state import MAX_AUDIO_BYTES, BrokerState
from .device_config import broker_settings, broker_token
from .model_process import ModelProcess

MAX_REQUEST_BYTES = MAX_AUDIO_BYTES * 4 // 3 + 8192


def make_server(state, token, port):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_POST(self):
            self.connection.settimeout(10)
            if self.path != "/rpc" or not hmac.compare_digest(
                self.headers.get("Authorization", ""), "Bearer " + token
            ):
                self.send_error(403)
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= MAX_REQUEST_BYTES:
                    raise ValueError("Invalid request length")
                payload = self.rfile.read(size)
                if len(payload) != size:
                    raise ValueError("Incomplete request")
                result = state.request(json.loads(payload))
                response = {"version": 1, "ok": True, "result": result}
            except (OSError, ValueError, RuntimeError, TypeError, KeyError) as exc:
                response = {"version": 1, "ok": False,
                            "error": {"code": getattr(exc, "code", "invalid_request"), "message": str(exc)}}
            data = json.dumps(response).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            try:
                self.wfile.write(data)
            except OSError:
                pass

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    server.daemon_threads = True
    return server


def main():
    root = Path.home() / ".cache/wa_whisper"
    root.mkdir(parents=True, exist_ok=True)
    if sys.stderr is None:
        sys.stderr = (root / "broker_server.log").open("a", encoding="utf-8", buffering=1)
    if sys.stdout is None:
        sys.stdout = sys.stderr
    from .whisper_backend import WhisperConfig

    settings = broker_settings()
    token = broker_token(settings)
    root = Path.home() / ".cache/wa_whisper"
    model = ModelProcess(WhisperConfig(device="cuda", compute_mode="gpu", fp16=True), root / "broker_model.log")
    # Bind before starting the scheduler: a duplicate broker cannot touch live spool files.
    class DeferredState:
        def request(self, request):
            return state.request(request)
    server = make_server(DeferredState(), token, settings["port"])
    state = BrokerState(model, root / "broker_spool")
    try:
        server.serve_forever(poll_interval=0.2)
    finally:
        server.server_close()
        state.close()


if __name__ == "__main__":
    main()
