"""Small local-only control protocol shared by Whisper and its CLI."""

from __future__ import annotations

import json
import math
import os
import socket
from pathlib import Path

PROTOCOL_VERSION = 1
MAX_FRAME_BYTES = 4096
MAX_WAIT_SECONDS = 300


def default_socket_path() -> Path:
    runtime = Path(os.environ.get("XDG_RUNTIME_DIR", f"/run/user/{os.getuid()}"))
    return runtime / "wa_whisper" / "control.sock"


def read_frame(connection: socket.socket) -> dict:
    data = bytearray()
    while len(data) <= MAX_FRAME_BYTES:
        chunk = connection.recv(1)
        if not chunk:
            raise ConnectionError("Control connection closed before a complete response")
        if chunk == b"\n":
            value = json.loads(data.decode("utf-8"))
            if not isinstance(value, dict):
                raise ValueError("Control frame must be a JSON object")
            return value
        data.extend(chunk)
    raise ValueError("Control frame exceeds 4096 bytes")


def send_frame(connection: socket.socket, response: dict) -> None:
    frame = json.dumps({"version": PROTOCOL_VERSION, **response}, separators=(",", ":"))
    connection.sendall(frame.encode("utf-8") + b"\n")


def validate_request(request: dict) -> tuple[str, float]:
    if type(request.get("version")) is not int or request["version"] != PROTOCOL_VERSION:
        raise ValueError("Unsupported control protocol version")
    command = request.get("command")
    if command in ("status", "switch_device", "cancel_switch"):
        return command, 0
    if command != "quiesce_stop":
        raise ValueError("Unknown control command")
    wait = request.get("wait_seconds", MAX_WAIT_SECONDS)
    if isinstance(wait, bool) or not isinstance(wait, (int, float)) or not math.isfinite(wait):
        raise ValueError("wait_seconds must be a finite number")
    if not 0 < wait <= MAX_WAIT_SECONDS:
        raise ValueError("wait_seconds must be greater than zero and at most 300")
    return command, float(wait)


def request_control(path: Path, request: dict) -> dict:
    _, wait = validate_request(request)
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
        connection.settimeout(wait + 5)
        connection.connect(str(path))
        send_frame(connection, request)
        response = read_frame(connection)
    if response.get("version") != PROTOCOL_VERSION:
        raise ValueError("Whisper returned an unsupported control protocol version")
    return response
