"""Private same-user socket for recording-safe GPU handoff."""

from __future__ import annotations

import fcntl
import os
import select
import socket
import stat
import struct
import threading
from pathlib import Path

from .control_protocol import (
    default_socket_path,
    read_frame,
    send_frame,
    validate_request,
)
from .control_state import HandoffBusyError, HandoffController


class ControlServer:
    def __init__(self, controller: HandoffController, path: Path | None = None) -> None:
        self._controller = controller
        self.path = path or default_socket_path()
        self._closed = threading.Event()
        self._listener: socket.socket | None = None
        self._lock_file = None
        self._inode: int | None = None
        self._accept_thread: threading.Thread | None = None
        self._clients: list[threading.Thread] = []

    def start(self) -> None:
        directory = self.path.parent
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        info = directory.lstat()
        if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
            raise PermissionError("Whisper control directory must be private and owned by this user")
        try:
            lock_fd = os.open(directory / "control.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
            self._lock_file = os.fdopen(lock_fd, "w")
            fcntl.flock(self._lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if self.path.exists() or self.path.is_symlink():
                prior = self.path.lstat()
                if not stat.S_ISSOCK(prior.st_mode) or prior.st_uid != os.getuid():
                    raise PermissionError("Refusing to replace an unexpected control path")
                self.path.unlink()
            self._listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            self._listener.bind(str(self.path))
            os.chmod(self.path, 0o600)
            self._inode = self.path.stat().st_ino
            self._listener.listen(4)
            self._listener.settimeout(0.2)
            self._accept_thread = threading.Thread(target=self._accept, name="whisper-control", daemon=True)
            self._accept_thread.start()
        except Exception:
            self.close()
            raise

    def close(self) -> None:
        self._closed.set()
        if self._listener is not None:
            self._listener.close()
        if self._accept_thread is not None:
            self._accept_thread.join(timeout=1)
        for client in self._clients:
            if client is not threading.current_thread():
                client.join(timeout=1)
        try:
            if self._inode is not None and self.path.stat().st_ino == self._inode:
                self.path.unlink()
        except FileNotFoundError:
            pass
        if self._lock_file is not None:
            self._lock_file.close()

    def _accept(self) -> None:
        while not self._closed.is_set():
            try:
                connection, _ = self._listener.accept()
            except TimeoutError:
                continue
            except OSError:
                return
            client = threading.Thread(target=self._handle, args=(connection,), daemon=True)
            self._clients = [thread for thread in self._clients if thread.is_alive()]
            self._clients.append(client)
            client.start()

    def _handle(self, connection: socket.socket) -> None:
        with connection:
            try:
                credentials = connection.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, struct.calcsize("3i"))
                _, peer_uid, _ = struct.unpack("3i", credentials)
                if peer_uid != os.getuid():
                    raise PermissionError("Control socket accepts only this user")
                connection.settimeout(5)
                command, wait = validate_request(read_frame(connection))
                if command == "status":
                    result = self._controller.status()
                elif command in ("switch_device", "cancel_switch"):
                    if self._controller.devices is None:
                        raise RuntimeError("Device switching is not configured")
                    action = (self._controller.devices.request_switch if command == "switch_device"
                              else self._controller.devices.cancel_switch)
                    result = action()
                else:
                    result = self._controller.quiesce_stop(wait, lambda: self._cancelled(connection))
                send_frame(connection, {"ok": True, "result": result})
            except (ConnectionError, BrokenPipeError):
                return
            except (OSError, RuntimeError, ValueError) as exc:
                if isinstance(exc, HandoffBusyError):
                    code = "busy"
                elif isinstance(exc, TimeoutError):
                    code = "timeout"
                elif isinstance(exc, InterruptedError):
                    code = "cancelled"
                else:
                    code = "unavailable" if isinstance(exc, RuntimeError) else "invalid_request"
                try:
                    send_frame(connection, {"ok": False, "error": {"code": code, "message": str(exc)}})
                except OSError:
                    return

    def _cancelled(self, connection: socket.socket) -> bool:
        if self._closed.is_set():
            return True
        readable, _, _ = select.select([connection], [], [], 0)
        if not readable:
            return False
        try:
            return connection.recv(1, socket.MSG_PEEK | socket.MSG_DONTWAIT) == b""
        except OSError:
            return True
