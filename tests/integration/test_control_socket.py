import os
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from wa_whisper.control_protocol import read_frame, request_control, send_frame
from wa_whisper.control_server import ControlServer
from wa_whisper.control_state import HandoffController
from wa_whisper.processing_queue import CaptureQueue, CaptureWorker


class Capture:
    def __init__(self):
        self.token = None
        self.idle = threading.Event()
        self.idle.set()
        self.acquired = threading.Event()
        self.released = threading.Event()

    def begin_quiesce(self, token):
        self.token = token
        self.acquired.set()

    def wait_capture_idle(self, token, deadline, cancelled):
        while not self.idle.is_set():
            if cancelled():
                raise InterruptedError("cancelled")
            if time.monotonic() >= deadline:
                raise TimeoutError("recording still active")
            self.idle.wait(0.01)

    def cancel_quiesce(self, token):
        if self.token == token:
            self.token = None
            self.released.set()

    def capture_state(self):
        return {"recording": not self.idle.is_set(), "quiescing": self.token is not None}


@pytest.fixture
def socket_path(tmp_path_factory):
    return tmp_path_factory.mktemp("s") / "control.sock"


@pytest.fixture
def running_control(tmp_path, socket_path):
    capture = Capture()
    tasks = CaptureQueue()
    worker = CaptureWorker(tasks, lambda _: None, tmp_path / "worker.log")
    worker.start()
    stops = []
    server = ControlServer(HandoffController(capture, worker, lambda: stops.append("stop")), socket_path)
    server.start()
    try:
        yield server, capture, stops
    finally:
        server.close()
        tasks.put(None)
        worker.join(1)


def test_private_status_and_successful_shutdown(running_control):
    server, capture, stops = running_control
    assert server.path.stat().st_mode & 0o777 == 0o600
    assert server.path.parent.stat().st_mode & 0o777 == 0o700
    status = request_control(server.path, {"version": 1, "command": "status"})
    assert status["ok"]
    assert status["result"]["pid"] == os.getpid()
    result = request_control(server.path, {"version": 1, "command": "quiesce_stop", "wait_seconds": 1})
    assert result["ok"]
    assert result["result"]["state"] == "stopping"
    assert stops == ["stop"]
    assert capture.token is not None


def test_active_capture_can_finish_and_parallel_handoff_is_busy(running_control):
    server, capture, stops = running_control
    capture.idle.clear()
    responses = []
    waiter = threading.Thread(target=lambda: responses.append(request_control(
        server.path, {"version": 1, "command": "quiesce_stop", "wait_seconds": 2})))
    waiter.start()
    try:
        assert capture.acquired.wait(1)
        busy = request_control(server.path, {"version": 1, "command": "quiesce_stop", "wait_seconds": 1})
        assert busy["error"]["code"] == "busy"
        assert stops == []
        capture.idle.set()
        waiter.join(1)
        assert responses[0]["ok"]
        assert stops == ["stop"]
    finally:
        capture.idle.set()
        waiter.join(2)


def test_disconnected_request_reopens_gate_without_stopping(running_control):
    server, capture, stops = running_control
    capture.idle.clear()
    connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    connection.connect(str(server.path))
    send_frame(connection, {"version": 1, "command": "quiesce_stop", "wait_seconds": 2})
    assert capture.acquired.wait(1)
    connection.close()
    assert capture.released.wait(1)
    assert capture.token is None
    assert stops == []


def test_timeout_reopens_gate_without_interrupting_capture(running_control):
    server, capture, stops = running_control
    capture.idle.clear()
    result = request_control(server.path, {"version": 1, "command": "quiesce_stop", "wait_seconds": 0.03})
    assert result["error"]["code"] == "timeout"
    assert capture.token is None
    assert not capture.idle.is_set()
    assert stops == []


def test_invalid_protocol_does_not_take_a_lease(running_control):
    server, capture, stops = running_control
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
        connection.connect(str(server.path))
        connection.sendall(b'{"version":99,"command":"quiesce_stop"}\n')
        result = read_frame(connection)
    assert result["error"]["code"] == "invalid_request"
    assert capture.token is None
    assert stops == []


def test_socket_owner_is_not_replaced_by_a_second_server(running_control):
    server, _, _ = running_control
    other = ControlServer(None, server.path)
    with pytest.raises(BlockingIOError):
        other.start()
    assert request_control(server.path, {"version": 1, "command": "status"})["ok"]


def test_nonprivate_directory_is_rejected(tmp_path):
    folder = tmp_path / "public"
    folder.mkdir(mode=0o755)
    with pytest.raises(PermissionError):
        ControlServer(None, folder / "control.sock").start()


def test_different_peer_uid_cannot_take_a_lease(running_control, monkeypatch):
    server, capture, stops = running_control
    actual_uid = os.getuid()
    monkeypatch.setattr("wa_whisper.control_server.os.getuid", lambda: actual_uid + 1)
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
        connection.connect(str(server.path))
        result = read_frame(connection)
    assert not result["ok"]
    assert capture.token is None
    assert stops == []


def test_failed_shutdown_callback_releases_admission(tmp_path):
    capture = Capture()
    tasks = CaptureQueue()
    worker = CaptureWorker(tasks, lambda _: None, tmp_path / "worker.log")
    worker.start()

    def fail_shutdown():
        raise RuntimeError("synthetic shutdown failure")

    controller = HandoffController(capture, worker, fail_shutdown)
    try:
        with pytest.raises(RuntimeError, match="synthetic"):
            controller.quiesce_stop(1, lambda: False)
        assert capture.token is None
        assert not controller.status()["stopping"]
    finally:
        tasks.put(None)
        worker.join(1)


def test_control_cli_import_does_not_import_torch():
    source = str(Path(__file__).resolve().parents[2] / "src")
    result = subprocess.run([sys.executable, "-c",
        "import wa_whisper.control_cli,sys; print('torch' in sys.modules)"],
        env={**os.environ, "PYTHONPATH": source}, capture_output=True, text=True, check=True)
    assert result.stdout.strip() == "False"


def test_owned_cpu_process_exits_after_cooperative_control(socket_path):
    source = str(Path(__file__).resolve().parents[2] / "src")
    path = socket_path
    script = '''
import sys,threading
from pathlib import Path
from wa_whisper.control_server import ControlServer
from wa_whisper.control_state import HandoffController
from wa_whisper.processing_queue import CaptureQueue,CaptureWorker
class Capture:
 def begin_quiesce(self, token): pass
 def wait_capture_idle(self, token, deadline, cancelled): pass
 def cancel_quiesce(self, token): pass
 def capture_state(self): return {"recording":False}
done=threading.Event()
tasks=CaptureQueue()
worker=CaptureWorker(tasks,lambda _:None,Path(sys.argv[1]).parent/"worker.log")
worker.start()
server=ControlServer(HandoffController(Capture(),worker,done.set),Path(sys.argv[1]))
server.start()
try:
 if not done.wait(8): raise RuntimeError("No handoff received")
finally:
 tasks.put(None)
 worker.join(1)
 server.close()
'''
    child = subprocess.Popen([sys.executable, "-c", script, str(path)],
        env={**os.environ, "PYTHONPATH": source}, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        deadline = time.monotonic() + 3
        while not path.exists() and child.poll() is None and time.monotonic() < deadline:
            time.sleep(0.02)
        result = request_control(path, {"version": 1, "command": "quiesce_stop", "wait_seconds": 2})
        assert result["result"]["pid"] == child.pid
        output, error = child.communicate(timeout=3)
        assert child.returncode == 0, (output, error)
        assert not path.exists()
    finally:
        if child.poll() is None:
            child.terminate()
            child.communicate(timeout=3)
