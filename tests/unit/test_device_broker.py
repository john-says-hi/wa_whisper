"""Shared model lifetime, routing, fairness, and retry tests without CUDA."""
import base64
import io
import threading
import time
import wave

import pytest

from wa_whisper.broker_state import BrokerState
from wa_whisper.model_process import BackendError


def audio():
    output = io.BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(b"\0\0" * 160)
    return base64.b64encode(output.getvalue()).decode()


def wait_until(predicate):
    deadline = time.monotonic() + 3
    while not predicate():
        assert time.monotonic() < deadline
        time.sleep(0.01)


class Model:
    def __init__(self):
        self.ready = False
        self.pid = None
        self.loads = 0
        self.calls = []
        self.gate = threading.Event()
        self.gate.set()
        self.failure = None

    def load(self, cancelled):
        self.loads += 1
        if self.failure:
            raise self.failure
        self.ready, self.pid = True, 123

    def transcribe(self, path, cancelled, options=None):
        self.calls.append(path)
        self.gate.wait(2)
        return {"text": "Hello", "segments": [], "info": {}}

    def close(self):
        self.ready, self.pid = False, None


@pytest.fixture
def broker(tmp_path):
    model = Model()
    state = BrokerState(model, tmp_path / "spool")
    yield state, model
    model.gate.set()
    state.close()


def rpc(state, action, session=None, **values):
    return state.request({"version": 1, "action": action, "session": session, **values})


def acquire(state, client):
    session = rpc(state, "acquire", client=client)["session"]
    wait_until(lambda: rpc(state, "status", session)["ready"])
    return session


def test_two_clients_share_model_and_release_independently(broker):
    state, model = broker
    desktop = acquire(state, "desktop")
    laptop = acquire(state, "laptop")
    assert model.loads == 1
    rpc(state, "release", laptop)
    assert rpc(state, "status", desktop)["ready"]
    rpc(state, "release", desktop)
    wait_until(lambda: not model.ready)


def test_duplicate_submit_only_transcribes_once_and_other_client_cannot_read(broker):
    state, model = broker
    desktop = acquire(state, "desktop")
    laptop = acquire(state, "laptop")
    for _ in range(2):
        rpc(state, "submit", desktop, job_id="record1", audio=audio())
    wait_until(lambda: rpc(state, "result", desktop, job_id="record1")["state"] == "completed")
    assert len(model.calls) == 1
    with pytest.raises(BackendError, match="not queued"):
        rpc(state, "result", laptop, job_id="record1")
    rpc(state, "acknowledge", desktop, job_id="record1")
    assert not list(state.spool.glob("*.wav"))


def test_queue_alternates_waiting_clients(broker):
    state, model = broker
    desktop = acquire(state, "desktop")
    laptop = acquire(state, "laptop")
    model.gate.clear()
    rpc(state, "submit", desktop, job_id="first", audio=audio())
    wait_until(lambda: len(model.calls) == 1)
    rpc(state, "submit", desktop, job_id="second", audio=audio())
    rpc(state, "submit", laptop, job_id="third", audio=audio())
    laptop_path = state.jobs[("laptop", "third")]["path"]
    model.gate.set()
    wait_until(lambda: len(model.calls) == 3)
    assert model.calls[1] == laptop_path


def test_lease_expiry_unloads_without_remote_release(broker):
    state, model = broker
    session = acquire(state, "desktop")
    with state._condition:
        state.sessions[session]["expires"] = 0
    wait_until(lambda: not model.ready)
    with pytest.raises(BackendError, match="expired"):
        rpc(state, "heartbeat", session)


def test_load_oom_is_reported_and_retry_can_load(broker):
    state, model = broker
    model.failure = BackendError("memory_full", "Laptop memory full")
    session = rpc(state, "acquire", client="desktop")["session"]
    wait_until(lambda: rpc(state, "status", session)["error"] is not None)
    assert rpc(state, "status", session)["error"]["code"] == "memory_full"
    model.failure = None
    acquire(state, "desktop")
    assert model.ready


def test_invalid_audio_and_unknown_sessions_rejected(broker):
    state, model = broker
    session = acquire(state, "desktop")
    with pytest.raises(ValueError):
        rpc(state, "submit", session, job_id="bad", audio="not audio")
    with pytest.raises(BackendError, match="expired"):
        rpc(state, "submit", "other", job_id="bad", audio=audio())
    assert not model.calls
