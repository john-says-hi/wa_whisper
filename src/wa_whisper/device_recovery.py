"""Durable retry markers in the original desktop recording directories."""
from __future__ import annotations

import json
import threading

from .device_config import atomic_json
from .dictation_archive import DictationRecord
from .log_utils import write_log
from .model_process import BackendError
from .recovery_queue import insert_transcript_into_recovery_queue
from .text_postprocess import postprocess_text

MARKER = "device_recovery.json"


class DeviceRecovery:
    def __init__(self, backend, archive):
        self.backend, self.archive = backend, archive
        self._active = set()
        self._outage_announced = False
        self._lock = threading.Lock()
        self._closed = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True, name="whisper-recording-recovery")

    def start(self):
        self._thread.start()

    def begin(self, record, options):
        with self._lock:
            self._active.add(record.record_id)
            atomic_json(record.record_dir / MARKER, {"version": 1, "options": options})

    def finish(self, record, succeeded):
        with self._lock:
            if succeeded:
                (record.record_dir / MARKER).unlink(missing_ok=True)
            self._active.discard(record.record_id)

    def _idle(self):
        state = self.backend.capture.capture_state()
        worker = self.backend.worker.status()
        return not (state["recording"] or state["finalizing"] or worker["processing"]
                    or worker["queued_items"] or self.backend.status()["switching"])

    def _recover(self, marker):
        root = marker.parent
        record = DictationRecord(root.name, root, root / "audio.wav", root / "transcript.txt", root / "metadata.json")
        request = json.loads(marker.read_text(encoding="utf-8"))
        if record.transcript_path.read_text(encoding="utf-8").strip():
            # A crash after persistence must not retranscribe or type the recording again.
            text = record.transcript_path.read_text(encoding="utf-8")
        else:
            result = self.backend.transcribe(record.audio_path, recovery=True)
            text = postprocess_text(result.text, **request["options"])
            self.archive.save_transcript(record, text, whisper_info=result.info)
        if text.strip():
            # Claim history delivery before the external call so a crash cannot duplicate it.
            history = None
            if not request.get("history_attempted"):
                request["history_attempted"] = True
                atomic_json(marker, request)
                history = insert_transcript_into_recovery_queue(text, self.backend.log_path)
            self.archive.update_record(record, status="recovered_to_history", recovery_queue=history.recovery_queue_metadata() if history else {"delivery_previously_attempted": True},
                                       injection={"succeeded": False, "reason": "Recovered recording is never auto-injected"})
        else:
            self.archive.update_record(record, status="no_text")
        self.backend.acknowledge(record.audio_path)
        marker.unlink(missing_ok=True)
        self.backend.notices.say("recovered")

    def _run(self):
        while not self._closed.wait(5):
            if not self._idle():
                continue
            # Reconnect a selected laptop without ever constructing a local CUDA model.
            if self.backend.destination == "laptop" and not self.backend.status()["ready"]:
                try:
                    remote = self.backend._remote()
                    remote.load(lambda: self._closed.is_set() or self.backend._switching)
                    self._outage_announced = False
                    self.backend.notices.say("laptop_online")
                except (OSError, ValueError, RuntimeError) as exc:
                    if not self._outage_announced and not self.backend._switching:
                        name = "laptop_memory_full" if isinstance(exc, BackendError) and exc.code == "memory_full" else "laptop_offline"
                        self.backend.notices.say(name, str(exc))
                        self._outage_announced = True
                    continue
            for marker in sorted(self.archive.root.glob("*/*/" + MARKER)):
                if self._closed.is_set() or not self._idle():
                    break
                with self._lock:
                    if marker.parent.name in self._active:
                        continue
                    self._active.add(marker.parent.name)
                try:
                    self._recover(marker)
                except (OSError, ValueError, RuntimeError) as exc:
                    write_log(f"Recording recovery postponed for {marker.parent.name}: {exc}", self.backend.log_path)
                    break
                finally:
                    with self._lock:
                        self._active.discard(marker.parent.name)

    def close(self):
        self._closed.set()
        self._thread.join(timeout=6)
