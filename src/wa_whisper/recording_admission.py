"""Admit a microphone recording only when its place in the FIFO is available."""
from __future__ import annotations

MAX_WAITING_RECORDINGS = 7


class RecordingAdmission:
    def __init__(self, tasks, backend):
        self.tasks = tasks
        self.backend = backend

    def __call__(self):
        # The hotkey lifecycle has one producer and finishes its enqueue before
        # admitting another capture; only the consumer can free a slot meanwhile.
        if self.tasks.waiting_recordings() >= MAX_WAITING_RECORDINGS:
            self.backend.notices.say("recording_queue_full")
            return "because the recording queue is full"
        return self.backend.capture_admission_reason()
