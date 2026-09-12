"""Cached device announcements that never allocate a speech model."""
from __future__ import annotations

import queue
import subprocess
import threading
import time
from pathlib import Path

from .log_utils import write_log

PHRASES = {
    "recording_queue_full": "Recording queue full",
    "transferring_voice": "Transferring voice",
    "voice_ready": "Voice ready",
    "desktop_online": "Desktop voice online",
    "laptop_online": "Laptop voice online",
    "laptop_offline": "Laptop offline. Recordings will be saved",
    "laptop_memory_full": "Laptop memory full",
    "desktop_memory_full": "Desktop memory full",
    "desktop_busy": "Desktop GPU busy",
    "switch_failed": "Voice switch failed",
    "recovered": "Recording recovered to history",
}


class DeviceNotices:
    def __init__(self, log_path):
        self.log_path = log_path
        self.capture = None
        self._queue = queue.Queue()
        self._last = {}
        self._closed = threading.Event()
        threading.Thread(target=self._run, daemon=True, name="whisper-device-notices").start()

    def say(self, name, detail=""):
        now = time.monotonic()
        if name not in ("transferring_voice", "voice_ready") and now - self._last.get(name, -100) < 15:
            return
        self._last[name] = now
        self._queue.put((name, detail))

    def _run(self):
        while not self._closed.is_set():
            try:
                name, detail = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue
            while self.capture and not self._closed.is_set():
                state = self.capture.capture_state()
                if not state["recording"] and not state["finalizing"]:
                    break
                self._closed.wait(0.1)
            if self._closed.is_set():
                return
            phrase = PHRASES[name]
            write_log(f"{phrase}: {detail}", self.log_path)
            try:
                subprocess.run(["notify-send", "-a", "Voice to text", phrase, detail], timeout=3, check=False)
            except (OSError, subprocess.SubprocessError) as exc:
                write_log(f"Device notification unavailable: {exc}", self.log_path)
            try:
                sound = Path.home() / ".local/share/wa_whisper/power_phrases" / (name + ".wav")
                if sound.exists():
                    subprocess.run(["paplay", str(sound)], timeout=8, check=False)
            except (OSError, subprocess.SubprocessError):
                pass

    def close(self):
        self._closed.set()
