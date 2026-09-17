"""Prevent desktop dictation from sending shortcuts into focused Resolve windows."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from .focused_window import FocusUnavailable, read_focused_window
from .log_utils import write_log

RESOLVE_FOCUSED = "resolve_focused"
DETECTION_UNAVAILABLE = "focus_detection_unavailable"


class InsertionBlocked(RuntimeError):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def insertion_block_reason() -> str | None:
    """Only the current keyboard destination decides whether delivery is allowed."""
    try:
        focused = read_focused_window()
    except FocusUnavailable:
        return DETECTION_UNAVAILABLE
    return RESOLVE_FOCUSED if focused.app_id.casefold() == "resolve" else None


def ensure_insertion_allowed() -> None:
    reason = insertion_block_reason()
    if reason:
        raise InsertionBlocked(reason)


def notify_saved_transcript(reason: str, destination: str, log_path: Path) -> None:
    explanation = (
        "DaVinci Resolve has keyboard focus."
        if reason == RESOLVE_FOCUSED
        else "The keyboard-focused application could not be identified."
    )
    body = f"{explanation} Automatic typing is paused. Retrieve your text from {destination}."
    write_log(f"Dictation saved without insertion: {reason}; retrieve from {destination}", log_path)
    gdbus = shutil.which("gdbus")
    if gdbus is None:
        write_log("Dictation notification unavailable: gdbus not found", log_path)
        return
    try:
        subprocess.run(
            [gdbus, "call", "--session", "--dest", "org.freedesktop.Notifications",
             "--object-path", "/org/freedesktop/Notifications",
             "--method", "org.freedesktop.Notifications.Notify",
             "Voice to text", "0", "", "Dictation saved", body, "[]", "{}", "8000"],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            timeout=2.0,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        write_log(f"Dictation notification failed: {exc}", log_path)
