"""Type into protected focused apps only from the one hotkey mode each app permits."""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .focused_window import FocusUnavailable, read_focused_window
from .log_utils import write_log
from .recording_mode import RecordingMode

RESOLVE_FOCUSED = "resolve_focused"
RUNELITE_FOCUSED = "runelite_focused"
DETECTION_UNAVAILABLE = "focus_detection_unavailable"


@dataclass(frozen=True)
class ProtectedApp:
    block_reason: str
    display_name: str
    allowed_mode: RecordingMode
    app_id: str
    match_prefix: bool = False

    def owns(self, app_id: str) -> bool:
        focused = app_id.casefold()
        return focused.startswith(self.app_id) if self.match_prefix else focused == self.app_id


PROTECTED_APPS = (
    ProtectedApp(RESOLVE_FOCUSED, "DaVinci Resolve", RecordingMode.HANDS_FREE, "resolve"),
    # Java reports its main class as the app ID, which differs between the Jagex
    # launcher (net-runelite-launcher-Launcher) and a direct client start.
    ProtectedApp(RUNELITE_FOCUSED, "RuneLite", RecordingMode.PUSH_TO_TALK, "net-runelite-", match_prefix=True),
)
_APPS_BY_REASON = {app.block_reason: app for app in PROTECTED_APPS}


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
    app = next((app for app in PROTECTED_APPS if app.owns(focused.app_id)), None)
    return app.block_reason if app else None


def ensure_insertion_allowed(recording_mode: RecordingMode = RecordingMode.PUSH_TO_TALK) -> None:
    reason = insertion_block_reason()
    app = _APPS_BY_REASON.get(reason)
    if app and recording_mode is app.allowed_mode:
        return
    if reason:
        raise InsertionBlocked(reason)


def notify_saved_transcript(reason: str, destination: str, log_path: Path) -> None:
    app = _APPS_BY_REASON.get(reason)
    explanation = (
        f"{app.display_name} has keyboard focus."
        if app
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
