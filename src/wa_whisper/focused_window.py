"""Read the keyboard-focused COSMIC window through the local Wayland helper."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

DEFAULT_HELPER = Path.home() / ".local/share/wa_whisper/bin/wa_toplevels"
FOCUS_QUERY_TIMEOUT_SECONDS = 1.0


class FocusUnavailable(RuntimeError):
    pass


@dataclass(frozen=True)
class FocusedWindow:
    window_id: str
    app_id: str


def read_focused_window(helper: Path = DEFAULT_HELPER) -> FocusedWindow:
    try:
        result = subprocess.run(
            [str(helper), "list"], check=True, capture_output=True, text=True,
            timeout=FOCUS_QUERY_TIMEOUT_SECONDS,
        )
        return focused_window_from_snapshot(json.loads(result.stdout))
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        raise FocusUnavailable("Could not identify the keyboard-focused application") from exc


def focused_window_from_snapshot(snapshot: object) -> FocusedWindow:
    if not isinstance(snapshot, dict) or not isinstance(snapshot.get("toplevels"), list):
        raise FocusUnavailable("Window inventory is unavailable")
    windows = snapshot["toplevels"]
    if any(not isinstance(window, dict) for window in windows):
        raise FocusUnavailable("Window inventory is malformed")
    focused = [window for window in windows if window.get("activated") is True]
    if len(focused) != 1:
        raise FocusUnavailable("Keyboard focus is missing or ambiguous")
    window = focused[0]
    window_id, app_id = window.get("id"), window.get("app_id")
    if (window.get("state_known") is not True or not isinstance(window_id, str) or not window_id
            or not isinstance(app_id, str) or not app_id.strip()):
        raise FocusUnavailable("The focused application identity is unavailable")
    return FocusedWindow(window_id=window_id, app_id=app_id)
