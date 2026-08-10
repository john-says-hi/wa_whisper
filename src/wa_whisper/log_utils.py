"""Logging helpers for wa_whisper."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

DEFAULT_LOG_PATH = Path.home() / ".cache" / "wa_whisper" / "push_to_talk.log"


def ensure_log_path(log_path: Path) -> Path:
    """Ensure the log directory exists and return the usable path."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if not log_path.exists():
        log_path.touch()
    return log_path


def write_log(message: str, log_path: Path = DEFAULT_LOG_PATH) -> None:
    """Best-effort append ``message`` without affecting runtime control flow.

    Ordinary filesystem and formatting failures are intentionally ignored.
    Process-control exceptions such as ``KeyboardInterrupt`` and ``SystemExit``
    remain visible because they do not inherit from ``Exception``.
    """
    try:
        timestamp = datetime.utcnow().isoformat(timespec="seconds") + "Z"
        path = ensure_log_path(log_path)
        with path.open("a", encoding="utf-8", errors="ignore") as fp:
            fp.write(f"{timestamp} {message}\n")
    except Exception:
        return
