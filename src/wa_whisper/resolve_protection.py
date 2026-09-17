"""Prevent desktop dictation from sending shortcuts while Resolve is running."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

from .log_utils import write_log

RESOLVE_EXECUTABLE = "/opt/resolve/bin/resolve"
RESOLVE_RUNNING = "resolve_running"
DETECTION_UNAVAILABLE = "resolve_detection_unavailable"


class InsertionBlocked(RuntimeError):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def insertion_block_reason(proc_root: Path = Path("/proc")) -> str | None:
    """Inspect executable identities; Resolve names its main thread 'GUI Thread'."""
    uncertain = False
    try:
        user_id = os.getuid()
        for process in proc_root.iterdir():
            if not process.name.isdecimal():
                continue
            try:
                if process.stat().st_uid != user_id:
                    continue
                is_resolve = process_is_resolve(process)
            except FileNotFoundError:
                continue
            except OSError:
                uncertain = True
                continue
            if is_resolve:
                return RESOLVE_RUNNING
    except OSError:
        return DETECTION_UNAVAILABLE
    return DETECTION_UNAVAILABLE if uncertain else None


def process_is_resolve(process: Path) -> bool:
    try:
        executable = str((process / "exe").readlink()).removesuffix(" (deleted)")
        return executable == RESOLVE_EXECUTABLE
    except PermissionError:
        # Some same-user daemons deny ptrace access but expose their launch command.
        command = (process / "cmdline").read_bytes().split(b"\0", 1)[0]
        if not command:
            raise PermissionError(f"Cannot identify process {process.name}")
        return Path(os.fsdecode(command)).name == "resolve"


def ensure_insertion_allowed() -> None:
    reason = insertion_block_reason()
    if reason:
        raise InsertionBlocked(reason)


def notify_saved_transcript(reason: str, destination: str, log_path: Path) -> None:
    explanation = (
        "DaVinci Resolve is open."
        if reason == RESOLVE_RUNNING
        else "DaVinci Resolve detection is unavailable."
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
