"""Best-effort system audio cues."""

from __future__ import annotations

import subprocess
from pathlib import Path

from .log_utils import write_log


SYSTEM_BELL_PLAYER = Path("/usr/bin/paplay")
SYSTEM_BELL_SOUND = Path("/usr/share/sounds/freedesktop/stereo/bell.oga")
SYSTEM_BELL_TIMEOUT_SECONDS = 1.0


def play_system_bell(log_path: Path, *, purpose: str) -> None:
    """Synchronously play the system bell without exposing failures to callers."""
    purpose_label = purpose.strip() or "unspecified purpose"

    try:
        if not SYSTEM_BELL_PLAYER.is_file():
            _write_log_safely(
                f"System bell for {purpose_label} skipped: player missing at {SYSTEM_BELL_PLAYER}",
                log_path,
            )
            return
        if not SYSTEM_BELL_SOUND.is_file():
            _write_log_safely(
                f"System bell for {purpose_label} skipped: sound missing at {SYSTEM_BELL_SOUND}",
                log_path,
            )
            return

        subprocess.run(
            [str(SYSTEM_BELL_PLAYER), str(SYSTEM_BELL_SOUND)],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            timeout=SYSTEM_BELL_TIMEOUT_SECONDS,
        )
    except subprocess.CalledProcessError as exc:
        stderr = _stderr_text(exc.stderr)
        stderr_detail = f"; stderr={stderr}" if stderr else ""
        _write_log_safely(
            f"System bell for {purpose_label} failed with exit code {exc.returncode}{stderr_detail}",
            log_path,
        )
    except subprocess.TimeoutExpired:
        _write_log_safely(
            f"System bell for {purpose_label} timed out after {SYSTEM_BELL_TIMEOUT_SECONDS:.1f}s",
            log_path,
        )
    except OSError as exc:
        _write_log_safely(f"System bell for {purpose_label} failed: {exc}", log_path)
    except Exception as exc:
        _write_log_safely(f"System bell for {purpose_label} failed unexpectedly: {exc}", log_path)


def _stderr_text(stderr: str | bytes | None) -> str:
    if isinstance(stderr, bytes):
        return stderr.decode("utf-8", "ignore").strip()
    return stderr.strip() if stderr else ""


def _write_log_safely(message: str, log_path: Path) -> None:
    try:
        write_log(message, log_path)
    except Exception:
        pass
