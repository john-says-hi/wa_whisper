"""Best-effort recovery queue for completed dictation transcripts."""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from .log_utils import write_log

COPYQ_HISTORY_ROW = 1
COPYQ_TIMEOUT_SECONDS = 1.0
RECOVERY_QUEUE_PROVIDER = "copyq"


@dataclass(frozen=True, slots=True)
class RecoveryQueueResult:
    """Outcome from adding a transcript to the manual recovery queue."""

    provider: str
    inserted: bool
    row: int
    active_clipboard_modified: bool = False
    error: Optional[str] = None

    def clipboard_metadata(self) -> dict[str, bool]:
        return {"active_clipboard_modified": self.active_clipboard_modified}

    def recovery_queue_metadata(self) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            "provider": self.provider,
            "inserted": self.inserted,
            "row": self.row,
        }
        if self.error:
            metadata["error"] = self.error
        return metadata


def insert_transcript_into_recovery_queue(
    text: str,
    log_path: Path,
    *,
    copyq_bin: Optional[Path] = None,
) -> RecoveryQueueResult:
    """Insert a transcript into CopyQ history without replacing the active clipboard."""
    resolved_copyq_bin = copyq_bin or resolve_copyq_path()
    if resolved_copyq_bin is None:
        return failed_result("copyq not found", log_path)

    try:
        subprocess.run(
            [str(resolved_copyq_bin), "insert", str(COPYQ_HISTORY_ROW), "--", text],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            timeout=COPYQ_TIMEOUT_SECONDS,
        )
    except subprocess.CalledProcessError as exc:
        return failed_result(process_error_message(exc), log_path)
    except subprocess.TimeoutExpired:
        return failed_result(f"copyq insert timed out after {COPYQ_TIMEOUT_SECONDS:.1f}s", log_path)
    except OSError as exc:
        return failed_result(str(exc), log_path)

    write_log(f"Transcript inserted into CopyQ history row {COPYQ_HISTORY_ROW}", log_path)
    return RecoveryQueueResult(
        provider=RECOVERY_QUEUE_PROVIDER,
        inserted=True,
        row=COPYQ_HISTORY_ROW,
    )


def skipped_recovery_queue_result() -> RecoveryQueueResult:
    return RecoveryQueueResult(
        provider=RECOVERY_QUEUE_PROVIDER,
        inserted=False,
        row=COPYQ_HISTORY_ROW,
    )


def resolve_copyq_path() -> Optional[Path]:
    resolved = shutil.which("copyq")
    return Path(resolved) if resolved else None


def failed_result(error: str, log_path: Path) -> RecoveryQueueResult:
    write_log(f"Transcript CopyQ recovery insert failed: {error}", log_path)
    return RecoveryQueueResult(
        provider=RECOVERY_QUEUE_PROVIDER,
        inserted=False,
        row=COPYQ_HISTORY_ROW,
        error=error,
    )


def process_error_message(exc: subprocess.CalledProcessError) -> str:
    stderr = (exc.stderr or "").strip()
    if stderr:
        return f"copyq insert failed with exit code {exc.returncode}: {stderr}"
    return f"copyq insert failed with exit code {exc.returncode}"
