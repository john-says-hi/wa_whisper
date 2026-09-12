"""Shared foreground admission and a nonblocking gate for new dictation.

The background service must finish already accepted audio while the media
coordinator waits for its quiesce response. It therefore does not acquire a
foreground ticket. Standalone GPU utilities do hold one until process exit.
"""

from __future__ import annotations

import importlib
import json
import sys
import threading
from pathlib import Path

_process_lease = None
_lock = threading.Lock()


def _runtime():
    path = Path.home() / ".config/local_media/admission.json"
    if not path.exists():
        return None
    config = json.loads(path.read_text())
    if not isinstance(config, dict) or not isinstance(config.get("enabled", False), bool):
        raise RuntimeError("Invalid local-media admission configuration")
    if not config.get("enabled", False):
        return None
    root = Path(config["module_root"])
    if not root.is_absolute() or not (root / "local_media/runtime/admission.py").is_file():
        raise RuntimeError("Local-media admission runtime is unavailable")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return importlib.import_module("local_media.runtime.admission")


def enabled() -> bool:
    return _runtime() is not None


def new_capture_reason() -> str | None:
    """Return a denial reason without joining or waiting on the GPU queue."""
    try:
        runtime = _runtime()
        if runtime is None:
            return None
        lease = runtime.configured_lease(owner="wa_whisper", kind="interactive")
        if lease.queue is None:
            return None
        for row in lease.queue.snapshot():
            if row["state"] == "waiting" and not row["owner_alive"]:
                continue
            if row["state"] == "active" and not row["owner_alive"]:
                children_live = any(
                    runtime.process_identity(child["pid"]) == child["start"]
                    for child in row["children"]
                )
                if row["metadata"].get("recover_when_process_exits") and not children_live:
                    continue
            return f"while shared GPU work is {row['state']} ({row['owner']})"
        return None
    except Exception as exc:
        return f"because shared GPU admission is unavailable: {exc}"


def ensure_process_admission() -> None:
    """Protect direct WhisperBackend/smoke-test GPU use until process death."""
    global _process_lease
    with _lock:
        if _process_lease is not None:
            return
        runtime = _runtime()
        if runtime is None:
            return
        lease = runtime.configured_lease(owner="wa_whisper_utility", kind="interactive")
        print("Waiting for shared GPU admission.", file=sys.stderr, flush=True)
        try:
            lease.acquire(timeout=3600)
            lease.recover_when_process_exits()
        except BaseException:
            lease.close()
            raise
        # No atexit release: Torch may retain its CUDA context until OS exit.
        _process_lease = lease
