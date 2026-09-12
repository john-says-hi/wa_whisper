"""Idempotent control from SSH or a second Desktop shortcut launch."""

import argparse
import os
import uuid
from pathlib import Path

STATE_DIR = Path.home() / ".cache" / "wa_whisper"


def request_power(action: str) -> None:
    if action not in ("start", "stop"):
        raise ValueError("Power action must be start or stop")
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    temporary = STATE_DIR / f"power_request_{uuid.uuid4().hex}.tmp"
    try:
        temporary.write_text(action, encoding="utf-8")
        os.replace(temporary, STATE_DIR / "windows_power_request.txt")
    finally:
        temporary.unlink(missing_ok=True)


def take_request() -> str | None:
    path = STATE_DIR / "windows_power_request.txt"
    try:
        action = path.read_text(encoding="utf-8").strip()
        path.unlink()
        return action if action in ("start", "stop") else None
    except FileNotFoundError:
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("start", "stop", "status"))
    action = parser.parse_args().action
    if action == "status":
        print((STATE_DIR / "windows_status.txt").read_text(encoding="utf-8"))
    else:
        request_power(action)
        print(f"Requested dictation {action}")


if __name__ == "__main__":
    main()
