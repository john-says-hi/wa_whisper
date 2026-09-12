"""Control dictation without loading Torch or the transcription model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .control_protocol import default_socket_path, request_control


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inspect Whisper or finish dictation before stopping it.")
    parser.add_argument("command", choices=["status", "quiesce-stop", "switch-device", "cancel-switch"])
    parser.add_argument("--wait-seconds", type=float, default=300)
    parser.add_argument("--socket", type=Path, default=default_socket_path())
    args = parser.parse_args(argv)
    try:
        response = request_control(args.socket, {
            "version": 1,
            "command": args.command.replace("-", "_"),
            "wait_seconds": args.wait_seconds,
        })
        print(json.dumps(response))
        return 0 if response.get("ok") else 1
    except (OSError, ValueError) as exc:
        print(f"wa-whisper-control: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
