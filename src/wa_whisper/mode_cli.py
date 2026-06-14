"""CLI for switching the persisted wa_whisper compute mode."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from .compute_mode import (
    ComputeMode,
    ComputeModeError,
    compute_mode_values,
    read_compute_mode,
    write_compute_mode,
)

DEFAULT_SERVICE_NAME = "wa-whisper-ptt.service"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Switch wa_whisper between GPU and system-RAM compute modes.")
    parser.add_argument(
        "action",
        choices=[*compute_mode_values(), "status"],
        help="Mode to activate or status to read.",
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=None,
        help="Override config directory. Defaults to XDG_CONFIG_HOME/wa_whisper or ~/.config/wa_whisper.",
    )
    parser.add_argument(
        "--service-name",
        default=DEFAULT_SERVICE_NAME,
        help="User systemd service to restart after switching modes.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    try:
        if args.action != "status":
            mode = ComputeMode.parse(args.action)
            path = write_compute_mode(mode, config_dir=args.config_dir)
            restart_service(args.service_name)
            service_state = read_service_state(args.service_name)
            print(f"wa_whisper compute mode: {mode.value}")
            print(f"config: {path}")
            print(f"service: {service_state}")
            return 0

        mode = read_compute_mode(config_dir=args.config_dir)
        service_state = read_service_state(args.service_name)
        print(f"wa_whisper compute mode: {mode.value}")
        print(f"service: {service_state}")
        return 0
    except ComputeModeError as exc:
        print(f"wa-whisper-mode: {exc}", file=sys.stderr)
        return 2
    except subprocess.CalledProcessError as exc:
        print(f"wa-whisper-mode: systemctl failed with exit code {exc.returncode}", file=sys.stderr)
        return exc.returncode or 1


def restart_service(service_name: str) -> None:
    subprocess.run(["systemctl", "--user", "restart", service_name], check=True)


def read_service_state(service_name: str) -> str:
    completed = subprocess.run(
        ["systemctl", "--user", "is-active", service_name],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    state = completed.stdout.strip()
    if state:
        return state
    if completed.returncode == 0:
        return "active"
    return f"unknown (systemctl exit {completed.returncode})"


if __name__ == "__main__":
    raise SystemExit(main())
