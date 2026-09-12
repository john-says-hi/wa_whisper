"""Persistent destination and private broker configuration, without CUDA imports."""
from __future__ import annotations

import json
import os
import uuid
from pathlib import Path

from .compute_mode import default_config_dir


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".pending")
    try:
        temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def read_destination() -> str:
    path = default_config_dir() / "destination.json"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("version") == 1 and value.get("destination") in ("desktop", "laptop"):
            return value["destination"]
    except (OSError, ValueError, AttributeError):
        pass
    return "desktop"


def save_destination(destination: str) -> None:
    if destination not in ("desktop", "laptop"):
        raise ValueError("Unknown dictation destination")
    atomic_json(default_config_dir() / "destination.json", {"version": 1, "destination": destination})


def broker_settings() -> dict:
    settings = {"host": "windows-laptop", "port": 47631, "local_port": 47632,
                "env_file": str(Path.home() / "Documents/wa_whisper/.env")}
    path = default_config_dir() / "broker.json"
    if path.exists():
        settings.update(json.loads(path.read_text(encoding="utf-8")))
    for name in ("port", "local_port"):
        if type(settings[name]) is not int or not 1024 <= settings[name] <= 65535:
            raise ValueError("Invalid broker port")
    return settings


def broker_token(settings: dict) -> str:
    for line in Path(settings["env_file"]).read_text(encoding="utf-8").splitlines():
        key, separator, value = line.partition("=")
        if separator and key.strip() == "WA_WHISPER_BROKER_TOKEN":
            token = value.strip().strip("\"'")
            if len(token) >= 32:
                return token
    raise RuntimeError("Whisper broker authentication is not configured")
