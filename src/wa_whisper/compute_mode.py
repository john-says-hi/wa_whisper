"""Persisted compute-mode selection for wa_whisper."""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

CONFIG_DIR_NAME = "wa_whisper"
MODE_FILE_NAME = "compute_mode"


class ComputeModeError(ValueError):
    """Raised when compute-mode configuration cannot be validated."""


class ComputeMode(str, Enum):
    GPU = "gpu"
    RAM = "ram"

    @classmethod
    def parse(cls, value: str) -> "ComputeMode":
        cleaned = value.strip()
        try:
            return cls(cleaned)
        except ValueError as exc:
            expected = ", ".join(mode.value for mode in cls)
            raise ComputeModeError(f"Unsupported compute mode {value!r}. Expected one of: {expected}.") from exc


@dataclass(frozen=True)
class ComputeDeviceSettings:
    compute_mode: ComputeMode | None
    device: str
    fp16: bool


def default_config_dir() -> Path:
    xdg_config_home = os.environ.get("XDG_CONFIG_HOME")
    if xdg_config_home:
        return Path(xdg_config_home) / CONFIG_DIR_NAME
    return Path.home() / ".config" / CONFIG_DIR_NAME


def compute_mode_file(*, config_dir: Path | None = None) -> Path:
    return (config_dir or default_config_dir()) / MODE_FILE_NAME


def read_compute_mode(*, config_dir: Path | None = None) -> ComputeMode:
    path = compute_mode_file(config_dir=config_dir)
    try:
        raw_value = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return ComputeMode.GPU

    try:
        return ComputeMode.parse(raw_value)
    except ComputeModeError as exc:
        raise ComputeModeError(
            f"Invalid compute mode in {path}: {raw_value.strip()!r}. Expected one of: gpu, ram.",
        ) from exc


def write_compute_mode(mode: ComputeMode | str, *, config_dir: Path | None = None) -> Path:
    parsed_mode = ensure_compute_mode(mode)
    path = compute_mode_file(config_dir=config_dir)
    path.parent.mkdir(parents=True, exist_ok=True)

    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            delete=False,
        ) as temp_file:
            temp_file.write(f"{parsed_mode.value}\n")
            temp_path = Path(temp_file.name)
        os.replace(temp_path, path)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()
    return path


def ensure_compute_mode(mode: ComputeMode | str) -> ComputeMode:
    if isinstance(mode, ComputeMode):
        return mode
    return ComputeMode.parse(mode)


def device_settings_for_mode(mode: ComputeMode | str) -> ComputeDeviceSettings:
    parsed_mode = ensure_compute_mode(mode)
    if parsed_mode is ComputeMode.GPU:
        return ComputeDeviceSettings(compute_mode=parsed_mode, device="cuda", fp16=True)
    return ComputeDeviceSettings(compute_mode=parsed_mode, device="cpu", fp16=False)


def resolve_compute_device(
    *,
    compute_mode_override: str | None = None,
    device_override: str | None = None,
    config_dir: Path | None = None,
) -> ComputeDeviceSettings:
    if compute_mode_override and device_override:
        raise ComputeModeError("--device cannot be used with --compute-mode.")

    if device_override:
        device = device_override.strip()
        return ComputeDeviceSettings(compute_mode=None, device=device, fp16=device != "cpu")

    if compute_mode_override:
        mode = ComputeMode.parse(compute_mode_override)
    else:
        mode = read_compute_mode(config_dir=config_dir)
    return device_settings_for_mode(mode)


def compute_mode_values() -> list[str]:
    return [mode.value for mode in ComputeMode]
