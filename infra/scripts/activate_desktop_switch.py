#!/usr/bin/env python3
"""Activate tested sources at a recording-safe desktop handoff, retaining rollback files."""
from __future__ import annotations

import fcntl
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from wa_whisper.control_protocol import default_socket_path, request_control  # noqa: E402

BACKUP = Path.home() / "Documents/wa_whisper_device_switch_deployment"
UNIT = "wa-whisper-ptt.service"


def save_original(path):
    manifest_path = BACKUP / "desktop_files.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    if str(path) in manifest:
        return
    stored = BACKUP / ("original_" + str(len(manifest)))
    if path.exists():
        shutil.copy2(path, stored)
        manifest[str(path)] = str(stored)
    else:
        manifest[str(path)] = None
    manifest_path.write_text(json.dumps(manifest, indent=2))


def main():
    BACKUP.mkdir(parents=True, exist_ok=True)
    runtime = Path(os.environ.get("XDG_RUNTIME_DIR", f"/run/user/{os.getuid()}"))
    with (runtime / "wa-whisper-power-toggle.lock").open("a") as power:
        fcntl.flock(power, fcntl.LOCK_EX | fcntl.LOCK_NB)
        active = subprocess.run(["systemctl", "--user", "is-active", "--quiet", UNIT]).returncode == 0
        if active:
            response = request_control(default_socket_path(), {"version": 1, "command": "quiesce_stop", "wait_seconds": 300})
            if not response.get("ok"):
                raise RuntimeError("Recording-safe shutdown refused: " + str(response))
            deadline = time.monotonic() + 30
            while subprocess.run(["systemctl", "--user", "is-active", "--quiet", UNIT]).returncode == 0:
                if time.monotonic() > deadline:
                    raise RuntimeError("Whisper did not stop; activation was cancelled")
                time.sleep(0.2)
        try:
            site = Path.home() / "Documents/wa_whisper/.venv/lib/python3.12/site-packages"
            pth = site / "zzzz_whisper_device_switch.pth"
            save_original(pth)
            pth.write_text("import sys; sys.path.insert(0, " + repr(str(ROOT / "src")) + ")\n")
            service = Path.home() / ".config/systemd/user/wa-whisper-ptt.service.d/90-device-switch.conf"
            save_original(service)
            service.parent.mkdir(parents=True, exist_ok=True)
            service.write_text("[Service]\nEnvironment=PYTHONPATH=" + str(ROOT / "src") + "\n")
            power_script = Path.home() / ".local/bin/wa-whisper-power-toggle"
            save_original(power_script)
            shutil.copy2(ROOT / "infra/scripts/wa-whisper-power-toggle", power_script)
            sounds = Path.home() / ".local/share/wa_whisper/power_phrases"
            sounds.mkdir(parents=True, exist_ok=True)
            for source in (ROOT / "assets/device_notices").glob("*.wav"):
                save_original(sounds / source.name)
                shutil.copy2(source, sounds / source.name)
            shortcuts = Path.home() / ".config/cosmic/com.system76.CosmicSettings.Shortcuts/v1/custom"
            save_original(shortcuts)
            contents = shortcuts.read_text()
            entry = '(modifiers: [Ctrl, Shift, Alt], key: "F1"): Spawn("/usr/bin/true"),'
            if entry not in contents:
                closing = contents.rfind('}')
                if closing < 0:
                    raise ValueError("Unexpected COSMIC shortcut format")
                shortcuts.write_text(contents[:closing] + '    ' + entry + '\n' + contents[closing:])
            subprocess.run(["systemctl", "--user", "daemon-reload"], check=True)
        finally:
            if active:
                subprocess.run(["systemctl", "--user", "start", UNIT], check=True)
    print("Desktop source activated. Rollback files: " + str(BACKUP))


if __name__ == "__main__":
    main()
