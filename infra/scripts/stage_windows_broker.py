#!/usr/bin/env python3
"""Stage the broker source and token without enabling the laptop microphone."""
from __future__ import annotations

import io
import json
import os
import secrets
import subprocess
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
APP = Path.home() / "Documents/wa_whisper"
BACKUP = Path.home() / "Documents/wa_whisper_device_switch_deployment"
SSH = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5", "windows-laptop"]


def main():
    BACKUP.mkdir(parents=True, exist_ok=True)
    env = APP / ".env"
    contents = env.read_text() if env.exists() else ""
    token = None
    for line in contents.splitlines():
        if line.startswith("WA_WHISPER_BROKER_TOKEN="):
            token = line.partition("=")[2].strip().strip("\"'")
    if token is None:
        token = secrets.token_hex(32)
        with env.open("a") as output:
            output.write(("\n" if contents and not contents.endswith("\n") else "")
                         + "WA_WHISPER_BROKER_TOKEN=" + token + "\n")
        os.chmod(env, 0o600)
    if len(token) < 32:
        raise RuntimeError("Existing broker token is invalid")
    bundle = io.BytesIO()
    with tarfile.open(fileobj=bundle, mode="w:gz") as archive:
        for path in (ROOT / "src/wa_whisper").glob("*.py"):
            archive.add(path, arcname="src/wa_whisper/" + path.name)
        archive.add(ROOT / "infra/scripts/install_windows_broker.ps1", arcname="infra/scripts/install_windows_broker.ps1")
    # Only package-owned paths enter this tar; no environment files or recordings.
    import base64
    payload = json.dumps({"token": token, "bundle": base64.b64encode(bundle.getvalue()).decode()})
    remote = r'''
import base64, csv, io, json, os, pathlib, subprocess, sys, tarfile
request = json.load(sys.stdin)
home = pathlib.Path.home()
app = home / 'Documents/wa_whisper'
release = home / 'Documents/wa_whisper_device_switch'
release.mkdir(parents=True, exist_ok=True)
with tarfile.open(fileobj=io.BytesIO(base64.b64decode(request['bundle'])), mode='r:gz') as archive:
    archive.extractall(release, filter='data')
env = app / '.env'
lines = env.read_text().splitlines() if env.exists() else []
lines = [line for line in lines if not line.startswith('WA_WHISPER_BROKER_TOKEN=')]
lines.append('WA_WHISPER_BROKER_TOKEN=' + request['token'])
env.write_text('\n'.join(lines) + '\n')
subprocess.run(['icacls', str(env), '/inheritance:r', '/grant:r',
                '*' + list(csv.reader([subprocess.check_output(['whoami', '/user', '/fo', 'csv', '/nh'], text=True).strip()]))[0][1] + ':(F)', 'SYSTEM:(F)'],
               check=True, stdout=subprocess.DEVNULL)
pth = app / '.venv/Lib/site-packages/zzzz_whisper_device_switch.pth'
pth.write_text('import sys; sys.path.insert(0, ' + repr(str(release / 'src')) + ')\n')
print('Windows broker staged; existing microphone power state preserved.')
'''
    encoded = base64.b64encode(remote.encode()).decode()
    command = ('C:\\Users\\John\\Documents\\wa_whisper\\.venv\\Scripts\\python.exe -c '
               '"import base64;exec(base64.b64decode(\'' + encoded + '\'))"')
    subprocess.run([*SSH, command], input=payload, text=True, check=True)
    subprocess.run([*SSH, "powershell -NoProfile -ExecutionPolicy Bypass -File "
                    "C:\\Users\\John\\Documents\\wa_whisper_device_switch\\infra\\scripts\\install_windows_broker.ps1"], check=True)


if __name__ == "__main__":
    main()
