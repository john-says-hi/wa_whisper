# Windows voice typing

The Windows launcher reuses the desktop's recorder, Right Alt / hands-free hotkey
state machine, OpenAI Whisper large-v3 backend, and text post-processing. The
Linux launcher is unchanged. Windows uses native Unicode keyboard injection and
an interactive logon task instead of Linux typing tools and systemd.

## Install

Use Python 3.12, FFmpeg (`winget install --id Gyan.FFmpeg --exact`), the Microsoft
Visual C++ x64 runtime (`winget install --id Microsoft.VCRedist.2015+.x64 --exact`), and an NVIDIA
driver compatible with the selected PyTorch CUDA wheel. Validate actual CUDA
transcription before deciding a driver upgrade is necessary.

In the app directory, create `.venv` with Python 3.12, then run:

```powershell
.venv\Scripts\python.exe -m pip install torch==2.9.0 --index-url https://download.pytorch.org/whl/cu128
.venv\Scripts\python.exe -m pip install -r requirements_windows.txt
.venv\Scripts\python.exe -m pip install --no-deps -e .
```

Copy the existing verified `large-v3.pt` model into
`C:\Users\John\.cache\huggingface\hub` (or the corresponding Windows user's home).
Run `powershell -NoProfile -ExecutionPolicy Bypass -File infra/scripts/install_windows_logon.ps1`
as the intended user. The execution policy override applies only to that setup
process; it does not change the machine policy. The script creates
the `WA Whisper Voice Typing` interactive scheduled task and Desktop shortcut.
The task starts at that user's Windows sign-in and runs without elevation or a
stored password. It can run on battery and has no execution time limit.

## Use

- Ctrl+Shift+F1 powers dictation off/on, unloading the GPU worker when it exits.
- System speakers mute during recording and return to their previous mute state
  afterward, including when recording fails or dictation is powered off. Volume
  level is unchanged. Audio already muted stays muted.
- Hold Right Alt to speak; release to transcribe into the focused text field.
  The Windows adapter normalizes pynput's AltGr representation of VK_RMENU to
  the shared recorder's Right Alt key on both press and release.
- Left Ctrl followed by Right Alt toggles hands-free recording.
- Closing the small control window minimizes it; the power shortcut stays active.
- Opening the Desktop shortcut again explicitly turns dictation on if it was off.
  It does not toggle a running worker off. The status file changes to OFF when
  the worker exits, rather than retaining the worker's previous Ready message.
- Turning off during a recording preserves audio. Turning off during inference
  waits for inference to finish, saves the transcript, and skips text injection.

Whisper runs locally. Native Windows Win+H remains separately available; do not
record with both at once. The Windows adapter does not provide CopyQ history integration or the Linux
`latest` symlink archive.
Avoid elevated applications: ordinary Windows processes cannot inject text into
an elevated application. Dictations go to the focused field when inference finishes.

Audio and transcripts are retained under `Music\wa_whisper_recordings` in unique
timestamped directories. Errors are saved in each record's `metadata.json`.
Logs and current status are under `.cache\wa_whisper`, including
`windows_worker.log`, `push_to_talk.log`, and `windows_status.txt`.

For remote administration, use the installed virtual environment's Python:

```powershell
.venv\Scripts\python.exe -m wa_whisper.windows_control status
.venv\Scripts\python.exe -m wa_whisper.windows_control start
.venv\Scripts\python.exe -m wa_whisper.windows_control stop
```

These send requests to the interactive controller; they do not start a second
GUI in the SSH session. A Running scheduled task means the controller is alive,
not necessarily that dictation is on. Confirm Ready, the worker process, and GPU
memory when diagnosing nonresponsive recording keys.

## Verify and disable

Test GPU transcription of a known WAV, interactive keyboard injection, the global
power shortcut in both directions, microphone dictation, and startup after sign-in.
Linux unit tests: `pytest tests/unit/test_windows_recovery.py tests/unit/test_windows_hotkey_shield.py`.
The scripts under `tests/integration/windows_*_probe.py` require an interactive
Windows session. The input probe refuses to type until its own test window has
focus. Click its text box if Windows prevents a background process taking focus.
The hotkey probe uses actual Windows keyboard hooks with synthetic keystrokes
and a harmless recorder, verifying push-to-talk and hands-free capture without
dictating text into another application.

To prevent future startup, disable the `WA Whisper Voice Typing` scheduled task.
Turn dictation off with Ctrl+Shift+F1 first so active audio is saved, then end the
controller task if its global shortcut should also be removed. Disabling Windows
Win+H is a separate setting; installing this adapter does not modify it.
