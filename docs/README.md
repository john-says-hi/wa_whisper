# wa_whisper

For Windows installation and Ctrl+Shift+F1 power control, see [Windows voice typing](windows.md).

Early prototype of a push-to-talk dictation utility that mirrors the ergonomics of `wa_parakeet` while using OpenAI Whisper `large-v3` as the ASR backend.

## Quickstart

1. Create the virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Launch the push-to-talk loop:
   ```bash
   wa-whisper
   ```

## Status

The prototype currently includes:

- Right-Alt push-to-talk capture with the Parakeet ergonomics you expect.
- Hands-free recording toggled with Left Ctrl followed by Right Alt.
- Background worker that runs Whisper `large-v3` with FP16 when CUDA is available.
- Lightweight text post-processing (number/acronym normalization, punctuation fixes).
- Optional desktop audio muting via `wpctl`/`pactl` and logging to `~/.cache/wa_whisper`.
- Durable local recovery for each dictation's original audio and final transcript.
- App-aware text injection: Warp and ordinary windows use the original `xdotool` typing path, while Orca terminals use Orca's daemon write path.
- Placeholder voice isolation hook for upcoming DTLN integration.

Use `scripts/bootstrap.sh` to provision dependencies, then run `wa-whisper` inside an active terminal
session. Hold Right Alt to dictate and release it to transcribe and inject text into the active window.

For hands-free recording, hold Left Ctrl first and press Right Alt once. A short bell confirms that
recording is latched on; releasing both keys does not stop it. Press Left Ctrl followed by Right Alt
again to stop, hear the bell, and begin normal transcription and injection. Right Alt by itself is
ignored while hands-free recording is active. Desktop output stays muted for the full recording to
avoid feedback, and the existing completion bell still confirms successful text injection. If the
microphone cannot start, a second immediate bell signals that hands-free mode returned to off.

Press `Ctrl+C` (or `Esc` when launched with `--exit-on-esc`) to exit. If the service exits during an
active recording, it preserves the finalized audio with an empty transcript and interrupted metadata
without transcribing or injecting it.

## Compute Modes

For a recording-safe GPU handoff to Mimic, see
[Cooperative desktop GPU handoff](gpu_handoff.md). Its local control client waits
for active dictation and pending processing before stopping the service.
The same guide covers the optional shared local-media queue, new-capture gating,
and process-lifetime admission for standalone GPU utilities.

`wa_whisper` has two restart-based compute modes:

```bash
wa-whisper-mode gpu
wa-whisper-mode ram
wa-whisper-mode status
```

`gpu` is the default when no mode file exists. It uses the existing OpenAI Whisper/PyTorch backend on
CUDA with FP16 enabled. `ram` uses the same backend on CPU/system RAM with FP16 disabled. The selected
mode is stored in `~/.config/wa_whisper/compute_mode`, then `wa-whisper-mode` restarts the user
service so the old model process exits and releases VRAM.

Manual runs can override the persisted mode without changing it:

```bash
wa-whisper --compute-mode ram
wa-whisper --compute-mode gpu
```

The lower-level `--device cpu|cuda` flag still works for direct debugging, but it cannot be combined
with `--compute-mode`.

Expect `ram` mode to be much slower than the RTX CUDA path because inference runs on the CPU, not only
from a different memory pool. Use it when freeing VRAM matters more than dictation latency.

Verification signs:

- `~/.cache/wa_whisper/service.log` contains `Whisper compute mode ram -> device=cpu fp16=False` or
  `Whisper compute mode gpu -> device=cuda fp16=True`.
- `nvidia-smi` should stop showing the `wa-whisper` Python process after switching to `ram`.
- Switching back to `gpu` should restart the service and load the model back onto CUDA on the next
  transcription.

## Dictation Recovery

`wa_whisper` saves every completed capture locally before transcription starts. This protects long
dictations if RAM-mode transcription takes a while, focus changes before text injection, or injection
reports success but the text does not land where expected.

Archived dictations live under:

```bash
~/Music/wa_whisper_recordings/
```

The pre-cutover archive remains unchanged under
`~/Music/wa_whisper_recordings/old_archive_through_july_28th_2026/`.

New records are grouped by local calendar date and use a human-readable local timestamp, timezone,
and unique suffix. For example:

```text
2026-07-28/2026-07-28_at_03-26-55_PM_PDT_ab12cd34/
```

Each timestamped record contains:

- `audio.wav`: the original recorder output.
- `transcript.txt`: the final post-processed transcript, or an empty file when transcription fails or
  produces no text.
- `metadata.json`: capture stats, backend mode/device details, CopyQ recovery queue status,
  injection status, and error details when processing failed.

An active capture interrupted by service shutdown or `Esc` is still archived. Its transcript remains
empty and its metadata records the interruption time and reason.

The latest recovery files are also copied to:

```bash
~/Music/wa_whisper_recordings/latest/audio.wav
~/Music/wa_whisper_recordings/latest/transcript.txt
~/Music/wa_whisper_recordings/latest/metadata.json
```

After every successful transcription, `wa_whisper` tries to insert the final transcript into CopyQ
history row `1` in both `gpu` and `ram` modes. This preserves the active clipboard item at row `0`, so
normal paste keeps using whatever you already had copied while the latest dictation stays nearby in
CopyQ. If CopyQ is unavailable, the archive still keeps the latest transcript on disk.

Permanent timestamped records are never deleted automatically. Delete them manually when needed and
monitor available disk space as the archive grows. `latest` is an atomic convenience link to a complete
derived snapshot, so its audio, transcript, and metadata always refer to the same record. Only its
current and previous derived snapshots are rotated; the corresponding timestamped records remain
permanent.

If a script reads all three `latest` files while another capture may finish, resolve the `latest`
link once and read the files from that resolved snapshot directory.

Audio and transcripts are stored as local plaintext files, and CopyQ also stores transcript text, so
treat both as sensitive.

Recording streams to disk and has no program timer, so hands-free sessions lasting minutes or hours
do not accumulate the entire WAV in memory. The current standard WAV format is not intended for
all-day captures and reaches its format boundary at roughly 12 hours with this workstation's 48 kHz
input.

## Text Injection Modes

The default CLI mode is `xdotool-type`, which preserves the original behavior:

```bash
wa-whisper
```

For day-to-day service use with Orca and Warp, use `auto`:

```bash
wa-whisper --injection-mode auto
```

In `auto` mode, `wa_whisper` inspects the active X11 window. Orca windows route through the Orca daemon so dictated text is written directly to the terminal session instead of being replayed as synthetic keystrokes. Warp and other windows continue to use the original `xdotool type --clearmodifiers` path. If Orca daemon delivery fails, `wa_whisper` logs the failure and does not fall back to `xdotool` for Orca, avoiding menu/settings hotkeys.

## Systemd Service

`systemd/wa-whisper-ptt.service` installs a user service so the push-to-talk loop launches automatically
at login. The service uses `--injection-mode auto` so Orca terminals receive daemon writes while Warp
keeps the original typing behavior. It also uses `--input-channel 1` to pin this workstation's SSL 2
capture to physical microphone input 1 while preserving a mono WAV. Install/update it with:

```bash
mkdir -p ~/.config/systemd/user
cp systemd/wa-whisper-ptt.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now wa-whisper-ptt.service
```

This replaces the legacy Parakeet service; disable it with `systemctl --user disable --now parakeet-ptt.service`.
