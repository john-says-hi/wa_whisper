# wa_whisper

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
- Background worker that runs Whisper `large-v3` with FP16 when CUDA is available.
- Lightweight text post-processing (number/acronym normalization, punctuation fixes).
- Optional desktop audio muting via `wpctl`/`pactl` and logging to `~/.cache/wa_whisper`.
- Durable local recovery for each dictation's original audio and final transcript.
- App-aware text injection: Warp and ordinary windows use the original `xdotool` typing path, while Orca terminals use Orca's daemon write path.
- Placeholder voice isolation hook for upcoming DTLN integration.

Use `scripts/bootstrap.sh` to provision dependencies, then run `wa-whisper` inside an active terminal session. Press Right Alt to dictate; release to transcribe and inject text into the active window. Press `Ctrl+C` (or `Esc` when launched with `--exit-on-esc`) to exit.

## Compute Modes

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
~/.local/share/wa_whisper/dictations/
```

Each timestamped record contains:

- `audio.wav`: the original recorder output.
- `transcript.txt`: the final post-processed transcript when Whisper produced text.
- `metadata.json`: capture stats, backend mode/device details, CopyQ recovery queue status,
  injection status, and error details when processing failed.

The latest recovery files are also copied to:

```bash
~/.local/share/wa_whisper/dictations/latest/audio.wav
~/.local/share/wa_whisper/dictations/latest/transcript.txt
~/.local/share/wa_whisper/dictations/latest/metadata.json
```

After every successful transcription, `wa_whisper` tries to insert the final transcript into CopyQ
history row `1` in both `gpu` and `ram` modes. This preserves the active clipboard item at row `0`, so
normal paste keeps using whatever you already had copied while the latest dictation stays nearby in
CopyQ. If CopyQ is unavailable, the archive still keeps the latest transcript on disk.

Records are retained for 90 days and pruned automatically at startup and after captures. Audio and
transcripts are stored as local plaintext files, and CopyQ also stores transcript text, so treat both as
sensitive.

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

`systemd/wa-whisper-ptt.service` installs a user service so the push-to-talk loop launches automatically at login. The service uses `--injection-mode auto` so Orca terminals receive daemon writes while Warp keeps the original typing behavior. Install/update it with:

```bash
mkdir -p ~/.config/systemd/user
cp systemd/wa-whisper-ptt.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now wa-whisper-ptt.service
```

This replaces the legacy Parakeet service; disable it with `systemctl --user disable --now parakeet-ptt.service`.
