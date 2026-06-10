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
- App-aware text injection: Warp and ordinary windows use the original `xdotool` typing path, while Orca terminals use Orca's daemon write path.
- Placeholder voice isolation hook for upcoming DTLN integration.

Use `scripts/bootstrap.sh` to provision dependencies, then run `wa-whisper` inside an active terminal session. Press Right Alt to dictate; release to transcribe and inject text into the active window. Press `Ctrl+C` (or `Esc` when launched with `--exit-on-esc`) to exit.

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
