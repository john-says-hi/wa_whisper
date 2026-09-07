# Dictation power toggle

Ctrl+Shift+F1 stops or starts `wa-whisper-ptt.service`. Left Ctrl and Left
Shift work; COSMIC's modifier matching also accepts the right-hand keys.
The shortcut runs outside the dictation process, so it works while the model
is unloaded. Stopping releases the service's GPU allocations. It never disables
autostart: the service still starts at the next login.

Install `infra/scripts/wa-whisper-power-toggle` as
`~/.local/bin/wa-whisper-power-toggle`, executable. Copy the existing Star Trek
recordings `var/phrases/af_bella/{online,powering_down}.wav` into
`~/.local/share/wa_whisper/power_phrases/`. No speech model or Star Trek service
is needed to play these cached recordings. Requires systemctl, flock, timeout,
paplay, and notify-send.

Merge the entry in `config/power_toggle_shortcut.ron` into COSMIC's custom
shortcuts, preserving other bindings. The installed COSMIC defaults and custom
bindings were checked for Ctrl+Shift+F1 conflicts. A global shortcut reserves
this combination from foreground applications.

The stop cue says “Powering down.” before stopping. The start cue says “Online.”
(the existing Star Trek recording) when startup is requested; allow the model
to finish loading before dictating. A desktop notification clarifies loading.
Overlapping toggle invocations are ignored. Audio playback failure does not
prevent service control; service command failures produce an error notification.

Avoid toggling during dictation or transcription: stopping terminates the
running process and may interrupt unfinished work.
