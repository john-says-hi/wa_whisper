# Desktop and laptop GPU switching

While desktop dictation is on, press **Left Ctrl + Left Shift + Left Alt + F1**
to switch the inference destination. “Transferring voice” confirms initiation;
“Voice ready” confirms the destination model has loaded and dictation is available.
Both use the same Kokoro af_bella voice as the power announcements. Ctrl+Shift+F1 still turns dictation off/on. Ordinary recording
shortcuts and the desktop microphone stay the same. The last successful destination
is saved in `~/.config/wa_whisper/destination.json` across power toggles and restarts.

Desktop-to-laptop transfers prioritize releasing the desktop GPU. After accepting
the switch, the app saves laptop as the destination, cancels any active desktop
inference, and waits for its owned process to exit before contacting/loading the
laptop model. The interrupted recording remains at the head of the transcription
queue and retries from the same WAV; later recordings do not overtake it. GPU
release follows process exit, rather than waiting through laptop model loading.

If laptop connection or loading fails, laptop remains selected and the desktop
model stays unloaded, including after restarting dictation. Pending audio waits
and retries when the laptop becomes ready. The seven-waiting-recording limit still
applies. Use the switch shortcut to explicitly return to the desktop; there is no
automatic desktop reload. “Voice ready” is spoken only after model readiness.
A configuration or preference-save failure before release leaves the original
working destination intact.

Laptop-to-desktop transfers still load and warm up the desktop first, subject to
GPU admission, then release the laptop lease. Failed desktop loads retain the
laptop. Repeated switch presses are ignored while a transfer runs.

Microphone hotkeys work throughout either transfer: finish push-to-talk or
hands-free recordings normally. Completed WAVs wait in the FIFO queue until the
selected model is ready. Recording does not delay a transfer. Shutdown preserves
queued audio through the normal archive/recovery path. Cached announcements load
no speech model and wait until microphone capture/finalization ends before speaking.

## Recording while transcription is busy

Both Right Alt push-to-talk and Left Ctrl + Right Alt hands-free can record again
while earlier audio is processing. The existing one-second accidental-repress
protection and capture-finalization guard remain; a two-second pause is sufficient
once the previous WAV has finished saving. There is one processing slot and up to
seven completed recordings waiting in FIFO order, including during a GPU transfer.

When seven are waiting, a new recording attempt says “Recording queue full” in the
same cached voice and leaves the microphone idle. Accepted recordings are never
evicted to make room. Once the worker takes another item, recording is available
again. Results use the existing transcript, CopyQ and insertion path in order.
Control drain/shutdown markers do not consume recording slots. Status `queued_items`
reports waiting recordings only, separate from the `processing` flag.

## Connection and shared ownership

The Windows sign-in task `WA Whisper Model Broker` owns one inference worker.
Desktop and laptop microphone clients share its serialized, alternating queue.
Turning the laptop microphone off releases only its lease; desktop dictation keeps
working. With no clients or accepted inference remaining, the model exits.
A parent watchdog prevents orphan inference processes from retaining CUDA memory.

The broker binds laptop loopback port 47631. Desktop uses local port 47632 through
host-key-verified `windows-laptop` SSH. Each application's root `.env` contains
`WA_WHISPER_BROKER_TOKEN`; never copy the token into documentation or logs.
`~/.config/wa_whisper/broker.json` can override the SSH host alias, ports and env-file
path. Any alternate alias must identify the same trusted laptop.
Requests have five-second timeouts, heartbeats run every five seconds, and leases
expire after twenty seconds. Switch preparation/drain is bounded to five minutes.
The broker preserves per-request decoding settings and rejects unknown options.

## Outages and recovery

Laptop mode stays selected during a disconnect and reconnects in the background.
It never automatically loads the desktop GPU. The switch shortcut can still return
to desktop without contacting the laptop, subject to local GPU admission.
Desktop WAVs and recovery markers are persisted before upload in the existing
archive. Recovered text goes to the desktop transcript and CopyQ history; it is
never typed later into an unrelated focused window. Failed history delivery still
leaves the transcript file. Transient laptop WAVs are removed after processing.

Inspect the running service with `infra/scripts/wa-whisper-control status`.
`switch-device` and `cancel-switch` provide the corresponding local controls.
The status reports destination, readiness, transition state and local model PID.
Mimic's remote-aware handoff leaves laptop dictation running when the local model
PID is absent. Local GPU admission still gates a switch back to desktop.

## Installation and rollback

The active desktop service is `wa-whisper-ptt.service`; its `90-device-switch.conf`
selects this worktree. `infra/scripts/activate_desktop_switch.py` activates at an
idle boundary. `stage_windows_broker.py` stages the laptop source and installs
its independent sign-in task. Laptop microphone power is controlled separately.
Keep this worktree present while the installed services point at it.

Rollback files are retained in `~/Documents/wa_whisper_device_switch_deployment`.
Stop desktop dictation at an idle point using `quiesce-stop`. Restore files listed
in `desktop_files.json` from their saved paths, removing entries whose original
value is null, then run `systemctl --user daemon-reload`. Restore the previous
Mimic launcher and admission manifest from the same directory. On Windows, stop
both Whisper tasks, remove `zzzz_whisper_device_switch.pth` from the original
app virtual environment, disable the model-broker task, then restart the original
voice-typing task. Restart desktop dictation only after its original source is
selected. Keep all audio archives, model caches, and settings.

## Verification and user acceptance

Automated coverage includes failed/successful switches, exact modifier handling,
shared leases, result ownership, alternating queues, protocol authentication,
archive recovery, hotkeys, capture drain, Windows power and GPU handoff.
Live checks in the implementation session verified large-v3 CUDA transcription
over SSH, desktop archive persistence and native text insertion, switching through
the keyboard path, and desktop model release. The final installed destination is
laptop, ready, with no desktop inference PID. Laptop microphone remains off.

User acceptance still includes speaking into the normal microphone through both
destinations, concurrent laptop microphone use, and physically unplugging Ethernet
while recording. These manual cases are not claimed as completed automated checks.
