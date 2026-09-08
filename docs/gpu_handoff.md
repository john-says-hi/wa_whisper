# Cooperative desktop GPU handoff

Whisper can finish the current recording and its pending transcription before
stopping for a local GPU job. It stops accepting new captures while it waits.
Right Alt release and the hands-free completion chord continue working. A wait
timeout or disconnected requester reopens recording admission without cutting
short an active recording.

The control client does not import Torch or load a model:

```bash
infra/scripts/wa-whisper-control status
infra/scripts/wa-whisper-control quiesce-stop --wait-seconds 300
```

`quiesce-stop` stops dictation once processing finishes. Run it through the
Mimic supervised handoff workflow when automatic restoration is wanted. A
standalone invocation does not start a restoration guardian.

## Socket contract

The Linux socket is `$XDG_RUNTIME_DIR/wa_whisper/control.sock`, defaulting to
`/run/user/<uid>/wa_whisper/control.sock`. Its directory is mode `0700`, its
socket is mode `0600`, and the server checks the peer UID. A lock prevents a
second server from replacing an active endpoint. A crashed server's socket is
replaced on the next startup. No transcript or audio content is returned.

Send one newline-terminated JSON object, at most 4096 bytes:

```json
{"version":1,"command":"status"}
{"version":1,"command":"quiesce_stop","wait_seconds":300}
```

`wait_seconds` must be greater than zero and at most 300. The server returns one
final response. Success is `{"version":1,"ok":true,"result":...}`. The status
result includes `pid`, `mode`, `recording`, `finalizing`, `quiescing`, `closed`,
`processing`, `queued_items`, `worker_alive`, `worker_failed`, and `stopping`.
A successful handoff result contains `pid`, a random `operation_id`, and
`state: "stopping"`. The caller must still wait for process exit and VRAM release.

Errors use `{"version":1,"ok":false,"error":{"code":...,"message":...}}`.
Codes are `busy`, `timeout`, `cancelled`, `unavailable`, or `invalid_request`.
The CLI prints JSON and exits zero only for success. Closing the connection
cancels a waiting handoff; it cannot undo an already committed shutdown.

## Recording and processing boundaries

The admission gate shares the hotkey state lock, so a concurrent key press
either starts a capture that finishes normally or observes the closed gate.
Quiescence waits for both idle recording mode and completed finalization. The
capture callback has finished queuing work before that boundary is released.
A barrier then crosses the single processing worker after all earlier audio,
transcript, CopyQ, injection, and metadata work. An empty queue alone does not
mean processing has finished. Queue shutdown rejects later barriers so explicit
shutdown cannot strand queue accounting.

The gate stays closed while the existing shutdown coordinator stops the app.
Explicit SIGTERM or Esc retains its normal interruption behavior. An unexpected
processing-worker failure makes cooperative handoff unavailable until restart,
rather than reporting pending work as completed. Ordinary transcription failures
continue through the existing archive/recovery behavior.

## Supervision and restoration

Mimic owns the external GPU job and restoration. Before requesting handoff it
records whether `wa-whisper-ptt.service` was active and acquires both its desktop
GPU lock and `$XDG_RUNTIME_DIR/wa-whisper-power-toggle.lock`. It waits for Whisper
to exit before loading a GPU workload. The job's systemd user service uses
`KillMode=control-group`; its post-stop cleanup restores originally-active
Whisper only after its owned workload exits. This restoration must also run when
the job fails or is cancelled. A stopped Whisper process cannot recover its own
lease, and an expiring timer must not restart Whisper over an active GPU job.

The control protocol's pre-shutdown lease is bounded by the request deadline
and connection lifetime. After shutdown, restoration belongs to the supervised
job's lifetime. If Whisper was already off, the job leaves it off. Restarting
Whisper does not change its persisted compute mode. Its model loads lazily on
the next transcription.

## Desktop deployment and rollback

This feature is developed from the running desktop snapshot in a separate
worktree. The existing service's Wayland launcher, virtual environment,
injection mode, and input-channel arguments are preserved. The prepared
`config/systemd/40-mimic-handoff.conf` adds only a `PYTHONPATH` override selecting
this worktree's source. Do not reinstall dependencies or change the original
checkout's editable package metadata.

First verify source selection without starting dictation:

```bash
PYTHONPATH=/home/johnwalton/Documents/wa_whisper_gpu_handoff/src \
  /home/johnwalton/Documents/wa_whisper/.venv/bin/python3 -c \
  'import importlib.util; print(importlib.util.find_spec("wa_whisper.main").origin)'
```

At the operator-coordinated service handoff, install the prepared file as
`~/.config/systemd/user/wa-whisper-ptt.service.d/40-mimic-handoff.conf` only after
checking that path is absent or contains this exact owned configuration. Run
`systemctl --user daemon-reload`, then start/restart the service at the coordinated
point. Verify the control CLI status and the `Cooperative handoff control ready`
log entry. The first restart upgrades the old runtime, which has no cooperative
endpoint; finish existing dictation before that restart.

Rollback removes only this owned drop-in, reloads systemd, and restarts the
service. Leave other drop-ins and the original service file untouched. Keep the
worktree until its PR is integrated and the service no longer imports from it.

## Validation

CPU-only tests cover active push-to-talk and hands-free completion, the
finalization callback boundary, competing leases, cancellation and deadlines,
queued and active transcription, worker failure, socket ownership, malformed
requests, and a complete owned child-process shutdown. They use fake audio and
processing and do not load models. New tests live under `tests/unit` and
`tests/integration`; existing hotkey and archive recovery tests are also targeted.

Live service restart and GPU handoff are separate deployment checks. The feature
implementation does not itself perform them or start training.
