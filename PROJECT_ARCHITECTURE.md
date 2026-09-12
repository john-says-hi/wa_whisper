# Whisper architecture

`src/wa_whisper/main.py` wires CLI configuration, recording, transcription,
archiving, text injection, and coordinated shutdown. The package initializer
loads dictation dependencies lazily so control clients need no Torch import.

| Module | Responsibility |
| --- | --- |
| `hotkeys.py` | Push-to-talk and hands-free lifecycle, capture admission gate |
| `recorder.py` | Microphone capture and finalized WAV files |
| `processing_queue.py` | Sequential capture processing and completion barriers |
| `whisper_backend.py` | Lazy model loading and transcription |
| `faster_whisper_backend.py` | Full-model CTranslate2 loading, decoding adaptation and segment conversion |
| `dictation_archive.py`, `recovery_queue.py` | Durable audio/text recovery and CopyQ recovery |
| `control_state.py` | Lease ownership, capture/processing drain, shutdown commitment |
| `control_server.py`, `control_protocol.py`, `control_cli.py` | Private local control transport and CLI |
| `compute_mode.py`, `mode_cli.py` | Persisted compute selection and service restart |

`infra/scripts` contains new operational entrypoints. `config/systemd` contains
the prepared desktop source-selection drop-in. Existing `scripts` launchers and
the installed service remain intact. Tests use CPU fakes under `tests`; new
handoff tests are categorized under `tests/unit` and `tests/integration`.

During GPU handoff, new captures are blocked atomically, existing captures finish
normally, and a queue barrier confirms complete processing before the normal
shutdown path runs. The external Mimic job owns restoring the stopped service;
Whisper never launches a training or writer workload. See `docs/gpu_handoff.md`
for protocol, deployment, and ownership boundaries.

## Shared laptop inference

`device_routing.py` owns the destination transaction and desktop admission;
`model_process.py`, `inference_worker.py` and `worker_guard.py` own disposable CUDA
processes. `broker_state.py` schedules one laptop model, scoped jobs and client
leases; `broker_server.py` exposes authenticated loopback RPC. `broker_client.py`
owns SSH, heartbeats and the lightweight Windows client. `device_recovery.py`
retries desktop archive records without delayed text injection. `device_config.py`
persists destination and connection settings; `device_notices.py` plays cached
announcements. See [operations and rollback](docs/device_switch.md).

Desktop inference and the laptop broker default to faster-whisper with full
large-v3 FP16. `WhisperConfig.engine` selects the implementation
inside the same disposable worker. Both engines share decoding options and the
WhisperResult contract. Warm-up consumes faster-whisper's lazy segment iterator
before readiness is published. Model disposal, leases and ordered queues retain
their existing ownership boundaries. Broker configuration can select `openai` for
rollback without replacing the deployed source; desktop supports `--engine openai`.

Device transfers gate inference rather than microphone admission. The existing
CaptureWorker FIFO holds completed recordings until the switch settles; the
inference lock protects transcription and model replacement. Desktop-to-laptop
transfer cancels active local inference before acquiring that lock; the same WAV
retries without surrendering its FIFO position. Cancellation
wakes waiting processing so shutdown can archive remaining recordings. External
GPU handoff still uses capture quiescence and a queue drain.

`recording_admission.py` combines the seven-waiting-recording limit with existing
GPU admission before either microphone hotkey starts capture. CaptureQueue counts
only audio jobs; control barriers and shutdown sentinels remain unbounded so queue
capacity cannot obstruct graceful shutdown. The microphone is the single producer,
and finalization completes enqueueing before the next recording is admitted.

Desktop GPU release precedes laptop load. The desktop inference cancellation
callback terminates the owned worker; the switch confirms close before constructing
or loading the remote client. Laptop preference is saved before release, so failed
loads and restarts cannot silently reload desktop CUDA. Pending laptop load retries
hold the current recording until readiness, shutdown or an explicit destination
switch. The reverse transfer retains target-first warm-up.
