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
