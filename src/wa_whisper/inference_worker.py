"""Owned CUDA subprocess. Only this process constructs a Whisper backend."""
from __future__ import annotations

import contextlib
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path


def main() -> None:
    from .worker_guard import watch_parent
    if len(sys.argv) == 3 and sys.argv[1] == "--parent-pid":
        watch_parent(int(sys.argv[2]))
    from .whisper_backend import WhisperBackend, WhisperConfig

    backend = None
    output = sys.stdout
    for line in sys.stdin:
        try:
            request = json.loads(line)
            with contextlib.redirect_stdout(sys.stderr):
                if request["command"] == "load":
                    config = request["config"]
                    config["cache_dir"] = Path(config["cache_dir"])
                    backend = WhisperBackend(WhisperConfig(**config), Path(request["log_path"]))
                    # The parent owns admission and the lifetime of this child.
                    base_config = backend._config
                    backend.enable_cooperative_capture()
                    backend.load()
                    if request.get("warmup", False):
                        import numpy as np
                        backend._model.transcribe(np.zeros(16000, dtype=np.float32), language="en",
                                                  fp16=backend._fp16, beam_size=backend._config.beam_size)
                    result = backend.archive_metadata()
                elif request["command"] == "transcribe" and backend is not None:
                    backend._config = replace(base_config, **request.get("decode_options", {}))
                    result = asdict(backend.transcribe(Path(request["path"])))
                else:
                    raise ValueError("Unsupported inference command")
            response = {"ok": True, "result": result}
        except Exception as exc:  # noqa: BLE001 - serialize third-party inference errors at the process boundary
            import torch
            code = "memory_full" if isinstance(exc, torch.cuda.OutOfMemoryError) else "inference_failed"
            response = {"ok": False, "error": {"code": code, "message": str(exc)}}
        output.write(json.dumps(response) + "\n")
        output.flush()


if __name__ == "__main__":
    main()
