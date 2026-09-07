"""Run explicitly in the signed-in Windows session with a known speech WAV."""

import argparse
import json
import os
import tempfile
import time
import tkinter as tk
from pathlib import Path

import torch

from wa_whisper.log_utils import DEFAULT_LOG_PATH
from wa_whisper.recorder import Recorder
from wa_whisper.text_postprocess import postprocess_text
from wa_whisper.whisper_backend import WhisperBackend, WhisperConfig
from wa_whisper.windows_input import type_text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("wave", type=Path)
    parser.add_argument("result", type=Path)
    args = parser.parse_args()
    result = {}
    try:
        tempfile.tempdir = str(args.result.parent)
        links = Path.home() / "AppData/Local/Microsoft/WinGet/Links"
        os.environ["PATH"] = str(links) + os.pathsep + os.environ.get("PATH", "")
        result["torch"] = torch.__version__
        result["gpu"] = torch.cuda.get_device_name()
        backend = WhisperBackend(WhisperConfig(device="cuda", compute_mode="gpu", fp16=True), DEFAULT_LOG_PATH)
        started = time.monotonic()
        text = postprocess_text(backend.transcribe(args.wave).text)
        result["transcript"] = text
        result["inference_with_load_seconds"] = round(time.monotonic() - started, 2)
        recorder = Recorder(16000, None, DEFAULT_LOG_PATH, 0.01)
        recorder.start()
        time.sleep(0.5)
        recorded = recorder.stop(0.0)
        result["microphone_capture"] = recorded is not None and recorded.stat().st_size > 44
        if recorded:
            recorded.unlink()
        root = tk.Tk()
        root.title("WA Whisper installation test")
        entry = tk.Text(root, width=75, height=5)
        entry.pack()
        root.lift()
        root.attributes("-topmost", True)
        entry.focus_force()

        def inject() -> None:
            type_text(text)
            root.after(500, verify)

        def verify() -> None:
            result["typed_text"] = entry.get("1.0", "end-1c")
            result["unicode_injection"] = result["typed_text"] == text and bool(text)
            root.destroy()

        root.after(500, inject)
        root.mainloop()
    except Exception as exc:
        result["error"] = repr(exc)
    args.result.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
