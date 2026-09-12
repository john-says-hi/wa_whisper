"""Explicit live probe: real remote inference and local archive, deferring UI insertion."""
import importlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from wa_whisper.broker_client import BrokerClient
from wa_whisper.device_recovery import DeviceRecovery
from wa_whisper.device_routing import RoutedBackend
from wa_whisper.dictation_archive import DictationArchive
from wa_whisper.whisper_backend import WhisperConfig


def main():
    root = Path.home() / "Documents/wa_whisper_validation/live_probe"
    root.mkdir(parents=True, exist_ok=True)
    source = Path(__file__).resolve().parents[2] / "assets/device_notices/desktop_online.wav"
    capture = root / "capture.wav"
    shutil.copy2(source, capture)
    backend = RoutedBackend(WhisperConfig(device="cuda", compute_mode="gpu", fp16=True), root / "probe.log")
    if backend.destination != "laptop":
        raise RuntimeError("Select laptop mode before this probe; no desktop model will be loaded")
    backend.remote = BrokerClient()
    backend.remote.settings["local_port"] = 47633
    archive = DictationArchive(root=root / "archive")
    recovery = DeviceRecovery(backend, archive)
    backend.recovery = recovery
    pipeline = importlib.import_module("wa_whisper.main")
    captured = []
    pipeline.inject_text = lambda text, *args, **kwargs: captured.append(text) or True
    try:
        pipeline.process_capture(audio_path=capture, stats=None, created_at=datetime.now(timezone.utc),
                                 backend=backend, voice_isolation=None, log_path=root / "probe.log",
                                 append_space=True, normalize_numbers=True, normalize_acronyms=True,
                                 ensure_punct=True, xdotool_bin=Path("/usr/bin/xdotool"), enable_beep=False,
                                 injection_mode=pipeline.InjectionMode.WTYPE, orca_daemon_dir=None,
                                 orca_session_id=None, archive=archive)
        if len(captured) != 1:
            raise RuntimeError("The transcription pipeline did not reach text insertion; inspect probe.log")
        report = {"text": captured[0], "archive": str(archive.root), "local_model_pid": backend.local.pid}
        (root / "result.json").write_text(json.dumps(report))
        print(json.dumps(report))
    finally:
        backend.recovery = None
        backend.close()


if __name__ == "__main__":
    main()
