"""Interrupted dictation remains in the original archive and never gets typed later."""
import json
from datetime import datetime, timezone
from types import SimpleNamespace

from wa_whisper import device_recovery as recovery_module
from wa_whisper.device_recovery import MARKER, DeviceRecovery
from wa_whisper.dictation_archive import DictationArchive


def test_recovery_uses_original_archive_and_persists_transcript(tmp_path, monkeypatch):
    source = tmp_path / "original.wav"
    source.write_bytes(b"saved audio")
    archive = DictationArchive(root=tmp_path / "archive")
    record = archive.start_record(source, stats=None, created_at=datetime.now(timezone.utc))
    calls = []
    backend = SimpleNamespace(
        log_path=tmp_path / "log", transcribe=lambda path, recovery: (
            calls.append(path) or SimpleNamespace(text="Recovered words", info={})
        ), acknowledge=lambda path: None, notices=SimpleNamespace(say=lambda message: None),
    )
    history = []
    monkeypatch.setattr(recovery_module, "insert_transcript_into_recovery_queue", lambda text, log: (
        history.append(text) or SimpleNamespace(recovery_queue_metadata=lambda: {"inserted": True})
    ))
    manager = DeviceRecovery(backend, archive)
    manager.begin(record, {})
    manager.finish(record, False)
    # A new instance represents restarting the desktop after losing the connection.
    restarted = DeviceRecovery(backend, archive)
    restarted._recover(record.record_dir / MARKER)
    assert calls == [record.audio_path]
    assert record.audio_path.read_bytes() == b"saved audio"
    assert record.transcript_path.read_text().strip()
    assert len(history) == 1
    metadata = json.loads(record.metadata_path.read_text())
    assert metadata["status"] == "recovered_to_history"
    assert metadata["injection"]["succeeded"] is False
    assert not (record.record_dir / MARKER).exists()


def test_crash_after_transcript_save_does_not_transcribe_again(tmp_path, monkeypatch):
    source = tmp_path / "source.wav"
    source.write_bytes(b"audio")
    archive = DictationArchive(root=tmp_path / "archive")
    record = archive.start_record(source, stats=None, created_at=datetime.now(timezone.utc))
    backend = SimpleNamespace(log_path=tmp_path / "log", acknowledge=lambda path: None,
                              notices=SimpleNamespace(say=lambda message: None))
    manager = DeviceRecovery(backend, archive)
    manager.begin(record, {})
    archive.save_transcript(record, "Already saved.")
    monkeypatch.setattr(recovery_module, "insert_transcript_into_recovery_queue", lambda *args: SimpleNamespace(
        recovery_queue_metadata=lambda: {"inserted": True}
    ))
    manager._recover(record.record_dir / MARKER)
    assert record.transcript_path.read_text() == "Already saved."


def test_background_recovery_discovers_date_partitioned_archive(tmp_path, monkeypatch):
    source = tmp_path / "capture.wav"
    source.write_bytes(b"saved audio")
    archive = DictationArchive(root=tmp_path / "archive")
    record = archive.start_record(source, stats=None, created_at=datetime.now(timezone.utc))
    backend = SimpleNamespace(destination="desktop")
    manager = DeviceRecovery(backend, archive)
    manager.begin(record, {})
    manager.finish(record, False)
    found = []
    monkeypatch.setattr(manager, "_idle", lambda: True)
    monkeypatch.setattr(manager, "_recover", found.append)
    waits = iter([False, True])
    monkeypatch.setattr(manager._closed, "wait", lambda _: next(waits))
    manager._run()
    assert found == [record.record_dir / MARKER]
