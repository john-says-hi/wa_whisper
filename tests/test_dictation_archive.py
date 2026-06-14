import json
import stat
from datetime import datetime, timedelta, timezone
from pathlib import Path

from wa_whisper.dictation_archive import (
    AUDIO_FILE_NAME,
    LATEST_DIR_NAME,
    METADATA_FILE_NAME,
    TRANSCRIPT_FILE_NAME,
    DictationArchive,
)
from wa_whisper.recorder import RecorderStats


def test_archive_start_record_copies_audio_and_clears_stale_latest_transcript(tmp_path):
    source_audio = tmp_path / "capture.wav"
    source_audio.write_bytes(b"audio bytes")
    archive_root = tmp_path / "archive"
    latest_transcript = archive_root / LATEST_DIR_NAME / TRANSCRIPT_FILE_NAME
    latest_transcript.parent.mkdir(parents=True)
    latest_transcript.write_text("stale", encoding="utf-8")
    archive = DictationArchive(root=archive_root)

    record = archive.start_record(
        source_audio,
        stats=sample_stats(),
        backend={"device": "cpu", "compute_mode": "ram"},
    )

    assert record.audio_path.read_bytes() == b"audio bytes"
    assert (archive_root / LATEST_DIR_NAME / AUDIO_FILE_NAME).read_bytes() == b"audio bytes"
    assert not latest_transcript.exists()
    assert stat.S_IMODE(record.record_dir.stat().st_mode) == 0o700
    assert stat.S_IMODE((archive_root / LATEST_DIR_NAME).stat().st_mode) == 0o700

    metadata = read_json(record.metadata_path)
    assert metadata["status"] == "audio_archived"
    assert metadata["retention_days"] == 90
    assert metadata["capture_stats"]["total_ms"] == 1250.0
    assert metadata["capture_stats"]["speech_ratio"] == 0.8
    assert metadata["backend"] == {"device": "cpu", "compute_mode": "ram"}
    assert metadata["paths"]["audio"] == str(record.audio_path)

    latest_metadata = read_json(archive_root / LATEST_DIR_NAME / METADATA_FILE_NAME)
    assert latest_metadata["record_id"] == record.record_id


def test_archive_save_transcript_updates_record_and_latest_files(tmp_path):
    source_audio = tmp_path / "capture.wav"
    source_audio.write_bytes(b"audio bytes")
    archive = DictationArchive(root=tmp_path / "archive")
    record = archive.start_record(source_audio, stats=None, backend=None)

    archive.save_transcript(record, "hello world", whisper_info={"duration": 1.5})

    assert record.transcript_path.read_text(encoding="utf-8") == "hello world"
    assert (archive.latest_dir / TRANSCRIPT_FILE_NAME).read_text(encoding="utf-8") == "hello world"

    metadata = read_json(record.metadata_path)
    assert metadata["status"] == "transcribed"
    assert metadata["transcription"] == {
        "text_chars": 11,
        "whisper_info": {"duration": 1.5},
    }


def test_archive_prune_expired_removes_old_records_but_keeps_latest(tmp_path):
    archive_root = tmp_path / "archive"
    archive = DictationArchive(root=archive_root, retention_days=90)
    old_record = create_record_dir(
        archive_root,
        "2026-01-01",
        "old",
        datetime.now(timezone.utc) - timedelta(days=91),
    )
    fresh_record = create_record_dir(
        archive_root,
        "2026-02-01",
        "fresh",
        datetime.now(timezone.utc) - timedelta(days=10),
    )
    latest_dir = archive_root / LATEST_DIR_NAME
    latest_dir.mkdir(parents=True)
    (latest_dir / METADATA_FILE_NAME).write_text("{}", encoding="utf-8")

    removed = archive.prune_expired()

    assert removed == 1
    assert not old_record.exists()
    assert fresh_record.exists()
    assert latest_dir.exists()


def test_archive_prune_expired_clears_stale_latest_files(tmp_path):
    archive_root = tmp_path / "archive"
    archive = DictationArchive(root=archive_root, retention_days=90)
    latest_dir = archive_root / LATEST_DIR_NAME
    latest_dir.mkdir(parents=True)
    stale_created_at = datetime.now(timezone.utc) - timedelta(days=91)
    (latest_dir / AUDIO_FILE_NAME).write_bytes(b"old audio")
    (latest_dir / TRANSCRIPT_FILE_NAME).write_text("old transcript", encoding="utf-8")
    (latest_dir / METADATA_FILE_NAME).write_text(
        json.dumps({"created_at": stale_created_at.isoformat().replace("+00:00", "Z")}),
        encoding="utf-8",
    )

    archive.prune_expired()

    assert list(latest_dir.iterdir()) == []


def sample_stats() -> RecorderStats:
    return RecorderStats(
        total_ms=1250.0,
        speech_ms=1000.0,
        silence_ms=250.0,
        max_rms=0.5,
        avg_silence_rms=0.01,
        speech_blocks=10,
        silence_blocks=2,
        total_blocks=12,
        max_speech_streak_ms=800.0,
    )


def create_record_dir(root: Path, date_dir: str, record_id: str, created_at: datetime) -> Path:
    record_dir = root / date_dir / record_id
    record_dir.mkdir(parents=True)
    (record_dir / AUDIO_FILE_NAME).write_bytes(b"audio")
    (record_dir / METADATA_FILE_NAME).write_text(
        json.dumps(
            {
                "record_id": record_id,
                "created_at": created_at.isoformat().replace("+00:00", "Z"),
            }
        ),
        encoding="utf-8",
    )
    return record_dir


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))
