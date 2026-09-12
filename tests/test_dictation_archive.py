import json
import os
import stat
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import pytest

import wa_whisper.dictation_archive as archive_module
from wa_whisper.dictation_archive import (
    AUDIO_FILE_NAME,
    DEFAULT_ARCHIVE_ROOT,
    LATEST_DIR_NAME,
    LATEST_LOCK_FILE_NAME,
    LATEST_POINTER_PREFIX,
    LATEST_SNAPSHOT_PREFIX,
    LATEST_SNAPSHOT_ROOT_NAME,
    METADATA_FILE_NAME,
    TRANSCRIPT_FILE_NAME,
    DictationArchive,
    DictationRecord,
)
from wa_whisper.recorder import RecorderStats


@pytest.fixture
def vancouver_system_timezone():
    original_timezone = os.environ.get("TZ")
    os.environ["TZ"] = "America/Vancouver"
    time.tzset()
    try:
        yield
    finally:
        if original_timezone is None:
            os.environ.pop("TZ", None)
        else:
            os.environ["TZ"] = original_timezone
        time.tzset()


def test_archive_defaults_to_music_recordings_folder():
    assert DEFAULT_ARCHIVE_ROOT == Path.home() / "Music" / "wa_whisper_recordings"
    assert DictationArchive().root == DEFAULT_ARCHIVE_ROOT


def test_archive_start_record_creates_complete_private_record_and_latest_snapshot(
    tmp_path,
    monkeypatch,
    vancouver_system_timezone,
):
    source_audio = tmp_path / "capture.wav"
    source_audio.write_bytes(b"audio bytes")
    source_audio.chmod(0o644)
    archive_root = tmp_path / "archive"
    latest_transcript = archive_root / LATEST_DIR_NAME / TRANSCRIPT_FILE_NAME
    latest_transcript.parent.mkdir(parents=True)
    latest_transcript.write_text("stale", encoding="utf-8")
    (latest_transcript.parent / AUDIO_FILE_NAME).write_bytes(b"legacy audio")
    (latest_transcript.parent / METADATA_FILE_NAME).write_text(
        '{"record_id": "legacy"}\n',
        encoding="utf-8",
    )
    archive = DictationArchive(root=archive_root)
    fixed_uuid = uuid.UUID("1234abcd-0000-0000-0000-000000000000")
    monkeypatch.setattr(archive_module.uuid, "uuid4", lambda: fixed_uuid)
    created_at = datetime(2026, 7, 28, 22, 26, 55, tzinfo=timezone.utc)

    record = archive.start_record(
        source_audio,
        stats=sample_stats(),
        backend={"device": "cpu", "compute_mode": "ram"},
        created_at=created_at,
    )

    assert record.record_dir.parent == archive_root / "2026-07-28"
    assert record.record_id == "2026-07-28_at_03-26-55_PM_PDT_1234abcd"
    assert record.audio_path.read_bytes() == b"audio bytes"
    assert record.transcript_path.read_text(encoding="utf-8") == ""
    assert (archive.latest_dir / AUDIO_FILE_NAME).read_bytes() == b"audio bytes"
    assert latest_transcript.read_text(encoding="utf-8") == ""
    assert archive.latest_dir.is_symlink()
    assert archive.latest_dir.resolve().parent == archive_root / LATEST_SNAPSHOT_ROOT_NAME
    assert (archive_root / LATEST_LOCK_FILE_NAME).is_file()
    assert stat.S_IMODE((archive_root / LATEST_LOCK_FILE_NAME).stat().st_mode) == 0o600
    migrated_snapshots = [
        snapshot
        for snapshot in snapshot_directories(archive)
        if snapshot != archive.latest_dir.resolve()
    ]
    assert len(migrated_snapshots) == 1
    assert (migrated_snapshots[0] / AUDIO_FILE_NAME).read_bytes() == b"legacy audio"
    assert (
        migrated_snapshots[0] / TRANSCRIPT_FILE_NAME
    ).read_text(encoding="utf-8") == "stale"

    for directory in (
        archive_root,
        record.record_dir.parent,
        record.record_dir,
        archive.latest_dir,
    ):
        assert stat.S_IMODE(directory.stat().st_mode) == 0o700
    for private_file in (
        record.audio_path,
        record.transcript_path,
        record.metadata_path,
        archive.latest_dir / AUDIO_FILE_NAME,
        archive.latest_dir / TRANSCRIPT_FILE_NAME,
        archive.latest_dir / METADATA_FILE_NAME,
    ):
        assert stat.S_IMODE(private_file.stat().st_mode) == 0o600

    metadata = read_json(record.metadata_path)
    assert metadata["created_at"] == "2026-07-28T22:26:55Z"
    assert metadata["updated_at"] == "2026-07-28T22:26:55Z"
    assert metadata["status"] == "audio_archived"
    assert "retention_days" not in metadata
    assert metadata["capture_stats"]["total_ms"] == 1250.0
    assert metadata["capture_stats"]["speech_ratio"] == 0.8
    assert metadata["backend"] == {"device": "cpu", "compute_mode": "ram"}
    assert metadata["paths"]["audio"] == str(record.audio_path)
    assert read_json(archive.latest_dir / METADATA_FILE_NAME) == metadata


def test_archive_completes_permanent_record_before_latest_refresh(tmp_path, monkeypatch):
    source_audio = tmp_path / "capture.wav"
    source_audio.write_bytes(b"audio")
    archive_root = tmp_path / "archive"
    archive = DictationArchive(root=archive_root)
    published_snapshots = []
    original_replace_latest_pointer = archive._replace_latest_pointer

    def assert_permanent_record_is_complete(snapshot_directory: Path) -> Path | None:
        metadata_paths = list(archive_root.glob("????-??-??/*/metadata.json"))
        assert len(metadata_paths) == 1
        record_dir = metadata_paths[0].parent
        assert (record_dir / AUDIO_FILE_NAME).is_file()
        assert (record_dir / TRANSCRIPT_FILE_NAME).is_file()
        assert (record_dir / METADATA_FILE_NAME).is_file()
        assert (snapshot_directory / AUDIO_FILE_NAME).is_file()
        assert (snapshot_directory / TRANSCRIPT_FILE_NAME).is_file()
        assert (snapshot_directory / METADATA_FILE_NAME).is_file()
        published_snapshots.append(snapshot_directory)
        return original_replace_latest_pointer(snapshot_directory)

    monkeypatch.setattr(
        archive,
        "_replace_latest_pointer",
        assert_permanent_record_is_complete,
    )

    archive.start_record(source_audio, stats=None)

    assert published_snapshots == [archive.latest_dir.resolve()]


def test_latest_failure_does_not_block_permanent_record_or_transcript(tmp_path):
    source_audio = tmp_path / "capture.wav"
    source_audio.write_bytes(b"audio bytes")
    archive_root = tmp_path / "archive"
    archive_root.mkdir()
    blocked_latest_path = archive_root / LATEST_DIR_NAME
    blocked_latest_path.write_text("not a directory", encoding="utf-8")
    archive = DictationArchive(root=archive_root)

    record = archive.start_record(source_audio, stats=None)

    assert record.audio_path.read_bytes() == b"audio bytes"
    assert record.transcript_path.read_text(encoding="utf-8") == ""
    assert read_json(record.metadata_path)["status"] == "audio_archived"

    archive.save_transcript(record, "durable transcript")
    archive.update_record(record, status="injected")

    assert record.transcript_path.read_text(encoding="utf-8") == "durable transcript"
    assert read_json(record.metadata_path)["status"] == "injected"
    assert blocked_latest_path.read_text(encoding="utf-8") == "not a directory"


def test_archive_uses_local_day_when_utc_capture_crosses_date_boundary(
    tmp_path,
    monkeypatch,
    vancouver_system_timezone,
):
    source_audio = tmp_path / "capture.wav"
    source_audio.write_bytes(b"audio")
    fixed_uuid = uuid.UUID("abcdef12-0000-0000-0000-000000000000")
    monkeypatch.setattr(archive_module.uuid, "uuid4", lambda: fixed_uuid)
    archive = DictationArchive(root=tmp_path / "archive")

    record = archive.start_record(
        source_audio,
        stats=None,
        created_at=datetime(2026, 7, 28, 6, 30, tzinfo=timezone.utc),
    )

    assert record.record_dir.parent.name == "2026-07-27"
    assert record.record_id == "2026-07-27_at_11-30-00_PM_PDT_abcdef12"
    assert read_json(record.metadata_path)["created_at"] == "2026-07-28T06:30:00Z"


def test_archive_same_second_records_have_unique_ids(
    tmp_path,
    monkeypatch,
    vancouver_system_timezone,
):
    source_audio = tmp_path / "capture.wav"
    source_audio.write_bytes(b"audio")
    generated_uuids = iter(
        (
            uuid.UUID("11111111-0000-0000-0000-000000000000"),
            uuid.UUID("22222222-0000-0000-0000-000000000000"),
        )
    )
    monkeypatch.setattr(archive_module.uuid, "uuid4", lambda: next(generated_uuids))
    archive = DictationArchive(root=tmp_path / "archive")
    created_at = datetime(2026, 7, 28, 22, 26, 55, tzinfo=timezone.utc)

    first_record = archive.start_record(source_audio, stats=None, created_at=created_at)
    second_record = archive.start_record(source_audio, stats=None, created_at=created_at)

    assert first_record.record_id == "2026-07-28_at_03-26-55_PM_PDT_11111111"
    assert second_record.record_id == "2026-07-28_at_03-26-55_PM_PDT_22222222"
    assert first_record.record_dir.is_dir()
    assert second_record.record_dir.is_dir()


def test_later_millisecond_wins_even_with_lexically_smaller_record_id(
    tmp_path,
    monkeypatch,
    vancouver_system_timezone,
):
    older_audio = tmp_path / "older.wav"
    newer_audio = tmp_path / "newer.wav"
    older_audio.write_bytes(b"older audio")
    newer_audio.write_bytes(b"newer audio")
    generated_uuids = iter(
        (
            uuid.UUID("ffffffff-0000-0000-0000-000000000000"),
            uuid.UUID("00000000-0000-0000-0000-000000000000"),
        )
    )
    monkeypatch.setattr(archive_module.uuid, "uuid4", lambda: next(generated_uuids))
    archive = DictationArchive(root=tmp_path / "archive")

    older_record = archive.start_record(
        older_audio,
        stats=None,
        created_at=datetime(2026, 7, 28, 20, 0, 0, 100_000, tzinfo=timezone.utc),
    )
    newer_record = archive.start_record(
        newer_audio,
        stats=None,
        created_at=datetime(2026, 7, 28, 20, 0, 0, 200_000, tzinfo=timezone.utc),
    )

    assert older_record.record_id.endswith("ffffffff")
    assert newer_record.record_id.endswith("00000000")
    assert read_json(older_record.metadata_path)["created_at"] == "2026-07-28T20:00:00.100000Z"
    assert read_json(newer_record.metadata_path)["created_at"] == "2026-07-28T20:00:00.200000Z"
    assert_latest_matches_record(archive, newer_record)


def test_latest_order_parses_legacy_second_timestamp_as_aware_utc():
    legacy_order = archive_module.latest_record_order(
        {
            "created_at": "2026-07-28T20:00:01+01:00",
            "record_id": "z-record",
        }
    )
    newer_utc_order = archive_module.latest_record_order(
        {
            "created_at": "2026-07-28T19:30:00Z",
            "record_id": "a-record",
        }
    )

    assert legacy_order is not None
    assert newer_utc_order is not None
    assert newer_utc_order > legacy_order


def test_archive_retries_uuid_collision(
    tmp_path,
    monkeypatch,
    vancouver_system_timezone,
):
    source_audio = tmp_path / "capture.wav"
    source_audio.write_bytes(b"audio")
    archive_root = tmp_path / "archive"
    date_dir = archive_root / "2026-07-28"
    colliding_record_id = "2026-07-28_at_03-26-55_PM_PDT_11111111"
    colliding_record_dir = date_dir / colliding_record_id
    colliding_record_dir.mkdir(parents=True)
    sentinel = colliding_record_dir / "sentinel.txt"
    sentinel.write_text("keep me", encoding="utf-8")
    generated_uuids = iter(
        (
            uuid.UUID("11111111-0000-0000-0000-000000000000"),
            uuid.UUID("22222222-0000-0000-0000-000000000000"),
        )
    )
    monkeypatch.setattr(archive_module.uuid, "uuid4", lambda: next(generated_uuids))
    archive = DictationArchive(root=archive_root)

    record = archive.start_record(
        source_audio,
        stats=None,
        created_at=datetime(2026, 7, 28, 22, 26, 55, tzinfo=timezone.utc),
    )

    assert record.record_id == "2026-07-28_at_03-26-55_PM_PDT_22222222"
    assert sentinel.read_text(encoding="utf-8") == "keep me"


def test_archive_rejects_naive_created_at(tmp_path):
    source_audio = tmp_path / "capture.wav"
    source_audio.write_bytes(b"audio")
    archive = DictationArchive(root=tmp_path / "archive")

    with pytest.raises(ValueError, match="timezone"):
        archive.start_record(
            source_audio,
            stats=None,
            created_at=datetime(2026, 7, 28, 15, 26, 55),
        )


def test_archive_save_transcript_updates_record_and_latest_files(tmp_path):
    source_audio = tmp_path / "capture.wav"
    source_audio.write_bytes(b"audio bytes")
    archive = DictationArchive(root=tmp_path / "archive")
    record = archive.start_record(source_audio, stats=None, backend=None)
    initial_snapshot = archive.latest_dir.resolve()

    archive.save_transcript(record, "hello world", whisper_info={"duration": 1.5})

    updated_snapshot = archive.latest_dir.resolve()
    assert updated_snapshot != initial_snapshot
    assert initial_snapshot.is_dir()
    assert (initial_snapshot / TRANSCRIPT_FILE_NAME).read_text(encoding="utf-8") == ""
    assert record.transcript_path.read_text(encoding="utf-8") == "hello world"
    assert (archive.latest_dir / TRANSCRIPT_FILE_NAME).read_text(encoding="utf-8") == "hello world"
    assert (initial_snapshot / AUDIO_FILE_NAME).stat().st_ino != record.audio_path.stat().st_ino
    assert (updated_snapshot / AUDIO_FILE_NAME).stat().st_ino != record.audio_path.stat().st_ino

    metadata = read_json(record.metadata_path)
    assert metadata["status"] == "transcribed"
    assert metadata["updated_at"].endswith("Z")
    assert metadata["transcription"] == {
        "text_chars": 11,
        "whisper_info": {"duration": 1.5},
    }
    assert read_json(archive.latest_dir / METADATA_FILE_NAME) == metadata


def test_late_older_updates_cannot_regress_or_mix_latest_snapshot(tmp_path):
    older_audio = tmp_path / "older.wav"
    newer_audio = tmp_path / "newer.wav"
    older_audio.write_bytes(b"older audio")
    newer_audio.write_bytes(b"newer audio")
    archive = DictationArchive(root=tmp_path / "archive")
    older_record = archive.start_record(
        older_audio,
        stats=None,
        created_at=datetime(2026, 7, 28, 20, 0, tzinfo=timezone.utc),
    )
    newer_record = archive.start_record(
        newer_audio,
        stats=None,
        created_at=datetime(2026, 7, 28, 20, 1, tzinfo=timezone.utc),
    )
    archive.save_transcript(newer_record, "newer transcript")
    archive.update_record(newer_record, status="injected")

    archive.save_transcript(older_record, "late older transcript")
    archive.update_record(older_record, status="injected")

    assert_latest_matches_record(archive, newer_record)
    assert older_record.transcript_path.read_text(encoding="utf-8") == "late older transcript"
    assert read_json(older_record.metadata_path)["status"] == "injected"


def test_source_copy_failure_does_not_partly_publish_latest_snapshot(tmp_path, monkeypatch):
    first_audio = tmp_path / "first.wav"
    second_audio = tmp_path / "second.wav"
    first_audio.write_bytes(b"first audio")
    second_audio.write_bytes(b"second audio")
    archive = DictationArchive(root=tmp_path / "archive")
    first_record = archive.start_record(
        first_audio,
        stats=None,
        created_at=datetime(2026, 7, 28, 20, 0, tzinfo=timezone.utc),
    )
    original_atomic_copy = archive_module.atomic_copy

    def fail_staged_transcript_copy(source: Path, destination: Path) -> None:
        if (
            destination.parent.name.startswith(LATEST_SNAPSHOT_PREFIX)
            and destination.parent.parent.name == LATEST_SNAPSHOT_ROOT_NAME
            and source.name == TRANSCRIPT_FILE_NAME
        ):
            raise OSError("staged transcript copy failed")
        original_atomic_copy(source, destination)

    monkeypatch.setattr(archive_module, "atomic_copy", fail_staged_transcript_copy)

    second_record = archive.start_record(
        second_audio,
        stats=None,
        created_at=datetime(2026, 7, 28, 20, 1, tzinfo=timezone.utc),
    )

    assert_latest_matches_record(archive, first_record)
    assert second_record.audio_path.read_bytes() == b"second audio"
    assert second_record.transcript_path.read_text(encoding="utf-8") == ""
    assert read_json(second_record.metadata_path)["record_id"] == second_record.record_id
    assert snapshot_directories(archive) == [archive.latest_dir.resolve()]


def test_pointer_failure_preserves_previous_latest_and_removes_unpublished_snapshot(
    tmp_path,
    monkeypatch,
):
    first_audio = tmp_path / "first.wav"
    second_audio = tmp_path / "second.wav"
    first_audio.write_bytes(b"first audio")
    second_audio.write_bytes(b"second audio")
    archive = DictationArchive(root=tmp_path / "archive")
    first_record = archive.start_record(
        first_audio,
        stats=None,
        created_at=datetime(2026, 7, 28, 20, 0, tzinfo=timezone.utc),
    )
    previous_snapshot = archive.latest_dir.resolve()
    original_replace = archive_module.os.replace

    def fail_latest_pointer_replace(source: Path, destination: Path) -> None:
        if source.name.startswith(LATEST_POINTER_PREFIX) and destination == archive.latest_dir:
            raise OSError("latest pointer replace failed")
        original_replace(source, destination)

    monkeypatch.setattr(archive_module.os, "replace", fail_latest_pointer_replace)

    second_record = archive.start_record(
        second_audio,
        stats=None,
        created_at=datetime(2026, 7, 28, 20, 1, tzinfo=timezone.utc),
    )

    assert archive.latest_dir.resolve() == previous_snapshot
    assert_latest_matches_record(archive, first_record)
    assert second_record.audio_path.read_bytes() == b"second audio"
    assert snapshot_directories(archive) == [previous_snapshot]
    assert not list(archive.root.glob(f"{LATEST_POINTER_PREFIX}*"))


def test_failed_atomic_concrete_latest_exchange_preserves_original_directory(
    tmp_path,
    monkeypatch,
):
    archive_root = tmp_path / "archive"
    concrete_latest = archive_root / LATEST_DIR_NAME
    concrete_latest.mkdir(parents=True)
    (concrete_latest / AUDIO_FILE_NAME).write_bytes(b"legacy audio")
    (concrete_latest / TRANSCRIPT_FILE_NAME).write_text("legacy transcript", encoding="utf-8")
    (concrete_latest / METADATA_FILE_NAME).write_text(
        '{"record_id": "legacy"}\n',
        encoding="utf-8",
    )
    source_audio = tmp_path / "capture.wav"
    source_audio.write_bytes(b"new audio")
    archive = DictationArchive(root=archive_root)

    def fail_atomic_exchange(left: Path, right: Path) -> None:
        raise OSError(f"exchange failed: {left} <-> {right}")

    monkeypatch.setattr(archive_module, "rename_exchange", fail_atomic_exchange)

    record = archive.start_record(source_audio, stats=None)

    assert not archive.latest_dir.is_symlink()
    assert (archive.latest_dir / AUDIO_FILE_NAME).read_bytes() == b"legacy audio"
    assert (
        archive.latest_dir / TRANSCRIPT_FILE_NAME
    ).read_text(encoding="utf-8") == "legacy transcript"
    assert record.audio_path.read_bytes() == b"new audio"
    assert not list(archive.root.glob(f"{LATEST_POINTER_PREFIX}*"))
    assert snapshot_directories(archive) == []


def test_concurrent_directory_reader_sees_complete_atomic_migration_snapshots(
    tmp_path,
    monkeypatch,
):
    archive_root = tmp_path / "archive"
    concrete_latest = archive_root / LATEST_DIR_NAME
    concrete_latest.mkdir(parents=True)
    (concrete_latest / AUDIO_FILE_NAME).write_bytes(b"legacy audio")
    (concrete_latest / TRANSCRIPT_FILE_NAME).write_text("legacy transcript", encoding="utf-8")
    (concrete_latest / METADATA_FILE_NAME).write_text(
        '{"record_id": "legacy"}\n',
        encoding="utf-8",
    )
    source_audio = tmp_path / "capture.wav"
    source_audio.write_bytes(b"new audio")
    archive = DictationArchive(root=archive_root)
    exchange_ready = threading.Event()
    allow_exchange = threading.Event()
    reader_ready = threading.Event()
    new_snapshot_seen = threading.Event()
    stop_reader = threading.Event()
    observations = []
    reader_errors = []
    original_rename_exchange = archive_module.rename_exchange

    def pause_atomic_exchange(left: Path, right: Path) -> None:
        exchange_ready.set()
        if not allow_exchange.wait(timeout=2):
            raise TimeoutError("test did not release atomic exchange")
        original_rename_exchange(left, right)

    def observe_latest() -> None:
        while not stop_reader.is_set():
            try:
                observation = read_latest_via_directory_descriptor(archive.latest_dir)
                observations.append(observation)
                if observation[0] == b"new audio":
                    new_snapshot_seen.set()
            except (OSError, ValueError) as exc:
                reader_errors.append(exc)
            reader_ready.set()

    monkeypatch.setattr(archive_module, "rename_exchange", pause_atomic_exchange)

    with ThreadPoolExecutor(max_workers=2) as executor:
        reader_future = executor.submit(observe_latest)
        assert reader_ready.wait(timeout=2)
        writer_future = executor.submit(archive.start_record, source_audio, stats=None)
        assert exchange_ready.wait(timeout=2)
        try:
            assert not reader_errors
            assert any(observation[0] == b"legacy audio" for observation in observations)
        finally:
            allow_exchange.set()
        record = writer_future.result(timeout=2)
        assert new_snapshot_seen.wait(timeout=2)
        stop_reader.set()
        reader_future.result(timeout=2)

    assert not reader_errors
    allowed_content = {
        (b"legacy audio", "legacy transcript", "legacy"),
        (b"new audio", "", record.record_id),
    }
    assert set(observations) <= allowed_content
    assert archive.latest_dir.is_symlink()
    assert_latest_matches_record(archive, record)


def test_canonical_file_lock_prevents_two_instances_from_publishing_older_last(
    tmp_path,
    monkeypatch,
):
    archive_root = tmp_path / "archive"
    older_audio = tmp_path / "older.wav"
    newer_audio = tmp_path / "newer.wav"
    older_audio.write_bytes(b"older audio")
    newer_audio.write_bytes(b"newer audio")
    older_archive = DictationArchive(root=archive_root)
    older_record = older_archive.start_record(
        older_audio,
        stats=None,
        created_at=datetime(2026, 7, 28, 20, 0, tzinfo=timezone.utc),
    )
    archive_alias = tmp_path / "archive_alias"
    archive_alias.symlink_to(archive_root, target_is_directory=True)
    newer_archive = DictationArchive(root=archive_alias)
    newer_record = start_record_without_latest(
        newer_archive,
        newer_audio,
        monkeypatch=monkeypatch,
        created_at=datetime(2026, 7, 28, 20, 1, tzinfo=timezone.utc),
    )
    older_snapshot_started = threading.Event()
    allow_older_snapshot = threading.Event()
    newer_refresh_started = threading.Event()
    newer_snapshot_started = threading.Event()
    original_older_snapshot = older_archive._create_latest_snapshot
    original_newer_snapshot = newer_archive._create_latest_snapshot

    def block_older_snapshot(record: DictationRecord) -> Path:
        older_snapshot_started.set()
        if not allow_older_snapshot.wait(timeout=2):
            raise TimeoutError("test did not release older snapshot")
        return original_older_snapshot(record)

    def track_newer_snapshot(record: DictationRecord) -> Path:
        newer_snapshot_started.set()
        return original_newer_snapshot(record)

    def refresh_newer_record() -> None:
        newer_refresh_started.set()
        newer_archive._refresh_latest(newer_record)

    monkeypatch.setattr(older_archive, "_create_latest_snapshot", block_older_snapshot)
    monkeypatch.setattr(newer_archive, "_create_latest_snapshot", track_newer_snapshot)

    with ThreadPoolExecutor(max_workers=2) as executor:
        older_future = executor.submit(older_archive._refresh_latest, older_record)
        assert older_snapshot_started.wait(timeout=2)
        newer_future = executor.submit(refresh_newer_record)
        assert newer_refresh_started.wait(timeout=2)
        try:
            assert not newer_snapshot_started.wait(timeout=0.2)
        finally:
            allow_older_snapshot.set()
        older_future.result(timeout=2)
        newer_future.result(timeout=2)

    assert older_archive.root.resolve() == newer_archive.root.resolve()
    assert newer_snapshot_started.is_set()
    assert_latest_matches_record(older_archive, newer_record)
    assert older_archive.latest_dir.resolve().is_dir()


def test_two_instance_cleanup_cannot_delete_new_live_snapshot(tmp_path, monkeypatch):
    archive_root = tmp_path / "archive"
    first_audio = tmp_path / "first.wav"
    second_audio = tmp_path / "second.wav"
    third_audio = tmp_path / "third.wav"
    first_audio.write_bytes(b"first audio")
    second_audio.write_bytes(b"second audio")
    third_audio.write_bytes(b"third audio")
    second_archive = DictationArchive(root=archive_root)
    third_archive = DictationArchive(root=archive_root)
    second_archive.start_record(
        first_audio,
        stats=None,
        created_at=datetime(2026, 7, 28, 20, 0, tzinfo=timezone.utc),
    )
    second_record = start_record_without_latest(
        second_archive,
        second_audio,
        monkeypatch=monkeypatch,
        created_at=datetime(2026, 7, 28, 20, 1, tzinfo=timezone.utc),
    )
    third_record = start_record_without_latest(
        third_archive,
        third_audio,
        monkeypatch=monkeypatch,
        created_at=datetime(2026, 7, 28, 20, 2, tzinfo=timezone.utc),
    )
    cleanup_started = threading.Event()
    allow_cleanup = threading.Event()
    third_refresh_started = threading.Event()
    third_snapshot_started = threading.Event()
    original_cleanup = second_archive._cleanup_obsolete_latest_snapshots
    original_third_snapshot = third_archive._create_latest_snapshot

    def pause_second_cleanup(*, previous_snapshot: Path | None) -> None:
        cleanup_started.set()
        if not allow_cleanup.wait(timeout=2):
            raise TimeoutError("test did not release second cleanup")
        original_cleanup(previous_snapshot=previous_snapshot)

    def track_third_snapshot(record: DictationRecord) -> Path:
        third_snapshot_started.set()
        return original_third_snapshot(record)

    def refresh_third_record() -> None:
        third_refresh_started.set()
        third_archive._refresh_latest(third_record)

    monkeypatch.setattr(
        second_archive,
        "_cleanup_obsolete_latest_snapshots",
        pause_second_cleanup,
    )
    monkeypatch.setattr(third_archive, "_create_latest_snapshot", track_third_snapshot)

    with ThreadPoolExecutor(max_workers=2) as executor:
        second_future = executor.submit(second_archive._refresh_latest, second_record)
        assert cleanup_started.wait(timeout=2)
        second_snapshot = second_archive.latest_dir.resolve()
        third_future = executor.submit(refresh_third_record)
        assert third_refresh_started.wait(timeout=2)
        try:
            assert not third_snapshot_started.wait(timeout=0.2)
            assert second_snapshot.is_dir()
        finally:
            allow_cleanup.set()
        second_future.result(timeout=2)
        third_future.result(timeout=2)

    live_snapshot = second_archive.latest_dir.resolve()
    assert third_snapshot_started.is_set()
    assert live_snapshot.is_dir()
    assert live_snapshot in snapshot_directories(second_archive)
    assert_latest_matches_record(second_archive, third_record)


def test_atomic_pointer_swap_exposes_only_complete_snapshots(tmp_path, monkeypatch):
    older_audio = tmp_path / "older.wav"
    newer_audio = tmp_path / "newer.wav"
    older_audio.write_bytes(b"older audio")
    newer_audio.write_bytes(b"newer audio")
    archive = DictationArchive(root=tmp_path / "archive")
    older_record = archive.start_record(
        older_audio,
        stats=None,
        created_at=datetime(2026, 7, 28, 20, 0, tzinfo=timezone.utc),
    )
    older_snapshot = archive.latest_dir.resolve()
    pointer_ready = threading.Event()
    allow_pointer_swap = threading.Event()
    pending_snapshot = []
    original_replace = archive_module.os.replace

    def pause_latest_pointer_swap(source: Path, destination: Path) -> None:
        if source.name.startswith(LATEST_POINTER_PREFIX) and destination == archive.latest_dir:
            pointer_target = Path(os.readlink(source))
            pending_snapshot.append(archive.root / pointer_target)
            pointer_ready.set()
            if not allow_pointer_swap.wait(timeout=2):
                raise TimeoutError("test did not release latest pointer swap")
        original_replace(source, destination)

    monkeypatch.setattr(archive_module.os, "replace", pause_latest_pointer_swap)

    with ThreadPoolExecutor(max_workers=1) as executor:
        newer_future = executor.submit(
            archive.start_record,
            newer_audio,
            stats=None,
            created_at=datetime(2026, 7, 28, 20, 1, tzinfo=timezone.utc),
        )
        assert pointer_ready.wait(timeout=2)
        try:
            assert archive.latest_dir.resolve() == older_snapshot
            assert_latest_matches_record(archive, older_record)
            assert_complete_snapshot(pending_snapshot[0])
        finally:
            allow_pointer_swap.set()
        newer_record = newer_future.result(timeout=2)

    assert archive.latest_dir.resolve() == pending_snapshot[0]
    assert_latest_matches_record(archive, newer_record)


def test_latest_snapshot_storage_keeps_only_current_and_previous_versions(tmp_path):
    source_audio = tmp_path / "capture.wav"
    source_audio.write_bytes(b"audio bytes")
    archive = DictationArchive(root=tmp_path / "archive")
    record = archive.start_record(source_audio, stats=None)

    archive.save_transcript(record, "first version")
    previous_snapshot = archive.latest_dir.resolve()
    archive.save_transcript(record, "second version")
    current_snapshot = archive.latest_dir.resolve()
    archive.save_transcript(record, "third version")

    retained_snapshots = snapshot_directories(archive)
    assert len(retained_snapshots) == 2
    assert archive.latest_dir.resolve() in retained_snapshots
    assert current_snapshot in retained_snapshots
    assert previous_snapshot not in retained_snapshots
    for snapshot in retained_snapshots:
        snapshot_audio = snapshot / AUDIO_FILE_NAME
        assert snapshot_audio.read_bytes() == record.audio_path.read_bytes()
        assert snapshot_audio.stat().st_ino != record.audio_path.stat().st_ino


def test_archive_never_changes_existing_permanent_records(tmp_path):
    archive_root = tmp_path / "archive"
    old_archive = archive_root / "old_archive_through_july_28th_2026"
    old_record = old_archive / "2026-01-01T00-00-00Z-old"
    old_record.mkdir(parents=True)
    old_audio = old_record / AUDIO_FILE_NAME
    old_transcript = old_record / TRANSCRIPT_FILE_NAME
    old_metadata = old_record / METADATA_FILE_NAME
    old_audio.write_bytes(b"old audio")
    old_transcript.write_text("old transcript", encoding="utf-8")
    old_metadata.write_text('{"status": "old"}\n', encoding="utf-8")
    source_audio = tmp_path / "capture.wav"
    source_audio.write_bytes(b"new audio")
    archive = DictationArchive(root=archive_root)

    record = archive.start_record(source_audio, stats=None)
    archive.save_transcript(record, "new transcript")

    assert old_audio.read_bytes() == b"old audio"
    assert old_transcript.read_text(encoding="utf-8") == "old transcript"
    assert old_metadata.read_text(encoding="utf-8") == '{"status": "old"}\n'
    assert not hasattr(archive, "prune_expired")


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


def assert_latest_matches_record(
    archive: DictationArchive,
    record: DictationRecord,
) -> None:
    assert (archive.latest_dir / AUDIO_FILE_NAME).read_bytes() == record.audio_path.read_bytes()
    assert (archive.latest_dir / TRANSCRIPT_FILE_NAME).read_text(
        encoding="utf-8"
    ) == record.transcript_path.read_text(encoding="utf-8")
    assert read_json(archive.latest_dir / METADATA_FILE_NAME) == read_json(record.metadata_path)


def assert_complete_snapshot(snapshot: Path) -> None:
    assert (snapshot / AUDIO_FILE_NAME).is_file()
    assert (snapshot / TRANSCRIPT_FILE_NAME).is_file()
    assert (snapshot / METADATA_FILE_NAME).is_file()


def snapshot_directories(archive: DictationArchive) -> list[Path]:
    snapshot_root = archive.root / LATEST_SNAPSHOT_ROOT_NAME
    if not snapshot_root.exists():
        return []
    return sorted(
        candidate
        for candidate in snapshot_root.iterdir()
        if candidate.is_dir() and candidate.name.startswith(LATEST_SNAPSHOT_PREFIX)
    )


def start_record_without_latest(
    archive: DictationArchive,
    audio_path: Path,
    *,
    monkeypatch: pytest.MonkeyPatch,
    created_at: datetime,
) -> DictationRecord:
    with monkeypatch.context() as scoped_monkeypatch:
        scoped_monkeypatch.setattr(archive, "_refresh_latest", lambda _record: None)
        return archive.start_record(
            audio_path,
            stats=None,
            created_at=created_at,
        )


def read_latest_via_directory_descriptor(latest_directory: Path) -> tuple[bytes, str, str]:
    directory_descriptor = os.open(
        latest_directory,
        os.O_RDONLY | os.O_DIRECTORY,
    )
    try:
        audio = read_file_from_directory(directory_descriptor, AUDIO_FILE_NAME, binary=True)
        transcript = read_file_from_directory(
            directory_descriptor,
            TRANSCRIPT_FILE_NAME,
            binary=False,
        )
        metadata_text = read_file_from_directory(
            directory_descriptor,
            METADATA_FILE_NAME,
            binary=False,
        )
    finally:
        os.close(directory_descriptor)
    metadata = json.loads(metadata_text)
    return audio, transcript, metadata["record_id"]


def read_file_from_directory(
    directory_descriptor: int,
    file_name: str,
    *,
    binary: bool,
):
    file_descriptor = os.open(file_name, os.O_RDONLY, dir_fd=directory_descriptor)
    mode = "rb" if binary else "r"
    encoding = None if binary else "utf-8"
    with os.fdopen(file_descriptor, mode, encoding=encoding) as file:
        return file.read()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))
