"""Durable local archive for wa_whisper dictation recovery.

Readers needing a coherent latest triplet should resolve the ``latest`` symlink once
and read all three files from that immutable target directory. The current and
previous targets remain available so a normal pointer swap cannot invalidate a
reader that already resolved the previous target. During the one-time conversion
of a concrete ``latest`` directory, a reader needing the same guarantee should
open that directory once and read the files relative to its directory descriptor.
"""

from __future__ import annotations

import contextlib
import ctypes
import errno
import fcntl
import json
import os
import secrets
import shutil
import tempfile
import threading
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping, Optional

from .recorder import RecorderStats

DEFAULT_ARCHIVE_ROOT = Path.home() / "Music" / "wa_whisper_recordings"
AUDIO_FILE_NAME = "audio.wav"
TRANSCRIPT_FILE_NAME = "transcript.txt"
METADATA_FILE_NAME = "metadata.json"
LATEST_DIR_NAME = "latest"
LATEST_SNAPSHOT_ROOT_NAME = ".latest_snapshots"
LATEST_SNAPSHOT_PREFIX = "snapshot_"
LATEST_POINTER_PREFIX = ".latest_pointer_"
LATEST_LOCK_FILE_NAME = ".latest.lock"
MAX_RECORD_ID_ATTEMPTS = 100
AT_FDCWD = -100
RENAME_EXCHANGE = 2


@dataclass(frozen=True, slots=True)
class DictationRecord:
    """Filesystem locations for one archived dictation."""

    record_id: str
    record_dir: Path
    audio_path: Path
    transcript_path: Path
    metadata_path: Path


class DictationArchive:
    """Save each dictation's original audio, transcript, and metadata."""

    def __init__(self, *, root: Path = DEFAULT_ARCHIVE_ROOT) -> None:
        self.root = root
        self._latest_lock = threading.RLock()

    @property
    def latest_dir(self) -> Path:
        return self.root / LATEST_DIR_NAME

    @property
    def _latest_snapshot_root(self) -> Path:
        return self.root / LATEST_SNAPSHOT_ROOT_NAME

    def start_record(
        self,
        audio_path: Path,
        *,
        stats: Optional[RecorderStats],
        backend: Optional[Mapping[str, Any]] = None,
        created_at: Optional[datetime] = None,
    ) -> DictationRecord:
        """Create an archive record and copy the original audio into it."""
        record_created_at = created_at or utc_now()
        require_aware_datetime(record_created_at)

        ensure_private_directory(self.root)
        record = self._create_record(record_created_at)

        atomic_copy(audio_path, record.audio_path)
        write_text_atomic(record.transcript_path, "")

        metadata = {
            "record_id": record.record_id,
            "created_at": format_timestamp(record_created_at),
            "updated_at": format_timestamp(record_created_at),
            "status": "audio_archived",
            "paths": {
                "audio": str(record.audio_path),
                "transcript": str(record.transcript_path),
                "metadata": str(record.metadata_path),
            },
            "source_audio_path": str(audio_path),
            "capture_stats": serialize_recorder_stats(stats),
            "backend": dict(backend or {}),
        }
        write_json_atomic(record.metadata_path, metadata)
        self._refresh_latest(record)
        return record

    def save_transcript(
        self,
        record: DictationRecord,
        text: str,
        *,
        whisper_info: Optional[Mapping[str, Any]] = None,
    ) -> None:
        """Persist the final post-processed transcript for a record."""
        write_text_atomic(record.transcript_path, text)
        self._update_permanent_metadata(
            record,
            status="transcribed",
            transcription={
                "text_chars": len(text),
                "whisper_info": dict(whisper_info or {}),
            },
        )
        self._refresh_latest(record)

    def update_record(self, record: DictationRecord, **updates: Any) -> None:
        """Merge metadata updates into a record and latest metadata."""
        self._update_permanent_metadata(record, **updates)
        self._refresh_latest(record)

    def _update_permanent_metadata(self, record: DictationRecord, **updates: Any) -> None:
        metadata = read_json(record.metadata_path)
        metadata.update(updates)
        metadata["updated_at"] = format_timestamp(utc_now())
        write_json_atomic(record.metadata_path, metadata)

    def _refresh_latest(self, record: DictationRecord) -> None:
        with self._latest_lock:
            try:
                with self._exclusive_latest_file_lock():
                    candidate_metadata = read_json(record.metadata_path)
                    if not self._should_refresh_latest(record, candidate_metadata):
                        return
                    self._publish_latest_snapshot(record)
            except (OSError, ValueError):
                return

    @contextlib.contextmanager
    def _exclusive_latest_file_lock(self) -> Iterator[None]:
        ensure_private_directory(self.root)
        canonical_root = self.root.resolve(strict=True)
        lock_path = canonical_root / LATEST_LOCK_FILE_NAME
        lock_descriptor = os.open(
            lock_path,
            os.O_CREAT | os.O_RDWR,
            0o600,
        )
        try:
            os.fchmod(lock_descriptor, 0o600)
            fcntl.flock(lock_descriptor, fcntl.LOCK_EX)
            yield
        finally:
            with contextlib.suppress(OSError):
                fcntl.flock(lock_descriptor, fcntl.LOCK_UN)
            os.close(lock_descriptor)

    def _should_refresh_latest(
        self,
        record: DictationRecord,
        candidate_metadata: Mapping[str, Any],
    ) -> bool:
        if candidate_metadata.get("record_id") != record.record_id:
            return False

        current_metadata = self._read_latest_metadata()
        if current_metadata.get("record_id") == record.record_id:
            return True

        candidate_order = latest_record_order(candidate_metadata)
        current_order = latest_record_order(current_metadata)
        return candidate_order is not None and (
            current_order is None or candidate_order > current_order
        )

    def _read_latest_metadata(self) -> dict[str, Any]:
        try:
            metadata = read_json(self.latest_dir / METADATA_FILE_NAME)
        except (OSError, ValueError):
            return {}
        return metadata if isinstance(metadata, dict) else {}

    def _publish_latest_snapshot(self, record: DictationRecord) -> None:
        snapshot_directory = self._create_latest_snapshot(record)
        previous_snapshot = self._current_latest_snapshot()
        try:
            migrated_snapshot = self._replace_latest_pointer(snapshot_directory)
        except OSError:
            shutil.rmtree(snapshot_directory, ignore_errors=True)
            raise
        if migrated_snapshot is not None:
            previous_snapshot = migrated_snapshot
        self._cleanup_obsolete_latest_snapshots(
            previous_snapshot=previous_snapshot,
        )

    def _create_latest_snapshot(self, record: DictationRecord) -> Path:
        ensure_private_directory(self._latest_snapshot_root)
        snapshot_directory = Path(
            tempfile.mkdtemp(
                dir=self._latest_snapshot_root,
                prefix=f"{LATEST_SNAPSHOT_PREFIX}{record.record_id}_",
            )
        )
        try:
            atomic_copy(record.audio_path, snapshot_directory / AUDIO_FILE_NAME)
            atomic_copy(record.transcript_path, snapshot_directory / TRANSCRIPT_FILE_NAME)
            atomic_copy(record.metadata_path, snapshot_directory / METADATA_FILE_NAME)
        except OSError:
            shutil.rmtree(snapshot_directory, ignore_errors=True)
            raise
        return snapshot_directory

    def _replace_latest_pointer(self, snapshot_directory: Path) -> Optional[Path]:
        pointer_path = self.root / f"{LATEST_POINTER_PREFIX}{secrets.token_hex(8)}"
        pointer_target = snapshot_directory.relative_to(self.root)
        os.symlink(pointer_target, pointer_path, target_is_directory=True)
        migrated_snapshot: Optional[Path] = None
        try:
            if self.latest_dir.is_symlink() or not self.latest_dir.exists():
                os.replace(pointer_path, self.latest_dir)
            elif self.latest_dir.is_dir():
                migrated_snapshot = self._migrate_concrete_latest(pointer_path)
            else:
                raise FileExistsError(
                    f"Cannot replace non-directory latest path: {self.latest_dir}"
                )
        finally:
            with contextlib.suppress(OSError):
                pointer_path.unlink()
        return migrated_snapshot

    def _migrate_concrete_latest(self, pointer_path: Path) -> Path:
        rename_exchange(pointer_path, self.latest_dir)
        migrated_snapshot = self._latest_snapshot_root / (
            f"{LATEST_SNAPSHOT_PREFIX}migrated_{secrets.token_hex(8)}"
        )
        try:
            os.replace(pointer_path, migrated_snapshot)
        except OSError:
            return pointer_path
        return migrated_snapshot

    def _current_latest_snapshot(self) -> Optional[Path]:
        if not self.latest_dir.is_symlink():
            return None
        try:
            pointer_target = Path(os.readlink(self.latest_dir))
        except OSError:
            return None
        candidate = pointer_target if pointer_target.is_absolute() else self.root / pointer_target
        try:
            relative_candidate = candidate.relative_to(self._latest_snapshot_root)
        except ValueError:
            return None
        if len(relative_candidate.parts) != 1:
            return None
        if not relative_candidate.name.startswith(LATEST_SNAPSHOT_PREFIX):
            return None
        return candidate

    def _cleanup_obsolete_latest_snapshots(
        self,
        *,
        previous_snapshot: Optional[Path],
    ) -> None:
        live_snapshot = self._current_latest_snapshot()
        if live_snapshot is None:
            return
        snapshots_to_keep = {live_snapshot, previous_snapshot}
        with contextlib.suppress(OSError):
            for candidate in self._latest_snapshot_root.iterdir():
                if candidate in snapshots_to_keep:
                    continue
                if not candidate.name.startswith(LATEST_SNAPSHOT_PREFIX):
                    continue
                remove_derived_path(candidate)
        with contextlib.suppress(OSError):
            for candidate in self.root.iterdir():
                if candidate in snapshots_to_keep:
                    continue
                if candidate.name.startswith(LATEST_POINTER_PREFIX):
                    remove_derived_path(candidate)

    def _create_record(self, created_at: datetime) -> DictationRecord:
        local_created_at = created_at.astimezone()
        date_dir = self.root / local_created_at.strftime("%Y-%m-%d")
        ensure_private_directory(date_dir)

        for _attempt in range(MAX_RECORD_ID_ATTEMPTS):
            record = self._build_record(local_created_at, date_dir)
            try:
                record.record_dir.mkdir(mode=0o700, exist_ok=False)
            except FileExistsError:
                continue
            ensure_private_directory(record.record_dir)
            return record

        raise RuntimeError("Unable to allocate a unique dictation record ID")

    @staticmethod
    def _build_record(created_at: datetime, date_dir: Path) -> DictationRecord:
        record_id = format_record_id(created_at, uuid.uuid4().hex[:8])
        record_dir = date_dir / record_id
        return DictationRecord(
            record_id=record_id,
            record_dir=record_dir,
            audio_path=record_dir / AUDIO_FILE_NAME,
            transcript_path=record_dir / TRANSCRIPT_FILE_NAME,
            metadata_path=record_dir / METADATA_FILE_NAME,
        )


def require_aware_datetime(value: datetime) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("created_at must include timezone information")


def format_record_id(created_at: datetime, unique_suffix: str) -> str:
    hour_12 = created_at.hour % 12 or 12
    meridiem = "AM" if created_at.hour < 12 else "PM"
    timezone_name = sanitize_timezone_name(created_at.tzname())
    return (
        f"{created_at:%Y-%m-%d}_at_{hour_12:02d}-{created_at:%M-%S}_"
        f"{meridiem}_{timezone_name}_{unique_suffix}"
    )


def sanitize_timezone_name(timezone_name: Optional[str]) -> str:
    safe_name = "".join(
        character if character.isalnum() else "-"
        for character in (timezone_name or "UTC")
    ).strip("-")
    return safe_name or "UTC"


def serialize_recorder_stats(stats: Optional[RecorderStats]) -> Optional[dict[str, Any]]:
    if stats is None:
        return None
    serialized = asdict(stats)
    serialized["speech_ratio"] = stats.speech_ratio
    serialized["speech_max_db"] = stats.speech_max_db
    serialized["silence_avg_db"] = stats.silence_avg_db
    return serialized


def latest_record_order(metadata: Mapping[str, Any]) -> Optional[tuple[datetime, str]]:
    created_at = metadata.get("created_at")
    record_id = metadata.get("record_id")
    parsed_created_at = parse_utc_timestamp(created_at)
    if parsed_created_at is None or not isinstance(record_id, str):
        return None
    if not record_id:
        return None
    return parsed_created_at, record_id


def parse_utc_timestamp(value: object) -> Optional[datetime]:
    if not isinstance(value, str) or not value:
        return None
    normalized_value = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed_value = datetime.fromisoformat(normalized_value)
    except ValueError:
        return None
    if parsed_value.tzinfo is None or parsed_value.utcoffset() is None:
        return None
    return parsed_value.astimezone(timezone.utc)


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_json_atomic(path: Path, data: Mapping[str, Any]) -> None:
    write_text_atomic(path, json.dumps(data, indent=2, sort_keys=True) + "\n")


def write_text_atomic(path: Path, text: str) -> None:
    ensure_private_directory(path.parent)
    temp_path: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temp_file:
            temp_file.write(text)
            temp_path = Path(temp_file.name)
        os.replace(temp_path, path)
        temp_path = None
    finally:
        if temp_path:
            temp_path.unlink(missing_ok=True)


def atomic_copy(source: Path, destination: Path) -> None:
    ensure_private_directory(destination.parent)
    with tempfile.NamedTemporaryFile(
        "wb",
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
        delete=False,
    ) as temp_file:
        temp_path = Path(temp_file.name)
    try:
        shutil.copy2(source, temp_path)
        temp_path.chmod(0o600)
        os.replace(temp_path, destination)
    finally:
        temp_path.unlink(missing_ok=True)


def rename_exchange(left: Path, right: Path) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise OSError(errno.ENOSYS, "renameat2 is unavailable")
    renameat2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    result = renameat2(
        AT_FDCWD,
        os.fsencode(left),
        AT_FDCWD,
        os.fsencode(right),
        RENAME_EXCHANGE,
    )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    raise OSError(
        error_number,
        os.strerror(error_number),
        f"{left} <-> {right}",
    )


def remove_derived_path(path: Path) -> None:
    if path.is_symlink():
        path.unlink(missing_ok=True)
    elif path.is_dir():
        shutil.rmtree(path, ignore_errors=True)


def ensure_private_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    with contextlib.suppress(OSError):
        path.chmod(0o700)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def format_timestamp(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat(timespec="auto").replace("+00:00", "Z")
