"""Durable local archive for wa_whisper dictation recovery."""

from __future__ import annotations

import contextlib
import json
import os
import shutil
import tempfile
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Optional

from .recorder import RecorderStats

DEFAULT_ARCHIVE_ROOT = Path.home() / ".local" / "share" / "wa_whisper" / "dictations"
DEFAULT_RETENTION_DAYS = 90
AUDIO_FILE_NAME = "audio.wav"
TRANSCRIPT_FILE_NAME = "transcript.txt"
METADATA_FILE_NAME = "metadata.json"
LATEST_DIR_NAME = "latest"


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

    def __init__(
        self,
        *,
        root: Path = DEFAULT_ARCHIVE_ROOT,
        retention_days: int = DEFAULT_RETENTION_DAYS,
    ) -> None:
        if retention_days <= 0:
            raise ValueError("retention_days must be positive")
        self.root = root
        self.retention_days = retention_days

    @property
    def latest_dir(self) -> Path:
        return self.root / LATEST_DIR_NAME

    def start_record(
        self,
        audio_path: Path,
        *,
        stats: Optional[RecorderStats],
        backend: Optional[Mapping[str, Any]] = None,
    ) -> DictationRecord:
        """Create an archive record and copy the original audio into it."""
        created_at = utc_now()
        record = self._build_record(created_at)
        ensure_private_directory(self.root)
        ensure_private_directory(record.record_dir.parent)
        record.record_dir.mkdir(mode=0o700, exist_ok=False)
        ensure_private_directory(record.record_dir)

        atomic_copy(audio_path, record.audio_path)
        clear_latest_transcript(self.latest_dir)
        atomic_copy(record.audio_path, self.latest_dir / AUDIO_FILE_NAME)

        metadata = {
            "record_id": record.record_id,
            "created_at": format_timestamp(created_at),
            "updated_at": format_timestamp(created_at),
            "status": "audio_archived",
            "retention_days": self.retention_days,
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
        write_json_atomic(self.latest_dir / METADATA_FILE_NAME, metadata)
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
        write_text_atomic(self.latest_dir / TRANSCRIPT_FILE_NAME, text)
        self.update_record(
            record,
            status="transcribed",
            transcription={
                "text_chars": len(text),
                "whisper_info": dict(whisper_info or {}),
            },
        )

    def update_record(self, record: DictationRecord, **updates: Any) -> None:
        """Merge metadata updates into a record and latest metadata."""
        metadata = read_json(record.metadata_path)
        metadata.update(updates)
        metadata["updated_at"] = format_timestamp(utc_now())
        write_json_atomic(record.metadata_path, metadata)
        write_json_atomic(self.latest_dir / METADATA_FILE_NAME, metadata)

    def prune_expired(self) -> int:
        """Delete records older than the configured retention period."""
        cutoff = utc_now() - timedelta(days=self.retention_days)
        removed = 0
        if not self.root.exists():
            return removed

        for record_dir in self._iter_record_dirs():
            if record_created_before(record_dir, cutoff):
                shutil.rmtree(record_dir, ignore_errors=True)
                removed += 1

        for child in self.root.iterdir():
            if child.is_dir() and child.name != LATEST_DIR_NAME and not any(child.iterdir()):
                child.rmdir()
        prune_latest_if_expired(self.latest_dir, cutoff)
        return removed

    def _build_record(self, created_at: datetime) -> DictationRecord:
        record_id = f"{created_at.strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"
        record_dir = self.root / created_at.strftime("%Y-%m-%d") / record_id
        return DictationRecord(
            record_id=record_id,
            record_dir=record_dir,
            audio_path=record_dir / AUDIO_FILE_NAME,
            transcript_path=record_dir / TRANSCRIPT_FILE_NAME,
            metadata_path=record_dir / METADATA_FILE_NAME,
        )

    def _iter_record_dirs(self) -> list[Path]:
        record_dirs: list[Path] = []
        for child in self.root.iterdir():
            if child.name == LATEST_DIR_NAME or not child.is_dir():
                continue
            if (child / METADATA_FILE_NAME).exists():
                record_dirs.append(child)
                continue
            record_dirs.extend(
                grandchild
                for grandchild in child.iterdir()
                if grandchild.is_dir() and (grandchild / METADATA_FILE_NAME).exists()
            )
        return record_dirs


def serialize_recorder_stats(stats: Optional[RecorderStats]) -> Optional[dict[str, Any]]:
    if stats is None:
        return None
    serialized = asdict(stats)
    serialized["speech_ratio"] = stats.speech_ratio
    serialized["speech_max_db"] = stats.speech_max_db
    serialized["silence_avg_db"] = stats.silence_avg_db
    return serialized


def record_created_before(record_dir: Path, cutoff: datetime) -> bool:
    metadata = read_json(record_dir / METADATA_FILE_NAME)
    created_at = parse_timestamp(metadata.get("created_at"))
    if created_at is None:
        created_at = datetime.fromtimestamp(record_dir.stat().st_mtime, tz=timezone.utc)
    return created_at < cutoff


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
        os.replace(temp_path, destination)
    finally:
        temp_path.unlink(missing_ok=True)


def clear_latest_transcript(latest_dir: Path) -> None:
    ensure_private_directory(latest_dir)
    (latest_dir / TRANSCRIPT_FILE_NAME).unlink(missing_ok=True)


def ensure_private_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    with contextlib.suppress(OSError):
        path.chmod(0o700)


def prune_latest_if_expired(latest_dir: Path, cutoff: datetime) -> None:
    if not latest_dir.exists():
        return
    metadata = read_json(latest_dir / METADATA_FILE_NAME)
    created_at = parse_timestamp(metadata.get("created_at"))
    if created_at is None or created_at >= cutoff:
        return
    for child in latest_dir.iterdir():
        if child.is_dir():
            shutil.rmtree(child, ignore_errors=True)
        else:
            child.unlink(missing_ok=True)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def format_timestamp(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def parse_timestamp(value: object) -> Optional[datetime]:
    if not isinstance(value, str) or not value:
        return None
    normalized = value.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)
