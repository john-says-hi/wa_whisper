import subprocess
from pathlib import Path

from wa_whisper import recovery_queue


def test_insert_transcript_into_recovery_queue_uses_copyq_insert_row_one(tmp_path, monkeypatch):
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        return subprocess.CompletedProcess(cmd, 0, stderr="")

    monkeypatch.setattr(recovery_queue.subprocess, "run", fake_run)

    result = recovery_queue.insert_transcript_into_recovery_queue(
        "final transcript",
        tmp_path / "log.txt",
        copyq_bin=Path("/usr/bin/copyq"),
    )

    assert result.inserted is True
    assert result.clipboard_metadata() == {"active_clipboard_modified": False}
    assert result.recovery_queue_metadata() == {"provider": "copyq", "inserted": True, "row": 1}
    assert calls == [
        (
            ["/usr/bin/copyq", "insert", "1", "--", "final transcript"],
            {
                "check": True,
                "stdout": subprocess.DEVNULL,
                "stderr": subprocess.PIPE,
                "text": True,
                "timeout": 1.0,
            },
        )
    ]
    assert "Transcript inserted into CopyQ history row 1" in (tmp_path / "log.txt").read_text(
        encoding="utf-8"
    )


def test_insert_transcript_into_recovery_queue_passes_multiline_dash_text_literally(
    tmp_path, monkeypatch
):
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        return subprocess.CompletedProcess(cmd, 0, stderr="")

    monkeypatch.setattr(recovery_queue.subprocess, "run", fake_run)
    text = "--flag-like text\nsecond line\twith tab"

    result = recovery_queue.insert_transcript_into_recovery_queue(
        text,
        tmp_path / "log.txt",
        copyq_bin=Path("/usr/bin/copyq"),
    )

    assert result.inserted is True
    assert calls[0][0] == ["/usr/bin/copyq", "insert", "1", "--", text]


def test_insert_transcript_into_recovery_queue_handles_missing_copyq(tmp_path, monkeypatch):
    monkeypatch.setattr(recovery_queue, "resolve_copyq_path", lambda: None)

    result = recovery_queue.insert_transcript_into_recovery_queue("hello", tmp_path / "log.txt")

    assert result.recovery_queue_metadata() == {
        "provider": "copyq",
        "inserted": False,
        "row": 1,
        "error": "copyq not found",
    }
    assert result.clipboard_metadata() == {"active_clipboard_modified": False}


def test_insert_transcript_into_recovery_queue_handles_copyq_failure(tmp_path, monkeypatch):
    def fake_run(*_args, **_kwargs):
        raise subprocess.CalledProcessError(7, "copyq", stderr="server unavailable")

    monkeypatch.setattr(recovery_queue.subprocess, "run", fake_run)

    result = recovery_queue.insert_transcript_into_recovery_queue(
        "hello",
        tmp_path / "log.txt",
        copyq_bin=Path("/usr/bin/copyq"),
    )

    assert result.inserted is False
    assert result.error == "copyq insert failed with exit code 7: server unavailable"


def test_insert_transcript_into_recovery_queue_handles_timeout(tmp_path, monkeypatch):
    def fake_run(*_args, **_kwargs):
        raise subprocess.TimeoutExpired("copyq", 1.0)

    monkeypatch.setattr(recovery_queue.subprocess, "run", fake_run)

    result = recovery_queue.insert_transcript_into_recovery_queue(
        "hello",
        tmp_path / "log.txt",
        copyq_bin=Path("/usr/bin/copyq"),
    )

    assert result.inserted is False
    assert result.error == "copyq insert timed out after 1.0s"
