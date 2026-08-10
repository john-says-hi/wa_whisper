import subprocess
from pathlib import Path

from wa_whisper import audio_cues


def configure_available_cue(monkeypatch, tmp_path: Path) -> tuple[Path, Path]:
    player = tmp_path / "paplay"
    sound = tmp_path / "bell.oga"
    player.touch()
    sound.touch()
    monkeypatch.setattr(audio_cues, "SYSTEM_BELL_PLAYER", player)
    monkeypatch.setattr(audio_cues, "SYSTEM_BELL_SOUND", sound)
    return player, sound


def test_system_bell_uses_expected_player_sound_and_timeout(monkeypatch, tmp_path):
    assert audio_cues.SYSTEM_BELL_PLAYER == Path("/usr/bin/paplay")
    assert audio_cues.SYSTEM_BELL_SOUND == Path("/usr/share/sounds/freedesktop/stereo/bell.oga")

    player, sound = configure_available_cue(monkeypatch, tmp_path)
    calls = []
    monkeypatch.setattr(audio_cues.subprocess, "run", lambda *args, **kwargs: calls.append((args, kwargs)))

    audio_cues.play_system_bell(tmp_path / "audio.log", purpose="completion")

    assert calls == [
        (
            ([str(player), str(sound)],),
            {
                "check": True,
                "stdout": subprocess.DEVNULL,
                "stderr": subprocess.PIPE,
                "text": True,
                "timeout": 1.0,
            },
        )
    ]


def test_system_bell_logs_missing_player_without_running(monkeypatch, tmp_path):
    missing_player = tmp_path / "missing-paplay"
    sound = tmp_path / "bell.oga"
    sound.touch()
    monkeypatch.setattr(audio_cues, "SYSTEM_BELL_PLAYER", missing_player)
    monkeypatch.setattr(audio_cues, "SYSTEM_BELL_SOUND", sound)
    monkeypatch.setattr(
        audio_cues.subprocess,
        "run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("player must not run")),
    )

    log_path = tmp_path / "audio.log"
    audio_cues.play_system_bell(log_path, purpose="hands-free on")

    log_text = log_path.read_text(encoding="utf-8")
    assert "hands-free on" in log_text
    assert f"player missing at {missing_player}" in log_text


def test_system_bell_logs_missing_sound_without_running(monkeypatch, tmp_path):
    player = tmp_path / "paplay"
    missing_sound = tmp_path / "missing-bell.oga"
    player.touch()
    monkeypatch.setattr(audio_cues, "SYSTEM_BELL_PLAYER", player)
    monkeypatch.setattr(audio_cues, "SYSTEM_BELL_SOUND", missing_sound)
    monkeypatch.setattr(
        audio_cues.subprocess,
        "run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("player must not run")),
    )

    log_path = tmp_path / "audio.log"
    audio_cues.play_system_bell(log_path, purpose="hands-free off")

    log_text = log_path.read_text(encoding="utf-8")
    assert "hands-free off" in log_text
    assert f"sound missing at {missing_sound}" in log_text


def test_system_bell_logs_called_process_error(monkeypatch, tmp_path):
    configure_available_cue(monkeypatch, tmp_path)

    def fail_playback(*_args, **_kwargs):
        raise subprocess.CalledProcessError(7, "paplay", stderr=b"pulse unavailable")

    monkeypatch.setattr(audio_cues.subprocess, "run", fail_playback)
    log_path = tmp_path / "audio.log"

    audio_cues.play_system_bell(log_path, purpose="completion")

    log_text = log_path.read_text(encoding="utf-8")
    assert "completion" in log_text
    assert "exit code 7" in log_text
    assert "pulse unavailable" in log_text


def test_system_bell_logs_timeout(monkeypatch, tmp_path):
    configure_available_cue(monkeypatch, tmp_path)

    def time_out(*_args, **_kwargs):
        raise subprocess.TimeoutExpired("paplay", 1.0)

    monkeypatch.setattr(audio_cues.subprocess, "run", time_out)
    log_path = tmp_path / "audio.log"

    audio_cues.play_system_bell(log_path, purpose="hands-free off")

    log_text = log_path.read_text(encoding="utf-8")
    assert "hands-free off" in log_text
    assert "timed out after 1.0s" in log_text


def test_system_bell_logs_os_error(monkeypatch, tmp_path):
    configure_available_cue(monkeypatch, tmp_path)

    def fail_playback(*_args, **_kwargs):
        raise OSError("audio service unavailable")

    monkeypatch.setattr(audio_cues.subprocess, "run", fail_playback)
    log_path = tmp_path / "audio.log"

    audio_cues.play_system_bell(log_path, purpose="hands-free on")

    log_text = log_path.read_text(encoding="utf-8")
    assert "hands-free on" in log_text
    assert "audio service unavailable" in log_text


def test_system_bell_never_raises_when_logging_fails(monkeypatch, tmp_path):
    configure_available_cue(monkeypatch, tmp_path)
    monkeypatch.setattr(
        audio_cues.subprocess,
        "run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("playback failed")),
    )
    monkeypatch.setattr(
        audio_cues,
        "write_log",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("log failed")),
    )

    audio_cues.play_system_bell(tmp_path / "audio.log", purpose="completion")
