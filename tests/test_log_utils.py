from pathlib import Path

import pytest

from wa_whisper import log_utils


def test_write_log_appends_timestamped_message(tmp_path):
    log_path = tmp_path / "nested" / "runtime.log"

    log_utils.write_log("capture started", log_path)

    log_line = log_path.read_text(encoding="utf-8").strip()
    assert log_line.endswith("Z capture started")


def test_write_log_ignores_unusable_log_path(tmp_path):
    blocking_file = tmp_path / "not_a_directory"
    blocking_file.write_text("blocked", encoding="utf-8")

    log_utils.write_log("capture started", blocking_file / "runtime.log")

    assert blocking_file.read_text(encoding="utf-8") == "blocked"


@pytest.mark.parametrize("control_error", [KeyboardInterrupt(), SystemExit(2)])
def test_write_log_does_not_suppress_process_control_exceptions(
    monkeypatch,
    control_error,
):
    def fail_log_setup(_log_path: Path) -> Path:
        raise control_error

    monkeypatch.setattr(log_utils, "ensure_log_path", fail_log_setup)

    with pytest.raises(type(control_error)):
        log_utils.write_log("capture started")
