import subprocess

from wa_whisper import mode_cli


def test_mode_cli_ram_writes_mode_restarts_service_and_reports_state(tmp_path, monkeypatch, capsys):
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        if cmd[:3] == ["systemctl", "--user", "is-active"]:
            return subprocess.CompletedProcess(cmd, 0, stdout="active\n")
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(mode_cli.subprocess, "run", fake_run)

    exit_code = mode_cli.main(["ram", "--config-dir", str(tmp_path)])

    assert exit_code == 0
    assert (tmp_path / "compute_mode").read_text(encoding="utf-8") == "ram\n"
    assert calls[0][0] == ["systemctl", "--user", "restart", "wa-whisper-ptt.service"]
    assert calls[1][0] == ["systemctl", "--user", "is-active", "wa-whisper-ptt.service"]
    assert "compute mode: ram" in capsys.readouterr().out


def test_mode_cli_gpu_writes_mode_restarts_service(tmp_path, monkeypatch):
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout="active\n")

    monkeypatch.setattr(mode_cli.subprocess, "run", fake_run)

    exit_code = mode_cli.main(["gpu", "--config-dir", str(tmp_path)])

    assert exit_code == 0
    assert (tmp_path / "compute_mode").read_text(encoding="utf-8") == "gpu\n"
    assert ["systemctl", "--user", "restart", "wa-whisper-ptt.service"] in calls


def test_mode_cli_status_reads_mode_without_restarting_service(tmp_path, monkeypatch, capsys):
    calls = []
    (tmp_path / "compute_mode").write_text("ram\n", encoding="utf-8")

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 3, stdout="inactive\n")

    monkeypatch.setattr(mode_cli.subprocess, "run", fake_run)

    exit_code = mode_cli.main(["status", "--config-dir", str(tmp_path)])

    assert exit_code == 0
    assert calls == [["systemctl", "--user", "is-active", "wa-whisper-ptt.service"]]
    output = capsys.readouterr().out
    assert "compute mode: ram" in output
    assert "service: inactive" in output
