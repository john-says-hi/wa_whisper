import json
import importlib
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

main_mod = importlib.import_module("wa_whisper.main")


class FakeOrcaStream:
    def __init__(self, responses):
        self.frames = []
        self._responses = [json.dumps(response).encode("utf-8") + b"\n" for response in responses]

    def write(self, data):
        self.frames.append(json.loads(data.decode("utf-8")))

    def flush(self):
        return None

    def readline(self):
        return self._responses.pop(0)


class FakeOrcaSocket:
    def __init__(self, stream):
        self.stream = stream
        self.timeout = None
        self.connected_to = None

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def settimeout(self, timeout):
        self.timeout = timeout

    def connect(self, path):
        self.connected_to = path

    def makefile(self, _mode):
        return self.stream


def test_parser_defaults_to_stable_xdotool_mode():
    args = main_mod.build_arg_parser().parse_args([])

    assert args.injection_mode == main_mod.InjectionMode.XDOTOOL_TYPE.value


def test_parser_accepts_auto_injection_mode():
    args = main_mod.build_arg_parser().parse_args(["--injection-mode", "auto"])

    assert args.injection_mode == main_mod.InjectionMode.AUTO.value


def test_parser_accepts_compute_mode():
    args = main_mod.build_arg_parser().parse_args(["--compute-mode", "ram"])

    assert args.compute_mode == "ram"


def test_parser_rejects_device_and_compute_mode():
    with pytest.raises(SystemExit):
        main_mod.build_arg_parser().parse_args(["--device", "cpu", "--compute-mode", "ram"])


def test_default_injection_uses_xdotool_type_and_beep(tmp_path, monkeypatch):
    calls = []
    events = []

    def fake_run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(main_mod.subprocess, "run", fake_run)
    monkeypatch.setattr(main_mod, "play_completion_beep", lambda *_args: events.append("beep"))

    delivered = main_mod.inject_text(
        "hello",
        Path("/usr/bin/xdotool"),
        tmp_path / "log.txt",
        enable_beep=True,
    )

    assert delivered is True
    assert calls[0][0] == ["/usr/bin/xdotool", "type", "--clearmodifiers", "hello"]
    assert events == ["beep"]


def test_orca_daemon_mode_does_not_fallback_to_xdotool(tmp_path, monkeypatch):
    def fail_xdotool(*_args, **_kwargs):
        raise AssertionError("orca-daemon mode must not fall back to xdotool")

    monkeypatch.setattr(main_mod, "send_text_to_orca_daemon", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(main_mod, "type_text_with_xdotool", fail_xdotool)

    delivered = main_mod.inject_text(
        "hello",
        Path("/usr/bin/xdotool"),
        tmp_path / "log.txt",
        enable_beep=False,
        injection_mode=main_mod.InjectionMode.ORCA_DAEMON,
    )

    assert delivered is False
    assert "skipped xdotool fallback" in (tmp_path / "log.txt").read_text(encoding="utf-8")


def test_auto_injection_routes_warp_to_xdotool(tmp_path, monkeypatch):
    calls = []
    events = []

    def fake_xdotool(text, xdotool_bin, log_path):
        calls.append((text, xdotool_bin, log_path))
        return True

    def fail_orca(*_args, **_kwargs):
        raise AssertionError("Warp auto mode must not use Orca daemon")

    monkeypatch.setattr(
        main_mod,
        "get_active_window_info",
        lambda *_args: main_mod.ActiveWindowInfo(
            window_id="123",
            wm_classes=("dev.warp.Warp", "dev.warp.Warp"),
            name="wise_apple",
            pid=7302,
            process_args="warp-terminal",
        ),
    )
    monkeypatch.setattr(main_mod, "type_text_with_xdotool", fake_xdotool)
    monkeypatch.setattr(main_mod, "send_text_to_orca_daemon", fail_orca)
    monkeypatch.setattr(main_mod, "play_completion_beep", lambda *_args: events.append("beep"))

    delivered = main_mod.inject_text(
        "hello warp",
        Path("/usr/bin/xdotool"),
        tmp_path / "log.txt",
        enable_beep=True,
        injection_mode=main_mod.InjectionMode.AUTO,
    )

    assert delivered is True
    assert calls == [("hello warp", Path("/usr/bin/xdotool"), tmp_path / "log.txt")]
    assert events == ["beep"]
    assert "Warp" in (tmp_path / "log.txt").read_text(encoding="utf-8")


def test_auto_injection_routes_orca_to_focused_clipboard_paste(tmp_path, monkeypatch):
    paste_calls = []
    events = []

    def fail_xdotool(*_args, **_kwargs):
        raise AssertionError("Orca auto mode must not use xdotool")

    def fail_orca(*_args, **_kwargs):
        raise AssertionError("Orca auto mode should try focused clipboard paste first")

    monkeypatch.setattr(
        main_mod,
        "get_active_window_info",
        lambda *_args: main_mod.ActiveWindowInfo(
            window_id="456",
            wm_classes=("orca", "Orca"),
            name="Orca",
            pid=8377,
            process_args="orca",
        ),
    )
    monkeypatch.setattr(main_mod, "type_text_with_xdotool", fail_xdotool)
    monkeypatch.setattr(
        main_mod,
        "paste_text_with_clipboard_shortcut",
        lambda *args: paste_calls.append(args) or True,
    )
    monkeypatch.setattr(main_mod, "send_text_to_orca_daemon", fail_orca)
    monkeypatch.setattr(main_mod, "play_completion_beep", lambda *_args: events.append("beep"))

    delivered = main_mod.inject_text(
        "hello orca",
        Path("/usr/bin/xdotool"),
        tmp_path / "log.txt",
        enable_beep=True,
        injection_mode=main_mod.InjectionMode.AUTO,
        orca_daemon_dir=tmp_path,
        orca_session_id="session-1",
    )

    assert delivered is True
    assert paste_calls == [("hello orca", Path("/usr/bin/xdotool"), tmp_path / "log.txt")]
    assert events == ["beep"]
    assert "selected orca-clipboard-paste" in (tmp_path / "log.txt").read_text(encoding="utf-8")


def test_auto_injection_orca_clipboard_failure_falls_back_to_daemon(tmp_path, monkeypatch):
    daemon_calls = []

    def fail_xdotool(*_args, **_kwargs):
        raise AssertionError("failed Orca clipboard paste must not fall back to xdotool typing")

    def fake_orca(text, log_path, **kwargs):
        daemon_calls.append((text, log_path, kwargs))
        return True

    monkeypatch.setattr(
        main_mod,
        "get_active_window_info",
        lambda *_args: main_mod.ActiveWindowInfo(
            window_id="456",
            wm_classes=("orca", "Orca"),
            name="Orca",
            pid=8377,
            process_args="orca",
        ),
    )
    monkeypatch.setattr(main_mod, "type_text_with_xdotool", fail_xdotool)
    monkeypatch.setattr(main_mod, "paste_text_with_clipboard_shortcut", lambda *_args: False)
    monkeypatch.setattr(main_mod, "send_text_to_orca_daemon", fake_orca)

    delivered = main_mod.inject_text(
        "hello orca",
        Path("/usr/bin/xdotool"),
        tmp_path / "log.txt",
        enable_beep=False,
        injection_mode=main_mod.InjectionMode.AUTO,
        orca_daemon_dir=tmp_path,
        orca_session_id="session-1",
    )

    assert delivered is True
    assert daemon_calls == [
        (
            "hello orca",
            tmp_path / "log.txt",
            {"daemon_dir": tmp_path, "preferred_session_id": "session-1"},
        )
    ]
    log_text = (tmp_path / "log.txt").read_text(encoding="utf-8")
    assert "focused clipboard paste failed; falling back to daemon" in log_text


def test_auto_injection_orca_failure_does_not_fallback_to_xdotool(tmp_path, monkeypatch):
    def fail_xdotool(*_args, **_kwargs):
        raise AssertionError("failed Orca auto mode must not fall back to xdotool typing")

    monkeypatch.setattr(
        main_mod,
        "get_active_window_info",
        lambda *_args: main_mod.ActiveWindowInfo(
            window_id="456",
            wm_classes=("orca", "Orca"),
            name="Orca",
            pid=8377,
            process_args="orca",
        ),
    )
    monkeypatch.setattr(main_mod, "type_text_with_xdotool", fail_xdotool)
    monkeypatch.setattr(main_mod, "paste_text_with_clipboard_shortcut", lambda *_args: False)
    monkeypatch.setattr(main_mod, "send_text_to_orca_daemon", lambda *_args, **_kwargs: False)

    delivered = main_mod.inject_text(
        "hello orca",
        Path("/usr/bin/xdotool"),
        tmp_path / "log.txt",
        enable_beep=False,
        injection_mode=main_mod.InjectionMode.AUTO,
    )

    assert delivered is False
    assert "skipped xdotool typing fallback" in (tmp_path / "log.txt").read_text(encoding="utf-8")


def test_clipboard_paste_shortcut_restores_previous_clipboard(tmp_path, monkeypatch):
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        if cmd == ["/usr/bin/xclip", "-selection", "clipboard", "-out"]:
            return subprocess.CompletedProcess(cmd, 0, stdout=b"old clipboard")
        return subprocess.CompletedProcess(cmd, 0, stdout=b"")

    monkeypatch.setattr(main_mod.subprocess, "run", fake_run)
    monkeypatch.setattr(main_mod.time, "sleep", lambda *_args: None)

    delivered = main_mod.paste_text_with_clipboard_shortcut(
        "hello orca",
        Path("/usr/bin/xdotool"),
        tmp_path / "log.txt",
        clipboard_bin=Path("/usr/bin/xclip"),
    )

    assert delivered is True
    assert calls[0][0] == ["/usr/bin/xclip", "-selection", "clipboard", "-out"]
    assert calls[1][0] == ["/usr/bin/xclip", "-selection", "clipboard", "-in"]
    assert calls[1][1]["input"] == b"hello orca"
    assert calls[2][0] == [
        "/usr/bin/xdotool",
        "key",
        "--clearmodifiers",
        "ctrl+shift+v",
    ]
    assert calls[3][0] == ["/usr/bin/xclip", "-selection", "clipboard", "-in"]
    assert calls[3][1]["input"] == b"old clipboard"


def test_copy_text_to_clipboard_leaves_transcript_on_clipboard(tmp_path, monkeypatch):
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        return subprocess.CompletedProcess(cmd, 0, stdout=b"")

    monkeypatch.setattr(main_mod.subprocess, "run", fake_run)

    copied = main_mod.copy_text_to_clipboard(
        "final transcript",
        tmp_path / "log.txt",
        clipboard_bin=Path("/usr/bin/xclip"),
    )

    assert copied is True
    assert calls == [
        (
            ["/usr/bin/xclip", "-selection", "clipboard", "-in"],
            {
                "input": b"final transcript",
                "check": True,
                "stdout": subprocess.DEVNULL,
                "stderr": subprocess.DEVNULL,
                "timeout": 1.0,
            },
        )
    ]
    assert "Transcript copied to clipboard" in (tmp_path / "log.txt").read_text(encoding="utf-8")


def test_auto_injection_unknown_window_uses_xdotool(tmp_path, monkeypatch):
    calls = []

    monkeypatch.setattr(main_mod, "get_active_window_info", lambda *_args: None)
    monkeypatch.setattr(main_mod, "type_text_with_xdotool", lambda *args: calls.append(args) or True)

    delivered = main_mod.inject_text(
        "hello",
        Path("/usr/bin/xdotool"),
        tmp_path / "log.txt",
        enable_beep=False,
        injection_mode=main_mod.InjectionMode.AUTO,
    )

    assert delivered is True
    assert calls == [("hello", Path("/usr/bin/xdotool"), tmp_path / "log.txt")]


def test_resolve_orca_daemon_paths_picks_latest_socket_with_token(tmp_path):
    (tmp_path / "daemon-v9.sock").touch()
    (tmp_path / "daemon-v9.token").write_text("old", encoding="utf-8")
    (tmp_path / "daemon-v10.sock").touch()
    (tmp_path / "daemon-v11.sock").touch()
    (tmp_path / "daemon-v11.token").write_text("new", encoding="utf-8")

    resolved = main_mod.resolve_orca_daemon_paths(tmp_path)

    assert resolved == (tmp_path / "daemon-v11.sock", tmp_path / "daemon-v11.token", 11)


def test_orca_session_selection_prefers_requested_session_id():
    sessions = [
        {"sessionId": "other", "state": "running", "isAlive": True},
        {"sessionId": "target", "state": "running", "isAlive": True},
    ]

    selected = main_mod.select_orca_daemon_session(
        sessions,
        preferred_session_id="target",
    )

    assert main_mod.get_orca_session_id(selected) == "target"


def test_orca_session_selection_prefers_active_orca_tab_state(tmp_path, monkeypatch):
    state_path = tmp_path / "orca-data.json"
    state_path.write_text(
        json.dumps(
            {
                "workspaceSession": {
                    "activeWorktreeId": "repo::worktree",
                    "activeTabId": "stale-global-tab",
                    "activeTabIdByWorktree": {
                        "repo::worktree": "active-tab",
                    },
                    "terminalLayoutsByTabId": {
                        "stale-global-tab": {
                            "activeLeafId": "leaf-stale",
                            "ptyIdsByLeafId": {
                                "leaf-stale": "first",
                            },
                        },
                        "active-tab": {
                            "activeLeafId": "leaf-active",
                            "ptyIdsByLeafId": {
                                "leaf-active": "second",
                            },
                        },
                    },
                    "tabsByWorktree": {
                        "repo::worktree": [
                            {"id": "first-tab", "ptyId": "first"},
                            {"id": "active-tab", "ptyId": "second"},
                        ],
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    sessions = [
        {"sessionId": "first", "state": "running", "isAlive": True, "pid": 11},
        {"sessionId": "second", "state": "running", "isAlive": True, "pid": 12},
    ]

    monkeypatch.setattr(main_mod, "process_tree_contains", lambda *_args: True)

    selected = main_mod.select_orca_daemon_session(
        sessions,
        active_state_path=state_path,
    )

    assert main_mod.get_orca_session_id(selected) == "second"


def test_orca_session_selection_prefers_codex_process(monkeypatch):
    sessions = [
        {"sessionId": "old", "state": "exited", "isAlive": False, "pid": 10},
        {"sessionId": "plain", "state": "running", "isAlive": True, "pid": 11},
        {"sessionId": "codex", "state": "running", "isAlive": True, "pid": 12},
    ]

    monkeypatch.setattr(main_mod, "process_tree_contains", lambda pid, _needle: pid == 12)

    selected = main_mod.select_orca_daemon_session(sessions)

    assert main_mod.get_orca_session_id(selected) == "codex"


def test_orca_session_selection_allows_single_live_session(monkeypatch):
    def fail_process_tree(*_args, **_kwargs):
        raise AssertionError("single live session should not need process tree scoring")

    sessions = [
        {"sessionId": "old", "state": "exited", "isAlive": False, "pid": 10},
        {"sessionId": "only-live", "state": "running", "isAlive": True, "pid": 11},
    ]

    monkeypatch.setattr(main_mod, "process_tree_contains", fail_process_tree)

    selected = main_mod.select_orca_daemon_session(sessions)

    assert main_mod.get_orca_session_id(selected) == "only-live"


def test_orca_session_selection_rejects_ambiguous_live_sessions(monkeypatch):
    sessions = [
        {"sessionId": "first", "state": "running", "isAlive": True, "pid": 11},
        {"sessionId": "second", "state": "running", "isAlive": True, "pid": 12},
    ]

    monkeypatch.setattr(main_mod, "process_tree_contains", lambda *_args: False)

    assert main_mod.select_orca_daemon_session(sessions) is None


def test_orca_bracketed_paste_wraps_text_without_enter():
    payload = main_mod.build_orca_bracketed_paste("hello")

    assert payload == "\x1b[200~hello\x1b[201~"
    assert not payload.endswith("\n")


def test_send_text_to_orca_daemon_writes_expected_frames(tmp_path, monkeypatch):
    socket_path = tmp_path / "daemon-v10.sock"
    token_path = tmp_path / "daemon-v10.token"
    socket_path.touch()
    token_path.write_text("test-daemon-token", encoding="utf-8")

    stream = FakeOrcaStream(
        [
            {"type": "hello", "ok": True},
            {
                "id": "list",
                "ok": True,
                "payload": {
                    "sessions": [
                        {
                            "sessionId": "session-1",
                            "state": "running",
                            "isAlive": True,
                            "pid": 12,
                            "cwd": "/workspace/project/",
                        }
                    ]
                },
            },
            {"id": "write", "ok": True, "payload": {}},
        ]
    )
    fake_socket = FakeOrcaSocket(stream)

    monkeypatch.setattr(main_mod.socket, "socket", lambda *_args: fake_socket)
    monkeypatch.setattr(main_mod, "process_tree_contains", lambda *_args: True)

    delivered = main_mod.send_text_to_orca_daemon(
        "hello",
        tmp_path / "log.txt",
        daemon_dir=tmp_path,
    )

    assert delivered is True
    assert fake_socket.timeout == main_mod.ORCA_DAEMON_TIMEOUT_SECONDS
    assert fake_socket.connected_to == str(socket_path)
    assert stream.frames[0]["type"] == "hello"
    assert stream.frames[0]["version"] == 10
    assert stream.frames[0]["token"] == "test-daemon-token"
    assert stream.frames[1]["type"] == "listSessions"
    assert stream.frames[2]["type"] == "write"
    assert stream.frames[2]["payload"] == {
        "sessionId": "session-1",
        "data": "\x1b[200~hello\x1b[201~",
    }


def test_window_matching_ignores_window_title():
    warp_window_titled_orca = main_mod.ActiveWindowInfo(
        window_id="789",
        wm_classes=("dev.warp.Warp", "dev.warp.Warp"),
        name="Debug voice-to-text integration with Orca terminal",
        pid=26709,
        process_args="warp-terminal",
    )

    assert main_mod.is_orca_window(warp_window_titled_orca) is False
    assert main_mod.is_warp_window(warp_window_titled_orca) is True

    orca_window_titled_warp = main_mod.ActiveWindowInfo(
        window_id="790",
        wm_classes=("orca", "Orca"),
        name="Researching the Warp terminal",
        pid=8377,
        process_args="orca-ide",
    )

    assert main_mod.is_orca_window(orca_window_titled_warp) is True
    assert main_mod.is_warp_window(orca_window_titled_warp) is False


def test_auto_injection_uses_xdotool_for_warp_window_titled_orca(tmp_path, monkeypatch):
    calls = []

    def fail_paste(*_args, **_kwargs):
        raise AssertionError("Warp window must not use Orca clipboard paste")

    monkeypatch.setattr(
        main_mod,
        "get_active_window_info",
        lambda *_args: main_mod.ActiveWindowInfo(
            window_id="789",
            wm_classes=("dev.warp.Warp", "dev.warp.Warp"),
            name="Debug voice-to-text integration with Orca terminal",
            pid=26709,
            process_args="warp-terminal",
        ),
    )
    monkeypatch.setattr(main_mod, "type_text_with_xdotool", lambda *args: calls.append(args) or True)
    monkeypatch.setattr(main_mod, "paste_text_with_clipboard_shortcut", fail_paste)
    monkeypatch.setattr(main_mod, "send_text_to_orca_daemon", fail_paste)

    delivered = main_mod.inject_text(
        "hello warp",
        Path("/usr/bin/xdotool"),
        tmp_path / "log.txt",
        enable_beep=False,
        injection_mode=main_mod.InjectionMode.AUTO,
    )

    assert delivered is True
    assert calls and calls[0][0] == "hello warp"
