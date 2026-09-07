"""Ensure Right Alt is delivered to dictation but hidden from application menus."""

from types import SimpleNamespace

import pytest

from wa_whisper import windows_hotkeys as hotkeys


def test_right_alt_is_queued_before_suppression(monkeypatch):
    events = []
    converted = (0x100, 0xA5)
    monkeypatch.setattr(hotkeys.keyboard.Listener, "_convert", lambda *args: converted, raising=False)
    listener = object.__new__(hotkeys.ShieldedWindowsListener)
    listener._UTF16_FLAG = 0x1000
    listener._WM_PROCESS = 0x410
    listener._message_loop = SimpleNamespace(post=lambda *args: events.append(args))

    def suppress():
        raise RuntimeError("suppressed")

    listener.suppress_event = suppress
    with pytest.raises(RuntimeError, match="suppressed"):
        listener._convert(0, 0x100, None)
    assert events == [(0x410, *converted)]


def test_unicode_yen_character_is_not_mistaken_for_right_alt(monkeypatch):
    converted = (0x1100, 0xA5)
    monkeypatch.setattr(hotkeys.keyboard.Listener, "_convert", lambda *args: converted, raising=False)
    listener = object.__new__(hotkeys.ShieldedWindowsListener)
    listener._UTF16_FLAG = 0x1000
    assert listener._convert(0, 0x100, None) == converted
