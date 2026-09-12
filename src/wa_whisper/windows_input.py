"""Unicode input with explicit Windows delivery failure reporting."""

import ctypes
from ctypes import wintypes


class KeyboardInput(ctypes.Structure):
    _fields_ = [("vk", wintypes.WORD), ("scan", wintypes.WORD), ("flags", wintypes.DWORD),
                ("time", wintypes.DWORD), ("extra", ctypes.c_size_t)]


class MouseInput(ctypes.Structure):
    _fields_ = [("x", wintypes.LONG), ("y", wintypes.LONG), ("data", wintypes.DWORD),
                ("flags", wintypes.DWORD), ("time", wintypes.DWORD), ("extra", ctypes.c_size_t)]


class InputValue(ctypes.Union):
    _fields_ = [("keyboard", KeyboardInput), ("mouse", MouseInput)]


class Input(ctypes.Structure):
    _fields_ = [("kind", wintypes.DWORD), ("value", InputValue)]


def type_text(text: str) -> None:
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    user32.SendInput.argtypes = [wintypes.UINT, ctypes.POINTER(Input), ctypes.c_int]
    user32.SendInput.restype = wintypes.UINT
    encoded = text.encode("utf-16le")
    events = []
    for offset in range(0, len(encoded), 2):
        unit = int.from_bytes(encoded[offset:offset + 2], "little")
        for flags in (0x0004, 0x0004 | 0x0002):
            events.append(Input(1, InputValue(keyboard=KeyboardInput(0, unit, flags, 0, 0))))
    if not events:
        return
    count = user32.SendInput(len(events), (Input * len(events))(*events), ctypes.sizeof(Input))
    if count != len(events):
        raise RuntimeError(
            f"Windows accepted {count}/{len(events)} input events. "
            "Focus a normal, non-administrator text field; transcript is saved locally."
        )
