"""Verify Windows input only, without repeating model inference."""

import ctypes
from ctypes import wintypes
import json
import os
import sys
import time
import tkinter as tk
from pathlib import Path

from wa_whisper.windows_input import type_text

result_path = Path(sys.argv[1])
result = {}
root = tk.Tk()
root.title("WA Whisper typing test")
entry = tk.Text(root, width=65, height=5)
tk.Label(root, text="Click the text box below to verify dictation typing.").pack()
entry.pack()
root.update()
root.attributes("-topmost", True)
root.lift()
root.focus_force()
entry.focus_force()
sample = "Windows voice typing works — café ¥."
deadline = time.monotonic() + 120


def inject():
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    user32.GetForegroundWindow.restype = wintypes.HWND
    user32.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
    user32.SetForegroundWindow.argtypes = [wintypes.HWND]
    foreground_thread = user32.GetWindowThreadProcessId(user32.GetForegroundWindow(), None)
    current_thread = ctypes.windll.kernel32.GetCurrentThreadId()
    attached = user32.AttachThreadInput(current_thread, foreground_thread, True)
    try:
        user32.GetAncestor.argtypes = [wintypes.HWND, wintypes.UINT]
        user32.GetAncestor.restype = wintypes.HWND
        user32.SetForegroundWindow(user32.GetAncestor(root.winfo_id(), 2))
        root.focus_force()
        entry.focus_force()
        root.update()
    finally:
        if attached:
            user32.AttachThreadInput(current_thread, foreground_thread, False)
    pid = wintypes.DWORD()
    user32.GetWindowThreadProcessId(user32.GetForegroundWindow(), ctypes.byref(pid))
    result["foreground_is_test"] = pid.value == os.getpid()
    result["foreground_pid"] = pid.value
    result["test_pid"] = os.getpid()
    if not result["foreground_is_test"] and time.monotonic() < deadline:
        root.after(500, inject)
        return
    try:
        if not result["foreground_is_test"]:
            raise RuntimeError("Windows did not focus the test window")
        type_text(sample)
    except Exception as exc:
        result["error"] = str(exc)
    root.after(1000, verify)


def verify():
    result["typed_text"] = entry.get("1.0", "end-1c")
    result["pass"] = result["typed_text"] == sample
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    root.destroy()


root.after(1000, inject)
root.mainloop()
