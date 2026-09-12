"""Windows logon launcher and Ctrl+Shift+F1 power control."""

from __future__ import annotations

import ctypes
from ctypes import wintypes
import queue
import os
import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path

from .log_utils import write_log
from .windows_control import request_power, take_request

STATE_DIR = Path.home() / ".cache" / "wa_whisper"
HOTKEY_ID = 1
WM_HOTKEY = 0x0312
MOD_CONTROL_SHIFT_NOREPEAT = 0x4006
VK_F1 = 0x70


def register_power_hotkey(events: queue.Queue) -> None:
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    if not user32.RegisterHotKey(None, HOTKEY_ID, MOD_CONTROL_SHIFT_NOREPEAT, VK_F1):
        events.put("hotkey_failed")
        return
    events.put("hotkey_registered")
    message = wintypes.MSG()
    try:
        while user32.GetMessageW(ctypes.byref(message), None, 0, 0) > 0:
            if message.message == WM_HOTKEY and message.wParam == HOTKEY_ID:
                events.put("toggle")
    finally:
        user32.UnregisterHotKey(None, HOTKEY_ID)


class DictationController:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.process: subprocess.Popen | None = None
        self.stopping = False
        self.start_pending = False
        self.events: queue.Queue = queue.Queue()
        self.status = tk.StringVar(value="Starting")
        root.title("WA Whisper — Voice to Text")
        root.geometry("500x200")
        tk.Label(root, textvariable=self.status, wraplength=470, font=("Segoe UI", 11)).pack(pady=16)
        tk.Label(root, text="Ctrl+Shift+F1: turn dictation on/off\nHold Right Alt: dictate\nLeft Ctrl + Right Alt: hands-free on/off").pack()
        tk.Button(root, text="Turn on / off", command=self.toggle).pack(pady=12)
        root.protocol("WM_DELETE_WINDOW", root.iconify)
        threading.Thread(target=register_power_hotkey, args=(self.events,), daemon=True).start()
        self.start()
        root.after(200, self.poll)

    def set_status(self, message: str) -> None:
        self.status.set(message)
        self.root.title("WA Whisper — " + message)
        (STATE_DIR / "windows_status.txt").write_text(message, encoding="utf-8")

    def start(self) -> None:
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        (STATE_DIR / "windows_status.txt").write_text("Starting", encoding="utf-8")
        executable = Path(sys.executable).with_name("python.exe")
        environment = os.environ.copy()
        ffmpeg_links = Path.home() / "AppData" / "Local" / "Microsoft" / "WinGet" / "Links"
        environment["PATH"] = str(ffmpeg_links) + os.pathsep + environment.get("PATH", "")
        with (STATE_DIR / "windows_worker.log").open("a", encoding="utf-8") as log:
            self.process = subprocess.Popen(
                [str(executable), "-m", "wa_whisper.windows_worker"],
                stdin=subprocess.PIPE,
                stdout=log,
                stderr=log,
                text=True,
                creationflags=subprocess.CREATE_NO_WINDOW,
                env=environment,
            )
        self.set_status("Starting — connecting to shared laptop model")
        write_log("Windows dictation power on")

    def toggle(self) -> None:
        if self.stopping:
            return
        if self.process is None or self.process.poll() is not None:
            self.start()
            return
        try:
            self.process.stdin.write("stop\n")
            self.process.stdin.flush()
            self.stopping = True
            self.set_status("Stopping safely; saving any active recording")
            write_log("Windows dictation power off requested")
        except (BrokenPipeError, OSError) as exc:
            self.status.set(f"Stopping: {exc}")

    def poll(self) -> None:
        request = take_request()
        if request == "start":
            if self.stopping:
                self.start_pending = True
            elif self.process is None or self.process.poll() is not None:
                self.start_pending = True
            self.root.deiconify()
            self.root.lift()
        elif request == "stop" and self.process is not None and self.process.poll() is None:
            self.toggle()
        while not self.events.empty():
            event = self.events.get_nowait()
            if event == "toggle":
                self.toggle()
            elif event == "hotkey_failed":
                write_log("Ctrl+Shift+F1 registration failed: shortcut already in use")
                tk.Label(self.root, text="Ctrl+Shift+F1 unavailable; use the button", fg="red").pack()
            elif event == "hotkey_registered":
                write_log("Ctrl+Shift+F1 registered")
        if self.process is not None:
            code = self.process.poll()
            if code is not None:
                self.process.stdin.close()
                self.process = None
                self.stopping = False
                self.set_status("OFF — Ctrl+Shift+F1 or the Desktop shortcut to start" if code == 0 else "Stopped with an error — check windows_worker.log")
                write_log(f"Windows dictation worker exited: {code}")
            elif not self.stopping:
                try:
                    self.status.set((STATE_DIR / "windows_status.txt").read_text(encoding="utf-8"))
                except OSError:
                    pass
        if self.start_pending and self.process is None:
            self.start_pending = False
            self.start()
        self.root.after(200, self.poll)


def main() -> None:
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateMutexW.restype = wintypes.HANDLE
    mutex = kernel32.CreateMutexW(None, False, "Local\\WAWhisperWindowsController")
    if not mutex:
        raise ctypes.WinError(ctypes.get_last_error())
    if ctypes.get_last_error() == 183:
        request_power("start")
        return
    root = tk.Tk()
    DictationController(root)
    root.mainloop()


if __name__ == "__main__":
    main()
