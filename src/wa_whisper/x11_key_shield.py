"""Server-side X11 grab that hides the push-to-talk key from focused apps."""

from __future__ import annotations

import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

from .log_utils import write_log

POLL_INTERVAL_SECONDS = 0.02


def load_xlib() -> SimpleNamespace:
    from Xlib import X, XK, display, error

    return SimpleNamespace(X=X, XK=XK, display=display, error=error)


class X11KeyShield:
    """Grab one key on the X11 root window so other clients never receive it.

    Electron apps (Orca included) reveal their menu bar when they observe a
    bare Alt press/release, which is exactly what every Right Alt
    push-to-talk gesture looks like. Grabbing the key makes the X server
    deliver its events to this client alone; pynput still observes the key
    through the XRecord extension, so recording keeps working while focused
    windows stay blind to the hotkey.
    """

    def __init__(self, keysym_name: str, log_path: Path) -> None:
        self._keysym_name = keysym_name
        self._log_path = log_path
        self._xlib: Optional[SimpleNamespace] = None
        self._display = None
        self._keycode: Optional[int] = None
        self._stop_event = threading.Event()
        self._drain_thread: Optional[threading.Thread] = None

    @property
    def active(self) -> bool:
        return self._display is not None

    def start(self) -> bool:
        if self.active:
            return True
        xlib = self._load_xlib_or_log()
        if xlib is None:
            return False
        display = self._open_display_or_log(xlib)
        if display is None:
            return False
        keycode = self._resolve_keycode(xlib, display)
        if not keycode:
            self._close_display_quietly(display)
            write_log(
                f"Hotkey shield disabled: no keycode for keysym {self._keysym_name}",
                self._log_path,
            )
            return False
        if not self._grab_key_or_log(xlib, display, keycode):
            self._close_display_quietly(display)
            return False

        self._xlib = xlib
        self._display = display
        self._keycode = keycode
        self._stop_event.clear()
        self._drain_thread = threading.Thread(
            target=self._drain_grabbed_events,
            name="wa-whisper-key-shield",
            daemon=True,
        )
        self._drain_thread.start()
        write_log(
            f"Hotkey shield active: {self._keysym_name} (keycode {keycode}) "
            "is hidden from focused apps",
            self._log_path,
        )
        return True

    def stop(self) -> None:
        self._stop_event.set()
        drain_thread, self._drain_thread = self._drain_thread, None
        if drain_thread:
            drain_thread.join(timeout=1.0)

        display, self._display = self._display, None
        xlib, self._xlib = self._xlib, None
        keycode, self._keycode = self._keycode, None
        if display is None:
            return
        try:
            if xlib is not None and keycode:
                display.screen().root.ungrab_key(keycode, xlib.X.AnyModifier)
                display.sync()
            write_log("Hotkey shield released", self._log_path)
        except Exception as exc:
            write_log(f"Hotkey shield release failed: {exc}", self._log_path)
        finally:
            self._close_display_quietly(display)

    def _load_xlib_or_log(self) -> Optional[SimpleNamespace]:
        try:
            return load_xlib()
        except Exception as exc:
            write_log(
                f"Hotkey shield disabled: python-xlib unavailable: {exc}",
                self._log_path,
            )
            return None

    def _open_display_or_log(self, xlib: SimpleNamespace):
        try:
            return xlib.display.Display()
        except Exception as exc:
            write_log(
                f"Hotkey shield disabled: cannot open X display: {exc}",
                self._log_path,
            )
            return None

    def _resolve_keycode(self, xlib: SimpleNamespace, display) -> int:
        keysym = xlib.XK.string_to_keysym(self._keysym_name)
        if not keysym:
            return 0
        try:
            return int(display.keysym_to_keycode(keysym) or 0)
        except Exception:
            return 0

    def _grab_key_or_log(self, xlib: SimpleNamespace, display, keycode: int) -> bool:
        try:
            grab_error = xlib.error.CatchError(xlib.error.BadAccess)
            display.screen().root.grab_key(
                keycode,
                xlib.X.AnyModifier,
                False,
                xlib.X.GrabModeAsync,
                xlib.X.GrabModeAsync,
                onerror=grab_error,
            )
            display.sync()
            if grab_error.get_error():
                raise RuntimeError("key is already grabbed by another client")
        except Exception as exc:
            write_log(f"Hotkey shield disabled: grab failed: {exc}", self._log_path)
            return False
        return True

    def _drain_grabbed_events(self) -> None:
        # The grabbed key's events are delivered to this connection; consume
        # them so the queue never grows. pynput sees the key via XRecord.
        display = self._display
        while not self._stop_event.is_set():
            try:
                while display.pending_events():
                    display.next_event()
            except Exception as exc:
                write_log(f"Hotkey shield event loop stopped: {exc}", self._log_path)
                return
            time.sleep(POLL_INTERVAL_SECONDS)

    @staticmethod
    def _close_display_quietly(display) -> None:
        try:
            display.close()
        except Exception:
            pass
