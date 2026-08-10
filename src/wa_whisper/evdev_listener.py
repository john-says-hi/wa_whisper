"""Display-server-independent global hotkey listener built on evdev.

pynput's Linux backend talks to X11 through Xlib and XRecord. Under Wayland the
compositor refuses global key grabs and never forwards keystrokes aimed at other
clients, so ``keyboard.Listener`` starts cleanly, reports no error, and then
never fires. Push-to-talk simply stops working with nothing in the log.

Reading ``/dev/input/event*`` sidesteps the problem entirely because evdev sits
*below* the compositor: the kernel publishes key events to anyone in the
``input`` group regardless of which display server is running.

This listener deliberately emits ``pynput.keyboard.Key`` values rather than raw
evdev codes, so :class:`~wa_whisper.hotkeys.PushToTalkHotkey` keeps its existing
event handling, its state machine and its tests unchanged — only the source of
the events differs.

Two consequences worth knowing:

* **The key is observed, not consumed.** Unlike the X11 key shield, an evdev
  reader cannot hide Right Alt from the focused window, and a device-wide
  ``EVIOCGRAB`` is not an option because it would swallow ordinary typing too.
  Apps that react to a bare Alt press (Electron menu bars) will still react.
* **Every matching device is read.** A laptop's built-in keyboard and an
  external USB keyboard are separate event nodes, and a keyboard may be plugged
  in after startup, so the device set is rescanned periodically.
"""

from __future__ import annotations

import errno
import os
import selectors
import threading
from pathlib import Path
from typing import Callable, Optional

from pynput import keyboard

from .log_utils import write_log

# Rescan for hot-plugged keyboards on this cadence. Long enough to be free,
# short enough that plugging a keyboard in feels instant.
DEVICE_RESCAN_INTERVAL_SECONDS = 2.0

KeyCallback = Callable[[keyboard.Key], None]


def load_evdev():
    """Import evdev lazily so X11 sessions never pay for it."""
    from evdev import InputDevice, ecodes, list_devices

    return InputDevice, ecodes, list_devices


def _key_map(ecodes) -> dict[int, keyboard.Key]:
    """Map the only three keys ``PushToTalkHotkey`` reacts to.

    Anything else is dropped here rather than in the handler, so this listener
    never has to synthesise a ``KeyCode`` for ordinary typing — which would mean
    reconstructing keyboard layout and modifier state from scratch.
    """
    return {
        ecodes.KEY_RIGHTALT: keyboard.Key.alt_r,
        ecodes.KEY_LEFTCTRL: keyboard.Key.ctrl_l,
        ecodes.KEY_ESC: keyboard.Key.esc,
    }


class EvdevKeyListener:
    """Watch every keyboard for the push-to-talk keys.

    Presents the subset of the ``pynput.keyboard.Listener`` interface that
    ``PushToTalkHotkey`` uses — ``start()``, ``stop()``, ``join()`` — so the two
    are interchangeable.
    """

    def __init__(
        self,
        *,
        on_press: KeyCallback,
        on_release: KeyCallback,
        log_path: Path,
    ) -> None:
        self._on_press = on_press
        self._on_release = on_release
        self._log_path = log_path
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._devices: dict[str, object] = {}
        self._warned_empty = False

    # Lifecycle ----------------------------------------------------------------

    def start(self) -> None:
        if self._thread is not None:
            return

        InputDevice, ecodes, list_devices = load_evdev()
        self._key_codes = _key_map(ecodes)
        self._ecodes = ecodes
        self._InputDevice = InputDevice
        self._list_devices = list_devices

        # Fail loudly at start() rather than silently never firing, which is the
        # exact failure mode this module exists to eliminate.
        found = self._scan_devices()
        if not found:
            raise RuntimeError(
                "No readable keyboard devices found under /dev/input. "
                "Add yourself to the 'input' group and log out and back in: "
                "sudo usermod -aG input $USER"
            )

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="wa-whisper-evdev-hotkeys",
            daemon=True,
        )
        self._thread.start()
        write_log(
            f"evdev hotkey listener started on {found} keyboard device(s)",
            self._log_path,
        )

    def stop(self) -> None:
        self._stop_event.set()
        thread = self._thread
        self._thread = None
        if thread and thread.is_alive():
            thread.join(timeout=2.0)
        self._close_devices()
        write_log("evdev hotkey listener stopped", self._log_path)

    def join(self, timeout: Optional[float] = None) -> None:
        thread = self._thread
        if thread:
            thread.join(timeout)

    # Device handling ----------------------------------------------------------

    def _scan_devices(self) -> int:
        """Open every device that can report the keys we care about.

        Returns the number of devices currently open.
        """
        wanted = set(self._key_codes)
        for path in self._list_devices():
            if path in self._devices:
                continue
            try:
                device = self._InputDevice(path)
            except OSError as exc:
                # EACCES is the common one: not in the 'input' group yet.
                if exc.errno not in (errno.EACCES, errno.ENOENT):
                    write_log(f"evdev: cannot open {path}: {exc}", self._log_path)
                continue

            capabilities = device.capabilities().get(self._ecodes.EV_KEY, [])
            if wanted.isdisjoint(capabilities):
                device.close()
                continue

            self._devices[path] = device
            write_log(f"evdev: watching {path} ({device.name})", self._log_path)

        return len(self._devices)

    def _drop_device(self, path: str) -> None:
        device = self._devices.pop(path, None)
        if device is None:
            return
        try:
            device.close()
        except OSError:
            pass
        write_log(f"evdev: stopped watching {path}", self._log_path)

    def _close_devices(self) -> None:
        for path in list(self._devices):
            self._drop_device(path)

    # Event loop ---------------------------------------------------------------

    def _run(self) -> None:
        try:
            while not self._stop_event.is_set():
                selector = selectors.DefaultSelector()
                registered = []
                for path, device in list(self._devices.items()):
                    try:
                        selector.register(device, selectors.EVENT_READ, path)
                        registered.append(path)
                    except (OSError, ValueError):
                        self._drop_device(path)

                if not registered and not self._warned_empty:
                    write_log("evdev: no keyboards currently readable", self._log_path)
                    self._warned_empty = True
                elif registered:
                    self._warned_empty = False

                # The timeout doubles as the hot-plug rescan cadence and as the
                # stop-event check, so shutdown never waits on a keypress.
                try:
                    ready = selector.select(timeout=DEVICE_RESCAN_INTERVAL_SECONDS)
                except OSError:
                    ready = []

                for selector_key, _mask in ready:
                    self._drain_device(selector_key.data, selector_key.fileobj)

                selector.close()
                self._scan_devices()
        except Exception as exc:  # pragma: no cover - defensive
            write_log(f"evdev listener crashed: {exc!r}", self._log_path)

    def _drain_device(self, path: str, device) -> None:
        try:
            for event in device.read():
                if event.type != self._ecodes.EV_KEY:
                    continue
                key = self._key_codes.get(event.code)
                if key is None:
                    continue
                # 1 = press, 0 = release, 2 = autorepeat. Autorepeat would look
                # like a storm of fresh presses to the state machine.
                if event.value == 1:
                    self._dispatch(self._on_press, key)
                elif event.value == 0:
                    self._dispatch(self._on_release, key)
        except BlockingIOError:
            return
        except OSError as exc:
            if exc.errno == errno.ENODEV:
                self._drop_device(path)  # keyboard unplugged mid-read
            else:
                write_log(f"evdev: read error on {path}: {exc}", self._log_path)
                self._drop_device(path)

    def _dispatch(self, callback: KeyCallback, key: keyboard.Key) -> None:
        # A raising handler must not kill the listener thread; that would leave
        # dictation silently dead for the rest of the session.
        try:
            callback(key)
        except Exception as exc:  # pragma: no cover - defensive
            write_log(f"evdev hotkey handler raised: {exc!r}", self._log_path)


def wayland_session() -> bool:
    """True when the current session cannot support the pynput/X11 path."""
    return os.environ.get("XDG_SESSION_TYPE", "").lower() == "wayland"
