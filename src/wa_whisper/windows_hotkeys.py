"""Keep the dictation key out of Windows application menus."""

from pynput import keyboard

from .hotkeys import PushToTalkHotkey


class ShieldedWindowsListener(keyboard.Listener):
    def _convert(self, code, message, data):
        converted = super()._convert(code, message, data)
        if converted is not None and not converted[0] & self._UTF16_FLAG and converted[1] == 0xA5:
            # pynput 1.8.1 posts conversion results to its callback thread. Post
            # Right Alt before suppressing so capture never blocks the OS hook.
            self._message_loop.post(self._WM_PROCESS, *converted)
            self.suppress_event()
        return converted


class WindowsPushToTalkHotkey(PushToTalkHotkey):
    def _create_listener(self):
        return ShieldedWindowsListener(
            on_press=self._handle_press,
            on_release=self._handle_release,
        )

    def _play_toggle_bell(self, purpose):
        import winsound

        winsound.MessageBeep(winsound.MB_OK)
