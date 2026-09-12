"""Exercise the real Windows hook with synthetic input and a harmless recorder."""

import json
import sys
import time
from pathlib import Path

from pynput import keyboard

from wa_whisper.windows_hotkeys import WindowsPushToTalkHotkey


class ProbeRecorder:
    def __init__(self, path):
        self.path = path
        self.starts = 0
        self.stops = 0

    def start(self):
        self.starts += 1
        self.path.touch()
        return self.path

    def stop(self, timeout):
        self.stops += 1
        return self.path

    def last_capture_stats(self):
        return None


class HardwareLikeProbe(WindowsPushToTalkHotkey):
    def _handle_press(self, key, injected=False):
        return super()._handle_press(key, injected=False)

    def _handle_release(self, key, injected=False):
        return super()._handle_release(key, injected=False)


def main():
    result_path = Path(sys.argv[1])
    recorder = ProbeRecorder(result_path.with_suffix(".wav"))
    captures = []
    hotkey = HardwareLikeProbe(recorder, silence_timeout=0, on_capture_finished=captures.append,
                               log_path=result_path.with_suffix(".log"), enable_audio_mute=False,
                               enable_hotkey_shield=False, exit_on_esc=False)
    result = {}
    try:
        hotkey.start()
        keys = keyboard.Controller()
        time.sleep(0.5)
        keys.press(keyboard.Key.alt_r)
        time.sleep(0.2)
        keys.release(keyboard.Key.alt_r)
        time.sleep(1.3)
        result["push_to_talk"] = recorder.starts == 1 and recorder.stops == 1
        for _ in range(2):
            keys.press(keyboard.Key.ctrl_l)
            time.sleep(0.1)
            keys.press(keyboard.Key.alt_r)
            time.sleep(0.2)
            keys.release(keyboard.Key.alt_r)
            keys.release(keyboard.Key.ctrl_l)
            time.sleep(0.3)
        time.sleep(0.5)
        result["hands_free"] = recorder.starts == 2 and recorder.stops == 2
        result["completed_captures"] = len(captures)
    except Exception as exc:
        result["error"] = repr(exc)
    finally:
        hotkey.stop()
        result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        recorder.path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
