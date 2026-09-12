"""Verify actual Windows speaker mute through capture start and shutdown."""
import json
import sys
from pathlib import Path

from pycaw.pycaw import AudioUtilities
from wa_whisper.hotkeys import RecordingMode
from wa_whisper.windows_hotkeys import WindowsPushToTalkHotkey
from windows_hotkey_probe import ProbeRecorder

result_path = Path(sys.argv[1])
volume = AudioUtilities.GetSpeakers().EndpointVolume
original = volume.GetMute()
level = volume.GetMasterVolumeLevelScalar()
results = []
try:
    for initially_muted in (False, True):
        volume.SetMute(int(initially_muted), None)
        recorder = ProbeRecorder(result_path.with_suffix('.wav'))
        hotkey = WindowsPushToTalkHotkey(
            recorder, silence_timeout=0, on_capture_finished=lambda capture: None,
            log_path=result_path.with_suffix('.log'), enable_hotkey_shield=False)
        try:
            with hotkey._lock:
                hotkey._start_capture_locked(RecordingMode.PUSH_TO_TALK)
            during = bool(volume.GetMute())
        finally:
            hotkey.stop()
        restored = bool(volume.GetMute()) == initially_muted
        unchanged = volume.GetMasterVolumeLevelScalar() == level
        results.append(dict(initially_muted=initially_muted, muted_during=during,
                            restored=restored, volume_unchanged=unchanged))
        assert during and restored and unchanged
        recorder.path.unlink(missing_ok=True)
finally:
    volume.SetMute(original, None)
    result_path.write_text(json.dumps(results, indent=2))
