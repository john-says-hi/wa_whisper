from threading import get_ident

import pytest

from wa_whisper.windows_audio import WindowsAudioMuteController


@pytest.mark.parametrize("original", [False, True])
def test_mute_restore_preserves_state_and_serializes_com(original):
    class Volume:
        muted = original
        threads = set()
        def GetMute(self):
            self.threads.add(get_ident())
            return self.muted
        def SetMute(self, value, context):
            self.threads.add(get_ident())
            self.muted = bool(value)
    volume = Volume()
    controller = WindowsAudioMuteController(lambda: volume, lambda: None)
    try:
        controller.mute()
        controller.mute()
        assert volume.muted
        controller.restore()
        controller.restore()
        assert volume.muted == original
        assert len(volume.threads) == 1
        assert get_ident() not in volume.threads
    finally:
        controller._executor.shutdown()


def test_failed_restore_retains_original_state_for_retry():
    class Volume:
        fail = False
        muted = False
        def GetMute(self):
            return self.muted
        def SetMute(self, value, context):
            if self.fail:
                raise OSError("endpoint unavailable")
            self.muted = bool(value)
    volume = Volume()
    controller = WindowsAudioMuteController(lambda: volume, lambda: None)
    try:
        controller.mute()
        volume.fail = True
        with pytest.raises(OSError):
            controller.restore()
        volume.fail = False
        controller.restore()
        assert not volume.muted
    finally:
        controller._executor.shutdown()
