"""Preserve Windows endpoint mute state on one COM apartment thread."""

from concurrent.futures import ThreadPoolExecutor


def _initialize_com():
    import comtypes
    comtypes.CoInitialize()


def _default_volume():
    from pycaw.pycaw import AudioUtilities
    return AudioUtilities.GetSpeakers().EndpointVolume


class WindowsAudioMuteController:
    def __init__(self, volume_factory=_default_volume, initializer=_initialize_com):
        self._volume_factory = volume_factory
        self._executor = ThreadPoolExecutor(max_workers=1, initializer=initializer,
                                            thread_name_prefix="windows_audio")
        self._volume = None
        self._previously_muted = None

    def mute(self):
        self._executor.submit(self._mute).result()

    def restore(self):
        self._executor.submit(self._restore).result()

    def _mute(self):
        if self._volume is not None:
            return
        volume = self._volume_factory()
        previous = bool(volume.GetMute())
        # Retain the original endpoint and state even if SetMute raises.
        self._volume = volume
        self._previously_muted = previous
        volume.SetMute(1, None)

    def _restore(self):
        if self._volume is None:
            return
        self._volume.SetMute(int(self._previously_muted), None)
        self._volume = None
        self._previously_muted = None
