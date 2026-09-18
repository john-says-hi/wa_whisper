"""Recording intent retained from the microphone hotkey through text delivery."""

from enum import Enum


class RecordingMode(Enum):
    IDLE = "idle"
    PUSH_TO_TALK = "push_to_talk"
    HANDS_FREE = "hands_free"
