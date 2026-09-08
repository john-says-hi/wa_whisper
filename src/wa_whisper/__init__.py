"""wa_whisper package exports."""

__all__ = ["main"]


def main(argv=None):
    """Load dictation dependencies only when starting dictation."""
    from .main import main as run

    return run(argv)
