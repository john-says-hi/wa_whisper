"""wa_whisper package exports."""

def main(argv=None):
    """Load the Linux CLI only when called, keeping Windows modules importable."""
    from .main import main as run

    return run(argv)

__all__ = ["main"]
