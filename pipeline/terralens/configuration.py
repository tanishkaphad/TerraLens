"""Server-only configuration; environment variables take precedence."""
from pathlib import Path

from dotenv import load_dotenv


def load_configuration() -> None:
    load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False, interpolate=False)
