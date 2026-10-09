#!/usr/bin/env python3
"""Serve the dynamic catalog and durable search backend, without selecting a city."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys


def main(argv: list[str] | None = None) -> int:
    pipeline = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(pipeline))
    from terralens.configuration import load_configuration
    load_configuration()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error("Port must be between 1 and 65535.")
    import uvicorn
    uvicorn.run("terralens.api:app", host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
