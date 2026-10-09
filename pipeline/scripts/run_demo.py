#!/usr/bin/env python3
"""Serve an already prepared local TerraLens sample; no downloads or generation.

Run from any directory using pipeline/.venv/bin/python /path/to/run_demo.py.
BUNDLE_DIR and ENCODER_CHECKPOINT environment settings take precedence over
the sample's script-relative defaults. The server listens on localhost:8000.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


PIPELINE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_BUNDLE = PIPELINE_DIR / "data" / "demo"
DEFAULT_CHECKPOINT = PIPELINE_DIR / "data" / "models" / "SkyCLIP_ViT_B32_top50pct.safetensors"


def valid_port(value: str) -> int:
    try:
        port = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Port must be a number between 1 and 65535.") from exc
    if not 1 <= port <= 65535:
        raise argparse.ArgumentTypeError("Port must be between 1 and 65535.")
    return port


def configure_environment() -> tuple[Path, Path]:
    os.environ.setdefault("BUNDLE_DIR", str(DEFAULT_BUNDLE))
    os.environ.setdefault("ENCODER_CHECKPOINT", str(DEFAULT_CHECKPOINT))
    for name in ("BUNDLE_DIR", "ENCODER_CHECKPOINT"):
        if not os.environ[name].strip():
            raise ValueError(f"{name} is empty. Set it to an existing local path or remove the override to use the sample default.")
    bundle = Path(os.environ["BUNDLE_DIR"]).expanduser().resolve()
    checkpoint = Path(os.environ["ENCODER_CHECKPOINT"]).expanduser().resolve()
    if not bundle.is_dir():
        raise ValueError(f"Prepared evidence bundle is missing: {bundle}. Prepare the sample first or set BUNDLE_DIR to a completed bundle.")
    required = ("manifest.json", "tiles.json", "embeddings.npy", "detections.geojson")
    missing = [name for name in required if not (bundle / name).is_file()]
    if missing:
        raise ValueError(f"Evidence bundle is incomplete at {bundle}; missing: {', '.join(missing)}. Finish preparation before starting the server.")
    if not checkpoint.is_file():
        raise ValueError(f"Matching local encoder checkpoint is missing: {checkpoint}. Set ENCODER_CHECKPOINT to the checkpoint used during bundle preparation.")
    return bundle, checkpoint


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1", help="Bind address (default: 127.0.0.1).")
    parser.add_argument("--port", type=valid_port, default=8000, help="Bind port (default: 8000).")
    args = parser.parse_args(argv)
    try:
        bundle, checkpoint = configure_environment()
    except ValueError as exc:
        parser.error(str(exc))
    # Application imports must work when the caller is outside pipeline/.
    sys.path.insert(0, str(PIPELINE_DIR))
    try:
        import uvicorn
    except ImportError as exc:
        raise SystemExit(f"Server dependencies are unavailable. Run this script with {PIPELINE_DIR / '.venv' / 'bin' / 'python'} after installing pipeline requirements.") from exc
    print(f"Evidence bundle: {bundle}")
    print(f"Encoder checkpoint: {checkpoint}")
    uvicorn.run("terralens.api:app", host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
