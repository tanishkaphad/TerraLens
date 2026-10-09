#!/usr/bin/env python3
"""Serve the prepared SpaceNet search without any conversation provider calls."""
from __future__ import annotations

import os
import sys

from run_demo import PIPELINE_DIR, REFERENCE_BUNDLE, configure_environment, main


def run() -> int:
    os.environ["GEOVERIFY_ROUND1"] = "1"
    os.environ["BUNDLE_DIR"] = str(REFERENCE_BUNDLE)
    configure_environment()
    sys.path.insert(0, str(PIPELINE_DIR))
    from terralens.api import get_bundle
    bundle = get_bundle("spacenet")
    if not bundle.retrieval_status()["enabled"]:
        raise SystemExit("The prepared SkyCLIP checkpoint must be available before Round 1 starts.")
    # Warm real text inference before admitting the first visitor's search.
    if bundle.encode_text("buildings") is None:
        raise SystemExit("The imagery search model failed its readiness check.")
    return main()


if __name__ == "__main__":
    raise SystemExit(run())
