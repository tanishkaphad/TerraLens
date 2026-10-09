#!/usr/bin/env python3
"""Fetch pinned MapLibre, official SkyCLIP B/32 and OWLv2 inference weights.

Model files stay in the ignored data directory. Download hashes and original
source URLs are recorded for reproducible preparation and serving.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import ssl
import urllib.request
import zipfile

import certifi


ROOT = Path(__file__).resolve().parents[1]
MAPLIBRE_VERSION = "5.5.0"
SKYCLIP_URL = "https://opendatasharing.s3.us-west-2.amazonaws.com/SkyScript/ckpt/SkyCLIP_ViT_B32_top50pct.zip"
OWL_ID = "google/owlv2-base-patch16-ensemble"
OWL_REVISION = "cfd3195ba4ea9592eec887ded089f4c08eff231d"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def download(url: str, target: Path, limit: int) -> dict:
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        partial = target.with_suffix(target.suffix + ".partial")
        request = urllib.request.Request(url, headers={"User-Agent": "TerraLens/0.1 dataset-preparation"})
        context = ssl.create_default_context(cafile=certifi.where())
        with urllib.request.urlopen(request, timeout=45, context=context) as response, partial.open("wb") as output:
            expected = int(response.headers.get("Content-Length", 0))
            if expected > limit:
                raise ValueError(f"Download exceeds byte budget: {url} ({expected})")
            received = 0
            for block in iter(lambda: response.read(1024 * 1024), b""):
                received += len(block)
                if received > limit:
                    raise ValueError(f"Download exceeds byte budget: {url}")
                output.write(block)
            if expected and received != expected:
                raise ValueError(f"Incomplete download: {received}/{expected} bytes")
        partial.replace(target)
    record = {"url": url, "file": str(target.relative_to(ROOT)), "bytes": target.stat().st_size, "sha256": digest(target)}
    print(json.dumps(record), flush=True)
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", choices=("map", "model", "detector", "all"), default="all")
    args = parser.parse_args()
    records = []
    if args.only in {"map", "all"}:
        vendor = ROOT / "terralens" / "static" / "vendor"
        for filename, path in (("maplibre-gl.js", "dist/maplibre-gl.js"), ("maplibre-gl.css", "dist/maplibre-gl.css"), ("MAPLIBRE_LICENSE.txt", "LICENSE.txt")):
            records.append(download(f"https://unpkg.com/maplibre-gl@{MAPLIBRE_VERSION}/{path}", vendor / filename, 8 * 1024 * 1024))
    if args.only in {"model", "all"}:
        model_dir = ROOT / "data" / "models"
        archive = model_dir / "SkyCLIP_ViT_B32_top50pct.zip"
        records.append(download(SKYCLIP_URL, archive, 2 * 1024**3))
        with zipfile.ZipFile(archive) as zipped:
            candidates = [entry for entry in zipped.infolist() if not entry.is_dir() and entry.filename.endswith((".pt", ".bin", ".ckpt", ".pth")) and "__MACOSX" not in entry.filename]
            if len(candidates) != 1:
                raise ValueError(f"Expected exactly one checkpoint in the official archive; found {[entry.filename for entry in candidates]}")
            entry = candidates[0]
            checkpoint = model_dir / "SkyCLIP_ViT_B32_top50pct.pt"
            if not checkpoint.exists():
                with zipped.open(entry) as source, checkpoint.open("wb") as output:
                    shutil.copyfileobj(source, output, length=1024 * 1024)
            checkpoint_record = {"file": str(checkpoint.relative_to(ROOT)), "archive_entry": entry.filename, "bytes": checkpoint.stat().st_size, "sha256": digest(checkpoint), "model_name": "ViT-B-32", "source": SKYCLIP_URL}
            records.append(checkpoint_record)
            print(json.dumps(checkpoint_record), flush=True)
    if args.only in {"detector", "all"}:
        detector_dir = ROOT / "data" / "models" / f"owlv2-{OWL_REVISION}"
        for filename in ("README.md", "config.json", "preprocessor_config.json", "tokenizer_config.json", "special_tokens_map.json", "added_tokens.json", "vocab.json", "merges.txt", "model.safetensors"):
            record = download(f"https://huggingface.co/{OWL_ID}/resolve/{OWL_REVISION}/{filename}", detector_dir / filename, 1024**3)
            record.update(model_id=OWL_ID, revision=OWL_REVISION, license="Apache-2.0")
            records.append(record)
    metadata = ROOT / "data" / "asset-provenance.json"
    metadata.parent.mkdir(parents=True, exist_ok=True)
    existing = json.loads(metadata.read_text()) if metadata.exists() else []
    by_path = {record["file"]: record for record in existing + records}
    metadata.write_text(json.dumps(list(by_path.values()), indent=2))


if __name__ == "__main__":
    main()
