#!/usr/bin/env python3
"""Fetch at most three labelled, georeferenced SpaceNet 1 RGB tiles (<10 MiB).

Original imagery and annotations are retained byte for byte. This deliberately
small training subset has partial building reference coverage and no road layer.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import ssl
import subprocess
import sys
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

BASE = "https://spacenet-dataset.s3.amazonaws.com/"
PREFIX = "spacenet/SN1_buildings/train/"
OFFICIAL = "https://spacenet.ai/spacenet-buildings-dataset-v1/"
LICENSE = "CC BY-SA 4.0"
ATTRIBUTION = "SpaceNet Partners (CosmiQ Works, Radiant Solutions and NVIDIA) / SpaceNet 1"
ROOT = Path(__file__).resolve().parents[1]
OWL_REVISION = "cfd3195ba4ea9592eec887ded089f4c08eff231d"
MAX_BYTES = 10 * 1024 * 1024
LABEL_PATTERN = re.compile(r"^spacenet/SN1_buildings/train/geojson/Geo_AOI_1_RIO_img(\d+)\.geojson$")


def digest(path: Path) -> str:
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def write_json(path: Path, value: dict) -> None:
    if path.exists():
        raise RuntimeError(f"Refusing to overwrite source artifact: {path}")
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def candidate_entries(raw: bytes) -> list[dict]:
    """Accept only documented annotation names; ignore empty/small label files."""
    root = ET.fromstring(raw)
    result = []
    for entry in root.findall("{*}Contents"):
        key = entry.findtext("{*}Key", "")
        match = LABEL_PATTERN.fullmatch(key)
        size = int(entry.findtext("{*}Size", "0"))
        if match and 500 < size < 2 * 1024 * 1024:
            result.append({"image_id": "img" + match.group(1), "label_key": key, "label_size": size,
                "raster_key": PREFIX + f"3band/3band_AOI_1_RIO_img{match.group(1)}.tif",
                # Object modification is download provenance, never image acquisition.
                "label_s3_last_modified": entry.findtext("{*}LastModified")})
    # File size is a bounded proxy for annotation richness, not an accuracy sample.
    return sorted(result, key=lambda item: (-item["label_size"], item["image_id"]))


class DownloadBudget:
    def __init__(self, maximum: int = MAX_BYTES):
        self.maximum = maximum
        self.used = 0

    def fetch(self, url: str, *, per_file: int = 4 * 1024 * 1024) -> tuple[bytes, dict]:
        import certifi

        remaining = min(per_file, self.maximum - self.used)
        if remaining <= 0:
            raise RuntimeError("SpaceNet download budget exhausted.")
        request = urllib.request.Request(url, headers={"User-Agent": "TerraLens bounded SpaceNet sample/1.0"})
        context = ssl.create_default_context(cafile=certifi.where())
        with urllib.request.urlopen(request, context=context, timeout=45) as response:
            declared = response.headers.get("Content-Length")
            if declared and int(declared) > remaining:
                raise RuntimeError("SpaceNet object exceeds the remaining bounded download budget.")
            raw = response.read(remaining + 1)
            self.used += len(raw)
            if len(raw) > remaining:
                raise RuntimeError("SpaceNet response exceeds the bounded download budget.")
            headers = {"http_etag": response.headers.get("ETag"),
                "http_last_modified": response.headers.get("Last-Modified")}
        return raw, headers


def source_url(key: str) -> str:
    return BASE + urllib.parse.quote(key, safe="/")


def annotations(raw: bytes, image_id: str) -> list[dict]:
    from shapely.geometry import shape

    document = json.loads(raw)
    if document.get("type") != "FeatureCollection" or not isinstance(document.get("features"), list):
        raise ValueError("Annotation must be a GeoJSON FeatureCollection.")
    features = []
    label_ids = set()
    for position, feature in enumerate(document["features"]):
        if feature.get("type") != "Feature" or not feature.get("geometry"):
            raise ValueError("Missing building annotation geometry.")
        geometry = shape(feature["geometry"])
        if geometry.is_empty or not geometry.is_valid or geometry.geom_type not in {"Polygon", "MultiPolygon"}:
            raise ValueError("Invalid or non-polygon building annotation.")
        west, south, east, north = geometry.bounds
        if not (-180 <= west <= east <= 180 and -90 <= south <= north <= 90):
            raise ValueError("SpaceNet GeoJSON coordinates are not plausible WGS84 longitude/latitude.")
        original = feature.get("properties") or {}
        label_id = original.get("BuildingId", original.get("building_id", feature.get("id", position)))
        if label_id is None:
            label_id = position
        identifier = f"spacenet:{image_id}:{label_id}"
        if identifier in label_ids:
            raise ValueError("Duplicate building label identity within a source tile.")
        label_ids.add(identifier)
        features.append({"type": "Feature", "id": identifier, "geometry": feature["geometry"], "properties": {
            "source_feature_id": identifier, "source_image_id": image_id, "source_label_id": str(label_id),
            "evidence_type": "dataset_annotation", "annotation_properties": original}})
    if not features:
        raise ValueError("Empty SpaceNet building annotations are not positive sample candidates.")
    return features


def inspect_raster(path: Path, features: list[dict]) -> dict:
    import numpy as np
    import rasterio
    from rasterio.warp import transform_bounds
    from shapely.geometry import box, shape

    with rasterio.open(path) as source:
        if source.crs is None or source.count < 3 or max(source.width, source.height) > 2048:
            raise ValueError("SpaceNet RGB tile needs known CRS, three bands and bounded dimensions.")
        rgb = source.read((1, 2, 3), masked=True)
        valid = ~np.any(np.ma.getmaskarray(rgb), axis=0)
        values = np.asarray(rgb.filled(0))
        valid &= np.any(values != 0, axis=0)
        if not valid.any() or np.count_nonzero(valid) / valid.size < 0.001:
            raise ValueError("SpaceNet tile has no usable RGB pixels.")
        if float(np.ptp(values[:, valid].astype(np.float64))) <= 0:
            raise ValueError("SpaceNet tile has no RGB variation.")
        extent = list(transform_bounds(source.crs, "EPSG:4326", *source.bounds))
        image_extent = box(*extent)
        if not all(shape(feature["geometry"]).intersects(image_extent) for feature in features):
            raise ValueError("Building labels do not align with the source raster extent.")
        # Acquisition is used only when explicitly labelled in raster metadata.
        tags = source.tags()
        acquisition = tags.get("acquisition_date") or tags.get("DATE_ACQUIRED")
        return {"crs": source.crs.to_string(), "transform": list(source.transform)[:6],
            "bbox_wgs84": extent, "width": source.width, "height": source.height,
            "dtype": source.dtypes[0], "pixel_size_native_units": [abs(source.transform.a), abs(source.transform.e)],
            "valid_rgb_pixel_fraction": float(np.count_nonzero(valid) / valid.size),
            "acquisition_date": acquisition, "acquisition_date_status": "raster_acquisition_tag" if acquisition else "unknown"}


def imagery_metadata(entry: dict, inspection: dict, path: Path, retrieved_at: str, headers: dict) -> dict:
    return {**inspection, "source": OFFICIAL, "source_url": source_url(entry["raster_key"]),
        "source_name": "SpaceNet 1 Rio three-band training imagery", "source_image_id": entry["image_id"],
        "license": LICENSE, "license_url": "https://creativecommons.org/licenses/by-sa/4.0/",
        "attribution": ATTRIBUTION, "retrieved_at": retrieved_at, "sha256": digest(path),
        "bands": ["red", "green", "blue"], "date_policy": "S3/HTTP modification time is not acquisition time.",
        "download_headers": headers, "profile": {"limits": {"maximum_distance_m": 500}}}


def verify_existing(output: Path, count: int) -> dict | None:
    path = output / "sample.json"
    if not path.is_file():
        return None
    sample = json.loads(path.read_text())
    if len(sample.get("imagery", [])) != count:
        raise RuntimeError("Existing sample count differs; choose a new source output directory.")
    for record in [*sample["imagery"], sample["buildings"]]:
        artifact = output / record["file"]
        sidecar = output / record["metadata"]
        metadata = json.loads(sidecar.read_text())
        if not artifact.is_file() or digest(artifact) != metadata.get("sha256"):
            raise RuntimeError("Existing SpaceNet sample failed checksum verification.")
    print(f"Verified existing SpaceNet sample ({count} RGB tiles).", flush=True)
    return sample


def fetch_sample(output: Path, count: int, candidate_limit: int) -> dict:
    existing = verify_existing(output, count)
    if existing:
        return existing
    if output.exists() and any(path.is_file() or path.is_dir() and any(path.iterdir()) for path in output.iterdir()):
        raise RuntimeError("Incomplete source directory exists; choose a new --output to preserve its inputs.")
    output.mkdir(parents=True, exist_ok=True)
    (output / "imagery").mkdir(exist_ok=True)
    (output / "annotations").mkdir(exist_ok=True)
    budget = DownloadBudget()
    listing_url = BASE + "?" + urllib.parse.urlencode({"list-type": 2, "prefix": PREFIX + "geojson/", "max-keys": 100})
    listing, _ = budget.fetch(listing_url, per_file=128 * 1024)
    (output / "listing.xml").write_bytes(listing)
    entries = candidate_entries(listing)
    imagery, combined, rejected = [], [], []
    for entry in entries[:candidate_limit]:
        if len(imagery) >= count:
            break
        image_id = entry["image_id"]
        try:
            labels_raw, _ = budget.fetch(source_url(entry["label_key"]), per_file=2 * 1024 * 1024)
            features = annotations(labels_raw, image_id)
            labels_path = output / "annotations" / Path(entry["label_key"]).name
            labels_path.write_bytes(labels_raw)
            rgb_raw, headers = budget.fetch(source_url(entry["raster_key"]))
            rgb_path = output / "imagery" / Path(entry["raster_key"]).name
            rgb_path.write_bytes(rgb_raw)
            inspection = inspect_raster(rgb_path, features)
            retrieved_at = datetime.now(timezone.utc).isoformat()
            sidecar = rgb_path.with_suffix(rgb_path.suffix + ".metadata.json")
            write_json(sidecar, imagery_metadata(entry, inspection, rgb_path, retrieved_at, headers))
            imagery.append({"file": str(rgb_path.relative_to(output)), "metadata": str(sidecar.relative_to(output)),
                "image_id": image_id, "label_file": str(labels_path.relative_to(output)), "label_sha256": digest(labels_path),
                "building_count": len(features)})
            combined.extend(features)
            print(f"Accepted {image_id}: {inspection['width']}×{inspection['height']} {inspection['crs']}, {len(features)} building annotations.", flush=True)
        except (ValueError, json.JSONDecodeError) as exc:
            rejected.append({"image_id": image_id, "reason": str(exc)})
            print(f"Rejected {image_id}: {exc}", flush=True)
    if len(imagery) < count:
        write_json(output / "incomplete.json", {"selected": imagery, "rejected": rejected, "network_bytes": budget.used})
        raise RuntimeError(f"Only {len(imagery)} usable tiles found among bounded candidates; requested {count}.")
    vector_path = output / "buildings.geojson"
    write_json(vector_path, {"type": "FeatureCollection", "features": combined})
    vector_metadata = vector_path.with_suffix(vector_path.suffix + ".metadata.json")
    write_json(vector_metadata, {"source": OFFICIAL, "source_name": "SpaceNet 1 building training annotations",
        "source_url": source_url(PREFIX + "geojson/"), "evidence_type": "dataset_annotation", "crs": "EPSG:4326",
        "acquisition_date": None, "extraction_date": None, "extraction_date_status": "unknown",
        "retrieved_at": datetime.now(timezone.utc).isoformat(), "license": LICENSE,
        "license_url": "https://creativecommons.org/licenses/by-sa/4.0/", "attribution": ATTRIBUTION,
        "coverage_status": "partial_training_labels", "reference_coverage_complete": False,
        "coverage_note": "Annotations only for sampled training tiles; no reference halo or municipal completeness.",
        "feature_count": len(combined), "sha256": digest(vector_path),
        "annotation_sources": [{"image_id": item["image_id"], "file": item["label_file"], "sha256": item["label_sha256"]} for item in imagery]})
    sample = {"source": OFFICIAL, "license": LICENSE, "attribution": ATTRIBUTION, "imagery": imagery,
        "buildings": {"file": vector_path.name, "metadata": vector_metadata.name}, "rejected": rejected,
        "network_bytes": budget.used, "network_budget_bytes": MAX_BYTES, "coverage_status": "partial_training_labels",
        "selection_policy": "largest_nonempty_annotation_files_among_first_100_listed_training_tiles",
        "sample_purpose": "Annotation-rich demonstration subset; not a representative or held-out accuracy benchmark."}
    write_json(output / "sample.json", sample)
    return sample


def preparation_command(output: Path, sample: dict, args) -> list[str]:
    command = [sys.executable, str(ROOT / "scripts/prepare_bundle.py"), "--area-name", "SpaceNet 1 Rio training sample",
        "--checkpoint", str(args.checkpoint.resolve()), "--owl-model", str(args.owl_model.resolve()),
        "--owl-revision", getattr(args, "owl_revision", OWL_REVISION),
        "--imagery-attribution", ATTRIBUTION + " · " + LICENSE,
        "--vector-attribution", ATTRIBUTION + " · " + LICENSE,
        "--concept", "buildings=buildings", "--associate", "buildings=buildings",
        "--vectors", f"buildings={output / sample['buildings']['file']}",
        "--vector-metadata", f"buildings={output / sample['buildings']['metadata']}",
        "--tile-size", "1024", "--overlap", "128", "--max-tiles", str(args.count),
        "--device", "cpu", "--output", str(args.bundle_output.resolve())]
    for record in sample["imagery"]:
        image = output / record["file"]
        command.extend(["--imagery", str(image), "--metadata", f"{image}={output / record['metadata']}"])
    return command


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "data/source-spacenet")
    parser.add_argument("--count", type=int, choices=(1, 2, 3), default=3)
    parser.add_argument("--candidate-limit", type=int, default=15)
    parser.add_argument("--prepare", action="store_true", help="Run bounded preparation using the existing local model checkpoints.")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "data/models/SkyCLIP_ViT_B32_top50pct.safetensors")
    parser.add_argument("--owl-model", type=Path, default=ROOT / "data/models/owlv2-cfd3195ba4ea9592eec887ded089f4c08eff231d")
    parser.add_argument("--owl-revision", default=OWL_REVISION, help="Pinned revision of the supplied local detector.")
    parser.add_argument("--bundle-output", type=Path, default=ROOT / "data/spacenet-sample")
    args = parser.parse_args()
    if not 1 <= args.candidate_limit <= 50:
        parser.error("candidate-limit must be between 1 and 50.")
    if args.prepare and (not args.checkpoint.is_file() or not args.owl_model.is_dir()):
        parser.error("Preparation requires the existing local SkyCLIP and OWLv2 model files.")
    if args.prepare and args.bundle_output.exists():
        parser.error("Bundle output already exists; select a new output to preserve the frozen bundle.")
    try:
        output = args.output.expanduser().resolve()
        sample = fetch_sample(output, args.count, args.candidate_limit)
        print(f"Source sample: {output}; downloaded {sample['network_bytes']} bytes (limit {MAX_BYTES}).", flush=True)
        command = preparation_command(output, sample, args)
        if args.prepare:
            environment = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "OMP_NUM_THREADS": "1"}
            subprocess.run(command, check=True, env=environment)
        else:
            import shlex
            print("Preparation command: " + shlex.join(command))
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"SpaceNet acquisition failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
