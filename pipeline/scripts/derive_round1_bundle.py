#!/usr/bin/env python3
"""Add a recorded OSM road snapshot to the frozen SpaceNet evidence bundle.

This derivation copies every cached image, embedding, and OWLv2 result byte for
byte. It validates and adds only the bounded road source, so no model inference
is repeated and the original evidence bundle remains immutable.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

from pyproj import Transformer
from shapely import transform
from shapely.geometry import LineString, box, shape

from prepare_bundle import bundle_version


QUERY_RADIUS_M = 50.0
HALO_CRS = "EPSG:32723"
OVERPASS_HOSTS = {"overpass-api.de", "overpass.kumi.systems"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def safe_child(root: Path, relative: str) -> Path:
    candidate = (root / relative).resolve()
    if candidate == root or root not in candidate.parents:
        raise ValueError(f"Artifact path leaves its root: {relative}")
    return candidate


def verify_hash(path: Path, expected: str, label: str) -> None:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"Missing regular {label}: {path}")
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"{label} hash mismatch: expected {expected}, received {actual}")


def bounds_close(left: list, right: list, tolerance: float = 1e-10) -> bool:
    return (len(left) == len(right) == 4 and all(
        isinstance(value, (int, float)) and math.isfinite(value)
        for value in [*left, *right]
    ) and all(abs(float(a) - float(b)) <= tolerance for a, b in zip(left, right)))


def validate_base(base: Path) -> dict:
    manifest = read_json(base / "manifest.json")
    artifacts = manifest.get("artifact_sha256")
    if not isinstance(artifacts, dict) or not artifacts:
        raise ValueError("The base bundle has no immutable artifact inventory.")
    for relative, expected in artifacts.items():
        if not isinstance(relative, str) or not isinstance(expected, str):
            raise ValueError("The base artifact inventory is malformed.")
        verify_hash(safe_child(base, relative), expected, f"base artifact '{relative}'")
    if manifest.get("coverage", {}).get("tile_count") != 3:
        raise ValueError("Round 1 derives only from the frozen three-tile SpaceNet sample.")
    if manifest.get("coverage", {}).get("bbox_wgs84") is None:
        raise ValueError("The base bundle has no recorded WGS84 coverage.")
    return manifest


def validate_osm_source(source: Path, base_manifest: dict) -> tuple[dict, dict, list[dict]]:
    source_record = read_json(source / "source.json")
    hashes = source_record.get("artifact_sha256")
    if not isinstance(hashes, dict) or not hashes:
        raise ValueError("The OSM source has no artifact inventory.")
    for relative, expected in hashes.items():
        if not isinstance(relative, str) or not isinstance(expected, str):
            raise ValueError("The OSM artifact inventory is malformed.")
        verify_hash(safe_child(source, relative), expected, f"OSM source artifact '{relative}'")

    metadata_path = source / "vectors" / "roads.geojson.metadata.json"
    roads_path = source / "vectors" / "roads.geojson"
    metadata = read_json(metadata_path)
    roads = read_json(roads_path)
    raw = read_json(source / "osm-response.json")
    study_bbox = base_manifest["coverage"]["bbox_wgs84"]
    if not bounds_close(source_record.get("study_bbox_wgs84", []), study_bbox):
        raise ValueError("The OSM source study bounds differ from the base imagery coverage.")
    if not bounds_close(metadata.get("source_study_bbox_wgs84", []), study_bbox):
        raise ValueError("The road metadata study bounds differ from the base imagery coverage.")
    if metadata.get("source_dataset_version") != base_manifest.get("dataset_version"):
        raise ValueError("The OSM source was prepared for a different base bundle.")
    if float(metadata.get("maximum_query_radius_m", 0)) != QUERY_RADIUS_M:
        raise ValueError("The road source must record an exact 50 metre query radius.")
    if metadata.get("halo_metric_crs") != HALO_CRS:
        raise ValueError("The road halo must be measured in the recorded Rio UTM CRS.")
    if metadata.get("sha256") != sha256_file(roads_path):
        raise ValueError("The road metadata does not bind the road GeoJSON.")
    if metadata.get("raw_response_sha256") != sha256_file(source / "osm-response.json"):
        raise ValueError("The road metadata does not bind the raw Overpass response.")
    if metadata.get("query_sha256") != sha256_file(source / "osm-query.txt"):
        raise ValueError("The road metadata does not bind the Overpass query.")
    endpoint = urlparse(str(metadata.get("source_url", "")))
    if endpoint.scheme != "https" or endpoint.hostname not in OVERPASS_HOSTS:
        raise ValueError("Roads must come from a recorded official Overpass endpoint.")

    query_bbox = metadata.get("coverage_bbox_wgs84", [])
    if len(query_bbox) != 4:
        raise ValueError("The road metadata has no query coverage bounds.")
    to_metric = Transformer.from_crs("EPSG:4326", HALO_CRS, always_xy=True).transform
    study = transform(box(*map(float, study_bbox)), to_metric, interleaved=False)
    west, south, east, north = map(float, query_bbox)
    query_area = transform(box(west, south, east, north), to_metric, interleaved=False)
    if not query_area.covers(study.buffer(QUERY_RADIUS_M)) \
            or query_area.boundary.distance(study) + 0.001 < QUERY_RADIUS_M:
        raise ValueError("The Overpass query bounds do not enclose the 50 metre imagery halo.")

    if roads.get("type") != "FeatureCollection" or not isinstance(roads.get("features"), list):
        raise ValueError("Roads must be a GeoJSON FeatureCollection.")
    features = roads["features"]
    if not features or metadata.get("feature_count") != len(features):
        raise ValueError("The road source is empty or its feature count is inconsistent.")
    raw_by_id = {
        f"osm/{element.get('type')}/{element.get('id')}": element
        for element in raw.get("elements", [])
        if isinstance(element, dict)
    }
    identifiers: set[str] = set()
    for feature in features:
        properties = feature.get("properties", {})
        identifier = feature.get("id")
        if not isinstance(identifier, str) or identifier in identifiers or properties.get("id") != identifier:
            raise ValueError("Road features need matching, unique OSM identities.")
        identifiers.add(identifier)
        geometry = shape(feature.get("geometry"))
        if not isinstance(geometry, LineString) or geometry.is_empty or not geometry.is_valid:
            raise ValueError(f"Road '{identifier}' is not a valid non-empty LineString.")
        if not isinstance(properties.get("highway"), str) or not properties["highway"]:
            raise ValueError(f"Road '{identifier}' has no source highway tag.")
        raw_element = raw_by_id.get(identifier)
        if raw_element is None or raw_element.get("tags", {}).get("highway") != properties["highway"]:
            raise ValueError(f"Road '{identifier}' is not traceable to the raw Overpass response.")
        raw_coordinates = [(point["lon"], point["lat"]) for point in raw_element.get("geometry", [])]
        if len(raw_coordinates) < 2 or not geometry.equals_exact(LineString(raw_coordinates), 1e-12):
            raise ValueError(f"Road '{identifier}' geometry differs from the raw Overpass response.")
    return source_record, metadata, features


def copy_file(source: Path, destination: Path) -> None:
    if source.is_symlink() or not source.is_file():
        raise ValueError(f"Refusing to copy a non-regular artifact: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)


def derive(base: Path, osm_source: Path, output: Path) -> dict:
    base, osm_source, output = base.resolve(), osm_source.resolve(), output.resolve()
    if output.exists():
        raise ValueError(f"Output must not already exist: {output}")
    if any(root == output or root in output.parents for root in (base, osm_source)):
        raise ValueError("Output must not be inside an immutable source bundle.")
    base_manifest = validate_base(base)
    source_record, roads_metadata, road_features = validate_osm_source(osm_source, base_manifest)

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    try:
        for relative in sorted(base_manifest["artifact_sha256"]):
            copy_file(safe_child(base, relative), safe_child(staging, relative))
        copy_file(osm_source / "vectors" / "roads.geojson", staging / "vectors" / "roads.geojson")
        provenance = staging / "provenance"
        copy_file(base / "manifest.json", provenance / "spacenet-grounded-manifest.json")
        copy_file(osm_source / "source.json", provenance / "osm-source.json")
        copy_file(osm_source / "osm-query.txt", provenance / "osm-query.txt")
        copy_file(osm_source / "osm-response.json", provenance / "osm-response.json")
        copy_file(osm_source / "vectors" / "roads.geojson.metadata.json",
                  provenance / "roads-source.metadata.json")

        study = box(*map(float, base_manifest["coverage"]["bbox_wgs84"]))
        geometries = [shape(feature["geometry"]) for feature in road_features]
        geometry_bounds = [
            min(geometry.bounds[0] for geometry in geometries),
            min(geometry.bounds[1] for geometry in geometries),
            max(geometry.bounds[2] for geometry in geometries),
            max(geometry.bounds[3] for geometry in geometries),
        ]
        study_count = sum(geometry.intersects(study) for geometry in geometries)
        derived_roads_metadata = copy.deepcopy(roads_metadata)
        derived_roads_metadata.update({
            "coverage_status": "bounded_osm_snapshot_partial",
            "reference_coverage_complete": False,
            "coverage_note": (
                "Highway-tagged ways selected by the recorded bounded Overpass query. "
                "Missing, untagged, or newly changed roads cannot establish absence."
            ),
            "temporal_alignment_status": "unknown",
            "temporal_alignment_note": (
                "SpaceNet imagery acquisition dates are unavailable; the OSM snapshot date "
                "must not be treated as the imagery date or as contemporaneous evidence."
            ),
            "geometry_extent_wgs84": geometry_bounds,
            "study_intersection_feature_count": study_count,
            "halo_only_feature_count": len(geometries) - study_count,
            "raw_response": "provenance/osm-response.json",
            "query": "provenance/osm-query.txt",
            "source_metadata": "provenance/roads-source.metadata.json",
            "source_inventory": "provenance/osm-source.json",
        })
        write_json(staging / "vectors" / "roads.geojson.metadata.json", derived_roads_metadata)

        properties = {
            key for feature in road_features
            for key in feature.get("properties", {})
        }
        road_artifact_hash = sha256_file(staging / "vectors" / "roads.geojson")
        road_metadata_hash = sha256_file(staging / "vectors" / "roads.geojson.metadata.json")
        manifest = copy.deepcopy(base_manifest)
        manifest.update({
            "area_name": "SpaceNet 1 Rio training sample with OSM road context",
            "created_by": "pipeline/scripts/derive_round1_bundle.py",
            "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
            "completed_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        })
        manifest["limits"] = {**manifest.get("limits", {}), "maximum_distance_m": QUERY_RADIUS_M}
        manifest["vector_layers"]["roads"] = {
            "file": "vectors/roads.geojson",
            "aliases": ["road", "roads", "street", "streets"],
            "fields": sorted(properties),
            "source_name": "roads.geojson",
            "source_layer_name": None,
            "source_sha256": source_record["artifact_sha256"]["vectors/roads.geojson"],
            "source": roads_metadata["source"],
            "source_url": roads_metadata["source_url"],
            "source_date": roads_metadata["source_date"],
            "extraction_date": roads_metadata["extraction_date"],
            "extraction_date_status": "recorded",
            "attribution": roads_metadata["attribution"],
            "license": roads_metadata["license"],
            "coverage_status": "bounded_osm_snapshot_partial",
            "reference_coverage_complete": False,
            "maximum_query_radius_m": QUERY_RADIUS_M,
            "metadata": derived_roads_metadata,
            "metadata_sha256": road_metadata_hash,
        }
        manifest["queryable_attributes"] = sorted({
            field
            for details in manifest["vector_layers"].values()
            for field in details.get("fields", [])
        })
        manifest["preparation"] = {
            **manifest.get("preparation", {}),
            "derived_from_dataset_version": base_manifest["dataset_version"],
            "derived_from_manifest_sha256": sha256_file(base / "manifest.json"),
            "model_inference_reused": True,
            "model_inference_repeated": False,
            "added_reference_layer": "roads",
            "reference_query_radius_m": QUERY_RADIUS_M,
            "reference_source_inventory_sha256": sha256_file(osm_source / "source.json"),
        }
        manifest["attribution"] = {
            **manifest.get("attribution", {}),
            "vectors": (
                "SpaceNet Partners (CosmiQ Works, Radiant Solutions and NVIDIA) / SpaceNet 1 "
                "(CC BY-SA 4.0); OpenStreetMap contributors (ODbL 1.0)"
            ),
        }

        artifact_hashes = {
            str(path.relative_to(staging)): sha256_file(path)
            for path in sorted(staging.rglob("*"))
            if path.is_file() and path.name != "manifest.json"
        }
        if artifact_hashes["vectors/roads.geojson"] != road_artifact_hash:
            raise ValueError("Road artifact changed during derivation.")
        manifest["artifact_sha256"] = artifact_hashes
        manifest["dataset_version"] = bundle_version(
            artifact_hashes,
            manifest["image_encoder"]["checkpoint_sha256"],
            manifest["object_detector"].get("revision"),
            manifest["supported_concepts"],
            manifest["limits"],
        )
        write_json(staging / "manifest.json", manifest)

        for relative, expected in manifest["artifact_sha256"].items():
            verify_hash(safe_child(staging, relative), expected, f"derived artifact '{relative}'")
        staging.rename(output)
        return manifest
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=Path("data/spacenet-grounded"))
    parser.add_argument("--osm-source", type=Path, default=Path("data/source-spacenet-round1"))
    parser.add_argument("--output", type=Path, default=Path("data/spacenet-round1"))
    args = parser.parse_args()
    try:
        manifest = derive(args.base, args.osm_source, args.output)
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    print(f"Wrote immutable Round 1 bundle: {args.output.resolve()}")
    print(f"Dataset version: {manifest['dataset_version']}")
    print("Reused all cached image embeddings and OWLv2 detections without inference.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
