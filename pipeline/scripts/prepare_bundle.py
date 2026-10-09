#!/usr/bin/env python3
"""Create an immutable TerraLens evidence bundle using LOCAL inputs and models.

Example (from pipeline/):
  python scripts/prepare_bundle.py --area-name "Demo area" --imagery /data/aerial.tif \
    --vectors roads=/data/osm.gpkg#roads --vectors buildings=/data/osm.gpkg#buildings \
    --concept solar_panels="solar panels" --associate solar_panels=buildings \
    --checkpoint /models/skyclip.bin --max-tiles 50 --output data/demo

--metadata image_path=sidecar.json supports SkyScript box/time/center_tags.
Bounds-only images require --bounds-georeference after verifying north-up mapping.
Source tags remain annotations, never visual detections. OWLv2 weights must be
local or already cached; this script does not download model weights.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
import math
import pickle
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from PIL import Image
from pyproj import Transformer
from rasterio.crs import CRS
from rasterio.transform import from_bounds
from rasterio.windows import Window
from shapely.geometry import Polygon, mapping, shape
from shapely.strtree import STRtree


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def rgb_image(dataset, window: Window) -> np.ndarray:
    """Preserve uint8 RGB values; stretch only higher-depth inputs."""
    if dataset.count < 3:
        raise ValueError(f"{dataset.name} needs at least three RGB bands.")
    bands = dataset.read((1, 2, 3), window=window, masked=True)
    rgb = np.zeros(bands.shape, dtype=np.uint8)
    for index, band in enumerate(bands):
        valid = ~np.ma.getmaskarray(band)  # Scalar True means all pixels are nodata.
        data = np.asarray(band.filled(0))
        if data.dtype == np.uint8:
            rgb[index] = np.where(valid, data, 0)
            continue
        data = data.astype(np.float32)
        valid &= np.isfinite(data)
        sample = data[valid]
        if not sample.size:
            continue
        low, high = np.percentile(sample, (2, 98))
        if high > low:
            scaled = np.zeros(data.shape, dtype=np.float32)
            scaled[valid] = np.clip((data[valid] - low) * (255.0 / (high - low)), 0, 255)
            rgb[index] = scaled.astype(np.uint8)
    return np.moveaxis(rgb, 0, -1)


def parse_vectors(values: list[str]) -> dict[str, tuple[Path, str | None]]:
    output = {}
    for value in values:
        name, separator, source = value.partition("=")
        if not separator or not re.fullmatch(r"[a-z][a-z0-9_]{0,62}", name):
            raise ValueError(f"Vector must use name=path or name=path#layer: {value}")
        path_text, hashmark, layer = source.partition("#")
        path = Path(path_text).expanduser().resolve()
        if not path.is_file() or (hashmark and not layer):
            raise ValueError(f"Vector source or layer does not exist: {value}")
        if name in output:
            raise ValueError(f"Vector layer name repeated: {name}")
        output[name] = (path, layer if hashmark else None)
    if not output:
        raise ValueError("Provide at least one vector layer with --vectors.")
    return output


def parse_concepts(values: list[str]) -> dict[str, dict]:
    concepts = {}
    for value in values:
        concept_id, separator, prompt = value.partition("=")
        concept_id, prompt = concept_id.strip(), prompt.strip()
        if not separator or not re.fullmatch(r"[a-z][a-z0-9_]{0,62}", concept_id) or not prompt:
            raise ValueError(f"Concept must use id=grounding phrase: {value}")
        if concept_id in concepts:
            raise ValueError(f"Concept repeated: {concept_id}")
        concepts[concept_id] = {"prompt": prompt, "aliases": [prompt], "association_layer": None,
                                "grounding_available": True, "evidence_type": "model_detection"}
    if not concepts:
        raise ValueError("Provide at least one supported class with --concept id=phrase.")
    return concepts


def apply_associations(values: list[str], concepts: dict, vectors: dict) -> None:
    seen = set()
    for value in values:
        concept_id, separator, layer_id = value.partition("=")
        if not separator or concept_id not in concepts or layer_id not in vectors:
            raise ValueError(f"Association requires a configured concept and vector layer: {value}")
        if concept_id in seen:
            raise ValueError(f"Association repeated: {concept_id}")
        seen.add(concept_id)
        concepts[concept_id].update(association_layer=layer_id, association_overlap_min=0.5)


class PrimitiveMetadataUnpickler(pickle.Unpickler):
    """Read primitive SkyScript metadata without executing pickle globals."""
    def find_class(self, module, name):
        raise ValueError(f"Metadata pickle contains unsupported executable global {module}.{name}; convert to JSON in a trusted environment.")

    def persistent_load(self, pid):
        raise ValueError("Persistent metadata objects are unsupported.")


def read_metadata(path: Path) -> dict:
    if path.suffix.lower() in {".pickle", ".pkl"}:
        value = PrimitiveMetadataUnpickler(io.BytesIO(path.read_bytes())).load()
    else:
        value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"Metadata must contain a dictionary: {path}")
    json.dumps(value, allow_nan=False)
    return value


def parse_metadata(values: list[str], keys: set[str], *, path_keys: bool) -> dict[str, Path]:
    result = {}
    for value in values:
        key, separator, source = value.partition("=")
        if path_keys:
            key = str(Path(key).expanduser().resolve())
        source_path = Path(source).expanduser().resolve()
        if not separator or key not in keys or not source_path.is_file():
            raise ValueError(f"Metadata must pair a configured source with a local sidecar: {value}")
        if key in result:
            raise ValueError(f"Metadata repeated: {key}")
        result[key] = source_path
    return result


def acquisition_metadata(metadata: dict, raster_tags: dict) -> tuple[str | None, str]:
    value = metadata.get("acquisition_date") or metadata.get("time")
    origin = "sidecar"
    if value is None:
        value = raster_tags.get("acquisition_date") or raster_tags.get("DATE_ACQUIRED")
        origin = "raster_acquisition_tag"
    if value is None or value == "":
        return None, "unknown"
    if isinstance(value, (list, tuple)):
        if len(value) != 5:
            raise ValueError("SkyScript time must contain year, month, day, hour, minute.")
        # Source documentation does not specify timezone; do not invent UTC.
        return datetime(*(int(part) for part in value)).isoformat(timespec="minutes"), origin
    text = str(value).strip()
    if re.fullmatch(r"\d{4}", text):
        datetime(int(text), 1, 1)
        return text, origin
    if re.fullmatch(r"\d{4}-\d{2}", text):
        year, month = (int(part) for part in text.split("-"))
        datetime(year, month, 1)
        return text, origin
    try:
        datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"Unrecognized acquisition date: {text}") from exc
    return text, origin


def raster_georeference(dataset, metadata: dict, allow_bounds: bool):
    if dataset.crs is not None:
        return dataset.crs, dataset.transform, {"method": "raster", "registration_status": "unverified"}
    bounds = metadata.get("box") or metadata.get("bbox_wgs84")
    if not allow_bounds or not isinstance(bounds, (list, tuple)) or len(bounds) != 4:
        raise ValueError(f"Raster has no CRS: {dataset.name}. Bounds-only imagery requires a sidecar box and --bounds-georeference after verifying north-up orientation.")
    west, south, east, north = (float(value) for value in bounds)
    if not all(math.isfinite(value) for value in (west, south, east, north)) or not (-180 <= west < east <= 180 and -90 <= south < north <= 90):
        raise ValueError(f"Invalid WGS84 sidecar bounds: {bounds}")
    transform = from_bounds(west, south, east, north, dataset.width, dataset.height)
    return CRS.from_epsg(4326), transform, {"method": "sidecar_bounds", "north_up_affirmed": True,
                                            "registration_status": "bounds_approximation", "source_box_wgs84": list(bounds)}


def pixel_box_geometry(pixel_box: list[float], tile_transform, transformer: Transformer):
    """One route only: window-local ORIGINAL pixels -> window transform."""
    x0, y0, x1, y1 = pixel_box
    corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]  # TL, TR, BR, BL
    coordinates = [list(transformer.transform(*(tile_transform * corner))) for corner in corners]
    if not all(math.isfinite(value) for coordinate in coordinates for value in coordinate):
        raise ValueError("Pixel transform produced non-finite geographic coordinates.")
    return Polygon(coordinates), coordinates


def tile_origins(length: int, tile_size: int, overlap: int) -> list[int]:
    """Cover an axis without making small redundant strips at its final edge."""
    if length < 1 or tile_size < 1 or not 0 <= overlap < tile_size:
        raise ValueError("Tiling requires positive dimensions and overlap in [0,tile_size).")
    if length <= tile_size:
        return [0]
    final_origin = length - tile_size
    origins = list(range(0, final_origin + 1, tile_size - overlap))
    if origins[-1] != final_origin:
        origins.append(final_origin)
    return origins


def overlap_epoch(properties: dict) -> str:
    epoch, date = properties.get("acquisition_epoch"), properties.get("acquisition_date")
    if epoch:
        # An explicit broad epoch label never overrides a contradictory date.
        return f"epoch:{epoch}:date:{date}"
    if not date or re.fullmatch(r"\d{4}(?:-\d{2})?", str(date)):
        # A year or month alone cannot establish the same exposure across rasters.
        return f"partial-or-undated-source:{properties['source_sha256']}:date:{date}"
    return f"date:{date}"


def profile_limits(metadata_values) -> dict[str, float]:
    """Use the most restrictive declared scene profile, without inventing one."""
    maximum_distances = []
    for metadata in metadata_values:
        profile = metadata.get("profile", {})
        if not isinstance(profile, dict) or not isinstance(profile.get("limits", {}), dict):
            raise ValueError("Source profile and profile.limits must be objects.")
        value = profile.get("limits", {}).get("maximum_distance_m")
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 < value <= 5000:
            raise ValueError("profile.limits.maximum_distance_m must be finite and in (0,5000].")
        maximum_distances.append(value)
    return {"maximum_distance_m": min(maximum_distances, default=5000)}


def deduplicate_detections(features: list[dict], threshold: float = 0.5) -> list[dict]:
    if not features:
        return []
    geometries = [shape(feature["geometry"]) for feature in features]
    tree, suppressed, result = STRtree(geometries), set(), []
    ordered = sorted(range(len(features)), key=lambda index: (-features[index]["properties"]["score"], features[index]["id"]))
    for index in ordered:
        if index in suppressed:
            continue
        feature = copy.deepcopy(features[index])
        properties, geometry = feature["properties"], geometries[index]
        observations = [{**copy.deepcopy(properties), "geometry": copy.deepcopy(feature["geometry"])}]
        for candidate in tree.query(geometry):
            candidate = int(candidate)
            if candidate == index or candidate in suppressed:
                continue
            other = features[candidate]["properties"]
            if properties["concept_id"] != other["concept_id"] or overlap_epoch(properties) != overlap_epoch(other):
                continue
            # Geographic overlap screens duplicates; it is not a metric distance.
            intersection = geometry.intersection(geometries[candidate]).area
            union = geometry.area + geometries[candidate].area - intersection
            if union > 0 and intersection / union >= threshold:
                suppressed.add(candidate)
                observations.append({**copy.deepcopy(other), "geometry": copy.deepcopy(features[candidate]["geometry"])})
        suppressed.add(index)
        properties.update(observations=observations, observation_count=len(observations),
                          deduplication="same_class_same_epoch_geographic_iou")
        result.append(feature)
    return result


def load_inference(args):
    """ML imports are optional for CLI help and lightweight geospatial tests."""
    import open_clip
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor, pipeline

    selected_device = args.device
    if selected_device == "auto":
        selected_device = "cuda" if torch.cuda.is_available() else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu"
    device = torch.device(selected_device)
    image_model, _, preprocess = open_clip.create_model_and_transforms(args.model_name, pretrained=None)
    open_clip.load_checkpoint(image_model, str(args.checkpoint))
    image_model = image_model.eval().to(device)
    local_model = Path(args.owl_model).expanduser()
    model_path = local_model.resolve() if local_model.is_dir() else Path(snapshot_download(args.owl_model, revision=args.owl_revision, local_files_only=True))
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True, use_fast=False)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_path, local_files_only=True)
    detector = pipeline("zero-shot-object-detection", model=model, tokenizer=processor.tokenizer,
                        image_processor=processor.image_processor, device=device)
    revision = (args.owl_revision or getattr(model.config, "_commit_hash", None)) if local_model.is_dir() else (getattr(model.config, "_commit_hash", None) or model_path.name)
    model_files = {str(path.relative_to(model_path)): sha256_file(path)
                   for path in sorted(model_path.rglob("*")) if path.is_file()}

    def infer_image(image):
        tensor = preprocess(image).unsqueeze(0).to(device)
        with torch.no_grad():
            vector = image_model.encode_image(tensor)
            vector = vector / vector.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        vector = vector[0].float().cpu().numpy()
        if not np.isfinite(vector).all() or not np.any(vector):
            raise ValueError("Image encoder produced an invalid embedding.")
        return vector

    return infer_image, detector, revision, model_files


def write_json(path: Path, value: Any, *, pretty=False):
    path.write_text(json.dumps(value, indent=2 if pretty else None, separators=None if pretty else (",", ":"), allow_nan=False))


def bundle_version(artifact_hashes: dict, encoder_sha256: str, detector_revision: str | None,
                   concepts: dict, limits: dict) -> str:
    """Bind saved query plans to this evidence and execution configuration."""
    signature = {
        "artifact_sha256": artifact_hashes, "encoder_checkpoint_sha256": encoder_sha256,
        "detector_revision": detector_revision, "supported_concepts": concepts, "limits": limits,
    }
    canonical = json.dumps(signature, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return "terralens-evidence-v1-" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--area-name", required=True)
    parser.add_argument("--imagery", action="append", required=True)
    parser.add_argument("--vectors", action="append", default=[])
    parser.add_argument("--concept", action="append", default=[])
    parser.add_argument("--associate", action="append", default=[], help="Explicit concept=layer, e.g. solar_panels=buildings.")
    parser.add_argument("--metadata", action="append", default=[], help="image_path=sidecar.json (or primitive SkyScript .pickle).")
    parser.add_argument("--vector-metadata", action="append", default=[], help="layer=sidecar.json with source/extraction_date/license/attribution.")
    parser.add_argument("--bounds-georeference", action="store_true", help="Affirm verified north-up mapping for bounds-only imagery.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model-name", default="ViT-B-32")
    parser.add_argument("--owl-model", default="google/owlv2-base-patch16-ensemble", help="Local directory or already cached detector ID.")
    parser.add_argument("--owl-revision", default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--tile-size", type=int, default=1024)
    parser.add_argument("--overlap", type=int, default=128)
    parser.add_argument("--max-tiles", type=int, default=200, help="Bound actual encoding/detection work to this many windows.")
    parser.add_argument("--detector-threshold", type=float, default=0.1)
    parser.add_argument("--dedup-iou", type=float, default=0.5)
    parser.add_argument("--imagery-attribution", default=None)
    parser.add_argument("--vector-attribution", default=None)
    parser.add_argument("--output", default="data/demo")
    return parser


def main(argv=None) -> int:
    parser, args = build_parser(), None
    args = parser.parse_args(argv)
    try:
        vectors, concepts = parse_vectors(args.vectors), parse_concepts(args.concept)
        apply_associations(args.associate, concepts, vectors)
        imagery_paths = [Path(value).expanduser().resolve() for value in args.imagery]
        if len(set(imagery_paths)) != len(imagery_paths):
            raise ValueError("Imagery sources must not be repeated.")
        for path in imagery_paths:
            if not path.is_file():
                raise ValueError(f"Imagery does not exist: {path}")
        args.checkpoint = Path(args.checkpoint).expanduser().resolve()
        if not args.checkpoint.is_file():
            raise ValueError(f"OpenCLIP checkpoint does not exist: {args.checkpoint}")
        if args.tile_size < 224 or not 0 <= args.overlap < args.tile_size or args.max_tiles < 1:
            raise ValueError("Use tile-size >= 224, overlap in [0,tile-size), and max-tiles >= 1.")
        if not 0 < args.detector_threshold < 1 or not 0 < args.dedup_iou <= 1:
            raise ValueError("Detector threshold must be in (0,1); dedup IoU in (0,1].")
        metadata_paths = parse_metadata(args.metadata, {str(path) for path in imagery_paths}, path_keys=True)
        vector_metadata_paths = parse_metadata(args.vector_metadata, set(vectors), path_keys=False)
        output = Path(args.output).expanduser().resolve()
        if output.exists() and (not output.is_dir() or any(output.iterdir())):
            raise ValueError(f"Output must be new or empty to preserve immutable bundles: {output}")
        source_metadata = {key: read_metadata(path) for key, path in metadata_paths.items()}
        vector_metadata = {key: read_metadata(path) for key, path in vector_metadata_paths.items()}
        limits = profile_limits(source_metadata.values())
        for path in imagery_paths:
            with rasterio.open(path) as dataset:
                raster_georeference(dataset, source_metadata.get(str(path), {}), args.bounds_georeference)
                acquisition_metadata(source_metadata.get(str(path), {}), dataset.tags())
                if dataset.count < 3:
                    raise ValueError(f"{path} needs three RGB bands.")
    except (ValueError, OSError, TypeError, pickle.UnpicklingError) as exc:
        parser.error(str(exc))
    source_paths = set(imagery_paths) | {value[0] for value in vectors.values()} | set(metadata_paths.values()) | set(vector_metadata_paths.values()) | {args.checkpoint}
    source_hashes = {path: sha256_file(path) for path in source_paths}
    created_at = utc_now()
    infer_image, detector, detector_revision, detector_files = load_inference(args)
    preview_dir, vector_dir = output / "previews", output / "vectors"
    preview_dir.mkdir(parents=True, exist_ok=True)
    vector_dir.mkdir(exist_ok=True)
    tile_records, embedding_rows, detections, source_records = [], [], [], []
    total_windows, empty_windows = 0, 0

    for source_path in imagery_paths:
        source_hash = source_hashes[source_path]
        metadata, metadata_path = source_metadata.get(str(source_path), {}), metadata_paths.get(str(source_path))
        with rasterio.open(source_path) as dataset:
            crs, parent_transform, georef = raster_georeference(dataset, metadata, args.bounds_georeference)
            transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
            acquisition_date, date_source = acquisition_metadata(metadata, dataset.tags())
            epoch, metadata_hash = metadata.get("acquisition_epoch"), source_hashes.get(metadata_path)
            source_records.append({
                "source_name": source_path.name, "source_sha256": source_hash,
                "width": dataset.width, "height": dataset.height, "crs": crs.to_string(),
                "transform": list(parent_transform)[:6], "georeferencing": georef,
                "acquisition_date": acquisition_date, "acquisition_date_status": "recorded" if acquisition_date else "unknown",
                "acquisition_date_source": date_source, "raster_tags": dataset.tags(),
                "metadata": metadata, "metadata_sha256": metadata_hash, "metadata_evidence_type": "source_annotation",
                "source": metadata.get("source") or metadata.get("image_source"),
                "attribution": metadata.get("attribution") or args.imagery_attribution, "license": metadata.get("license"),
                "rgb_processing": "preserve_uint8" if all(value == "uint8" for value in dataset.dtypes[:3]) else "per_window_2_98_percentile_stretch",
            })
            row_origins = tile_origins(dataset.height, args.tile_size, args.overlap)
            column_origins = tile_origins(dataset.width, args.tile_size, args.overlap)
            total_windows += len(row_origins) * len(column_origins)
            for row in row_origins:
                if len(tile_records) >= args.max_tiles:
                    break
                for col in column_origins:
                    if len(tile_records) >= args.max_tiles:
                        break
                    window = Window(col, row, min(args.tile_size, dataset.width - col), min(args.tile_size, dataset.height - row))
                    valid_fraction = float(np.all(dataset.read_masks((1, 2, 3), window=window) > 0, axis=0).mean())
                    if valid_fraction == 0:
                        empty_windows += 1
                        continue
                    tile_seed = f"{source_hash}:{metadata_hash}:{row}:{col}:{int(window.height)}:{int(window.width)}"
                    tile_id = hashlib.sha256(tile_seed.encode()).hexdigest()[:20]
                    image, timestamp = Image.fromarray(rgb_image(dataset, window)), utc_now()
                    embedding_rows.append(infer_image(image))
                    preview, preview_path = image.copy(), preview_dir / f"{tile_id}.jpg"
                    preview.thumbnail((800, 800), Image.Resampling.LANCZOS)
                    preview.save(preview_path, quality=88, optimize=True)
                    tile_transform = rasterio.windows.transform(window, parent_transform)
                    tile_geometry, corners = pixel_box_geometry([0, 0, image.width, image.height], tile_transform, transformer)
                    tile_records.append({
                        "tile_id": tile_id, "source_name": source_path.name, "source_sha256": source_hash,
                        "preview": f"previews/{tile_id}.jpg", "preview_sha256": sha256_file(preview_path),
                        "preview_width": preview.width, "preview_height": preview.height,
                        "acquisition_date": acquisition_date, "acquisition_epoch": epoch,
                        "acquisition_date_status": "recorded" if acquisition_date else "unknown", "acquisition_date_source": date_source,
                        "crs": crs.to_string(), "transform": list(tile_transform)[:6], "width": image.width, "height": image.height,
                        "source_width": dataset.width, "source_height": dataset.height,
                        "window_offset": [col, row], "pixel_coordinate_space": "original_window_pixels",
                        "valid_rgb_fraction": valid_fraction, "georeferencing": georef,
                        "resolution_units": [math.hypot(tile_transform.a, tile_transform.d), math.hypot(tile_transform.b, tile_transform.e)],
                        "resolution_unit_name": "degrees" if crs.is_geographic else crs.linear_units,
                        "image_coordinates_wgs84": corners, "geometry": mapping(tile_geometry), "inference_timestamp": timestamp,
                        "metadata_sha256": metadata_hash,
                        "source_annotation": {"evidence_type": "source_annotation", "center_tags": metadata.get("center_tags"), "surrounding_tags": metadata.get("surrounding_tags")},
                    })
                    for concept_id, details in concepts.items():
                        for prediction in detector(image, candidate_labels=[details["prompt"]], threshold=args.detector_threshold):
                            box = prediction.get("box", {})
                            values = [float(box.get(key, 0)) for key in ("xmin", "ymin", "xmax", "ymax")]
                            score = float(prediction.get("score", 0))
                            if not all(math.isfinite(value) for value in values) or not math.isfinite(score) or not args.detector_threshold <= score <= 1:
                                continue
                            x0, y0, x1, y1 = values
                            pixel_box = [max(0, min(x0, image.width)), max(0, min(y0, image.height)), max(0, min(x1, image.width)), max(0, min(y1, image.height))]
                            if pixel_box[2] <= pixel_box[0] or pixel_box[3] <= pixel_box[1]:
                                continue
                            geographic, _ = pixel_box_geometry(pixel_box, tile_transform, transformer)
                            detection_id = hashlib.sha256(f"{tile_id}:{concept_id}:{pixel_box}:{detector_revision}:{args.detector_threshold}".encode()).hexdigest()[:20]
                            detections.append({"type": "Feature", "id": detection_id, "geometry": mapping(geographic), "properties": {
                                "detection_id": detection_id, "tile_id": tile_id, "concept_id": concept_id,
                                "label": prediction.get("label", details["prompt"]), "prompt": details["prompt"],
                                "score": score, "model_id": args.owl_model, "model_revision": detector_revision,
                                "evidence_type": "model_detection", "inference_timestamp": timestamp,
                                "pixel_box": pixel_box, "pixel_coordinate_space": "original_window_pixels",
                                "window_offset": [col, row], "source_sha256": source_hash,
                                "acquisition_date": acquisition_date, "acquisition_epoch": epoch}})
                    print(f"Prepared {len(tile_records)}/{args.max_tiles} tiles; {len(detections)} raw model detections", file=sys.stderr)
    if not tile_records:
        raise ValueError("No valid RGB raster windows were produced.")
    import geopandas as gpd
    vector_layers = {}
    for layer_name, (path, layer) in vectors.items():
        metadata, metadata_path = vector_metadata.get(layer_name, {}), vector_metadata_paths.get(layer_name)
        frame = gpd.read_file(path, layer=layer) if layer else gpd.read_file(path)
        if frame.crs is None:
            raise ValueError(f"Vector layer '{layer_name}' has no CRS.")
        frame = frame[frame.geometry.notna() & ~frame.geometry.is_empty].to_crs("EPSG:4326")
        if frame.empty or not frame.geometry.is_valid.all():
            raise ValueError(f"Vector layer '{layer_name}' has empty or invalid geometry; repair source explicitly before preparation.")
        if "source_feature_id" not in frame.columns:
            identity = next((name for name in ("osm_id", "@id", "id", "fid") if name in frame.columns), None)
            frame["source_feature_id"] = frame[identity].astype(str) if identity else frame.index.astype(str)
        frame = frame.rename(columns={name: f"source_{name}" for name in ("fid", "geom") if name in frame.columns and name != frame.geometry.name})
        for column in frame.columns:
            if column != frame.geometry.name and frame[column].dtype == object:
                frame[column] = frame[column].map(lambda value: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list, tuple)) else value)
        frame.to_file(vector_dir / f"{layer_name}.geojson", driver="GeoJSON", index=False)
        aliases = [layer_name.replace("_", " ")] + {"drains": ["drain", "drainage", "waterway"], "roads": ["road", "roads", "street"], "buildings": ["building", "buildings", "roof", "roofs"]}.get(layer_name, [])
        vector_layers[layer_name] = {
            "file": f"vectors/{layer_name}.geojson", "aliases": sorted(set(aliases)), "fields": [column for column in frame.columns if column != frame.geometry.name],
            "source_name": path.name, "source_layer_name": layer, "source_sha256": source_hashes[path], "source": metadata.get("source"),
            "extraction_date": metadata.get("extraction_date"), "extraction_date_status": "recorded" if metadata.get("extraction_date") else "unknown",
            "attribution": metadata.get("attribution") or args.vector_attribution, "license": metadata.get("license"),
            "metadata": metadata, "metadata_sha256": source_hashes.get(metadata_path),
        }
    raw_count, detections = len(detections), deduplicate_detections(detections, args.dedup_iou)
    coordinates = [coordinate for tile in tile_records for coordinate in tile["image_coordinates_wgs84"]]
    bounds = [min(value[0] for value in coordinates), min(value[1] for value in coordinates), max(value[0] for value in coordinates), max(value[1] for value in coordinates)]
    for path, expected in source_hashes.items():
        if sha256_file(path) != expected:
            raise ValueError(f"Input changed during preparation; bundle incomplete: {path}")
    np.save(output / "embeddings.npy", np.asarray(embedding_rows, dtype=np.float32), allow_pickle=False)
    write_json(output / "tiles.json", tile_records)
    write_json(output / "detections.geojson", {"type": "FeatureCollection", "features": detections})
    artifact_hashes = {str(path.relative_to(output)): sha256_file(path) for path in sorted(output.rglob("*")) if path.is_file()}
    manifest = {
        "dataset_version": bundle_version(artifact_hashes, source_hashes[args.checkpoint], detector_revision, concepts, limits),
        "area_name": args.area_name, "created_by": "pipeline/scripts/prepare_bundle.py",
        "created_at_utc": created_at, "completed_at_utc": utc_now(), "evidence_mode": "precomputed_model_inference",
        "image_encoder": {"model_name": args.model_name, "checkpoint_sha256": source_hashes[args.checkpoint], "embedding_dimension": int(embedding_rows[0].shape[0])},
        "object_detector": {"model_id": args.owl_model, "revision": detector_revision, "requested_revision": args.owl_revision, "score_threshold": args.detector_threshold, "model_files_sha256": detector_files},
        "supported_concepts": concepts, "vector_layers": vector_layers,
        "limits": limits,
        "queryable_attributes": sorted({field for details in vector_layers.values() for field in details["fields"]}),
        "source_records": source_records, "artifact_sha256": artifact_hashes,
        "preparation": {"tile_size": args.tile_size, "overlap": args.overlap, "max_tiles": args.max_tiles, "dedup_iou": args.dedup_iou, "device_requested": args.device,
                        "pixel_coordinate_space": "original_window_pixels", "corner_order": ["top_left", "top_right", "bottom_right", "bottom_left"],
                        "tile_origin_policy": "full_size_windows_with_final_edge_aligned"},
        "coverage": {"bbox_wgs84": bounds, "tile_count": len(tile_records), "detection_count": len(detections), "raw_detection_count": raw_count,
                     "source_window_count": total_windows, "empty_windows_skipped": empty_windows,
                     "preparation_truncated": len(tile_records) + empty_windows < total_windows,
                     "acquisition_dates": sorted({tile["acquisition_date"] for tile in tile_records if tile["acquisition_date"]}), "registration_status": "unverified",
                     "source_epoch_policy": "different_dates_never_merged; partial_or_unknown_dates_are_source_specific_unless_explicit_epoch"},
        "attribution": {"imagery": args.imagery_attribution or "See source_records; unknown unless supplied.", "vectors": args.vector_attribution or "See vector_layers; unknown unless supplied."},
    }
    # The manifest is written LAST, and marks a completed immutable bundle.
    write_json(output / "manifest.json", manifest, pretty=True)
    print(f"Wrote {len(tile_records)} tiles and {len(detections)} grounded detections to {output}")
    print(f"Image/text checkpoint SHA-256: {source_hashes[args.checkpoint]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
