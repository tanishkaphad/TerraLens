#!/usr/bin/env python3
"""Create a TerraLens evidence bundle from georeferenced rasters and vector layers.

Example:
  python scripts/prepare_bundle.py --area-name "Demo area" --imagery /data/aerial.tif \
    --vectors roads=/data/osm.gpkg#roads --vectors buildings=/data/osm.gpkg#buildings \
    --concept solar_panels="solar panels" --checkpoint /models/skyclIP.bin \
    --output data/demo

The source imagery and vectors must already be available and licensed for this use.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import open_clip
import rasterio
import torch
from PIL import Image
from pyproj import Transformer
from rasterio.windows import Window
from shapely.geometry import Polygon, mapping
from shapely.ops import transform as transform_geometry
from transformers import pipeline


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rgb_image(dataset: rasterio.DatasetReader, window: Window) -> np.ndarray:
    if dataset.count < 3:
        raise ValueError(f"{dataset.name} needs at least three RGB bands.")
    bands = dataset.read((1, 2, 3), window=window, masked=True)
    rgb = np.zeros(bands.shape, dtype=np.uint8)
    for index, band in enumerate(bands):
        data = np.asarray(band.filled(0), dtype=np.float32)
        valid = np.asarray(~band.mask) if np.ndim(band.mask) else np.ones(data.shape, dtype=bool)
        sample = data[valid]
        if sample.size == 0:
            continue
        low, high = np.percentile(sample, (2, 98))
        if high <= low:
            high = low + 1
        rgb[index] = np.clip((data - low) * (255.0 / (high - low)), 0, 255).astype(np.uint8)
    return np.moveaxis(rgb, 0, -1)


def parse_vectors(values: list[str]) -> dict[str, tuple[Path, str | None]]:
    output: dict[str, tuple[Path, str | None]] = {}
    for value in values:
        name, separator, source = value.partition("=")
        if not separator or not re.fullmatch(r"[a-z][a-z0-9_]{0,62}", name):
            raise ValueError(f"Vector must use name=path or name=path#layer: {value}")
        path_text, hashmark, layer = source.partition("#")
        path = Path(path_text).expanduser().resolve()
        if not path.is_file():
            raise ValueError(f"Vector source does not exist: {path}")
        if name in output:
            raise ValueError(f"Vector layer name repeated: {name}")
        output[name] = (path, layer if hashmark else None)
    if not output:
        raise ValueError("Provide at least one vector layer with --vectors.")
    return output


def parse_concepts(values: list[str]) -> dict[str, dict[str, object]]:
    concepts: dict[str, dict[str, object]] = {}
    for value in values:
        concept_id, separator, prompt = value.partition("=")
        concept_id = concept_id.strip()
        prompt = prompt.strip()
        if not separator or not re.fullmatch(r"[a-z][a-z0-9_]{0,62}", concept_id) or not prompt:
            raise ValueError(f"Concept must use id=grounding phrase: {value}")
        if concept_id in concepts:
            raise ValueError(f"Concept repeated: {concept_id}")
        concepts[concept_id] = {"prompt": prompt, "aliases": [prompt], "association_layer": None}
    if not concepts:
        raise ValueError("Provide at least one supported class with --concept id=phrase.")
    return concepts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--area-name", required=True)
    parser.add_argument("--imagery", action="append", required=True, help="Georeferenced raster; repeat for more inputs.")
    parser.add_argument("--vectors", action="append", default=[], help="Layer spec name=path.gpkg#layer; repeat per layer.")
    parser.add_argument("--concept", action="append", default=[], help="Grounding concept id=prompt; repeat per class.")
    parser.add_argument("--checkpoint", required=True, help="Local OpenCLIP checkpoint used by both image and text towers.")
    parser.add_argument("--model-name", default="ViT-B-32")
    parser.add_argument("--owl-model", default="google/owlv2-base-patch16-ensemble")
    parser.add_argument("--owl-revision", default=None, help="Optional Hugging Face commit SHA for reproducible detector weights.")
    parser.add_argument("--tile-size", type=int, default=1024)
    parser.add_argument("--overlap", type=int, default=128)
    parser.add_argument("--detector-threshold", type=float, default=0.1)
    parser.add_argument("--output", default="data/demo")
    args = parser.parse_args()

    try:
        vectors = parse_vectors(args.vectors)
        concepts = parse_concepts(args.concept)
        imagery_paths = [Path(value).expanduser().resolve() for value in args.imagery]
        for image_path in imagery_paths:
            if not image_path.is_file():
                raise ValueError(f"Imagery does not exist: {image_path}")
        checkpoint = Path(args.checkpoint).expanduser().resolve()
        if not checkpoint.is_file():
            raise ValueError(f"OpenCLIP checkpoint does not exist: {checkpoint}")
        if args.tile_size < 224 or not 0 <= args.overlap < args.tile_size:
            raise ValueError("Use tile-size >= 224 and overlap in [0, tile-size).")
        if not 0 < args.detector_threshold < 1:
            raise ValueError("Detector threshold must be between 0 and 1.")
    except ValueError as exc:
        parser.error(str(exc))

    output = Path(args.output).expanduser().resolve()
    preview_dir = output / "previews"
    preview_dir.mkdir(parents=True, exist_ok=True)
    vector_dir = output / "vectors"
    vector_dir.mkdir(exist_ok=True)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    image_model, _, preprocess = open_clip.create_model_and_transforms(args.model_name, pretrained=None)
    open_clip.load_checkpoint(image_model, str(checkpoint))
    image_model = image_model.eval().to(device)
    detector = pipeline("zero-shot-object-detection", model=args.owl_model, revision=args.owl_revision, device=device)
    detector_revision = getattr(getattr(detector.model, "config", None), "_commit_hash", None) or args.owl_revision

    tile_records: list[dict[str, object]] = []
    embedding_rows: list[np.ndarray] = []
    detection_features: list[dict[str, object]] = []
    for source_path in imagery_paths:
        source_hash = sha256_file(source_path)
        with rasterio.open(source_path) as dataset:
            if dataset.crs is None:
                raise ValueError(f"Raster has no coordinate reference system: {source_path}")
            transformer = Transformer.from_crs(dataset.crs, "EPSG:4326", always_xy=True)
            step = args.tile_size - args.overlap
            for row in range(0, dataset.height, step):
                for col in range(0, dataset.width, step):
                    window = Window(col, row, min(args.tile_size, dataset.width - col), min(args.tile_size, dataset.height - row))
                    rgb = rgb_image(dataset, window)
                    if rgb.size == 0:
                        continue
                    tile_seed = f"{source_hash}:{row}:{col}:{int(window.height)}:{int(window.width)}"
                    tile_id = hashlib.sha256(tile_seed.encode()).hexdigest()[:20]
                    image = Image.fromarray(rgb, mode="RGB")
                    tensor = preprocess(image).unsqueeze(0).to(device)
                    with torch.no_grad():
                        vector = image_model.encode_image(tensor)
                        vector = vector / vector.norm(dim=-1, keepdim=True).clamp(min=1e-12)
                    embedding_rows.append(vector[0].float().cpu().numpy())

                    preview_path = preview_dir / f"{tile_id}.jpg"
                    preview = image.copy()
                    preview.thumbnail((800, 800), Image.Resampling.LANCZOS)
                    preview.save(preview_path, quality=88, optimize=True)

                    tile_transform = dataset.window_transform(window)
                    pixel_corners = [(0, 0), (window.width, 0), (window.width, window.height), (0, window.height)]
                    map_corners = [tile_transform * point for point in pixel_corners]
                    lonlat_corners = [transformer.transform(x, y) for x, y in map_corners]
                    tile_geometry = Polygon(lonlat_corners)
                    tile_tags = dataset.tags()
                    acquisition_date = tile_tags.get("acquisition_date") or tile_tags.get("DATE_ACQUIRED") or tile_tags.get("TIFFTAG_DATETIME")
                    tile_records.append({
                        "tile_id": tile_id, "source_name": source_path.name, "source_sha256": source_hash,
                        "preview": f"previews/{tile_id}.jpg", "acquisition_date": acquisition_date,
                        "crs": dataset.crs.to_string(), "transform": list(tile_transform)[:6],
                        "width": int(window.width), "height": int(window.height),
                        "resolution_units": [abs(tile_transform.a), abs(tile_transform.e)],
                        "geometry": mapping(tile_geometry),
                    })

                    for concept_id, details in concepts.items():
                        predictions = detector(image, candidate_labels=[details["prompt"]], threshold=args.detector_threshold)
                        for prediction in predictions:
                            box = prediction.get("box", {})
                            x0, y0 = float(box.get("xmin", 0)), float(box.get("ymin", 0))
                            x1, y1 = float(box.get("xmax", 0)), float(box.get("ymax", 0))
                            x0, x1 = max(0, min(x0, image.width)), max(0, min(x1, image.width))
                            y0, y1 = max(0, min(y0, image.height)), max(0, min(y1, image.height))
                            if x1 <= x0 or y1 <= y0:
                                continue
                            pixels = [
                                tile_transform * (x0, y0), tile_transform * (x1, y0),
                                tile_transform * (x1, y1), tile_transform * (x0, y1),
                            ]
                            geographic = Polygon([transformer.transform(x, y) for x, y in pixels])
                            detection_id = hashlib.sha256(f"{tile_id}:{concept_id}:{x0:.1f}:{y0:.1f}:{x1:.1f}:{y1:.1f}".encode()).hexdigest()[:20]
                            detection_features.append({
                                "type": "Feature", "id": detection_id, "geometry": mapping(geographic),
                                "properties": {
                                    "detection_id": detection_id, "tile_id": tile_id,
                                    "concept_id": concept_id, "label": prediction.get("label", details["prompt"]),
                                    "score": float(prediction.get("score", 0)), "model_id": args.owl_model,
                                    "model_revision": detector_revision,
                                    "pixel_box": [x0, y0, x1, y1], "window_offset": [int(col), int(row)],
                                },
                            })
                    print(f"Prepared {len(tile_records)} tiles; {len(detection_features)} grounded detections", file=sys.stderr)

    if not tile_records:
        raise ValueError("No raster windows were produced.")
    vector_layers: dict[str, dict[str, object]] = {}
    for layer_name, (path, layer) in vectors.items():
        frame = gpd.read_file(path, layer=layer) if layer else gpd.read_file(path)
        if frame.crs is None:
            raise ValueError(f"Vector layer '{layer_name}' has no CRS.")
        frame = frame[frame.geometry.notna() & ~frame.geometry.is_empty].to_crs("EPSG:4326")
        if frame.empty:
            raise ValueError(f"Vector layer '{layer_name}' has no non-empty geometries.")
        if any(name in {"fid", "geom"} for name in frame.columns if name != frame.geometry.name):
            frame = frame.rename(columns={name: f"source_{name}" for name in ("fid", "geom") if name in frame.columns and name != frame.geometry.name})
        for column in frame.columns:
            if column != frame.geometry.name and frame[column].dtype == object:
                sample = frame[column].dropna().head(1)
                if len(sample) and isinstance(sample.iloc[0], (dict, list, tuple)):
                    frame[column] = frame[column].map(lambda value: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list, tuple)) else value)
        frame.to_file(vector_dir / f"{layer_name}.geojson", driver="GeoJSON", index=False)
        aliases = [layer_name.replace("_", " ")]
        aliases.extend({"drains": ["drain", "drainage", "waterway"], "roads": ["road", "roads", "street"], "buildings": ["building", "buildings", "roof", "roofs"]}.get(layer_name, []))
        vector_layers[layer_name] = {"file": f"vectors/{layer_name}.geojson", "aliases": sorted(set(aliases)), "fields": [column for column in frame.columns if column != frame.geometry.name]}

    # Estimate overall area and preserve a WGS84 bounding box for the UI.
    tile_geometries = [gpd.GeoSeries([Polygon(tile["geometry"]["coordinates"][0])], crs="EPSG:4326").iloc[0] for tile in tile_records]
    coverage_frame = gpd.GeoDataFrame(geometry=tile_geometries, crs="EPSG:4326")
    coverage_bounds = [float(value) for value in coverage_frame.total_bounds]
    checkpoint_hash = sha256_file(checkpoint)
    field_names = sorted({field for details in vector_layers.values() for field in details["fields"]})
    if "buildings" in vector_layers:
        for details in concepts.values():
            details["association_layer"] = "buildings"
    manifest = {
        "dataset_version": "terralens-demo-v1", "area_name": args.area_name,
        "created_by": "pipeline/scripts/prepare_bundle.py",
        "image_encoder": {"model_name": args.model_name, "checkpoint_sha256": checkpoint_hash, "embedding_dimension": int(embedding_rows[0].shape[0])},
        "object_detector": {"model_id": args.owl_model, "revision": detector_revision, "score_threshold": args.detector_threshold},
        "supported_concepts": concepts, "vector_layers": vector_layers,
        "queryable_attributes": field_names,
        "coverage": {"bbox_wgs84": coverage_bounds, "tile_count": len(tile_records), "detection_count": len(detection_features), "acquisition_dates": sorted({tile["acquisition_date"] for tile in tile_records if tile["acquisition_date"]})},
        "attribution": {"imagery": "See source imagery terms and metadata.", "vectors": "Record source attribution and extraction date for every input layer."},
    }
    np.save(output / "embeddings.npy", np.asarray(embedding_rows, dtype=np.float32), allow_pickle=False)
    (output / "tiles.json").write_text(json.dumps(tile_records, separators=(",", ":")))
    (output / "detections.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": detection_features}, separators=(",", ":")))
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Wrote {len(tile_records)} tiles and {len(detection_features)} detections to {output}")
    print(f"Image/text checkpoint SHA-256: {checkpoint_hash}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
