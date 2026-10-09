"""Bounded, local open-vocabulary inference with reusable evidence runs.

The detector never downloads weights during a request. Its observations stay
separate from the immutable source bundle and from cached concept detections.
"""
from __future__ import annotations

from collections import OrderedDict
import copy
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import threading
from time import monotonic, perf_counter
from typing import Any
import uuid

from affine import Affine
from PIL import Image
from pyproj import Transformer
from pyproj.exceptions import CRSError
from shapely.geometry import Polygon, mapping, shape

from .bundle import Bundle, BundleError, file_sha256
from .planner import QueryPlan


class GroundingError(RuntimeError):
    def __init__(self, message: str, status_code: int = 503):
        super().__init__(message)
        self.status_code = status_code


@dataclass
class GroundingRun:
    identifier: str
    source_version: str
    source_identity: str
    signature: tuple
    bundle: Bundle
    metadata: dict[str, Any]
    expires_at: float


def visual_signature(plan: QueryPlan) -> tuple:
    return plan.concept_id, plan.visual_query.strip().casefold(), plan.subject_layer


def source_identity(source: Bundle) -> str:
    """Bind runs to a loaded immutable bundle, including model and artifact identity."""
    if not getattr(source, "_grounding_identity", None):
        record = {"root": str(source.root.resolve()), "manifest": source.manifest, "tiles": source.tiles,
            "model_path": str(model_path(source))}
        digest = hashlib.sha256(json.dumps(record, sort_keys=True, default=str).encode())
        digest.update(source.embeddings.tobytes())
        source._grounding_identity = digest.hexdigest()
    return source._grounding_identity


def eligible_tiles(bundle: Bundle) -> list[dict]:
    result = []
    for tile in bundle.tiles:
        transform = tile.get("transform", [])
        dimensions = [tile.get("width"), tile.get("height")]
        if not tile.get("crs") or len(transform) != 6 or not all(isinstance(value, (int, float)) and math.isfinite(value) for value in transform):
            continue
        if not all(isinstance(value, int) and 0 < value <= 8192 for value in dimensions):
            continue
        path = bundle.asset_path(tile["tile_id"])
        if path is None:
            continue
        try:
            observation_geometry(tile, [0, 0, dimensions[0], dimensions[1]])
            with Image.open(path) as image:
                if max(image.size) > 1600:
                    continue
                image.verify()
        except (OSError, ValueError, CRSError, BundleError, GroundingError):
            continue
        result.append(tile)
    return result


def model_path(bundle: Bundle) -> Path:
    configured = os.getenv("GROUNDING_MODEL_PATH")
    if configured:
        return Path(configured).expanduser().resolve()
    revision = bundle.manifest.get("object_detector", {}).get("revision", "")
    return Path(__file__).resolve().parents[1] / "data" / "models" / f"owlv2-{revision}"


def grounding_status(bundle: Bundle) -> dict:
    detector = bundle.manifest.get("object_detector", {})
    path = model_path(bundle)
    hashes = detector.get("model_files_sha256", {})
    required = ("config.json", "model.safetensors", "preprocessor_config.json")
    model_available = bool(detector.get("revision")) and all(
        name in hashes and (path / name).is_file() for name in required
    )
    usable_tiles = len(eligible_tiles(bundle))
    available = model_available and usable_tiles > 0
    return {"available": available, "model_id": "google/owlv2-base-patch16-ensemble",
        "model_revision": detector.get("revision"), "max_tiles": 4,
        "reason": None if available else "Local pinned detector or tile coordinate transforms are unavailable.",
        "status": "configured_not_loaded" if available else "unavailable", "eligible_tiles": usable_tiles,
        "note": "New prompts produce model candidates, with uncalibrated scores and bounded image coverage."}


def observation_geometry(tile: dict, pixel_box: list[float]) -> dict:
    """Convert ORIGINAL window pixels through the retained affine transform."""
    transform = tile.get("transform")
    if not tile.get("crs") or not isinstance(transform, list) or len(transform) != 6:
        raise GroundingError("The selected image has no usable coordinate transform.", 422)
    if not all(isinstance(value, (int, float)) and math.isfinite(value) for value in transform):
        raise GroundingError("The selected image transform is invalid.", 422)
    affine = Affine(*transform)
    transformer = Transformer.from_crs(tile["crs"], "EPSG:4326", always_xy=True)
    x0, y0, x1, y1 = pixel_box
    corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    polygon = Polygon([transformer.transform(*(affine @ point)) for point in corners])
    return mapping(Bundle._geometry(mapping(polygon), "Live detector observation"))


def merge_observations(features: list[dict]) -> list[dict]:
    """NMS preserves source-specific, same-date constituent observations."""
    result: list[dict] = []
    geometries = []
    for feature in sorted(features, key=lambda item: -item["properties"]["score"]):
        props = feature["properties"]
        geometry = shape(feature["geometry"])
        epoch = props.get("acquisition_date") or props.get("tile_id")
        observation = {**copy.deepcopy(props), "geometry": copy.deepcopy(feature["geometry"])}
        target = None
        for index, previous in enumerate(result):
            old = previous["properties"]
            if not props.get("source_sha256") or (old.get("source_sha256"), old.get("acquisition_date") or old.get("tile_id")) != (props.get("source_sha256"), epoch):
                continue
            union = geometry.union(geometries[index]).area
            if union > 0 and geometry.intersection(geometries[index]).area / union >= 0.5:
                target = previous
                break
        if target is None:
            retained = copy.deepcopy(feature)
            retained["properties"]["observations"] = [observation]
            result.append(retained)
            geometries.append(geometry)
        else:
            target["properties"]["observations"].append(observation)
    return result


class GroundingService:
    """One inference lock, bounded in-memory runs, and no saved user imagery."""
    def __init__(self, *, max_runs: int = 8, ttl_seconds: int = 3600):
        self.max_runs, self.ttl_seconds = max_runs, ttl_seconds
        self._runs: OrderedDict[str, GroundingRun] = OrderedDict()
        self._cache: dict[str, str] = {}
        self._lock = threading.Lock()
        self._inference_lock = threading.Lock()
        self._detector = None
        self._model_identity = None

    def _expire(self):
        for identifier, run in list(self._runs.items()):
            if run.expires_at <= monotonic():
                del self._runs[identifier]
        self._cache = {key: identifier for key, identifier in self._cache.items() if identifier in self._runs}

    def get(self, identifier: str, source: Bundle, plan: QueryPlan) -> GroundingRun:
        with self._lock:
            self._expire()
            run = self._runs.get(identifier)
            if run is None:
                raise GroundingError("This detection run expired or the server restarted. Run detection again before continuing or exporting.", 409)
            if run.source_identity != source_identity(source) or run.signature != visual_signature(plan):
                raise GroundingError("The dataset or visual target differs from this detection run. Run a new detection.", 409)
            self._runs.move_to_end(identifier)
            return run

    def _load_detector(self, source: Bundle):
        path = model_path(source)
        info = source.manifest.get("object_detector", {})
        identity = (str(path), info.get("revision"), json.dumps(info.get("model_files_sha256", {}), sort_keys=True))
        if self._detector is not None and identity == self._model_identity:
            return self._detector
        if not grounding_status(source)["available"]:
            raise GroundingError(grounding_status(source)["reason"])
        for relative, expected in info.get("model_files_sha256", {}).items():
            candidate = (path / relative).resolve()
            if path not in candidate.parents or not candidate.is_file() or file_sha256(candidate) != expected:
                raise GroundingError("Local detector integrity differs from the prepared bundle. Restore the pinned model before detection.")
        try:
            import torch
            from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor, pipeline
            processor = AutoProcessor.from_pretrained(path, local_files_only=True, use_fast=False)
            model = AutoModelForZeroShotObjectDetection.from_pretrained(path, local_files_only=True)
            # CPU is reliable for the tiny default corpus; CUDA can be opted in.
            requested = os.getenv("GROUNDING_DEVICE", "cpu")
            if requested not in {"cpu", "cuda", "mps"}:
                raise GroundingError("GROUNDING_DEVICE must be cpu, cuda or mps.")
            self._detector = pipeline("zero-shot-object-detection", model=model,
                tokenizer=processor.tokenizer, image_processor=processor.image_processor,
                device=torch.device(requested))
            self._model_identity = identity
            return self._detector
        except GroundingError:
            raise
        except Exception as exc:
            raise GroundingError("The local object detector could not load. Prepared evidence and image search remain available.") from exc

    def create(self, source: Bundle, plan: QueryPlan, *, tile_limit: int = 2, threshold: float = 0.1) -> tuple[GroundingRun, bool]:
        if not 1 <= tile_limit <= 4 or not math.isfinite(threshold) or not 0.05 <= threshold <= 0.9:
            raise GroundingError("Choose 1–4 tiles and a screening threshold from 0.05 to 0.9.", 422)
        if plan.query_type != "objects":
            raise GroundingError("Live detection requires a visual object query.", 422)
        # Serialize actual inference, but never indefinitely queue another costly request.
        if not self._inference_lock.acquire(timeout=1):
            raise GroundingError("Another detection is running. Wait for it to finish and retry.", 429)
        try:
            vector = source.encode_text(plan.visual_query)
            if vector is None:
                raise GroundingError("Detection needs the matching SkyCLIP text encoder to select image candidates.")
            scores = source.embeddings @ vector
            available = eligible_tiles(source)
            if not available:
                raise GroundingError("No prepared image has a usable transform and bounded detection preview.")
            selected = sorted(available, key=lambda tile: (-float(scores[source.tile_index[tile["tile_id"]]]), tile["tile_id"]))[:tile_limit]
            identity = source_identity(source)
            key = hashlib.sha256(json.dumps({"source": identity,
                "visual": visual_signature(plan), "tiles": [tile["tile_id"] for tile in selected],
                "threshold": threshold}, sort_keys=True).encode()).hexdigest()
            with self._lock:
                self._expire()
                cached = self._runs.get(self._cache.get(key, ""))
                if cached:
                    self._runs.move_to_end(cached.identifier)
                    return cached, True
            started = perf_counter()
            detector = self._load_detector(source)
            features = []
            capped = False
            for tile in selected:
                path = source.asset_path(tile["tile_id"])
                if path is None:
                    raise GroundingError("A selected source image is missing.")
                width, height = tile.get("width"), tile.get("height")
                if not isinstance(width, int) or not isinstance(height, int) or not 0 < width <= 8192 or not 0 < height <= 8192:
                    raise GroundingError("Original image dimensions are missing or outside the detection budget.", 422)
                try:
                    with Image.open(path) as original:
                        image = original.convert("RGB")
                except OSError as exc:
                    raise GroundingError("A selected prepared image cannot be decoded; no run was saved.") from exc
                if max(image.size) > 1600:
                    raise GroundingError("The prepared detection image exceeds the bounded inference size.", 422)
                timestamp = datetime.now(timezone.utc).isoformat()
                try:
                    predictions = detector(image, candidate_labels=[plan.visual_query], threshold=threshold)
                except Exception as exc:
                    raise GroundingError("Object inference failed; no detection result was saved.") from exc
                predictions = sorted(predictions, key=lambda prediction: float(prediction.get("score", 0)), reverse=True)
                capped |= len(predictions) > 200
                for prediction in predictions[:200]:
                    box = prediction.get("box", {})
                    values = [float(box.get(name, math.nan)) for name in ("xmin", "ymin", "xmax", "ymax")]
                    score = float(prediction.get("score", math.nan))
                    if not all(math.isfinite(value) for value in [*values, score]) or not threshold <= score <= 1:
                        continue
                    x0, y0, x1, y1 = values
                    pixels = [max(0, min(width, x0 * width / image.width)), max(0, min(height, y0 * height / image.height)),
                        max(0, min(width, x1 * width / image.width)), max(0, min(height, y1 * height / image.height))]
                    if pixels[2] <= pixels[0] or pixels[3] <= pixels[1]:
                        continue
                    identifier = hashlib.sha256(f"{key}:{tile['tile_id']}:{pixels}".encode()).hexdigest()[:24]
                    features.append({"type": "Feature", "id": identifier, "geometry": observation_geometry(tile, pixels), "properties": {
                        "detection_id": identifier, "tile_id": tile["tile_id"], "concept_id": plan.concept_id,
                        "score": score, "model_id": "google/owlv2-base-patch16-ensemble",
                        "model_revision": source.manifest.get("object_detector", {}).get("revision"), "prompt": plan.visual_query,
                        "pixel_box": pixels, "pixel_coordinate_space": "original_window_pixels",
                        "geometry_role": "visual_detection_envelope", "evidence_type": "live_model_detection",
                        "inference_timestamp": timestamp, "inference_image_role": "prepared_rgb_preview",
                        "inference_image_size": list(image.size), "source_sha256": tile.get("source_sha256"),
                        "acquisition_date": tile.get("acquisition_date")}})
            view = copy.copy(source)
            view.manifest = copy.deepcopy(source.manifest)
            concept = copy.deepcopy(view.manifest.get("supported_concepts", {}).get(plan.concept_id, {}))
            concept.update(prompt=plan.visual_query, aliases=[plan.visual_query], grounding_available=True,
                detector_threshold=threshold, association_layer=plan.subject_layer or concept.get("association_layer"))
            view.manifest.setdefault("supported_concepts", {})[plan.concept_id] = concept
            view.detections = merge_observations(features)
            identifier = uuid.uuid4().hex
            metadata = {"model_id": "google/owlv2-base-patch16-ensemble",
                "model_revision": source.manifest.get("object_detector", {}).get("revision"), "visual_query": plan.visual_query,
                "threshold": threshold, "threshold_meaning": "uncalibrated_screening_cutoff",
                "tiles_processed": len(selected), "tiles_available": len(source.tiles), "eligible_tiles": len(available),
                "selected_tile_ids": [tile["tile_id"] for tile in selected],
                "raw_observations": len(features), "deduplicated_observations": len(view.detections),
                "observation_limit_reached": capped, "inference_ms": round((perf_counter() - started) * 1000, 2),
                "inference_image_role": "prepared_rgb_preview", "expires_in_seconds": self.ttl_seconds,
                "scope": "Model candidates from selected image tiles. This is not an exhaustive object survey or proof of absence."}
            view.manifest["live_grounding"] = {"run_id": identifier, **metadata}
            run = GroundingRun(identifier, source.manifest["dataset_version"], identity, visual_signature(plan), view, metadata, monotonic() + self.ttl_seconds)
            with self._lock:
                self._runs[identifier] = run
                self._cache[key] = identifier
                while len(self._runs) > self.max_runs:
                    self._runs.popitem(last=False)
                self._expire()
            return run, False
        finally:
            self._inference_lock.release()
