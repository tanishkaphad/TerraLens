"""Complete per-window visual evidence for resumable geographic search jobs.

Retrieval orders every window; it never decides which windows may be grounded.
Predictions are retained under the recorded policy with no top-k/object cap.
The immutable source bundle is never extended with newly requested concepts.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import json
import math
import os
from pathlib import Path
import threading
from time import perf_counter
from typing import Any
from urllib.parse import quote
import uuid

from affine import Affine
import numpy as np
from PIL import Image
from shapely.geometry import shape

from .bundle import Bundle, BundleError, file_sha256, json_value
from .grounding import GroundingError, GroundingService, model_path, observation_geometry


class GroundingFeatures(list):
    """A feature-list interface with coverage metadata even when zero boxes occur."""
    def __init__(self, features=(), metadata: dict | None = None):
        super().__init__(features)
        self.metadata = metadata or {}


def detector_path(bundle: Bundle) -> Path:
    configured = os.getenv("GROUNDING_MODEL_PATH")
    if configured:
        return Path(configured).expanduser().resolve()
    info = bundle.manifest.get("object_detector", {})
    configured = info.get("local_path") or info.get("path")
    if configured:
        return (bundle.root / configured).expanduser().resolve()
    return model_path(bundle)


class CompleteGroundingService(GroundingService):
    """Installed Transformers detectors, independent of the legacy preview caps."""
    def _load_detector(self, source: Bundle):
        info = source.manifest.get("object_detector", {})
        path = detector_path(source)
        hashes = info.get("model_files_sha256", {})
        identity = (str(path), info.get("revision"), json.dumps(hashes, sort_keys=True),
            tuple((name, tuple(_model_stat(path / name) or ())) for name in sorted(hashes)))
        if self._detector is not None and self._model_identity == identity:
            return self._detector
        if not info.get("revision") or not hashes:
            raise GroundingError("A pinned local open-vocabulary detector and its artifact hashes are not configured.")
        for relative, expected in hashes.items():
            candidate = (path / relative).resolve()
            if path not in candidate.parents or not candidate.is_file() or file_sha256(candidate) != expected:
                raise GroundingError("The installed detector is unavailable or differs from the catalog's recorded artifact hashes.")
        try:
            import torch
            from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor, pipeline
            processor = AutoProcessor.from_pretrained(path, local_files_only=True, use_fast=False)
            model = AutoModelForZeroShotObjectDetection.from_pretrained(path, local_files_only=True)
            requested = os.getenv("GROUNDING_DEVICE", "cpu")
            if requested not in {"cpu", "cuda", "mps"}:
                raise GroundingError("GROUNDING_DEVICE must name a supported installed device.")
            self._detector = pipeline("zero-shot-object-detection", model=model,
                tokenizer=processor.tokenizer, image_processor=processor.image_processor,
                device=torch.device(requested))
            self._model_identity = identity
            return self._detector
        except GroundingError:
            raise
        except Exception as exc:
            raise GroundingError(f"The installed object-grounding processor could not load: {type(exc).__name__}.") from exc


@lru_cache(maxsize=1)
def get_search_grounding_service() -> CompleteGroundingService:
    return CompleteGroundingService()


def assess_tile(bundle: Bundle, tile_id: str) -> dict[str, Any]:
    tile_index = bundle.tile_index.get(tile_id)
    if tile_index is None:
        return {"tile_id": tile_id, "available": False, "status": "unavailable", "reason": "Unknown image window."}
    tile = bundle.tiles[tile_index]
    base = {"tile_id": tile_id, "geometry": deepcopy(tile.get("geometry")),
        "acquisition_date": tile.get("acquisition_date"), "source_sha256": tile.get("source_sha256")}
    try:
        dimensions = tile.get("width"), tile.get("height")
        if not all(isinstance(value, int) and not isinstance(value, bool) and value > 0 for value in dimensions):
            raise ValueError("Original window dimensions are missing or invalid.")
        transform = tile.get("transform")
        if not isinstance(transform, list) or len(transform) != 6 or not all(isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) for value in transform):
            raise ValueError("The image has no valid affine coordinate transform.")
        if Affine(*transform).determinant == 0:
            raise ValueError("The image coordinate transform is singular.")
        observation_geometry(tile, [0, 0, dimensions[0], dimensions[1]])
        path = bundle.asset_path(tile_id)
        if path is None:
            raise ValueError("The image source is unavailable.")
        with Image.open(path) as image:
            size = list(image.size)
            image.verify()
        return {**base, "available": True, "status": "ready", "reason": None,
            "inference_image_size": size, "original_window_size": list(dimensions)}
    except Exception as exc:
        return {**base, "available": False, "status": "unavailable", "reason": str(exc)}


def _policy(bundle: Bundle, threshold_policy: dict | float | None) -> dict:
    if threshold_policy is None:
        detector = bundle.manifest.get("object_detector", {})
        threshold_policy = detector.get("threshold_policy") or {"threshold": detector.get("score_threshold", detector.get("threshold", 0.1)),
            "id": "local-screening", "version": "unvalidated", "calibrated": False}
    policy = dict(threshold_policy) if isinstance(threshold_policy, dict) else {"threshold": threshold_policy}
    threshold = policy.get("threshold", policy.get("score_threshold"))
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise GroundingError("The grounding policy requires a finite screening threshold in [0, 1].", 422)
    policy["threshold"] = float(threshold)
    policy.setdefault("id", "requested-screening")
    policy.setdefault("version", "unvalidated")
    policy.setdefault("calibrated", False)
    policy.setdefault("score_meaning", "uncalibrated_model_screening_score")
    return json_value(policy)


def _cache_root(cache_dir: Path | str | None) -> Path:
    if cache_dir is not None:
        return Path(cache_dir).expanduser().resolve()
    configured = os.getenv("SEARCH_GROUNDING_CACHE_DIR")
    return Path(configured).expanduser().resolve() if configured else Path(__file__).resolve().parents[1] / "data" / "search-cache" / "grounding"


def _source_identity(bundle: Bundle, tile: dict, policy: dict, prompt: str, path: Path) -> dict:
    detector = bundle.manifest.get("object_detector", {})
    # Hash the actual inference pixels, not just a declared source image digest.
    return {"adapter_version": "complete-window-v1", "dataset_id": getattr(bundle, "catalog_id", bundle.manifest.get("dataset_id")),
        "root": str(bundle.root), "dataset_version": bundle.manifest.get("dataset_version"),
        "catalog_version": getattr(bundle, "catalog_version", None), "tile": tile,
        "preview_sha256": file_sha256(path), "prompt": prompt, "threshold_policy": policy,
        "detector": detector, "detector_path": str(detector_path(bundle)),
        "model_file_stats": {name: _model_stat(detector_path(bundle) / name) for name in detector.get("model_files_sha256", {})}}


def _model_stat(path: Path) -> list | None:
    try:
        stat = path.stat()
        return [stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns]
    except OSError:
        return None


def ground_tile(bundle: Bundle, tile_id: str, visual_query: str,
        threshold_policy: dict | float | None = None, *, service: GroundingService | None = None,
        cache_dir: Path | str | None = None) -> GroundingFeatures:
    if not isinstance(visual_query, str) or not visual_query.strip():
        raise GroundingError("A non-empty visual description is required.", 422)
    prompt = visual_query.strip()
    eligibility = assess_tile(bundle, tile_id)
    if not eligibility["available"]:
        raise GroundingError(f"Image window '{tile_id}' cannot be grounded: {eligibility['reason']}", 422)
    tile = bundle.tiles[bundle.tile_index[tile_id]]
    path = bundle.asset_path(tile_id)
    policy = _policy(bundle, threshold_policy)
    identity = _source_identity(bundle, tile, policy, prompt, path)
    key = hashlib.sha256(json.dumps(identity, sort_keys=True, default=str).encode()).hexdigest()
    cache_path = _cache_root(cache_dir) / f"{key}.json"
    service = service or get_search_grounding_service()
    lock = getattr(service, "_inference_lock", None) or threading.Lock()
    with lock:
        if cache_path.is_file():
            try:
                cached = json.loads(cache_path.read_text())
                if cached.get("key") == key and isinstance(cached.get("features"), list) and isinstance(cached.get("metadata"), dict):
                    metadata = deepcopy(cached["metadata"])
                    metadata["cache_hit"] = True
                    return GroundingFeatures(deepcopy(cached["features"]), metadata)
            except (OSError, ValueError, TypeError):
                pass
        detector = service._load_detector(bundle)
        started = perf_counter()
        with Image.open(path) as source:
            image = source.convert("RGB")
        timestamp = datetime.now(timezone.utc).isoformat()
        try:
            predictions = detector(image, candidate_labels=[prompt], threshold=policy["threshold"])
        except Exception as exc:
            raise GroundingError(f"Grounding failed for image window '{tile_id}': {type(exc).__name__}.") from exc
        if not isinstance(predictions, list):
            raise GroundingError("The installed detector returned an invalid prediction contract.")
        features, invalid, below_policy = [], 0, 0
        width, height = tile["width"], tile["height"]
        detector_info = deepcopy(bundle.manifest.get("object_detector", {}))
        model_id = detector_info.get("model_id") or detector_info.get("id") or detector_info.get("model_name") or "installed_zero_shot_object_detector"
        for index, prediction in enumerate(predictions):
            try:
                raw_box = prediction["box"]
                values = [float(raw_box[name]) for name in ("xmin", "ymin", "xmax", "ymax")]
                score = float(prediction["score"])
                if not all(math.isfinite(value) for value in [*values, score]) or not 0 <= score <= 1:
                    raise ValueError("Non-finite prediction.")
                if score < policy["threshold"]:
                    below_policy += 1
                    continue
                pixels = [max(0, min(width, values[0] * width / image.width)), max(0, min(height, values[1] * height / image.height)),
                    max(0, min(width, values[2] * width / image.width)), max(0, min(height, values[3] * height / image.height))]
                if pixels[2] <= pixels[0] or pixels[3] <= pixels[1]:
                    raise ValueError("Empty prediction box.")
                geometry = observation_geometry(tile, pixels)
            except (KeyError, TypeError, ValueError, OverflowError, BundleError, GroundingError):
                invalid += 1
                continue
            identifier = hashlib.sha256(f"{key}:{index}:{pixels}".encode()).hexdigest()[:32]
            evidence = {"query": prompt, "status": "pass", "evidence_type": "live_model_detection",
                "model_id": model_id, "model_revision": detector_info.get("revision"), "model": detector_info,
                "threshold_policy": policy, "source_sha256": tile.get("source_sha256"),
                "preview_sha256": identity["preview_sha256"], "tile_id": tile_id,
                "acquisition_date": tile.get("acquisition_date"), "inference_timestamp": timestamp}
            properties = {**evidence, "detection_id": identifier, "visual_query": prompt, "prompt": prompt,
                "score": score, "score_meaning": policy["score_meaning"], "evidence_status": "grounded_model_observation",
                "geometry_role": "object_box", "pixel_box": pixels, "inference_pixel_box": values,
                "pixel_coordinate_space": "original_window_pixels", "crs": tile["crs"], "transform": deepcopy(tile["transform"]),
                "inference_image_role": "prepared_rgb_preview", "inference_image_size": list(image.size),
                "source_name": tile.get("source_name"), "dataset_id": getattr(bundle, "catalog_id", bundle.manifest.get("dataset_id")),
                "dataset_version": bundle.manifest.get("dataset_version"), "cache_key": key, "visual_evidence": [evidence],
                "asset_url": "/api/assets/" + quote(tile_id, safe="")}
            properties["observations"] = [{**deepcopy(properties), "geometry": deepcopy(geometry)}]
            features.append({"type": "Feature", "id": identifier, "geometry": geometry, "bbox": list(shape(geometry).bounds), "properties": properties})
        metadata = {"status": "processed" if invalid == 0 else "processed_with_invalid_predictions", "tile_id": tile_id,
            "visual_query": prompt, "dataset_version": bundle.manifest.get("dataset_version"), "cache_key": key,
            "cache_hit": False, "model_id": model_id, "model_revision": detector_info.get("revision"),
            "threshold_policy": policy, "provenance": identity, "raw_predictions": len(predictions),
            "retained_observations": len(features), "invalid_predictions": invalid, "below_policy_predictions": below_policy,
            "observation_limit_reached": False, "inference_timestamp": timestamp,
            "inference_ms": round((perf_counter() - started) * 1000, 2),
            "absence_verified": False, "note": "No returned observation does not establish real-world absence."}
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = cache_path.with_suffix(f".{uuid.uuid4().hex}.tmp")
        try:
            temporary.write_text(json.dumps(json_value({"key": key, "features": features, "metadata": metadata}), ensure_ascii=False, allow_nan=False))
            temporary.replace(cache_path)
        finally:
            temporary.unlink(missing_ok=True)
        return GroundingFeatures(features, metadata)


def rank_scenes(bundle: Bundle, visual_query: str, tile_ids: list[str] | None = None) -> GroundingFeatures:
    """Return every requested scene ranked by a compatible image/text encoder."""
    if not isinstance(visual_query, str) or not visual_query.strip():
        raise GroundingError("A non-empty scene description is required.", 422)
    selected = list(bundle.tile_index) if tile_ids is None else list(dict.fromkeys(tile_ids))
    unknown = set(selected) - set(bundle.tile_index)
    if unknown:
        raise GroundingError("Scene retrieval references an unknown image window.", 422)
    if not selected:
        return GroundingFeatures([], {"status": "unavailable", "reason": "No image windows are available.", "tiles_ranked": 0})
    vector = bundle.encode_text(visual_query.strip())
    if vector is None:
        raise GroundingError("Scene ranking requires the matching installed text encoder; grounding can scan all image windows without retrieval ranking.")
    vector = np.asarray(vector)
    if vector.shape != (bundle.embeddings.shape[1],) or not np.isfinite(vector).all():
        raise GroundingError("The text encoder is incompatible with the prepared image embeddings.")
    scores = bundle.embeddings @ vector
    selected.sort(key=lambda identifier: (-float(scores[bundle.tile_index[identifier]]), identifier))
    features = []
    for rank, identifier in enumerate(selected):
        tile = bundle.tiles[bundle.tile_index[identifier]]
        properties = {"tile_id": identifier, "visual_query": visual_query.strip(), "prompt": visual_query.strip(),
            "rank": rank + 1, "retrieval_similarity": float(scores[bundle.tile_index[identifier]]),
            "score_meaning": "image_text_cosine_similarity_not_probability", "geometry_role": "scene_footprint",
            "evidence_type": "image_text_retrieval", "acquisition_date": tile.get("acquisition_date"),
            "source_sha256": tile.get("source_sha256"), "source_name": tile.get("source_name"),
            "asset_url": "/api/assets/" + quote(identifier, safe=""),
            "dataset_id": getattr(bundle, "catalog_id", bundle.manifest.get("dataset_id")),
            "dataset_version": bundle.manifest.get("dataset_version"), "image_encoder": deepcopy(bundle.manifest.get("image_encoder", {})),
            "visual_evidence": [{"query": visual_query.strip(), "status": "unknown", "evidence_type": "scene_similarity"}],
            "note": "A scene similarity score does not establish an object or every requested condition."}
        features.append({"type": "Feature", "id": "scene:" + identifier, "geometry": deepcopy(tile["geometry"]), "properties": properties})
    return GroundingFeatures(features, {"status": "ranked", "tiles_ranked": len(features), "truncated": False,
        "visual_query": visual_query.strip(), "image_encoder": deepcopy(bundle.manifest.get("image_encoder", {}))})
