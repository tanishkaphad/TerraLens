"""Discover local evidence and expose its observed schema without place whitelists.

Catalog refreshes inspect source files; they do not download or load model weights.
An explicitly selected dataset is required for new searches. ``legacy_default_id``
exists only to keep older, explicitly dataset-serving endpoints operational.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import re
import threading
from typing import Any

import geopandas as gpd
import numpy as np
from shapely.geometry import mapping
from shapely.ops import unary_union

from .bundle import Bundle, BundleError, file_sha256, json_value
from .search_grounding import assess_tile, detector_path


class CatalogError(BundleError):
    pass


def _safe_id(value: str) -> str:
    slug = re.sub(r"[^a-z0-9_-]+", "-", value.casefold()).strip("-_")
    if not slug:
        slug = "dataset-" + hashlib.sha256(value.encode()).hexdigest()[:12]
    if len(slug) > 40:
        slug = slug[:27].rstrip("-_") + "-" + hashlib.sha256(value.encode()).hexdigest()[:12]
    return slug


def _file_stat(path: Path) -> tuple:
    try:
        stat = path.stat()
        return str(path), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns
    except OSError:
        return str(path), None


def _source_version(root: Path, manifest: dict) -> str:
    """Invalidate loaded evidence even when the operator forgot a version bump."""
    files = {root / "manifest.json", root / "tiles.json", root / "embeddings.npy", root / "detections.geojson"}
    for layer_id, layer in manifest.get("vector_layers", {}).items():
        if isinstance(layer, dict):
            files.add(root / layer.get("file", f"vectors/{layer_id}.geojson"))
    for relative in manifest.get("artifact_sha256", {}):
        files.add(root / relative)
    try:
        for tile in json.loads((root / "tiles.json").read_text()):
            if isinstance(tile, dict) and tile.get("preview"):
                files.add(root / tile["preview"])
    except (OSError, ValueError, TypeError):
        pass
    record = {"manifest": manifest, "files": [_file_stat(path) for path in sorted(files)]}
    return hashlib.sha256(json.dumps(record, sort_keys=True, default=str).encode()).hexdigest()


def _geometry_counts(frame: gpd.GeoDataFrame) -> dict:
    """Retain enumeration gaps before the usable-geometry view is constructed."""
    missing = int(frame.geometry.isna().sum())
    empty = int(frame.geometry.is_empty.sum())
    return {"source_record_count": len(frame), "feature_count": len(frame) - missing - empty,
        "omitted_geometry": missing + empty,
        "omitted_geometry_reasons": {"null_geometry": missing, "empty_geometry": empty}}


class VectorBundle(Bundle):
    """The Bundle contract for valid vector sources with no prepared image index."""
    def __init__(self, root: Path, manifest: dict):
        self.root = root.resolve()
        self.manifest = deepcopy(manifest)
        self.tiles, self.detections, self.tile_index = [], [], {}
        self.embeddings = np.empty((0, 0), dtype=np.float32)
        self._text_model = self._tokenizer = None
        self._encoder_attempted = True
        self._encoder_lock = threading.RLock()
        self.text_model_error = "image_index_unavailable"
        self.layers = {}
        self.vector_geometry_metadata = {}
        for layer_id, info in manifest.get("vector_layers", {}).items():
            if not isinstance(info, dict):
                raise BundleError(f"Vector layer '{layer_id}' metadata must be an object.")
            path = self.local_path(info.get("file", f"vectors/{layer_id}.geojson"))
            if not path.is_file():
                raise BundleError(f"Vector file for layer '{layer_id}' is missing.")
            expected = manifest.get("artifact_sha256", {}).get(str(path.relative_to(self.root)))
            if expected and file_sha256(path) != expected:
                raise BundleError(f"Bundle artifact integrity check failed: {path.name}.")
            source_layer = info.get("source_layer")
            frame = gpd.read_file(path, layer=source_layer) if source_layer and path.suffix.lower() != ".geojson" else gpd.read_file(path)
            if frame.crs is None:
                raise BundleError(f"Vector layer '{layer_id}' has no CRS.")
            self.vector_geometry_metadata[str(layer_id)] = _geometry_counts(frame)
            frame = frame[~frame.geometry.isna() & ~frame.geometry.is_empty].to_crs("EPSG:4326").reset_index(drop=True)
            for geometry in frame.geometry:
                self._geometry(mapping(geometry), f"Vector layer '{layer_id}'")
            if set(info.get("fields", [])) - set(frame.columns):
                raise BundleError(f"Declared fields missing from vector layer '{layer_id}'.")
            self.layers[str(layer_id)] = frame
        if not self.layers:
            raise BundleError("The source has neither a prepared image index nor a usable vector layer.")


class SourceCatalog:
    def __init__(self, roots: list[Path | str] | None = None, bundle_dir: Path | str | None = None):
        if roots is None:
            configured = os.getenv("BUNDLE_CATALOG_ROOTS")
            roots = [Path(value) for value in configured.split(os.pathsep) if value.strip()] if configured else [Path(__file__).resolve().parents[1] / "data"]
        self.roots = tuple(dict.fromkeys(Path(value).expanduser().resolve() for value in roots))
        explicit = bundle_dir if bundle_dir is not None else os.getenv("BUNDLE_DIR")
        self.bundle_dir = Path(explicit).expanduser().resolve() if explicit else None
        self._lock = threading.RLock()
        self._records: dict[str, dict] = {}
        self._bundles: dict[str, tuple[str, Bundle]] = {}
        self._descriptions: dict[str, tuple[str, dict]] = {}
        self._windows: dict[str, tuple[str, list[dict]]] = {}
        self._hashes: dict[tuple, str] = {}
        self._errors: list[dict] = []
        self.revision = ""
        self.refresh(force=True)

    def refresh(self, force: bool = False) -> bool:
        """Rediscover additions, removals and modified versions on each catalog read."""
        with self._lock:
            candidates: dict[Path, str] = {}
            for root in self.roots:
                if not root.exists():
                    continue
                for path in sorted(root.rglob("manifest.json")):
                    relative = path.parent.relative_to(root)
                    candidates[path.parent.resolve()] = str(relative) if str(relative) != "." else root.name
            if self.bundle_dir is not None:
                candidates[self.bundle_dir] = candidates.get(self.bundle_dir, self.bundle_dir.name)
            entries, errors = [], []
            for root, relative in sorted(candidates.items()):
                try:
                    manifest = json.loads((root / "manifest.json").read_text())
                    if not isinstance(manifest, dict):
                        raise ValueError("Manifest must be an object.")
                    # Model/package manifests are not local geographic datasets.
                    if "vector_layers" not in manifest and not (root / "tiles.json").is_file():
                        continue
                    identifier = _safe_id(str(manifest.get("dataset_id") or relative))
                    entries.append({"id": identifier, "path": root, "manifest": manifest,
                        "catalog_version": _source_version(root, manifest)})
                except (OSError, ValueError, TypeError) as exc:
                    errors.append({"path": str(root), "status": "unavailable", "reason": str(exc)})
            collisions = {}
            for entry in entries:
                collisions.setdefault(entry["id"], []).append(entry)
            records = {}
            for identifier, group in collisions.items():
                for entry in group:
                    actual = identifier if len(group) == 1 else identifier[:27].rstrip("-_") + "-" + hashlib.sha256(str(entry["path"]).encode()).hexdigest()[:12]
                    entry["id"] = actual
                    records[actual] = entry
            revision = hashlib.sha256(json.dumps({"records": [(key, item["catalog_version"]) for key, item in sorted(records.items())], "errors": errors}, sort_keys=True).encode()).hexdigest()
            changed = force or revision != self.revision
            if changed:
                self._bundles = {key: value for key, value in self._bundles.items() if key in records and value[0] == records[key]["catalog_version"]}
                self._descriptions = {key: value for key, value in self._descriptions.items() if key in records and value[0] == records[key]["catalog_version"]}
                self._windows = {key: value for key, value in self._windows.items() if key in records and value[0] == records[key]["catalog_version"]}
                self._records, self._errors, self.revision = records, errors, revision
            return changed

    @property
    def legacy_default_id(self) -> str | None:
        self.refresh()
        configured = os.getenv("DEFAULT_DATASET_ID")
        if configured is not None:
            return configured if configured in self._records else None
        if self.bundle_dir is not None:
            return next((key for key, item in self._records.items() if item["path"] == self.bundle_dir), None)
        return next(iter(sorted(self._records)), None)

    def datasets(self) -> dict[str, dict]:
        self.refresh()
        with self._lock:
            default = self.legacy_default_id
            result = {}
            for identifier, item in sorted(self._records.items()):
                manifest = item["manifest"]
                attribution = manifest.get("attribution", {})
                source = manifest.get("source") or manifest.get("source_name") or attribution
                result[identifier] = {"id": identifier, "path": item["path"],
                    "name": manifest.get("area_name") or manifest.get("name") or identifier,
                    "source": source, "default": identifier == default,
                    "dataset_version": manifest.get("dataset_version") or item["catalog_version"],
                    "catalog_version": item["catalog_version"]}
            return result

    def get_bundle(self, dataset_id: str | None = None) -> Bundle:
        if not dataset_id:
            raise CatalogError("Select a dataset or an explicit search area; no default geography is used for new searches.")
        self.refresh()
        with self._lock:
            item = self._records.get(dataset_id)
            if item is None:
                raise CatalogError("The selected dataset is absent from the configured source catalog.")
            cached = self._bundles.get(dataset_id)
            if cached and cached[0] == item["catalog_version"]:
                return cached[1]
            root = item["path"]
            prepared = all((root / name).is_file() for name in ("tiles.json", "detections.geojson", "embeddings.npy"))
            if prepared:
                try:
                    bundle = Bundle(root)
                except BundleError as exc:
                    # Independently valid vectors remain useful when image/index
                    # evidence fails. VectorBundle validates those files again.
                    bundle = VectorBundle(root, item["manifest"])
                    bundle.imagery_unavailable_reason = str(exc)
            else:
                bundle = VectorBundle(root, item["manifest"])
                bundle.imagery_unavailable_reason = "No complete prepared image index is available."
            bundle.catalog_id = dataset_id
            bundle.catalog_version = item["catalog_version"]
            if not hasattr(bundle, "vector_geometry_metadata"):
                bundle.vector_geometry_metadata = {}
                for layer_id, info in bundle.manifest.get("vector_layers", {}).items():
                    path = bundle.local_path(info.get("file", f"vectors/{layer_id}.geojson"))
                    source_layer = info.get("source_layer")
                    original = gpd.read_file(path, layer=source_layer) if source_layer and path.suffix.lower() != ".geojson" else gpd.read_file(path)
                    bundle.vector_geometry_metadata[layer_id] = _geometry_counts(original)
            self._bundles[dataset_id] = item["catalog_version"], bundle
            return bundle

    def layer_features(self, dataset_id: str, layer_id: str) -> gpd.GeoDataFrame:
        bundle = self.get_bundle(dataset_id)
        if layer_id not in bundle.layers:
            raise CatalogError(f"Layer '{layer_id}' is unavailable in the selected dataset.")
        return bundle.layers[layer_id].copy(deep=True)

    def _hash(self, path: Path) -> str:
        key = _file_stat(path)
        if key not in self._hashes:
            self._hashes[key] = file_sha256(path)
        return self._hashes[key]

    def _capabilities(self, bundle: Bundle) -> dict:
        encoder = bundle.manifest.get("image_encoder", {})
        checkpoint = os.getenv("ENCODER_CHECKPOINT")
        checkpoint_path = Path(checkpoint).expanduser().resolve() if checkpoint else None
        retrieval_ready = bool(len(bundle.tiles) and checkpoint_path and checkpoint_path.is_file() and encoder.get("checkpoint_sha256"))
        if retrieval_ready:
            retrieval_ready = self._hash(checkpoint_path) == encoder["checkpoint_sha256"]
        detector = bundle.manifest.get("object_detector", {})
        path = detector_path(bundle)
        required = ("config.json", "model.safetensors", "preprocessor_config.json")
        hashes = detector.get("model_files_sha256", {})
        detector_ready = bool(detector.get("revision") and all(name in hashes and (path / name).is_file() for name in required))
        if detector_ready:
            detector_ready = all(path in (path / name).resolve().parents and (path / name).is_file() and self._hash(path / name) == expected for name, expected in hashes.items())
        identity, version = bundle.catalog_id, bundle.catalog_version
        window_cache = self._windows.get(identity)
        if window_cache is None or window_cache[0] != version:
            windows = [assess_tile(bundle, tile["tile_id"]) for tile in bundle.tiles]
            self._windows[identity] = version, windows
        else:
            windows = window_cache[1]
        eligible = sum(item["available"] for item in windows)
        return {"mapped_features": {"available": bool(bundle.layers)},
            "scene_retrieval": {"available": bool(retrieval_ready), "model": deepcopy(encoder),
                "status": "configured_not_loaded" if retrieval_ready else "unavailable", "reason": None if retrieval_ready else getattr(bundle, "imagery_unavailable_reason", "A matching local text encoder and image index are required.")},
            "object_grounding": {"available": bool(detector_ready and eligible), "model": deepcopy(detector),
                "status": "configured_not_loaded" if detector_ready and eligible else "unavailable", "eligible_windows": eligible,
                "reason": None if detector_ready and eligible else getattr(bundle, "imagery_unavailable_reason", "A locally verified detector and readable georeferenced windows are required.")},
            "windows": windows, "temporal_change": {"available": False, "reason": "No temporal processor is registered for this source."}}

    def describe(self) -> dict[str, Any]:
        self.refresh()
        with self._lock:
            datasets, layers, errors = {}, {}, deepcopy(self._errors)
            for identifier, details in self.datasets().items():
                item = self._records[identifier]
                try:
                    bundle = self.get_bundle(identifier)
                    cached = self._descriptions.get(identifier)
                    if cached and cached[0] == item["catalog_version"]:
                        description = deepcopy(cached[1])
                        capabilities = self._capabilities(bundle)
                        description.update(capabilities=capabilities,
                            live_grounding=capabilities["object_grounding"]["available"],
                            retrieval=capabilities["scene_retrieval"]["available"])
                        # A legacy serving default can change without a source change.
                        description.update(details)
                        datasets[identifier] = description
                        layers.update({identifier + ":" + key: value for key, value in description["layers"].items()})
                        continue
                    dataset_layers = {}
                    footprints = [Bundle._geometry(tile["geometry"], "Tile coverage") for tile in bundle.tiles]
                    for layer_id, frame in bundle.layers.items():
                        info = deepcopy(bundle.manifest.get("vector_layers", {}).get(layer_id, {}))
                        fields = [str(field) for field in frame.columns if field != frame.geometry.name]
                        observed = {}
                        missing = {}
                        for field in fields:
                            values = {}
                            for value in frame[field]:
                                converted = json_value(value)
                                if converted is None:
                                    continue
                                values[json.dumps(converted, sort_keys=True)] = converted
                            observed[field] = [values[key] for key in sorted(values)]
                            missing[field] = int(frame[field].isna().sum())
                        geometry = unary_union(list(frame.geometry)) if len(frame) else None
                        if geometry is not None:
                            footprints.append(geometry)
                        metadata = {**info, "dataset_id": identifier, "layer_id": layer_id,
                            "catalog_key": identifier + ":" + layer_id, "fields": fields, "observed_values": observed,
                            "missing_values": missing, "field_types": {field: str(frame[field].dtype) for field in fields},
                            "geometry_types": sorted(set(frame.geometry.geom_type)), "feature_count": len(frame),
                            "crs": "EPSG:4326", "footprint": mapping(geometry) if geometry is not None else None,
                            "dates": {key: value for key, value in info.items() if any(token in key.casefold() for token in ("date", "time"))}}
                        metadata.update(bundle.vector_geometry_metadata.get(layer_id, {}))
                        dataset_layers[layer_id] = metadata
                        layers[identifier + ":" + layer_id] = metadata
                    imagery_dates = sorted({str(tile["acquisition_date"]) for tile in bundle.tiles if tile.get("acquisition_date")})
                    capabilities = self._capabilities(bundle)
                    datasets[identifier] = {**details, "manifest": deepcopy(bundle.manifest), "status": "available",
                        "imagery_unavailable_reason": getattr(bundle, "imagery_unavailable_reason", None),
                        "layers": dataset_layers, "fields": sorted({field for layer in dataset_layers.values() for field in layer["fields"]}),
                        "footprint": mapping(unary_union(footprints)) if footprints else None,
                        "coverage": deepcopy(bundle.manifest.get("coverage", {})),
                        "dates": {"imagery": imagery_dates, "unknown_imagery_dates": sum(not tile.get("acquisition_date") for tile in bundle.tiles)},
                        "provenance": bundle.provenance(), "capabilities": capabilities,
                        "live_grounding": capabilities["object_grounding"]["available"],
                        "retrieval": capabilities["scene_retrieval"]["available"],
                        "attribute_examples": {key: value["observed_values"] for key, value in dataset_layers.items()}}
                    self._descriptions[identifier] = item["catalog_version"], deepcopy(datasets[identifier])
                except (BundleError, OSError, ValueError, TypeError) as exc:
                    datasets[identifier] = {**details, "manifest": deepcopy(item["manifest"]), "status": "unavailable", "reason": str(exc), "layers": {}, "capabilities": {}}
                    errors.append({"dataset_id": identifier, "status": "unavailable", "reason": str(exc)})
            return json_value({"datasets": datasets, "layers": layers, "revision": self.revision, "errors": errors,
                "scope_policy": "Explicit dataset IDs, explicit area or all loaded data; no implicit geography."})
