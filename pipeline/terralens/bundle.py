"""Validate and load a frozen geospatial evidence bundle without fetching models."""
from __future__ import annotations
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any
import geopandas as gpd
import numpy as np
from shapely.geometry import shape

class BundleError(RuntimeError):
    pass

def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def json_value(value: Any) -> Any:
    """Vector formats may return NaN, timestamps or NumPy scalars; JSON must not."""
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    if isinstance(value, np.generic):
        return json_value(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)

class Bundle:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self._text_model = self._tokenizer = None
        self._encoder_attempted = False
        self.text_model_error: str | None = None
        try:
            self.manifest = json.loads((self.root / "manifest.json").read_text())
            self.tiles = json.loads((self.root / "tiles.json").read_text())
            collection = json.loads((self.root / "detections.geojson").read_text())
            self.embeddings = np.load(self.root / "embeddings.npy", allow_pickle=False)
            if not isinstance(self.manifest, dict) or not isinstance(self.tiles, list) or not self.tiles:
                raise BundleError("The manifest must be an object and tiles.json a non-empty array.")
            limits = self.manifest.get("limits", {})
            maximum_distance = limits.get("maximum_distance_m", 5000)
            explore_limit = limits.get("explore_tile_limit", 100)
            if not isinstance(maximum_distance, (int, float)) or not math.isfinite(maximum_distance) or not 0 < maximum_distance <= 5000:
                raise BundleError("maximum_distance_m must be finite and in (0, 5000].")
            if not isinstance(explore_limit, int) or isinstance(explore_limit, bool) or not 1 <= explore_limit <= 1000:
                raise BundleError("explore_tile_limit must be an integer from 1 to 1000.")
            allowance = self.manifest.get("distance_uncertainty_m")
            if allowance is not None and (not isinstance(allowance, (int, float)) or not math.isfinite(allowance) or not 0 <= allowance <= 1000):
                raise BundleError("distance_uncertainty_m must be finite and in [0, 1000].")
            if collection.get("type") != "FeatureCollection" or not isinstance(collection.get("features"), list):
                raise BundleError("detections.geojson must be a FeatureCollection.")
            if self.embeddings.ndim != 2 or self.embeddings.shape[0] != len(self.tiles) or self.embeddings.shape[1] < 1:
                raise BundleError("Embedding rows must match the tiles.json order and have a non-zero dimension.")
            if not np.issubdtype(self.embeddings.dtype, np.floating) or not np.isfinite(self.embeddings).all():
                raise BundleError("Image embeddings must be finite floating point vectors.")
            norms = np.linalg.norm(self.embeddings, axis=1)
            if not np.allclose(norms, 1.0, atol=0.02):
                raise BundleError("Image embeddings must have unit length for cosine retrieval.")
            dimension = self.manifest.get("image_encoder", {}).get("embedding_dimension")
            if dimension is not None and int(dimension) != self.embeddings.shape[1]:
                raise BundleError("The manifest embedding dimension does not match embeddings.npy.")
            self.tile_index = {}
            for index, tile in enumerate(self.tiles):
                tile_id = tile.get("tile_id")
                if not isinstance(tile_id, str) or not tile_id or tile_id in self.tile_index:
                    raise BundleError("Tile identifiers must be unique non-empty strings.")
                self._geometry(tile.get("geometry"), f"Tile '{tile_id}'")
                if tile["geometry"].get("type") != "Polygon":
                    raise BundleError("Image tile coverage must be a WGS84 polygon.")
                if self.asset_path_for(tile) is None:
                    raise BundleError(f"Preview missing for tile '{tile_id}'.")
                self.tile_index[tile_id] = index
            self.detections = collection["features"]
            detection_ids = set()
            for feature in self.detections:
                props = feature.get("properties", {})
                identifier = props.get("detection_id") or feature.get("id")
                if not identifier or str(identifier) in detection_ids:
                    raise BundleError("Detection identifiers must be unique non-empty values.")
                detection_ids.add(str(identifier))
                if props.get("tile_id") not in self.tile_index:
                    raise BundleError("A detection refers to an unknown tile.")
                if props.get("concept_id") not in self.manifest.get("supported_concepts", {}):
                    raise BundleError("A detection refers to an undeclared visual concept.")
                score = props.get("score")
                if not isinstance(score, (int, float)) or not math.isfinite(score) or not 0 <= score <= 1:
                    raise BundleError("Detection scores must be finite numbers between zero and one.")
                self._geometry(feature.get("geometry"), f"Detection '{identifier}'")
                observations = props.get("observations", [])
                if not isinstance(observations, list):
                    raise BundleError("Detection observations must be an array.")
                for observation in observations:
                    if not isinstance(observation, dict) or observation.get("tile_id") not in self.tile_index:
                        raise BundleError("A constituent detection observation refers to an unknown tile.")
                    observed_score = observation.get("score")
                    if not isinstance(observed_score, (int, float)) or not math.isfinite(observed_score) or not 0 <= observed_score <= 1:
                        raise BundleError("Constituent detection scores must be finite and in [0, 1].")
                    if observation.get("geometry") is not None:
                        self._geometry(observation["geometry"], "Constituent detection observation")
            self.layers: dict[str, gpd.GeoDataFrame] = {}
            for layer_id, details in self.manifest.get("vector_layers", {}).items():
                path = self.local_path(details.get("file", f"vectors/{layer_id}.geojson"))
                if not path.is_file():
                    raise BundleError(f"Vector file for layer '{layer_id}' is missing.")
                source_layer = details.get("source_layer")
                # Bundled GeoJSON is one layer even when its original source was a GeoPackage.
                frame = gpd.read_file(path, layer=source_layer) if source_layer and path.suffix.lower() != ".geojson" else gpd.read_file(path)
                if frame.crs is None:
                    raise BundleError(f"Vector layer '{layer_id}' has no CRS.")
                frame = frame[frame.geometry.notna() & ~frame.geometry.is_empty].to_crs("EPSG:4326").reset_index(drop=True)
                if not frame.empty and not frame.geometry.is_valid.all():
                    raise BundleError(f"Vector layer '{layer_id}' contains invalid geometries.")
                for geometry in frame.geometry:
                    self._geometry(geometry.__geo_interface__, f"Vector layer '{layer_id}'")
                declared_fields = set(details.get("fields", []))
                if declared_fields - set(frame.columns):
                    raise BundleError(f"Declared fields missing from vector layer '{layer_id}'.")
                self.layers[layer_id] = frame
            for concept_id, concept in self.manifest.get("supported_concepts", {}).items():
                associated = concept.get("association_layer")
                if associated is not None and associated not in self.layers:
                    raise BundleError(f"Association layer for '{concept_id}' is unavailable.")
                overlap = float(concept.get("association_overlap_min", 0.5))
                if not math.isfinite(overlap) or not 0 < overlap <= 1:
                    raise BundleError("Association overlap thresholds must be in (0, 1].")
            for name, expected_hash in self.manifest.get("artifact_sha256", {}).items():
                path = self.local_path(name)
                if not path.is_file() or file_sha256(path) != expected_hash:
                    raise BundleError(f"Bundle artifact integrity check failed: {name}.")
        except BundleError:
            raise
        except Exception as exc:
            raise BundleError(f"Could not load evidence bundle under {self.root}: {exc}") from exc

    @staticmethod
    def _geometry(raw: dict | None, label: str):
        try:
            geometry = shape(raw)
            bounds = geometry.bounds
            if geometry.is_empty or not geometry.is_valid or len(bounds) != 4 or not all(math.isfinite(value) for value in bounds):
                raise ValueError("empty, non-finite or invalid geometry")
            if not (-180 <= bounds[0] <= bounds[2] <= 180 and -90 <= bounds[1] <= bounds[3] <= 90):
                raise ValueError("coordinates are outside WGS84 bounds")
            return geometry
        except Exception as exc:
            raise BundleError(f"{label} has invalid WGS84 geometry: {exc}") from exc

    def local_path(self, relative: str) -> Path:
        path = (self.root / relative).resolve()
        if self.root not in path.parents:
            raise BundleError("An artifact path leaves the evidence bundle.")
        return path

    def asset_path_for(self, tile: dict) -> Path | None:
        try:
            path = self.local_path(tile.get("preview", ""))
        except BundleError:
            return None
        return path if path.is_file() else None

    def asset_path(self, tile_id: str) -> Path | None:
        index = self.tile_index.get(tile_id)
        return None if index is None else self.asset_path_for(self.tiles[index])

    def text_encoder(self):
        """Load only a local checkpoint with the image index's exact identity."""
        if self._text_model is not None:
            return self._text_model, self._tokenizer
        if self._encoder_attempted:
            return None
        self._encoder_attempted = True
        checkpoint = os.getenv("ENCODER_CHECKPOINT")
        if not checkpoint:
            self.text_model_error = "checkpoint_not_configured"
            return None
        path = Path(checkpoint).expanduser().resolve()
        if not path.is_file():
            self.text_model_error = "checkpoint_not_found"
            return None
        expected = self.manifest.get("image_encoder", {}).get("checkpoint_sha256")
        if not expected or file_sha256(path) != expected:
            self.text_model_error = "checkpoint_identity_missing_or_mismatched"
            return None
        try:
            import open_clip
            import torch
            name = self.manifest.get("image_encoder", {}).get("model_name", "ViT-B-32")
            model = open_clip.create_model(name, pretrained=None)
            open_clip.load_checkpoint(model, str(path))
            if torch.cuda.is_available():
                device = torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = torch.device("mps")
            else:
                device = torch.device("cpu")
            self._text_model = model.eval().to(device)
            self._tokenizer = open_clip.get_tokenizer(name)
            self._device = device
            probe = self.encode_text("aerial imagery")
            if probe is None or probe.shape != (self.embeddings.shape[1],):
                raise ValueError("text/image embedding dimensions differ")
            self.text_model_error = None
        except Exception as exc:
            self._text_model = self._tokenizer = None
            self.text_model_error = f"encoder_load_failed: {type(exc).__name__}"
            return None
        return self._text_model, self._tokenizer

    def encode_text(self, text: str) -> np.ndarray | None:
        loaded = self.text_encoder()
        if loaded is None:
            return None
        try:
            import torch
            model, tokenizer = loaded
            with torch.no_grad():
                vector = model.encode_text(tokenizer([text]).to(self._device))
                vector = vector / vector.norm(dim=-1, keepdim=True).clamp(min=1e-12)
            result = vector[0].float().cpu().numpy()
            if result.shape != (self.embeddings.shape[1],) or not np.isfinite(result).all() or np.linalg.norm(result) < 0.99:
                raise ValueError("invalid text embedding")
            return result
        except Exception as exc:
            self.text_model_error = f"encoder_inference_failed: {type(exc).__name__}"
            return None

    def retrieval_status(self) -> dict[str, Any]:
        loaded = self.text_encoder()
        return {"enabled": loaded is not None, "status": "ready" if loaded is not None else self.text_model_error}

    def provenance(self) -> dict[str, Any]:
        return json_value({"dataset_version": self.manifest.get("dataset_version"),
            "created_at_utc": self.manifest.get("created_at_utc"), "coverage": self.manifest.get("coverage", {}),
            "created_by": self.manifest.get("created_by"), "preparation": self.manifest.get("preparation", {}),
            "attribution": self.manifest.get("attribution", {}), "image_encoder": self.manifest.get("image_encoder", {}),
            "object_detector": self.manifest.get("object_detector", {}), "source_records": self.manifest.get("source_records", []),
            "vector_layers": self.manifest.get("vector_layers", {}), "artifact_sha256": self.manifest.get("artifact_sha256", {})})
