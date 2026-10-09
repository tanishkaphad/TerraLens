"""Loading the immutable, precomputed TerraLens evidence bundle."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np


class BundleError(RuntimeError):
    pass


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class Bundle:
    def __init__(self, root: Path):
        self.root = root.resolve()
        try:
            self.manifest = json.loads((self.root / "manifest.json").read_text())
            self.tiles = json.loads((self.root / "tiles.json").read_text())
            collection = json.loads((self.root / "detections.geojson").read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise BundleError(f"Could not read bundle files under {self.root}: {exc}") from exc
        self.embeddings = np.load(self.root / "embeddings.npy", allow_pickle=False)
        if self.embeddings.ndim != 2 or self.embeddings.shape[0] != len(self.tiles):
            raise BundleError("Embedding rows must match the tiles.json order.")
        self.tile_index = {tile["tile_id"]: i for i, tile in enumerate(self.tiles)}
        self.detections = collection.get("features", [])
        self.layers: dict[str, gpd.GeoDataFrame] = {}
        for layer_id, layer_details in self.manifest.get("vector_layers", {}).items():
            relative = Path(layer_details.get("file", f"vectors/{layer_id}.geojson"))
            vector_path = (self.root / relative).resolve()
            if self.root not in vector_path.parents or not vector_path.is_file():
                raise BundleError(f"Vector file for layer '{layer_id}' is missing or outside the bundle.")
            try:
                layer_name = layer_details.get("source_layer")
                frame = gpd.read_file(vector_path, layer=layer_name) if layer_name else gpd.read_file(vector_path)
            except Exception as exc:
                raise BundleError(f"Could not load vector layer '{layer_id}': {exc}") from exc
            if frame.crs is None:
                raise BundleError(f"Vector layer '{layer_id}' has no CRS.")
            self.layers[layer_id] = frame.to_crs("EPSG:4326")
        self._text_model = None
        self._text_preprocess = None
        self._tokenizer = None
        self.text_model_error: str | None = None

    def text_encoder(self):
        """Lazily load the exact local checkpoint used to make the image index."""
        if self._text_model is not None:
            return self._text_model, self._tokenizer
        checkpoint = os.getenv("ENCODER_CHECKPOINT")
        if not checkpoint:
            return None
        checkpoint_path = Path(checkpoint).expanduser().resolve()
        if not checkpoint_path.is_file():
            self.text_model_error = f"Encoder checkpoint not found: {checkpoint_path}"
            return None
        expected = self.manifest.get("image_encoder", {}).get("checkpoint_sha256")
        actual = file_sha256(checkpoint_path)
        if expected and actual != expected:
            self.text_model_error = "Text checkpoint does not match the image-embedding checkpoint in this bundle."
            return None
        try:
            import open_clip
            import torch

            model_name = self.manifest.get("image_encoder", {}).get("model_name", "ViT-B-32")
            model = open_clip.create_model(model_name, pretrained=None)
            open_clip.load_checkpoint(model, str(checkpoint_path))
            if torch.cuda.is_available():
                device = torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = torch.device("mps")
            else:
                device = torch.device("cpu")
            self._text_model = model.eval().to(device)
            self._tokenizer = open_clip.get_tokenizer(model_name)
            self._device = device
        except Exception as exc:
            self.text_model_error = f"Could not load the image-index encoder: {exc}"
            return None
        return self._text_model, self._tokenizer

    def encode_text(self, text: str) -> np.ndarray | None:
        loaded = self.text_encoder()
        if loaded is None:
            return None
        import torch

        model, tokenizer = loaded
        tokens = tokenizer([text]).to(self._device)
        with torch.no_grad():
            vector = model.encode_text(tokens)
            vector = vector / vector.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        return vector[0].float().cpu().numpy()

    def asset_path(self, tile_id: str) -> Path | None:
        index = self.tile_index.get(tile_id)
        if index is None:
            return None
        relative = Path(self.tiles[index].get("preview", ""))
        candidate = (self.root / relative).resolve()
        if self.root not in candidate.parents or not candidate.is_file():
            return None
        return candidate
