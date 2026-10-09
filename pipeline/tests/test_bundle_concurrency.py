"""Thread-safety checks for lazy text-encoder setup and inference."""
from __future__ import annotations

import copy
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
import sys
import threading
import time
from types import SimpleNamespace

import numpy as np

from terralens.bundle import file_sha256
from test_backend import make_bundle


class FakeTensor:
    def __init__(self, values):
        self.values = np.asarray(values, dtype=np.float32)

    def norm(self, dim=-1, keepdim=False):
        return FakeTensor(np.linalg.norm(self.values, axis=dim, keepdims=keepdim))

    def clamp(self, min):
        return FakeTensor(np.maximum(self.values, min))

    def __truediv__(self, other):
        return FakeTensor(self.values / other.values)

    def __getitem__(self, index):
        return FakeTensor(self.values[index])

    def float(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.values.copy()


class FakeTokens:
    def to(self, device):
        return self


class FakeModel:
    def __init__(self):
        self.active = 0
        self.max_active = 0
        self.guard = threading.Lock()

    def eval(self):
        return self

    def to(self, device):
        return self

    def encode_text(self, tokens):
        with self.guard:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            time.sleep(0.02)
            return FakeTensor([[1.0, 0.0]])
        finally:
            with self.guard:
                self.active -= 1


def test_encoder_first_use_and_inference_are_serialized_and_shared_by_views(tmp_path, monkeypatch):
    manifest = {
        "dataset_version": "encoder-thread-test",
        "analysis_crs": "EPSG:32631",
        "supported_concepts": {"solar_panels": {
            "prompt": "solar panels", "aliases": ["solar panels"],
            "association_layer": "buildings", "grounding_available": True,
        }},
        "vector_layers": {
            "roads": {"file": "vectors/roads.geojson", "aliases": ["roads"], "fields": ["id", "surface"]},
            "buildings": {"file": "vectors/buildings.geojson", "aliases": ["buildings"], "fields": ["id", "building:use"]},
        },
        "image_encoder": {"embedding_dimension": 2},
    }
    bundle_root = tmp_path / "bundle"
    bundle_root.mkdir()
    bundle = make_bundle(bundle_root, manifest)
    checkpoint = tmp_path / "encoder.safetensors"
    checkpoint.write_bytes(b"thread-test-checkpoint")
    bundle.manifest["image_encoder"].update(
        model_name="test-encoder", checkpoint_sha256=file_sha256(checkpoint)
    )
    monkeypatch.setenv("ENCODER_CHECKPOINT", str(checkpoint))

    model = FakeModel()
    create_calls = []
    fake_open_clip = SimpleNamespace(
        create_model=lambda *args, **kwargs: create_calls.append(args) or model,
        load_checkpoint=lambda *args, **kwargs: None,
        get_tokenizer=lambda name: lambda texts: FakeTokens(),
    )
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: False),
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False)),
        device=lambda value: value,
        no_grad=nullcontext,
    )
    monkeypatch.setitem(sys.modules, "open_clip", fake_open_clip)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    gate = threading.Barrier(2)

    def encode(value):
        gate.wait()
        return bundle.encode_text(value)

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(encode, value) for value in ("trees", "roads")]
        first_results = [future.result(timeout=2) for future in futures]

    assert len(create_calls) == 1
    assert all(np.allclose(result, [1.0, 0.0]) for result in first_results)
    assert model.max_active == 1

    model.max_active = 0
    gate = threading.Barrier(2)
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(encode, value) for value in ("buildings", "water")]
        second_results = [future.result(timeout=2) for future in futures]
    assert all(np.allclose(result, [1.0, 0.0]) for result in second_results)
    assert model.max_active == 1

    view = copy.copy(bundle)
    assert view._encoder_lock is bundle._encoder_lock
    assert view._text_model is bundle._text_model
