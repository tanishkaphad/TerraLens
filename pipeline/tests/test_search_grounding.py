"""Real geometry/cache behavior; detector predictions are declared test fixtures."""
from copy import deepcopy
import json
from types import SimpleNamespace
import threading

import numpy as np
import pytest
from shapely.geometry import shape

from terralens.grounding import GroundingError
from terralens.search_grounding import assess_tile, ground_tile, rank_scenes
from test_backend import make_bundle, manifest


@pytest.fixture
def source(tmp_path, manifest):
    bundle = make_bundle(tmp_path, manifest, duplicate=True)
    for tile in bundle.tiles:
        tile.update(crs="EPSG:32631", transform=[1, 0, 500000, 0, -1, 100020])
    return bundle


def fixture_service(predictions):
    calls = []
    def detector(image, **kwargs):
        calls.append(kwargs)
        return deepcopy(predictions)
    return SimpleNamespace(_inference_lock=threading.Lock(), _load_detector=lambda bundle: detector, calls=calls)


def test_arbitrary_prompt_without_encoder_no_object_cap_and_source_immutable(source, tmp_path):
    predictions = [{"score": .8, "box": {"xmin": 2, "ymin": 3, "xmax": 12, "ymax": 14}}] * 250
    service = fixture_service(predictions)
    original_manifest, original_detections = deepcopy(source.manifest), deepcopy(source.detections)
    result = ground_tile(source, "tile0", "unlisted visual structure", {"threshold": .3, "version": "test-policy-v1"}, service=service, cache_dir=tmp_path / "cache")
    assert len(result) == 250
    assert result.metadata["raw_predictions"] == 250 and not result.metadata["observation_limit_reached"]
    assert service.calls == [{"candidate_labels": ["unlisted visual structure"], "threshold": .3}]
    assert source.manifest == original_manifest and source.detections == original_detections
    first = result[0]
    assert first["properties"]["visual_query"] == "unlisted visual structure"
    assert first["properties"]["geometry_role"] == "object_box"
    assert first["properties"]["visual_evidence"][0]["query"] == "unlisted visual structure"
    assert first["properties"]["observations"][0]["geometry"] == first["geometry"]
    import geopandas as gpd
    restored = gpd.GeoSeries([shape(first["geometry"])], crs="EPSG:4326").to_crs("EPSG:32631").iloc[0]
    assert restored.bounds == pytest.approx((500002, 100006, 500012, 100017), abs=1e-5)


def test_cache_reuses_exact_policy_and_invalidates_prompt_version_source(source, tmp_path):
    service = fixture_service([])
    cache = tmp_path / "cache"
    first = ground_tile(source, "tile0", "new object", .2, service=service, cache_dir=cache)
    again = ground_tile(source, "tile0", "new object", .2, service=service, cache_dir=cache)
    assert again.metadata["cache_hit"] and len(service.calls) == 1
    assert len(first) == 0 and first.metadata["status"] == "processed" and first.metadata["absence_verified"] is False
    assert json.loads(next(cache.glob("*.json")).read_text())["metadata"]["retained_observations"] == 0
    ground_tile(source, "tile0", "different object", .2, service=service, cache_dir=cache)
    ground_tile(source, "tile0", "new object", .3, service=service, cache_dir=cache)
    source.manifest["dataset_version"] = "updated-v2"
    ground_tile(source, "tile0", "new object", .2, service=service, cache_dir=cache)
    source.tiles[0]["transform"][2] += 1
    ground_tile(source, "tile0", "new object", .2, service=service, cache_dir=cache)
    assert len(service.calls) == 5


def test_preview_boxes_scale_to_original_pixels_and_invalid_predictions_reported(source, tmp_path):
    source.tiles[0].update(width=40, height=40)
    service = fixture_service([
        {"score": .8, "box": {"xmin": 2, "ymin": 3, "xmax": 12, "ymax": 14}},
        {"score": .8, "box": {"xmin": 4, "ymin": 4, "xmax": 1, "ymax": 1}},
        {"score": .05, "box": {"xmin": 2, "ymin": 3, "xmax": 12, "ymax": 14}},
        {"score": float("nan"), "box": {"xmin": 2, "ymin": 3, "xmax": 12, "ymax": 14}},
    ])
    result = ground_tile(source, "tile0", "objects", .1, service=service, cache_dir=tmp_path / "cache")
    assert len(result) == 1 and result[0]["properties"]["pixel_box"] == [4, 6, 24, 28]
    assert result.metadata["invalid_predictions"] == 2 and result.metadata["below_policy_predictions"] == 1
    assert result.metadata["status"] == "processed_with_invalid_predictions"


def test_invalid_tile_reports_unavailable_instead_of_silent_skip(source, tmp_path):
    source.tiles[0]["transform"] = [0, 0, 0, 0, 0, 0]
    assert not assess_tile(source, "tile0")["available"]
    with pytest.raises(GroundingError, match="singular"):
        ground_tile(source, "tile0", "objects", service=fixture_service([]), cache_dir=tmp_path / "cache")
    with pytest.raises(GroundingError, match="Unknown image"):
        ground_tile(source, "absent", "objects", service=fixture_service([]), cache_dir=tmp_path / "cache")


def test_ranking_preserves_every_scene_and_uses_matching_encoder(source, monkeypatch):
    monkeypatch.setattr(source, "encode_text", lambda text: np.array([1., 0.], dtype=np.float32))
    result = rank_scenes(source, "unseen scene concept")
    assert len(result) == 2 and result.metadata["truncated"] is False
    assert [feature["properties"]["tile_id"] for feature in result] == ["tile0", "tile1"]
    assert all(feature["properties"]["geometry_role"] == "scene_footprint" for feature in result)
    assert all(feature["properties"]["visual_evidence"][0]["status"] == "unknown" for feature in result)
    monkeypatch.setattr(source, "encode_text", lambda text: None)
    with pytest.raises(GroundingError, match="matching installed text encoder"):
        rank_scenes(source, "scene")


def test_large_readable_preview_has_no_legacy_1600_pixel_exclusion(source, tmp_path):
    from PIL import Image
    Image.new("RGB", (1700, 20)).save(source.asset_path("tile0"))
    source.tiles[0]["width"] = 1700
    assert assess_tile(source, "tile0")["available"]
    result = ground_tile(source, "tile0", "objects", service=fixture_service([]), cache_dir=tmp_path / "cache")
    assert result.metadata["status"] == "processed"
