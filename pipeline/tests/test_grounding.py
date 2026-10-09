"""Real coordinate maths and bounded run lifecycle; detector output is a fixture."""
import copy
import json
import threading
import numpy as np
import pytest
from fastapi.testclient import TestClient
from shapely.geometry import mapping, box, shape
from terralens import api
from terralens.grounding import GroundingError, GroundingService, eligible_tiles, merge_observations, observation_geometry
from terralens.planner import plan_query
from test_backend import make_bundle, manifest, local_parser


@pytest.fixture
def source(tmp_path, manifest):
    bundle = make_bundle(tmp_path, manifest, duplicate=True)
    for tile in bundle.tiles:
        tile.update(crs="EPSG:32631", transform=[1, 0, 500000, 0, -1, 100020])
    bundle.encode_text = lambda text: np.array([1, 0], dtype=np.float32)
    return bundle


@pytest.fixture
def service(monkeypatch):
    calls = []
    def detector(image, **kwargs):
        calls.append(kwargs)
        return [{"score": .8, "box": {"xmin": 2, "ymin": 3, "xmax": 12, "ymax": 14}}]
    value = GroundingService(max_runs=2)
    monkeypatch.setattr(value, "_load_detector", lambda bundle: detector)
    value.calls = calls
    return value


def test_original_pixels_transform_to_geographic_coordinates(source):
    geometry = shape(observation_geometry(source.tiles[0], [2, 3, 12, 14]))
    import geopandas as gpd
    restored = gpd.GeoSeries([geometry], crs="EPSG:4326").to_crs("EPSG:32631").iloc[0]
    assert restored.bounds == pytest.approx((500002, 100006, 500012, 100017), abs=1e-5)


def test_preview_boxes_scale_to_original_and_do_not_mutate_source(source, service):
    source.tiles[0]["width"] = 40
    source.tiles[0]["height"] = 40
    original = copy.deepcopy(source.detections)
    plan = plan_query("Find trees", source.manifest, mode="ground")
    run, hit = service.create(source, plan, tile_limit=1)
    assert not hit and len(service.calls) == 1
    assert run.bundle.detections[0]["properties"]["pixel_box"] == [4, 6, 24, 28]
    assert run.bundle.detections[0]["properties"]["prompt"] == "trees"
    assert source.detections == original
    assert None not in source.manifest["supported_concepts"]
    again, hit = service.create(source, plan, tile_limit=1)
    assert hit and again.identifier == run.identifier and len(service.calls) == 1


def test_reuse_allows_spatial_changes_but_rejects_new_visual_target(source, service):
    plan = plan_query("Find trees within 100 metres of roads", source.manifest, mode="ground")
    run, _ = service.create(source, plan, tile_limit=1)
    changed = plan_query("make it 50 metres", source.manifest, plan.model_dump(), "ground")
    assert service.get(run.identifier, source, changed) is run
    with pytest.raises(GroundingError, match="visual target"):
        service.get(run.identifier, source, plan_query("Find trucks", source.manifest, mode="ground"))
    run.expires_at = 0
    with pytest.raises(GroundingError, match="expired"):
        service.get(run.identifier, source, plan)


def test_merge_preserves_observations_without_combining_different_dates():
    def feature(identifier, date, tile="tile0"):
        return {"id": identifier, "geometry": mapping(box(0, 0, 1, 1)), "properties": {
            "detection_id": identifier, "score": .8, "source_sha256": "image", "acquisition_date": date, "tile_id": tile}}
    merged = merge_observations([feature("a", "2020-01-01"), feature("b", "2020-01-01"), feature("c", "2021-01-01")])
    assert len(merged) == 2 and len(merged[0]["properties"]["observations"]) == 2
    assert len(merge_observations([feature("a", None), feature("b", None, "tile1")])) == 2
    unknown_a, unknown_b = feature("a", "2020-01-01"), feature("b", "2020-01-01")
    unknown_a["properties"]["source_sha256"] = unknown_b["properties"]["source_sha256"] = None
    assert len(merge_observations([unknown_a, unknown_b])) == 2


def test_run_identity_isolated_even_when_versions_and_tile_ids_repeat(source, service, tmp_path):
    plan = plan_query("Find trees", source.manifest, mode="ground")
    first, _ = service.create(source, plan, tile_limit=1)
    other = copy.copy(source)
    other.root = tmp_path / "other-bundle"
    other._grounding_identity = None
    with pytest.raises(GroundingError, match="dataset"):
        service.get(first.identifier, other, plan)


def test_existing_run_remains_readable_during_other_inference(source, service, monkeypatch):
    plan = plan_query("Find trees", source.manifest, mode="ground")
    first, _ = service.create(source, plan, tile_limit=1)
    entered, release = threading.Event(), threading.Event()
    def slow_detector(image, **kwargs):
        entered.set()
        assert release.wait(3)
        return []
    monkeypatch.setattr(service, "_load_detector", lambda bundle: slow_detector)
    second_plan = plan_query("Find trucks", source.manifest, mode="ground")
    errors = []
    def background():
        try:
            service.create(source, second_plan, tile_limit=1)
        except Exception as exc:
            errors.append(exc)
    thread = threading.Thread(target=background)
    thread.start()
    assert entered.wait(1)
    try:
        assert service.get(first.identifier, source, plan) is first
    finally:
        release.set()
        thread.join(3)
    assert not thread.is_alive() and not errors


def test_selection_skips_ungeoreferenced_high_ranked_tile(source, service):
    source.tiles[0].pop("transform")
    plan = plan_query("Find trees", source.manifest, mode="ground")
    run, _ = service.create(source, plan, tile_limit=1)
    assert run.metadata["selected_tile_ids"] == ["tile1"]
    assert run.metadata["eligible_tiles"] == 1


@pytest.mark.parametrize("invalid", [{"crs": "not-a-real-crs"}, {"crs": "EPSG:4326", "transform": [1, 0, 1000, 0, -1, 1000]}, {"transform": [0, 0, 0, 0, 0, 0]}])
def test_invalid_crs_or_transformed_extent_is_ineligible(source, invalid):
    source.tiles[0].update(invalid)
    assert [tile["tile_id"] for tile in eligible_tiles(source)] == ["tile1"]


def test_api_grounding_export_and_pagination_reuse_exact_run(source, service, monkeypatch):
    monkeypatch.setattr(api, "get_bundle", lambda: source)
    monkeypatch.setattr(api, "get_grounding_service", lambda: service)
    client = TestClient(api.app)
    missing = client.post("/api/export", json={"query": "Find trees", "mode": "ground"})
    assert missing.status_code == 409 and len(service.calls) == 0
    response = client.post("/api/query", json={"query": "Find trees", "mode": "ground", "grounding_tile_limit": 1})
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["evidence_kind"] == "live_model_detection" and result["tiles_processed"] == 1
    assert result["candidate_selection_truncated"] is True
    body = {"plan": result["plan"], "mode": "ground", "dataset_version": "test-v1", "grounding_run_id": result["grounding_run_id"]}
    paged = client.post("/api/query", json={**body, "page": 1, "page_size": 1})
    assert paged.status_code == 200 and paged.json()["grounding"]["cache_hit"]
    exported = client.post("/api/export", json=body)
    assert exported.status_code == 200, exported.text
    native = client.get("/api/export", params={**{key: value for key, value in body.items() if key != "plan"}, "plan": json.dumps(result["plan"])})
    assert native.status_code == 200
    assert native.json()["features"] == exported.json()["features"]
    assert exported.json()["metadata"]["grounding_run_id"] == result["grounding_run_id"]
    assert len(service.calls) == 1
    stale = client.post("/api/query", json={**body, "dataset_version": "old"})
    assert stale.status_code == 409
    changed = client.post("/api/query", json={"query": "Find trucks", "mode": "ground", "grounding_run_id": result["grounding_run_id"]})
    assert changed.status_code == 409


@pytest.mark.parametrize("changes", [{"grounding_tile_limit": 5}, {"grounding_tile_limit": 1.5}, {"grounding_threshold": 0}, {"grounding_run_id": "../model"}, {"dataset_id": "../data"}])
def test_api_rejects_out_of_budget_or_path_inputs(changes):
    assert TestClient(api.app).post("/api/query", json={"query": "Find trees", "mode": "ground", **changes}).status_code == 422


def test_dataset_assets_and_unknown_ids_are_isolated(source, monkeypatch):
    monkeypatch.setattr(api, "_dataset_registry", lambda: ({"sample": {"id": "sample", "path": source.root, "default": True}}, "sample"))
    monkeypatch.setattr(api, "get_bundle", lambda identifier=None: source)
    client = TestClient(api.app)
    result = client.post("/api/query", json={"query": "Find buildings", "dataset_id": "sample"}).json()
    assert result["evidence_kind"] == "mapped_feature"
    data = client.get("/api/map", params={"dataset_id": "sample"}).json()
    assert data["tiles"][0]["asset_url"].endswith("?dataset_id=sample")
    assert client.get(data["tiles"][0]["asset_url"]).status_code == 200
    assert client.get("/api/map", params={"dataset_id": "unlisted"}).status_code == 404
