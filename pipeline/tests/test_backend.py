"""Focused correctness checks using synthetic geometries, never production evidence."""
from __future__ import annotations
import json
import math
from pathlib import Path
import geopandas as gpd
import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image
from shapely.geometry import LineString, box, mapping
from terralens import api
from terralens.bundle import Bundle, BundleError
from terralens.engine import condition_state, execute
from terralens.planner import PlanError, _guard_interpretation, plan_query, validate_plan

@pytest.fixture(autouse=True)
def local_parser(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    monkeypatch.delenv("ENCODER_CHECKPOINT", raising=False)

@pytest.fixture
def manifest():
    return {"dataset_version": "test-v1", "area_name": "Synthetic geometry test",
        "analysis_crs": "EPSG:32631", "supported_concepts": {"solar_panels": {
            "prompt": "solar panels", "aliases": ["solar panels"], "association_layer": "buildings", "grounding_available": True}},
        "vector_layers": {"roads": {"file": "vectors/roads.geojson", "aliases": ["road", "roads"], "fields": ["id", "surface"], "extraction_date": "2025-01-01"},
            "buildings": {"file": "vectors/buildings.geojson", "aliases": ["buildings"], "fields": ["id", "building:use"]}},
        "queryable_attributes": ["id", "surface", "building:use"], "coverage": {"test_only": True},
        "image_encoder": {"embedding_dimension": 2}, "attribution": {"imagery": "Synthetic test fixture"}}

def geographic(geometry):
    from shapely import affinity
    geometry = affinity.translate(geometry, xoff=500000, yoff=100000)
    return gpd.GeoSeries([geometry], crs="EPSG:32631").to_crs("EPSG:4326").iloc[0]

def make_bundle(tmp_path: Path, manifest, *, duplicate=False, ambiguous=False, absent_surface=False, distance=50):
    (tmp_path / "vectors").mkdir()
    (tmp_path / "previews").mkdir()
    tiles = []
    detections = []
    for index in range(2 if duplicate else 1):
        tile_id = f"tile{index}"
        Image.new("RGB", (20, 20), (20, 50, 80)).save(tmp_path / "previews" / f"{tile_id}.jpg")
        tiles.append({"tile_id": tile_id, "preview": f"previews/{tile_id}.jpg", "geometry": mapping(geographic(box(-100, -100, 200, 200))),
            "source_name": "test.tif", "source_sha256": "fixture-image", "width": 20, "height": 20, "acquisition_date": "2024-06-01"})
        detections.append({"type": "Feature", "geometry": mapping(geographic(box(3, 3, 10, 10))), "properties": {
            "detection_id": f"detection{index}", "tile_id": tile_id, "concept_id": "solar_panels", "score": 0.8,
            "model_id": "fixture-model", "pixel_box": [3, 3, 10, 10]}})
    buildings = [{"id": "b1", "building:use": "commercial", "geometry": geographic(box(0, 0, 20, 20))}]
    if ambiguous:
        buildings.append({"id": "b2", "building:use": None, "geometry": geographic(box(0, 0, 22, 22))})
    roads = [{"id": "paved", "surface": None if absent_surface else "asphalt", "geometry": geographic(LineString([(20 + distance, -50), (20 + distance, 50)]))},
        {"id": "unpaved", "surface": "unpaved", "geometry": geographic(LineString([(25 + distance, -50), (25 + distance, 50)]))}]
    gpd.GeoDataFrame(buildings, crs="EPSG:4326").to_file(tmp_path / "vectors/buildings.geojson", driver="GeoJSON")
    gpd.GeoDataFrame(roads, crs="EPSG:4326").to_file(tmp_path / "vectors/roads.geojson", driver="GeoJSON")
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    (tmp_path / "tiles.json").write_text(json.dumps(tiles))
    (tmp_path / "detections.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": detections}))
    np.save(tmp_path / "embeddings.npy", np.array([[1., 0.]] * len(tiles), dtype=np.float32))
    return Bundle(tmp_path)

def object_plan(manifest, **changes):
    raw = {"concept_id": "solar_panels", "visual_query": "solar panels", "reference_layer": "roads", "predicate": "within_distance", "distance_m": 200, "attributes": []}
    raw.update(changes)
    return validate_plan(raw, manifest)

def test_decimal_compact_radius_and_followup_preserve_attributes(manifest):
    plan = plan_query("Find solar panels within 0.2km of roads", manifest)
    assert plan.distance_m == 200
    prior = object_plan(manifest, attributes=[{"role": "subject", "field": "building:use", "operator": "equals", "value": "commercial"}])
    edited = plan_query("make it 0.1 km", manifest, prior.model_dump())
    assert edited.distance_m == 100
    assert edited.attributes == prior.attributes

def test_followup_cannot_silently_drop_added_clause(manifest):
    prior = object_plan(manifest)
    with pytest.raises(PlanError):
        plan_query("make it 100 metres and commercial buildings", manifest, prior.model_dump())

def test_subject_building_alias_cannot_steal_road_reference(manifest):
    manifest["supported_concepts"]["building_roofs"] = {"prompt": "building roofs", "aliases": ["building roofs"], "association_layer": "buildings", "grounding_available": True}
    manifest["vector_layers"]["buildings"]["aliases"] = ["building", "buildings", "roof", "roofs"]
    plan = plan_query("Find building roofs within 100 metres of roads", manifest)
    assert plan.concept_id == "building_roofs"
    assert plan.reference_layer == "roads"

def test_plain_supported_visual_query_in_explore_is_tile_retrieval(manifest):
    manifest["supported_concepts"]["swimming_pools"] = {"prompt": "swimming pools", "aliases": ["swimming pools"], "association_layer": None, "grounding_available": True}
    plan = plan_query("Find swimming pools", manifest, mode="explore")
    assert plan.query_type == "tiles"
    assert plan.reference_layer is None
    assert plan.predicate is None
    assert plan.concept_id == "swimming_pools"

@pytest.mark.parametrize("distance", [math.inf, math.nan, -1, 0, 5001])
def test_finite_bounded_distance(manifest, distance):
    with pytest.raises(PlanError):
        object_plan(manifest, distance_m=distance)

def test_role_specific_capabilities(manifest):
    with pytest.raises(PlanError, match="reference"):
        object_plan(manifest, attributes=[{"role": "reference", "field": "building:use", "operator": "equals", "value": "commercial"}])

@pytest.mark.parametrize("query", [
    "Find commercial solar panels within 200 metres of roads",
    "Find recently installed solar panels within 200 metres of roads",
    "Find solar panels without roads",
    "Find solar panels within 100 metres of roads",
])
def test_valid_schema_cannot_omit_known_requested_condition(manifest, query):
    with pytest.raises(PlanError):
        _guard_interpretation(query, object_plan(manifest), manifest)

def test_cached_concept_cannot_claim_extra_visual_adjective(manifest):
    with pytest.raises(PlanError, match="additional visual"):
        object_plan(manifest, visual_query="broken solar panels")

def test_reference_attribute_uses_qualifying_reference_not_empty_dict(tmp_path, manifest):
    bundle = make_bundle(tmp_path, manifest)
    plan = object_plan(manifest, attributes=[{"role": "reference", "field": "surface", "operator": "equals", "value": "unpaved"}])
    result = execute(bundle, plan)
    assert result["total_matches"] == 1
    relation = result["features"][0]["properties"]["conditions"][-2]
    assert relation["reference_feature_ids"] == ["unpaved"]
    assert result["features"][0]["properties"]["conditions"][-1]["observed"] == "unpaved"

def test_missing_reference_attribute_is_unknown(tmp_path, manifest):
    bundle = make_bundle(tmp_path, manifest, absent_surface=True)
    plan = object_plan(manifest, attributes=[{"role": "reference", "field": "surface", "operator": "equals", "value": "asphalt"}])
    result = execute(bundle, plan)
    assert result["total_matches"] == 0
    assert result["total_uncertain"] == 1
    assert result["uncertain_candidates"][0]["geometry"]

def test_failure_overrides_unknown():
    assert condition_state([{"status": "unknown"}, {"status": "fail"}]) == "fail"

def test_spatial_failure_excludes_even_missing_subject_attribute(tmp_path, manifest):
    bundle = make_bundle(tmp_path, manifest, distance=400)
    bundle.layers["buildings"]["building:use"] = None
    plan = object_plan(manifest, distance_m=200, attributes=[{"role": "subject", "field": "building:use", "operator": "equals", "value": "commercial"}])
    result = execute(bundle, plan)
    assert result["total_matches"] == 0
    assert result["total_uncertain"] == 0
    assert result["objects_excluded"] == 1

def test_ambiguous_association_preserves_geometry_and_evidence(tmp_path, manifest):
    bundle = make_bundle(tmp_path, manifest, ambiguous=True, distance=400)
    result = execute(bundle, object_plan(manifest))
    candidate = result["uncertain_candidates"][0]
    assert candidate["properties"]["association"] == "ambiguous"
    assert shape_equal(candidate["geometry"], bundle.detections[0]["geometry"])
    assert candidate["properties"]["evidence_geometries"]

def shape_equal(left, right):
    from shapely.geometry import shape
    return shape(left).equals(shape(right))

def test_duplicate_tiles_do_not_overcount_same_vector_object(tmp_path, manifest):
    bundle = make_bundle(tmp_path, manifest, duplicate=True)
    result = execute(bundle, object_plan(manifest))
    assert result["total_matches"] == 1
    assert result["duplicate_observations_collapsed"] == 1
    assert len(result["features"][0]["properties"]["observations"]) == 2

def test_distance_allowance_straddles_threshold(tmp_path, manifest):
    manifest["distance_uncertainty_m"] = 5
    bundle = make_bundle(tmp_path, manifest, distance=198)
    result = execute(bundle, object_plan(manifest, distance_m=200))
    assert result["total_matches"] == 0
    assert result["total_uncertain"] == 1

def test_unknown_concept_explores_real_embedding_without_object_claim(tmp_path, manifest, monkeypatch):
    bundle = make_bundle(tmp_path, manifest)
    monkeypatch.setattr(bundle, "encode_text", lambda text: np.array([1., 0.]))
    plan = plan_query("Find swimming pools within 200 metres of roads", manifest, mode="explore")
    result = execute(bundle, plan, mode="explore")
    assert result["result_kind"] == "tile_retrieval"
    assert result["total_matches"] == 0
    assert result["total_tile_candidates"] == 1
    assert result["tile_candidates"][0]["properties"]["retrieval_similarity"] == 1
    assert result["unresolved_conditions"][0]["status"] == "unknown"

def test_bundle_rejects_nonfinite_embeddings(tmp_path, manifest):
    make_bundle(tmp_path, manifest)
    np.save(tmp_path / "embeddings.npy", np.array([[math.nan, 0.]], dtype=np.float32))
    with pytest.raises(BundleError, match="finite"):
        Bundle(tmp_path)

def test_map_export_and_truthful_readiness(tmp_path, manifest):
    bundle = make_bundle(tmp_path, manifest, duplicate=True)
    import pandas as pd
    additional = gpd.GeoDataFrame([{"id": "b3", "building:use": "commercial", "geometry": geographic(box(0, 30, 20, 50))}], crs="EPSG:4326")
    bundle.layers["buildings"] = gpd.GeoDataFrame(pd.concat([bundle.layers["buildings"], additional], ignore_index=True), crs="EPSG:4326")
    detection = json.loads(json.dumps(bundle.detections[0]))
    detection["properties"]["detection_id"] = "detection3"
    detection["geometry"] = mapping(geographic(box(3, 33, 10, 40)))
    bundle.detections.append(detection)
    api.app.dependency_overrides.clear()
    old = api.get_bundle
    api.get_bundle = lambda: bundle
    try:
        client = TestClient(api.app)
        ready = client.get("/ready").json()
        assert ready["status"] == "degraded"
        assert ready["capabilities"]["live_text_image_retrieval"] is False
        mapped = client.get("/api/map?tile_limit=1&vector_limit=2").json()
        assert len(mapped["tiles"][0]["coordinates"]) == 4
        assert mapped["counts"]["tiles"]["truncated"]
        assert sum(len(layer["features"]) for layer in mapped["vector_layers"].values()) == 2
        assert client.get("/api/map?bbox=nan,0,1,2").status_code == 422
        assert client.get(mapped["tiles"][0]["asset_url"]).status_code == 200
        search = client.post("/api/query", json={"query": "Find solar panels within 200 metres of roads", "page_size": 1}).json()
        assert search["total_matches"] == 2
        assert search["truncated"] is True
        page = client.post("/api/query", json={"plan": search["plan"], "dataset_version": "test-v1", "page": 1, "page_size": 1})
        assert page.status_code == 200
        assert page.json()["plan"] == search["plan"]
        assert page.json()["page"] == 1
        assert len(page.json()["features"]) == 1
        assert page.json()["features"][0]["id"] != search["features"][0]["id"]
        assert client.post("/api/query", json={"plan": search["plan"], "dataset_version": "stale"}).status_code == 409
        assert client.post("/api/query", json={}).status_code == 422
        export = client.post("/api/export", json={"plan": search["plan"], "dataset_version": "test-v1"})
        assert export.status_code == 200
        exported = export.json()
        assert exported["metadata"]["truncated"] is False
        assert sum(feature["properties"]["role"] == "match" for feature in exported["features"]) == 2
        assert exported["metadata"]["provenance"]["vector_layers"]["roads"]["extraction_date"] == "2025-01-01"
        assert any(feature["properties"]["role"] == "visual_detection" for feature in exported["features"])
        assert client.post("/api/export", json={"plan": search["plan"], "dataset_version": "other"}).status_code == 409
        native = client.get("/api/export", params={"plan": json.dumps(search["plan"]), "dataset_version": "test-v1", "mode": "catalog_scan", "include_uncertain": "false"})
        assert native.status_code == 200
        assert native.headers["content-type"] == "application/geo+json"
        assert native.headers["content-disposition"] == export.headers["content-disposition"]
        assert native.json()["features"] == exported["features"]
        assert native.json()["metadata"]["plan"] == search["plan"]
        assert client.get("/api/export", params={"plan": json.dumps(search["plan"]), "dataset_version": "stale"}).status_code == 409
        assert client.get("/api/export", params={"plan": "{invalid json"}).status_code == 422
        assert client.get("/api/export", params={"plan": "x" * 8193}).status_code == 422
    finally:
        api.get_bundle = old
