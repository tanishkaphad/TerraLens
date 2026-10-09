"""Route and export checks use provider fixtures, never fabricated live matches."""
from copy import deepcopy
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from terralens import api, router
from terralens.planner import PlanError, QueryPlan


def place(name="Pune", *, longitude=73.85, latitude=18.52, kind="city", identifier="pune"):
    return {"type": "Feature", "id": identifier,
        "geometry": {"type": "Point", "coordinates": [longitude, latitude]},
        "properties": {"name": name, "display_name": name, "place_type": kind,
            "source": "provider fixture", "source_url": "https://example.test/place",
            "evidence_kind": "mapped_place", "retrieved_at": "fixture-only", "bbox": None}}


@pytest.fixture
def setup_router(monkeypatch):
    monkeypatch.delenv("GEOVERIFY_ROUND1", raising=False)
    registry = {"phoenix": {"id": "phoenix", "name": "Phoenix / Encanto study area"}}
    bundle = SimpleNamespace(manifest={"area_name": "Phoenix / Encanto study area",
        "supported_concepts": {}, "coverage": {"bbox_wgs84": [-112.1, 33.48, -112.09, 33.49]}})
    calls = {"places": [], "vectors": [], "local": []}

    def lookup(text, limit=6):
        calls["places"].append(text)
        return [place()]

    def vector(subject, reference, distance, location, **kwargs):
        calls["vectors"].append((subject, reference, distance, location, kwargs))
        return {"type": "FeatureCollection", "features": [], "context_features": [], "total_matches": 0}

    def local(plan, mode, dataset_id):
        calls["local"].append((plan, mode, dataset_id))
        return {"features": [], "uncertain_candidates": [], "tile_candidates": [],
            "context_features": [], "result_kind": "grounded_objects", "total_matches": 0,
            "total_uncertain": 0, "plan": plan.model_dump()}

    monkeypatch.setattr(router.world_data, "lookup_places", lookup)
    monkeypatch.setattr(router.world_data, "vector_search", vector)

    def run(query, **kwargs):
        return router.discover(query, dataset_id=kwargs.pop("dataset_id", "phoenix"),
            previous_plan=kwargs.pop("previous_plan", None), registry=registry, default_dataset="phoenix",
            bundle_loader=lambda identifier: bundle, local_search=local, **kwargs)

    return run, calls


def test_named_place_ignores_imagery_hint_and_preserves_city(setup_router):
    run, calls = setup_router
    result = run("Find Sinhagad College in Pune", dataset_id="unprepared")
    assert result["route"] == "places"
    assert result["result_kind"] == "place_lookup"
    assert result["dataset_id"] is None
    assert calls["places"] == ["Sinhagad College in Pune"]
    assert calls["local"] == calls["vectors"] == []


def test_ambiguous_named_institution_returns_real_candidates(setup_router, monkeypatch):
    run, calls = setup_router
    candidates = [place("Sinhagad College A", identifier="a"), place("Sinhagad College B", identifier="b")]
    monkeypatch.setattr(router.world_data, "lookup_places", lambda *args, **kwargs: deepcopy(candidates))
    result = run("Find Sinhagad College in Pune")
    assert result["status"] == "needs_clarification"
    assert [item["id"] for item in result["features"]] == ["a", "b"]
    assert result["total_matches"] == 2
    assert calls["local"] == calls["vectors"] == []


@pytest.mark.parametrize("distance", ["300 metres", "0.3km", "300m"])
def test_global_spatial_metric_search_preserves_radius(setup_router, distance):
    run, calls = setup_router
    result = run(f"Find buildings within {distance} of a river in Pune")
    assert result["route"] == "vectors" and result["status"] == "completed"
    assert result["result_kind"] == "vector_features"
    assert calls["vectors"][0][:3] == ("buildings", "rivers", 300)
    assert result["interpretation"]["distance_m"] == 300
    assert result["interpretation"]["search_radius_m"] == 2000
    assert "2 km" in result["claim_scope"]
    assert calls["local"] == []


def test_unique_exact_locality_does_not_choose_arbitrary_first_result(setup_router, monkeypatch):
    run, calls = setup_router
    monkeypatch.setattr(router.world_data, "lookup_places", lambda *args, **kwargs: [
        place("Pune Ward 4", kind="suburb", identifier="ward"), place(),
        place("Pune Road", kind="street", identifier="street")])
    result = run("Find buildings within 300 metres of rivers in Pune")
    assert result["status"] == "completed"
    assert calls["vectors"][0][3]["id"] == "pune"


def test_ambiguous_locality_stops_before_spatial_execution(setup_router, monkeypatch):
    run, calls = setup_router
    monkeypatch.setattr(router.world_data, "lookup_places", lambda *args, **kwargs: [
        place(identifier="one"), place(longitude=74.1, identifier="two")])
    result = run("Find buildings within 300 metres of rivers in Pune")
    assert result["status"] == "needs_clarification"
    assert len(result["context_features"]) == 2
    assert calls["vectors"] == calls["local"] == []


@pytest.mark.parametrize("distance", ["-1m", "0m", "5001m", "nan m", "infinity m", "1e9m"])
def test_bad_distance_is_not_a_place_search_or_silent_default(setup_router, distance):
    run, calls = setup_router
    result = run(f"Find buildings within {distance} of rivers in Pune")
    assert result["status"] == "needs_clarification"
    assert calls["places"] == calls["vectors"] == calls["local"] == []


def test_solar_near_named_place_needs_radius_and_imagery(setup_router):
    run, calls = setup_router
    result = run("Find rooftop solar panels near PMC, Pune")
    assert result["status"] == "needs_clarification"
    assert result["interpretation"]["distance_m"] is None
    assert "radius" in result["message"] and "imagery" in result["message"]
    assert calls["places"] == ["PMC, Pune"]
    assert calls["vectors"] == calls["local"] == []


def test_pune_visual_coverage_never_falls_back_to_phoenix(setup_router):
    run, calls = setup_router
    result = run("Find rooftop solar panels in Pune")
    assert result["status"] == "unavailable" and result["route"] == "visual"
    assert "Pune" in result["message"] and "Phoenix" in result["message"]
    assert calls["local"] == calls["vectors"] == []


def test_named_visual_reference_preserves_numeric_distance_and_coverage_notice(setup_router):
    run, calls = setup_router
    result = run("Find rooftop solar panels within 500m of Pune Municipal Corporation in Pune")
    assert result["status"] == "unavailable"
    assert result["interpretation"]["distance_m"] == 500
    assert "named reference" in result["message"]
    assert calls["local"] == calls["vectors"] == []


@pytest.mark.parametrize("query", [
    "Find red roofs in Pune with solar panels",
    "Find buildings in Pune near rivers",
    "Find solar panels in Pune and Mumbai",
    "Find swimming pools in Phoenix within 100 metres of roads",
    "Find buildings in Pune; with building=house",
])
def test_trailing_conditions_cannot_erase_requested_geography(setup_router, query):
    run, calls = setup_router
    result = run(query)
    assert result["status"] == "needs_clarification"
    assert result["interpretation"]["original_query"] == query
    assert calls["places"] == calls["vectors"] == calls["local"] == []


def test_complex_global_modifier_is_not_lost_or_geocoded_as_a_place(setup_router):
    run, calls = setup_router
    query = "Find unpaved roads intersecting cleared forest patches in Pune"
    result = run(query)
    assert result["status"] == "needs_clarification"
    assert result["interpretation"]["unsupported_conditions"] == ["cleared"]
    assert "cleared" in result["message"]
    assert calls["places"] == calls["vectors"] == calls["local"] == []


def test_declared_phoenix_suffix_selects_correct_local_bundle(setup_router, monkeypatch):
    run, calls = setup_router
    plan = QueryPlan(query_type="objects", concept_id="swimming_pools", visual_query="swimming pools",
        reference_layer="roads", predicate="within_distance", distance_m=100)
    seen = []

    def parse(query, manifest):
        seen.append(query)
        return plan, "catalog_scan"

    monkeypatch.setattr(router, "discovery_plan", parse)
    result = run("Find swimming pools within 100 metres of roads in Phoenix", dataset_id="spacenet")
    assert result["status"] == "completed" and result["dataset_id"] == "phoenix"
    assert seen == ["Find swimming pools within 100 metres of roads"]
    assert calls["local"][0][2] == "phoenix"
    assert calls["places"] == calls["vectors"] == []
    assert "prepared imagery area" in result["message"]


def test_generic_colleges_require_location(setup_router):
    run, calls = setup_router
    result = run("Find colleges")
    assert result["status"] == "needs_clarification"
    assert calls["places"] == calls["vectors"] == calls["local"] == []


def test_world_distance_limit_is_clarified_before_a_provider_request(setup_router):
    run, calls = setup_router
    result = run("Find buildings within 3km of rivers in Pune")
    assert result["status"] == "needs_clarification"
    assert result["interpretation"]["distance_m"] == 3000
    assert calls["places"] == calls["vectors"] == calls["local"] == []


@pytest.mark.parametrize("query", ["Find illegal buildings in Pune", "Find buildings from 2019 in Pune"])
def test_legal_or_historical_conditions_are_not_geocoded_as_places(setup_router, query):
    run, calls = setup_router
    result = run(query)
    assert result["status"] == "needs_clarification"
    assert result["interpretation"]["unsupported_conditions"]
    assert calls["places"] == calls["vectors"] == calls["local"] == []


def test_api_snapshot_export_keeps_all_evidence_without_rerun(monkeypatch):
    monkeypatch.delenv("GEOVERIFY_ROUND1", raising=False)
    calls = []
    matched = place("building", identifier="b")
    matched["properties"]["evidence_geometries"] = [place("observation", identifier="obs")]
    original = {"type": "FeatureCollection", "status": "completed", "route": "visual",
        "result_kind": "object_detections", "features": [matched],
        "uncertain_candidates": [place("uncertain", identifier="u")],
        "context_features": [place("road", identifier="r")],
        "tile_candidates": [place("tile", identifier="t")],
        "total_matches": 1, "total_uncertain": 1, "dataset_id": "phoenix",
        "message": "Fixture evidence", "claim_scope": "Fixture only", "interpretation": {"distance_m": 100},
        "export_scope": {"type": "bounded_returned_subset", "citywide": False, "returned_matches": 1}}

    def run(*args, **kwargs):
        calls.append(args)
        return deepcopy(original)

    monkeypatch.setattr(api.router, "discover", run)
    with TestClient(api.app) as client:
        response = client.post("/api/discover", json={"query": "Find fixture"})
        assert response.status_code == 200
        result = response.json()
        original["features"][0]["properties"]["name"] = "changed after run"
        export = client.get(result["export_url"])
        assert export.status_code == 200
        saved = export.json()
        assert {feature["id"] for feature in saved["features"]} == {"b", "obs", "u", "r", "t"}
        assert saved["metadata"]["interpretation"]["distance_m"] == 100
        assert saved["metadata"]["export_scope"] == {"type": "bounded_returned_subset", "citywide": False, "returned_matches": 1}
        assert "no query, model or provider was rerun" in saved["metadata"]["export_note"]
        assert next(item for item in saved["features"] if item["id"] == "b")["properties"]["name"] == "building"
        assert len(calls) == 1
        assert client.get("/api/discovery/runs/invalid/export").status_code == 404


def test_unified_discovery_preserves_round1_restriction(monkeypatch):
    monkeypatch.setenv("GEOVERIFY_ROUND1", "1")
    with TestClient(api.app) as client:
        response = client.post("/api/discover", json={"query": "Find Sinhagad College in Pune"})
        assert response.status_code == 422 and "SpaceNet" in response.json()["detail"]


def test_world_vector_plan_cannot_become_a_default_phoenix_followup(monkeypatch):
    """World vector plans lack the mandatory local-plan visual identity fields."""
    monkeypatch.delenv("GEOVERIFY_ROUND1", raising=False)
    calls = []
    monkeypatch.setattr(api.router, "discover", lambda *args, **kwargs: calls.append(kwargs))
    # This is the actual on-demand mapped vector plan shape, not a local plan.
    world_plan = {"query_type": "vectors", "subject_layer": "buildings", "reference_layer": "roads",
        "predicate": "within_distance", "distance_m": 300, "attributes": []}
    with TestClient(api.app) as client:
        response = client.post("/api/discover", json={"query": "make it 100 metres",
            "dataset_id": "phoenix", "previous_plan": world_plan})
        assert response.status_code == 422
        assert {error["loc"][-1] for error in response.json()["detail"]} >= {"concept_id", "visual_query"}
        assert calls == []
