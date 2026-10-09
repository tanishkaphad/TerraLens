"""Synthetic geometry contracts and provider boundary tests, without network IO."""
from concurrent.futures import ThreadPoolExecutor
import json
import threading
import time

from pyproj import Transformer
import pytest
import requests
from shapely.geometry import LineString, Point, box, mapping
from shapely.ops import transform

from terralens import world_data as world


RETRIEVED = "2026-10-09T07:00:00Z"
SOURCE_DATE = "2026-10-09T06:45:00Z"
LOCATION = {"type": "Feature", "geometry": {"type": "Point", "coordinates": [73.86, 18.52]},
    "properties": {"name": "Synthetic center", "display_name": "Synthetic center, India", "osm_type": "node", "osm_id": 99}}
TO_METRIC = Transformer.from_crs("EPSG:4326", "EPSG:32643", always_xy=True)
TO_GEO = Transformer.from_crs("EPSG:32643", "EPSG:4326", always_xy=True)
ORIGIN = TO_METRIC.transform(*LOCATION["geometry"]["coordinates"])


def geographic(geometry):
    return transform(lambda x, y, z=None: TO_GEO.transform(x + ORIGIN[0], y + ORIGIN[1]), geometry)


def way(identifier, geometry, tags):
    geo = geographic(geometry)
    coordinates = geo.exterior.coords if geo.geom_type == "Polygon" else geo.coords
    return {"type": "way", "id": identifier, "tags": tags,
        "geometry": [{"lon": lon, "lat": lat} for lon, lat in coordinates]}


def snapshot(elements):
    return {"elements": elements, "osm3s": {"timestamp_osm_base": SOURCE_DATE}}


def mock_snapshot(monkeypatch, elements):
    calls = []
    def fetch(provider, url, **arguments):
        calls.append((provider, url, arguments))
        return snapshot(elements), RETRIEVED, False
    monkeypatch.setattr(world, "_cached_json", fetch)
    return calls


@pytest.fixture(autouse=True)
def isolated_cache(monkeypatch, tmp_path):
    world._CACHE.clear()
    world._LAST_REQUEST.clear()
    monkeypatch.setenv("WORLD_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.delenv("PHOTON_URL", raising=False)
    monkeypatch.delenv("OVERPASS_URL", raising=False)
    monkeypatch.setattr(world, "_REQUEST_INTERVAL", {"places": 0, "vectors": 0})


def test_place_normalization_geojson_extent_identity_and_no_fabricated_spelling(monkeypatch):
    calls = []
    raw = {"type": "Feature", "geometry": {"type": "Point", "coordinates": [73.8, 18.4]},
        "properties": {"name": "Sinhgad College", "city": "Pune", "state": "Maharashtra", "country": "India",
            "osm_type": "W", "osm_id": 123, "osm_key": "amenity", "osm_value": "college",
            "extent": [73.7, 18.5, 73.9, 18.3], "source_url": "https://untrusted.invalid/?key=private"}}
    def fetch(provider, url, **arguments):
        calls.append((provider, url, arguments))
        return {"features": [raw, raw]}, RETRIEVED, True
    monkeypatch.setattr(world, "_cached_json", fetch)
    results = world.lookup_places("Find Sinhagad College in Pune")
    assert calls[0][2]["params"]["q"] == "Sinhagad College, Pune"
    assert calls[0][1] == "https://photon.komoot.io/api/"
    assert len(results) == 1
    assert results[0]["bbox"] == [73.7, 18.3, 73.9, 18.5]
    assert results[0]["properties"]["name"] == "Sinhgad College"
    assert results[0]["properties"]["source_url"] == "https://www.openstreetmap.org/way/123"
    assert results[0]["properties"]["evidence_kind"] == "mapped_place"
    assert results[0]["properties"]["retrieved_at"] == RETRIEVED
    assert "private" not in json.dumps(results)


def test_places_preserve_real_ambiguity_and_skip_invalid_coordinates(monkeypatch):
    def feature(identifier, coordinates):
        return {"type": "Feature", "geometry": {"type": "Point", "coordinates": coordinates},
            "properties": {"name": "College", "osm_type": "N", "osm_id": identifier}}
    monkeypatch.setattr(world, "_cached_json", lambda *args, **kwargs: (
        {"features": [feature(1, [73.8, 18.4]), feature(2, [73.9, 18.5]), feature(3, [181, 18.5])]}, RETRIEVED, False))
    result = world.lookup_places("College, Pune")
    assert [f["properties"]["osm_id"] for f in result] == [1, 2]


@pytest.mark.parametrize("value", ["", "a", "a" * 301, "Pune\nIndia"])
def test_invalid_place_queries_fail_before_request(value, monkeypatch):
    monkeypatch.setattr(world, "_cached_json", lambda *a, **k: pytest.fail("Unexpected network call"))
    with pytest.raises(world.WorldDataError):
        world.lookup_places(value)


def test_building_distance_uses_footprint_not_centroid(monkeypatch):
    elements = [way(1, box(0, -10, 200, 10), {"building": "yes"}),
        way(2, LineString([(210, -100), (210, 100)]), {"highway": "residential"})]
    mock_snapshot(monkeypatch, elements)
    result = world.vector_search("buildings", "roads", 20, LOCATION)
    assert result["total_matches"] == 1
    condition = result["features"][0]["properties"]["conditions"][-1]
    assert condition["distance_m"] == pytest.approx(10, abs=0.001)
    assert condition["distance_basis"] == "source_geometry_to_source_geometry"
    assert condition["reference_feature_ids"] == ["osm:way:2"]
    assert result["context_features"][0]["geometry"]["type"] == "LineString"
    assert result["features"][0]["properties"]["evidence_kind"] == "mapped_vector"
    assert result["features"][0]["properties"]["source_date"] == SOURCE_DATE


def test_reference_halo_covers_subjects_at_area_edge(monkeypatch):
    elements = [way(1, box(80, -5, 100, 5), {"building": "yes"}),
        way(2, LineString([(120, -20), (120, 20)]), {"highway": "service"}),
        way(3, box(140, 0, 160, 20), {"building": "yes"})]
    calls = mock_snapshot(monkeypatch, elements)
    result = world.vector_search("buildings", "roads", 25, LOCATION, radius_m=100)
    query = calls[0][2]["data"]["data"]
    assert "around:100.00" in query and "around:125.00" in query
    assert result["objects_evaluated"] == 1
    assert result["total_matches"] == 1
    assert result["scope"]["radius_m"] == 100
    assert result["scope"]["reference_radius_m"] == 125
    assert result["scope"]["citywide"] is False


def test_missing_references_are_unknown_not_absence(monkeypatch):
    mock_snapshot(monkeypatch, [way(1, box(0, 0, 20, 20), {"building": "yes"})])
    result = world.vector_search("buildings", "rivers", 100, LOCATION)
    assert result["total_matches"] == 0
    assert result["total_uncertain"] == 1
    condition = result["uncertain_candidates"][0]["properties"]["conditions"][-1]
    assert condition["status"] == "unknown"
    assert condition["reference_feature_ids"] == []
    assert "absence is not established" in condition["note"]
    assert result["scope"]["reference_coverage_complete"] is False


def test_nonqualifying_mapped_relationship_remains_unknown_coverage(monkeypatch):
    mock_snapshot(monkeypatch, [way(1, box(0, 0, 20, 20), {"building": "yes"}),
        way(2, LineString([(120, -100), (120, 100)]), {"highway": "residential"})])
    result = world.vector_search("buildings", "roads", 30, LOCATION)
    assert result["total_matches"] == 0
    assert result["total_uncertain"] == 1
    assert not result["context_features"]


def test_building_node_does_not_substitute_for_a_footprint(monkeypatch):
    mock_snapshot(monkeypatch, [{"type": "node", "id": 1, "lon": 73.86, "lat": 18.52, "tags": {"building": "yes"}},
        way(2, LineString([(-5, 0), (5, 0)]), {"highway": "service"})])
    result = world.vector_search("buildings", "roads", 30, LOCATION)
    assert result["objects_evaluated"] == result["total_matches"] == 0


def test_plain_mapped_college_can_be_a_real_osm_point(monkeypatch):
    mock_snapshot(monkeypatch, [{"type": "node", "id": 1, "lon": 73.86, "lat": 18.52,
        "tags": {"amenity": "college", "name": "Synthetic college"}}])
    result = world.vector_search("colleges", None, None, LOCATION)
    assert result["total_matches"] == 1
    assert result["features"][0]["geometry"]["type"] == "Point"
    assert result["plan"]["predicate"] is None


def test_intersection_requires_actual_geometry_contact(monkeypatch):
    elements = [way(1, LineString([(-10, 0), (10, 0)]), {"highway": "track"}),
        way(2, box(-5, -5, 5, 5), {"landuse": "forest"})]
    mock_snapshot(monkeypatch, elements)
    result = world.vector_search("roads", "forests", None, LOCATION, predicate="intersects")
    assert result["total_matches"] == 1
    assert result["features"][0]["properties"]["conditions"][-1]["condition"] == "intersects"
    assert result["features"][0]["properties"]["conditions"][-1]["distance_m"] == 0


def test_multipolygon_holes_and_incomplete_boundaries_are_respected():
    outer = way(1, box(-20, -20, 20, 20), {})["geometry"]
    inner = way(2, box(-5, -5, 5, 5), {})["geometry"]
    relation = {"type": "relation", "id": 100, "tags": {"type": "multipolygon", "building": "yes"},
        "members": [{"type": "way", "role": "outer", "geometry": outer},
            {"type": "way", "role": "inner", "geometry": inner}]}
    geometry = world._geometry(relation)
    assert geometry is not None and len(geometry.interiors) == 1
    assert not geometry.covers(Point(LOCATION["geometry"]["coordinates"]))
    relation["members"].append({"type": "way", "role": "outer",
        "geometry": way(3, LineString([(50, 50), (60, 60)]), {})["geometry"]})
    assert world._geometry(relation) is None


@pytest.mark.parametrize("subject,reference,distance,radius", [
    ("solar panels", "roads", 30, 2000), ("unpaved roads", "forests", 30, 2000),
    ("buildings", "roads", 2001, 2000), ("buildings", "roads", 30, 2001),
    ("buildings", "roads", float("nan"), 2000), ("buildings", "roads", None, 2000),
    ("buildings", "roads", True, 2000),
])
def test_unsupported_attributes_and_unbounded_queries_fail_before_request(monkeypatch, subject, reference, distance, radius):
    monkeypatch.setattr(world, "_cached_json", lambda *a, **k: pytest.fail("Unexpected network call"))
    with pytest.raises(world.WorldDataError):
        world.vector_search(subject, reference, distance, LOCATION, radius_m=radius)


def test_all_displayed_witnesses_have_context_after_result_cap(monkeypatch):
    elements = []
    # Reverse IDs exercise sorting after evaluation and the 100-feature cap.
    for number in reversed(range(105)):
        x = number * 5 - 260
        elements.extend([way(number + 1, box(x, 0, x + 2, 2), {"building": "yes"}),
            way(number + 1001, LineString([(x + 3, -5), (x + 3, 5)]), {"highway": "service"})])
    mock_snapshot(monkeypatch, elements)
    result = world.vector_search("buildings", "roads", 1.1, LOCATION)
    assert result["total_matches"] == 105
    assert len(result["features"]) == 100
    assert result["truncated"] is True
    witnesses = {identifier for feature in result["features"] for condition in feature["properties"]["conditions"]
        for identifier in condition.get("reference_feature_ids", [])}
    assert witnesses == {feature["id"] for feature in result["context_features"]}
    assert result["export_scope"]["all_returned_relationship_witnesses_included"] is True
    assert result["export_scope"]["kind"] == "bounded_returned_subset"


def test_total_returned_subject_features_remains_bounded(monkeypatch):
    elements = [way(number + 1, box(number * 3 - 200, 0, number * 3 - 198, 2), {"building": "yes"})
        for number in range(105)]
    elements.append(way(2000, LineString([(-300, 10), (0, 10)]), {"highway": "service"}))
    mock_snapshot(monkeypatch, elements)
    result = world.vector_search("buildings", "roads", 10, LOCATION)
    assert len(result["features"]) + len(result["uncertain_candidates"]) <= 100
    assert result["uncertain_truncated"] is True


def test_provider_timeout_partial_snapshot_is_rejected(monkeypatch):
    monkeypatch.setattr(world, "_cached_json", lambda *a, **k: (
        {"elements": [], "remark": "runtime error at private URL/key"}, RETRIEVED, False))
    with pytest.raises(world.WorldDataError, match="did not finish") as caught:
        world.vector_search("buildings", None, None, LOCATION)
    assert caught.value.status_code == 503
    assert "private" not in str(caught.value)


class FakeResponse:
    status_code = 200
    def __init__(self, payload):
        self.payload = payload
    def __enter__(self):
        return self
    def __exit__(self, *args):
        return False
    def iter_content(self, chunk_size):
        yield json.dumps(self.payload).encode()


def test_cached_responses_keep_retrieval_date_and_survive_memory_reset(monkeypatch, tmp_path):
    calls = []
    def request(*args, **kwargs):
        calls.append(kwargs)
        return FakeResponse({"features": []})
    monkeypatch.setattr(world.requests, "request", request)
    first = world._cached_json("places", "https://example.invalid/api", params={"q": "Pune"})
    second = world._cached_json("places", "https://example.invalid/api", params={"q": "Pune"})
    world._CACHE.clear()
    third = world._cached_json("places", "https://example.invalid/api", params={"q": "Pune"})
    assert first[2] is False and second[2] is third[2] is True
    assert first[1] == second[1] == third[1]
    assert len(calls) == 1
    assert calls[0]["stream"] is True
    assert calls[0]["allow_redirects"] is False


def test_simultaneous_identical_requests_share_one_serialized_call(monkeypatch):
    lock = threading.Lock()
    count = 0
    def request(*args, **kwargs):
        nonlocal count
        with lock:
            count += 1
        time.sleep(0.02)
        return FakeResponse({"features": []})
    monkeypatch.setattr(world.requests, "request", request)
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(lambda _: world._cached_json("places", "https://example.invalid/api", params={"q": "Pune"}), range(4)))
    assert count == 1
    assert sum(not result[2] for result in results) == 1


def test_provider_error_never_discloses_endpoint_credentials(monkeypatch):
    def request(*args, **kwargs):
        raise requests.ConnectionError("https://example.invalid/?key=private-secret")
    monkeypatch.setattr(world.requests, "request", request)
    with pytest.raises(world.WorldDataError) as caught:
        world._cached_json("places", "https://example.invalid/?key=private-secret", params={"q": "Pune"})
    assert caught.value.status_code == 503
    assert "secret" not in str(caught.value) and "example.invalid" not in str(caught.value)


def test_relative_cache_path_is_anchored_to_pipeline(monkeypatch):
    monkeypatch.setenv("WORLD_CACHE_DIR", "data/world-cache")
    assert world._cache_path("a").parent == world.Path(world.__file__).resolve().parents[1] / "data" / "world-cache"


def test_status_does_not_expose_provider_url_or_key(monkeypatch):
    monkeypatch.setenv("PHOTON_URL", "https://private.invalid/?key=private-secret")
    assert "private" not in json.dumps(world.status())
