import json
import time

import pytest
from fastapi.testclient import TestClient
from shapely.geometry import Point, box, mapping

from terralens import api, search_api
from terralens.catalog import SourceCatalog
from terralens.search_service import SearchService, SearchServiceError
from terralens.search_sources import ConfiguredOSMSource
from terralens.search_store import SearchStore
from terralens.search_workers import SearchWorkerRunner


def source(tmp_path, *, count=205):
    root = tmp_path / "newly_registered_area"
    root.mkdir()
    features = [{"type": "Feature", "id": f"point-{i}",
        "geometry": mapping(Point(73.815 + i * 0.000001, 18.515)),
        "properties": {"id": f"point-{i}", "custom:use": "testing_category"}} for i in range(count)]
    (root / "objects.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": features}))
    (root / "manifest.json").write_text(json.dumps({"dataset_id": "unseen-area", "dataset_version": "version-1",
        "area_name": "An operator-added area", "vector_layers": {"objects": {"file": "objects.geojson",
            "fields": ["id", "custom:use"], "reference_coverage_complete": True,
            "source": "independent-test-fixture", "extraction_date": "2026-10-09"}}}))
    return root


def explicit_plan(dataset_id="unseen-area", *, clauses=None, expression=None):
    return {"original_query": "Find the requested category in this explicitly selected area",
        "scope": {"mode": "geometry", "geometry": mapping(box(73.81, 18.51, 73.82, 18.52)),
            "dataset_ids": [dataset_id]},
        "entities": [{"id": "subject", "description": "A previously unseen mapped category", "kind": "mapped",
            "dataset_id": dataset_id, "layer_id": "objects"}], "subject_entity_ids": ["subject"],
        "clauses": clauses or [], "expression": expression}


@pytest.fixture
def service(tmp_path, monkeypatch):
    source(tmp_path)
    monkeypatch.setenv("SEARCH_VECTOR_CHUNK_SIZE", "27")
    monkeypatch.delenv("SEARCH_OSM_CATALOG", raising=False)
    monkeypatch.delenv("BUNDLE_DIR", raising=False)
    store = SearchStore(tmp_path / "search.sqlite3")
    runner = SearchWorkerRunner(store, workers=3, max_pending=5)
    instance = SearchService(store, runner=runner, catalog=SourceCatalog([tmp_path]), osm=ConfiguredOSMSource())
    yield instance
    runner.shutdown()


def finish(service, run_id):
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        run = service.store.get_run(run_id)
        if run["status"] not in {"queued", "running"}:
            return run
        time.sleep(0.01)
    pytest.fail("Search failed to finish within the test deadline")


def test_new_catalog_layer_runs_persists_pages_and_exports_every_result(service):
    run = service.create(plan=explicit_plan())
    final = finish(service, run["run_id"])
    assert final["status"] == "completed", final
    assert final["counts"]["matches"] == 205
    assert final["total_results"] == 205
    identifiers = []
    page = service.store.results(run["run_id"], page_size=19)
    revision = page["revision"]
    while True:
        identifiers.extend(record["id"] for record in page["records"])
        if page["next_cursor"] is None:
            break
        page = service.store.results(run["run_id"], cursor=page["next_cursor"], page_size=19)
    assert len(identifiers) == len(set(identifiers)) == 205
    exported = service.store.export(run["run_id"], revision=revision)
    subjects = [feature for feature in exported["features"] if feature["properties"].get("result_group") == "matches"]
    assert len(subjects) == 205
    assert len(service.store.map_data(run["run_id"])["features"]) == 205
    # Result revision, not a provider or a subsequently changed source, controls export.
    original = json.dumps(exported, sort_keys=True)
    source_file = service.catalog.datasets()["unseen-area"]["path"] / "objects.geojson"
    source_file.write_text('{"type":"FeatureCollection","features":[]}')
    assert json.dumps(service.store.export(run["run_id"], revision=revision), sort_keys=True) == original


def test_source_mutation_blocks_remaining_units_without_erasing_snapshot(service):
    run = service.create(plan=explicit_plan(), start=False)
    unit = service.store.claim_unit(run["run_id"])
    output = service.process_unit(run["run_id"], unit, lambda _: None)
    service.store.commit_unit(run["run_id"], unit["unit_id"], unit["lease_token"], output["records"], output["coverage"])
    revision = service.store.snapshot(run["run_id"])
    prior = service.store.export(run["run_id"], revision=revision)
    path = service.catalog.datasets()["unseen-area"]["path"] / "objects.geojson"
    path.write_text('{"type":"FeatureCollection","features":[]}')
    next_unit = service.store.claim_unit(run["run_id"])
    with pytest.raises(SearchServiceError, match="changed"):
        service.process_unit(run["run_id"], next_unit, lambda _: None)
    assert service.store.export(run["run_id"], revision=revision) == prior


def test_missing_attribute_stays_in_plan_and_review_group(service):
    clause = {"id": "needed", "entity_id": "subject", "kind": "attribute", "operator": "equals",
        "field": "not_in_source", "value": "must_be_known", "original_text": "with this unknown attribute"}
    plan = explicit_plan(clauses=[clause], expression={"operator": "clause", "clause_id": "needed"})
    run = service.create(plan=plan)
    final = finish(service, run["run_id"])
    assert final["status"] == "completed", final
    assert final["counts"]["needs_review"] == 205
    assert final["counts"]["matches"] == 0
    assert final["plan"]["clauses"][0]["field"] == "not_in_source"


def test_area_scope_is_not_silently_substituted(service):
    plan = explicit_plan()
    plan["scope"]["geometry"] = mapping(box(77, 28, 77.01, 28.01))
    run = service.create(plan=plan)
    final = finish(service, run["run_id"])
    assert final["status"] == "completed", final
    assert final["counts"]["matches"] == 0
    assert final["plan"]["scope"]["geometry"] == json.loads(json.dumps(plan["scope"]["geometry"]))


def test_http_end_to_end_controls_snapshots_and_complete_map(service, monkeypatch):
    monkeypatch.setattr(search_api, "get_search_service", lambda: service)
    with TestClient(api.app) as client:
        catalog = client.get("/api/catalog")
        assert catalog.status_code == 200
        assert "unseen-area:objects" in catalog.json()["layers"]
        queued = client.post("/api/search", json={"plan": explicit_plan(), "start": False})
        assert queued.status_code == 202, queued.text
        run = queued.json()
        run_id = run["run_id"]
        assert client.post(f"/api/search/runs/{run_id}/pause").status_code == 200
        assert client.post(f"/api/search/runs/{run_id}/resume").status_code == 202
        final = finish(service, run_id)
        assert final["status"] == "completed", final
        paged = client.get(run["results_url"], params={"page_size": 5}).json()
        assert len(paged["records"]) == 5
        assert len(client.get(run["map_url"]).json()["features"]) == 205
        exported = client.get(run["export_url"], params={"revision": final["revision"]})
        assert exported.status_code == 200
        assert exported.headers["content-type"].startswith("application/geo+json")
        assert len(exported.json()["features"]) == 205
        assert client.get("/api/search/schema").status_code == 200
        assert client.get("/api/search/runs/missing").status_code == 404
