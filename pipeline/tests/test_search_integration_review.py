"""Cross-module regression cases found during scope/provenance review.

Local image/geometry data is real synthetic fixture data. Inference and provider
responses are explicit small test doubles, never reported as live validation.
"""
from copy import deepcopy
import json
import threading
from types import SimpleNamespace

import numpy as np
import pytest
from shapely.geometry import box, mapping

from terralens.catalog import SourceCatalog
from terralens.grounding import GroundingError
from terralens.search_service import SearchService
from terralens.search_sources import ConfiguredOSMSource
from terralens.search_store import SearchStore
from terralens.search_workers import SearchWorkerRunner
from terralens import world_data
from test_backend import make_bundle, manifest


@pytest.fixture
def review_service(tmp_path, manifest, monkeypatch):
    monkeypatch.delenv("BUNDLE_DIR", raising=False)
    monkeypatch.delenv("SEARCH_OSM_CATALOG", raising=False)
    root = tmp_path / "explicit_source"
    root.mkdir()
    manifest = deepcopy(manifest)
    manifest["dataset_id"] = "review-source"
    bundle = make_bundle(root, manifest, duplicate=True)
    for tile in bundle.tiles:
        tile.update(crs="EPSG:32631", transform=[1, 0, 500000, 0, -1, 100020])
    (root / "tiles.json").write_text(json.dumps(bundle.tiles))
    catalog = SourceCatalog([tmp_path])
    catalog.get_bundle("review-source").encode_text = lambda text: np.array([1., 0.], dtype=np.float32)
    def unavailable(_bundle):
        raise GroundingError("Fixture detector deliberately unavailable.")
    grounding = SimpleNamespace(_inference_lock=threading.Lock(), _load_detector=unavailable)
    store = SearchStore(tmp_path / "search.sqlite3")
    runner = SearchWorkerRunner(store, workers=2, max_pending=2)
    service = SearchService(store, catalog=catalog, runner=runner, osm=ConfiguredOSMSource(), grounding=grounding)
    yield service
    runner.shutdown()


def run_synchronously(service, plan):
    run = service.create(plan=plan, start=False)
    service.runner.run(run["run_id"], service._processor(run["run_id"]))
    return service.store.get_run(run["run_id"])


def test_grounding_fallback_preserves_missing_processor_and_scene_provenance(review_service):
    plan = {"original_query": "Search these loaded images for an unseen visual object",
        "scope": {"mode": "all_loaded", "dataset_ids": ["review-source"]},
        "entities": [{"id": "subject", "kind": "visual", "description": "unseen visual object",
            "visual_query": "unseen visual object", "dataset_id": "review-source"}],
        "subject_entity_ids": ["subject"], "clauses": []}
    final = run_synchronously(review_service, plan)
    assert final["status"] == "completed", final
    assert final["counts"]["matches"] == 0
    assert final["counts"]["visual_candidates"] == 2
    assert final["counts"]["coverage_gaps"] == 2
    coverage = [unit["coverage"] for unit in final["coverage"]["units"]
        if unit["coverage"].get("entity_id") == "subject"]
    assert all(item["status"] == "missing_processor" for item in coverage)
    assert all(item["source_enumeration"] == "ranked_registered_window" for item in coverage)
    features = review_service.store.export(final["run_id"])["features"]
    candidates = [feature for feature in features if feature["properties"].get("result_group") == "visual_candidates"]
    assert all(feature["properties"]["geometry_role"] == "scene_footprint" for feature in candidates)
    assert all(feature["properties"]["visual_evidence"][0]["status"] == "unknown" for feature in candidates)


def test_vector_snapshot_outside_requested_dates_cannot_be_full_match(review_service):
    plan = {"original_query": "Find roads observed during the requested historical period",
        "scope": {"mode": "all_loaded", "dataset_ids": ["review-source"],
            "start_date": "2022-01-01", "end_date": "2022-12-31"},
        "entities": [{"id": "roads", "kind": "mapped", "description": "mapped roads",
            "dataset_id": "review-source", "layer_id": "roads"}],
        "subject_entity_ids": ["roads"], "clauses": []}
    final = run_synchronously(review_service, plan)
    assert final["status"] == "completed", final
    assert final["counts"]["matches"] == 0
    assert final["counts"]["coverage_gaps"] > 0
    assert final["has_coverage_gaps"] is True
    assert final["plan"]["scope"]["start_date"] == "2022-01-01"


def test_polygon_osm_layer_rejects_tagged_node_geometry(tmp_path, monkeypatch):
    configuration = tmp_path / "osm.json"
    configuration.write_text(json.dumps({"layers": {"administrative_areas": {
        "geometry_type": "polygon", "fields": ["boundary"],
        "selectors": [{"field": "boundary", "value": "administrative"}]}}}))
    provider = ConfiguredOSMSource(configuration)
    payload = {"elements": [{"type": "node", "id": 1, "lat": 18.5, "lon": 73.8,
        "tags": {"boundary": "administrative"}}], "osm3s": {"timestamp_osm_base": "2026-10-09T00:00:00Z"}}
    monkeypatch.setattr(world_data, "_cached_json", lambda *args, **kwargs: (payload, "2026-10-09T00:01:00Z", False))
    features, coverage = provider.extract("administrative_areas", mapping(box(73.79, 18.49, 73.81, 18.51)),
        catalog_version=provider.revision)
    assert features == []
    assert coverage["omitted_geometry"] == 1
    assert coverage["source_enumeration"] == "incomplete_geometry"
