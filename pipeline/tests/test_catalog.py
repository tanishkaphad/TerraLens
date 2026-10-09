"""Catalog discovery uses actual source metadata and never chooses a search area."""
import json
from pathlib import Path

import geopandas as gpd
import pytest
from shapely.geometry import LineString, mapping

from terralens.catalog import CatalogError, SourceCatalog
from test_backend import make_bundle, manifest


def vector_source(root: Path, *, dataset_id=None, layer_id="survey_assets"):
    root.mkdir(parents=True)
    layer = root / "actual.geojson"
    gpd.GeoDataFrame([{"asset_type": "custom observation", "inspection_year": 2026,
        "geometry": LineString([(72.8, 18.9), (72.81, 18.91)])}], crs="EPSG:4326").to_file(layer, driver="GeoJSON")
    details = {"dataset_version": "survey-v1", "area_name": "Administrator survey",
        "vector_layers": {layer_id: {"file": layer.name, "source": "field survey", "extraction_date": "2026-10-01"}},
        "attribution": {"vectors": "Field team"}}
    if dataset_id is not None:
        details["dataset_id"] = dataset_id
    (root / "manifest.json").write_text(json.dumps(details))
    return details


def test_discovers_arbitrary_names_and_observed_schema(tmp_path, monkeypatch):
    monkeypatch.delenv("BUNDLE_DIR", raising=False)
    monkeypatch.delenv("DEFAULT_DATASET_ID", raising=False)
    vector_source(tmp_path / "some-new-region", dataset_id="survey-2026")
    catalog = SourceCatalog([tmp_path])
    assert list(catalog.datasets()) == ["survey-2026"]
    description = catalog.describe()
    layer = description["layers"]["survey-2026:survey_assets"]
    assert layer["fields"] == ["asset_type", "inspection_year"]
    assert layer["observed_values"] == {"asset_type": ["custom observation"], "inspection_year": [2026]}
    assert layer["geometry_types"] == ["LineString"]
    assert layer["feature_count"] == 1
    assert layer["dates"]["extraction_date"] == "2026-10-01"
    assert description["datasets"]["survey-2026"]["capabilities"]["mapped_features"]["available"]
    assert not description["datasets"]["survey-2026"]["capabilities"]["scene_retrieval"]["available"]
    assert catalog.get_bundle("survey-2026").tiles == []
    with pytest.raises(CatalogError, match="no default geography"):
        catalog.get_bundle()


def test_env_roots_and_explicit_bundle_are_additive(tmp_path, monkeypatch):
    other = tmp_path / "outside" / "arbitrary"
    root = tmp_path / "catalog"
    vector_source(root / "nested" / "new_area")
    vector_source(other, dataset_id="configured-id")
    monkeypatch.setenv("BUNDLE_CATALOG_ROOTS", str(root))
    monkeypatch.setenv("BUNDLE_DIR", str(other))
    monkeypatch.delenv("DEFAULT_DATASET_ID", raising=False)
    catalog = SourceCatalog()
    assert set(catalog.datasets()) == {"nested-new_area", "configured-id"}
    assert catalog.legacy_default_id == "configured-id"
    assert catalog.datasets()["configured-id"]["default"] is True


def test_safe_stable_ids_and_collision_handling(tmp_path, monkeypatch):
    monkeypatch.delenv("BUNDLE_DIR", raising=False)
    vector_source(tmp_path / "first", dataset_id="Repeated / geography")
    vector_source(tmp_path / "second", dataset_id="Repeated / geography")
    catalog = SourceCatalog([tmp_path])
    identifiers = list(catalog.datasets())
    assert len(identifiers) == 2 and len(set(identifiers)) == 2
    assert all(len(identifier) <= 40 and "/" not in identifier for identifier in identifiers)
    assert SourceCatalog([tmp_path]).datasets().keys() == catalog.datasets().keys()
    with pytest.raises(CatalogError):
        catalog.get_bundle("../first")


def test_changed_manifest_and_vector_invalidate_loaded_source(tmp_path, monkeypatch):
    monkeypatch.delenv("BUNDLE_DIR", raising=False)
    root = tmp_path / "source"
    manifest = vector_source(root, dataset_id="mutable-source")
    catalog = SourceCatalog([tmp_path])
    first = catalog.get_bundle("mutable-source")
    revision = catalog.revision
    assert catalog.get_bundle("mutable-source") is first
    manifest["dataset_version"] = "survey-v2"
    (root / "manifest.json").write_text(json.dumps(manifest))
    assert catalog.refresh()
    assert catalog.revision != revision
    second = catalog.get_bundle("mutable-source")
    assert second is not first and second.manifest["dataset_version"] == "survey-v2"
    data = json.loads((root / "actual.geojson").read_text())
    data["features"][0]["properties"]["asset_type"] = "new observation"
    (root / "actual.geojson").write_text(json.dumps(data))
    assert catalog.get_bundle("mutable-source") is not second
    assert catalog.describe()["layers"]["mutable-source:survey_assets"]["observed_values"]["asset_type"] == ["new observation"]


def test_invalid_sources_visible_and_layer_return_does_not_mutate(tmp_path, monkeypatch):
    monkeypatch.delenv("BUNDLE_DIR", raising=False)
    root = tmp_path / "source"
    vector_source(root, dataset_id="survey")
    malformed = tmp_path / "bad"
    malformed.mkdir()
    (malformed / "manifest.json").write_text("not json")
    catalog = SourceCatalog([tmp_path])
    assert catalog.describe()["errors"]
    frame = catalog.layer_features("survey", "survey_assets")
    frame.loc[0, "asset_type"] = "altered copy"
    assert catalog.get_bundle("survey").layers["survey_assets"].loc[0, "asset_type"] == "custom observation"


def test_prepared_bundle_still_uses_frozen_validation(tmp_path, manifest, monkeypatch):
    monkeypatch.delenv("BUNDLE_DIR", raising=False)
    make_bundle(tmp_path, manifest)
    manifest["dataset_id"] = "imagery-source"
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    catalog = SourceCatalog([tmp_path])
    bundle = catalog.get_bundle("imagery-source")
    assert len(bundle.tiles) == 1 and bundle.embeddings.shape == (1, 2)
    assert catalog.describe()["datasets"]["imagery-source"]["dates"]["imagery"] == ["2024-06-01"]


def test_damaged_image_index_keeps_independently_valid_vector_evidence(tmp_path, manifest, monkeypatch):
    monkeypatch.delenv("BUNDLE_DIR", raising=False)
    make_bundle(tmp_path, manifest)
    (tmp_path / "embeddings.npy").write_bytes(b"damaged index")
    catalog = SourceCatalog([tmp_path])
    identifier = next(iter(catalog.datasets()))
    bundle = catalog.get_bundle(identifier)
    assert not bundle.tiles and len(bundle.layers["roads"]) == 2
    description = catalog.describe()["datasets"][identifier]
    assert description["status"] == "available"
    assert description["imagery_unavailable_reason"]
    assert description["capabilities"]["mapped_features"]["available"]
    assert not description["capabilities"]["object_grounding"]["available"]


def test_vector_metadata_counts_records_without_map_geometry(tmp_path, monkeypatch):
    monkeypatch.delenv("BUNDLE_DIR", raising=False)
    root = tmp_path / "source"
    vector_source(root, dataset_id="survey")
    collection = json.loads((root / "actual.geojson").read_text())
    collection["features"].append({"type": "Feature", "geometry": None,
        "properties": {"asset_type": "location missing", "inspection_year": 2026}})
    collection["features"].append({"type": "Feature", "geometry": {"type": "LineString", "coordinates": []},
        "properties": {"asset_type": "empty source geometry", "inspection_year": 2026}})
    (root / "actual.geojson").write_text(json.dumps(collection))
    catalog = SourceCatalog([tmp_path])
    bundle = catalog.get_bundle("survey")
    description = catalog.describe()["layers"]["survey:survey_assets"]
    assert description["source_record_count"] == 3
    assert description["feature_count"] == 1
    assert description["omitted_geometry"] == 2
    assert sum(description["omitted_geometry_reasons"].values()) == 2
    assert bundle.vector_geometry_metadata["survey_assets"]["omitted_geometry"] == 2
    assert len(catalog.layer_features("survey", "survey_assets")) == 1


def test_prepared_bundle_reports_omitted_vector_records(tmp_path, manifest, monkeypatch):
    monkeypatch.delenv("BUNDLE_DIR", raising=False)
    make_bundle(tmp_path, manifest)
    path = tmp_path / "vectors" / "roads.geojson"
    collection = json.loads(path.read_text())
    collection["features"].append({"type": "Feature", "geometry": None,
        "properties": {"id": "location-missing", "surface": "unknown"}})
    path.write_text(json.dumps(collection))
    catalog = SourceCatalog([tmp_path])
    dataset_id = next(iter(catalog.datasets()))
    layer = catalog.describe()["layers"][dataset_id + ":roads"]
    assert layer["source_record_count"] == 3 and layer["feature_count"] == 2
    assert layer["omitted_geometry"] == 1
    assert len(catalog.get_bundle(dataset_id).layers["roads"]) == 2
