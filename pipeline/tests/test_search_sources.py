from copy import deepcopy
import json

import pytest
from shapely.geometry import box, mapping, shape
from shapely.ops import unary_union

from terralens.search_sources import ConfiguredOSMSource, SourceError, buffered_geometry, partition_geometry
from terralens import world_data


def test_partition_preserves_actual_polygon_and_requested_reference_margin():
    area = box(73.81, 18.51, 73.825, 18.525).difference(box(73.82, 18.52, 73.825, 18.525))
    partitions = partition_geometry(mapping(area), 500)
    assert len(partitions) > 4
    combined = unary_union([shape(item["geometry"]) for item in partitions])
    assert combined.symmetric_difference(area).area < 1e-10
    expanded = shape(buffered_geometry(mapping(area), 350))
    assert expanded.contains(area)
    assert expanded.area > area.area
    with pytest.raises(SourceError):
        partition_geometry(mapping(area), -1)


def test_provider_uses_registered_arbitrary_selectors_and_keeps_all_outputs(tmp_path, monkeypatch):
    path = tmp_path / "providers.json"
    path.write_text(json.dumps({"layers": {"charging_points": {"geometry_type": "point",
        "fields": ["amenity", "operator"], "selectors": [{"field": "amenity", "value": "charging_station"}]}}}))
    provider = ConfiguredOSMSource(path)
    assert "osm:charging_points" in provider.describe()
    captured = []
    elements = [{"type": "node", "id": index, "lon": 73.815, "lat": 18.515,
        "tags": {"amenity": "charging_station", "operator": "source_record"}} for index in range(225)]
    def fetch(kind, url, *, data):
        captured.append(data["data"])
        return {"elements": deepcopy(elements), "osm3s": {"timestamp_osm_base": "2026-10-09T01:00:00Z"}}, "retrieval", False
    monkeypatch.setattr(world_data, "_cached_json", fetch)
    features, coverage = provider.extract("charging_points", mapping(box(73.81, 18.51, 73.82, 18.52)),
        catalog_version=provider.revision)
    assert len(features) == 225
    assert '["amenity"="charging_station"]' in captured[0]
    assert "around:" not in captured[0]
    assert coverage["source_enumeration"] == "response_complete"
    assert coverage["data_coverage"] == "map_completeness_unknown"
    assert all(item["properties"]["snapshot_date"] == "2026-10-09T01:00:00Z" for item in features)


def test_provider_does_not_mask_incomplete_geometry_or_mutating_configuration(tmp_path, monkeypatch):
    path = tmp_path / "providers.json"
    config = {"layers": {"parcels": {"geometry_type": "polygon", "fields": ["custom:kind"],
        "selectors": [{"field": "custom:kind", "value": 'value";out;'}]}}}
    path.write_text(json.dumps(config))
    provider = ConfiguredOSMSource(path)
    captured = []
    def fetch(kind, url, *, data):
        captured.append(data["data"])
        return {"elements": [{"type": "way", "id": 1, "tags": {"custom:kind": 'value";out;'},
            "geometry": [{"lat": 18.515, "lon": 73.815}]}]}, "retrieval", False
    monkeypatch.setattr(world_data, "_cached_json", fetch)
    features, coverage = provider.extract("parcels", mapping(box(73.81, 18.51, 73.82, 18.52)),
        catalog_version=provider.revision)
    assert features == [] and coverage["omitted_geometry"] == 1
    assert '\\";out;' in captured[0]
    old = provider.revision
    config["layers"]["parcels"]["fields"].append("new")
    path.write_text(json.dumps(config))
    with pytest.raises(SourceError, match="changed"):
        provider.extract("parcels", mapping(box(73.81, 18.51, 73.82, 18.52)), catalog_version=old)
