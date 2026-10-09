"""General execution contracts tested with synthetic metre geometries only."""
from __future__ import annotations

from copy import deepcopy

import geopandas as gpd
import numpy as np
import pytest
from shapely import affinity
from shapely.geometry import LineString, box, mapping, shape

from terralens.engine import execute
from terralens.planner import QueryPlan


def geographic(geometry):
    moved = affinity.translate(geometry, xoff=500000, yoff=100000)
    return gpd.GeoSeries([moved], crs="EPSG:32631").to_crs("EPSG:4326").iloc[0]


def frame(records):
    return gpd.GeoDataFrame(records, geometry="geometry", crs="EPSG:4326")


class ExampleBundle:
    def __init__(self):
        self.manifest = {
            "dataset_version": "synthetic-general-v1", "analysis_crs": "EPSG:32631",
            "limits": {"maximum_distance_m": 500},
            "supported_concepts": {
                "solar_panels": {"prompt": "solar panels", "aliases": ["solar panels"], "association_layer": "buildings"},
                "swimming_pools": {"prompt": "swimming pools", "aliases": ["swimming pools"], "association_layer": None}},
            "vector_layers": {
                "buildings": {"fields": ["id", "building:use"], "extraction_date": "2026-01-01", "source": "synthetic-fixture"},
                "roads": {"fields": ["id", "surface"], "extraction_date": "2026-01-02"},
                "waterways": {"fields": ["id", "waterway"], "extraction_date": "2026-01-03"}}}
        self.layers = {
            "buildings": frame([
                {"id": "b1", "building:use": "commercial", "geometry": geographic(box(0, 0, 20, 20))},
                {"id": "outside", "building:use": "commercial", "geometry": geographic(box(450, 0, 470, 20))}]),
            "roads": frame([
                {"id": "paved", "surface": "asphalt", "geometry": geographic(LineString([(70, -50), (70, 50)]))},
                {"id": "unpaved", "surface": "unpaved", "geometry": geographic(LineString([(75, -50), (75, 50)]))}]),
            "waterways": frame([
                {"id": "stream", "waterway": "stream", "geometry": geographic(LineString([(110, -50), (110, 50)]))}])}
        self.tiles = [{"tile_id": "t1", "geometry": mapping(geographic(box(-100, -100, 200, 200))),
            "source_name": "synthetic.tif", "source_sha256": "synthetic-only", "width": 300, "height": 300,
            "acquisition_date": "2024-06-01"}]
        self.tile_index = {"t1": 0}
        self.embeddings = np.array([[1, 0]], dtype=np.float32)
        self.detections = [{"type": "Feature", "geometry": mapping(geographic(box(3, 3, 10, 10))),
            "properties": {"detection_id": "d1", "tile_id": "t1", "concept_id": "solar_panels",
                "score": 0.8, "model_id": "test-model", "pixel_box": [3, 3, 10, 10]}}]
        self.text_model_error = "checkpoint_not_configured"
        self.encoder_calls = 0

    def encode_text(self, query):
        self.encoder_calls += 1
        return None

    def provenance(self):
        return {"test_only": True, "vector_layers": self.manifest["vector_layers"]}


@pytest.fixture
def bundle():
    return ExampleBundle()


def plan(**changes):
    raw = {"query_type": "objects", "concept_id": "solar_panels", "visual_query": "solar panels",
        "reference_layer": None, "predicate": None, "distance_m": None, "attributes": []}
    raw.update(changes)
    return QueryPlan.model_validate(raw)


def relation(layer="roads", distance=200, **changes):
    raw = {"reference_layer": layer, "predicate": "within_distance", "distance_m": distance, "attributes": []}
    raw.update(changes)
    return raw


def filter_reference(field, value):
    return {"role": "reference", "field": field, "operator": "equals", "value": value}


def test_cached_visual_only_query_needs_no_spatial_relation(bundle):
    result = execute(bundle, plan())
    assert result["total_matches"] == 1
    assert result["features"][0]["properties"]["association"] == "unique"
    assert not result["context_features"]
    assert [item["condition"] for item in result["features"][0]["properties"]["conditions"]] == ["visual_detection", "feature_association"]


def test_visual_only_unassociated_concept_preserves_detection(bundle):
    bundle.detections[0]["properties"]["concept_id"] = "swimming_pools"
    result = execute(bundle, plan(concept_id="swimming_pools", visual_query="swimming pools"))
    assert result["total_matches"] == 1
    assert result["features"][0]["properties"]["geometry_role"] == "visual_detection_envelope"
    assert shape(result["features"][0]["geometry"]).equals(shape(bundle.detections[0]["geometry"]))


def test_multiple_relations_keep_independent_witnesses_and_all_reference_context(bundle):
    query = plan(reference_layer="roads", predicate="within_distance", distance_m=200,
        attributes=[filter_reference("surface", "asphalt")],
        relations=[relation(attributes=[filter_reference("surface", "unpaved")]), relation("waterways", 100)])
    result = execute(bundle, query, all_results=True)
    assert result["total_matches"] == 1
    clauses = [item for item in result["features"][0]["properties"]["conditions"] if "reference_feature_ids" in item]
    assert [item["reference_feature_ids"] for item in clauses] == [["paved"], ["unpaved"], ["stream"]]
    assert [item["relation_index"] for item in clauses] == [0, 1, 2]
    assert {feature["id"] for feature in result["context_features"]} == {"roads:paved", "roads:unpaved", "waterways:stream"}


def test_additional_known_failure_dominates_primary_missing_attribute(bundle):
    bundle.layers["roads"]["surface"] = None
    query = plan(reference_layer="roads", predicate="within_distance", distance_m=200,
        attributes=[filter_reference("surface", "asphalt")], relations=[relation("waterways", 10)])
    result = execute(bundle, query)
    assert result["total_matches"] == result["total_uncertain"] == 0
    assert result["objects_excluded"] == 1
    assert not result["context_features"]


def test_additional_unknown_preserves_subject_and_witness_identity(bundle):
    bundle.layers["waterways"]["waterway"] = None
    query = plan(reference_layer="roads", predicate="within_distance", distance_m=200,
        relations=[relation("waterways", 100, attributes=[filter_reference("waterway", "stream")])])
    result = execute(bundle, query)
    assert result["total_matches"] == 0
    candidate = result["uncertain_candidates"][0]
    assert candidate["id"] == "buildings:b1"
    clause = next(item for item in candidate["properties"]["conditions"] if item.get("relation_index") == 1)
    assert clause["status"] == "unknown"
    assert clause["reference_feature_ids"] == ["stream"]
    assert shape(candidate["geometry"]).equals(bundle.layers["buildings"].iloc[0].geometry)


def test_vector_subjects_skip_encoder_exclude_outside_coverage_and_keep_source_dates(bundle):
    query = plan(query_type="vectors", concept_id=None, visual_query="buildings", subject_layer="buildings")
    result = execute(bundle, query)
    assert bundle.encoder_calls == 0
    assert result["result_kind"] == "vector_features"
    assert result["text_encoder_status"] == "not_required_for_vector_query"
    assert [feature["id"] for feature in result["features"]] == ["buildings:b1"]
    props = result["features"][0]["properties"]
    assert props["source_attributes"]["building:use"] == "commercial"
    assert props["vector_extraction_date"] == "2026-01-01"
    assert props["imagery_coverage_dates"] == ["2024-06-01"]
    assert props["observations"] == []
    assert all(item["condition"] != "visual_detection" for item in props["conditions"])


def test_vector_subject_spatial_queries_retain_references_outside_image_coverage(bundle):
    bundle.layers["roads"] = frame([
        {"id": "halo-road", "surface": "asphalt", "geometry": geographic(LineString([(370, -50), (370, 50)]))}])
    query = plan(query_type="vectors", concept_id=None, visual_query="buildings", subject_layer="buildings",
        reference_layer="roads", predicate="within_distance", distance_m=400)
    result = execute(bundle, query)
    assert result["total_matches"] == 1
    assert result["context_features"][0]["id"] == "roads:halo-road"
    assert result["features"][0]["properties"]["conditions"][-1]["distance_m"] == pytest.approx(350, abs=0.01)


def test_vector_subject_attributes_and_spatial_intersections_are_not_visual_claims(bundle):
    bundle.layers["roads"] = frame([
        {"id": "crossing", "surface": "asphalt", "geometry": geographic(LineString([(-10, 10), (30, 10)]))}])
    query = plan(query_type="vectors", concept_id=None, visual_query="buildings", subject_layer="buildings",
        reference_layer="roads", predicate="intersects", attributes=[
            {"role": "subject", "field": "building:use", "operator": "equals", "value": "commercial"}])
    result = execute(bundle, query)
    assert result["total_matches"] == 1
    assert result["features"][0]["properties"]["conditions"][-1]["reference_feature_ids"] == ["crossing"]


def test_empty_reference_layer_is_unknown_instead_of_proof_of_absence(bundle):
    bundle.layers["waterways"] = bundle.layers["waterways"].iloc[0:0]
    query = plan(query_type="vectors", concept_id=None, visual_query="buildings", subject_layer="buildings",
        reference_layer="waterways", predicate="within_distance", distance_m=200)
    result = execute(bundle, query)
    assert result["total_matches"] == 0
    assert result["total_uncertain"] == 1
    assert "absence is not established" in result["uncertain_candidates"][0]["properties"]["conditions"][-1]["note"]


def test_vector_self_join_excludes_subject_identity(bundle):
    query = plan(query_type="vectors", concept_id=None, visual_query="buildings", subject_layer="buildings",
        reference_layer="buildings", predicate="within_distance", distance_m=100)
    result = execute(bundle, query)
    assert result["total_matches"] == result["total_uncertain"] == 0
    bundle.layers["buildings"] = frame([
        {"id": "b1", "building:use": "commercial", "geometry": geographic(box(0, 0, 20, 20))},
        {"id": "neighbor", "building:use": "commercial", "geometry": geographic(box(50, 0, 70, 20))}])
    result = execute(bundle, query)
    assert result["total_matches"] == 2
    witnesses = {feature["properties"]["id"]: feature["properties"]["conditions"][-1]["reference_feature_ids"]
        for feature in result["features"]}
    assert witnesses == {"b1": ["neighbor"], "neighbor": ["b1"]}


def test_partial_reference_labels_cannot_establish_negative_relation(bundle):
    bundle.manifest["vector_layers"]["buildings"]["coverage_status"] = "partial_training_labels"
    query = plan(query_type="vectors", concept_id=None, visual_query="buildings", subject_layer="buildings",
        reference_layer="buildings", predicate="within_distance", distance_m=100)
    result = execute(bundle, query)
    assert result["total_matches"] == 0
    assert result["total_uncertain"] == 1
    condition = result["uncertain_candidates"][0]["properties"]["conditions"][-1]
    assert condition["reference_coverage_status"] == "partial_training_labels"
    assert "nonmatch does not establish absence" in condition["note"]


def test_partial_reference_metadata_nested_by_preparation_is_respected(bundle):
    bundle.manifest["vector_layers"]["buildings"]["metadata"] = {
        "coverage_status": "partial_training_labels", "reference_coverage_complete": False}
    query = plan(query_type="vectors", concept_id=None, visual_query="buildings", subject_layer="buildings",
        reference_layer="buildings", predicate="within_distance", distance_m=100)
    result = execute(bundle, query)
    assert result["total_matches"] == 0
    assert result["total_uncertain"] == 1
    assert result["uncertain_candidates"][0]["properties"]["conditions"][-1]["reference_coverage_status"] == "partial_training_labels"


def test_ground_mode_accepts_ephemeral_null_concept_and_arbitrary_prompt(bundle):
    bundle.manifest["supported_concepts"][None] = {"prompt": "blue rectangular backyard pools", "association_layer": None}
    bundle.detections[0]["properties"]["concept_id"] = None
    result = execute(bundle, plan(concept_id=None, visual_query="blue rectangular backyard pools"), mode="ground")
    assert result["total_matches"] == 1
    assert result["features"][0]["properties"]["model_prompt"] == "blue rectangular backyard pools"


def duplicated_pool_observations(bundle, *, second_date="2024-06-01", source_hash="synthetic-only", same_tile=False):
    bundle.tiles[0]["source_sha256"] = source_hash
    bundle.detections[0]["properties"]["concept_id"] = "swimming_pools"
    duplicate = deepcopy(bundle.detections[0])
    duplicate["properties"]["detection_id"] = "d2"
    if not same_tile:
        tile = deepcopy(bundle.tiles[0])
        tile.update(tile_id="t2", acquisition_date=second_date)
        bundle.tiles.append(tile)
        bundle.tile_index["t2"] = 1
        bundle.embeddings = np.array([[1, 0], [1, 0]], dtype=np.float32)
        duplicate["properties"]["tile_id"] = "t2"
    else:
        bundle.tiles[0]["acquisition_date"] = second_date
    bundle.detections.append(duplicate)


@pytest.mark.parametrize("second_date,source_hash", [
    ("2025-06-01", "synthetic-only"),
    ("2024-06-01", None),
    ("2024-06-01", ""),
])
def test_unassociated_observations_do_not_merge_different_dates_or_missing_source_identity(bundle, second_date, source_hash):
    duplicated_pool_observations(bundle, second_date=second_date, source_hash=source_hash)
    result = execute(bundle, plan(concept_id="swimming_pools", visual_query="swimming pools"))
    assert result["total_matches"] == 2
    assert result["duplicate_observations_collapsed"] == 0


def test_unassociated_same_source_same_known_epoch_observations_merge(bundle):
    duplicated_pool_observations(bundle)
    result = execute(bundle, plan(concept_id="swimming_pools", visual_query="swimming pools"))
    assert result["total_matches"] == 1
    assert result["duplicate_observations_collapsed"] == 1
    assert len(result["features"][0]["properties"]["observations"]) == 2


@pytest.mark.parametrize("same_tile,expected", [(False, 2), (True, 1)])
def test_unknown_epoch_unassociated_observations_require_same_tile(bundle, same_tile, expected):
    bundle.tiles[0]["acquisition_date"] = None
    duplicated_pool_observations(bundle, second_date=None, same_tile=same_tile)
    result = execute(bundle, plan(concept_id="swimming_pools", visual_query="swimming pools"))
    assert result["total_matches"] == expected


def test_uniquely_identified_vector_object_keeps_observations_across_dates(bundle):
    duplicated_pool_observations(bundle, second_date="2025-06-01")
    for detection in bundle.detections:
        detection["properties"]["concept_id"] = "solar_panels"
    result = execute(bundle, plan())
    assert result["total_matches"] == 1
    dates = {observation["imagery_date"] for observation in result["features"][0]["properties"]["observations"]}
    assert dates == {"2024-06-01", "2025-06-01"}
