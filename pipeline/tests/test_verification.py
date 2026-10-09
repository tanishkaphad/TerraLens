"""Evidence and geometric verification with synthetic projected fixtures."""
from __future__ import annotations

from copy import deepcopy

import geopandas as gpd
import pytest
from shapely import affinity
from shapely.geometry import LineString, box, mapping

from terralens.verification import apply_verification, evidence_receipt, verify_result


def geographic(geometry):
    shifted = affinity.translate(geometry, xoff=500000, yoff=100000)
    return mapping(gpd.GeoSeries([shifted], crs="EPSG:32631").to_crs("EPSG:4326").iloc[0])


def result_fixture(*, subject=None, reference=None, radius=30, predicate="within_distance", vector=False):
    subject = subject if subject is not None else box(0, 0, 100, 100)
    reference = reference if reference is not None else LineString([(0, -100), (0, 200)])
    observation = {"condition": "vector_source" if vector else "visual_detection", "status": "pass",
        "source": "synthetic-only"} if vector else {"condition": "visual_detection", "status": "pass",
        "score": 0.8, "checkpoint": "fixture-owl", "model_revision": "fixed-revision"}
    relation = {"condition": predicate, "status": "pass", "relation_index": 0, "reference_layer": "roads",
        "radius_m": radius if predicate == "within_distance" else None, "reference_feature_ids": ["r1"],
        "reference_candidates": [{"id": "r1", "status": "pass", "attributes": []}]}
    properties = {"id": "s1", "role": "match", "geometry_role": "source_vector_feature" if vector else "visual_detection_envelope",
        "association": "unique" if vector else "not_required", "subject_layer": "buildings" if vector else None,
        "source_sha256": "fixture-raster-sha", "source_image": "fixture.tif", "tile_ids": ["tile1"],
        "model_id": None if vector else "fixture-owl", "model_revision": None if vector else "fixed-revision",
        "model_prompt": None if vector else "swimming pools", "conditions": [observation, relation]}
    return {"plan": {"visual_query": "swimming pools", "reference_layer": "roads", "distance_m": radius,
        "predicate": predicate}, "dataset_version": "fixture-v1", "result_kind": "vector_features" if vector else "grounded_objects",
        "mode": "catalog_scan", "analysis_crs": "EPSG:32631", "page": 0, "page_size": 50,
        "total_matches": 1, "total_uncertain": 0, "objects_excluded": 0, "truncated": False, "uncertain_truncated": False,
        "features": [{"type": "Feature", "id": "subject:s1", "geometry": geographic(subject), "properties": properties}],
        "uncertain_candidates": [], "tile_candidates": [], "context_features": [{"type": "Feature", "id": "roads:r1",
            "geometry": geographic(reference), "properties": {"id": "r1", "layer": "roads", "role": "reference"}}],
        "provenance": {"source_records": [{"sha256": "fixture-raster-sha", "source": "synthetic-only"}],
            "vector_layers": {"roads": {"source": "synthetic-roads", "extraction_date": "2026-01-01"}}}}


def clause(audit):
    return audit["features"][0]["conditions"][-1]


def witness(audit):
    return clause(audit)["verification"]["witnesses"][0]


def test_envelope_touching_road_does_not_prove_object_proximity():
    result = result_fixture(radius=30)
    audit = verify_result(result)
    assert audit["features"][0]["status"] == "unknown"
    assert witness(audit)["distance_interval_m"][0] == pytest.approx(0, abs=1e-6)
    assert witness(audit)["distance_interval_m"][1] > 100
    assert "does not establish object proximity" in witness(audit)["reason"]
    adjusted = apply_verification(result)
    assert adjusted["total_matches"] == 0
    assert adjusted["total_uncertain"] == 1
    assert adjusted["uncertain_candidates"][0]["id"] == "subject:s1"


def test_envelope_intersecting_road_does_not_prove_object_intersection():
    audit = verify_result(result_fixture(predicate="intersects"))
    assert clause(audit)["status"] == "unknown"
    assert "Envelope intersection" in witness(audit)["reason"]


def test_small_envelope_wholly_within_distance_is_supported_conditionally():
    result = result_fixture(subject=box(-5, -5, 5, 5), reference=LineString([(20, -100), (20, 100)]), radius=40)
    audit = verify_result(result)
    assert audit["features"][0]["status"] == "pass"
    interval = witness(audit)["distance_interval_m"]
    assert interval[0] == pytest.approx(15, abs=0.01)
    assert 27 < interval[1] < 28
    assert witness(audit)["sensitivity"]["additional_error_before_geometric_status_change_m"] > 12
    assert "contained within" in clause(audit)["verification"]["assumption"]


def test_registration_allowance_widens_interval_and_can_remove_support():
    result = result_fixture(subject=box(-5, -5, 5, 5), reference=LineString([(20, -100), (20, 100)]), radius=30)
    assert clause(verify_result(result))["status"] == "pass"
    audit = verify_result(result, registration_error_m=5)
    assert clause(audit)["status"] == "unknown"
    item = witness(audit)
    assert item["registration_adjusted_interval_m"][0] == pytest.approx(10, abs=0.01)
    assert item["registration_adjusted_interval_m"][1] > 32
    assert audit["registration_status"] == "supplied_assumption"


def test_unknown_registration_is_recorded_without_invented_error():
    audit = verify_result(result_fixture())
    assert audit["registration_error_m"] is None
    assert audit["registration_status"] == "not_estimated"
    assert witness(audit)["registration_adjusted_interval_m"] is None
    assert audit["calibration"]["status"] == "uncalibrated"
    assert "not probabilities" in audit["calibration"]["score_meaning"]


@pytest.mark.parametrize("error", [-1, True, float("nan"), float("inf"), "5"])
def test_invalid_registration_allowance_is_rejected(error):
    with pytest.raises(ValueError, match="finite, nonnegative"):
        verify_result(result_fixture(), registration_error_m=error)


def test_minimum_distance_beyond_radius_fails_for_complete_reference_evidence():
    result = result_fixture(subject=box(0, 0, 10, 10), reference=LineString([(100, -100), (100, 100)]), radius=30)
    audit = verify_result(result)
    assert clause(audit)["status"] == "fail"
    adjusted = apply_verification(result)
    assert adjusted["total_matches"] == adjusted["total_uncertain"] == 0
    assert adjusted["objects_excluded"] == 1
    assert adjusted["verification_excluded"][0]["id"] == "subject:s1"


def test_partial_reference_coverage_preserves_nonmatch_as_unknown():
    result = result_fixture(subject=box(0, 0, 10, 10), reference=LineString([(100, -100), (100, 100)]), radius=30)
    result["provenance"]["vector_layers"]["roads"]["metadata"] = {
        "coverage_status": "partial_training_labels", "reference_coverage_complete": False}
    audit = verify_result(result)
    assert witness(audit)["status"] == "fail"
    assert clause(audit)["status"] == "unknown"
    assert clause(audit)["verification"]["partial_reference_coverage"] is True


def test_known_positive_witness_can_pass_with_partial_reference_coverage():
    result = result_fixture(subject=box(0, 0, 1, 1), reference=LineString([(5, -100), (5, 100)]), radius=30)
    result["provenance"]["vector_layers"]["roads"]["metadata"] = {"reference_coverage_complete": False}
    audit = verify_result(result)
    assert clause(audit)["status"] == "pass"
    assert clause(audit)["verification"]["partial_reference_coverage"] is True


def test_missing_reference_geometry_cannot_validate_original_passing_condition():
    result = result_fixture()
    result["context_features"] = []
    audit = verify_result(result)
    assert clause(audit)["status"] == "unknown"
    assert clause(audit)["verification"]["missing_reference_geometry_ids"] == ["r1"]


def test_missing_or_nonmetric_analysis_crs_is_unknown():
    result = result_fixture()
    result["analysis_crs"] = "EPSG:4326"
    assert clause(verify_result(result))["status"] == "unknown"
    result.pop("analysis_crs")
    assert clause(verify_result(result))["status"] == "unknown"


def test_reference_attribute_unknown_is_not_repaired_by_nearby_geometry():
    result = result_fixture(subject=box(0, 0, 1, 1), reference=LineString([(5, -100), (5, 100)]), radius=30)
    result["features"][0]["properties"]["conditions"][-1]["reference_candidates"][0]["attributes"] = [
        {"condition": "reference_attribute", "field": "surface", "expected": "unpaved", "observed": None, "status": "unknown"}]
    audit = verify_result(result)
    assert witness(audit)["geometric_status"] == "pass"
    assert witness(audit)["status"] == clause(audit)["status"] == "unknown"


def test_each_reference_has_independent_interval_and_any_supported_witness_can_pass():
    result = result_fixture(subject=box(0, 0, 1, 1), reference=LineString([(100, -100), (100, 100)]), radius=30)
    relation = result["features"][0]["properties"]["conditions"][-1]
    relation["reference_feature_ids"].append("r2")
    relation["reference_candidates"].append({"id": "r2", "status": "pass", "attributes": []})
    result["context_features"].append({"type": "Feature", "id": "roads:r2", "geometry": geographic(LineString([(5, -100), (5, 100)])),
        "properties": {"id": "r2", "layer": "roads", "role": "reference"}})
    audit = verify_result(result)
    assert clause(audit)["status"] == "pass"
    assert clause(audit)["verification"]["supported_reference_feature_ids"] == ["r2"]
    assert [item["status"] for item in clause(audit)["verification"]["witnesses"]] == ["fail", "pass"]


def test_failure_dominates_other_unresolved_clauses():
    result = result_fixture(subject=box(0, 0, 10, 10), reference=LineString([(100, -100), (100, 100)]), radius=30)
    result["features"][0]["properties"]["conditions"].insert(1, {"condition": "subject_attribute", "status": "unknown"})
    audit = verify_result(result)
    assert audit["features"][0]["status"] == "fail"


def test_annotation_evidence_is_explicit_and_never_becomes_model_observation():
    result = result_fixture(subject=box(0, 0, 1, 1), reference=LineString([(5, -100), (5, 100)]), radius=30, vector=True)
    result["features"][0]["properties"]["source_attributes"] = {"evidence_type": "dataset_annotation", "source_image_id": "img1"}
    audit = verify_result(result)
    assert audit["features"][0]["evidence_kinds"] == ["dataset_annotation"]
    assert "model_observation" not in audit["features"][0]["evidence_kinds"]
    assert audit["features"][0]["model_id"] is None


def test_receipt_is_deterministic_and_covers_geometry_conditions_source_model_and_prompt():
    result = result_fixture()
    receipt = evidence_receipt(result)
    assert receipt == evidence_receipt(deepcopy(result))
    changes = []
    for key, value in (("source_sha256", "changed"), ("model_id", "changed"), ("model_prompt", "changed")):
        changed = deepcopy(result)
        changed["features"][0]["properties"][key] = value
        changes.append(changed)
    changed = deepcopy(result)
    changed["features"][0]["geometry"] = geographic(box(0, 0, 101, 100))
    changes.append(changed)
    changed = deepcopy(result)
    changed["features"][0]["properties"]["conditions"][-1]["radius_m"] = 31
    changes.append(changed)
    changed = deepcopy(result)
    changed["context_features"][0]["geometry"] = geographic(LineString([(1, -100), (1, 200)]))
    changes.append(changed)
    assert all(evidence_receipt(changed)["sha256"] != receipt["sha256"] for changed in changes)
    reordered = deepcopy(result)
    reordered["timings_ms"] = {"total": 999}
    assert evidence_receipt(reordered) == receipt


def test_verification_and_adjustment_never_mutate_input():
    result = result_fixture()
    before = deepcopy(result)
    verify_result(result)
    adjusted = apply_verification(result)
    assert result == before
    adjusted["uncertain_candidates"][0]["properties"]["conditions"][0]["score"] = 0
    assert result == before


def test_paginated_aggregate_counts_are_preserved_for_unseen_rows():
    result = result_fixture()
    result.update(total_matches=10, truncated=True)
    adjusted = apply_verification(result)
    assert adjusted["total_matches"] == 10
    assert adjusted["total_uncertain"] == 0
    assert len(adjusted["features"]) == 0
    assert len(adjusted["uncertain_candidates"]) == 1
    assert adjusted["verification"]["scope"] == "returned_page_only"
    assert adjusted["verification"]["counts"] == {"pass": 0, "unknown": 1, "fail": 0}
    assert adjusted["verification_totals_status"] == "original_aggregate_totals_preserved_for_unseen_pages"
