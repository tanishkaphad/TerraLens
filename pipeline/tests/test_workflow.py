"""Offline workflow execution tests using explicit synthetic metre geometries."""
from copy import deepcopy

import geopandas as gpd
import pytest
from pydantic import ValidationError
from shapely import affinity
from shapely.geometry import LineString, box, mapping

from terralens.workflow import (Workflow, WorkflowError, aggregate_result, execute_workflow,
    guard_numeric_conditions, spatial_filter, validate_workflow)


def geographic(geometry):
    shifted = affinity.translate(geometry, xoff=500000, yoff=100000)
    return mapping(gpd.GeoSeries([shifted], crs="EPSG:32631").to_crs("EPSG:4326").iloc[0])


def feature(identifier, geometry, *, envelope=False, uncertain=False, attributes=None):
    return {"type": "Feature", "id": identifier, "geometry": geographic(geometry), "properties": {
        "id": identifier, "geometry_role": "visual_detection_envelope" if envelope else "source_vector_feature",
        "association": "not_required" if envelope else "unique", "subject_layer": None if envelope else "buildings",
        "role": "uncertain_candidate" if uncertain else "match", "subject_attributes": attributes or {},
        "conditions": [{"condition": "visual_detection" if envelope else "vector_source", "status": "pass"},
            *([{"condition": "subject_attribute", "status": "unknown"}] if uncertain else [])]}}


def result(records, uncertain=None, *, version="v1", coverage=None):
    return {"dataset_version": version, "result_kind": "grounded_objects", "analysis_crs": "EPSG:32631",
        "plan": {"subject_layer": "buildings"}, "features": records, "uncertain_candidates": uncertain or [],
        "tile_candidates": [], "context_features": [], "total_matches": len(records), "total_uncertain": len(uncertain or []),
        "objects_excluded": 0, "truncated": False, "uncertain_truncated": False,
        "coverage": {"bbox_wgs84": coverage or list(gpd.GeoSeries([affinity.translate(box(-100, -100, 500, 500), xoff=500000, yoff=100000)], crs="EPSG:32631").to_crs("EPSG:4326").total_bounds)},
        "provenance": {"vector_layers": {"buildings": {"fields": ["use"], "source": "synthetic"}}}, "claim_scope": "Synthetic local sample only"}


@pytest.fixture
def catalog():
    coverage = result([])["coverage"]
    manifest = {"dataset_version": "v1", "coverage": coverage, "limits": {"maximum_distance_m": 500},
        "supported_concepts": {}, "vector_layers": {"buildings": {"fields": ["use"]}}}
    return {"sample": {"manifest": manifest, "live_grounding": True, "retrieval": True}}


def search_step(identifier="objects", *, dataset="sample", **changes):
    step = {"id": identifier, "operation": "search", "label": identifier, "dataset_id": dataset, "mode": "catalog_scan",
        "plan": {"query_type": "vectors", "concept_id": None, "subject_layer": "buildings", "visual_query": "buildings",
            "reference_layer": None, "predicate": None, "distance_m": None, "attributes": [], "relations": []}}
    step.update(changes)
    return step


def workflow(steps):
    return Workflow.model_validate({"title": "Synthetic workflow", "status": "ready", "steps": steps,
        "requirements": [{"text": "Find and summarize supplied records", "status": "planned", "step_ids": [step["id"] for step in steps]}]})


@pytest.mark.parametrize("operation,extra", [
    ("search", {"metric": "count"}), ("search", {"distance_m": 20}),
    ("coverage", {"mode": "catalog_scan"}), ("coverage", {"bbox": [0, 0, 1, 1]}),
    ("aggregate", {"dataset_id": "sample"}), ("spatial_filter", {"group_by": "use"})])
def test_operation_specific_fields_cannot_be_silently_ignored(catalog, operation, extra):
    if operation == "search":
        steps = [search_step(**extra)]
    elif operation == "coverage":
        steps = [{"id": "inventory", "operation": operation, "label": "inventory", "dataset_id": "sample", **extra}]
    else:
        steps = [search_step("first"), search_step("second"), {"id": "analysis", "operation": operation,
            "label": "analysis", "input_id": "first", **({"metric": "count"} if operation == "aggregate" else {
                "reference_id": "second", "predicate": "within_distance", "distance_m": 20}), **extra}]
    with pytest.raises(WorkflowError, match="must be null"):
        validate_workflow(workflow(steps), catalog)


def test_partial_requested_bbox_is_rejected_instead_of_silently_clipped(catalog):
    area = catalog["sample"]["manifest"]["coverage"]["bbox_wgs84"]
    partial = [area[0] - 1, area[1], area[2], area[3]]
    with pytest.raises(WorkflowError, match="complete requested area"):
        validate_workflow(workflow([search_step(bbox=partial)]), catalog)


@pytest.mark.parametrize("field,value", [("distance_m", True), ("distance_m", float("nan")), ("bbox", [0, True, 1, 2])])
def test_boolean_nonfinite_and_nonliteral_coordinates_are_rejected(field, value):
    with pytest.raises(ValidationError):
        workflow([search_step(**{field: value})])


def test_detection_and_dataset_distance_budgets_are_enforced(catalog):
    ground_plan = {"query_type": "objects", "concept_id": None, "visual_query": "trees", "reference_layer": None,
        "predicate": None, "distance_m": None, "subject_layer": None, "attributes": []}
    with pytest.raises(WorkflowError, match="six-tile"):
        validate_workflow(workflow([search_step("first", mode="ground", plan=ground_plan, tile_limit=4),
            search_step("second", mode="ground", plan=ground_plan, tile_limit=4)]), catalog)
    with pytest.raises(WorkflowError, match="distance limit"):
        validate_workflow(workflow([search_step("first"), search_step("second"), {"id": "join", "operation": "spatial_filter",
            "label": "join", "input_id": "first", "reference_id": "second", "predicate": "within_distance", "distance_m": 600}]), catalog)


def test_wrong_city_cross_dataset_join_is_rejected(catalog):
    catalog["other"] = deepcopy(catalog["sample"])
    catalog["other"]["manifest"]["coverage"]["bbox_wgs84"] = [77, 18, 78, 19]
    with pytest.raises(WorkflowError, match="different areas"):
        validate_workflow(workflow([search_step("first"), search_step("second", dataset="other"),
            {"id": "join", "operation": "spatial_filter", "label": "join", "input_id": "first", "reference_id": "second",
                "predicate": "within_distance", "distance_m": 100}]), catalog)


def test_numeric_guard_preserves_unit_conversion_and_duplicate_clauses():
    planned = workflow([search_step("first"), search_step("second"), {"id": "join", "operation": "spatial_filter",
        "label": "join", "input_id": "first", "reference_id": "second", "predicate": "within_distance", "distance_m": 30.48}])
    guard_numeric_conditions("Find buildings within100 feet of roofs", planned)
    with pytest.raises(WorkflowError, match="reused"):
        guard_numeric_conditions("Buildings within100feet of roofs and within100feet of roads", planned)
    with pytest.raises(WorkflowError, match="positive"):
        guard_numeric_conditions("Find buildings within-1m of roads", planned)
    guard_numeric_conditions("Report 100m² of source roof area", workflow([search_step()]))


def test_exact_vector_join_uses_resolvable_ids_and_excludes_self_identity():
    subjects = result([feature("building:b1", box(0, 0, 10, 10))])
    refs = result([feature("building:b1", box(0, 0, 10, 10)), feature("building:b2", box(20, 0, 30, 10))])
    joined = spatial_filter(subjects, refs, "within_distance", 15)
    assert joined["total_matches"] == 1
    condition = joined["features"][0]["properties"]["conditions"][-1]
    assert [candidate["original_feature_id"] for candidate in condition["reference_candidates"]] == ["building:b2"]
    assert condition["verification"]["missing_reference_geometry_ids"] == []
    assert condition["reference_candidates"][0]["distance_interval_m"] == pytest.approx([10, 10], abs=0.01)


def test_both_detection_boxes_need_distance_interval_not_envelope_touch():
    subjects = result([feature("pool:p1", box(0, 0, 100, 100), envelope=True)])
    refs = result([feature("tree:t1", box(95, 0, 195, 100), envelope=True)])
    joined = spatial_filter(subjects, refs, "within_distance", 10)
    assert joined["total_matches"] == 0
    assert joined["total_uncertain"] == 1
    candidate = joined["uncertain_candidates"][0]["properties"]["conditions"][-1]["reference_candidates"][0]
    assert candidate["distance_interval_m"][0] == pytest.approx(0, abs=0.001)
    assert candidate["distance_interval_m"][1] > 230


def test_small_subject_and_reference_detection_boxes_can_support_large_radius():
    subjects = result([feature("pool:p1", box(0, 0, 1, 1), envelope=True)])
    refs = result([feature("tree:t1", box(5, 0, 6, 1), envelope=True)])
    joined = spatial_filter(subjects, refs, "within_distance", 20)
    assert joined["total_matches"] == 1


@pytest.mark.parametrize("subject_unknown,reference_unknown", [(True, False), (False, True), (True, True)])
def test_spatial_join_retains_upstream_unknown_for_either_source(subject_unknown, reference_unknown):
    subject = feature("b1", box(0, 0, 10, 10), uncertain=subject_unknown)
    reference = feature("b2", box(20, 0, 30, 10), uncertain=reference_unknown)
    subjects = result([] if subject_unknown else [subject], [subject] if subject_unknown else [])
    refs = result([] if reference_unknown else [reference], [reference] if reference_unknown else [])
    joined = spatial_filter(subjects, refs, "within_distance", 20)
    assert joined["total_matches"] == 0
    assert joined["total_uncertain"] == 1


def test_empty_reference_set_is_unknown_and_does_not_mutate_inputs():
    subjects = result([feature("b1", box(0, 0, 10, 10))])
    refs = result([])
    before = deepcopy(subjects)
    joined = spatial_filter(subjects, refs, "within_distance", 20)
    assert joined["total_uncertain"] == 1
    assert subjects == before


def test_spatial_pair_budget_stops_dense_joins_before_expanding_context(monkeypatch):
    from terralens import workflow as workflow_module
    monkeypatch.setattr(workflow_module, "MAX_SPATIAL_PAIRS", 1)
    subjects = result([feature("b1", box(0, 0, 10, 10))])
    refs = result([feature("b2", box(20, 0, 30, 10)), feature("b3", box(30, 0, 40, 10))])
    with pytest.raises(WorkflowError, match="candidate-pair budget"):
        spatial_filter(subjects, refs, "within_distance", 100)


def test_aggregation_uses_only_supported_source_geometries_and_scalar_groups():
    records = result([feature("b1", box(0, 0, 10, 20), attributes={"use": "commercial"})],
        [feature("b2", box(20, 0, 30, 20), uncertain=True)])
    area = aggregate_result(records, "area_m2", "use")
    assert area["rows"] == [{"group": "commercial", "area_m2": pytest.approx(200, abs=0.01)}]
    assert area["uncertain_records_excluded"] == 1
    records["features"][0]["properties"]["subject_attributes"]["use"] = {"nested": "annotation"}
    with pytest.raises(WorkflowError, match="scalar"):
        aggregate_result(records, "count", "use")


@pytest.mark.parametrize("metric,envelope,geometry", [
    ("area_m2", True, box(0, 0, 1, 1)), ("length_m", False, box(0, 0, 1, 1)),
    ("area_m2", False, LineString([(0, 0), (1, 1)]))])
def test_aggregation_rejects_model_box_areas_and_wrong_geometry_metrics(metric, envelope, geometry):
    with pytest.raises(WorkflowError):
        aggregate_result(result([feature("a", geometry, envelope=envelope)]), metric, None)


def test_aggregate_requires_complete_counts_and_metric_crs():
    records = result([feature("a", box(0, 0, 1, 1))])
    records["total_matches"] = 2
    with pytest.raises(WorkflowError, match="complete"):
        aggregate_result(records, "count", None)
    records["total_matches"] = 1
    records["analysis_crs"] = "EPSG:4326"
    with pytest.raises(WorkflowError, match="metre"):
        aggregate_result(records, "area_m2", None)


def test_workflow_end_to_end_outputs_actual_requirement_evidence_and_timing_independent_receipt(catalog):
    steps = [search_step("first"), search_step("second"), {"id": "join", "operation": "spatial_filter", "label": "Join source buildings",
        "input_id": "first", "reference_id": "second", "predicate": "within_distance", "distance_m": 20},
        {"id": "summary", "operation": "aggregate", "label": "Count by use", "input_id": "join", "metric": "count", "group_by": "use"}]
    first = result([feature("b1", box(0, 0, 10, 10), attributes={"use": "commercial"})])
    second = result([feature("b2", box(20, 0, 30, 10))])
    def search(step):
        returned = deepcopy(first if step.id == "first" else second)
        returned["timings_ms"] = {"total": 15}
        returned["grounding"] = {"inference_ms": 20, "model_id": "fixture"}
        return returned
    executed = execute_workflow(workflow(steps), catalog, search)
    assert executed["artifacts"][-1]["rows"] == [{"group": "commercial", "count": 1}]
    requirement = executed["requirement_evidence"][0]
    assert requirement["status"] == "supported_by_executed_steps"
    assert requirement["step_evidence"][2]["supported_records"] == 1
    assert requirement["step_evidence"][2]["conditions"]
    def other_timings(step):
        returned = search(step)
        returned["timings_ms"]["total"] = 999
        returned["grounding"]["inference_ms"] = 888
        returned["grounding"]["cache_hit"] = True
        returned["grounding"]["expires_in_seconds"] = 30
        return returned
    repeated = execute_workflow(workflow(steps), catalog, other_timings)
    assert executed["receipt"]["sha256"] == repeated["receipt"]["sha256"]
    assert "timings_ms" not in first


def test_zero_matches_requirement_never_claims_municipal_absence(catalog):
    executed = execute_workflow(workflow([search_step()]), catalog, lambda step: result([]))
    evidence = executed["requirement_evidence"][0]
    assert evidence["status"] == "no_supported_matches_in_sample"
    assert "does not establish municipal absence" in evidence["step_evidence"][0]["note"]


def test_workflow_search_cannot_switch_dataset_version_or_outside_area(catalog):
    with pytest.raises(WorkflowError, match="different dataset"):
        execute_workflow(workflow([search_step()]), catalog, lambda step: result([], version="other"))
    outside = result([feature("outside", box(10000, 10000, 10010, 10010))])
    with pytest.raises(WorkflowError, match="outside"):
        execute_workflow(workflow([search_step()]), catalog, lambda step: outside)
