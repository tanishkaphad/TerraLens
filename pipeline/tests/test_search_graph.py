"""Behavioral evidence tests for Boolean joins, uncertainty and source identity."""
from __future__ import annotations

from copy import deepcopy

import pytest
from pyproj import Transformer
from shapely.geometry import Point, Polygon, box, mapping
from shapely.ops import transform

from terralens.search_graph import evaluate_expression, execute_graph, record_id
from terralens.search_plan import SearchPlanError


def feature(identifier, geometry, **properties):
    return {"type": "Feature", "id": identifier, "geometry": mapping(geometry), "properties": properties}


def leaf(identifier):
    return {"operator": "clause", "clause_id": identifier}


def plan(clauses=(), expression=None, entities=None):
    return {"entities": entities or [{"id": "asset", "description": "mapped assets", "kind": "mapped"},
        {"id": "reference", "description": "reference records", "kind": "mapped"}],
        "subject_entity_ids": ["asset"], "clauses": list(clauses), "expression": expression}


def attribute(identifier, field, value, entity="asset", **extra):
    return {"id": identifier, "kind": "attribute", "entity_id": entity, "field": field, "operator": "equals", "value": value, **extra}


def spatial(identifier="near", **extra):
    return {"id": identifier, "kind": "spatial", "entity_id": "asset", "reference_entity_id": "reference", "operator": "within_distance", "distance_m": 200, **extra}


@pytest.mark.parametrize(("operator", "states", "expected"), [
    ("AND", {"a": "fail", "b": "unknown"}, "fail"),
    ("AND", {"a": "pass", "b": "unknown"}, "unknown"),
    ("OR", {"a": "pass", "b": "unknown"}, "pass"),
    ("OR", {"a": "fail", "b": "unknown"}, "unknown"),
    ("OR", {"a": "fail", "b": "fail"}, "fail"),
])
def test_boolean_expression_preserves_unknowns_and_known_decisive_evidence(operator, states, expected):
    assert evaluate_expression({"operator": operator, "operands": [leaf("a"), leaf("b")]}, states) == expected
    assert evaluate_expression({"operator": "NOT", "operands": [leaf("a")]}, {"a": "unknown"}) == "unknown"


def test_missing_source_attribute_does_not_become_negative_match():
    query = plan([attribute("a", "material", "stone")], {"operator": "NOT", "operands": [leaf("a")]})
    records = execute_graph(query, {"asset": [feature("one", Point(0, 0))]})
    assert records[0]["group"] == "needs_review"
    assert records[0]["feature"]["properties"]["conditions"][0]["observed"] is None


def test_or_allows_known_match_and_retains_unknown_alternative():
    query = plan([attribute("a", "material", "stone"), attribute("b", "status", "inspected")], {"operator": "OR", "operands": [leaf("a"), leaf("b")]})
    records = execute_graph(query, {"asset": [feature("one", Point(0, 0), material="stone"), feature("two", Point(.001, 0), material="wood")]})
    assert {item["feature"]["properties"]["source_feature_id"]: item["group"] for item in records} == {"one": "matches", "two": "needs_review"}
    assert records[0]["feature"]["properties"]["conditions"][1]["status"] == "unknown"


def test_all_spatial_witnesses_are_measured_and_retained_beyond_subject_boundary():
    query = plan([spatial()], leaf("near"))
    query["scope"] = {"mode": "geometry", "geometry": mapping(box(-.0001, -.0001, .0001, .0001))}
    records = execute_graph(query, {"asset": [feature("one", Point(0, 0)), feature("outside_subject", Point(.01, 0))],
        "reference": [feature("left", Point(-.001, 0)), feature("right", Point(.001, 0)), feature("far", Point(.1, 0))]},
        coverage={"reference": {"reference_coverage_complete": True}})
    assert len(records) == 1
    condition = records[0]["feature"]["properties"]["conditions"][0]
    assert records[0]["group"] == "matches"
    assert set(condition["reference_feature_ids"]) == {"left", "right"}
    assert len(condition["reference_candidates"]) == 2
    assert condition["references_total"] == 3
    assert condition["definite_spatial_rejects_count"] == 1
    assert all(110 < witness["distance_m"] < 112 for witness in condition["witnesses"])
    assert all(witness["reference_feature"]["geometry"] for witness in condition["witnesses"])


def test_reference_attributes_apply_to_the_same_reference_that_satisfies_distance():
    clauses = [spatial(reference_clause_ids=["open"]), attribute("open", "covered", "no", entity="reference")]
    query = plan(clauses, leaf("near"))
    inputs = {"asset": [feature("one", Point(0, 0))], "reference": [feature("close_closed", Point(.0001, 0), covered="yes"), feature("far_open", Point(.01, 0), covered="no")]}
    records = execute_graph(query, inputs, coverage={"reference": {"enumeration_complete": True}})
    assert records[0]["group"] == "excluded"
    inputs["reference"].append(feature("close_missing_tag", Point(.0002, 0)))
    records = execute_graph(query, inputs, coverage={"reference": {"enumeration_complete": True}})
    assert records[0]["group"] == "needs_review"


def test_distinct_relationships_keep_each_independent_witness():
    entities = [{"id": "asset", "description": "test assets", "kind": "mapped"},
        {"id": "reference", "description": "inspection lines", "kind": "mapped"},
        {"id": "area", "description": "inspection zones", "kind": "mapped"}]
    query = plan([spatial(), spatial("touches", reference_entity_id="area", operator="intersects", distance_m=None)],
        {"operator": "AND", "operands": [leaf("near"), leaf("touches")]}, entities)
    records = execute_graph(query, {"asset": [feature("one", Point(0, 0))], "reference": [feature("line", Point(.001, 0))],
        "area": [feature("zone", box(-.001, -.001, .001, .001))]})
    assert records[0]["group"] == "matches"
    assert [condition["reference_feature_ids"] for condition in records[0]["feature"]["properties"]["conditions"]] == [["line"], ["zone"]]


def test_shared_binding_prevents_different_objects_from_satisfying_same_object_request():
    # One witness is west, another east; no single witness satisfies both.
    clauses = [spatial("west", binding_id="same_reference", reference_clause_ids=["is_west"]),
        spatial("east", binding_id="same_reference", reference_clause_ids=["is_east"]),
        attribute("is_west", "position", "west", entity="reference"), attribute("is_east", "position", "east", entity="reference")]
    query = plan(clauses, {"operator": "AND", "operands": [leaf("west"), leaf("east")]})
    inputs = {"asset": [feature("one", Point(0, 0))], "reference": [feature("west", Point(-.001, 0), position="west"), feature("east", Point(.001, 0), position="east")]}
    result = execute_graph(query, inputs, coverage={"reference": {"enumeration_complete": True}})[0]
    assert all(condition["status"] == "pass" for condition in result["feature"]["properties"]["conditions"])
    assert result["group"] == "excluded"
    query["clauses"][1]["binding_id"] = "other_reference"
    assert execute_graph(query, inputs, coverage={"reference": {"enumeration_complete": True, "absence_supported": True, "absence_policy_id": "enumerated-records-policy"}})[0]["group"] == "matches"


def test_negation_applies_after_existential_binding_instead_of_selecting_far_reference():
    query = plan([spatial(binding_id="witness")], {"operator": "NOT", "operands": [leaf("near")]})
    inputs = {"asset": [feature("one", Point(0, 0))], "reference": [feature("near", Point(.001, 0)), feature("far", Point(.1, 0))]}
    assert execute_graph(query, inputs, coverage={"reference": {"enumeration_complete": True}})[0]["group"] == "excluded"
    inputs["reference"] = [feature("far", Point(.1, 0))]
    assert execute_graph(query, inputs, coverage={"reference": {"enumeration_complete": True, "absence_supported": True, "absence_policy_id": "enumerated-records-policy"}})[0]["group"] == "matches"


def test_reference_clause_negation_uses_selected_shared_witness():
    clauses = [spatial(binding_id="drain"), attribute("covered", "covered", "yes", entity="reference", binding_id="drain")]
    query = plan(clauses, {"operator": "AND", "operands": [leaf("near"), {"operator": "NOT", "operands": [leaf("covered")]}]})
    inputs = {"asset": [feature("one", Point(0, 0))], "reference": [feature("close_closed", Point(.001, 0), covered="yes"), feature("far_open", Point(.1, 0), covered="no")]}
    assert execute_graph(query, inputs, coverage={"reference": {"enumeration_complete": True}})[0]["group"] == "excluded"
    inputs["reference"].append(feature("close_open", Point(-.001, 0), covered="no"))
    assert execute_graph(query, inputs, coverage={"reference": {"enumeration_complete": True}})[0]["group"] == "matches"


def test_partial_reference_coverage_and_empty_visual_detection_never_prove_absence():
    query = plan([spatial()], {"operator": "NOT", "operands": [leaf("near")]})
    inputs = {"asset": [feature("one", Point(0, 0))], "reference": []}
    assert execute_graph(query, inputs)[0]["group"] == "needs_review"
    assert execute_graph(query, inputs, coverage={"reference": {"enumeration_complete": True, "absence_supported": False}})[0]["group"] == "needs_review"
    query["entities"][1].update(kind="visual", visual_query="newly described installation")
    assert execute_graph(query, inputs, coverage={"reference": {"enumeration_complete": True}})[0]["group"] == "needs_review"
    query["clauses"][0]["binding_id"] = "visual_witness"
    assert execute_graph(query, inputs, coverage={"reference": {"enumeration_complete": True}})[0]["group"] == "needs_review"


def test_geometry_boundary_contact_and_distance_equality_are_inclusive():
    crs = "EPSG:32631"
    to_wgs = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    left = Point(*to_wgs.transform(500000, 0))
    right = Point(*to_wgs.transform(500200, 0))
    # Use a tolerance smaller than positional uncertainty for the deliberate
    # grid-boundary case; inside and outside are independently established.
    inputs = {"asset": [feature("one", left)], "reference": [feature("right", right)]}
    clause = spatial(distance_m=200.000001)
    assert execute_graph(plan([clause], leaf("near")), inputs, analysis_crs=crs)[0]["group"] == "matches"
    clause["distance_m"] = 199.999999
    assert execute_graph(plan([clause], leaf("near")), inputs, analysis_crs=crs, coverage={"reference": {"enumeration_complete": True}})[0]["group"] == "excluded"
    query = plan([spatial(operator="intersects", distance_m=None)], leaf("near"))
    assert execute_graph(query, {"asset": [feature("one", box(0, 0, .001, .001))], "reference": [feature("boundary", Point(.001, .001))]})[0]["group"] == "matches"


def test_visual_attribute_on_another_object_does_not_qualify_subject():
    entities = [{"id": "asset", "description": "commercial roofs", "kind": "mapped"}, {"id": "panels", "description": "panel installation", "kind": "visual", "visual_query": "rooftop solar arrays"}]
    query = plan([{"id": "has_panels", "kind": "visual", "entity_id": "asset", "reference_entity_id": "panels", "operator": "associated_with"}], leaf("has_panels"), entities)
    evidence = [{"query": "rooftop solar arrays", "status": "pass", "evidence_type": "model_detection"}]
    inputs = {"asset": [feature("A", box(0, 0, .001, .001)), feature("B", box(.002, 0, .003, .001))], "panels": [feature("p", box(.0001, .0001, .0002, .0002), visual_evidence=evidence)]}
    records = execute_graph(query, inputs, association_links=[{"subject_entity_id": "asset", "subject_feature_id": "A", "reference_entity_id": "panels", "reference_feature_id": "p", "status": "pass", "policy_id": "registered-association-v1"}])
    assert {item["feature"]["properties"]["source_feature_id"]: item["group"] for item in records} == {"A": "matches", "B": "needs_review"}


def test_exact_prompt_and_live_detection_required_for_visual_results():
    entities = [{"id": "asset", "description": "faded red staging shelters", "visual_query": "faded red staging shelters", "kind": "visual"}]
    query = plan(entities=entities)
    inputs = {"asset": [feature("fallback", box(0, 0, .001, .001), geometry_role="image_tile_coverage"),
        feature("wrong_prompt", Point(0, 0), visual_evidence=[{"query": "shelters", "status": "pass"}]),
        feature("grounded", Point(0, 0), visual_evidence=[{"query": "faded red staging shelters", "status": "pass", "evidence_type": "live_model_detection"}])]}
    groups = {record["feature"]["properties"]["source_feature_id"]: record["group"] for record in execute_graph(query, inputs)}
    assert groups == {"fallback": "visual_candidates", "wrong_prompt": "visual_candidates", "grounded": "matches"}


def test_temporal_and_approximate_boundary_claims_remain_reviewable():
    temporal = {"id": "changed", "kind": "temporal", "entity_id": "asset", "operator": "changed", "original_text": "cleared since last year"}
    assert execute_graph(plan([temporal], leaf("changed")), {"asset": [feature("one", Point(0, 0), imagery_date="2026-10-01")]})[0]["group"] == "needs_review"
    query = plan([spatial(operator="intersects", distance_m=None)], leaf("near"))
    inputs = {"asset": [feature("box", box(0, 0, .001, .001), geometry_role="object_box")], "reference": [feature("edge", Point(.001, .001))]}
    assert execute_graph(query, inputs)[0]["group"] == "needs_review"


def test_stable_identity_preserves_colocated_entities_and_dated_observations():
    inputs = {"asset": [feature("first", Point(0, 0)), feature("second", Point(0, 0)),
        feature("dated", Point(0, 0), imagery_date="2025-01-01"), feature("dated", Point(0, 0), imagery_date="2026-01-01")]}
    assert len(execute_graph(plan(), inputs)) == 4
    duplicate = deepcopy(inputs["asset"][0])
    inputs["asset"].append(duplicate)
    assert len(execute_graph(plan(), inputs)) == 4


def test_exact_source_duplicates_merge_provenance_but_different_source_ids_do_not():
    first = feature("local-a", Point(0, 0), dataset_id="a", source_feature_id="source-1", source_metadata={"source": "source-uri", "extraction_date": "2026-10-01"}, source_attributes={"material": "stone"})
    second = deepcopy(first)
    second["id"] = "local-b"
    second["properties"]["dataset_id"] = "b"
    assert record_id("asset", first) == record_id("asset", second)
    records = execute_graph(plan(), {"asset": [first, second]})
    assert len(records) == 1
    assert records[0]["feature"]["properties"]["source_dataset_ids"] == ["a", "b"]
    second["properties"]["source_feature_id"] = "source-2"
    assert len(execute_graph(plan(), {"asset": [first, second]})) == 2


def test_no_result_pagination_or_nearby_box_deduplication_in_graph():
    inputs = {"asset": [feature(str(index), Point(0, 0)) for index in range(150)]}
    assert len(execute_graph(plan(), inputs)) == 150
    assert inputs["asset"][0]["properties"] == {}


def test_invalid_candidate_entity_and_geographic_crs_are_rejected():
    with pytest.raises(SearchPlanError, match="unknown plan entity"):
        execute_graph(plan(), {"unknown": []})
    with pytest.raises(SearchPlanError, match="metre"):
        execute_graph(plan(), {"asset": [feature("one", Point(0, 0))]}, analysis_crs="EPSG:4326")


def test_wide_area_spatial_processing_has_truthful_unknown_instead_of_degree_distance():
    query = plan([spatial()], leaf("near"))
    result = execute_graph(query, {"asset": [feature("one", Point(0, 0))], "reference": [feature("far", Point(20, 20))]}, coverage={"reference": {"enumeration_complete": True}})[0]
    assert result["group"] == "needs_review"
    assert result["feature"]["properties"]["analysis_crs"] is None


def test_source_observation_period_limits_subject_and_reference_evidence():
    subject = feature("one", Point(0, 0), scope_time_status="unknown", scope_time_reason="The requested historical date has no source snapshot.")
    assert execute_graph(plan(), {"asset": [subject]})[0]["group"] == "needs_review"
    assert execute_graph(plan(), {"asset": [subject]})[0]["feature"]["properties"]["entity_attribute_conditions"][0]["condition"] == "scope_observation_period"
    query = plan([spatial()], {"operator": "NOT", "operands": [leaf("near")]})
    inputs = {"asset": [feature("one", Point(0, 0))], "reference": [feature("current_far_record", Point(.1, 0), scope_time_status="unknown")]}
    assert execute_graph(query, inputs, coverage={"reference": {"enumeration_complete": True}})[0]["group"] == "needs_review"
    query["clauses"][0]["binding_id"] = "dated_reference"
    assert execute_graph(query, inputs, coverage={"reference": {"enumeration_complete": True, "absence_supported": True, "absence_policy_id": "current-records-only"}})[0]["group"] == "needs_review"
    historical_attribute = plan([attribute("material", "material", "wood")], leaf("material"))
    subject["properties"]["material"] = "stone"
    assert execute_graph(historical_attribute, {"asset": [subject]})[0]["group"] == "needs_review"


def test_spatial_index_retains_unknown_geometries_and_uncertain_threshold_pairs():
    query = plan([spatial()], leaf("near"))
    invalid_reference = {"type": "Feature", "id": "unlocated", "geometry": None, "properties": {}}
    inputs = {"asset": [feature("one", Point(0, 0))], "reference": [feature("beyond_nominal_threshold", Point(.002, 0), positional_uncertainty_m=50), invalid_reference]}
    result = execute_graph(query, inputs, coverage={"reference": {"enumeration_complete": True}})[0]
    assert result["group"] == "needs_review"
    condition = result["feature"]["properties"]["conditions"][0]
    assert condition["references_evaluated"] == 2
    assert condition["definite_spatial_rejects_count"] == 0


def test_all_quantifier_accounts_for_definitely_pruned_reference_failures():
    query = plan([spatial(quantifier="all")], leaf("near"))
    inputs = {"asset": [feature("one", Point(0, 0))], "reference": [feature("near", Point(.001, 0)), feature("far", Point(.1, 0))]}
    result = execute_graph(query, inputs, coverage={"reference": {"enumeration_complete": True}})[0]
    assert result["group"] == "excluded"
    assert result["feature"]["properties"]["conditions"][0]["definite_spatial_rejects_count"] == 1
