"""Contracts for arbitrary runtime concepts and complete structured interpretation."""
from __future__ import annotations

import copy
import json

import pytest
from shapely.geometry import box, mapping

from terralens.search_plan import SearchPlan, SearchPlanError, parse_search_plan, validate_search_plan


def leaf(identifier):
    return {"operator": "clause", "clause_id": identifier}


@pytest.fixture
def catalog():
    return {"datasets": {"survey-42": {"layers": {}}}, "layers": {
        "survey-42:untested_assets": {"dataset_id": "survey-42", "layer_id": "untested_assets", "fields": ["material", "capacity"]},
        "survey-42:inspection_lines": {"dataset_id": "survey-42", "layer_id": "inspection_lines", "fields": ["inspection_state"]},
    }}


def raw_plan():
    return {"original_query": "Find test objects satisfying the supplied conditions.",
        "entities": [{"id": "assets", "description": "assets from the supplied survey", "kind": "mapped", "dataset_id": "survey-42", "layer_id": "untested_assets"}],
        "subject_entity_ids": ["assets"]}


def test_arbitrary_new_layers_and_visual_concepts_need_no_central_edits(catalog):
    raw = raw_plan()
    raw["entities"].append({"id": "new_concept", "description": "faded violet emergency staging shelters", "kind": "visual", "visual_query": "faded violet emergency staging shelters"})
    raw["subject_entity_ids"].append("new_concept")
    result = parse_search_plan(raw, catalog)
    assert result.entities[1].visual_query == "faded violet emergency staging shelters"
    assert result.entities[0].layer_id == "untested_assets"
    assert result.scope.dataset_ids == []


def test_unknown_attribute_and_operator_preserve_original_requirement(catalog):
    raw = raw_plan()
    raw["clauses"] = [{"id": "missing", "kind": "attribute", "entity_id": "assets", "operator": "equals", "field": "ownership", "value": "municipal", "original_text": "owned by the municipality"},
        {"id": "unsupported", "kind": "temporal", "entity_id": "assets", "operator": "arbitrary_future_operation", "original_text": "becoming accessible next year"}]
    raw["expression"] = {"operator": "AND", "operands": [leaf("missing"), leaf("unsupported")]}
    result = validate_search_plan(raw, catalog)
    assert [(clause.availability, clause.original_text) for clause in result.clauses] == [("unknown", "owned by the municipality"), ("unsupported", "becoming accessible next year")]
    assert raw["clauses"][0].get("availability") is None


@pytest.mark.parametrize("mutation", [
    lambda raw: raw.update(subject_entity_ids=["unknown"]),
    lambda raw: raw["entities"].append(copy.deepcopy(raw["entities"][0])),
    lambda raw: raw.update(clauses=[{"id": "bad", "kind": "spatial", "entity_id": "assets", "reference_entity_id": "unknown", "operator": "intersects"}], expression=leaf("bad")),
    lambda raw: raw.update(expression=leaf("absent")),
])
def test_references_are_validated_even_before_execution(catalog, mutation):
    raw = raw_plan()
    mutation(raw)
    with pytest.raises(SearchPlanError):
        validate_search_plan(raw, catalog)


def test_model_cannot_drop_unreferenced_clauses_or_boolean_grouping(catalog):
    raw = raw_plan()
    raw["clauses"] = [{"id": "one", "kind": "attribute", "entity_id": "assets", "operator": "equals", "field": "material", "value": "stone"}]
    with pytest.raises(SearchPlanError, match="explicit Boolean"):
        validate_search_plan(raw, catalog)
    raw["expression"] = {"operator": "AND", "operands": []}
    with pytest.raises(SearchPlanError, match="at least one operand"):
        validate_search_plan(raw, catalog)


def test_same_reference_qualifiers_are_typed_not_global_filters(catalog):
    raw = raw_plan()
    raw["entities"].append({"id": "line", "description": "survey inspection lines", "kind": "mapped", "dataset_id": "survey-42", "layer_id": "inspection_lines"})
    raw["clauses"] = [{"id": "near", "kind": "spatial", "entity_id": "assets", "reference_entity_id": "line", "operator": "within_distance", "distance_m": 200,
        "reference_clause_ids": ["open"]}, {"id": "open", "kind": "attribute", "entity_id": "line", "field": "inspection_state", "operator": "equals", "value": "open"}]
    raw["expression"] = leaf("near")
    assert validate_search_plan(raw, catalog).clauses[0].reference_clause_ids == ["open"]
    raw["clauses"][1]["entity_id"] = "assets"
    with pytest.raises(SearchPlanError, match="same reference entity"):
        validate_search_plan(raw, catalog)


@pytest.mark.parametrize("distance", [float("nan"), float("inf"), -1, True])
def test_distances_are_explicit_finite_nonnegative_numbers(catalog, distance):
    raw = raw_plan()
    raw["clauses"] = [{"id": "near", "kind": "spatial", "entity_id": "assets", "reference_entity_id": "assets", "operator": "within_distance", "distance_m": distance}]
    raw["expression"] = leaf("near")
    with pytest.raises(SearchPlanError):
        validate_search_plan(raw, catalog)


def test_finite_geometry_dates_and_unresolved_area_cannot_be_silently_substituted(catalog):
    raw = raw_plan()
    raw["scope"] = {"mode": "geometry", "geometry": {"type": "Point", "coordinates": [float("nan"), 0]}}
    with pytest.raises(SearchPlanError, match="finite"):
        validate_search_plan(raw, catalog)
    raw["scope"] = {"mode": "all_loaded", "place_query": "an unresolved named area"}
    with pytest.raises(SearchPlanError, match="unresolved named area"):
        validate_search_plan(raw, catalog)
    raw["status"] = "needs_clarification"
    assert validate_search_plan(raw, catalog).scope.place_query == "an unresolved named area"
    raw["scope"] = {"mode": "all_loaded", "start_date": "2026-10-09", "end_date": "2025-01-01"}
    with pytest.raises(SearchPlanError, match="ends before"):
        validate_search_plan(raw, catalog)


def test_ambiguous_layer_requires_explicit_dataset(catalog):
    catalog["datasets"]["other"] = {"layers": {}}
    catalog["layers"]["other:untested_assets"] = {"dataset_id": "other", "layer_id": "untested_assets", "fields": ["material"]}
    raw = raw_plan()
    raw["entities"][0].pop("dataset_id")
    with pytest.raises(SearchPlanError, match="ambiguous"):
        validate_search_plan(raw, catalog)


def test_fully_qualified_catalog_layer_can_also_specify_its_dataset(catalog):
    raw = raw_plan()
    raw["entities"][0]["layer_id"] = "survey-42:untested_assets"
    assert validate_search_plan(raw, catalog).entities[0].dataset_id == "survey-42"


def test_unconfigured_interpreter_requires_structured_builder_and_retains_query(catalog, monkeypatch):
    import terralens.assistant as assistant
    monkeypatch.setattr(assistant, "status", lambda: {"configured": False, "provider": "gemini", "model": None})
    for key in ("SEARCH_PARSER_MODEL", "SEARCH_PARSER_BASE_URL", "SEARCH_PARSER_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    query = "Search any newly described concept anywhere I choose"
    with pytest.raises(SearchPlanError) as caught:
        parse_search_plan(query, catalog)
    assert caught.value.code == "needs_structured_plan"
    assert caught.value.details["original_query"] == query


def test_parser_uses_configured_provider_and_real_catalog_with_preserved_followup(catalog, monkeypatch):
    import terralens.assistant as assistant
    monkeypatch.setattr(assistant, "status", lambda: {"configured": True, "provider": "gemini", "model": "configured-model"})
    previous = raw_plan()
    previous["clauses"] = [{"id": "attribute", "kind": "attribute", "entity_id": "assets", "operator": "equals", "field": "material", "value": "stone"}]
    previous["expression"] = leaf("attribute")
    calls = []
    def fake_call(client, model, name, schema, messages, description, provider):
        calls.append((model, provider, json.loads(messages[-1]["content"]), schema))
        raw = copy.deepcopy(previous)
        raw["clauses"][0]["value"] = "timber"
        raw["expression"] = json.dumps(raw["expression"])
        raw["revision"] = 2
        return raw
    monkeypatch.setattr(assistant, "_call", fake_call)
    result = parse_search_plan("Change the material to timber", catalog, previous, client=object())
    assert result.clauses[0].value == "timber"
    assert result.entities[0].layer_id == "untested_assets"
    assert calls[0][0:2] == ("configured-model", "gemini")
    assert calls[0][2]["runtime_catalog"]["layers"]["survey-42:untested_assets"]["fields"] == ["material", "capacity"]
    assert calls[0][2]["previous_validated_plan"]["clauses"][0]["value"] == "stone"
    assert previous["clauses"][0]["value"] == "stone"


def test_parser_generated_coordinates_need_resolved_geometry_evidence(catalog, monkeypatch):
    import terralens.assistant as assistant
    monkeypatch.setattr(assistant, "status", lambda: {"configured": True, "provider": "gemini", "model": "configured-model"})
    raw = raw_plan()
    raw["scope"] = {"mode": "geometry", "geometry": json.dumps(mapping(box(0, 0, .1, .1)))}
    monkeypatch.setattr(assistant, "_call", lambda *args: copy.deepcopy(raw))
    with pytest.raises(SearchPlanError) as caught:
        parse_search_plan("Find things in the named area", catalog, client=object())
    assert caught.value.code == "needs_clarification"
    catalog["active_scope"] = {"geometry": mapping(box(0, 0, .1, .1))}
    assert parse_search_plan("Search the selected area", catalog, client=object()).scope.mode == "geometry"


def test_parser_rate_limit_is_actionable_without_provider_secrets(catalog, monkeypatch):
    import terralens.assistant as assistant
    from openai import RateLimitError
    import httpx
    monkeypatch.setattr(assistant, "status", lambda: {"configured": True, "provider": "gemini", "model": "configured-model"})
    response = httpx.Response(429, request=httpx.Request("POST", "https://example.invalid/private?key=SECRET_API_KEY"))
    exception = RateLimitError("Do not expose SECRET_API_KEY or private provider body", response=response,
        body={"error": {"code": 429, "status": "RESOURCE_EXHAUSTED", "message": "private account data SECRET_API_KEY"}})
    def fail(*args):
        raise exception
    monkeypatch.setattr(assistant, "_call", fail)
    with pytest.raises(SearchPlanError) as caught:
        parse_search_plan("Find every mapped asset", catalog, client=object())
    assert caught.value.code == "parser_rate_limited"
    assert caught.value.details["error_type"] == "RateLimitError"
    assert caught.value.details["provider_status"] == 429
    assert caught.value.details["provider_error_code"] == "RESOURCE_EXHAUSTED"
    public = str(caught.value) + json.dumps(caught.value.details)
    assert "SECRET_API_KEY" not in public
    assert "example.invalid" not in public
    assert "private account" not in public


def test_parser_generic_failure_diagnostics_filter_unknown_provider_codes(catalog, monkeypatch):
    import terralens.assistant as assistant
    monkeypatch.setattr(assistant, "status", lambda: {"configured": True, "provider": "gemini", "model": "configured-model"})
    class ProviderFailure(Exception):
        status_code = 400
        code = "https://private.invalid/?key=SECRET"
        body = {"error": {"code": "account-sensitive-SECRET", "message": "private body"}}
    def fail(*args):
        raise ProviderFailure("SECRET request contents")
    monkeypatch.setattr(assistant, "_call", fail)
    with pytest.raises(SearchPlanError) as caught:
        parse_search_plan("Find every mapped asset", catalog, client=object())
    assert caught.value.code == "parser_unavailable"
    assert caught.value.details["provider_status"] == 400
    assert caught.value.details["provider_error_code"] is None
    assert caught.value.details["parser_stage"] == "provider_request"
    assert "SECRET" not in json.dumps(caught.value.details)
