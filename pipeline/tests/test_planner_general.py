"""General planner contracts for vector, live-grounding and multi-relation searches."""
from __future__ import annotations

import json
import sys
from copy import deepcopy
from types import SimpleNamespace

import pytest

from terralens.planner import (
    PlanError,
    QueryPlan,
    SpatialRelation,
    _guard_interpretation,
    _llm_plan,
    plan_query,
    validate_plan,
)


@pytest.fixture(autouse=True)
def deterministic_parser(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)


@pytest.fixture
def manifest():
    return {
        "supported_concepts": {
            "solar_panels": {
                "prompt": "solar panels",
                "aliases": ["solar panels"],
                "association_layer": "buildings",
                "grounding_available": True,
            },
            "swimming_pools": {
                "prompt": "swimming pools",
                "aliases": ["swimming pools"],
                "association_layer": None,
                "grounding_available": True,
            },
        },
        "vector_layers": {
            "buildings": {
                "aliases": ["building", "buildings"],
                "fields": ["id", "building", "name"],
            },
            "roads": {
                "aliases": ["road", "roads", "street", "streets"],
                "fields": ["id", "highway", "surface"],
            },
            "waterways": {
                "aliases": ["waterway", "waterways", "drain", "drains"],
                "fields": ["id", "waterway", "covered"],
            },
            "parks": {
                "aliases": ["park", "parks"],
                "fields": ["id", "leisure", "name"],
                "attribute_filters": {
                    "public access": {"field": "leisure", "operator": "equals", "value": "park"}
                },
            },
        },
        "supported_predicates": ["within_distance", "intersects"],
        "limits": {"maximum_distance_m": 1000},
    }


def test_schema_extends_legacy_plan_without_breaking_defaults(manifest):
    plan = validate_plan({
        "concept_id": "swimming_pools",
        "visual_query": "swimming pools",
        "reference_layer": None,
        "predicate": None,
        "distance_m": None,
    }, manifest)
    assert plan.query_type == "objects"
    assert plan.subject_layer is None
    assert plan.relations == []
    assert plan.attributes == []

    with pytest.raises(PlanError, match="primary relationship"):
        validate_plan({
            **plan.model_dump(),
            "reference_layer": None,
            "relations": [{
                "reference_layer": "roads", "predicate": "intersects", "distance_m": None, "attributes": []
            }],
        }, manifest)

    with pytest.raises(ValueError, match="reference layer"):
        SpatialRelation(
            reference_layer="roads",
            predicate="intersects",
            attributes=[{"role": "subject", "field": "highway", "operator": "equals", "value": "primary"}],
        )


def test_cached_object_and_vector_subjects_need_no_relation(manifest):
    cached = plan_query("Find swimming pools", manifest)
    assert cached.query_type == "objects"
    assert cached.concept_id == "swimming_pools"
    assert cached.reference_layer is None

    vector = plan_query("Find residential buildings", manifest)
    assert vector.query_type == "vectors"
    assert vector.concept_id is None
    assert vector.subject_layer == "buildings"
    assert vector.reference_layer is None
    assert [(item.role, item.field, item.value) for item in vector.attributes] == [
        ("subject", "building", "residential")
    ]


def test_explicit_mapped_subject_resolves_concept_layer_collision(manifest):
    collision = deepcopy(manifest)
    collision["supported_concepts"]["buildings"] = {
        "prompt": "buildings",
        "aliases": ["building", "buildings"],
        "association_layer": "buildings",
        "grounding_available": True,
    }

    bare = plan_query("Find buildings", collision)
    assert bare.query_type == "objects"
    assert bare.concept_id == "buildings"

    for qualifier in ("mapped", "annotated"):
        vector = plan_query(f"Find {qualifier} buildings within 100 metres of roads", collision)
        assert vector.query_type == "vectors"
        assert vector.concept_id is None
        assert vector.subject_layer == "buildings"
        assert vector.visual_query == "buildings"
        assert vector.reference_layer == "roads"

    with pytest.raises(PlanError, match="vector subject layer"):
        _guard_interpretation("Find mapped buildings", bare, collision)


def test_exact_vector_filters_and_multiple_and_relations(manifest):
    plan = plan_query(
        "Find buildings with building=house within 100 metres of roads with highway=residential "
        "and intersecting waterways with waterway=stream",
        manifest,
    )
    assert plan.query_type == "vectors"
    assert plan.subject_layer == "buildings"
    assert (plan.reference_layer, plan.predicate, plan.distance_m) == ("roads", "within_distance", 100)
    assert [(item.role, item.field, item.value) for item in plan.attributes] == [
        ("subject", "building", "house"),
        ("reference", "highway", "residential"),
    ]
    assert len(plan.relations) == 1
    assert plan.relations[0].reference_layer == "waterways"
    assert plan.relations[0].predicate == "intersects"
    assert plan.relations[0].distance_m is None
    assert [(item.role, item.field, item.value) for item in plan.relations[0].attributes] == [
        ("reference", "waterway", "stream")
    ]


@pytest.mark.parametrize("message", ["make it 50 metres", "Change the distance to 50 metres", "Set the radius to 50 metres"])
def test_radius_followup_changes_primary_only_and_preserves_every_other_clause(manifest, message):
    original = plan_query(
        "Find buildings with building=house within 100 metres of residential roads "
        "and within 300 metres of parks with public access",
        manifest,
    )
    edited = plan_query(message, manifest, original.model_dump())
    assert edited.distance_m == 50
    assert edited.subject_layer == original.subject_layer
    assert edited.attributes == original.attributes
    assert edited.relations == original.relations
    assert edited.relations[0].distance_m == 300


def test_ground_mode_keeps_complete_free_object_description(manifest):
    plan = plan_query("Find faded red emergency vehicles within 250 metres of roads", manifest, mode="ground")
    assert plan.query_type == "objects"
    assert plan.concept_id is None
    assert plan.visual_query == "faded red emergency vehicles"
    assert plan.reference_layer == "roads"
    assert plan.distance_m == 250

    known = plan_query("Find damaged solar panels", manifest, mode="ground")
    assert known.concept_id == "solar_panels"
    assert known.visual_query == "damaged solar panels"


@pytest.mark.parametrize("query", [
    "Find vehicles without license plates",
    "Find recently cleared land",
    "Find illegal buildings",
])
def test_ground_mode_rejects_claims_needing_absence_history_or_legal_evidence(manifest, query):
    with pytest.raises(PlanError):
        plan_query(query, manifest, mode="ground")


def test_parser_rejects_unknown_meaning_instead_of_inventing_or_dropping_it(manifest):
    with pytest.raises(PlanError, match="family"):
        plan_query("Find family buildings", manifest)
    with pytest.raises(PlanError, match="reference layer"):
        plan_query("Find buildings within 100 metres of schools", manifest)
    with pytest.raises(PlanError):
        plan_query("Find buildings near roads", manifest)


def test_validation_keeps_modes_and_attribute_roles_truthful(manifest):
    vector = plan_query("Find roads", manifest)
    with pytest.raises(PlanError, match="catalog-scan"):
        validate_plan(vector, manifest, mode="explore")
    with pytest.raises(PlanError, match="visual concept"):
        validate_plan({**vector.model_dump(), "concept_id": "solar_panels"}, manifest)
    with pytest.raises(PlanError, match="reference layer"):
        validate_plan({
            **vector.model_dump(),
            "reference_layer": "roads",
            "predicate": "intersects",
            "relations": [{
                "reference_layer": "parks",
                "predicate": "intersects",
                "distance_m": None,
                "attributes": [{"role": "subject", "field": "leisure", "operator": "equals", "value": "park"}],
            }],
        }, manifest)


def test_guard_detects_a_dropped_additional_layer_or_distance(manifest):
    complete = plan_query(
        "Find buildings within 100 metres of roads and within 300 metres of parks",
        manifest,
    )
    dropped = QueryPlan.model_validate({**complete.model_dump(), "relations": []})
    with pytest.raises(PlanError):
        _guard_interpretation(
            "Find buildings within 100 metres of roads and within 300 metres of parks",
            dropped,
            manifest,
        )

    attribute_plan = plan_query("Find buildings", manifest)
    with pytest.raises(PlanError, match="not preserved"):
        _guard_interpretation(
            "Find buildings with building=house",
            attribute_plan,
            manifest,
        )


def test_language_model_schema_contains_generalized_contract(manifest, monkeypatch):
    captured = {}
    output = {
        "query_type": "vectors",
        "concept_id": None,
        "visual_query": "buildings",
        "reference_layer": None,
        "predicate": None,
        "distance_m": None,
        "attributes": [],
        "unsupported_conditions": [],
        "subject_layer": "buildings",
        "relations": [],
    }

    class Completions:
        def create(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(output), refusal=None), finish_reason="stop")]
            )

    class FakeOpenAI:
        def __init__(self, **kwargs):
            self.chat = SimpleNamespace(completions=Completions())

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=FakeOpenAI))
    monkeypatch.setenv("OPENAI_MODEL", "test-model")
    plan = _llm_plan("Find buildings", manifest, None, "catalog_scan")
    schema = captured["response_format"]["json_schema"]["schema"]
    assert plan.query_type == "vectors"
    assert schema["properties"]["query_type"]["enum"] == ["objects", "tiles", "vectors"]
    assert schema["properties"]["subject_layer"]
    assert schema["properties"]["relations"]["maxItems"] == 3
    relation = schema["properties"]["relations"]["items"]
    assert relation["required"] == ["reference_layer", "predicate", "distance_m", "attributes"]
    assert relation["properties"]["attributes"]["items"]["properties"]["role"]["enum"] == ["reference"]
