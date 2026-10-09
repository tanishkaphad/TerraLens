from __future__ import annotations

import copy

import pytest

from terralens.discovery import discovery_plan
from terralens.planner import PlanError


@pytest.fixture
def manifest():
    return {"supported_concepts": {"buildings": {"prompt": "buildings", "aliases": ["buildings"], "association_layer": "buildings"}},
        "vector_layers": {"buildings": {"aliases": ["buildings"], "fields": []}, "roads": {"aliases": ["roads"], "fields": ["highway"]}},
        "limits": {"maximum_distance_m": 500}, "supported_predicates": ["within_distance", "intersects"]}


def test_routes_cached_visual_spatial_query_and_preserves_radius(manifest):
    plan, mode = discovery_plan("Find buildings within 50 metres of roads", manifest)
    assert mode == "catalog_scan"
    assert (plan.concept_id, plan.reference_layer, plan.distance_m) == ("buildings", "roads", 50)


def test_routes_new_visible_category_without_a_category_list(manifest):
    plan, mode = discovery_plan("Find blue shipping containers within 50 metres of roads", manifest)
    assert mode == "ground"
    assert plan.visual_query == "blue shipping containers"
    assert plan.concept_id is None
    assert plan.reference_layer == "roads"


def test_routes_new_visible_category_without_spatial_constraints(manifest):
    plan, mode = discovery_plan("Find swimming pools", manifest)
    assert mode == "ground"
    assert plan.query_type == "objects"
    assert plan.concept_id is None
    assert plan.visual_query == "swimming pools"


def test_routes_explicit_mapped_subject_as_vector_evidence(manifest):
    plan, mode = discovery_plan("Find mapped roads", manifest)
    assert mode == "catalog_scan"
    assert plan.query_type == "vectors"
    assert plan.subject_layer == "roads"


def test_keeps_visible_appearance_words_in_the_grounding_prompt(manifest):
    plan, mode = discovery_plan("Find red-roofed buildings", manifest)
    assert mode == "ground"
    assert plan.concept_id == "buildings"
    assert plan.visual_query == "red-roofed buildings"


def test_new_object_can_use_a_registered_reference_attribute(manifest):
    plan, mode = discovery_plan(
        "Find blue shipping containers within 50 metres of residential roads", manifest
    )
    assert mode == "ground"
    assert plan.visual_query == "blue shipping containers"
    assert [(item.role, item.field, item.value) for item in plan.attributes] == [
        ("reference", "highway", "residential")
    ]


def test_semantic_condition_uses_catalog_attribute_when_it_is_registered(manifest):
    catalog = copy.deepcopy(manifest)
    catalog["vector_layers"]["buildings"]["fields"] = ["building:use"]
    plan, mode = discovery_plan("Find commercial buildings", catalog)
    assert mode == "catalog_scan"
    assert [(item.role, item.field, item.value) for item in plan.attributes] == [
        ("subject", "building:use", "commercial")
    ]


def test_scene_description_is_preserved_and_never_claims_object_grounding(manifest):
    plan, mode = discovery_plan("Find satellite imagery of densely built neighbourhoods", manifest)
    assert mode == "explore"
    assert plan.query_type == "tiles"
    assert plan.visual_query == "densely built neighbourhoods"


@pytest.mark.parametrize("query", ["Find buildings within 500 metres of rivers", "Find buildings constructed between 2020 and 2025", "Find NDVI", "Find infrastructure near flood-prone areas", "Find images of buildings within 50 metres of roads"])
def test_missing_evidence_does_not_fall_back_to_an_unconstrained_visual_search(manifest, query):
    with pytest.raises(PlanError):
        discovery_plan(query, manifest)


@pytest.mark.parametrize(
    ("query", "missing_evidence"),
    [
        ("Find commercial buildings", "land-use"),
        ("Find residential buildings", "land-use"),
        ("Find government-owned buildings within 50 metres of roads", "ownership"),
        ("Find buildings with permits", "permit"),
        ("Find insured buildings", "insurance"),
        ("Find properties for sale", "market"),
        ("Find vacant buildings", "occupancy"),
        ("Find buildings with more than 50 residents", "census"),
        ("Find schools open today", "operational"),
        ("Find hospitals with available beds", "operational"),
        ("Find high-crime neighbourhoods", "crime"),
        ("Find structurally unsafe buildings", "inspection"),
        ("Find buildings at risk of collapse", "inspection"),
        ("Find energy-efficient buildings", "energy audit"),
        ("Find affordable housing", "market"),
        ("Find healthy crops", "spectral"),
        ("Find construction sites", "construction"),
        ("Find damaged buildings", "inspection"),
        ("Find roofs suitable for solar", "spectral"),
        ("Find roads from 2020", "dated source"),
        ("Find images of hospitals with available beds", "operational"),
    ],
)
def test_nonvisual_claims_never_become_raw_model_prompts(manifest, query, missing_evidence):
    with pytest.raises(PlanError) as raised:
        discovery_plan(query, manifest)
    message = str(raised.value).casefold()
    assert "cannot establish" in message
    assert missing_evidence in message


@pytest.mark.parametrize(
    "query",
    ["Find trees north of roads", "Find trees in roads", "Find images of buildings beside roads"],
)
def test_unexecutable_spatial_language_is_not_embedded_in_a_visual_prompt(manifest, query):
    with pytest.raises(PlanError, match="spatial"):
        discovery_plan(query, manifest)
