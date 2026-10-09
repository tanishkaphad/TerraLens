"""Deterministic dataset and coordinate preservation without provider calls."""
from copy import deepcopy

import pytest

from terralens.intent import explicit_wgs84_boxes, guard_intent_constraints
from terralens.workflow import Workflow, WorkflowError

BOX = [-43.12, -23.02, -43.10, -23.00]


@pytest.fixture
def catalog():
    manifest = {"dataset_version": "v1", "coverage": {"bbox_wgs84": [-44, -24, -42, -22]},
        "vector_layers": {"buildings": {"fields": []}}, "supported_concepts": {}}
    return {"spacenet": {"manifest": deepcopy(manifest)}, "demo": {"manifest": deepcopy(manifest)}}


def search(*, dataset="spacenet", bbox=None, identifier="find"):
    return {"id": identifier, "operation": "search", "label": "Search supplied buildings", "dataset_id": dataset,
        "mode": "catalog_scan", "bbox": bbox, "plan": {"query_type": "vectors", "concept_id": None,
            "visual_query": "buildings", "subject_layer": "buildings", "reference_layer": None,
            "predicate": None, "distance_m": None, "attributes": [], "relations": []}}


def workflow(steps):
    return Workflow.model_validate({"title": "Search", "status": "ready", "steps": steps,
        "requirements": [{"text": "Requested area and source", "status": "planned", "step_ids": [step["id"] for step in steps]}]})


@pytest.mark.parametrize("message", [
    "Search bbox [-43.12,-23.02,-43.10,-23.00]",
    "Use WGS84 bounds: (-43.12; -23.02; -43.10; -23.00)",
    "west=-43.12 south=-23.02 east=-43.10 north=-23.00",
    "north=-23.00, east=-43.10, south=-23.02, west=-43.12",
])
def test_explicit_coordinate_formats_keep_longitude_latitude_order(message):
    assert explicit_wgs84_boxes(message) == [tuple(BOX)]


def test_current_explicit_coordinates_are_accepted_and_reported(catalog):
    report = guard_intent_constraints("Find buildings in bbox [-43.12,-23.02,-43.10,-23.00]", "spacenet", [],
        workflow([search(bbox=BOX)]), catalog)
    assert report["status"] == "passed"
    assert report["search_steps"][0]["coordinate_source"] == "current_user_message"


def test_fabricated_place_name_coordinates_are_rejected(catalog):
    with pytest.raises(WorkflowError, match="invented coordinates"):
        guard_intent_constraints("Find buildings in Rio", "spacenet", [], workflow([search(bbox=BOX)]), catalog)


def test_even_geographically_overlapping_dataset_cannot_replace_selected_source(catalog):
    with pytest.raises(WorkflowError, match="selected dataset"):
        guard_intent_constraints("Find buildings", "spacenet", [], workflow([search(dataset="demo")]), catalog)


def test_coverage_inventory_may_list_other_registered_sources(catalog):
    inventory = {"id": "inventory", "operation": "coverage", "label": "List datasets", "dataset_id": "demo"}
    report = guard_intent_constraints("What datasets are available?", "spacenet", [], workflow([inventory]), catalog)
    assert report["status"] == "passed"
    assert report["search_steps"] == []


def test_explicit_current_bbox_cannot_be_dropped(catalog):
    with pytest.raises(WorkflowError, match="area was dropped"):
        guard_intent_constraints("Find buildings in bbox [-43.12,-23.02,-43.10,-23.00]", "spacenet", [],
            workflow([search()]), catalog)


def test_model_cannot_replace_current_box_with_history_box(catalog):
    prior = {"message": "Find in bbox [-43.12,-23.02,-43.10,-23.00]", "result": {"workflow": workflow([search(bbox=BOX)]).model_dump()}}
    with pytest.raises(WorkflowError, match="explicit user"):
        guard_intent_constraints("Now use bbox [-43.20,-23.20,-43.15,-23.15]", "spacenet", [prior],
            workflow([search(bbox=BOX)]), catalog)


def test_successful_validated_same_dataset_history_can_authorise_followup(catalog):
    prior = {"message": "Find in bbox [-43.12,-23.02,-43.10,-23.00]", "result": {"workflow": workflow([search(bbox=BOX)]).model_dump()}}
    report = guard_intent_constraints("Count those buildings", "spacenet", [prior], workflow([search(bbox=BOX)]), catalog)
    assert report["search_steps"][0]["coordinate_source"] == "validated_same_dataset_history"


def test_prior_model_invented_box_is_not_authority(catalog):
    prior = {"message": "Find buildings in Rio", "result": {"workflow": workflow([search(bbox=BOX)]).model_dump()}}
    with pytest.raises(WorkflowError, match="explicit user"):
        guard_intent_constraints("Count those buildings", "spacenet", [prior], workflow([search(bbox=BOX)]), catalog)


def test_prior_coordinates_from_other_dataset_are_not_authority(catalog):
    prior = {"message": "Find in bbox [-43.12,-23.02,-43.10,-23.00]", "result": {"workflow": workflow([search(dataset="demo", bbox=BOX)]).model_dump()}}
    with pytest.raises(WorkflowError, match="same dataset"):
        guard_intent_constraints("Count those buildings", "spacenet", [prior], workflow([search(bbox=BOX)]), catalog)


def test_failed_or_invalid_prior_workflow_cannot_authorise_bbox(catalog):
    prior = {"message": "Find in bbox [-43.12,-23.02,-43.10,-23.00]", "result": {"status": "failed", "workflow": workflow([search(bbox=BOX)]).model_dump()}}
    with pytest.raises(WorkflowError):
        guard_intent_constraints("Count those", "spacenet", [prior], workflow([search(bbox=BOX)]), catalog)


def test_latest_area_reset_removes_old_coordinate_authority(catalog):
    history = [{"message": "Find in bbox [-43.12,-23.02,-43.10,-23.00]", "result": {"workflow": workflow([search(bbox=BOX)]).model_dump()}},
        {"message": "Search the whole sample", "result": {"workflow": workflow([search()]).model_dump()}}]
    with pytest.raises(WorkflowError):
        guard_intent_constraints("Count those", "spacenet", history, workflow([search(bbox=BOX)]), catalog)


@pytest.mark.parametrize("message", ["bbox [200,0,201,1]", "bbox [1,2,0,3]", "west=1 south=2 east=3"])
def test_invalid_explicit_coordinates_do_not_pass_as_authority(message):
    with pytest.raises(WorkflowError):
        explicit_wgs84_boxes(message)


def test_no_dataset_selection_allows_any_registered_source_but_not_invented_area(catalog):
    report = guard_intent_constraints("Find buildings", None, [], workflow([search(dataset="demo")]), catalog)
    assert report["status"] == "passed"


def test_guard_does_not_mutate_history_workflow_or_catalog(catalog):
    selected = workflow([search(bbox=BOX)])
    prior = {"message": "Find in bbox [-43.12,-23.02,-43.10,-23.00]", "result": {"workflow": selected.model_dump()}}
    before = deepcopy((catalog, [prior], selected.model_dump()))
    guard_intent_constraints("Count those", "spacenet", [prior], selected, catalog)
    assert (catalog, [prior], selected.model_dump()) == before
