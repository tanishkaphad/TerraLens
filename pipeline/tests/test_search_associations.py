"""Association policies act on real geometry and explicitly supplied provenance."""
from copy import deepcopy

import pytest
from shapely.geometry import box, mapping

from terralens.search_associations import build_association_links
from terralens.search_graph import execute_graph


def feature(identifier, geometry, **properties):
    return {"type": "Feature", "id": identifier, "geometry": mapping(geometry) if geometry is not None else None,
        "properties": properties}


def source():
    return {"footprints": [feature("one", box(73.8, 18.5, 73.801, 18.501), geometry_role="survey_footprint")],
        "observations": [feature("observation", box(73.8002, 18.5002, 73.8003, 18.5003),
            geometry_role="object_box", visual_query="arbitrary visual property",
            visual_evidence=[{"query": "arbitrary visual property", "status": "pass", "evidence_type": "model_detection"}])]}


def plan(policy_id="source-policy-v1"):
    return {"scope": {"mode": "all_loaded"}, "entities": [
        {"id": "footprints", "kind": "mapped", "description": "source footprints"},
        {"id": "observations", "kind": "visual", "description": "arbitrary visual property", "visual_query": "arbitrary visual property"}],
        "subject_entity_ids": ["footprints"], "clauses": [{"id": "associated", "kind": "visual", "entity_id": "footprints",
            "reference_entity_id": "observations", "operator": "associated_with",
            "evidence_requirement": {"association_policy_id": policy_id}}],
        "expression": {"operator": "clause", "clause_id": "associated"}}


def catalog(**overrides):
    policy = {"method": "unique_overlap", "minimum_overlap": .9,
        "subject_geometry_roles": ["survey_footprint"], "reference_geometry_roles": ["object_box"],
        "center_mode": "centroid", "version": "fixture-v1", "validation_status": "unvalidated"}
    policy.update(overrides)
    return {"association_policies": {"source-policy-v1": policy}}


def test_unique_geometric_association_links_exact_subject_and_graph_condition():
    candidates, query, metadata = source(), plan(), catalog()
    before = deepcopy(candidates)
    links = build_association_links(query, candidates, metadata)
    assert len(links) == 1 and links[0]["status"] == "pass"
    assert links[0]["subject_feature_id"] == "one" and links[0]["reference_feature_id"] == "observation"
    assert links[0]["candidate_subject_feature_ids"] == ["one"]
    assert links[0]["overlap_fraction"] == pytest.approx(1)
    assert links[0]["reference_area_m2"] > 0 and links[0]["analysis_crs"]
    assert links[0]["validation_status"] == "unvalidated" and links[0]["accuracy_claim"] is None
    assert candidates == before
    records = execute_graph(query, candidates, association_links=links)
    assert records[0]["group"] == "matches"


def test_competing_footprints_preserve_both_and_make_association_unknown():
    candidates = source()
    candidates["footprints"].append(feature("two", box(73.8001, 18.5001, 73.8011, 18.5011), geometry_role="survey_footprint"))
    links = build_association_links(plan(), candidates, catalog())
    assert len(links) == 2 and all(link["status"] == "unknown" for link in links)
    assert all(link["candidate_subject_count"] == 2 for link in links)
    assert {link["subject_feature_id"] for link in links} == {"one", "two"}
    assert all(record["group"] == "needs_review" for record in execute_graph(plan(), candidates, association_links=links))


def test_observation_on_neighbour_does_not_qualify_another_subject():
    candidates = source()
    candidates["footprints"].append(feature("neighbour", box(73.802, 18.5, 73.803, 18.501), geometry_role="survey_footprint"))
    links = build_association_links(plan(), candidates, catalog())
    assert len(links) == 1 and links[0]["subject_feature_id"] == "one"
    records = execute_graph(plan(), candidates, association_links=links)
    assert {record["feature"]["properties"]["source_feature_id"]: record["group"] for record in records} == {
        "one": "matches", "neighbour": "needs_review"}


@pytest.mark.parametrize("metadata", [{}, {"association_policies": {"source-policy-v1": {"method": "invented"}}},
    catalog(minimum_overlap=float("nan")), catalog(minimum_overlap=0), catalog(minimum_overlap=1.1),
    catalog(subject_geometry_roles=[])])
def test_missing_or_invalid_policy_never_infers_association(metadata):
    links = build_association_links(plan(), source(), metadata)
    assert links and all(link["status"] == "unknown" for link in links)


def test_date_requirements_keep_unknown_or_mismatched_epochs_unknown():
    candidates = source()
    assert build_association_links(plan(), candidates, catalog(require_contemporaneous=True))[0]["status"] == "unknown"
    candidates["footprints"][0]["properties"]["snapshot_date"] = "2026-10-09"
    candidates["observations"][0]["properties"]["acquisition_date"] = "2024-01-01"
    link = build_association_links(plan(), candidates, catalog(require_contemporaneous=True, maximum_date_delta_days=1))[0]
    assert link["status"] == "unknown" and link["date_delta_days"] > 1
    candidates["observations"][0]["properties"]["acquisition_date"] = "2026-10-09"
    assert build_association_links(plan(), candidates, catalog(require_contemporaneous=True))[0]["status"] == "pass"


def test_registration_requirements_and_positional_bounds_are_explicit():
    candidates = source()
    metadata = catalog(require_registration=True, max_positional_uncertainty_m=1)
    assert build_association_links(plan(), candidates, metadata)[0]["status"] == "unknown"
    for features in candidates.values():
        features[0]["properties"]["registration"] = {"status": "registered", "id": "fixture-registration", "positional_uncertainty_m": .2}
    assert build_association_links(plan(), candidates, metadata)[0]["status"] == "pass"
    candidates["observations"][0]["properties"]["registration"]["positional_uncertainty_m"] = 3
    assert build_association_links(plan(), candidates, metadata)[0]["status"] == "unknown"


def test_incompatible_scene_role_and_missing_subject_geometry_remain_unknown():
    candidates = source()
    candidates["observations"][0]["properties"]["geometry_role"] = "scene_footprint"
    assert build_association_links(plan(), candidates, catalog())[0]["status"] == "unknown"
    candidates = source()
    candidates["footprints"].append(feature("missing", None, geometry_role="survey_footprint"))
    link = build_association_links(plan(), candidates, catalog())[0]
    assert link["status"] == "unknown" and link["invalid_subject_feature_ids"] == ["missing"]


def test_uncertain_boundary_overlap_cannot_be_promoted_to_pass():
    candidates = source()
    candidates["observations"][0] = feature("observation", box(73.8009, 18.5002, 73.801, 18.5003), geometry_role="object_box", positional_uncertainty_m=3)
    candidates["footprints"][0]["properties"]["positional_uncertainty_m"] = 0
    link = build_association_links(plan(), candidates, catalog())[0]
    assert link["overlap_fraction"] == pytest.approx(1, abs=1e-4)
    assert link["overlap_lower_bound"] < .9 and link["status"] == "unknown"
