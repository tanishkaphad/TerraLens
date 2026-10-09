from shapely.geometry import Point, box, mapping

from terralens.search_service import SearchService


def feature(identifier, geometry, **properties):
    return {"type": "Feature", "id": identifier, "geometry": mapping(geometry), "properties": properties}


def test_counts_per_boundary_preserve_unassigned_and_ambiguous_subjects():
    boundaries = [feature("west", box(0, 0, 1, 1), ward="west"),
                  feature("east", box(1, 0, 2, 1), ward="east")]
    records = [{"id": str(index), "group": "matches", "feature": feature(str(index), Point(*point))}
               for index, point in enumerate([(0.5, .5), (1.5, .5), (1, .5), (3, .5)])]
    counts = {"matches": 4, "needs_review": 0, "visual_candidates": 0, "excluded": 0}
    result = SearchService._aggregate({"output": {"kind": "count", "group_by_entity_id": "wards",
        "group_by_field": "ward"}}, records, {"wards": boundaries}, counts)
    assert sum(row["counts"]["matches"] for row in result["rows"]) == 4
    assert {row["status"] for row in result["rows"]} == {"known", "ambiguous_boundary", "unassigned"}
    assert result["grouping_basis"] == "representative_point_in_boundary"


def test_unknown_group_field_and_unspecified_statistics_are_not_fabricated():
    records = [{"id": "subject", "group": "needs_review", "feature": feature("subject", Point(0, 0))}]
    counts = {"matches": 0, "needs_review": 1, "visual_candidates": 0, "excluded": 0}
    result = SearchService._aggregate({"output": {"kind": "count", "group_by_field": "missing"}}, records, {}, counts)
    assert result["rows"][0]["status"] == "unknown_attribute"
    unsupported = SearchService._aggregate({"output": {"kind": "statistics"}}, records, {}, counts)
    assert unsupported["missing_evidence"]
