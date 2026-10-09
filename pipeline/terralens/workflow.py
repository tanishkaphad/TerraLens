"""Typed, bounded GIS workflows over registered evidence. No generated code runs."""
from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from collections import Counter
from time import perf_counter
from typing import Callable, Literal

import geopandas as gpd
from pydantic import BaseModel, ConfigDict, Field, field_validator
from pyproj import CRS
from pyproj.exceptions import ProjError
from shapely.geometry import Point, box, mapping, shape
from shapely.errors import ShapelyError
from shapely.ops import unary_union

from .bundle import json_value
from .planner import QueryPlan, PlanError, validate_plan


class WorkflowError(ValueError):
    pass


class MissingEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")
    category: Literal["data", "tool", "resolution", "time", "clarification", "model"]
    message: str = Field(min_length=1, max_length=500)


class Requirement(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str = Field(min_length=1, max_length=500)
    status: Literal["planned", "missing"]
    step_ids: list[str] = Field(max_length=6)


class WorkflowStep(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(pattern=r"^[a-z][a-z0-9_]{0,31}$")
    operation: Literal["search", "spatial_filter", "aggregate", "coverage"]
    label: str = Field(min_length=1, max_length=160)
    dataset_id: str | None = None
    plan: QueryPlan | None = None
    mode: Literal["catalog_scan", "explore", "ground"] | None = None
    input_id: str | None = None
    reference_id: str | None = None
    predicate: Literal["within_distance", "intersects"] | None = None
    distance_m: float | None = Field(default=None, gt=0, le=5000, allow_inf_nan=False, strict=True)
    group_by: str | None = Field(default=None, max_length=80)
    metric: Literal["count", "area_m2", "length_m"] | None = None
    tile_limit: int | None = Field(default=None, ge=1, le=4, strict=True)
    bbox: list[float] | None = Field(default=None, min_length=4, max_length=4)

    @field_validator("bbox", mode="before")
    @classmethod
    def finite_numeric_bbox(cls, value):
        if value is not None and (not isinstance(value, list) or any(
                isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(item) for item in value)):
            raise ValueError("Area coordinates must be explicit finite numbers.")
        return value

    @field_validator("plan", mode="before")
    @classmethod
    def explicit_plan_distances(cls, value):
        if isinstance(value, dict):
            for relation in [value, *value.get("relations", [])]:
                distance = relation.get("distance_m")
                if distance is not None and (isinstance(distance, bool) or not isinstance(distance, (int, float)) or not math.isfinite(distance)):
                    raise ValueError("Query distances must be explicit finite numeric metre values.")
        return value


class Workflow(BaseModel):
    model_config = ConfigDict(extra="forbid")
    title: str = Field(min_length=1, max_length=160)
    status: Literal["ready", "needs_input", "unsupported"]
    requirements: list[Requirement] = Field(min_length=1, max_length=24)
    missing_evidence: list[MissingEvidence] = Field(default_factory=list, max_length=12)
    steps: list[WorkflowStep] = Field(default_factory=list, max_length=6)


TOOLS = {
    "search": {"id": "search", "description": "Search vector attributes, cached detections, SkyCLIP image scenes, or new OWLv2 visual descriptions using a validated QueryPlan. Select one registered dataset; optional WGS84 bbox filters results. Scores are not probabilities."},
    "spatial_filter": {"id": "spatial_filter", "description": "Join two earlier object/vector search results with within_distance or intersects. Can combine independently grounded visual concepts. Tile similarity cannot be used as object geometry. Distances use projected metre coordinates."},
    "aggregate": {"id": "aggregate", "description": "Count, sum geometric area, or sum geometric length of earlier supported results, optionally grouped by an available source attribute. Produce a table and chart. Counts describe records, not true population estimates."},
    "coverage": {"id": "coverage", "description": "Inspect a registered dataset's spatial coverage, image dimensions and dates, vector fields, model capabilities and evidence limitations. No geographic claims beyond that inventory."},
}
MAX_SPATIAL_PAIRS = 100000


def strict_schema() -> dict:
    schema = Workflow.model_json_schema()
    def visit(item):
        if isinstance(item, dict):
            item.pop("default", None)
            if item.get("type") == "object":
                item["additionalProperties"] = False
                item["required"] = list(item.get("properties", {}))
            for value in item.values():
                visit(value)
        elif isinstance(item, list):
            for value in item:
                visit(value)
    visit(schema)
    return schema


def _coverage_bounds(coverage: dict) -> list[float] | None:
    value = coverage.get("bbox_wgs84")
    if not isinstance(value, (list, tuple)) or len(value) != 4 or any(
            isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(item) for item in value):
        return None
    west, south, east, north = value
    return list(value) if -180 <= west < east <= 180 and -90 <= south < north <= 90 else None


def _maximum_radius(manifest: dict) -> float:
    value = manifest.get("limits", {}).get("maximum_distance_m", 5000)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise WorkflowError("The dataset distance limit is invalid.")
    return min(5000.0, float(value))


def _ancestors(identifier: str, seen: dict) -> set[str]:
    result, pending = set(), [identifier]
    while pending:
        current = pending.pop()
        if current in result:
            continue
        result.add(current)
        step = seen[current]
        pending.extend(value for value in (step.input_id, step.reference_id) if value is not None)
    return result


def _metric_crs(records: list[dict], configured=None):
    try:
        if configured:
            crs = CRS.from_user_input(configured)
        elif records:
            series = gpd.GeoSeries([shape(item["geometry"]) for item in records], crs="EPSG:4326")
            bounds = series.total_bounds
            if not all(math.isfinite(value) for value in bounds) or bounds[2] - bounds[0] > 6 or bounds[3] - bounds[1] > 6:
                raise WorkflowError("These geometries are too broad for one local metric analysis.")
            estimated = series.estimate_utm_crs()
            if estimated is None:
                raise WorkflowError("Could not choose a local metric coordinate system.")
            crs = CRS.from_user_input(estimated)
        else:
            raise WorkflowError("A projected metre coordinate system is required.")
        if not crs.is_projected or any(abs(axis.unit_conversion_factor - 1) > 1e-9 for axis in crs.axis_info[:2]):
            raise WorkflowError("Spatial metrics require a projected coordinate system with metre units.")
        return crs
    except (ValueError, TypeError, KeyError, ProjError) as exc:
        if isinstance(exc, WorkflowError):
            raise
        raise WorkflowError("The metric coordinate system or source geometry is invalid.") from exc


def _areas_compatible(first: list[float], second: list[float], distance: float) -> bool:
    if box(*first).intersects(box(*second)):
        return True
    if not distance:
        return False
    if abs((first[0] + first[2] - second[0] - second[2]) / 2) > 1 or abs((first[1] + first[3] - second[1] - second[3]) / 2) > 1:
        return False
    shapes = gpd.GeoSeries([box(*first), box(*second)], crs="EPSG:4326")
    projected = shapes.to_crs(shapes.estimate_utm_crs())
    return projected.iloc[0].distance(projected.iloc[1]) <= distance


def validate_workflow(raw, catalog: dict) -> Workflow:
    workflow = Workflow.model_validate(raw)
    if workflow.status != "ready":
        if workflow.steps or not workflow.missing_evidence or any(item.step_ids for item in workflow.requirements):
            raise WorkflowError("An unsupported or ambiguous request must identify missing evidence and execute no steps.")
        return workflow
    if not workflow.steps or workflow.missing_evidence or any(item.status == "missing" for item in workflow.requirements):
        raise WorkflowError("Every requested condition needs a complete workflow before execution.")
    seen = {}
    origins = {}
    detection_budget = 0
    for step in workflow.steps:
        if step.id in seen or step.operation not in TOOLS:
            raise WorkflowError("Step identifiers must be unique and operations must be registered.")
        allowed_fields = {
            "search": {"dataset_id", "plan", "mode", "bbox", "tile_limit"},
            "coverage": {"dataset_id"},
            "spatial_filter": {"input_id", "reference_id", "predicate", "distance_m"},
            "aggregate": {"input_id", "group_by", "metric"},
        }[step.operation]
        for field in ("dataset_id", "plan", "mode", "input_id", "reference_id", "predicate", "distance_m", "group_by", "metric", "tile_limit", "bbox"):
            if getattr(step, field) is not None and field not in allowed_fields:
                raise WorkflowError(f"Field '{field}' is not used by operation '{step.operation}'; it must be null.")
        if step.operation in {"search", "coverage"}:
            if step.dataset_id not in catalog:
                raise WorkflowError("The workflow selected an unavailable dataset.")
            if step.input_id or step.reference_id:
                raise WorkflowError("Source operations cannot depend on result identifiers.")
            origins[step.id] = {step.dataset_id}
        if step.operation == "search":
            if step.plan is None or step.mode is None:
                raise WorkflowError("Every search needs an explicit validated plan and execution mode.")
            validate_plan(step.plan, catalog[step.dataset_id]["manifest"], step.mode)
            if step.tile_limit is not None and step.mode != "ground":
                raise WorkflowError("A tile limit belongs only to live grounding searches.")
            if step.mode == "ground":
                if not catalog[step.dataset_id].get("live_grounding"):
                    raise WorkflowError("The selected dataset has no usable local grounding model.")
                detection_budget += step.tile_limit or 2
            if step.mode == "explore" and not catalog[step.dataset_id].get("retrieval"):
                raise WorkflowError("Image discovery needs the matching local text encoder.")
            if step.bbox:
                west, south, east, north = step.bbox
                if not all(math.isfinite(value) for value in step.bbox) or not (-180 <= west < east <= 180 and -90 <= south < north <= 90):
                    raise WorkflowError("The requested area must be finite west,south,east,north coordinates.")
                coverage = _coverage_bounds(catalog[step.dataset_id]["manifest"].get("coverage", {}))
                if coverage is None or not box(*coverage).covers(box(*step.bbox)):
                    raise WorkflowError("The complete requested area must be inside the selected imagery coverage.")
        elif step.operation in {"spatial_filter", "aggregate"}:
            if step.input_id not in seen or seen[step.input_id].operation not in {"search", "spatial_filter"}:
                raise WorkflowError("Analysis must reference an earlier geographic result.")
            source = seen[step.input_id]
            if source.mode == "explore" or any(seen[identifier].mode == "explore" for identifier in _ancestors(source.id, seen)):
                raise WorkflowError("Image similarity tiles cannot be counted or joined as grounded objects.")
            origins[step.id] = set(origins[step.input_id])
            if step.operation == "spatial_filter":
                if step.reference_id not in seen or seen[step.reference_id].operation not in {"search", "spatial_filter"}:
                    raise WorkflowError("A spatial join requires an earlier geographic reference result.")
                if seen[step.reference_id].mode == "explore" or step.reference_id == step.input_id:
                    raise WorkflowError("Use distinct object/vector sources for the spatial join.")
                if step.predicate is None or (step.predicate == "within_distance") != (step.distance_m is not None):
                    raise WorkflowError("Distance joins need a radius; intersection joins do not.")
                origins[step.id].update(origins[step.reference_id])
                maximum = min(_maximum_radius(catalog[identifier]["manifest"]) for identifier in origins[step.id])
                if step.distance_m is not None and step.distance_m > maximum:
                    raise WorkflowError(f"The spatial join exceeds its source datasets' {maximum:g} metre distance limit.")
                source_areas = [_coverage_bounds(catalog[identifier]["manifest"].get("coverage", {})) for identifier in origins[step.input_id]]
                reference_areas = [_coverage_bounds(catalog[identifier]["manifest"].get("coverage", {})) for identifier in origins[step.reference_id]]
                if any(area is None for area in [*source_areas, *reference_areas]):
                    raise WorkflowError("Both spatial sources need recorded geographic coverage.")
                if not any(_areas_compatible(area, other, step.distance_m or 0) for area in source_areas for other in reference_areas):
                    raise WorkflowError("The selected datasets cover different areas outside the requested relationship.")
            elif step.metric is None:
                raise WorkflowError("Choose count, area_m2 or length_m for aggregation.")
        seen[step.id] = step
    if detection_budget > 6:
        raise WorkflowError("This workflow exceeds the six-tile live detection budget; narrow the area or split the request.")
    for requirement in workflow.requirements:
        if not requirement.step_ids or any(identifier not in seen for identifier in requirement.step_ids):
            raise WorkflowError("Every condition must cite existing execution steps.")
    used = {identifier for item in workflow.requirements for identifier in item.step_ids}
    if used != set(seen):
        raise WorkflowError("Every workflow step must serve a declared user requirement.")
    return workflow


def guard_numeric_conditions(message: str, workflow: Workflow) -> None:
    """Preserve each explicit distance occurrence, including its source units."""
    scales = {"m": 1, "metre": 1, "meter": 1, "metres": 1, "meters": 1,
        "km": 1000, "kilometre": 1000, "kilometer": 1000, "kilometres": 1000, "kilometers": 1000,
        "cm": 0.01, "centimetre": 0.01, "centimeter": 0.01, "centimetres": 0.01, "centimeters": 0.01,
        "mm": 0.001, "millimetre": 0.001, "millimeter": 0.001, "millimetres": 0.001, "millimeters": 0.001,
        "ft": 0.3048, "foot": 0.3048, "feet": 0.3048, "yd": 0.9144, "yard": 0.9144, "yards": 0.9144,
        "mi": 1609.344, "mile": 1609.344, "miles": 1609.344}
    units = "|".join(sorted(scales, key=len, reverse=True))
    message = re.sub(r"\b(within|radius|distance|under|over|below|above)(?=[+-]?(?:\d|\.\d))", r"\1 ", message, flags=re.I)
    expressions = re.findall(rf"(?<![\w.])([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?)\s*({units})\b(?![²³]|\^\s*[23])", message, re.I)
    requested = [float(number) * scales[unit.lower()] for number, unit in expressions]
    planned = []
    for step in workflow.steps:
        if step.distance_m is not None:
            planned.append(step.distance_m)
        if step.plan:
            if step.plan.distance_m is not None:
                planned.append(step.plan.distance_m)
            planned.extend(relation.distance_m for relation in step.plan.relations if relation.distance_m is not None)
    if workflow.status == "ready":
        remaining = list(planned)
        for value in requested:
            if not math.isfinite(value) or value <= 0:
                raise WorkflowError("An explicitly requested distance must be finite and positive.")
            match = next((index for index, actual in enumerate(remaining) if math.isclose(value, actual, rel_tol=1e-9, abs_tol=1e-9)), None)
            if match is None:
                raise WorkflowError("An explicitly requested distance was dropped, changed or reused for multiple clauses by the proposed workflow.")
            remaining.pop(match)


def _records(result):
    return result.get("features", [])


def _complete_geographic_result(result: dict) -> None:
    if result.get("result_kind") == "tile_retrieval":
        raise WorkflowError("Image similarity tiles are not grounded object records.")
    if result.get("truncated") or result.get("uncertain_truncated") or result.get("total_matches") != len(result.get("features", [])) or result.get("total_uncertain") != len(result.get("uncertain_candidates", [])):
        raise WorkflowError("Workflow analysis requires complete geographic result sets, not unseen pages.")


def _source_attributes(feature: dict) -> dict:
    properties = feature.get("properties", {})
    return properties.get("subject_attributes") or properties.get("source_attributes") or {}


def _valid_geometries(records: list[dict]) -> list:
    try:
        geometries = [shape(item["geometry"]) for item in records]
        if any(geometry.is_empty or not geometry.is_valid or not all(math.isfinite(value) for value in geometry.bounds) for geometry in geometries):
            raise WorkflowError("Source geometries must be valid, nonempty and finite.")
        return geometries
    except (ValueError, TypeError, KeyError, ShapelyError) as exc:
        if isinstance(exc, WorkflowError):
            raise
        raise WorkflowError("A geographic source has unreadable geometry.") from exc


def aggregate_result(result: dict, metric: str, group_by: str | None) -> dict:
    _complete_geographic_result(result)
    if metric not in {"count", "area_m2", "length_m"}:
        raise WorkflowError("Choose count, area_m2 or length_m for aggregation.")
    records = _records(result)
    if len(records) > 10000:
        raise WorkflowError("Aggregate input exceeds the 10,000-feature budget.")
    groups = Counter()
    fields = {key for feature in records for key in _source_attributes(feature)}
    layer = result.get("plan", {}).get("subject_layer")
    fields.update(result.get("provenance", {}).get("vector_layers", {}).get(layer, {}).get("fields", []))
    if group_by and group_by not in fields:
        raise WorkflowError(f"Grouping attribute '{group_by}' is unavailable in these results.")
    geometries = _valid_geometries(records)
    if metric != "count":
        expected = {"Polygon", "MultiPolygon"} if metric == "area_m2" else {"LineString", "MultiLineString"}
        if any(feature["geometry"]["type"] not in expected for feature in records):
            raise WorkflowError("The chosen metric does not match the supplied geometry types.")
        if any(feature["properties"].get("geometry_role") in {"visual_detection_envelope", "image_tile_coverage"} for feature in records):
            raise WorkflowError("Detection boxes and image extents do not establish true object area or length.")
        frame = gpd.GeoSeries(geometries, crs="EPSG:4326")
        if records:
            metric_frame = frame.to_crs(_metric_crs(records, result.get("analysis_crs")))
            values = metric_frame.area if metric == "area_m2" else metric_frame.length
        else:
            values = []
    else:
        values = [1] * len(records)
    for feature, value in zip(records, values):
        raw = _source_attributes(feature).get(group_by) if group_by else "All supported records"
        if raw is not None and not isinstance(raw, (str, int, float, bool)):
            raise WorkflowError("Grouping requires a scalar source attribute, not an annotation object.")
        if isinstance(raw, float) and not math.isfinite(raw):
            raw = None
        group = "Unknown / missing" if raw is None else str(raw)
        groups[group] += float(value)
    ordered = sorted(groups.items(), key=lambda pair: (-pair[1], pair[0]))
    if len(ordered) > 50:
        ordered = ordered[:49] + [("Other groups", sum(value for _, value in ordered[49:]))]
    rows = [{"group": group, metric: int(value) if metric == "count" else round(value, 3)} for group, value in ordered]
    return {"rows": rows, "chart": {"labels": [row["group"] for row in rows], "values": [row[metric] for row in rows], "unit": metric},
        "supported_records": len(records), "uncertain_records_excluded": len(result.get("uncertain_candidates", [])),
        "scope": result.get("claim_scope"), "metric": metric,
        "note": "Sum of source geometries, including any overlaps." if metric != "count" else "Count of supported evidence records; not an estimate of all real-world objects."}


def _condition_state(conditions: list[dict]) -> str:
    states = {item.get("status", "unknown") for item in conditions}
    return "fail" if "fail" in states else "unknown" if "unknown" in states else "pass"


def _object_identity(feature: dict, result: dict):
    properties = feature.get("properties", {})
    attributes = _source_attributes(feature)
    identifier = attributes.get("source_feature_id") or properties.get("source_feature_id")
    if identifier:
        return ("source", str(identifier))
    return (result.get("dataset_version"), properties.get("association_layer") or properties.get("subject_layer"), str(feature.get("id")))


def _envelope(feature: dict) -> bool:
    return feature.get("properties", {}).get("geometry_role") == "visual_detection_envelope" and feature.get("properties", {}).get("association") != "unique"


def _covering_radius(geometry) -> tuple:
    center = geometry.representative_point()
    left, bottom, right, top = geometry.bounds
    return center, max(center.distance(Point(x, y)) for x, y in ((left, bottom), (left, top), (right, bottom), (right, top)))


def _join_interval(subject, reference, subject_box: bool, reference_box: bool) -> tuple[float, float]:
    lower = float(subject.distance(reference))
    if subject_box and reference_box:
        first, first_radius = _covering_radius(subject)
        second, second_radius = _covering_radius(reference)
        upper = first.distance(second) + first_radius + second_radius
    elif subject_box:
        center, radius = _covering_radius(subject)
        upper = center.distance(reference) + radius
    elif reference_box:
        center, radius = _covering_radius(reference)
        upper = subject.distance(center) + radius
    else:
        upper = lower
    return lower, float(upper)


def spatial_filter(subjects: dict, references: dict, predicate: str, distance: float | None) -> dict:
    """Join supplied evidence sets with distance bounds for either object's box."""
    from .verification import apply_verification, evidence_receipt
    _complete_geographic_result(subjects)
    _complete_geographic_result(references)
    if predicate not in {"within_distance", "intersects"} or (predicate == "within_distance") != (distance is not None):
        raise WorkflowError("Distance joins need a radius; intersection joins do not.")
    if distance is not None and (isinstance(distance, bool) or not isinstance(distance, (int, float)) or not math.isfinite(distance) or not 0 < distance <= 5000):
        raise WorkflowError("Join radius must be a finite positive distance of at most 5000 metres.")
    result = copy.deepcopy(subjects)
    source_records = [*result.get("features", []), *result.get("uncertain_candidates", [])]
    ref_records = [*references.get("features", []), *references.get("uncertain_candidates", [])]
    if len(source_records) > 10000 or len(ref_records) > 10000:
        raise WorkflowError("Spatial inputs exceed the 10,000-feature budget.")
    coverage, other_coverage = _coverage_bounds(subjects.get("coverage", {})), _coverage_bounds(references.get("coverage", {}))
    if coverage is None or other_coverage is None or not _areas_compatible(coverage, other_coverage, distance or 0):
        raise WorkflowError("Both source areas must be known and compatible with the requested relationship.")
    analysis_crs = _metric_crs([*source_records, *ref_records], subjects.get("analysis_crs"))
    metric_subjects = gpd.GeoSeries(_valid_geometries(source_records), crs="EPSG:4326").to_crs(analysis_crs)
    metric_refs = gpd.GeoSeries(_valid_geometries(ref_records), crs="EPSG:4326").to_crs(analysis_crs)
    subject_receipt = evidence_receipt(_receipt_content(subjects))
    reference_receipt = evidence_receipt(_receipt_content(references))
    reference_digest = reference_receipt["sha256"][:16]
    layer_id = "workflow_reference_" + reference_digest
    context = copy.deepcopy(result.get("context_features", []))
    uncertain_subject_ids = {str(feature.get("id")) for feature in subjects.get("uncertain_candidates", [])}
    uncertain_reference_ids = {str(feature.get("id")) for feature in references.get("uncertain_candidates", [])}
    supported, uncertain = [], []
    candidate_pair_count = 0
    for position, feature in enumerate(source_records):
        geometry = metric_subjects.iloc[position]
        positions = metric_refs.sindex.query(geometry.buffer(distance) if predicate == "within_distance" else geometry, predicate="intersects")
        candidate_pair_count += len(positions)
        if candidate_pair_count > MAX_SPATIAL_PAIRS:
            raise WorkflowError("The spatial join exceeds its candidate-pair budget; narrow the preceding searches.")
        witnesses = []
        for reference_position in sorted(int(item) for item in positions):
            reference = ref_records[reference_position]
            if _object_identity(feature, subjects) == _object_identity(reference, references):
                continue
            reference_geometry = metric_refs.iloc[reference_position]
            lower, upper = _join_interval(geometry, reference_geometry, _envelope(feature), _envelope(reference))
            if predicate == "within_distance":
                geometry_status = "fail" if lower > distance else "pass" if upper <= distance else "unknown"
            elif _envelope(feature) and _envelope(reference):
                geometry_status = "unknown"
            elif _envelope(feature):
                geometry_status = "pass" if reference_geometry.covers(geometry) else "unknown"
            elif _envelope(reference):
                geometry_status = "pass" if geometry.covers(reference_geometry) else "unknown"
            else:
                geometry_status = "pass" if geometry.intersects(reference_geometry) else "fail"
            upstream = _condition_state(reference.get("properties", {}).get("conditions", []))
            if str(reference.get("id")) in uncertain_reference_ids and upstream == "pass":
                upstream = "unknown"
            reference_id = str(reference_position)
            attributes = [{"condition": "workflow_reference_evidence", "status": upstream,
                "original_feature_id": reference.get("id"), "source_dataset_version": references.get("dataset_version"),
                "conditions": copy.deepcopy(reference.get("properties", {}).get("conditions", []))},
                {"condition": "workflow_geometry_interval", "status": geometry_status,
                    "distance_interval_m": [lower, upper], "subject_is_envelope": _envelope(feature),
                    "reference_is_envelope": _envelope(reference), "registration_error_status": "not_estimated",
                    "assumption": "Each detection envelope contains its intended nonempty object."}]
            witnesses.append({"id": reference_id, "original_feature_id": reference.get("id"),
                "status": _condition_state(attributes), "distance_m": lower, "distance_interval_m": [lower, upper],
                "attributes": attributes, "properties": copy.deepcopy(reference.get("properties", {}))})
            context.append({**copy.deepcopy(reference), "id": f"{layer_id}:{reference_id}", "properties": {
                **copy.deepcopy(reference.get("properties", {})), "id": reference_id, "role": "reference", "layer": layer_id,
                "original_feature_id": reference.get("id"), "source_dataset_version": references.get("dataset_version")}})
        relation_status = "pass" if any(item["status"] == "pass" for item in witnesses) else "unknown"
        properties = feature.setdefault("properties", {})
        if str(feature.get("id")) in uncertain_subject_ids and _condition_state(properties.get("conditions", [])) == "pass":
            properties.setdefault("conditions", []).append({"condition": "workflow_subject_evidence", "status": "unknown",
                "note": "The subject was unresolved in the preceding search."})
        properties.setdefault("conditions", []).append({"condition": predicate, "status": relation_status,
            "reference_layer": layer_id, "reference_feature_ids": [item["id"] for item in witnesses if item["status"] != "fail"],
            "reference_candidates": witnesses, "radius_m": distance,
            "distance_m": min((item["distance_m"] for item in witnesses), default=None),
            "reference_coverage_status": "partial_workflow_results", "note": "Join over recorded evidence sets; nonmatches do not prove municipal absence."})
        if _condition_state(properties["conditions"]) == "pass":
            supported.append(feature)
        else:
            properties.update(role="uncertain_candidate", evidence_status="unknown")
            uncertain.append(feature)
    context_by_id = {str(feature["id"]): feature for feature in context}
    result.update(features=supported, uncertain_candidates=uncertain, context_features=list(context_by_id.values()),
        total_matches=len(supported), total_uncertain=len(uncertain), analysis_crs=analysis_crs.to_string(),
        result_kind="spatial_workflow_objects", truncated=False, uncertain_truncated=False, page=0,
        workflow_sources={"subjects": subject_receipt, "references": reference_receipt},
        claim_scope="Spatial join over preceding evidence sets; geometry bounds, upstream uncertainty and incomplete sample coverage remain explicit.")
    return apply_verification(result)


def execute_workflow(workflow: Workflow, catalog: dict, search: Callable, progress: Callable = lambda *_: None) -> dict:
    from .verification import apply_verification, evidence_receipt
    workflow = validate_workflow(workflow, catalog)
    if workflow.status != "ready":
        return {"status": workflow.status, "message": " ".join(item.message for item in workflow.missing_evidence),
            "workflow": workflow.model_dump(), "artifacts": [], "tool_trace": [], "receipt": None}
    results, artifacts, trace = {}, [], []
    for step in workflow.steps:
        progress(step.id, step.label)
        started = perf_counter()
        if step.operation == "search":
            result = copy.deepcopy(search(step))
            if result.get("truncated") or result.get("uncertain_truncated") or result.get("total_matches") != len(result.get("features", [])) or result.get("total_uncertain") != len(result.get("uncertain_candidates", [])):
                raise WorkflowError("A workflow search was truncated; refusing to calculate incomplete totals.")
            expected_version = catalog[step.dataset_id]["manifest"].get("dataset_version")
            if result.get("dataset_version") != expected_version:
                raise WorkflowError("A search returned a different dataset version than its validated source.")
            coverage = _coverage_bounds(catalog[step.dataset_id]["manifest"].get("coverage", {}))
            if coverage is not None:
                source_area = box(*coverage)
                for feature in [*result.get("features", []), *result.get("uncertain_candidates", []), *result.get("tile_candidates", [])]:
                    if not shape(feature["geometry"]).intersects(source_area):
                        raise WorkflowError("A search result lies outside its selected dataset's geographic coverage.")
            if step.bbox:
                area = box(*step.bbox)
                for field in ("features", "uncertain_candidates", "tile_candidates"):
                    result[field] = [feature for feature in result.get(field, []) if shape(feature["geometry"]).intersects(area)]
                result.update(total_matches=len(result["features"]), total_uncertain=len(result["uncertain_candidates"]),
                    total_tile_candidates=len(result.get("tile_candidates", [])), requested_bbox=step.bbox)
            result = apply_verification(result)
            artifacts.append({"id": step.id, "kind": "map", "title": step.label, "result": result})
        elif step.operation == "spatial_filter":
            result = spatial_filter(results[step.input_id], results[step.reference_id], step.predicate, step.distance_m)
            artifacts.append({"id": step.id, "kind": "map", "title": step.label, "result": result})
        elif step.operation == "aggregate":
            result = aggregate_result(results[step.input_id], step.metric, step.group_by)
            artifacts.append({"id": step.id, "kind": "chart", "title": step.label, **result})
        else:
            entry = catalog[step.dataset_id]
            manifest = entry["manifest"]
            result = {"rows": [{"item": "Area", "value": manifest.get("area_name")},
                {"item": "Image tiles", "value": manifest.get("coverage", {}).get("tile_count")},
                {"item": "Imagery dates", "value": ", ".join(manifest.get("coverage", {}).get("acquisition_dates", [])) or "Unknown"},
                {"item": "Available vector layers", "value": ", ".join(manifest.get("vector_layers", {}))},
                {"item": "Live object detection", "value": entry.get("live_grounding", False)}],
                "dataset_version": manifest.get("dataset_version")}
            artifacts.append({"id": step.id, "kind": "table", "title": step.label, **result})
        results[step.id] = result
        trace.append({"step_id": step.id, "operation": step.operation, "status": "completed",
            "duration_ms": round((perf_counter() - started) * 1000, 2),
            "summary": f"{result.get('total_matches', result.get('supported_records', 0))} supported records" if step.operation != "coverage" else "Dataset inventory checked"})
    # Narration is assembled from executed facts; the planner cannot invent a result.
    messages = []
    for artifact in artifacts:
        if artifact["kind"] == "map":
            result = artifact["result"]
            if result.get("result_kind") == "tile_retrieval":
                messages.append(f"{artifact['title']}: {len(result.get('tile_candidates', []))} image candidates. These are image similarities, not object detections.")
            else:
                messages.append(f"{artifact['title']}: {result['total_matches']} supported records; {result['total_uncertain']} need review.")
        elif artifact["kind"] == "chart":
            total = sum(artifact["chart"]["values"])
            messages.append(f"{artifact['title']}: {total:g} {artifact['metric']}; {artifact['uncertain_records_excluded']} uncertain records excluded.")
        else:
            messages.append(f"{artifact['title']}: the available dataset inventory is shown below.")
    result = json_value({"status": "completed", "message": " ".join(messages), "workflow": workflow.model_dump(),
        "artifacts": artifacts, "tool_trace": trace, "requirement_evidence": _requirement_evidence(workflow, results)})
    content = _receipt_content({"workflow": result["workflow"], "artifacts": result["artifacts"], "requirement_evidence": result["requirement_evidence"]})
    digest = hashlib.sha256(json.dumps(content, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    result["receipt"] = {"id": "workflow-" + digest[:20], "sha256": digest,
        "canonicalization": "Sorted JSON object keys; operational timings and derived receipts excluded.",
        "meaning": "Content fingerprint of the executed workflow and evidence. Not a digital signature or proof of scientific correctness."}
    return result


def _receipt_content(value):
    if isinstance(value, dict):
        excluded = {"timings_ms", "duration_ms", "inference_ms", "elapsed_ms", "elapsed_seconds",
            "model_load_ms", "text_encoding_ms", "detection_ms", "preprocessing_ms", "retrieval_ms", "spatial_ms",
            "encoding_ms", "planning_ms", "cache_hit", "expires_in_seconds", "duration_seconds",
            "receipt", "evidence_receipt", "receipt_id", "sha256_receipt"}
        return {key: _receipt_content(item) for key, item in value.items() if key not in excluded}
    if isinstance(value, list):
        return [_receipt_content(item) for item in value]
    return value


def _requirement_evidence(workflow: Workflow, results: dict) -> list[dict]:
    evidence = []
    for requirement in workflow.requirements:
        steps = []
        for identifier in requirement.step_ids:
            result = results[identifier]
            if "total_matches" in result:
                supported, uncertain = result["total_matches"], result["total_uncertain"]
                status = "supported_with_uncertain_records" if supported and uncertain else "supported" if supported else "uncertain" if uncertain else "no_supported_matches_in_sample"
                if result.get("result_kind") == "tile_retrieval":
                    status = "image_candidates_require_grounding"
                steps.append({"step_id": identifier, "status": status, "supported_records": supported,
                    "uncertain_records": uncertain, "image_candidates": len(result.get("tile_candidates", [])),
                    "dataset_version": result.get("dataset_version"), "scope": result.get("claim_scope"),
                    "evidence_receipt": result.get("evidence_receipt"),
                    "conditions": [{"feature_id": feature.get("id"), "conditions": feature.get("properties", {}).get("conditions", [])}
                        for feature in [*result.get("features", []), *result.get("uncertain_candidates", [])]],
                    "note": "No supported matches describes available sample evidence; it does not establish municipal absence."})
            elif "supported_records" in result:
                steps.append({"step_id": identifier, "status": "aggregate_computed", "supported_records": result["supported_records"],
                    "uncertain_records": result["uncertain_records_excluded"], "metric": result["metric"], "rows": result["rows"], "scope": result.get("scope")})
            else:
                steps.append({"step_id": identifier, "status": "inventory_checked", "dataset_version": result.get("dataset_version")})
        statuses = {step["status"] for step in steps}
        status = "uncertain" if any(value in statuses for value in ("uncertain", "supported_with_uncertain_records", "image_candidates_require_grounding")) else "no_supported_matches_in_sample" if "no_supported_matches_in_sample" in statuses else "supported_by_executed_steps"
        evidence.append({"text": requirement.text, "status": status, "step_evidence": steps,
            "interpretation": "Execution evidence for cited steps; not independent proof that language interpretation or model detections are correct."})
    return evidence
