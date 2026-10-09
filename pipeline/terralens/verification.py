"""Immutable evidence receipts and conservative audits of supplied geometries.

This module checks evidence consistency and geometric sensitivity. It does not
calibrate a detector, prove that a detection is correct, or estimate registration
error. Passing geometric clauses are conditional on the recorded geometries and,
for detection boxes, on the object being contained by its box.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from typing import Any

from pyproj import CRS, Transformer
from pyproj.exceptions import ProjError
from shapely.errors import ShapelyError
from shapely.geometry import Point, shape
from shapely.ops import transform

VERSION = "terralens-geometric-verification-v1"
_COLLECTIONS = ("features", "uncertain_candidates", "tile_candidates", "context_features", "verification_excluded")


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def evidence_receipt(result: dict) -> dict:
    """Hash the query, actual feature geometries, conditions and provenance.

    The receipt detects changes to recorded evidence; it is not a signature or
    proof of correctness. Runtime timings and display pagination are excluded.
    The evidence arrays are canonically sorted without modifying the input.
    """
    evidence = {key: result.get(key) for key in (
        "plan", "dataset_version", "mode", "result_kind", "analysis_crs", "coverage",
        "provenance", "grounding", "verification_policy")}
    for collection in _COLLECTIONS:
        evidence[collection] = sorted(deepcopy(result.get(collection, [])), key=_canonical)
    checksum = hashlib.sha256(_canonical(evidence)).hexdigest()
    return {"receipt_id": "sha256:" + checksum, "sha256": checksum, "algorithm": "SHA-256",
        "canonicalization": "UTF-8 JSON; sorted object keys and evidence arrays; exact recorded coordinate values",
        "schema": VERSION, "dataset_version": result.get("dataset_version"),
        "scope": "Recorded query, geometries, conditions, source/model/prompt metadata and provenance.",
        "meaning": "Evidence fingerprint; not a digital signature, detector calibration or correctness certificate."}


def _state(conditions: list[dict]) -> str:
    statuses = {condition.get("status", "unknown") for condition in conditions}
    return "fail" if "fail" in statuses else "unknown" if "unknown" in statuses else "pass"


def _registration_error(value: float | None) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError("registration_error_m must be a finite, nonnegative combined positional allowance, or None.")
    return float(value)


def _projection(result: dict):
    try:
        crs = CRS.from_user_input(result.get("analysis_crs"))
        if not crs.is_projected or any(abs(axis.unit_conversion_factor - 1) > 1e-9 for axis in crs.axis_info[:2]):
            return None
        return Transformer.from_crs("EPSG:4326", crs, always_xy=True).transform
    except (ValueError, TypeError, ProjError):
        return None


def _metric_geometry(feature: dict, project):
    if project is None or not feature.get("geometry"):
        return None
    try:
        geometry = transform(project, shape(feature["geometry"]))
        if geometry.is_empty or not geometry.is_valid or not all(math.isfinite(value) for value in geometry.bounds):
            return None
        return geometry
    except (ValueError, TypeError, KeyError, ShapelyError, ProjError):
        return None


def _metadata(result: dict, layer: str | None) -> dict:
    return result.get("provenance", {}).get("vector_layers", {}).get(layer, {})


def _partial_reference(result: dict, condition: dict) -> bool:
    metadata = _metadata(result, condition.get("reference_layer"))
    nested = metadata.get("metadata", {})
    coverage = condition.get("reference_coverage_status") or metadata.get("coverage_status") or nested.get("coverage_status")
    complete = metadata.get("reference_coverage_complete", nested.get("reference_coverage_complete"))
    return complete is False or coverage is not None and "partial" in str(coverage).lower()


def _reference_index(result: dict) -> dict[str, dict]:
    references = {}
    for feature in result.get("context_features", []):
        properties = feature.get("properties", {})
        layer, identifier = properties.get("layer"), properties.get("id")
        if layer is not None and identifier is not None:
            references[f"{layer}:{identifier}"] = feature
        if feature.get("id") is not None:
            references[str(feature["id"])] = feature
    return references


def _distance_bounds(subject, reference, envelope: bool) -> tuple[float, float]:
    lower = float(subject.distance(reference))
    if not envelope:
        return lower, lower
    # The projected bounding rectangle covers the full projected detection
    # polygon. A ball through its four corners covers every possible object
    # position inside it; triangle inequality bounds distance to the reference.
    center = subject.representative_point()
    left, bottom, right, top = subject.bounds
    corners = (Point(left, bottom), Point(left, top), Point(right, bottom), Point(right, top))
    covering_radius = max(float(center.distance(corner)) for corner in corners)
    upper = float(center.distance(reference)) + covering_radius
    return lower, upper


def _witness_audit(subject, reference, condition: dict, identifier: str, candidate: dict,
        *, envelope: bool, error: float | None) -> dict:
    attributes = deepcopy(candidate.get("attributes", []))
    attribute_status = _state(attributes)
    radius = condition.get("radius_m")
    lower, upper = _distance_bounds(subject, reference, envelope)
    allowance = error if error is not None else 0.0
    effective_lower, effective_upper = max(0.0, lower - allowance), upper + allowance
    if condition.get("condition") == "within_distance":
        if isinstance(radius, bool) or not isinstance(radius, (int, float)) or not math.isfinite(radius) or radius < 0:
            geometric_status, reason = "unknown", "A finite radius is required for geometric verification."
        elif effective_lower > radius:
            geometric_status, reason = "fail", "Even the minimum possible distance exceeds the radius."
        elif effective_upper <= radius:
            geometric_status, reason = "pass", "The conservative upper distance bound is within the radius."
        else:
            geometric_status, reason = "unknown", "The distance interval straddles the radius; envelope proximity does not establish object proximity."
    else:
        if envelope:
            if effective_lower > 0:
                geometric_status, reason = "fail", "The supplied detection envelope cannot reach this reference within the allowance."
            elif (reference.buffer(-allowance).covers(subject) if allowance else reference.covers(subject)):
                geometric_status, reason = "pass", "The entire supplied envelope is covered by the reference geometry."
            else:
                geometric_status, reason = "unknown", "Envelope intersection does not establish that the detected object intersects the reference."
        elif not allowance:
            geometric_status = "pass" if subject.intersects(reference) else "fail"
            reason = "Intersection checked against the supplied object geometry; registration error is not inferred."
        elif subject.distance(reference) > allowance:
            geometric_status, reason = "fail", "The geometry separation exceeds the supplied positional allowance."
        else:
            geometric_status, reason = "unknown", "Intersection may change under the supplied positional allowance."
    status = _state(attributes + [{"status": geometric_status}])
    if condition.get("condition") == "within_distance" and isinstance(radius, (int, float)) and not isinstance(radius, bool):
        sensitivity = {"guaranteed_inside_from_radius_m": effective_upper,
            "guaranteed_outside_below_radius_m": effective_lower,
            "additional_error_before_geometric_status_change_m": (
                max(0.0, radius - effective_upper) if geometric_status == "pass" else
                max(0.0, effective_lower - radius) if geometric_status == "fail" else 0.0)}
    else:
        sensitivity = None
    return {"reference_feature_id": identifier, "status": status, "geometric_status": geometric_status,
        "attribute_status": attribute_status, "attribute_conditions": attributes,
        "distance_interval_m": [lower, upper],
        "registration_adjusted_interval_m": [effective_lower, effective_upper] if error is not None else None,
        "tested_interval_m": [effective_lower, effective_upper], "radius_m": radius,
        "distance_basis": "object_inside_detection_envelope" if envelope else "supplied_object_geometry",
        "registration_error_m": error, "registration_status": "not_estimated" if error is None else "supplied_assumption",
        "sensitivity": sensitivity, "reason": reason}


def _audit_spatial(result: dict, feature: dict, condition: dict, references: dict, project, error: float | None) -> dict:
    audited = deepcopy(condition)
    input_status = condition.get("status", "unknown")
    properties = feature.get("properties", {})
    envelope = properties.get("geometry_role") == "visual_detection_envelope" and properties.get("association") != "unique"
    subject = _metric_geometry(feature, project)
    candidates = {str(item.get("id")): item for item in condition.get("reference_candidates", [])}
    identifiers = [str(identifier) for identifier in condition.get("reference_feature_ids", [])]
    witness_audits, missing = [], []
    for identifier in identifiers:
        reference_feature = references.get(f"{condition.get('reference_layer')}:{identifier}")
        reference = _metric_geometry(reference_feature or {}, project)
        if subject is None or reference is None:
            missing.append(identifier)
            witness_audits.append({"reference_feature_id": identifier, "status": "unknown",
                "reason": "Valid subject/reference geometry and a projected metre CRS are required."})
            continue
        witness_audits.append(_witness_audit(subject, reference, condition, identifier,
            candidates.get(identifier, {}), envelope=envelope, error=error))
    states = {item["status"] for item in witness_audits}
    partial = _partial_reference(result, condition)
    # A relation is existential: one fully supported reference is sufficient.
    # Missing or partial references cannot establish a global negative answer.
    verified_status = "pass" if "pass" in states else "unknown" if "unknown" in states or partial or not witness_audits else "fail"
    if input_status == "fail":
        verified_status = "unknown" if partial else "fail"
    elif input_status == "unknown" and verified_status == "pass":
        # Geometry cannot repair a previously missing attribute, assumed error
        # allowance, or other unresolved clause in the source execution.
        verified_status = "unknown"
    if properties.get("association") in {"unmatched", "ambiguous"}:
        verified_status = "unknown"
    audited["status"] = verified_status
    audited["verification"] = {"input_status": input_status, "status": verified_status,
        "reference_layer": condition.get("reference_layer"), "witnesses": witness_audits,
        "supported_reference_feature_ids": [item["reference_feature_id"] for item in witness_audits if item["status"] == "pass"],
        "uncertain_reference_feature_ids": [item["reference_feature_id"] for item in witness_audits if item["status"] == "unknown"],
        "missing_reference_geometry_ids": missing, "partial_reference_coverage": partial,
        "geometry_role": properties.get("geometry_role"), "registration_error_m": error,
        "registration_status": "not_estimated" if error is None else "supplied_assumption",
        "assumption": "The true object is contained within the detection envelope." if envelope else "The recorded object geometry represents the requested subject.",
        "note": "Geometric evidence is conditional on supplied coordinates; model correctness and map completeness are not established."}
    return audited


def _evidence_kinds(result: dict, feature: dict) -> list[str]:
    properties = feature.get("properties", {})
    conditions = properties.get("conditions", [])
    kinds = set()
    if any(item.get("condition") == "visual_detection" for item in conditions) or properties.get("model_id"):
        kinds.add("model_observation")
    subject = properties.get("subject_attributes") or properties.get("source_attributes") or {}
    layer = properties.get("subject_layer") or properties.get("association_layer")
    metadata = _metadata(result, layer)
    evidence_type = properties.get("evidence_type") or subject.get("evidence_type") or metadata.get("evidence_type") or metadata.get("metadata", {}).get("evidence_type")
    if evidence_type == "dataset_annotation":
        kinds.add("dataset_annotation")
    elif subject or any(item.get("condition") == "vector_source" for item in conditions):
        kinds.add("source_vector")
    return sorted(kinds or {"recorded_evidence"})


def _complete_result(result: dict) -> bool:
    if result.get("truncated") or result.get("uncertain_truncated"):
        return False
    if result.get("total_matches") != len(result.get("features", [])):
        return False
    if result.get("total_uncertain") != len(result.get("uncertain_candidates", [])):
        return False
    if result.get("result_kind") == "tile_retrieval" and result.get("total_tile_candidates") != len(result.get("tile_candidates", [])):
        return False
    return True


def verify_result(result: dict, *, registration_error_m: float | None = None) -> dict:
    """Return an audit; leave the supplied engine result completely unchanged."""
    error = _registration_error(registration_error_m)
    references = _reference_index(result)
    project = _projection(result)
    features = []
    for collection in ("features", "uncertain_candidates"):
        for feature in result.get(collection, []):
            properties = feature.get("properties", {})
            conditions = []
            for condition in properties.get("conditions", []):
                if condition.get("condition") in {"within_distance", "intersects"}:
                    audited = _audit_spatial(result, feature, condition, references, project, error)
                else:
                    audited = deepcopy(condition)
                    audited["verification"] = {"status": condition.get("status", "unknown"),
                        "evidence_kind": "model_observation" if condition.get("condition") == "visual_detection" else "recorded_source_condition",
                        "note": "Screened model observation; score is not a probability and correctness has not been independently established."
                            if condition.get("condition") == "visual_detection" else "Retained recorded condition; this geometric audit does not add missing source evidence."}
                conditions.append(audited)
            source_ids = {str(value) for value in [properties.get("source_image"), properties.get("source_sha256"),
                properties.get("source_feature_id"), properties.get("id"), *properties.get("tile_ids", []),
                *properties.get("detection_ids", [])] if value is not None}
            features.append({"feature_id": feature.get("id"), "input_collection": collection,
                "status": _state(conditions) if conditions else "unknown", "evidence_kinds": _evidence_kinds(result, feature),
                "source_ids": sorted(source_ids), "model_id": properties.get("model_id"),
                "model_revision": properties.get("model_revision"), "model_prompt": properties.get("model_prompt"),
                "conditions": conditions, "supported_conditions": [item for item in conditions if item.get("status") == "pass"],
                "uncertain_conditions": [item for item in conditions if item.get("status") == "unknown"],
                "failed_conditions": [item for item in conditions if item.get("status") == "fail"]})
    return {"version": VERSION, "scope": "complete_recorded_result" if _complete_result(result) else "returned_page_only",
        "receipt": evidence_receipt(result), "registration_error_m": error,
        "registration_status": "not_estimated" if error is None else "supplied_assumption",
        "calibration": {"status": "uncalibrated", "score_meaning": "Model scores and text similarity are not probabilities.",
            "evidence": "No empirical detector calibration is performed by this audit."},
        "features": features, "counts": {status: sum(item["status"] == status for item in features) for status in ("pass", "unknown", "fail")},
        "assumptions": ["Detection envelopes contain their intended objects.",
            "Source coordinates and the selected projected CRS represent the recorded area.",
            "Any supplied registration allowance bounds combined relative imagery/vector position error; it is not learned or measured here."],
        "limitations": ["A receipt detects evidence changes; it does not prove source authenticity or model correctness.",
            "Unestimated registration error stays unknown; no numerical allowance is invented.",
            "Missing/partial reference coverage cannot establish real-world absence.",
            "Envelope intersection or a small minimum envelope distance alone does not establish the requested object relationship."]}


def apply_verification(result: dict, registration_error_m: float | None = None) -> dict:
    """Copy and partition audited rows; recompute totals only for complete results.

    For paginated inputs, original aggregate totals remain unchanged and the
    returned-page audit counts are available under ``verification.counts``.
    The API should use complete engine execution before auditing then paginate.
    """
    verified = deepcopy(result)
    audit = verify_result(result, registration_error_m=registration_error_m)
    records, uncertain, excluded = [], [], []
    original_features = [*verified.get("features", []), *verified.get("uncertain_candidates", [])]
    for feature, item in zip(original_features, audit["features"]):
        properties = feature.setdefault("properties", {})
        properties["conditions"] = deepcopy(item["conditions"])
        properties["verification_status"] = item["status"]
        properties["calibration_status"] = "uncalibrated"
        properties["registration_error_status"] = audit["registration_status"]
        if item["status"] == "pass":
            properties.update(role="match", evidence_status="supported_by_available_evidence")
            properties["reason"] = None
            records.append(feature)
        elif item["status"] == "unknown":
            properties.update(role="uncertain_candidate", evidence_status="unknown")
            properties["reason"] = "; ".join(
                str(condition.get("condition", "condition")) + " remains unresolved after evidence verification"
                for condition in item["uncertain_conditions"])
            uncertain.append(feature)
        else:
            properties.update(role="verification_excluded", evidence_status="failed_geometric_verification")
            properties["reason"] = "; ".join(
                str(condition.get("condition", "condition")) + " fails evidence verification"
                for condition in item["failed_conditions"])
            excluded.append(feature)
    verified.update(features=records, uncertain_candidates=uncertain, verification_excluded=excluded,
        verification=audit, verification_policy={"version": VERSION, "registration_error_m": audit["registration_error_m"],
            "registration_status": audit["registration_status"]})
    if _complete_result(result):
        verified["engine_totals_before_verification"] = {key: result.get(key) for key in ("total_matches", "total_uncertain", "objects_excluded")}
        verified.update(total_matches=len(records), total_uncertain=len(uncertain),
            objects_excluded=int(result.get("objects_excluded", 0)) + len(excluded),
            verification_totals_status="recomputed_from_complete_recorded_result")
    else:
        verified["verification_totals_status"] = "original_aggregate_totals_preserved_for_unseen_pages"
    verified["evidence_receipt"] = evidence_receipt(verified)
    return verified
