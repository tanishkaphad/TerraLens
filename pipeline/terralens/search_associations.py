"""Explicit, versioned geometric policies for linking visual observations.

The processor records a heuristic association between exact supplied identities.
It neither merges neighbouring footprints nor learns ownership from proximity.
No policy is installed by default and no object vocabulary is built into it.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import math
from typing import Any

from pyproj import CRS, Transformer
from pyproj.exceptions import CRSError
from shapely import STRtree, transform as transform_geometry
from shapely.errors import GEOSException
from shapely.geometry import Point, box, mapping, shape

from .search_graph import feature_id


def _finite(value: Any, *, minimum=0, maximum=None) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= minimum and (maximum is None or value <= maximum)


def _policy(raw: Any) -> tuple[dict | None, str | None]:
    if not isinstance(raw, dict):
        return None, "The requested association policy is not registered."
    policy = deepcopy(raw)
    if policy.get("method") != "unique_overlap":
        return None, "No installed association processor executes the registered method."
    overlap = policy.get("minimum_overlap")
    if not _finite(overlap, minimum=0, maximum=1) or overlap == 0:
        return None, "The association policy needs a finite minimum_overlap in (0, 1]."
    for field in ("subject_geometry_roles", "reference_geometry_roles"):
        if not isinstance(policy.get(field), list) or not policy[field] or not all(isinstance(role, str) and role for role in policy[field]):
            return None, "The association policy must explicitly declare accepted geometry roles."
    policy.setdefault("center_mode", "none")
    if policy["center_mode"] not in {"none", "centroid", "bounds_center"}:
        return None, "The association policy's center containment method is unsupported."
    for field in ("require_contemporaneous", "require_registration"):
        policy.setdefault(field, False)
        if not isinstance(policy[field], bool):
            return None, "Association evidence requirements must be Boolean values."
    for field in ("maximum_date_delta_days", "max_positional_uncertainty_m"):
        if field in policy and not _finite(policy[field]):
            return None, "Association date and positional bounds must be finite and nonnegative."
    policy.setdefault("maximum_date_delta_days", 0)
    policy.setdefault("version", "unversioned")
    policy.setdefault("validation_status", "unvalidated")
    return policy, None


def _geometry(feature: dict):
    try:
        if not isinstance(feature.get("geometry"), dict):
            return None
        geometry = shape(feature.get("geometry"))
        if geometry.is_empty or not geometry.is_valid or geometry.geom_type not in {"Polygon", "MultiPolygon"}:
            return None
        west, south, east, north = geometry.bounds
        if not all(math.isfinite(value) for value in geometry.bounds) or not (-180 <= west <= east <= 180 and -90 <= south <= north <= 90):
            return None
        return geometry
    except (TypeError, ValueError, KeyError, GEOSException):
        return None


def _date(feature: dict) -> tuple[datetime | None, str | None]:
    props = feature.get("properties") or {}
    metadata = props.get("source_metadata") or {}
    value = next((source[key] for source, keys in ((props, ("observation_date", "imagery_date", "acquisition_date", "snapshot_date", "vector_extraction_date")),
        (metadata, ("observation_date", "source_date", "acquisition_date", "snapshot_date", "extraction_date")))
        for key in keys if source.get(key)), None)
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00")) if value else None
        if parsed is not None and parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed, str(value) if value else None
    except ValueError:
        return None, str(value)


def _registration(feature: dict) -> dict:
    props = feature.get("properties") or {}
    metadata = props.get("source_metadata") or {}
    registration = props.get("registration") or metadata.get("registration") or {}
    if not isinstance(registration, dict):
        registration = {}
    return {"status": registration.get("status", props.get("registration_status", metadata.get("registration_status"))),
        "id": registration.get("id", props.get("registration_id", metadata.get("registration_id"))),
        "positional_uncertainty_m": registration.get("positional_uncertainty_m", props.get("positional_uncertainty_m", metadata.get("positional_uncertainty_m")))}


def _projection(reference, configured) -> tuple[CRS | None, str | None]:
    west, south, east, north = reference.bounds
    if east - west > 6 or north - south > 6 or south < -80 or north > 84:
        return None, "The observation needs a regional or geodesic association processor."
    try:
        if configured:
            crs = CRS.from_user_input(configured)
            if not crs.is_projected or not crs.axis_info or any(abs(axis.unit_conversion_factor - 1) > 1e-10 for axis in crs.axis_info[:2]):
                return None, "The registered association CRS must use projected metre units."
        else:
            center = reference.representative_point()
            crs = CRS.from_proj4(f"+proj=aeqd +lat_0={center.y} +lon_0={center.x} +datum=WGS84 +units=m +no_defs")
        return crs, None
    except (ValueError, TypeError, CRSError):
        return None, "The registered association CRS is invalid."


def _link(clause: dict, subject: dict | None, reference: dict | None, policy: dict | None, **details) -> dict:
    return {"clause_id": clause["id"], "subject_entity_id": clause["entity_id"],
        "subject_feature_id": feature_id(subject) if subject is not None else None,
        "reference_entity_id": clause["reference_entity_id"],
        "reference_feature_id": feature_id(reference) if reference is not None else None,
        "policy_id": (clause.get("evidence_requirement") or {}).get("association_policy_id"),
        "status": "unknown", "method": "unique_overlap", "association_role": "geometric_heuristic",
        "policy_version": (policy or {}).get("version"),
        "validation_status": (policy or {}).get("validation_status", "unvalidated"),
        "accuracy_claim": None, "policy": deepcopy(policy),
        "uniqueness_scope": "all_supplied_subject_candidates", **details}


def build_association_links(plan, candidates: dict[str, list[dict]], catalog: dict) -> list[dict]:
    """Build all requested association evidence under explicit catalog policies.

    A pass requires one unique possible footprint and a robust overlap under
    supplied positional bounds. Missing geometry, role, epoch, registration or
    a competing plausible footprint leaves association unknown.
    """
    raw = plan.model_dump() if hasattr(plan, "model_dump") else deepcopy(plan)
    policies = catalog.get("association_policies", {})
    policies = policies if isinstance(policies, dict) else {}
    output = []
    for clause in raw.get("clauses", []):
        if clause.get("kind") != "visual" or clause.get("operator") != "associated_with" or not clause.get("reference_entity_id"):
            continue
        subjects = candidates.get(clause["entity_id"], [])
        references = candidates.get(clause["reference_entity_id"], [])
        policy_id = (clause.get("evidence_requirement") or {}).get("association_policy_id")
        policy, policy_error = _policy(policies.get(policy_id)) if policy_id else (None, "The clause must select an explicitly registered association policy.")
        if not references:
            output.extend(_link(clause, subject, None, policy, reason="No supplied visual observation can establish this association.") for subject in subjects)
            continue
        if policy_error:
            # No unregistered rule is applied, even as a guessed fallback.
            # Unresolved endpoints describe the gap without creating every
            # Cartesian pair. The graph keeps all unlinked pairs unknown.
            output.extend(_link(clause, subject, None, policy, reason=policy_error) for subject in subjects)
            output.extend(_link(clause, None, reference, policy, reason=policy_error) for reference in references)
            continue
        valid_subjects, geometries, invalid_subjects = [], [], []
        maximum_uncertainty = 0.0
        invalid_uncertainty = False
        for subject in subjects:
            geometry = _geometry(subject)
            if geometry is None:
                invalid_subjects.append(feature_id(subject))
                continue
            valid_subjects.append(subject)
            geometries.append(geometry)
            value = _registration(subject)["positional_uncertainty_m"]
            if value is not None:
                if _finite(value):
                    maximum_uncertainty = max(maximum_uncertainty, value)
                else:
                    invalid_uncertainty = True
        tree = STRtree(geometries)
        for reference in references:
            reference_geometry = _geometry(reference)
            role = (reference.get("properties") or {}).get("geometry_role")
            if reference_geometry is None or role not in policy["reference_geometry_roles"]:
                reason = "Reference geometry is unavailable or its role is incompatible with the policy."
                output.extend(_link(clause, subject, reference, policy, reason=reason) for subject in subjects)
                if not subjects:
                    output.append(_link(clause, None, reference, policy, reason=reason))
                continue
            crs, projection_error = _projection(reference_geometry, policy.get("analysis_crs") or catalog.get("analysis_crs"))
            if projection_error:
                output.extend(_link(clause, subject, reference, policy, reason=projection_error) for subject in subjects)
                if not subjects:
                    output.append(_link(clause, None, reference, policy, reason=projection_error))
                continue
            forward = Transformer.from_crs("EPSG:4326", crs, always_xy=True).transform
            inverse = Transformer.from_crs(crs, "EPSG:4326", always_xy=True).transform
            projected_reference = transform_geometry(reference_geometry, forward, interleaved=False)
            if not math.isfinite(projected_reference.area) or projected_reference.area <= 0:
                output.extend(_link(clause, subject, reference, policy, reason="The reference has no finite measurable area.") for subject in subjects)
                continue
            reference_registration = _registration(reference)
            reference_uncertainty = reference_registration["positional_uncertainty_m"]
            reference_error = reference_uncertainty is not None and not _finite(reference_uncertainty)
            reference_allowance = reference_uncertainty if _finite(reference_uncertainty) else 0.0
            margin = maximum_uncertainty + reference_allowance
            # A metric rectangle encloses the whole uncertainty neighbourhood.
            west, south, east, north = projected_reference.bounds
            envelope = box(west - margin, south - margin, east + margin, north + margin)
            envelope = envelope.segmentize(max(1.0, max(east - west + 2 * margin, north - south + 2 * margin) / 32))
            geographic_query = transform_geometry(envelope, inverse, interleaved=False)
            positions = sorted(int(index) for index in tree.query(geographic_query.envelope))
            center = projected_reference.centroid if policy["center_mode"] == "centroid" else Point(
                (west + east) / 2, (south + north) / 2) if policy["center_mode"] == "bounds_center" else None
            evaluated, possible = [], []
            for position in positions:
                subject, subject_geometry = valid_subjects[position], transform_geometry(geometries[position], forward, interleaved=False)
                registration = _registration(subject)
                raw_uncertainty = registration["positional_uncertainty_m"]
                allowance = (raw_uncertainty if _finite(raw_uncertainty) else 0.0) + reference_allowance
                nominal = projected_reference.intersection(subject_geometry).area / projected_reference.area
                lower_geometry = subject_geometry.buffer(-allowance) if allowance else subject_geometry
                upper_geometry = subject_geometry.buffer(allowance) if allowance else subject_geometry
                lower = projected_reference.intersection(lower_geometry).area / projected_reference.area
                upper = projected_reference.intersection(upper_geometry).area / projected_reference.area
                center_nominal = center is None or subject_geometry.covers(center)
                center_robust = center is None or lower_geometry.covers(center)
                center_possible = center is None or upper_geometry.covers(center)
                potentially_qualifying = upper >= policy["minimum_overlap"] and center_possible
                reasons = []
                if (subject.get("properties") or {}).get("geometry_role") not in policy["subject_geometry_roles"]:
                    reasons.append("Subject geometry role is incompatible with the policy.")
                if any((feature.get("properties") or {}).get("scope_time_status") not in {None, "pass"} for feature in (subject, reference)):
                    reasons.append("The observations are not established for the requested period.")
                subject_date, subject_date_text = _date(subject)
                reference_date, reference_date_text = _date(reference)
                date_delta = abs((subject_date - reference_date).total_seconds()) / 86400 if subject_date and reference_date else None
                if policy["require_contemporaneous"] and (date_delta is None or date_delta > policy["maximum_date_delta_days"]):
                    reasons.append("Observation dates are missing or exceed the policy's contemporaneous bound.")
                if policy["require_registration"] and any(record["status"] not in {"registered", "aligned"} for record in (registration, reference_registration)):
                    reasons.append("Source registration is not established by both observations.")
                if "max_positional_uncertainty_m" in policy and (raw_uncertainty is None or reference_uncertainty is None or allowance > policy["max_positional_uncertainty_m"]):
                    reasons.append("Positional uncertainty is missing or exceeds the registered policy bound.")
                if reference_error or raw_uncertainty is not None and not _finite(raw_uncertainty):
                    reasons.append("Positional uncertainty metadata is invalid.")
                if lower < policy["minimum_overlap"] or not center_robust:
                    reasons.append("Overlap or center containment does not pass robustly under positional bounds.")
                details = {"overlap_fraction": min(1.0, max(0.0, float(nominal))),
                    "minimum_overlap": policy["minimum_overlap"], "overlap_lower_bound": min(1.0, max(0.0, float(lower))),
                    "overlap_upper_bound": min(1.0, max(0.0, float(upper))), "reference_area_m2": float(projected_reference.area),
                    "center_mode": policy["center_mode"], "center_contained": center_nominal,
                    "center_robustly_contained": center_robust, "analysis_crs": crs.to_string(),
                    "positional_uncertainty_m": allowance, "subject_registration": registration,
                    "reference_registration": reference_registration, "subject_date": subject_date_text,
                    "reference_date": reference_date_text, "date_delta_days": date_delta,
                    "potentially_qualifying": potentially_qualifying,
                    "provenance": {"subject": deepcopy(subject.get("properties") or {}), "reference": deepcopy(reference.get("properties") or {})}}
                evaluated.append((subject, details, reasons))
                if potentially_qualifying:
                    possible.append(feature_id(subject))
            incomplete_uniqueness = bool(invalid_subjects or invalid_uncertainty)
            for subject, details, reasons in evaluated:
                passed = details["potentially_qualifying"] and len(possible) == 1 and not reasons and not incomplete_uniqueness
                if len(possible) > 1:
                    reasons.append("More than one supplied footprint can satisfy the association policy.")
                if incomplete_uniqueness:
                    reasons.append("Some supplied subject geometry or uncertainty is unavailable, so uniqueness is unresolved.")
                output.append(_link(clause, subject, reference, policy, **details,
                    status="pass" if passed else "unknown", candidate_subject_feature_ids=possible,
                    candidate_subject_count=len(possible), invalid_subject_feature_ids=invalid_subjects,
                    reason="One unique footprint satisfies the registered geometric policy." if passed else " ".join(reasons) or "This footprint does not qualify under the association policy."))
            if not evaluated:
                output.append(_link(clause, None, reference, policy, reason="No supplied subject footprint can be uniquely associated with this observation.",
                    candidate_subject_feature_ids=[], candidate_subject_count=0, invalid_subject_feature_ids=invalid_subjects,
                    analysis_crs=crs.to_string()))
    return output
