"""Deterministic evaluation of entity graphs against supplied geographic evidence.

Every candidate produces a persisted evaluation record. Unknown evidence is kept
unknown under negation; reference qualification and shared relationship bindings
retain the identity of the witness that actually satisfied the condition.
"""
from __future__ import annotations

import copy
import hashlib
import itertools
import json
import math
from typing import Any, Literal

from pyproj import CRS, Transformer
from shapely import STRtree, transform as transform_geometry

from .search_plan import (AttributeCondition, Clause, Entity, Expression, SearchPlan,
    SearchPlanError, expression_clause_ids, resolve_entity_layer, validate_geometry, validate_search_plan)


State = Literal["pass", "fail", "unknown"]


def boolean_state(operator: str, states: list[State]) -> State:
    if operator == "NOT":
        if len(states) != 1:
            raise ValueError("NOT requires one state.")
        return {"pass": "fail", "fail": "pass", "unknown": "unknown"}[states[0]]
    if operator == "AND":
        return "fail" if "fail" in states else "unknown" if "unknown" in states else "pass"
    if operator == "OR":
        return "pass" if "pass" in states else "unknown" if "unknown" in states else "fail"
    raise ValueError("Unknown Boolean operation.")


def evaluate_expression(expression: Expression | dict | None, states: dict[str, State]) -> State:
    if expression is None:
        return "pass"
    expression = Expression.model_validate(expression)
    if expression.operator == "clause":
        return states.get(expression.clause_id, "unknown")
    return boolean_state(expression.operator, [evaluate_expression(child, states) for child in expression.operands])


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False, default=str)


def feature_id(feature: dict) -> str:
    identity = feature.get("id", feature.get("properties", {}).get("id"))
    if identity is not None:
        return str(identity)
    # No nearby-box deduplication: only identical anonymous evidence gets an ID.
    return "anonymous:" + hashlib.sha256(_canonical(feature).encode()).hexdigest()


def record_id(entity_id: str, feature: dict) -> str:
    """The ID used for both provisional and final subject records.

    Source/epoch keys avoid ID collisions between providers and dated evidence.
    Existing IDs remain readable for sources that already namespace them.
    """
    props = feature.get("properties") or {}
    metadata = props.get("source_metadata") or {}
    source_feature = props.get("source_feature_id")
    if props.get("source_sha256"):
        source_identity = {"source_sha256": props["source_sha256"],
            "source_feature_id": source_feature if source_feature is not None else feature_id(feature),
            "geometry": feature.get("geometry"), "date": props.get("imagery_date", props.get("observation_date")),
            "visual_query": props.get("visual_query")}
        return entity_id + ":source:" + hashlib.sha256(_canonical(source_identity).encode()).hexdigest()
    epoch = metadata.get("source_date") or (metadata.get("metadata") or {}).get("source_date") or metadata.get("snapshot_date") or metadata.get("extraction_date") or props.get("vector_extraction_date")
    if metadata.get("source") and source_feature is not None and epoch:
        source_identity = {"source": metadata["source"], "source_feature_id": source_feature,
            "geometry": feature.get("geometry"), "snapshot": epoch}
        return entity_id + ":source:" + hashlib.sha256(_canonical(source_identity).encode()).hexdigest()
    source = {key: props[key] for key in ("dataset_id", "layer_id", "source_layer", "source_uri",
        "source_sha256", "dataset_version", "imagery_date", "observation_date", "vector_extraction_date") if props.get(key) is not None}
    suffix = ":" + hashlib.sha256(_canonical(source).encode()).hexdigest()[:20] if source else ""
    return f"{entity_id}:{feature_id(feature)}{suffix}"


def _merge_candidates(entity_id: str, candidates: list[dict]) -> list[dict]:
    unique: dict[str, dict] = {}
    for raw in candidates:
        if not isinstance(raw, dict) or raw.get("type") != "Feature":
            raise SearchPlanError("Graph candidates must be GeoJSON Features.")
        item = copy.deepcopy(raw)
        item.setdefault("properties", {})
        if not isinstance(item["properties"], dict):
            raise SearchPlanError("Feature properties must be an object.")
        # A changed geometry is distinct evidence, even for the same source ID.
        key = record_id(entity_id, item) + ":" + _canonical(item.get("geometry"))
        if key not in unique:
            item["properties"]["source_dataset_ids"] = sorted(set(item["properties"].get("source_dataset_ids", []) + ([item["properties"]["dataset_id"]] if item["properties"].get("dataset_id") else [])))
            unique[key] = item
            continue
        existing = unique[key]
        old, new = existing["properties"], item["properties"]
        old["source_dataset_ids"] = sorted(set(old.get("source_dataset_ids", []) + new.get("source_dataset_ids", []) + ([new["dataset_id"]] if new.get("dataset_id") else [])))
        for name in ("observations", "visual_evidence", "associations", "evidence_geometries"):
            values = old.get(name, []) + new.get(name, [])
            old[name] = list({_canonical(value): value for value in values}.values())
        if _canonical({key: value for key, value in old.items() if key not in {"observations", "visual_evidence", "associations", "evidence_geometries", "source_variants"}}) != _canonical({key: value for key, value in new.items() if key not in {"observations", "visual_evidence", "associations", "evidence_geometries", "source_variants"}}):
            variants = old.get("source_variants", []) + [raw.get("properties", {})]
            old["source_variants"] = list({_canonical(value): copy.deepcopy(value) for value in variants}.values())
    # Same source ID with conflicting geometry must not overwrite a store row.
    counts: dict[str, int] = {}
    for item in unique.values():
        identifier = record_id(entity_id, item)
        counts[identifier] = counts.get(identifier, 0) + 1
    result = []
    for item in unique.values():
        identifier = record_id(entity_id, item)
        if counts[identifier] > 1:
            item["properties"]["identity_ambiguity"] = "The same source ID has different supplied geometries."
            item["properties"]["graph_record_id"] = identifier + ":geometry:" + hashlib.sha256(_canonical(item.get("geometry")).encode()).hexdigest()[:20]
        result.append(item)
    return result


def _source_properties(feature: dict) -> dict:
    props = feature.get("properties") or {}
    return props.get("source_attributes", props.get("subject_attributes", props))


def _numeric(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("A finite numeric attribute is required.")
    return value


def evaluate_attribute(feature: dict, condition: AttributeCondition | Clause) -> dict:
    properties = _source_properties(feature)
    actual = properties.get(condition.field)
    result = {"status": "unknown", "field": condition.field, "operator": condition.operator,
        "expected": condition.value, "observed": actual}
    if (feature.get("properties") or {}).get("scope_time_status") not in {None, "pass"}:
        return {**result, "reason": "This attribute observation is not established for the requested period."}
    if condition.field not in properties or actual is None or actual == "":
        return {**result, "reason": "The source attribute is missing; missing tags do not establish a negative."}
    for variant in (feature.get("properties") or {}).get("source_variants", []):
        observed = variant.get("source_attributes", variant.get("subject_attributes", variant)).get(condition.field)
        if observed != actual:
            return {**result, "reason": "Supplied observations of the same source feature disagree on this attribute."}
    try:
        operator, expected = condition.operator, condition.value
        if operator == "equals":
            passed = type(actual) is type(expected) and actual == expected or not isinstance(actual, bool) and not isinstance(expected, bool) and isinstance(actual, (int, float)) and isinstance(expected, (int, float)) and actual == expected
        elif operator == "not_equals":
            passed = actual != expected
        elif operator == "contains":
            if not isinstance(actual, (str, list, tuple)):
                raise ValueError("The source field does not support containment.")
            passed = expected in actual
        elif operator == "in":
            if not isinstance(expected, list):
                raise ValueError("The in operator requires an explicit value list.")
            passed = actual in expected
        elif operator == "exists":
            passed = True
        elif operator in {"lt", "lte", "gt", "gte"}:
            left, right = _numeric(actual), _numeric(expected)
            passed = {"lt": left < right, "lte": left <= right, "gt": left > right, "gte": left >= right}[operator]
        else:
            return {**result, "reason": "No tested attribute operator is configured."}
        return {**result, "status": "pass" if passed else "fail", "reason": "Checked against the supplied source attribute."}
    except (TypeError, ValueError):
        return {**result, "reason": "The source value and operator have incompatible types."}


def _geometry(feature: dict):
    try:
        return validate_geometry(feature.get("geometry"))
    except (ValueError, TypeError, KeyError):
        return None


def _metric_projection(geometries: list, configured: str | CRS | None):
    if configured is not None:
        crs = CRS.from_user_input(configured)
        if not crs.is_projected or any(axis.unit_conversion_factor != 1 for axis in crs.axis_info[:2]):
            raise SearchPlanError("Spatial analysis CRS must be projected and use metre units.")
        return crs
    if not geometries:
        return None
    west = min(value.bounds[0] for value in geometries)
    south = min(value.bounds[1] for value in geometries)
    east = max(value.bounds[2] for value in geometries)
    north = max(value.bounds[3] for value in geometries)
    # A local UTM calculation is unsuitable for a broad corpus. A configured
    # appropriate CRS/geodesic processor must handle that case.
    if east - west > 6 or north - south > 6 or south < -80 or north > 84:
        return None
    longitude, latitude = (west + east) / 2, (south + north) / 2
    zone = min(60, max(1, int((longitude + 180) // 6) + 1))
    return CRS.from_epsg((32600 if latitude >= 0 else 32700) + zone)


def _coverage_for(entity: Entity, coverage: dict, catalog: dict | None) -> dict:
    result = {}
    if catalog and entity.layer_id:
        layer = resolve_entity_layer(entity, catalog)
        if layer:
            details = layer[1]
            result.update(details.get("metadata", {}))
            result.update({key: value for key, value in details.items() if key not in {"metadata", "fields", "observed_values"}})
    for key in (entity.dataset_id, f"{entity.dataset_id}:{entity.layer_id}" if entity.dataset_id and entity.layer_id else None, entity.layer_id, entity.id):
        if key is not None and isinstance(coverage.get(key), dict):
            result.update(coverage[key])
    return result


def _complete(metadata: dict) -> bool:
    return metadata.get("enumeration_complete") is True or metadata.get("reference_coverage_complete") is True


def _clause_base(clause: Clause) -> dict:
    return {"clause_id": clause.id, "condition": clause.kind, "entity_id": clause.entity_id,
        "reference_entity_id": clause.reference_entity_id, "operator": clause.operator,
        "original_text": clause.original_text, "binding_id": clause.binding_id,
        "evidence_requirement": clause.evidence_requirement.model_dump(), "status": "unknown"}


def execute_graph(plan: SearchPlan | dict, candidates: dict[str, list[dict]], scope_geometry: dict | None = None,
        *, catalog: dict | None = None, analysis_crs: str | CRS | None = None,
        coverage: dict | None = None, association_links: list[dict] | None = None) -> list[dict]:
    """Evaluate all supplied subject candidates; never paginate or prune unknowns.

    ``coverage`` is keyed by entity/dataset/layer and must explicitly declare
    ``enumeration_complete`` (or ``reference_coverage_complete``) for absence.
    Association links need matching subject/reference IDs, ``status`` and a
    ``policy_id``. Reference candidates outside subject scope remain available.
    """
    plan = validate_search_plan(plan, catalog)
    if plan.status != "ready":
        raise SearchPlanError("Resolve the plan clarifications before geographic execution.", "needs_clarification", plan.clarifications)
    entity_by_id = {entity.id: entity for entity in plan.entities}
    if candidates.keys() - entity_by_id.keys():
        raise SearchPlanError("Candidate inputs reference an unknown plan entity.")
    candidate_by_id = {identifier: _merge_candidates(identifier, candidates.get(identifier, [])) for identifier in entity_by_id}
    clauses = {clause.id: clause for clause in plan.clauses}
    coverage = coverage or {}
    association_links = association_links or []
    scope = validate_geometry(scope_geometry or plan.scope.geometry) if scope_geometry is not None or plan.scope.geometry is not None else None
    geometry_cache = {id(feature): _geometry(feature) for features in candidate_by_id.values() for feature in features}
    all_geometries = [value for value in geometry_cache.values() if value is not None]
    crs = _metric_projection(all_geometries + ([scope] if scope is not None else []), analysis_crs or (catalog or {}).get("analysis_crs"))
    transformer = Transformer.from_crs("EPSG:4326", crs, always_xy=True) if crs else None
    metric_cache = {}
    reference_indexes = {}
    qualification_cache = {}
    identity_cache = {}
    definite_rejects = {}

    def candidate_identity(identifier: str, feature: dict):
        key = identifier, id(feature)
        if key not in identity_cache:
            identity_cache[key] = (feature.get("properties") or {}).get("graph_record_id") or record_id(identifier, feature)
        return identity_cache[key]

    def reference_complete(identifier: str):
        return _complete(_coverage_for(entity_by_id[identifier], coverage, catalog)) and not any(
            (feature.get("properties") or {}).get("scope_time_status") not in {None, "pass"}
            for feature in candidate_by_id[identifier])

    def absence_supported(identifier: str, clause: Clause | None = None):
        metadata = _coverage_for(entity_by_id[identifier], coverage, catalog)
        expected = clause.evidence_requirement.absence_policy_id if clause else None
        return reference_complete(identifier) and metadata.get("absence_supported") is True and bool(metadata.get("absence_policy_id")) and (
            expected is None or expected == metadata.get("absence_policy_id"))

    def inverted(state: State):
        return boolean_state("NOT", [state])

    def expression_pair(node: Expression | None, positive: dict, negative: dict):
        if node is None:
            return "pass", "fail"
        if node.operator == "clause":
            return positive.get(node.clause_id, "unknown"), negative.get(node.clause_id, "unknown")
        children = [expression_pair(child, positive, negative) for child in node.operands]
        if node.operator == "NOT":
            return children[0][1], children[0][0]
        opposite = "OR" if node.operator == "AND" else "AND"
        return boolean_state(node.operator, [child[0] for child in children]), boolean_state(opposite, [child[1] for child in children])

    def metric(feature):
        key = id(feature)
        if key not in metric_cache:
            geometry = geometry_cache[key]
            if geometry is None or transformer is None:
                metric_cache[key] = None
            else:
                def project(coordinates):
                    result = coordinates.copy()
                    result[:, 0], result[:, 1] = transformer.transform(coordinates[:, 0], coordinates[:, 1])
                    return result
                projected = transform_geometry(geometry, project)
                metric_cache[key] = projected if projected.is_valid and all(math.isfinite(value) for value in projected.bounds) else None
        return metric_cache[key]

    def qualify(reference: dict, entity: Entity, clause: Clause):
        cache_key = clause.id, id(reference)
        if cache_key in qualification_cache:
            return qualification_cache[cache_key]
        attributes = [evaluate_attribute(reference, item) for item in entity.attributes]
        time_condition = scope_time_condition(reference)
        if time_condition is not None:
            attributes.append(time_condition)
        if entity.kind in {"visual", "scene"}:
            attributes.append(visual(Clause(id="entity_visual_evidence", kind="visual", entity_id=entity.id,
                operator="present", value=entity.visual_query or entity.description), reference, entity))
        checks = attributes + [evaluate(clauses[identifier], entity.id, reference) for identifier in clause.reference_clause_ids]
        result = boolean_state("AND", [check["status"] for check in checks]), checks
        qualification_cache[cache_key] = result
        return result

    def allowance(feature):
        value = (feature.get("properties") or {}).get("positional_uncertainty_m", 0)
        return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0 else None

    def possible_references(clause: Clause, subject: dict, references: list[dict]):
        """Prune only pairs that cannot satisfy a positive spatial predicate."""
        subject_geometry = metric(subject)
        subject_allowance = allowance(subject)
        if subject_geometry is None or subject_allowance is None or clause.operator not in {"within_distance", "intersects", "within", "covered_by", "contains", "covers"}:
            return references, set()
        identifier = clause.reference_entity_id
        if identifier not in reference_indexes:
            features, geometries, pending, maximum = [], [], [], 0.0
            for reference in references:
                geometry, uncertainty = metric(reference), allowance(reference)
                if geometry is None or uncertainty is None or (reference.get("properties") or {}).get("scope_time_status") not in {None, "pass"}:
                    pending.append(reference)
                else:
                    features.append(reference)
                    geometries.append(geometry)
                    maximum = max(maximum, uncertainty)
            reference_indexes[identifier] = STRtree(geometries), features, pending, maximum
        tree, indexed, pending, maximum = reference_indexes[identifier]
        distance = (clause.distance_m or 0) if clause.operator == "within_distance" else 0
        margin = distance + subject_allowance + maximum
        # The expanded rectangle contains the complete distance neighbourhood;
        # unlike a polygonal circle buffer it cannot miss diagonal boundary pairs.
        west, south, east, north = subject_geometry.bounds
        from shapely.geometry import box
        envelope = box(west - margin, south - margin, east + margin, north + margin)
        positions = {int(position) for position in tree.query(envelope)}
        possible = [reference for position, reference in enumerate(indexed) if position in positions] + pending
        rejected = {id(reference) for position, reference in enumerate(indexed) if position not in positions}
        return possible, rejected

    def scope_time_condition(feature: dict):
        props = feature.get("properties") or {}
        if "scope_time_status" not in props:
            return None
        return {"condition": "scope_observation_period", "status": "pass" if props["scope_time_status"] == "pass" else "unknown",
            "source_time": (props.get("source_metadata") or {}).get("source_date", props.get("imagery_date")),
            "reason": props.get("scope_time_reason", "The source observation period has not been established for the requested dates.")}

    def spatial_pair(clause, subject, reference):
        left, right = metric(subject), metric(reference)
        result = {"status": "unknown", "spatial_status": "unknown", "distance_m": None,
            "analysis_crs": crs.to_string() if crs else None}
        if left is None or right is None:
            return {**result, "reason": "Valid geometries and a suitable metric/geodesic processor are required."}
        if any((item.get("properties") or {}).get("scope_time_status") not in {None, "pass"} for item in (subject, reference)):
            return {**result, "reason": "The measured geometry is not established for the requested observation period."}
        distance = float(left.distance(right))
        result["distance_m"] = distance
        operator = clause.operator
        if operator == "within_distance":
            passed = distance <= clause.distance_m
        elif operator == "nearest":
            passed = True
        else:
            predicates = {"intersects": left.intersects, "within": left.within, "covered_by": left.covered_by,
                "contains": left.contains, "covers": left.covers, "outside": left.disjoint, "disjoint": left.disjoint}
            passed = predicates[operator](right)
        status: State = "pass" if passed else "fail"
        roles = [(item.get("properties") or {}).get("geometry_role", "source_vector_feature") for item in (subject, reference)]
        unknown_box = any(role in {"object_box", "detection_box", "object_envelope", "visual_detection_envelope", "scene_footprint", "image_tile_coverage", "image_tile_footprint"} for role in roles)
        if passed and unknown_box and not clause.evidence_requirement.geometry_roles:
            status = "unknown"
            result["reason"] = "A detector envelope is approximate; this boundary-dependent object relationship needs association or an accepted geometry policy."
        allowance = 0.0
        for item in (subject, reference):
            raw_allowance = (item.get("properties") or {}).get("positional_uncertainty_m", 0)
            if isinstance(raw_allowance, (int, float)) and not isinstance(raw_allowance, bool) and math.isfinite(raw_allowance) and raw_allowance >= 0:
                allowance += raw_allowance
            else:
                status = "unknown"
                result["reason"] = "Positional uncertainty metadata is invalid."
        if operator == "within_distance" and allowance:
            status = "pass" if distance + allowance <= clause.distance_m else "fail" if distance - allowance > clause.distance_m else "unknown"
            if unknown_box and passed and not clause.evidence_requirement.geometry_roles:
                status = "unknown"
        elif allowance and operator != "nearest":
            status = "unknown"
            result["reason"] = "Declared registration uncertainty requires review of this boundary relationship."
        required = clause.evidence_requirement.geometry_roles
        if required and any(role not in required for role in roles):
            status = "unknown"
            result["reason"] = "A supplied geometry role does not meet this clause's evidence requirement."
        return {**result, "status": status, "spatial_status": status, "distance_uncertainty_m": allowance,
            "geometry_roles": roles, "radius_m": clause.distance_m,
            "reason": result.get("reason", "Evaluated exact predicate on the declared geometries in a projected metre CRS.")}

    def visual(clause: Clause, feature: dict, entity: Entity):
        base = _clause_base(clause)
        query = clause.value if isinstance(clause.value, str) else entity.visual_query or entity.description
        entries = [entry for entry in (feature.get("properties") or {}).get("visual_evidence", [])
            if isinstance(entry, dict) and entry.get("query") == query]
        supported = [entry for entry in entries if entry.get("evidence_type") not in {"scene_similarity", "retrieval", "tile_retrieval"}]
        positive = [entry for entry in supported if entry.get("status") == "pass"]
        absence = [entry for entry in supported if entry.get("status") == "fail"
            and entry.get("absence_policy_id") and entry.get("coverage_complete") is True and entry.get("adequate_observation") is True
            and (clause.evidence_requirement.absence_policy_id is None or entry.get("absence_policy_id") == clause.evidence_requirement.absence_policy_id)]
        if positive:
            state = "fail" if clause.operator == "absent" else "pass"
        elif absence:
            state = "pass" if clause.operator == "absent" else "fail"
        else:
            state = "unknown"
        return {**base, "status": state, "visual_query": query, "observations": entries,
            "reason": "Exact-prompt visual evidence was evaluated." if state != "unknown" else
            "No exact-prompt observation establishes this condition; non-detection or scene similarity is not proof of absence."}

    def relationship(clause: Clause, subject: dict, association=False):
        base = _clause_base(clause)
        reference_entity = entity_by_id[clause.reference_entity_id]
        references = candidate_by_id[reference_entity.id]
        metadata = _coverage_for(reference_entity, coverage, catalog)
        complete = reference_complete(reference_entity.id)
        total_references = len(references)
        spatial_rejects = set()
        if not association:
            references, spatial_rejects = possible_references(clause, subject, references)
            definite_rejects[(clause.id, id(subject))] = spatial_rejects
        pairs = []
        for reference in references:
            if feature_id(reference) == feature_id(subject) and clause.entity_id == clause.reference_entity_id:
                continue
            qualified, checks = qualify(reference, reference_entity, clause)
            if association:
                links = association_links + (subject.get("properties") or {}).get("associations", [])
                matching = [item for item in links if item.get("subject_entity_id", clause.entity_id) == clause.entity_id
                    and str(item.get("subject_feature_id")) == feature_id(subject)
                    and item.get("reference_entity_id", reference_entity.id) == reference_entity.id
                    and str(item.get("reference_feature_id")) == feature_id(reference)]
                supported = [item for item in matching if item.get("policy_id") and
                    (clause.evidence_requirement.association_policy_id is None or item["policy_id"] == clause.evidence_requirement.association_policy_id)]
                state = "pass" if any(item.get("status") == "pass" for item in supported) else "unknown"
                detail = {"status": state, "spatial_status": state, "association_links": matching,
                    "reason": "A recorded association policy links these exact entities." if state == "pass" else "Object association is missing or ambiguous; nearby evidence cannot establish ownership by this subject."}
            else:
                detail = spatial_pair(clause, subject, reference)
            pairs.append({**detail, "status": boolean_state("AND", [detail["status"], qualified]),
                "reference_feature_id": feature_id(reference), "reference_record_id": candidate_identity(reference_entity.id, reference),
                "reference_entity_id": reference_entity.id, "reference_feature": copy.deepcopy(reference),
                "reference_conditions": checks})
        if clause.operator == "nearest" and pairs:
            finite = [item["distance_m"] for item in pairs if item.get("distance_m") is not None and item["status"] != "fail"]
            minimum = min(finite) if finite else None
            for item in pairs:
                if minimum is not None and item.get("distance_m") is not None and not math.isclose(item["distance_m"], minimum, rel_tol=1e-12, abs_tol=1e-9):
                    item["status"] = "fail"
                elif not complete and item["status"] == "pass":
                    item["status"] = "unknown"
                    item["reason"] = "Nearest requires complete enumeration of the declared reference universe."
        statuses = [item["status"] for item in pairs]
        if clause.quantifier == "all":
            state = "fail" if spatial_rejects else boolean_state("AND", statuses) if statuses else "unknown"
            if not complete and state == "pass":
                state = "unknown"
        else:
            state = boolean_state("OR", statuses) if statuses else "fail"
            if state == "fail" and (not complete or association or reference_entity.kind in {"visual", "scene"}):
                state = "unknown"
        negative = "unknown" if state == "fail" and not absence_supported(reference_entity.id, clause) else inverted(state)
        return {**base, "status": state, "negation_status": negative, "quantifier": clause.quantifier, "reference_coverage": metadata,
            "references_total": total_references, "references_evaluated": len(pairs), "definite_spatial_rejects_count": len(spatial_rejects),
            "reference_coverage_complete": complete, "reference_candidates": pairs,
            "witnesses": [item for item in pairs if item["status"] in {"pass", "unknown"}],
            "reference_feature_ids": [item["reference_feature_id"] for item in pairs if item["status"] == "pass"],
            "reason": "All qualifying witnesses were retained." if state == "pass" else
            "The relationship is unsupported by complete supplied reference evidence." if state == "fail" else
            "Reference evidence, geometry, association or coverage is incomplete."}

    def evaluate(clause: Clause, current_entity: str, feature: dict):
        base = _clause_base(clause)
        if clause.availability != "available":
            return {**base, "reason": clause.unavailability_reason or "This requirement has no available processor."}
        if clause.entity_id != current_entity:
            return {**base, "reason": "The clause describes another entity; a recorded relationship binding is required."}
        if clause.kind == "attribute":
            return {**base, **evaluate_attribute(feature, clause)}
        if clause.kind == "temporal":
            # Paired dates alone are not a validated change processor result.
            return {**base, "reason": "No validated temporal processor output is supplied; a single-date observation cannot establish change."}
        if clause.kind == "spatial":
            return relationship(clause, feature)
        if clause.operator == "associated_with":
            if not clause.reference_entity_id:
                return {**base, "reason": "An object association requires a reference entity."}
            return relationship(clause, feature, association=True)
        return visual(clause, feature, entity_by_id[current_entity])

    records = []
    root_clause_ids = expression_clause_ids(plan.expression)
    for entity_id in plan.subject_entity_ids:
        entity = entity_by_id[entity_id]
        for subject in candidate_by_id[entity_id]:
            geometry = geometry_cache[id(subject)]
            if scope is not None and geometry is not None:
                inside = geometry.intersects(scope) if plan.scope.inclusion == "intersects" else geometry.within(scope) if plan.scope.inclusion == "within" else scope.covers(geometry.centroid)
                if not inside:
                    continue
            conditions = [evaluate(clauses[identifier], entity_id, subject) for identifier in sorted(root_clause_ids)]
            state_by_clause = {condition["clause_id"]: condition["status"] for condition in conditions}
            negative_by_clause = {condition["clause_id"]: condition.get("negation_status", inverted(condition["status"])) for condition in conditions}
            entity_checks = [evaluate_attribute(subject, item) for item in entity.attributes]
            time_condition = scope_time_condition(subject)
            if time_condition is not None:
                entity_checks.append(time_condition)
            if entity.kind in {"visual", "scene"}:
                entity_checks.append(visual(Clause(id="entity_visual_evidence", kind="visual", entity_id=entity.id,
                    operator="present", value=entity.visual_query or entity.description), subject, entity))
            binding_entities = {}
            binding_paths = {}
            def collect_paths(node: Expression | None, path: tuple[int, ...] = ()):
                if node is None:
                    return
                if node.operator == "clause":
                    clause = clauses[node.clause_id]
                    if clause.binding_id and clause.quantifier == "any":
                        binding_paths.setdefault(clause.binding_id, []).append(path)
                for position, child in enumerate(node.operands):
                    collect_paths(child, path + (position,))
            collect_paths(plan.expression)
            for identifier in root_clause_ids:
                clause = clauses[identifier]
                if clause.binding_id and clause.quantifier == "any":
                    binding_entities[clause.binding_id] = clause.reference_entity_id if clause.reference_entity_id else clause.entity_id
            bindings = []
            if binding_entities:
                universes = {binding: candidate_by_id[identifier] or [None] for binding, identifier in binding_entities.items()}
                # An incomplete source may contain an unobserved possible witness.
                for binding, identifier in binding_entities.items():
                    if universes[binding] != [None] and not reference_complete(identifier):
                        universes[binding] = universes[binding] + [None]
                budget = plan.execution_policy.max_binding_evaluations
                scopes = {}
                for name, paths in binding_paths.items():
                    common = paths[0]
                    for path in paths[1:]:
                        common = common[:next((index for index, (left, right) in enumerate(zip(common, path)) if left != right), min(len(common), len(path)))]
                    scopes.setdefault(common, []).append(name)
                evaluations = 0
                def bound_clause(clause: Clause, assignment: dict):
                    if clause.availability != "available":
                        return "unknown", {"reason": clause.unavailability_reason}
                    reference = assignment[clause.binding_id]
                    if reference is None:
                        target = entity_by_id[binding_entities[clause.binding_id]]
                        complete = reference_complete(target.id)
                        negative_safe = target.kind == "mapped" and clause.kind != "visual"
                        return ("fail" if complete and negative_safe else "unknown"), {"reason": "No supplied bound reference; visual absence remains unproven."}
                    if clause.entity_id != entity_id and not clause.reference_entity_id:
                        detail = evaluate(clause, clause.entity_id, reference)
                        return detail["status"], detail
                    evaluated = next((item for item in conditions if item["clause_id"] == clause.id), {})
                    pair = next((item for item in evaluated.get("reference_candidates", []) if item["reference_record_id"] == candidate_identity(binding_entities[clause.binding_id], reference)), None)
                    if pair is None and id(reference) in definite_rejects.get((clause.id, id(subject)), set()):
                        return "fail", {"spatial_status": "fail", "reference_feature_id": feature_id(reference),
                            "reason": "The conservative spatial index established this pair cannot satisfy the requested predicate."}
                    return (pair["status"], pair) if pair else ("unknown", {"reason": "The supplied pair has no bound relationship evidence."})
                def expression_body(node: Expression, path, assignment):
                    if node.operator == "clause":
                        clause = clauses[node.clause_id]
                        if clause.binding_id in assignment and clause.quantifier == "any":
                            state, evidence = bound_clause(clause, assignment)
                            bindings.append({"status": state, "clause_id": clause.id, "assignments": {key: feature_id(value) if value else None for key, value in assignment.items()}, "evidence": evidence})
                            return state, inverted(state)
                        return state_by_clause.get(node.clause_id, "unknown"), negative_by_clause.get(node.clause_id, "unknown")
                    children = [evaluate_bound(child, path + (position,), assignment) for position, child in enumerate(node.operands)]
                    if node.operator == "NOT":
                        return children[0][1], children[0][0]
                    opposite = "OR" if node.operator == "AND" else "AND"
                    return boolean_state(node.operator, [child[0] for child in children]), boolean_state(opposite, [child[1] for child in children])
                def evaluate_bound(node: Expression, path, assignment):
                    nonlocal evaluations
                    names = scopes.get(path, [])
                    if not names:
                        return expression_body(node, path, assignment)
                    states = []
                    for values in itertools.product(*(universes[name] for name in names)):
                        evaluations += 1
                        if budget is not None and evaluations > budget:
                            bindings.append({"status": "unknown", "reason": "The shared-binding evaluation budget was reached; no combinations were silently discarded.", "budget": budget})
                            states.append(("unknown", "unknown"))
                            break
                        bound = {**assignment, **dict(zip(names, values))}
                        states.append(expression_body(node, path, bound))
                    positive = boolean_state("OR", [state[0] for state in states])
                    negative = boolean_state("AND", [state[1] for state in states])
                    if negative == "pass" and not all(absence_supported(binding_entities[name]) for name in names):
                        negative = "unknown"
                    return positive, negative
                expression_state = evaluate_bound(plan.expression, (), {})[0]
            else:
                expression_state = expression_pair(plan.expression, state_by_clause, negative_by_clause)[0]
            state = boolean_state("AND", [expression_state] + [item["status"] for item in entity_checks])
            if geometry is None:
                state = boolean_state("AND", [state, "unknown"])
                conditions.append({"condition": "geometry", "status": "unknown", "reason": "The subject has no valid geographic geometry."})
            if plan.unresolved_conditions:
                state = boolean_state("AND", [state, "unknown"])
                conditions.extend({"condition": "unresolved_requirement", "original_text": text, "status": "unknown", "reason": "The original requirement remains unresolved."} for text in plan.unresolved_conditions)
            visual_subject = entity.kind in {"visual", "scene"}
            group = "matches" if state == "pass" else "excluded" if state == "fail" else "visual_candidates" if visual_subject else "needs_review"
            result_feature = copy.deepcopy(subject)
            identifier = result_feature["properties"].get("graph_record_id") or record_id(entity_id, subject)
            result_feature["id"] = identifier
            result_feature["properties"].update({"source_feature_id": feature_id(subject), "entity_id": entity_id,
                "entity_description": entity.description, "result_group": group, "expression_status": state,
                "conditions": conditions, "entity_attribute_conditions": entity_checks,
                "relationship_bindings": bindings, "analysis_crs": crs.to_string() if crs else None,
                "geometry_role": result_feature["properties"].get("geometry_role", "source_vector_feature" if entity.kind == "mapped" else "unknown"),
                "source_evidence": {"dataset_id": entity.dataset_id, "layer_id": entity.layer_id,
                    "visual_query": entity.visual_query, "original_query": plan.original_query},
                "reason": "; ".join(item.get("reason", "Unresolved condition") for item in conditions if item["status"] == "unknown") if state == "unknown" else None})
            if geometry is not None:
                result_feature["bbox"] = list(geometry.bounds)
                result_feature["properties"]["map_location"] = {"type": "Point", "coordinates": list(geometry.representative_point().coords[0]), "role": "representative_point"}
            records.append({"id": identifier, "group": group, "feature": result_feature})
    return sorted(records, key=lambda record: record["id"])
