"""Image retrieval and explicit, auditable object-to-vector spatial checks."""
from __future__ import annotations
from time import perf_counter
import math
from typing import Any
from urllib.parse import quote
import geopandas as gpd
from pyproj import CRS
from shapely.geometry import mapping, shape
from .bundle import Bundle, json_value
from .planner import QueryPlan, validate_plan

def _feature_id(row: Any, index: Any) -> str:
    for key in ("osm_id", "osm_way_id", "source_feature_id", "id", "@id"):
        value = json_value(row.get(key))
        if value is not None and str(value).strip():
            return str(value)
    return str(index)

def _test_attribute(properties: dict[str, Any], condition) -> tuple[str, str | None]:
    raw = json_value(properties.get(condition.field))
    actual = None if raw is None else str(raw)
    if actual is None or not actual.strip():
        return "unknown", actual
    passed = actual.casefold() == condition.value.casefold() if condition.operator == "equals" else condition.value.casefold() in actual.casefold()
    return ("pass" if passed else "fail"), actual

def _attribute_evidence(properties: dict, conditions: list) -> list[dict]:
    result = []
    for condition in conditions:
        status, actual = _test_attribute(properties, condition)
        result.append({"condition": f"{condition.role}_attribute", "status": status, "field": condition.field,
            "operator": condition.operator, "expected": condition.value, "observed": actual})
    return result

def condition_state(conditions: list[dict]) -> str:
    # Conjunction: a known false clause always excludes a record, even if another is unknown.
    states = {item["status"] for item in conditions}
    return "fail" if "fail" in states else "unknown" if "unknown" in states else "pass"

def _properties(row, geometry_name: str) -> dict:
    return json_value(row.drop(labels=[geometry_name], errors="ignore").to_dict())

def _asset_url(tile_id: str) -> str:
    return "/api/assets/" + quote(tile_id, safe="")

def _tile_candidate(tile: dict, score: float | None) -> dict:
    return {"type": "Feature", "id": "tile:" + tile["tile_id"], "geometry": tile["geometry"], "properties": {
        "id": "tile:" + tile["tile_id"], "tile_id": tile["tile_id"], "role": "retrieval_candidate",
        "geometry_role": "image_tile_coverage", "evidence_status": "tile_candidate_requires_grounding",
        "retrieval_similarity": score, "imagery_date": tile.get("acquisition_date"),
        "imagery_date_status": tile.get("acquisition_date_status", "known" if tile.get("acquisition_date") else "unknown"),
        "source_image": tile.get("source_name"), "source_sha256": tile.get("source_sha256"),
        "asset_url": _asset_url(tile["tile_id"]), "conditions": []}}

def _analysis_crs(bundle: Bundle) -> CRS:
    configured = bundle.manifest.get("analysis_crs")
    if configured:
        crs = CRS.from_user_input(configured)
        if not crs.is_projected or any(axis.unit_name.lower() not in {"metre", "meter"} for axis in crs.axis_info[:2]):
            raise ValueError("The configured analysis CRS must use projected metre units.")
        return crs
    coverage = gpd.GeoSeries([shape(tile["geometry"]) for tile in bundle.tiles], crs="EPSG:4326")
    bounds = coverage.total_bounds
    if bounds[2] - bounds[0] > 6 or bounds[3] - bounds[1] > 6:
        raise ValueError("This corpus is too broad for one local UTM spatial analysis; prepare a local area or metric CRS.")
    crs = coverage.estimate_utm_crs()
    if crs is None:
        raise ValueError("Could not choose a local metric CRS for spatial checks.")
    return CRS.from_user_input(crs)

def _merge_observations(features: list[dict]) -> tuple[list[dict], int]:
    """Merge identified vector objects and strongly overlapping unassociated envelopes."""
    groups: list[dict] = []
    geometries = []
    collapsed = 0
    for feature in features:
        props = feature["properties"]
        geometry = shape(feature["geometry"])
        found = None
        for index, existing in enumerate(groups):
            old = existing["properties"]
            if props.get("association") == "unique" and old.get("association") == "unique":
                if (props.get("association_layer"), props["id"]) == (old.get("association_layer"), old["id"]):
                    found = index
                    break
            elif props.get("association") == old.get("association") == "not_required" and props.get("source_sha256") == old.get("source_sha256"):
                other = geometries[index]
                union = geometry.union(other).area
                if union > 0 and geometry.intersection(other).area / union >= 0.65:
                    found = index
                    break
        if found is None:
            groups.append(feature)
            geometries.append(geometry)
        else:
            collapsed += 1
            target = groups[found]["properties"]
            target["observations"].extend(props.get("observations", []))
            target["evidence_geometries"].extend(props.get("evidence_geometries", []))
            target["detection_ids"] = sorted(set(target["detection_ids"] + props["detection_ids"]))
            target["tile_ids"] = sorted(set(target["tile_ids"] + props["tile_ids"]))
            similarities = [value for value in (target.get("retrieval_similarity"), props.get("retrieval_similarity")) if value is not None]
            target["retrieval_similarity"] = max(similarities) if similarities else None
            target["retrieval_similarity_aggregation"] = "maximum_across_observations"
    return groups, collapsed

def execute(bundle: Bundle, plan: QueryPlan, mode: str = "catalog_scan", page_size: int = 50, page: int = 0, *, all_results: bool = False) -> dict[str, Any]:
    started = perf_counter()
    plan = validate_plan(plan, bundle.manifest, mode)
    if not 1 <= page_size <= 100 or not 0 <= page <= 1000:
        raise ValueError("Pagination must use page_size 1–100 and page 0–1000.")
    text_vector = bundle.encode_text(plan.visual_query)
    if mode == "explore" and text_vector is None:
        raise ValueError("Exploration needs a verified, compatible local text encoder; cached catalog scanning remains available.")
    scores = {} if text_vector is None else {tile["tile_id"]: float(value) for tile, value in zip(bundle.tiles, bundle.embeddings @ text_vector)}
    ranked = sorted(bundle.tile_index, key=lambda identifier: (-scores.get(identifier, 0), identifier))
    limit = min(1000, max(1, int(bundle.manifest.get("limits", {}).get("explore_tile_limit", 100))))
    selected = ranked[:limit] if mode == "explore" else ranked
    allowed = set(selected)
    retrieval_ms = round((perf_counter() - started) * 1000, 2)
    base = {"plan": plan.model_dump(), "dataset_version": bundle.manifest.get("dataset_version"), "mode": mode,
        "tiles_considered": len(bundle.tiles), "tiles_processed": len(allowed), "tiles_available": len(bundle.tiles),
        "page": page, "page_size": page_size, "text_similarity_enabled": text_vector is not None,
        "text_encoder_status": "ready" if text_vector is not None else bundle.text_model_error or "checkpoint_not_configured",
        "coverage": bundle.manifest.get("coverage", {}), "provenance": bundle.provenance(),
        "timings_ms": {"retrieval": retrieval_ms}, "candidate_selection_truncated": len(selected) < len(bundle.tiles)}
    start, end = (0, None) if all_results else (page * page_size, (page + 1) * page_size)
    if plan.query_type == "tiles":
        candidates = [_tile_candidate(bundle.tiles[bundle.tile_index[tile_id]], scores.get(tile_id)) for tile_id in selected]
        base.update({"result_kind": "tile_retrieval", "features": [], "tile_candidates": candidates[start:end],
            "uncertain_candidates": [], "context_features": [], "objects_evaluated": 0, "total_matches": 0,
            "total_tile_candidates": len(candidates), "total_uncertain": 0,
            "truncated": not all_results and (page + 1) * page_size < len(candidates),
            "unresolved_conditions": ([{"condition": plan.predicate, "reference_layer": plan.reference_layer,
                "distance_m": plan.distance_m, "status": "unknown", "reason": "Tile similarity does not localize an object or establish its spatial relationship."}] if plan.predicate else []),
            "claim_scope": "Ranked image tiles only. Similarity is not a probability; object identity and spatial relationships require grounding."})
        base["timings_ms"]["total"] = round((perf_counter() - started) * 1000, 2)
        return json_value(base)

    concept = bundle.manifest["supported_concepts"][plan.concept_id]
    associated_layer = concept.get("association_layer")
    analysis_crs = _analysis_crs(bundle)
    reference = bundle.layers[plan.reference_layer]
    reference_metric = reference.to_crs(analysis_crs)
    subjects = bundle.layers.get(associated_layer)
    subjects_metric = None if subjects is None else subjects.to_crs(analysis_crs)
    reference_conditions = [condition for condition in plan.attributes if condition.role == "reference"]
    subject_conditions = [condition for condition in plan.attributes if condition.role == "subject"]
    allowance = bundle.manifest.get("distance_uncertainty_m")
    if allowance is not None and (not isinstance(allowance, (int, float)) or not math.isfinite(allowance) or not 0 <= allowance <= 1000):
        raise ValueError("Distance uncertainty allowance must be a finite value in metres from 0 to 1000.")
    margin = float(allowance or 0)
    detections = [feature for feature in bundle.detections if feature["properties"].get("concept_id") == plan.concept_id
        and any(identifier in allowed for identifier in [feature["properties"].get("tile_id"), *[observation.get("tile_id") for observation in feature["properties"].get("observations", [])]])]
    records, uncertain, excluded = [], [], 0
    context_by_id = {}
    for detection in detections:
        props = detection["properties"]
        detected_geometry = shape(detection["geometry"])
        detection_metric = gpd.GeoSeries([detected_geometry], crs="EPSG:4326").to_crs(analysis_crs).iloc[0]
        tile = bundle.tiles[bundle.tile_index[props["tile_id"]]]
        detection_id = str(props.get("detection_id") or detection.get("id"))
        subject_id, subject_geometry, subject_properties = detection_id, detected_geometry, {}
        association_state, candidates = "not_required", []
        if subjects is not None:
            positions = subjects_metric.sindex.query(detection_metric, predicate="intersects")
            center = detection_metric.representative_point()
            for position in positions:
                row = subjects_metric.iloc[int(position)]
                overlap = row.geometry.intersection(detection_metric).area / max(detection_metric.area, 1e-9)
                if overlap >= float(concept.get("association_overlap_min", 0.5)) and row.geometry.covers(center):
                    candidates.append(int(position))
            if len(candidates) == 1:
                row = subjects.iloc[candidates[0]]
                subject_geometry, subject_properties = row.geometry, _properties(row, subjects.geometry.name)
                subject_id, association_state = _feature_id(row, candidates[0]), "unique"
            else:
                association_state = "ambiguous" if candidates else "unmatched"
        conditions = [{"condition": "visual_detection", "status": "pass", "concept_id": plan.concept_id,
            "score": props.get("score"), "score_meaning": "uncalibrated_model_score", "checkpoint": props.get("model_id"),
            "model_revision": props.get("model_revision"), "screening_policy": "preparation_threshold",
            "screening_threshold": concept.get("detector_threshold", bundle.manifest.get("object_detector", {}).get("score_threshold"))}]
        if associated_layer:
            conditions.append({"condition": "feature_association", "status": "pass" if association_state == "unique" else "unknown",
                "layer": associated_layer, "feature_id": subject_id if association_state == "unique" else None,
                "candidate_feature_ids": [_feature_id(subjects.iloc[position], position) for position in candidates], "association": association_state})
        conditions += _attribute_evidence(subject_properties, subject_conditions)
        subject_metric = gpd.GeoSeries([subject_geometry], crs="EPSG:4326").to_crs(analysis_crs).iloc[0]
        # Unknown association cannot become a passing footprint relationship. Preserve its visual envelope.
        search_geometry = subject_metric.buffer(float(plan.distance_m) + margin) if plan.predicate == "within_distance" else subject_metric
        positions = reference_metric.sindex.query(search_geometry, predicate="intersects")
        references = []
        for position in positions:
            position = int(position)
            row, metric_row = reference.iloc[position], reference_metric.iloc[position]
            distance = float(subject_metric.distance(metric_row.geometry))
            if plan.predicate == "within_distance":
                if distance > float(plan.distance_m) + margin:
                    continue
                spatial_status = "unknown" if allowance is not None and distance + margin > float(plan.distance_m) else "pass"
            else:
                if not subject_metric.intersects(metric_row.geometry):
                    continue
                spatial_status = "pass"
                if association_state != "unique" and props.get("geometry_role", "visual_detection_envelope") not in {"object_mask", "segmented_object_boundary"}:
                    spatial_status = "unknown"
            reference_props = _properties(row, reference.geometry.name)
            checks = _attribute_evidence(reference_props, reference_conditions)
            state = condition_state(checks + [{"status": spatial_status}])
            rid = _feature_id(row, position)
            references.append({"id": rid, "position": position, "distance_m": distance, "status": state,
                "spatial_status": spatial_status, "attributes": checks, "properties": reference_props})
        qualified = [item for item in references if item["status"] == "pass"]
        pending = [item for item in references if item["status"] == "unknown"]
        witness_pool = qualified or pending or references
        witness = min(witness_pool, key=lambda item: item["distance_m"]) if witness_pool else None
        relation_status = "pass" if qualified else "unknown" if pending else "fail"
        if associated_layer and association_state != "unique":
            relation_status = "unknown"
        conditions.append({"condition": plan.predicate, "status": relation_status, "reference_layer": plan.reference_layer,
            "radius_m": plan.distance_m, "distance_m": round(witness["distance_m"], 3) if witness else None,
            "distance_basis": "associated_vector_footprint" if association_state == "unique" else "visual_detection_envelope",
            "distance_uncertainty_m": allowance, "uncertainty_status": "assumed_allowance" if allowance is not None else "not_estimated",
            "reference_feature_ids": [item["id"] for item in (qualified or pending)],
            "reference_candidates": [{key: value for key, value in item.items() if key != "position"} for item in references],
            "note": "Spatial predicates apply to the supplied geometry and frozen vector layer."})
        if witness and (association_state == "unique" or not associated_layer or qualified or pending):
            conditions += witness["attributes"]
        state = condition_state(conditions)
        if state == "fail":
            excluded += 1
            continue
        for item in (qualified or pending):
            row = reference.iloc[item["position"]]
            key = f"{plan.reference_layer}:{item['id']}"
            context_by_id[key] = {"type": "Feature", "id": key, "geometry": mapping(row.geometry),
                "properties": {"id": item["id"], "layer": plan.reference_layer, "role": "reference", "source_attributes": item["properties"],
                    "vector_extraction_date": bundle.manifest["vector_layers"][plan.reference_layer].get("extraction_date")}}
        observations, evidence_geometries = [], []
        for raw in props.get("observations") or [props]:
            observation_tile_id = raw.get("tile_id", props["tile_id"])
            observation_tile = bundle.tiles[bundle.tile_index[observation_tile_id]]
            observation_id = str(raw.get("detection_id", detection_id))
            observations.append({"detection_id": observation_id, "tile_id": observation_tile_id,
                "visual_score": raw.get("score"), "model_id": raw.get("model_id"), "model_revision": raw.get("model_revision"),
                "pixel_box": raw.get("pixel_box"), "original_width": observation_tile.get("width"), "original_height": observation_tile.get("height"),
                "inference_timestamp": raw.get("inference_timestamp") or observation_tile.get("inference_timestamp"),
                "asset_url": _asset_url(observation_tile_id), "source_sha256": observation_tile.get("source_sha256"),
                "imagery_date": observation_tile.get("acquisition_date"), "retrieval_similarity": scores.get(observation_tile_id)})
            raw_geometry = raw.get("geometry") or (detection["geometry"] if observation_id == detection_id else None)
            if raw_geometry:
                evidence_geometries.append({"type": "Feature", "id": "detection:" + observation_id, "geometry": raw_geometry,
                    "properties": {"id": observation_id, "role": "visual_detection", "tile_id": observation_tile_id,
                        "concept_id": plan.concept_id, "geometry_role": raw.get("geometry_role", "visual_detection_envelope")}})
        observation_scores = [item["retrieval_similarity"] for item in observations if item["retrieval_similarity"] is not None]
        feature = {"type": "Feature", "id": f"{associated_layer or 'detection'}:{subject_id}", "geometry": mapping(subject_geometry), "properties": {
            "id": subject_id, "detection_id": detection_id, "detection_ids": sorted({item["detection_id"] for item in observations}),
            "tile_id": props["tile_id"], "tile_ids": sorted({item["tile_id"] for item in observations}),
            "role": "match" if state == "pass" else "uncertain_candidate", "geometry_role": "associated_vector_footprint" if association_state == "unique" else "visual_detection_envelope",
            "evidence_status": "supported_by_available_evidence" if state == "pass" else "unknown", "association": association_state,
            "association_layer": associated_layer, "visual_concept": plan.concept_id, "visual_score": props.get("score"),
            "retrieval_similarity": max(observation_scores) if observation_scores else None,
            "retrieval_similarity_aggregation": "maximum_across_observations", "imagery_date": tile.get("acquisition_date"),
            "pixel_box": props.get("pixel_box"), "original_width": tile.get("width"), "original_height": tile.get("height"),
            "model_id": props.get("model_id"), "model_revision": props.get("model_revision"),
            "model_prompt": concept.get("prompt"), "inference_timestamp": props.get("inference_timestamp") or tile.get("inference_timestamp"),
            "imagery_date_status": tile.get("acquisition_date_status", "known" if tile.get("acquisition_date") else "unknown"),
            "source_image": tile.get("source_name"), "source_sha256": tile.get("source_sha256"), "subject_attributes": subject_properties,
            "vector_extraction_date": bundle.manifest.get("vector_layers", {}).get(associated_layer, {}).get("extraction_date"),
            "asset_url": _asset_url(props["tile_id"]), "conditions": conditions,
            "observations": observations, "evidence_geometries": evidence_geometries,
            "reason": "; ".join(item["condition"] + " is unresolved" for item in conditions if item["status"] == "unknown") if state != "pass" else None}}
        (records if state == "pass" else uncertain).append(feature)
    records, collapsed = _merge_observations(records)
    uncertain, uncertain_collapsed = _merge_observations(uncertain)
    sort_key = lambda feature: (feature["properties"]["retrieval_similarity"] is None, -(feature["properties"]["retrieval_similarity"] or 0), feature["id"])
    records.sort(key=sort_key)
    uncertain.sort(key=sort_key)
    base.update({"result_kind": "grounded_objects", "analysis_crs": analysis_crs.to_string(),
        "objects_evaluated": len(detections), "duplicate_observations_collapsed": collapsed + uncertain_collapsed,
        "objects_excluded": excluded, "total_matches": len(records), "total_uncertain": len(uncertain),
        "features": records[start:end], "uncertain_candidates": uncertain[start:end], "tile_candidates": [],
        "context_features": list(context_by_id.values()), "truncated": not all_results and (page + 1) * page_size < len(records),
        "uncertain_truncated": not all_results and (page + 1) * page_size < len(uncertain),
        "claim_scope": "Matches within the cached model detections and supplied vector features; no guarantee of real-world completeness or calibrated correctness."})
    base["timings_ms"].update({"spatial_and_evidence": round((perf_counter() - started) * 1000 - retrieval_ms, 2), "total": round((perf_counter() - started) * 1000, 2)})
    return json_value(base)
