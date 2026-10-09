"""Evidence retrieval, object association and metric spatial execution."""

from __future__ import annotations

from typing import Any

import geopandas as gpd
from shapely.geometry import shape, mapping

from .bundle import Bundle
from .planner import QueryPlan


def _feature_id(row: Any, index: Any) -> str:
    for key in ("osm_id", "osm_way_id", "id", "@id"):
        if key in row and row[key] is not None:
            return str(row[key])
    return str(index)


def _attribute_value(properties: dict[str, Any], field: str) -> str | None:
    value = properties.get(field)
    return None if value is None else str(value)


def _test_attribute(properties: dict[str, Any], condition) -> tuple[str, str | None]:
    actual = _attribute_value(properties, condition.field)
    if actual is None or actual.strip() == "":
        return "unknown", None
    if condition.operator == "equals":
        passed = actual.casefold() == condition.value.casefold()
    else:
        passed = condition.value.casefold() in actual.casefold()
    return ("pass" if passed else "fail"), actual


def execute(bundle: Bundle, plan: QueryPlan, mode: str = "catalog_scan", page_size: int = 50, page: int = 0) -> dict[str, Any]:
    concept = bundle.manifest["supported_concepts"][plan.concept_id]
    associated_layer = concept.get("association_layer")
    if associated_layer and associated_layer not in bundle.layers:
        associated_layer = None
    reference = bundle.layers.get(plan.reference_layer)
    if reference is None:
        raise ValueError(f"Reference layer '{plan.reference_layer}' is unavailable in this bundle.")

    text_vector = bundle.encode_text(plan.visual_query)
    scores: dict[str, float] = {}
    if text_vector is not None:
        similarities = bundle.embeddings @ text_vector
        scores = {tile["tile_id"]: float(similarities[i]) for i, tile in enumerate(bundle.tiles)}

    if mode not in {"catalog_scan", "explore"}:
        raise ValueError("Mode must be 'catalog_scan' or 'explore'.")
    if mode == "explore":
        if text_vector is None:
            raise ValueError("Exploration mode needs a compatible local text encoder; use catalog scan for the cached concept.")
        tile_order = sorted(scores, key=scores.get, reverse=True)[:max(page_size * 4, 100)]
        allowed_tiles = set(tile_order)
    else:
        allowed_tiles = set(bundle.tile_index)

    image_detections = [
        feature for feature in bundle.detections
        if feature.get("properties", {}).get("concept_id") == plan.concept_id
        and feature.get("properties", {}).get("tile_id") in allowed_tiles
    ]
    reference_wgs = reference[reference.geometry.notna() & ~reference.geometry.is_empty].copy()
    analysis_crs = reference_wgs.estimate_utm_crs()
    if analysis_crs is None:
        raise ValueError("Could not estimate a local metric CRS for spatial checks.")
    reference_metric = reference_wgs.to_crs(analysis_crs)
    records: list[dict[str, Any]] = []
    uncertain: list[dict[str, Any]] = []
    context_by_id: dict[str, dict[str, Any]] = {}

    if associated_layer:
        subjects = bundle.layers[associated_layer]
        subjects = subjects[subjects.geometry.notna() & ~subjects.geometry.is_empty].copy()
        subjects_metric = subjects.to_crs(analysis_crs)
    else:
        subjects = None
        subjects_metric = None

    ref_ids = [_feature_id(row, idx) for idx, row in reference_wgs.iterrows()]
    ref_props = [row.drop(labels=[reference_wgs.geometry.name], errors="ignore").to_dict() for _, row in reference_wgs.iterrows()]
    for detection in image_detections:
        props = detection.get("properties", {})
        try:
            detected_geometry = shape(detection["geometry"])
        except Exception:
            continue
        if detected_geometry.is_empty:
            continue
        tile_id = props["tile_id"]
        subject_id = props.get("detection_id", tile_id)
        subject_geometry = detected_geometry
        subject_properties: dict[str, Any] = {}
        association_state = "not_required"
        subject_feature_id = str(subject_id)
        if subjects is not None and subjects_metric is not None:
            detection_metric = gpd.GeoSeries([detected_geometry], crs="EPSG:4326").to_crs(analysis_crs).iloc[0]
            center = detection_metric.representative_point()
            overlaps = subjects_metric.geometry.intersection(detection_metric).area / max(detection_metric.area, 1e-9)
            candidates = [
                idx for idx, fraction in overlaps.items()
                if fraction >= float(concept.get("association_overlap_min", 0.5))
                and subjects_metric.loc[idx].geometry.covers(center)
            ]
            if len(candidates) != 1:
                association_state = "ambiguous" if candidates else "unmatched"
            else:
                idx = candidates[0]
                row = subjects.loc[idx]
                subject_geometry = row.geometry
                subject_properties = row.drop(labels=[subjects.geometry.name], errors="ignore").to_dict()
                subject_feature_id = _feature_id(row, idx)
                association_state = "unique"

        if associated_layer and association_state != "unique":
            uncertain.append({
                "detection_id": str(subject_id), "tile_id": tile_id,
                "reason": f"Detection-to-{associated_layer} association is {association_state}.",
                "visual_score": props.get("score"), "asset_url": f"/api/assets/{tile_id}",
            })
            continue

        conditions: list[dict[str, Any]] = [{
            "condition": "visual_detection", "status": "pass",
            "concept_id": plan.concept_id, "score": props.get("score"),
            "checkpoint": props.get("model_id"),
        }]
        if associated_layer:
            conditions.append({"condition": "feature_association", "status": "pass", "layer": associated_layer, "feature_id": subject_feature_id})
        roles = {"subject": subject_properties, "reference": {}}
        qualified_refs: list[tuple[str, dict[str, Any], Any]] = []
        for idx, row in reference_wgs.iterrows():
            rid = ref_ids[list(reference_wgs.index).index(idx)]
            qualified_refs.append((rid, ref_props[list(reference_wgs.index).index(idx)], reference_metric.loc[idx].geometry))

        spatial_matches: list[tuple[str, dict[str, Any], Any, float]] = []
        subject_metric = gpd.GeoSeries([subject_geometry], crs="EPSG:4326").to_crs(analysis_crs).iloc[0]
        for rid, rprops, rgeom_metric in qualified_refs:
            if plan.predicate == "within_distance":
                distance = float(subject_metric.distance(rgeom_metric))
                if distance <= float(plan.distance_m):
                    spatial_matches.append((rid, rprops, rgeom_metric, distance))
            else:
                if subject_metric.intersects(rgeom_metric):
                    spatial_matches.append((rid, rprops, rgeom_metric, float(subject_metric.distance(rgeom_metric))))

        for condition in plan.attributes:
            values = roles.get(condition.role, {})
            status, actual = _test_attribute(values, condition)
            conditions.append({
                "condition": f"{condition.role}_attribute", "status": status,
                "field": condition.field, "operator": condition.operator,
                "expected": condition.value, "observed": actual,
            })

        if spatial_matches:
            witness = min(spatial_matches, key=lambda item: item[3])
            conditions.append({
                "condition": plan.predicate, "status": "pass", "reference_layer": plan.reference_layer,
                "distance_m": round(witness[3], 2), "radius_m": plan.distance_m,
                "reference_feature_ids": [match[0] for match in spatial_matches],
            })
            context_by_id.update({
                f"{plan.reference_layer}:{rid}": {
                    "type": "Feature", "geometry": mapping(reference_wgs.loc[idx].geometry),
                    "properties": {"id": rid, "layer": plan.reference_layer, "role": "reference"},
                }
                for idx, rid in zip(reference_wgs.index, ref_ids)
                if any(match[0] == rid for match in spatial_matches)
            })
        else:
            conditions.append({
                "condition": plan.predicate, "status": "fail", "reference_layer": plan.reference_layer,
                "radius_m": plan.distance_m, "reference_feature_ids": [],
                "note": "No feature in the supplied reference layer satisfies this condition.",
            })

        state = "pass" if all(item["status"] == "pass" for item in conditions) else (
            "unknown" if any(item["status"] == "unknown" for item in conditions) else "fail"
        )
        if state != "pass":
            if state == "unknown":
                uncertain.append({"detection_id": str(subject_id), "tile_id": tile_id, "reason": "One or more requested attributes are missing.", "conditions": conditions, "asset_url": f"/api/assets/{tile_id}"})
            continue

        tile = bundle.tiles[bundle.tile_index[tile_id]]
        score = scores.get(tile_id)
        geometry_role = "associated_vector_footprint" if associated_layer else "visual_detection_envelope"
        result_geometry = subject_geometry
        records.append({
            "type": "Feature", "geometry": mapping(result_geometry),
            "properties": {
                "id": str(subject_feature_id), "detection_id": str(subject_id), "tile_id": tile_id,
                "geometry_role": geometry_role, "evidence_status": "supported_by_available_evidence",
                "association": association_state, "visual_concept": plan.concept_id,
                "visual_score": props.get("score"), "retrieval_similarity": score,
                "imagery_date": tile.get("acquisition_date"), "source_image": tile.get("source_name"),
                "asset_url": f"/api/assets/{tile_id}", "conditions": conditions,
            },
        })

    records.sort(key=lambda feature: (
        feature["properties"]["retrieval_similarity"] is None,
        -(feature["properties"]["retrieval_similarity"] or 0.0),
    ))
    start = page * page_size
    total = len(records)
    return {
        "plan": plan.model_dump(), "dataset_version": bundle.manifest.get("dataset_version"),
        "mode": mode, "tiles_processed": len(allowed_tiles), "tiles_available": len(bundle.tiles),
        "objects_evaluated": len(image_detections), "total_matches": total,
        "page": page, "page_size": page_size, "truncated": start + page_size < total,
        "text_similarity_enabled": text_vector is not None,
        "text_encoder_status": bundle.text_model_error or ("ready" if text_vector is not None else "checkpoint_not_configured"),
        "features": records[start:start + page_size], "uncertain_candidates": uncertain[:page_size],
        "context_features": list(context_by_id.values()),
        "coverage": bundle.manifest.get("coverage", {}),
        "claim_scope": "Matches within the precomputed detections and vector features in this frozen bundle; not a completeness claim about the real world.",
    }
