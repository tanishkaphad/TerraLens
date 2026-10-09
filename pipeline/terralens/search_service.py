"""Catalog-driven search orchestration over durable, independently leased units."""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
from urllib.parse import quote

from shapely.geometry import box, mapping, shape
from shapely.ops import unary_union

from .bundle import json_value
from .catalog import SourceCatalog
from .grounding import GroundingError, GroundingService
from .search_graph import execute_graph, record_id
from .search_associations import build_association_links
from .search_grounding import CompleteGroundingService, assess_tile, ground_tile, rank_scenes
from .search_plan import SearchPlan, parse_search_plan
from .search_sources import ConfiguredOSMSource, SourceError, buffered_geometry, partition_geometry
from .search_store import SearchStore
from .search_workers import SearchWorkerRunner


class SearchServiceError(ValueError):
    def __init__(self, message: str, status_code: int = 422, *, details: dict | None = None):
        super().__init__(message)
        self.status_code = status_code
        self.details = details or {}


def _positive_setting(name: str, default: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
        if value < 1:
            raise ValueError()
        return value
    except ValueError:
        raise SearchServiceError(f"{name} must be a positive operational setting.", 503) from None


def _dates_match(date: str | None, scope: dict) -> bool | None:
    start, end = scope.get("start_date"), scope.get("end_date")
    if not start and not end:
        return True
    if not date:
        return None
    try:
        current = datetime.fromisoformat(date.replace("Z", "+00:00")).date()
        return (not start or current >= datetime.fromisoformat(start.replace("Z", "+00:00")).date()) and (
            not end or current <= datetime.fromisoformat(end.replace("Z", "+00:00")).date())
    except (ValueError, TypeError):
        return None


class SearchService:
    def __init__(self, store: SearchStore, *, catalog: SourceCatalog | None = None,
                 runner: SearchWorkerRunner | None = None, osm: ConfiguredOSMSource | None = None,
                 grounding: GroundingService | None = None):
        self.store = store
        self.catalog = catalog or SourceCatalog()
        self.osm = osm or ConfiguredOSMSource()
        self.grounding = grounding or CompleteGroundingService()
        self.runner = runner or SearchWorkerRunner(store,
            workers=_positive_setting("SEARCH_WORKERS", 4),
            max_pending=_positive_setting("SEARCH_MAX_PENDING", 32),
            max_units_per_run=int(os.environ["SEARCH_UNIT_BUDGET"]) if os.getenv("SEARCH_UNIT_BUDGET") else None,
            max_runtime_seconds=float(os.environ["SEARCH_TIME_BUDGET_SECONDS"]) if os.getenv("SEARCH_TIME_BUDGET_SECONDS") else None)

    def describe(self) -> dict:
        description = self.catalog.describe()
        external = self.osm.describe()
        description["layers"].update(external)
        if external:
            description["datasets"]["osm"] = {"id": "osm", "name": "Configured OpenStreetMap layers",
                "status": "available", "catalog_version": self.osm.revision, "dataset_version": self.osm.revision,
                "layers": {item["layer_id"]: item for item in external.values()},
                "coverage": {"status": "provider_dependent"}, "capabilities": {"mapped_records": True}}
        # Paths are server configuration, not part of the public or model catalog.
        for value in description["datasets"].values():
            value.pop("path", None)
        policies = {}
        for identifier, dataset in description["datasets"].items():
            for policy_id, policy in dataset.get("manifest", {}).get("association_policies", {}).items():
                if policy_id in policies and policies[policy_id] != policy:
                    raise SourceError("Association policy IDs must be unique across the catalog.")
                policies[policy_id] = deepcopy(policy)
        configured_policies = os.getenv("SEARCH_ASSOCIATION_POLICIES")
        if configured_policies:
            try:
                configured = json.loads(Path(configured_policies).expanduser().read_text())
                if not isinstance(configured, dict):
                    raise ValueError()
                policies.update(configured)
            except (OSError, ValueError, TypeError):
                raise SourceError("The server association-policy catalog is unreadable or invalid.") from None
        description["association_policies"] = policies
        description["search_pipeline"] = {"persistent": True, "complete_results": True,
            "workers": _positive_setting("SEARCH_WORKERS", 4), "dynamic_grounding": True,
            "natural_language": "configured_provider" if os.getenv("SEARCH_PARSER_ENABLED", "0") == "1" else "external_teammate_adapter",
            "structured_plan": True,
            "unsupported_processors": ["temporal_change", "network_analysis"],
            "scope_policy": "Explicit area or all loaded evidence; no implicit sample geography."}
        return json_value(description)

    def plan(self, query, *, previous_plan=None, scope: dict | None = None) -> SearchPlan:
        catalog = self.describe()
        context = deepcopy(catalog)
        if scope is not None:
            context["active_scope"] = scope
            selected_ids = set(scope.get("dataset_ids", []))
            if selected_ids:
                context["datasets"] = {key: value for key, value in context["datasets"].items() if key in selected_ids}
                context["layers"] = {key: value for key, value in context["layers"].items() if value.get("dataset_id") in selected_ids}
        # Metadata samples constrain request size, never result membership or
        # legal query values. The full observed schema stays in /api/catalog.
        sample_size = _positive_setting("SEARCH_PARSER_SAMPLE_VALUES", 20)
        for layer in context["layers"].values():
            observed = layer.get("observed_values", {})
            layer["observed_value_counts"] = {field: len(values) for field, values in observed.items()}
            layer["observed_values"] = {field: values[:sample_size] for field, values in observed.items()}
            layer["metadata_values_sampled"] = True
        for dataset in context["datasets"].values():
            dataset.pop("attribute_examples", None)
            dataset["layers"] = {value["layer_id"]: value for value in context["layers"].values()
                                 if value.get("dataset_id") == dataset.get("id")}
            # The parser needs source semantics, not entire prepared-artifact
            # manifests. Execution still uses the complete server-side catalog.
            dataset.pop("manifest", None)
        plan = parse_search_plan(query, context, previous_plan=previous_plan)
        if scope is not None:
            # Explicit UI/builder scope wins over model-generated extent.
            raw = plan.model_dump()
            raw["scope"] = scope
            plan = parse_search_plan(raw, context)
        return plan

    def create(self, query=None, *, plan=None, scope=None, parent_run_id=None, start=True) -> dict:
        parent = self.store.get_run(parent_run_id) if parent_run_id else None
        catalog = self.describe()
        resolved = self.plan(plan if plan is not None else query,
            previous_plan=parent["plan"] if parent else None, scope=scope)
        raw = resolved.model_dump()
        if resolved.status == "needs_clarification":
            raise SearchServiceError("The search needs clarification before execution.", details={
                "status": "needs_clarification", "clarifications": resolved.clarifications, "plan": raw})
        area = raw["scope"].get("geometry")
        if raw["scope"].get("place_query") and area is None:
            raise SearchServiceError("Resolve the named place to a boundary or provide an explicit area before spatial execution.",
                details={"status": "needs_clarification", "place_query": raw["scope"]["place_query"],
                         "plan": raw, "place_lookup_url": "/api/catalog/places"})
        configured_ids = raw["scope"].get("dataset_ids", [])
        if configured_ids:
            unknown = set(configured_ids) - set(catalog["datasets"])
            if unknown:
                raise SearchServiceError("The selected scope contains an unregistered dataset.")
        else:
            configured_ids = [key for key in catalog["datasets"]
                if raw["scope"]["mode"] != "all_loaded" or key != "osm"]
        if not configured_ids:
            raise SearchServiceError("No usable sources are registered. Add a source manifest to the configured catalog.",
                details={"status": "missing_evidence", "catalog_errors": catalog.get("errors", [])})
        entities = []
        for entity in raw["entities"]:
            dataset_id, layer_id = entity.get("dataset_id"), entity.get("layer_id")
            if layer_id and ":" in layer_id and layer_id in catalog["layers"]:
                layer_meta = catalog["layers"][layer_id]
                dataset_id, layer_id = layer_meta["dataset_id"], layer_meta["layer_id"]
            eligible = [key for key in configured_ids if dataset_id is None or key == dataset_id]
            if not eligible:
                raise SearchServiceError("An entity selects a dataset outside the declared search scope.")
            if entity["kind"] == "mapped":
                if not layer_id:
                    raise SearchServiceError("Mapped entities require a catalog-resolved layer, preserving the original description.")
                eligible = [key for key in eligible if layer_id in catalog["datasets"][key].get("layers", {})]
                if not eligible:
                    raise SearchServiceError(f"No selected source supplies mapped layer '{layer_id}'.")
            else:
                eligible = [key for key in eligible if key != "osm" and catalog["datasets"][key].get("status") == "available"]
            entity["layer_id"] = layer_id
            entities.append((entity, eligible))
        raw["entities"] = [item for item, _ids in entities]
        if not raw["output"].get("sort_by") and all(entity["kind"] == "scene" for entity in raw["entities"]
                if entity["id"] in raw["subject_entity_ids"]):
            raw["output"]["sort_by"] = "-retrieval_similarity"
        resolved = parse_search_plan(raw, catalog)
        versions = {key: catalog["datasets"][key].get("catalog_version") or
                    catalog["datasets"][key].get("dataset_version") for key in configured_ids}
        scope_geometry = shape(area) if area else None
        if scope_geometry is None:
            footprints = [shape(catalog["datasets"][key]["footprint"]) for key in configured_ids
                if catalog["datasets"][key].get("footprint")]
            scope_geometry = unary_union(footprints) if footprints else None
        effective_area = mapping(scope_geometry) if scope_geometry is not None and not scope_geometry.is_empty else None
        units = []
        chunk_size = _positive_setting("SEARCH_VECTOR_CHUNK_SIZE", 1000)
        subjects = set(raw["subject_entity_ids"])
        for entity, identifiers in entities:
            identifier = entity["id"]
            margin = max([float(clause.get("distance_m") or 0) for clause in raw["clauses"]
                if clause.get("reference_entity_id") == identifier] or [0])
            acquisition_area = buffered_geometry(area, margin) if area and margin else area
            count = 0
            declared_footprints, extent_unknown = [], False
            for dataset_id in identifiers:
                version = versions[dataset_id]
                if dataset_id == "osm":
                    if acquisition_area:
                        declared_footprints.append(shape(acquisition_area))
                    if acquisition_area is None:
                        raise SearchServiceError("Connected OSM extraction requires an explicit polygon area; all-loaded applies to local data.")
                    layer = catalog["layers"][f"osm:{entity['layer_id']}"]
                    size = float(layer.get("partition_size_m", os.getenv("SEARCH_PARTITION_SIZE_M", "2000")))
                    for cell in partition_geometry(acquisition_area, size):
                        unit_id = f"source:{identifier}:osm:{cell['id']}"
                        units.append({"id": unit_id, "payload": {"kind": "osm", "entity": entity,
                            "dataset_id": dataset_id, "version": version, "geometry": cell["geometry"],
                            "subject": identifier in subjects}})
                        count += 1
                    continue
                bundle = self.catalog.get_bundle(dataset_id)
                if bundle.catalog_version != version:
                    raise SearchServiceError("The source changed while planning; replan against the current catalog.", 409)
                if entity["kind"] == "mapped":
                    frame = bundle.layers[entity["layer_id"]].copy(deep=True)
                    layer_metadata = bundle.manifest.get("vector_layers", {}).get(entity["layer_id"], {})
                    source_metadata = layer_metadata.get("metadata", {})
                    declared_geometry = layer_metadata.get("coverage_geometry") or source_metadata.get("coverage_geometry")
                    declared_bounds = layer_metadata.get("coverage_bbox_wgs84") or source_metadata.get("coverage_bbox_wgs84")
                    if declared_geometry:
                        declared_footprints.append(shape(declared_geometry))
                    elif declared_bounds:
                        declared_footprints.append(box(*declared_bounds))
                    else:
                        extent_unknown = True
                    if acquisition_area:
                        selected = sorted(int(i) for i in frame.sindex.query(shape(acquisition_area), predicate="intersects"))
                    else:
                        selected = list(range(len(frame)))
                    # Empty layers still get a work unit and explicit coverage evidence.
                    for offset in range(0, max(1, len(selected)), chunk_size):
                        units.append({"id": f"source:{identifier}:{dataset_id}:{offset}", "payload": {
                            "kind": "vector", "entity": entity, "dataset_id": dataset_id, "version": version,
                            "indices": selected[offset:offset + chunk_size], "geometry": acquisition_area,
                            "subject": identifier in subjects, "last_chunk": offset + chunk_size >= len(selected)}})
                        count += 1
                else:
                    declared_footprints.extend(shape(tile["geometry"]) for tile in bundle.tiles)
                    selected_tiles = [tile for tile in bundle.tiles if not acquisition_area or
                                      shape(tile["geometry"]).intersects(shape(acquisition_area))]
                    if entity["kind"] == "scene":
                        eligible_ids = []
                        for tile in selected_tiles:
                            if _dates_match(tile.get("acquisition_date"), raw["scope"]) is True:
                                eligible_ids.append(tile["tile_id"])
                            else:
                                units.append({"id": f"date:{identifier}:{dataset_id}:{tile['tile_id']}", "payload": {
                                    "kind": "missing", "entity": entity, "geometry": tile["geometry"],
                                    "reason": "Imagery date is unknown or outside the requested period."}})
                                count += 1
                        if eligible_ids:
                            units.append({"id": f"source:{identifier}:{dataset_id}:scenes", "payload": {
                                "kind": "scene", "entity": entity, "dataset_id": dataset_id, "version": version,
                                "tile_ids": eligible_ids, "subject": identifier in subjects}})
                            count += 1
                        continue
                    for tile in selected_tiles:
                        units.append({"id": f"source:{identifier}:{dataset_id}:{tile['tile_id']}", "payload": {
                            "kind": "scene" if entity["kind"] == "scene" else "ground",
                            "entity": entity, "dataset_id": dataset_id, "version": version,
                            "tile_id": tile["tile_id"], "subject": identifier in subjects,
                            "scope_dates": raw["scope"], "policy": raw.get("execution_policy", {})}})
                        count += 1
            if not count:
                units.append({"id": f"missing:{identifier}", "payload": {"kind": "missing", "entity": entity,
                    "geometry": area, "reason": "No suitable source windows cover this entity in the declared scope."}})
            if acquisition_area and declared_footprints and not extent_unknown:
                uncovered = shape(acquisition_area).difference(unary_union(declared_footprints))
                if not uncovered.is_empty and uncovered.area > 0:
                    units.append({"id": f"coverage:{identifier}", "payload": {"kind": "missing", "entity": entity,
                        "geometry": mapping(uncovered), "reason": "Requested area extends beyond this entity's registered evidence coverage."}})
        dependencies = [item["id"] for item in units]
        units.append({"id": "evaluate", "depends_on": dependencies,
            "payload": {"kind": "evaluate", "plan": resolved.model_dump(), "scope_geometry": effective_area,
                        "catalog": catalog, "source_unit_ids": dependencies}})
        created = self.store.create_run(resolved.model_dump(), versions, units, parent_run_id=parent_run_id)
        if start:
            try:
                self.runner.submit(created["run_id"], self._processor(created["run_id"]))
            except Exception:
                self.store.pause(created["run_id"], reason="Worker queue is full or unavailable; resume this saved run.")
                raise
        return self.store.get_run(created["run_id"])

    def _processor(self, run_id):
        return lambda unit, checkpoint: self.process_unit(run_id, unit, checkpoint)

    def resume(self, run_id) -> dict:
        self.store.resume(run_id)
        try:
            self.runner.submit(run_id, self._processor(run_id))
        except Exception:
            self.store.pause(run_id, reason="Worker queue is full or unavailable; resume this saved run.")
            raise
        return self.store.get_run(run_id)

    def _gap(self, unit, reason: str, *, geometry=None):
        payload = unit["payload"].get("payload", unit["payload"])
        identifier = "gap:" + unit["unit_id"] + ":" + hashlib.sha256(reason.encode()).hexdigest()[:16]
        return {"id": identifier, "group": "coverage_gaps", "feature": {"type": "Feature", "id": identifier,
            "geometry": geometry or payload.get("geometry"), "properties": {"feature_role": "coverage_gap",
                "entity_id": payload.get("entity", {}).get("id"), "reason": reason, "evidence_status": "unknown"}}}

    def process_unit(self, run_id: str, unit: dict, checkpoint) -> dict:
        payload = unit["payload"].get("payload", unit["payload"])
        kind = payload["kind"]
        if kind == "evaluate":
            records = self.store.get_records(run_id, groups=["evidence"])
            candidates = {}
            for record in records:
                feature = record["feature"]
                candidates.setdefault(feature["properties"]["entity_id"], []).append(feature)
            run = self.store.get_run(run_id)
            coverage = self._graph_coverage(payload["plan"], payload["catalog"], run)
            associations = build_association_links(payload["plan"], candidates, payload["catalog"])
            result = execute_graph(payload["plan"], candidates, scope_geometry=payload["scope_geometry"],
                catalog=payload["catalog"], coverage=coverage, association_links=associations)
            counts = {group: sum(item["group"] == group for item in result) for group in
                      ("matches", "needs_review", "visual_candidates", "excluded")}
            aggregate = self._aggregate(payload["plan"], result, candidates, counts)
            if aggregate.get("missing_evidence"):
                result.append(self._gap(unit, aggregate["missing_evidence"], geometry=payload["scope_geometry"]))
            return {"records": result, "coverage": {"stage": "graph_evaluation", "objects_evaluated": len(result),
                "requested_geometry": payload["scope_geometry"], "complete_real_world_inventory": False,
                "data_coverage": coverage, "aggregate": aggregate,
                "association_links": len(associations),
                "threshold_policy": payload["plan"].get("execution_policy", {})}}
        if kind == "missing":
            return {"records": [self._gap(unit, payload["reason"])], "coverage": {
                "entity_id": payload["entity"]["id"], "status": "missing_evidence", "geometry": payload.get("geometry")}}
        dataset_id, entity = payload["dataset_id"], payload["entity"]
        if dataset_id == "osm":
            self.osm.refresh()
            current = {"catalog_version": self.osm.revision}
        else:
            current = self.catalog.datasets().get(dataset_id)
        if current is None or (current.get("catalog_version") or current.get("dataset_version")) != payload["version"]:
            raise SearchServiceError("Source data changed during this run. Saved evidence remains available; create a new run.", 409)
        coverage = {"entity_id": entity["id"], "dataset_id": dataset_id, "status": "processed", "data_completeness": "unknown"}
        extra_records = []
        if kind == "osm":
            features, information = self.osm.extract(entity["layer_id"], payload["geometry"], catalog_version=payload["version"])
            coverage.update(information)
            if information["omitted_geometry"]:
                extra_records.append(self._gap(unit, "Some provider records lack usable geometry.", geometry=payload["geometry"]))
        else:
            bundle = self.catalog.get_bundle(dataset_id)
            if bundle.catalog_version != payload["version"]:
                raise SearchServiceError("Source changed while acquiring its data; resume only against the saved source version.", 409)
            if kind == "vector":
                frame = bundle.layers[entity["layer_id"]].copy(deep=True)
                metadata = deepcopy(bundle.manifest.get("vector_layers", {}).get(entity["layer_id"], {}))
                features = []
                for position in payload["indices"]:
                    row = frame.iloc[position]
                    props = json_value(row.drop(labels=[frame.geometry.name]).to_dict())
                    stable = next((props[key] for key in ("source_feature_id", "osm_id", "osm_way_id", "id", "@id")
                                   if props.get(key) is not None), position)
                    identifier = f"{dataset_id}:{entity['layer_id']}:{stable}"
                    features.append({"type": "Feature", "id": identifier, "geometry": mapping(row.geometry),
                        "properties": {**props, "source_attributes": props, "source_metadata": metadata,
                            "source_feature_id": stable, "layer_id": entity["layer_id"],
                            "geometry_role": "source_vector_feature", "snapshot_date": metadata.get("source_date") or metadata.get("metadata", {}).get("source_date") or metadata.get("extraction_date"),
                            "retrieved_at": metadata.get("retrieved_at") or metadata.get("extraction_date"),
                            "reference_coverage_complete": metadata.get("reference_coverage_complete", False)}})
                coverage.update(geometry=payload.get("geometry"), records_received=len(features),
                    source_enumeration="all_registered_records", source_metadata=metadata)
                geometry_counts = getattr(bundle, "vector_geometry_metadata", {}).get(entity["layer_id"], {})
                if payload.get("last_chunk") and geometry_counts.get("omitted_geometry", 0):
                    extra_records.append(self._gap(unit, "Source records with null or empty geometry cannot be located.", geometry=payload.get("geometry")))
                    coverage.update(status="missing_geometry", **geometry_counts)
                scope_dates = self.store.get_run(run_id)["plan"]["scope"]
                source_date = metadata.get("source_date") or metadata.get("metadata", {}).get("source_date") or metadata.get("snapshot_date") or metadata.get("extraction_date")
                if _dates_match(source_date, scope_dates) is not True:
                    reason = "The mapped snapshot date is unknown or outside the requested period; historical feature conditions remain unresolved."
                    for feature in features:
                        feature["properties"].update(scope_time_status="unknown", scope_time_reason=reason)
                    extra_records.append(self._gap(unit, reason, geometry=payload.get("geometry")))
                    coverage.update(status="unsuitable_date")
            elif kind == "scene":
                features = rank_scenes(bundle, entity.get("visual_query") or entity["description"], payload["tile_ids"])
                coverage.update(getattr(features, "metadata", {}))
                coverage.update(geometry=mapping(unary_union([shape(bundle.tiles[bundle.tile_index[tile_id]]["geometry"])
                    for tile_id in payload["tile_ids"]])), source_enumeration="ranked_registered_windows")
            else:
                tile = bundle.tiles[bundle.tile_index[payload["tile_id"]]]
                date_status = _dates_match(tile.get("acquisition_date"), payload.get("scope_dates", {}))
                if date_status is not True:
                    reason = "Imagery date is unknown for the requested period." if date_status is None else "Imagery is outside the requested period."
                    return {"records": [self._gap(unit, reason, geometry=tile["geometry"])],
                        "coverage": {**coverage, "status": "unsuitable_date", "geometry": tile["geometry"]}}
                if kind == "scene":
                    features = rank_scenes(bundle, entity.get("visual_query") or entity["description"], [payload["tile_id"]])
                    coverage.update(geometry=tile["geometry"], source_enumeration="ranked_registered_window")
                else:
                    suitability = assess_tile(bundle, payload["tile_id"])
                    if not suitability.get("available"):
                        return {"records": [self._gap(unit, suitability.get("reason") or "Image window is unavailable for grounding.", geometry=tile["geometry"])],
                            "coverage": {**coverage, "status": "unsuitable", "geometry": tile["geometry"]}}
                    policy = payload.get("policy", {}).get("threshold_policy")
                    try:
                        features = ground_tile(bundle, payload["tile_id"], entity.get("visual_query") or entity["description"],
                            threshold_policy=policy, service=self.grounding)
                    except GroundingError as exc:
                        # Preserve a readable scene candidate when grounding is
                        # unavailable, with the missing processor kept explicit.
                        try:
                            features = rank_scenes(bundle, entity.get("visual_query") or entity["description"], [payload["tile_id"]])
                        except (GroundingError, ValueError):
                            features = []
                        extra_records.append(self._gap(unit, str(exc), geometry=tile["geometry"]))
                        coverage.update(status="missing_processor", missing_reason=str(exc))
                    missing_reason = coverage.get("missing_reason")
                    coverage.update(getattr(features, "metadata", {}))
                    coverage.update(geometry=tile["geometry"], source_enumeration="ranked_registered_window" if missing_reason else "grounded_registered_window")
                    if missing_reason:
                        coverage.update(status="missing_processor", missing_reason=missing_reason)
        if kind == "osm":
            scope_dates = self.store.get_run(run_id)["plan"]["scope"]
            if _dates_match(coverage.get("snapshot_date"), scope_dates) is not True:
                reason = "The provider snapshot does not establish mapped conditions in the requested period."
                for feature in features:
                    feature["properties"].update(scope_time_status="unknown", scope_time_reason=reason)
                extra_records.append(self._gap(unit, reason, geometry=payload["geometry"]))
                coverage.update(status="unsuitable_date")
        records = extra_records
        for feature in features:
            feature = deepcopy(feature)
            props = feature.setdefault("properties", {})
            props.update(entity_id=entity["id"], dataset_id=dataset_id, dataset_version=payload["version"])
            if props.get("asset_url", "").startswith("/api/assets/") and "?" not in props["asset_url"]:
                props["asset_url"] += "?dataset_id=" + quote(dataset_id, safe="")
            if kind == "ground" and props.get("geometry_role") not in {"image_tile_coverage", "scene_footprint"}:
                props.setdefault("visual_evidence", [{"query": entity.get("visual_query") or entity["description"],
                    "status": "pass", "evidence_type": "model_detection", "model_id": props.get("model_id"),
                    "raw_score": props.get("score"), "calibration": "uncalibrated", "human_review": "not_reviewed"}])
            records.append({"id": f"evidence:{entity['id']}:{feature['id']}", "group": "evidence", "feature": feature})
            if payload.get("subject"):
                provisional = deepcopy(feature)
                provisional["properties"].update(evidence_status="provisional", expression_status="unknown",
                    pending_conditions=True, feature_role="subject_candidate")
                records.append({"id": record_id(entity["id"], feature),
                    "group": "visual_candidates" if kind in {"ground", "scene"} else "needs_review", "feature": provisional})
        checkpoint({"records_received": len(features), "source_version": payload["version"]})
        return {"records": records, "coverage": coverage}

    @staticmethod
    def _aggregate(plan, records, candidates, counts):
        output = plan["output"]
        grouping_entity, field = output.get("group_by_entity_id"), output.get("group_by_field")
        result = {"counts": counts, "basis": "saved_evidence_entities", "population_completeness": "not_established"}
        if not grouping_entity and not field:
            if output["kind"] == "statistics":
                result["missing_evidence"] = "A statistical measurement must be specified and supported; only saved-entity counts are available."
            return result
        boundaries = candidates.get(grouping_entity, []) if grouping_entity else []
        if grouping_entity and not boundaries:
            return {**result, "missing_evidence": "The requested grouping entities have no usable evidence.", "grouping_status": "unknown"}
        if grouping_entity and any(shape(item["geometry"]).geom_type not in {"Polygon", "MultiPolygon"} for item in boundaries):
            return {**result, "missing_evidence": "Geographic grouping requires polygon boundaries for the selected grouping entity.", "grouping_status": "unknown"}
        rows = {}
        for record in records:
            if record["group"] == "excluded":
                continue
            feature = record["feature"]
            props = feature.get("properties", {})
            if grouping_entity:
                point = shape(feature["geometry"]).representative_point()
                matched = [boundary for boundary in boundaries if shape(boundary["geometry"]).covers(point)]
                if len(matched) == 1:
                    boundary = matched[0]
                    boundary_props = boundary.get("properties", {})
                    value = boundary_props.get(field, boundary_props.get("source_attributes", {}).get(field)) if field else boundary["id"]
                    label = {"value": value, "group_entity_id": grouping_entity,
                             "reference_feature_ids": [boundary["id"]], "status": "known" if value is not None else "unknown_attribute"}
                else:
                    label = {"value": None, "group_entity_id": grouping_entity,
                             "reference_feature_ids": sorted(item["id"] for item in matched),
                             "status": "ambiguous_boundary" if matched else "unassigned"}
            else:
                value = props.get(field, props.get("source_attributes", {}).get(field))
                label = {"value": value, "field": field, "status": "known" if value is not None else "unknown_attribute"}
            key = json.dumps(label, sort_keys=True)
            row = rows.setdefault(key, {**label, "counts": {name: 0 for name in counts}})
            row["counts"][record["group"]] += 1
        result.update(rows=[rows[key] for key in sorted(rows)], group_by_field=field, group_by_entity_id=grouping_entity,
                      grouping_basis="representative_point_in_boundary" if grouping_entity else "source_attribute")
        if output["kind"] == "statistics":
            result["missing_evidence"] = "Grouped saved-entity counts are available; an additional statistical measurement needs a specified supported processor."
        return result

    @staticmethod
    def _graph_coverage(plan, catalog, run):
        # Actual layer completeness is independent of finishing every work unit.
        result = {}
        for entity in plan["entities"]:
            layer_id = entity.get("layer_id")
            entries = [details for details in catalog["layers"].values()
                if details.get("layer_id") == layer_id and
                (entity.get("dataset_id") is None or details.get("dataset_id") == entity["dataset_id"])]
            complete = bool(entries) and all(details.get("reference_coverage_complete") is True for details in entries)
            entity_units = [item["coverage"] for item in run["coverage"]["units"]
                            if item["coverage"].get("entity_id") == entity["id"]]
            if any(item.get("status") in {"missing_evidence", "unsuitable", "unsuitable_date", "missing_processor"}
                   or item.get("omitted_geometry", 0) for item in entity_units):
                complete = False
            if entity["kind"] != "mapped":
                complete = False
            result[entity["id"]] = {"complete": complete, "reference_coverage_complete": complete,
                "absence_supported": False, "status": "source_declared_complete" if complete else "unknown",
                "processing_finished": True}
        return result
