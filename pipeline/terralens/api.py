"""FastAPI search, bounded imagery map, evidence assets and complete GeoJSON export."""
from __future__ import annotations
import mimetypes
import json
import math
import os
import re
from functools import lru_cache
from time import monotonic
from pathlib import Path
from typing import Literal
from urllib.parse import quote
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from shapely.geometry import box, mapping, shape
from .bundle import Bundle, BundleError, json_value
from .engine import execute
from .grounding import GroundingError, GroundingService, grounding_status
from .planner import PlanError, QueryPlan, plan_query, validate_plan

app = FastAPI(title="TerraLens", version="0.3.0")
STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

class SearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str | None = Field(default=None, min_length=3, max_length=500)
    plan: QueryPlan | None = None
    dataset_version: str | None = Field(default=None, max_length=160)
    previous_plan: QueryPlan | None = None
    mode: Literal["catalog_scan", "explore", "ground"] = "catalog_scan"
    dataset_id: str | None = Field(default=None, max_length=40, pattern=r"^[a-z0-9_-]+$")
    grounding_run_id: str | None = Field(default=None, pattern=r"^[a-f0-9]{32}$")
    grounding_tile_limit: int = Field(default=2, ge=1, le=4, strict=True)
    grounding_threshold: float = Field(default=0.1, ge=0.05, le=0.9, allow_inf_nan=False)
    page: int = Field(default=0, ge=0, le=1000)
    page_size: int = Field(default=50, ge=1, le=100)

class ExportRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str | None = Field(default=None, min_length=3, max_length=500)
    plan: QueryPlan | None = None
    previous_plan: QueryPlan | None = None
    mode: Literal["catalog_scan", "explore", "ground"] = "catalog_scan"
    dataset_id: str | None = Field(default=None, max_length=40, pattern=r"^[a-z0-9_-]+$")
    grounding_run_id: str | None = Field(default=None, pattern=r"^[a-f0-9]{32}$")
    include_uncertain: bool = False
    dataset_version: str | None = Field(default=None, max_length=160)
    # Accepted for parity with search; export deliberately evaluates all results.
    page: int = Field(default=0, ge=0, le=1000)
    page_size: int = Field(default=50, ge=1, le=100)

def _dataset_registry() -> tuple[dict[str, dict], str]:
    """Only operator-configured local bundles; clients cannot supply filesystem paths."""
    data = Path(__file__).resolve().parents[1] / "data"
    reference = data / "spacenet-grounded"
    if not (reference / "manifest.json").is_file():
        reference = data / "spacenet-urban"
    if not (reference / "manifest.json").is_file():
        reference = data / "spacenet-sample"
    paths = {"spacenet": reference, "demo": data / "demo"}
    configured = os.getenv("BUNDLE_DIR")
    default = "spacenet" if (paths["spacenet"] / "manifest.json").is_file() else "demo"
    if configured:
        configured_path = Path(configured).expanduser().resolve()
        default = next((key for key, path in paths.items() if path.resolve() == configured_path), "configured")
        paths[default] = configured_path
    datasets = {}
    for identifier, path in paths.items():
        try:
            manifest = json.loads((path / "manifest.json").read_text())
        except (OSError, ValueError):
            continue
        datasets[identifier] = {"id": identifier, "path": path, "name": manifest.get("area_name") or identifier,
            "source": "SpaceNet 1 reference sample" if identifier == "spacenet" else "Additional imagery and vector sample" if identifier == "demo" else "Configured local evidence bundle",
            "default": identifier == default, "dataset_version": manifest.get("dataset_version")}
    return datasets, default

@lru_cache(maxsize=4)
def get_bundle(dataset_id: str | None = None) -> Bundle:
    datasets, default = _dataset_registry()
    identifier = dataset_id or default
    if identifier not in datasets:
        raise BundleError("The requested local dataset is not prepared.")
    return Bundle(datasets[identifier]["path"])

@lru_cache(maxsize=1)
def get_grounding_service() -> GroundingService:
    return GroundingService()

def _bundle(dataset_id: str | None = None) -> Bundle:
    if dataset_id is not None and dataset_id not in _dataset_registry()[0]:
        raise HTTPException(status_code=404, detail="Unknown or unprepared dataset.")
    try:
        return get_bundle(dataset_id) if dataset_id is not None else get_bundle()
    except (BundleError, OSError, ValueError) as exc:
        raise HTTPException(status_code=503, detail=f"Evidence bundle unavailable: {exc}") from exc

def _dataset_assets(value, dataset_id: str | None):
    if not dataset_id:
        return value
    if isinstance(value, dict):
        return {key: _dataset_assets(item, dataset_id) for key, item in value.items()}
    if isinstance(value, list):
        return [_dataset_assets(item, dataset_id) for item in value]
    if isinstance(value, str) and value.startswith("/api/assets/") and "?" not in value:
        return value + "?dataset_id=" + quote(dataset_id, safe="")
    return value

def _run(request, *, all_results: bool = False):
    bundle = _bundle(request.dataset_id)
    if getattr(request, "dataset_version", None) is not None and request.dataset_version != bundle.manifest.get("dataset_version"):
        raise HTTPException(status_code=409, detail="The displayed dataset version differs from the current bundle; rerun the search before exporting.")
    try:
        if getattr(request, "plan", None) is not None:
            plan = validate_plan(request.plan, bundle.manifest, request.mode)
        elif request.query:
            previous = request.previous_plan.model_dump() if request.previous_plan else None
            plan = plan_query(request.query, bundle.manifest, previous, request.mode)
        else:
            raise PlanError("Provide a validated plan or a natural-language query.")
        run = None
        cache_hit = False
        if request.mode == "ground":
            service = get_grounding_service()
            if request.grounding_run_id:
                run = service.get(request.grounding_run_id, bundle, plan)
                cache_hit = True
            elif all_results:
                raise GroundingError("Run detection first, then export its displayed evidence run.", 409)
            else:
                run, cache_hit = service.create(bundle, plan, tile_limit=request.grounding_tile_limit,
                    threshold=request.grounding_threshold)
        result = execute(run.bundle if run else bundle, plan, request.mode, request.page_size, request.page, all_results=all_results)
        result["dataset_id"] = request.dataset_id
        if run:
            result.update({"grounding_run_id": run.identifier, "grounding": {**run.metadata, "cache_hit": cache_hit,
                    "expires_in_seconds": max(0, int(run.expires_at - monotonic()))},
                "tiles_processed": run.metadata["tiles_processed"], "tiles_considered": len(bundle.tiles),
                "candidate_selection_truncated": run.metadata["tiles_processed"] < len(bundle.tiles),
                "claim_scope": run.metadata["scope"], "visual_concept": plan.visual_query})
        kind = "live_model_detection" if run else "mapped_feature" if plan.query_type == "vectors" else "image_similarity" if request.mode == "explore" else "cached_model_detection"
        result["evidence_kind"] = kind
        for feature in result["features"] + result["uncertain_candidates"]:
            feature["properties"]["evidence_kind"] = kind
            if run:
                feature["properties"]["grounding_run_id"] = run.identifier
        return _dataset_assets(result, request.dataset_id)
    except GroundingError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
    except (PlanError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

@app.get("/", include_in_schema=False)
def home():
    return FileResponse(STATIC_DIR / "index.html")

@app.get("/health")
def health():
    return {"status": "alive", "service": "terralens"}

@app.get("/api/datasets")
def datasets():
    entries, default = _dataset_registry()
    return {"datasets": [{key: value for key, value in entry.items() if key != "path"} for entry in entries.values()],
        "default_dataset_id": default}

@app.get("/ready")
def ready(dataset_id: str | None = None):
    bundle = _bundle(dataset_id)
    retrieval = bundle.retrieval_status()
    grounded = [key for key, details in bundle.manifest.get("supported_concepts", {}).items() if details.get("grounding_available", True)]
    has_vectors = any(not frame.empty for frame in bundle.layers.values())
    full = retrieval["enabled"] and bool(grounded) and has_vectors
    return {"status": "ready" if full else "degraded", "bundle_loaded": True,
        "dataset_version": bundle.manifest.get("dataset_version"), "tiles": len(bundle.tiles), "detections": len(bundle.detections),
        "layers": sorted(bundle.layers), "retrieval": retrieval, "grounded_concepts": grounded,
        "capabilities": {"live_text_image_retrieval": retrieval["enabled"], "cached_object_spatial_search": bool(grounded) and has_vectors,
            "live_object_grounding": grounding_status(bundle)["available"], "mapped_feature_search": has_vectors},
        "note": "Precomputed model grounding; live text ranking and spatial checks." if full else "Evidence bundle loaded; full multimodal readiness is incomplete. See capabilities."}

@app.get("/api/capabilities")
def capabilities(dataset_id: str | None = None):
    bundle = _bundle(dataset_id)
    retrieval = bundle.retrieval_status()
    grounding = grounding_status(bundle)
    tasks = []
    for identifier, concept in bundle.manifest.get("supported_concepts", {}).items():
        if concept.get("grounding_available", True):
            aliases = concept.get("aliases", [])
            subject = aliases[0] if aliases else identifier.replace("_", " ")
            tasks.append({"query": "Find " + subject, "mode": "catalog_scan", "evidence_kind": "cached_model_detection"})
    for layer in bundle.layers:
        collision = any(layer.replace("_", " ").casefold() in {str(value).casefold() for value in
            [identifier.replace("_", " "), concept.get("prompt", ""), *concept.get("aliases", [])]}
            for identifier, concept in bundle.manifest.get("supported_concepts", {}).items())
        tasks.append({"query": "Find " + ("mapped " if collision else "") + layer,
            "mode": "catalog_scan", "evidence_kind": "mapped_feature"})
    if {"buildings", "roads"} <= set(bundle.layers):
        tasks.append({"query": "Find buildings within 100 metres of roads", "mode": "catalog_scan", "evidence_kind": "mapped_feature"})
        buildings, roads = bundle.layers["buildings"], bundle.layers["roads"]
        if "building" in buildings and "highway" in roads and buildings["building"].eq("house").any() and roads["highway"].eq("residential").any():
            tasks.insert(0, {"query": "Find buildings with building=house within 100 metres of roads with highway=residential",
                "mode": "catalog_scan", "evidence_kind": "mapped_feature"})
    if grounding["available"] and retrieval["enabled"]:
        tasks.append({"query": "Find trees within 100 metres of roads" if "roads" in bundle.layers else "Find trees",
            "mode": "ground", "evidence_kind": "live_model_detection"})
    tasks.append({"query": "Find densely built neighbourhoods", "mode": "explore", "evidence_kind": "image_similarity"})
    return json_value({"dataset_version": bundle.manifest.get("dataset_version"), "area_name": bundle.manifest.get("area_name"),
        "coverage": bundle.manifest.get("coverage", {}), "supported_concepts": bundle.manifest.get("supported_concepts", {}),
        "vector_layers": bundle.manifest.get("vector_layers", {}), "queryable_attributes": bundle.manifest.get("queryable_attributes", []),
        "text_similarity_enabled": retrieval["enabled"], "retrieval": retrieval,
        "parser": "schema_constrained_language_model" if os.getenv("OPENAI_API_KEY") and os.getenv("OPENAI_MODEL") else "conservative_templates",
        "supported_predicates": bundle.manifest.get("supported_predicates", ["within_distance", "intersects"]),
        "limits": {"maximum_distance_m": min(5000, bundle.manifest.get("limits", {}).get("maximum_distance_m", 5000)),
            "page_size": 100, "explore_tile_limit": min(1000, bundle.manifest.get("limits", {}).get("explore_tile_limit", 100)),
            "map_tile_limit": 64, "map_vector_limit": 5000},
        "provenance": bundle.provenance(), "live_grounding": grounding["available"] and retrieval["enabled"],
        "grounding": grounding, "task_catalog": tasks, "dataset_id": dataset_id,
        "query_families": ["mapped_features", "cached_objects", "image_scenes", "new_visual_objects"],
        "maximum_spatial_clauses": 4})

@app.post("/api/query")
def query(request: SearchRequest):
    return _run(request)

@app.post("/api/export")
def export(request: ExportRequest):
    result = _run(request, all_results=True)
    features = list(result["features"])
    if request.include_uncertain:
        features.extend(result["uncertain_candidates"])
    if result["result_kind"] == "tile_retrieval":
        features.extend(result["tile_candidates"])
    related = {}
    reference_ids = set()
    for feature in features:
        for evidence in feature["properties"].get("evidence_geometries", []):
            related[evidence["id"]] = evidence
        for condition in feature["properties"].get("conditions", []):
            for identifier in condition.get("reference_feature_ids", []):
                reference_ids.add(f"{condition.get('reference_layer')}:{identifier}")
    for feature in result["context_features"]:
        if feature["id"] in reference_ids:
            related[feature["id"]] = feature
    features.extend(related.values())
    metadata = {key: value for key, value in result.items() if key not in {"features", "uncertain_candidates", "context_features", "tile_candidates"}}
    metadata.update({"include_uncertain": request.include_uncertain, "export_scope": "all_results_in_selected_execution_mode",
        "page": None, "page_size": None, "truncated": False, "uncertain_truncated": False,
        "exported_geojson_features": len(features), "related_features": len(related)})
    version = re.sub(r"[^a-zA-Z0-9_-]", "_", str(result.get("dataset_version") or "dataset"))[:80]
    return JSONResponse({"type": "FeatureCollection", "features": features, "metadata": metadata},
        media_type="application/geo+json", headers={"Content-Disposition": f'attachment; filename="terralens-{version}.geojson"'})

@app.get("/api/export")
def export_download(
    plan: str = Query(min_length=2, max_length=8192),
    mode: Literal["catalog_scan", "explore", "ground"] = "catalog_scan",
    dataset_id: str | None = Query(default=None, max_length=40, pattern=r"^[a-z0-9_-]+$"),
    grounding_run_id: str | None = Query(default=None, pattern=r"^[a-f0-9]{32}$"),
    dataset_version: str | None = Query(default=None, max_length=160),
    include_uncertain: bool = False,
):
    """Native browser attachment download using the already executed, bounded plan."""
    try:
        validated = QueryPlan.model_validate_json(plan)
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=f"Invalid export plan: {exc.errors()[0]['msg']}") from exc
    return export(ExportRequest(plan=validated, mode=mode, dataset_version=dataset_version,
        dataset_id=dataset_id, grounding_run_id=grounding_run_id, include_uncertain=include_uncertain))

@app.get("/api/map")
def map_data(bbox: str | None = None, tile_limit: int = Query(default=24, ge=1, le=64), vector_limit: int = Query(default=2000, ge=1, le=5000), dataset_id: str | None = None):
    bundle = _bundle(dataset_id)
    bounds_geometry = None
    if bbox is not None:
        try:
            bounds = [float(value.strip()) for value in bbox.split(",")]
            if len(bounds) != 4 or not all(math.isfinite(value) for value in bounds):
                raise ValueError
            if not (-180 <= bounds[0] < bounds[2] <= 180 and -90 <= bounds[1] < bounds[3] <= 90):
                raise ValueError
            bounds_geometry = box(*bounds)
        except (ValueError, TypeError):
            raise HTTPException(status_code=422, detail="bbox must be finite west,south,east,north WGS84 bounds with positive width and height.")
    available_tiles = [tile for tile in bundle.tiles if bounds_geometry is None or shape(tile["geometry"]).intersects(bounds_geometry)]
    tiles = []
    for tile in available_tiles[:tile_limit]:
        geometry = tile["geometry"]
        coordinates = tile.get("image_coordinates_wgs84")
        if coordinates is None and geometry.get("type") == "Polygon" and len(geometry["coordinates"][0]) == 5:
            coordinates = geometry["coordinates"][0][:4]
        tiles.append({"tile_id": tile["tile_id"], "asset_url": "/api/assets/" + quote(tile["tile_id"], safe="") + ("?dataset_id=" + quote(dataset_id, safe="") if dataset_id else ""),
            "geometry": geometry, "coordinates": coordinates, "acquisition_date": tile.get("acquisition_date"),
            "acquisition_date_status": tile.get("acquisition_date_status", "known" if tile.get("acquisition_date") else "unknown"),
            "source_name": tile.get("source_name"), "source_sha256": tile.get("source_sha256"),
            "width": tile.get("width"), "height": tile.get("height")})
    layers, counts, remaining = {}, {}, vector_limit
    for layer_id, frame in bundle.layers.items():
        positions = range(len(frame)) if bounds_geometry is None else sorted(frame.sindex.query(bounds_geometry, predicate="intersects").tolist())
        total = len(positions)
        features = []
        for position in list(positions)[:remaining]:
            row = frame.iloc[int(position)]
            properties = json_value(row.drop(labels=[frame.geometry.name], errors="ignore").to_dict())
            properties.update({"layer": layer_id, "role": "base_vector"})
            features.append({"type": "Feature", "geometry": mapping(row.geometry), "properties": properties})
        remaining -= len(features)
        layers[layer_id] = {"type": "FeatureCollection", "features": features}
        counts[layer_id] = {"available": total, "returned": len(features), "truncated": len(features) < total}
    return json_value({"tiles": tiles, "vector_layers": layers, "coverage": bundle.manifest.get("coverage", {}),
        "area_name": bundle.manifest.get("area_name"), "dataset_version": bundle.manifest.get("dataset_version"),
        "attribution": bundle.manifest.get("attribution", {}), "provenance": bundle.provenance(),
        "counts": {"tiles": {"available": len(available_tiles), "returned": len(tiles), "truncated": len(tiles) < len(available_tiles)}, "vector_layers": counts},
        "bounds_requested": bounds_geometry.bounds if bounds_geometry is not None else None})

@app.get("/api/assets/{tile_id}", include_in_schema=False)
def asset(tile_id: str, dataset_id: str | None = None):
    path = _bundle(dataset_id).asset_path(tile_id)
    if path is None:
        raise HTTPException(status_code=404, detail="Evidence image not found.")
    return FileResponse(path, media_type=mimetypes.guess_type(path.name)[0] or "application/octet-stream")
