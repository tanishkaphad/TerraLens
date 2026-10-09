"""FastAPI search, bounded imagery map, evidence assets and complete GeoJSON export."""
from __future__ import annotations
import mimetypes
import math
import os
import re
from functools import lru_cache
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
from .planner import PlanError, QueryPlan, plan_query, validate_plan

app = FastAPI(title="TerraLens", version="0.2.0")
STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

class SearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str | None = Field(default=None, min_length=3, max_length=500)
    plan: QueryPlan | None = None
    dataset_version: str | None = Field(default=None, max_length=160)
    previous_plan: QueryPlan | None = None
    mode: Literal["catalog_scan", "explore"] = "catalog_scan"
    page: int = Field(default=0, ge=0, le=1000)
    page_size: int = Field(default=50, ge=1, le=100)

class ExportRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str | None = Field(default=None, min_length=3, max_length=500)
    plan: QueryPlan | None = None
    previous_plan: QueryPlan | None = None
    mode: Literal["catalog_scan", "explore"] = "catalog_scan"
    include_uncertain: bool = False
    dataset_version: str | None = Field(default=None, max_length=160)
    # Accepted for parity with search; export deliberately evaluates all results.
    page: int = Field(default=0, ge=0, le=1000)
    page_size: int = Field(default=50, ge=1, le=100)

@lru_cache(maxsize=1)
def get_bundle() -> Bundle:
    return Bundle(Path(os.getenv("BUNDLE_DIR", "data/demo")).expanduser())

def _bundle() -> Bundle:
    try:
        return get_bundle()
    except (BundleError, OSError, ValueError) as exc:
        raise HTTPException(status_code=503, detail=f"Evidence bundle unavailable: {exc}") from exc

def _run(request, *, all_results: bool = False):
    bundle = _bundle()
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
        return execute(bundle, plan, request.mode, request.page_size, request.page, all_results=all_results)
    except (PlanError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

@app.get("/", include_in_schema=False)
def home():
    return FileResponse(STATIC_DIR / "index.html")

@app.get("/health")
def health():
    return {"status": "alive", "service": "terralens"}

@app.get("/ready")
def ready():
    bundle = _bundle()
    retrieval = bundle.retrieval_status()
    grounded = [key for key, details in bundle.manifest.get("supported_concepts", {}).items() if details.get("grounding_available", True)]
    has_vectors = any(not frame.empty for frame in bundle.layers.values())
    full = retrieval["enabled"] and bool(grounded) and has_vectors
    return {"status": "ready" if full else "degraded", "bundle_loaded": True,
        "dataset_version": bundle.manifest.get("dataset_version"), "tiles": len(bundle.tiles), "detections": len(bundle.detections),
        "layers": sorted(bundle.layers), "retrieval": retrieval, "grounded_concepts": grounded,
        "capabilities": {"live_text_image_retrieval": retrieval["enabled"], "cached_object_spatial_search": bool(grounded) and has_vectors, "live_object_grounding": False},
        "note": "Precomputed model grounding; live text ranking and spatial checks." if full else "Evidence bundle loaded; full multimodal readiness is incomplete. See capabilities."}

@app.get("/api/capabilities")
def capabilities():
    bundle = _bundle()
    retrieval = bundle.retrieval_status()
    return json_value({"dataset_version": bundle.manifest.get("dataset_version"), "area_name": bundle.manifest.get("area_name"),
        "coverage": bundle.manifest.get("coverage", {}), "supported_concepts": bundle.manifest.get("supported_concepts", {}),
        "vector_layers": bundle.manifest.get("vector_layers", {}), "queryable_attributes": bundle.manifest.get("queryable_attributes", []),
        "text_similarity_enabled": retrieval["enabled"], "retrieval": retrieval,
        "parser": "schema_constrained_language_model" if os.getenv("OPENAI_API_KEY") and os.getenv("OPENAI_MODEL") else "conservative_templates",
        "supported_predicates": bundle.manifest.get("supported_predicates", ["within_distance", "intersects"]),
        "limits": {"maximum_distance_m": min(5000, bundle.manifest.get("limits", {}).get("maximum_distance_m", 5000)),
            "page_size": 100, "explore_tile_limit": min(1000, bundle.manifest.get("limits", {}).get("explore_tile_limit", 100)),
            "map_tile_limit": 64, "map_vector_limit": 5000},
        "provenance": bundle.provenance(), "live_grounding": False})

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
    mode: Literal["catalog_scan", "explore"] = "catalog_scan",
    dataset_version: str | None = Query(default=None, max_length=160),
    include_uncertain: bool = False,
):
    """Native browser attachment download using the already executed, bounded plan."""
    try:
        validated = QueryPlan.model_validate_json(plan)
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=f"Invalid export plan: {exc.errors()[0]['msg']}") from exc
    return export(ExportRequest(plan=validated, mode=mode, dataset_version=dataset_version, include_uncertain=include_uncertain))

@app.get("/api/map")
def map_data(bbox: str | None = None, tile_limit: int = Query(default=24, ge=1, le=64), vector_limit: int = Query(default=2000, ge=1, le=5000)):
    bundle = _bundle()
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
        tiles.append({"tile_id": tile["tile_id"], "asset_url": "/api/assets/" + quote(tile["tile_id"], safe=""),
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
def asset(tile_id: str):
    path = _bundle().asset_path(tile_id)
    if path is None:
        raise HTTPException(status_code=404, detail="Evidence image not found.")
    return FileResponse(path, media_type=mimetypes.guess_type(path.name)[0] or "application/octet-stream")
