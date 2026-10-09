"""FastAPI service and same-origin TerraLens search interface."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .bundle import Bundle, BundleError
from .engine import execute
from .planner import PlanError, QueryPlan, plan_query

app = FastAPI(title="TerraLens", version="0.1.0")
STATIC = Path(__file__).parent / "static" / "index.html"


class SearchRequest(BaseModel):
    query: str = Field(min_length=3, max_length=500)
    previous_plan: dict | None = None
    mode: str = "catalog_scan"
    page: int = Field(default=0, ge=0, le=1000)
    page_size: int = Field(default=50, ge=1, le=100)


@lru_cache(maxsize=1)
def get_bundle() -> Bundle:
    root = Path(os.getenv("BUNDLE_DIR", "data/demo")).expanduser()
    return Bundle(root)


@app.get("/", include_in_schema=False)
def home():
    return FileResponse(STATIC)


@app.get("/health")
def health():
    return {"status": "alive", "service": "terralens"}


@app.get("/ready")
def ready():
    try:
        bundle = get_bundle()
    except (BundleError, OSError, ValueError) as exc:
        raise HTTPException(status_code=503, detail=f"Evidence bundle unavailable: {exc}") from exc
    return {
        "status": "ready", "dataset_version": bundle.manifest.get("dataset_version"),
        "tiles": len(bundle.tiles), "detections": len(bundle.detections),
        "layers": sorted(bundle.layers),
    }


@app.get("/api/capabilities")
def capabilities():
    try:
        bundle = get_bundle()
    except (BundleError, OSError, ValueError) as exc:
        raise HTTPException(status_code=503, detail=f"Evidence bundle unavailable: {exc}") from exc
    return {
        "dataset_version": bundle.manifest.get("dataset_version"),
        "area_name": bundle.manifest.get("area_name"), "coverage": bundle.manifest.get("coverage", {}),
        "supported_concepts": bundle.manifest.get("supported_concepts", {}),
        "vector_layers": bundle.manifest.get("vector_layers", {}),
        "queryable_attributes": bundle.manifest.get("queryable_attributes", []),
        "text_similarity_enabled": bool(os.getenv("ENCODER_CHECKPOINT")),
        "limits": {"maximum_distance_m": 5000, "page_size": 100},
    }


@app.post("/api/query")
def query(request: SearchRequest):
    try:
        bundle = get_bundle()
        plan = plan_query(request.query, bundle.manifest, request.previous_plan)
        result = execute(bundle, plan, request.mode, request.page_size, request.page)
    except PlanError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except (BundleError, OSError, ValueError) as exc:
        raise HTTPException(status_code=503 if isinstance(exc, (BundleError, OSError)) else 422, detail=str(exc)) from exc
    return result


@app.get("/api/assets/{tile_id}", include_in_schema=False)
def asset(tile_id: str):
    try:
        bundle = get_bundle()
    except (BundleError, OSError, ValueError) as exc:
        raise HTTPException(status_code=503, detail="Evidence bundle unavailable.") from exc
    path = bundle.asset_path(tile_id)
    if path is None:
        raise HTTPException(status_code=404, detail="Evidence image not found.")
    return FileResponse(path, media_type="image/jpeg")
