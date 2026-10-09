"""FastAPI search, bounded imagery map, evidence assets and complete GeoJSON export."""
from __future__ import annotations
import mimetypes
import json
import math
import os
import re
import threading
from collections import deque
from functools import lru_cache
from time import monotonic
from pathlib import Path
from typing import Literal
from urllib.parse import quote
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.gzip import GZipMiddleware
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from shapely.geometry import box, mapping, shape
from .bundle import Bundle, BundleError, json_value
from . import assistant
from .configuration import load_configuration
from .engine import execute
from .discovery import discovery_plan
from .grounding import GroundingError, GroundingService, grounding_status
from .jobs import JobError, JobManager
from .planner import PlanError, QueryPlan, _rule_plan, plan_query, validate_plan
from .workflow import WorkflowError

load_configuration()
app = FastAPI(title="GeoVerify-X", version="0.4.0")
app.add_middleware(GZipMiddleware, minimum_size=1000)
STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

_CHAT_LIMIT_LOCK = threading.Lock()
_CHAT_REQUESTS: deque[float] = deque()
_CHAT_REQUESTS_DAY: deque[float] = deque()


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    message: str = Field(min_length=2, max_length=2000)
    conversation_id: str | None = Field(default=None, pattern=r"^[a-f0-9]{32}$")
    dataset_id: str | None = Field(default=None, max_length=40, pattern=r"^[a-z0-9_-]+$")


@lru_cache(maxsize=1)
def get_job_manager() -> JobManager:
    configured = os.getenv("JOBS_DB", "data/conversations.sqlite3")
    path = Path(configured).expanduser()
    if not path.is_absolute():
        path = Path(__file__).resolve().parents[1] / path
    return JobManager(path, workers=2, max_pending=8)


def _assistant_catalog(dataset_id: str | None) -> dict:
    datasets, _default = _dataset_registry()
    if dataset_id is not None and dataset_id not in datasets:
        raise WorkflowError("The selected dataset is unavailable.")
    catalog = {}
    for identifier, detail in datasets.items():
        bundle = _bundle(identifier)
        retrieval = bundle.retrieval_status()
        grounding = grounding_status(bundle)
        attributes = {}
        for layer_id, frame in bundle.layers.items():
            layer_fields = {}
            for field in frame.columns:
                if field == frame.geometry.name:
                    continue
                try:
                    values = frame[field].dropna().astype(str).unique()[:8]
                    if len(values):
                        layer_fields[field] = [value[:120] for value in values]
                except (TypeError, ValueError):
                    continue
            attributes[layer_id] = layer_fields
        catalog[identifier] = {"manifest": bundle.manifest, "live_grounding": grounding["available"],
            "retrieval": retrieval["enabled"], "attribute_examples": attributes,
            "bundle": bundle, "name": detail["name"]}
    return catalog


def _workflow_search(step):
    request = SearchRequest(plan=step.plan, mode=step.mode, dataset_id=step.dataset_id,
        grounding_tile_limit=step.tile_limit or 2, page_size=100)
    if step.mode == "ground":
        first = _run(request)
        if not first.get("grounding_run_id"):
            raise WorkflowError("The object-grounding run did not return a reusable evidence identifier.")
        return _run(ExportRequest(plan=step.plan, mode=step.mode, dataset_id=step.dataset_id,
            grounding_run_id=first["grounding_run_id"], dataset_version=first["dataset_version"],
            include_uncertain=True), all_results=True)
    return _run(request, all_results=True)


def _assistant_run(message: str, dataset_id: str | None, history: list, progress) -> dict:
    catalog = _assistant_catalog(dataset_id)
    result = assistant.respond(message, dataset_id, history, progress, catalog=catalog, search=_workflow_search)
    result["summary"] = result.get("message", "")[:4000]
    return result

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
    reference = data / "spacenet-round1"
    if not (reference / "manifest.json").is_file():
        reference = data / "spacenet-grounded"
    if not (reference / "manifest.json").is_file():
        reference = data / "spacenet-urban"
    if not (reference / "manifest.json").is_file():
        reference = data / "spacenet-sample"
    paths = {"phoenix": data / "phoenix-demo", "spacenet": reference, "demo": data / "demo"}
    configured = os.getenv("BUNDLE_DIR")
    default = next((identifier for identifier in ("phoenix", "spacenet", "demo")
        if (paths[identifier] / "manifest.json").is_file()), "demo")
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
            "source": "SpaceNet 1 reference sample" if identifier == "spacenet" else "USDA NAIP imagery and OpenStreetMap context" if identifier in {"phoenix", "demo"} else "Configured local evidence bundle",
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
    if isinstance(value, dict):
        return {key: _dataset_assets(item, dataset_id) for key, item in value.items()}
    if isinstance(value, list):
        return [_dataset_assets(item, dataset_id) for item in value]
    if isinstance(value, str) and value.startswith("/") and "/owlv2-" in value:
        return "google/owlv2-base-patch16-ensemble"
    if dataset_id and isinstance(value, str) and value.startswith("/api/assets/") and "?" not in value:
        return value + "?dataset_id=" + quote(dataset_id, safe="")
    return value

def _run(request, *, all_results: bool = False):
    round1 = os.getenv("GEOVERIFY_ROUND1") == "1"
    if round1 and request.dataset_id != "spacenet":
        raise HTTPException(status_code=422, detail="Round 1 searches the prepared SpaceNet imagery only.")
    bundle = _bundle(request.dataset_id)
    if getattr(request, "dataset_version", None) is not None and request.dataset_version != bundle.manifest.get("dataset_version"):
        raise HTTPException(status_code=409, detail="The displayed dataset version differs from the current bundle; rerun the search before exporting.")
    try:
        if getattr(request, "plan", None) is not None:
            plan = validate_plan(request.plan, bundle.manifest, request.mode)
        elif request.query:
            previous = request.previous_plan.model_dump() if request.previous_plan else None
            plan = (_rule_plan(request.query, bundle.manifest, None, request.mode) if round1
                else plan_query(request.query, bundle.manifest, previous, request.mode))
        else:
            raise PlanError("Provide a validated plan or a natural-language query.")
        if round1:
            if plan.query_type != "vectors" and not bundle.retrieval_status()["enabled"]:
                raise HTTPException(status_code=503, detail="The local imagery search model is not ready. Please try again shortly.")
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


@app.get("/api/assistant/status")
def assistant_status():
    if os.getenv("GEOVERIFY_ROUND1") == "1":
        return {"configured": False, "provider": None, "model": None, "tools": [],
            "reason": "Conversation is paused for Round 1. The imagery search is available."}
    return assistant.status()


def _check_chat_capacity(now: float) -> None:
    """Check the process-local request guard while ``_CHAT_LIMIT_LOCK`` is held."""
    try:
        hourly = max(1, min(200, int(os.getenv("CHAT_REQUESTS_PER_HOUR", "20"))))
        daily = max(hourly, min(1000, int(os.getenv("CHAT_REQUESTS_PER_DAY", "100"))))
    except ValueError:
        hourly, daily = 20, 100
    while _CHAT_REQUESTS and _CHAT_REQUESTS[0] <= now - 3600:
        _CHAT_REQUESTS.popleft()
    while _CHAT_REQUESTS_DAY and _CHAT_REQUESTS_DAY[0] <= now - 86400:
        _CHAT_REQUESTS_DAY.popleft()
    if len(_CHAT_REQUESTS) >= hourly or len(_CHAT_REQUESTS_DAY) >= daily:
        raise HTTPException(status_code=429, detail="The conversation service has reached its request limit. Try again later.")


def _admit_chat(request: ChatRequest) -> dict:
    """Atomically check capacity, admit the durable job, then record usage."""
    with _CHAT_LIMIT_LOCK:
        now = monotonic()
        _check_chat_capacity(now)
        submitted = get_job_manager().submit(
            request.message.strip(), request.dataset_id, request.conversation_id, _assistant_run)
        _CHAT_REQUESTS.append(now)
        _CHAT_REQUESTS_DAY.append(now)
        return submitted


@app.post("/api/chat", status_code=202)
def chat(request: ChatRequest):
    if os.getenv("GEOVERIFY_ROUND1") == "1":
        raise HTTPException(status_code=503, detail="Conversation is paused for Round 1. Use imagery search.")
    status = assistant.status()
    if not status["configured"]:
        raise HTTPException(status_code=503, detail=status["reason"])
    if request.dataset_id is not None and request.dataset_id not in _dataset_registry()[0]:
        raise HTTPException(status_code=404, detail="Unknown or unprepared dataset.")
    try:
        return _admit_chat(request)
    except JobError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc


@app.get("/api/jobs/{job_id}")
def assistant_job(job_id: str, conversation_id: str = Query(pattern=r"^[a-f0-9]{32}$")):
    try:
        job = get_job_manager().get(job_id, conversation_id)
    except JobError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
    if job["status"] == "succeeded":
        result = job["result"]
        result["conversation_id"] = conversation_id
        has_geo = any(artifact.get("kind") == "map" for artifact in result.get("artifacts", []))
        if has_geo:
            result["export_url"] = f"/api/jobs/{job_id}/export?conversation_id={conversation_id}"
    return job


@app.get("/api/jobs/{job_id}/export")
def assistant_export(job_id: str, conversation_id: str = Query(pattern=r"^[a-f0-9]{32}$")):
    try:
        job = get_job_manager().get(job_id, conversation_id)
    except JobError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
    if job["status"] != "succeeded":
        raise HTTPException(status_code=409, detail="The evidence workflow has not completed.")
    result = job["result"]
    features = []

    def feature_id(feature: dict) -> str | None:
        identifier = feature.get("id")
        if identifier is not None:
            return str(identifier)
        properties = feature.get("properties", {})
        identifier, layer = properties.get("id"), properties.get("layer")
        return f"{layer}:{identifier}" if layer is not None and identifier is not None else None

    def exported_feature(feature: dict, step_id: str, status: str, *, parent_id: str | None = None) -> dict:
        properties = dict(feature.get("properties", {}))
        properties.update(workflow_step=step_id, workflow_status=status)
        if parent_id is not None:
            properties["workflow_parent_feature_id"] = parent_id
        exported = {"type": "Feature", "geometry": feature.get("geometry"), "properties": properties}
        if feature.get("id") is not None:
            exported["id"] = feature["id"]
        return exported

    def reference_index(context: list[dict]) -> dict[str, dict]:
        indexed = {}
        for feature in context:
            properties = feature.get("properties", {})
            identifiers = {str(value) for value in (
                feature.get("id"), properties.get("id"), properties.get("original_feature_id")) if value is not None}
            layer = properties.get("layer")
            for identifier in list(identifiers):
                indexed.setdefault(identifier, feature)
                if layer is not None:
                    indexed.setdefault(f"{layer}:{identifier}", feature)
        return indexed

    for artifact in result.get("artifacts", []):
        if artifact.get("kind") != "map":
            continue
        step_id = str(artifact.get("id") or "map")
        search = artifact.get("result", {})
        collections = (
            ("supported", search.get("features", [])),
            ("uncertain", search.get("uncertain_candidates", [])),
            ("verification_excluded", search.get("verification_excluded", [])),
            ("tile_candidate", search.get("tile_candidates", []) if search.get("result_kind") == "tile_retrieval" else []),
        )
        selected = [feature for _status, records in collections for feature in records]
        context = list(search.get("context_features", []))
        indexed = reference_index(context)
        related = {}
        for feature in selected:
            for condition in feature.get("properties", {}).get("conditions", []):
                layer = condition.get("reference_layer")
                identifiers = list(condition.get("reference_feature_ids", []))
                if condition.get("reference_feature_id") is not None:
                    identifiers.append(condition["reference_feature_id"])
                for identifier in identifiers:
                    identifier = str(identifier)
                    reference = indexed.get(f"{layer}:{identifier}") if layer is not None else None
                    reference = reference or indexed.get(identifier)
                    if reference is not None:
                        related[feature_id(reference) or identifier] = reference
        # Context features are recorded evidence, including references retained for
        # uncertain or verification-excluded records. Export all of them so a GIS
        # consumer can audit the saved result without a model or database rerun.
        for reference in context:
            related.setdefault(feature_id(reference) or f"context:{len(related)}", reference)
        for status, records in collections:
            for feature in records:
                parent_id = feature_id(feature)
                features.append(exported_feature(feature, step_id, status))
                for evidence in feature.get("properties", {}).get("evidence_geometries", []):
                    features.append(exported_feature(evidence, step_id, "evidence_geometry", parent_id=parent_id))
        for reference in related.values():
            features.append(exported_feature(reference, step_id, "reference"))
    metadata = {key: result.get(key) for key in ("workflow", "receipt", "validation", "tool_trace")}
    metadata.update({"conversation_id": conversation_id, "job_id": job_id,
        "export_scope": "saved completed workflow; supported, uncertain and excluded records with observation and reference geometries",
        "exported_geojson_features": len(features), "workflow_result": result})
    filename = re.sub(r"[^a-zA-Z0-9_-]", "_", (result.get("workflow") or {}).get("title", "geoverify-evidence"))[:80]
    return JSONResponse({"type": "FeatureCollection", "features": features, "metadata": json_value(metadata)},
        media_type="application/geo+json", headers={"Content-Disposition": f'attachment; filename="{filename or "geoverify-evidence"}.geojson"'})

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
    # Examples must execute through the same credential-free discovery route as
    # a typed prompt, preserving their advertised evidence and dataset limit.
    maximum_distance = min(5000, bundle.manifest.get("limits", {}).get("maximum_distance_m", 5000))
    example_distance = min(20, maximum_distance)
    distance_label = f"{example_distance:g}"
    for identifier, concept in bundle.manifest.get("supported_concepts", {}).items():
        if concept.get("grounding_available", True):
            aliases = concept.get("aliases", [])
            subject = aliases[0] if aliases else identifier.replace("_", " ")
            tasks.append({"query": "Find " + subject, "mode": "catalog_scan", "evidence_kind": "cached_model_detection"})
            if "roads" in bundle.layers:
                tasks.append({"query": f"Find {subject} within {distance_label} metres of roads",
                    "mode": "catalog_scan", "evidence_kind": "cached_model_detection"})
    for layer in bundle.layers:
        collision = any(layer.replace("_", " ").casefold() in {str(value).casefold() for value in
            [identifier.replace("_", " "), concept.get("prompt", ""), *concept.get("aliases", [])]}
            for identifier, concept in bundle.manifest.get("supported_concepts", {}).items())
        tasks.append({"query": "Find " + ("mapped " if collision else "") + layer,
            "mode": "catalog_scan", "evidence_kind": "mapped_feature"})
    if {"buildings", "roads"} <= set(bundle.layers):
        tasks.append({"query": f"Find mapped buildings within {distance_label} metres of roads", "mode": "catalog_scan", "evidence_kind": "mapped_feature"})
        buildings, roads = bundle.layers["buildings"], bundle.layers["roads"]
        if "building" in buildings and "highway" in roads and buildings["building"].eq("house").any() and roads["highway"].eq("residential").any():
            tasks.insert(0, {"query": f"Find mapped buildings with building=house within {distance_label} metres of roads with highway=residential",
                "mode": "catalog_scan", "evidence_kind": "mapped_feature"})
    if grounding["available"] and retrieval["enabled"]:
        tasks.append({"query": f"Find trees within {distance_label} metres of roads" if "roads" in bundle.layers else "Find trees",
            "mode": "ground", "evidence_kind": "live_model_detection"})
    if retrieval["enabled"]:
        tasks.append({"query": "Find satellite imagery of densely built neighbourhoods", "mode": "explore", "evidence_kind": "image_similarity"})
    return json_value({"dataset_version": bundle.manifest.get("dataset_version"), "area_name": bundle.manifest.get("area_name"),
        "coverage": bundle.manifest.get("coverage", {}), "supported_concepts": bundle.manifest.get("supported_concepts", {}),
        "vector_layers": bundle.manifest.get("vector_layers", {}), "queryable_attributes": bundle.manifest.get("queryable_attributes", []),
        "text_similarity_enabled": retrieval["enabled"], "retrieval": retrieval,
        "parser": "schema_constrained_language_model" if os.getenv("GEOVERIFY_ROUND1") != "1" and os.getenv("OPENAI_API_KEY") and os.getenv("OPENAI_MODEL") else "conservative_templates",
        "conversation": assistant_status(),
        "supported_predicates": bundle.manifest.get("supported_predicates", ["within_distance", "intersects"]),
        "limits": {"maximum_distance_m": maximum_distance,
            "page_size": 100, "explore_tile_limit": min(1000, bundle.manifest.get("limits", {}).get("explore_tile_limit", 100)),
            "map_tile_limit": 64, "map_vector_limit": 5000},
        "provenance": _dataset_assets(bundle.provenance(), dataset_id), "live_grounding": grounding["available"] and retrieval["enabled"],
        "grounding": grounding, "task_catalog": tasks, "dataset_id": dataset_id,
        "query_families": ["mapped_features", "cached_objects", "image_scenes", "new_visual_objects"],
        "maximum_spatial_clauses": 4})

@app.post("/api/query")
def query(request: SearchRequest):
    return _run(request)


@app.post("/api/search")
def discover(request: SearchRequest):
    if not request.query or request.plan or request.previous_plan or request.grounding_run_id:
        raise HTTPException(status_code=422, detail="Enter a complete natural-language search question.")
    bundle = _bundle(request.dataset_id)
    try:
        plan, mode = discovery_plan(request.query.strip(), bundle.manifest)
    except PlanError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    resolved = request.model_copy(update={"query": None, "plan": plan, "mode": mode})
    result = _run(resolved)
    result["execution_mode"] = mode
    result["interpretation"] = {"method": "catalog constraints and open-vocabulary visual description",
        "note": "Only registered evidence operations execute. Complex language that cannot be safely translated needs clarification."}
    return result

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
        media_type="application/geo+json", headers={"Content-Disposition": f'attachment; filename="geoverify-{version}.geojson"'})

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
        "attribution": bundle.manifest.get("attribution", {}), "provenance": _dataset_assets(bundle.provenance(), dataset_id),
        "counts": {"tiles": {"available": len(available_tiles), "returned": len(tiles), "truncated": len(tiles) < len(available_tiles)}, "vector_layers": counts},
        "bounds_requested": bounds_geometry.bounds if bounds_geometry is not None else None})

@app.get("/api/assets/{tile_id}", include_in_schema=False)
def asset(tile_id: str, dataset_id: str | None = None):
    path = _bundle(dataset_id).asset_path(tile_id)
    if path is None:
        raise HTTPException(status_code=404, detail="Evidence image not found.")
    return FileResponse(path, media_type=mimetypes.guess_type(path.name)[0] or "application/octet-stream")
