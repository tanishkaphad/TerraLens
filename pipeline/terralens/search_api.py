"""Persistent, catalog-driven search API independent of legacy preview routes."""
from __future__ import annotations

from functools import lru_cache
import os
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from . import assistant, world_data
from .search_plan import SearchPlan, SearchPlanError
from .search_service import SearchService, SearchServiceError
from .search_sources import SourceError, place_boundary
from .search_store import SearchStore, SearchStoreError


router = APIRouter(tags=["Durable search"])


@lru_cache(maxsize=1)
def get_search_service() -> SearchService:
    path = Path(os.getenv("SEARCH_DB", "data/search.sqlite3")).expanduser()
    if not path.is_absolute():
        path = Path(__file__).resolve().parents[1] / path
    return SearchService(SearchStore(path))


class PlanRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str | None = Field(default=None, min_length=1)
    plan: dict[str, Any] | None = None
    scope: dict[str, Any] | None = None
    parent_run_id: str | None = None

    @model_validator(mode="after")
    def one_intent(self):
        if (self.query is None) == (self.plan is None):
            raise ValueError("Provide either a natural-language query or an explicit structured plan.")
        return self


class RunRequest(PlanRequest):
    start: bool = True


class PauseRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    reason: str | None = None


def _call(operation):
    try:
        return operation()
    except SearchPlanError as exc:
        raise HTTPException(429 if exc.code == "parser_rate_limited" else 422,
            detail={"code": exc.code, "message": str(exc), "details": exc.details}) from exc
    except SearchServiceError as exc:
        raise HTTPException(exc.status_code, detail={"message": str(exc), **exc.details}) from exc
    except SearchStoreError as exc:
        raise HTTPException(exc.status_code, detail=str(exc)) from exc
    except world_data.WorldDataError as exc:
        raise HTTPException(exc.status_code, detail=str(exc)) from exc
    except (SourceError, ValidationError) as exc:
        raise HTTPException(422, detail=str(exc)) from exc


def _links(run: dict) -> dict:
    identifier = run["run_id"]
    aggregate = next((unit["coverage"]["aggregate"] for unit in reversed(run["coverage"]["units"])
                      if "aggregate" in unit["coverage"]), None)
    return {**run, "status_url": f"/api/search/runs/{identifier}",
        "results_url": f"/api/search/runs/{identifier}/results", "map_url": f"/api/search/runs/{identifier}/map",
        "export_url": f"/api/search/runs/{identifier}/export", "units_url": f"/api/search/runs/{identifier}/units",
        "aggregate": aggregate}


@router.get("/api/catalog")
def catalog():
    return _call(lambda: {**get_search_service().describe(), "interpreter": {
        **{key: value for key, value in assistant.status().items() if key in {"configured", "provider", "model", "reason"}},
        "enabled": os.getenv("SEARCH_PARSER_ENABLED", "0") == "1",
        "integration_owner": "teammate"}})


@router.get("/api/search/schema")
def schema():
    return SearchPlan.model_json_schema()


@router.post("/api/search/plan")
def plan(request: PlanRequest):
    def resolve():
        service = get_search_service()
        previous = service.store.get_run(request.parent_run_id)["plan"] if request.parent_run_id else None
        result = service.plan(request.plan if request.plan is not None else request.query,
                              previous_plan=previous, scope=request.scope)
        return {"plan": result.model_dump(), "status": result.status,
            "catalog_revision": service.catalog.revision, "execution_started": False}
    return _call(resolve)


@router.post("/api/search/runs", status_code=202)
@router.post("/api/search", status_code=202)
def create_run(request: RunRequest):
    return _call(lambda: _links(get_search_service().create(request.query, plan=request.plan,
        scope=request.scope, parent_run_id=request.parent_run_id, start=request.start)))


@router.get("/api/search/runs")
def runs(status: list[str] | None = Query(default=None)):
    return _call(lambda: {"runs": [_links(run) for run in get_search_service().store.list_runs(statuses=status)]})


@router.get("/api/search/runs/{run_id}")
def run_status(run_id: str):
    return _call(lambda: _links(get_search_service().store.get_run(run_id)))


@router.get("/api/search/runs/{run_id}/units")
def work_units(run_id: str):
    # Lease tokens belong to workers; callers only need progress and coverage.
    return _call(lambda: {"units": [{key: value for key, value in unit.items()
        if key not in {"lease_token", "payload"}} for unit in get_search_service().store.units(run_id)]})


@router.get("/api/search/runs/{run_id}/results")
def results(run_id: str, revision: int | None = Query(default=None, ge=0),
            group: list[str] | None = Query(default=None), cursor: str | None = None,
            page_size: int = Query(default=50, ge=1)):
    return _call(lambda: get_search_service().store.results(run_id, revision=revision, groups=group,
                                                          cursor=cursor, page_size=page_size))


@router.get("/api/search/runs/{run_id}/map")
def map_results(run_id: str, revision: int | None = Query(default=None, ge=0),
                group: list[str] | None = Query(default=None)):
    return _call(lambda: get_search_service().store.map_data(run_id, revision=revision, groups=group))


@router.get("/api/search/runs/{run_id}/export")
def export(run_id: str, revision: int | None = Query(default=None, ge=0),
           group: list[str] | None = Query(default=None)):
    collection = _call(lambda: get_search_service().store.export(run_id, revision=revision, groups=group))
    return JSONResponse(collection, media_type="application/geo+json", headers={
        "Content-Disposition": f'attachment; filename="geoai-search-{run_id}.geojson"'})


@router.post("/api/search/runs/{run_id}/pause")
def pause(run_id: str, request: PauseRequest | None = None):
    return _call(lambda: _links(get_search_service().store.pause(run_id, reason=request.reason if request else None)))


@router.post("/api/search/runs/{run_id}/resume", status_code=202)
def resume(run_id: str):
    return _call(lambda: _links(get_search_service().resume(run_id)))


@router.post("/api/search/runs/{run_id}/cancel")
def cancel(run_id: str):
    return _call(lambda: _links(get_search_service().store.cancel(run_id)))


@router.get("/api/catalog/places")
def places(query: str = Query(min_length=1)):
    return _call(lambda: {"candidates": world_data.lookup_places(query), "enumeration_complete": False,
        "note": "Provider-ranked identity suggestions; choose an identity and resolve its actual boundary for area search."})


@router.get("/api/catalog/places/{osm_type}/{osm_id}/boundary")
def boundary(osm_type: str, osm_id: int):
    return _call(lambda: place_boundary(osm_type, osm_id))
