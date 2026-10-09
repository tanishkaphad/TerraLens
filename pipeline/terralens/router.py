"""Conservative routing between worldwide places, mapped vectors and local imagery.

The selected imagery dataset is a hint, never a substitute for an explicit place.
Unresolved distances, modifiers and locations remain visible rather than becoming
unrelated searches against the default bundle.
"""
from __future__ import annotations

from copy import deepcopy
import math
import re
from typing import Callable

from . import world_data
from .discovery import discovery_plan
from .planner import PlanError, QueryPlan, plan_query

_START = re.compile(r"^\s*(?:(?:please\s+)?(?:find|locate|search\s+for|show\s+me|show|look\s+for))\s+", re.I)
_UNIT = r"kilometres?|kilometers?|km|metres?|meters?|m"
_NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)"
_WITHIN = re.compile(rf"\bwithin\s+({_NUMBER})\s*({_UNIT})\s+(?:of|from)\s+", re.I)
_RELATION = re.compile(r"\b(?:within|near|nearby|close\s+to|next\s+to|beside|adjacent|intersects?|intersecting|crossing)\b", re.I)
_LOCATION = re.compile(r"\s+(?:in|at)\s+(.+?)\s*$", re.I)
_NEAR = re.compile(r"\s+(?:near|close\s+to|next\s+to|beside|adjacent\s+to)\s+(.+?)\s*$", re.I)
_INTERSECT = re.compile(r"\s+(?:intersects?|intersecting|crossing)\s+", re.I)
_VISUAL = re.compile(r"\b(?:solar|panels?|roofs?|rooftop|pools?|trees?|vegetation|imagery|satellite|aerial|potholes?|unpaved|cleared|patches?|construction|damaged|collapsed|commercial|industrial|residential|warehouses?|parking|cars?|vehicles?|dense|densely|illegal|unauthori[sz]ed|historical|recent|newly|between|before|after|since|during|built|constructed|flood|healthy|abandoned|energy|crime)\b", re.I)
_CLAUSE = re.compile(r"\b(?:and|or|without|with|except|not|where|whose|having|between|before|after|since|during)\b|[;=]", re.I)
_UNSUPPORTED_CLAIMS = re.compile(r"\b(?:illegal|illegally|unauthori[sz]ed|encroachment|cleared|deforestation|newly|recently|historical|constructed|built|damaged|collapsed)\b|\b(?:19|20)\d{2}\b", re.I)
_GENERIC = {
    "building": "buildings", "buildings": "buildings", "building footprints": "buildings",
    "road": "roads", "roads": "roads", "street": "roads", "streets": "roads",
    "college": "colleges", "colleges": "colleges", "university": "colleges", "universities": "colleges",
    "river": "rivers", "rivers": "rivers", "waterway": "rivers", "waterways": "rivers",
    "forest": "forests", "forests": "forests", "forest area": "forests", "forest areas": "forests",
}


def _generic(text: str) -> str | None:
    clean = re.sub(r"^(?:(?:all|mapped|the|a|an)\s+)+", "", text.strip(), flags=re.I)
    return _GENERIC.get(clean.casefold())


def _clean(query: str) -> str:
    return _START.sub("", query.strip()).strip(" .!?")


def _base(query: str, route: str, *, status="completed", message="", dataset_id=None) -> dict:
    return {"type": "FeatureCollection", "status": status, "route": route,
        "result_kind": "place_lookup" if route == "places" else "coverage_notice",
        "features": [], "uncertain_candidates": [], "context_features": [], "tile_candidates": [],
        "total_matches": 0, "total_uncertain": 0, "message": message, "claim_scope": "",
        "interpretation": {"original_query": query, "method": "conservative geographic intent router"},
        "dataset_id": dataset_id, "export_url": None}


def _notice(query: str, route: str, message: str, *, status="needs_clarification", **interpretation) -> dict:
    result = _base(query, route, status=status, message=message)
    result["claim_scope"] = "No discovery was executed; the request needs resolved geography or suitable evidence."
    result["interpretation"].update(interpretation)
    return result


def _places(query: str, text: str, *, for_location=False) -> dict:
    try:
        places = world_data.lookup_places(text, limit=6)
    except world_data.WorldDataError as exc:
        return _notice(query, "places", str(exc), status="unavailable", location_text=text,
            provider_status_code=exc.status_code)
    if for_location and len(places) > 1:
        normal = lambda value: re.sub(r"[^a-z0-9]+", " ", str(value).casefold()).strip()
        wanted = normal(text)
        exact_localities = [feature for feature in places
            if normal(feature.get("properties", {}).get("name", "")) == wanted
            and (feature.get("properties", {}).get("osm_value") in {"city", "town", "village", "municipality"}
                or feature.get("properties", {}).get("place_type") in {"city", "town", "village", "municipality"})]
        # A unique exact locality is a semantic match, not provider first-result
        # guessing. Two distinct cities with the same name remain ambiguous.
        if len(exact_localities) == 1:
            places = exact_localities
    count = len(places)
    result = _base(query, "places", status="needs_clarification" if count > 1 else "completed",
        message=(f"Found {count} mapped places. Choose the intended location before a geographic analysis."
            if count > 1 else "Found one mapped place." if count else "No matching mapped place was returned by the place index."))
    result.update(features=places, total_matches=count, result_kind="place_lookup", execution_mode="place_lookup")
    result["claim_scope"] = "Mapped place records from the worldwide place index; names and coordinates are not visual detections or a completeness guarantee."
    result["interpretation"].update(search_text=text, location_resolution=for_location,
        ambiguous=count > 1)
    if for_location and not count:
        result["status"] = "needs_clarification"
        result["message"] = "The requested location could not be resolved. Include a more specific place name and city."
    return result


def _bundle_matches_location(detail: dict, location: str) -> bool:
    # Only a complete supplied place name may be removed from the query. Token
    # overlap (for example 'Pune' with an arbitrary site name) is not coverage.
    name = str(detail.get("name", "")).casefold()
    wanted = re.sub(r"\s+", " ", location.casefold().strip())
    alternatives = [name, *[piece.strip() for piece in re.split(r"[/,]", name)]]
    alternatives += [re.sub(r"\s+(?:study area|demo|sample|imagery|evidence bundle)$", "", item) for item in alternatives]
    return bool(wanted) and wanted in alternatives


def _spatial_parse(text: str) -> tuple[str | None, str | None, float | None, str | None, str | None]:
    """Parse only exact mapped subject + one metric relation; return an issue."""
    matches = list(_WITHIN.finditer(text))
    if matches:
        if len(matches) != 1:
            return None, None, None, None, "This worldwide vector search supports one spatial relationship. Simplify the request without dropping any condition."
        match = matches[0]
        subject, reference = _generic(text[:match.start()]), _generic(text[match.end():])
        distance = float(match.group(1)) * (1000 if match.group(2).lower().startswith("k") else 1)
        if not math.isfinite(distance) or not 0 < distance <= 5000:
            return subject, reference, distance, "within_distance", "Choose a finite distance greater than zero and at most 5000 metres."
        return subject, reference, distance, "within_distance", None
    match = _INTERSECT.search(text)
    if match:
        return _generic(text[:match.start()]), _generic(text[match.end():]), None, "intersects", None
    if _RELATION.search(text):
        return _generic(text), None, None, None, "Specify an explicit distance in metres or kilometres, or an exact intersection relationship. 'Nearby' has no implicit radius."
    return _generic(text), None, None, None, None


def _local(query: str, search_text: str, identifier: str, bundle_loader: Callable,
        local_search: Callable, previous_plan=None, *, location_text=None) -> dict:
    bundle = bundle_loader(identifier)
    try:
        if previous_plan:
            previous = previous_plan.model_dump() if isinstance(previous_plan, QueryPlan) else previous_plan
            mode = ("explore" if previous.get("query_type") == "tiles" else "ground"
                if previous.get("query_type", "objects") == "objects" and previous.get("concept_id") not in bundle.manifest.get("supported_concepts", {})
                else "catalog_scan")
            plan = plan_query(search_text, bundle.manifest, previous, mode)
            if plan.query_type == "tiles":
                mode = "explore"
        else:
            plan, mode = discovery_plan("Find " + search_text, bundle.manifest)
        result = local_search(plan, mode, identifier)
    except PlanError as exc:
        return _notice(query, "visual", str(exc), location_text=location_text,
            search_text=search_text, selected_dataset_id=identifier)
    result = dict(result)
    raw_kind = result.get("result_kind")
    result.update(type="FeatureCollection", status="completed", route="vectors" if plan.query_type == "vectors" else "visual",
        result_kind="vector_features" if plan.query_type == "vectors" else "tile_retrieval" if raw_kind == "tile_retrieval" else "object_detections",
        dataset_id=identifier, execution_mode=mode, export_url=None)
    result.setdefault("features", [])
    result.setdefault("uncertain_candidates", [])
    result.setdefault("context_features", [])
    result.setdefault("tile_candidates", [])
    name = bundle.manifest.get("area_name", identifier)
    count = result.get("total_tile_candidates", len(result["tile_candidates"])) if raw_kind == "tile_retrieval" else result.get("total_matches", len(result["features"]))
    result["message"] = (f"Found {count} image candidate tiles in {name}. Inspect imagery to assess the requested appearance."
        if raw_kind == "tile_retrieval" else f"Found {count} supported records in {name}, with {result.get('total_uncertain', 0)} uncertain candidates.")
    result["message"] += " Search covers the prepared imagery area only."
    result["claim_scope"] = f"Prepared local coverage: {name}. " + result.get("claim_scope", "")
    result["interpretation"] = {"original_query": query, "search_text": search_text,
        "location_text": location_text, "selected_dataset_id": identifier,
        "method": "registered local coverage and evidence-constrained visual/spatial plan",
        "scope_note": "A city name selects its prepared study area; the analysis does not cover the entire city."}
    return result


def discover(query: str, *, dataset_id: str | None, previous_plan, registry: dict,
        default_dataset: str, bundle_loader: Callable, local_search: Callable) -> dict:
    """Return a consistent evidence response; no arbitrary worldwide AI claims."""
    clean = _clean(query)
    if not clean:
        return _notice(query, "places", "Enter a place, geographic object or spatial relationship.")
    search_text, location_text = clean, None
    location_match = _LOCATION.search(clean)
    if location_match:
        candidate = location_match.group(1).strip()
        if not _generic(candidate) and (_CLAUSE.search(candidate) or _RELATION.search(candidate)):
            return _notice(query, "visual" if _VISUAL.search(clean) else "vectors",
                "The location and additional conditions need clarification. Put the city at the end and keep every spatial condition explicit, for example 'Find buildings within 300 metres of rivers in Pune'.",
                location_text=candidate, unresolved_conditions=[candidate])
        # A generic vector layer is a spatial condition, not a named locality.
        if not _generic(candidate) and not _CLAUSE.search(candidate):
            search_text, location_text = clean[:location_match.start()].strip(), candidate
    near = _NEAR.search(search_text)
    if near and not _generic(near.group(1)):
        named = near.group(1).strip()
        location = f"{named}, {location_text}" if location_text else named
        resolved = _places(query, location, for_location=True)
        resolved["route"] = "visual" if _VISUAL.search(search_text) else "vectors"
        resolved["result_kind"] = "coverage_notice"
        resolved["context_features"] = resolved.pop("features")
        resolved["features"] = []
        resolved["total_matches"] = 0
        resolved["interpretation"].update(location_text=location, distance_m=None,
            unsupported_conditions=["undefined nearby radius"])
        if resolved["status"] != "unavailable":
            resolved["status"] = "needs_clarification"
            resolved["message"] = ("Several mapped locations match. Choose the intended place and give a radius in metres or kilometres."
                if resolved["interpretation"].get("ambiguous") else "Define how far 'near' means with a radius in metres or kilometres before searching.")
            if _VISUAL.search(search_text):
                resolved["message"] += " Visual discovery also needs suitable prepared imagery for that area; worldwide basemap imagery is not detector evidence."
        return resolved

    subject, reference, distance, predicate, issue = _spatial_parse(search_text)
    route = "vectors" if subject else "visual" if previous_plan or _VISUAL.search(search_text) or _RELATION.search(search_text) or _CLAUSE.search(search_text) or _UNSUPPORTED_CLAIMS.search(search_text) else "places"
    # A named institution stays a worldwide place lookup regardless of selected
    # imagery. Generic entities need an area rather than a misleading global list.
    if route == "places":
        return _places(query, clean)
    if subject == "colleges" and not location_text:
        return _notice(query, "places", "Give a city or area for the college search, for example 'Find colleges in Pune'.", subject=subject)
    if issue:
        return _notice(query, route, issue, location_text=location_text, distance_m=distance,
            subject=subject, reference=reference, predicate=predicate)
    matched = [identifier for identifier, detail in registry.items()
        if location_text and _bundle_matches_location(detail, location_text)]
    if matched:
        chosen = dataset_id if dataset_id in matched else matched[0]
        return _local(query, search_text, chosen, bundle_loader, local_search, previous_plan, location_text=location_text)
    if not location_text:
        chosen = dataset_id or default_dataset
        if chosen in registry:
            return _local(query, search_text, chosen, bundle_loader, local_search, previous_plan)
        return _notice(query, route, "No prepared local imagery area is selected. Include a city for mapped-place discovery, or prepare a suitable imagery bundle.", status="unavailable")

    unsupported = _UNSUPPORTED_CLAIMS.findall(search_text)
    if unsupported:
        return _notice(query, route,
            "The conditions " + ", ".join(repr(item) for item in unsupported) + " need dated imagery or compatible source attributes; the worldwide mapped-vector executor cannot establish them. No conditions were dropped.",
            location_text=location_text, search_text=search_text, distance_m=distance,
            unsupported_conditions=unsupported)
    if subject and distance is not None and distance > world_data.MAX_DISTANCE_M:
        return _notice(query, "vectors", f"Worldwide mapped-vector proximity is bounded to {world_data.MAX_DISTANCE_M:g} metres. Choose a smaller distance.",
            location_text=location_text, subject=subject, reference=reference, distance_m=distance)

    resolved = _places(query, location_text, for_location=True)
    location_features = resolved["features"]
    interpretation = {"location_text": location_text, "subject": subject, "reference": reference,
        "distance_m": distance, "predicate": predicate, "search_text": search_text}
    if resolved["status"] != "completed" or len(location_features) != 1:
        result = _notice(query, route, resolved["message"], status=resolved["status"], **interpretation)
        result["context_features"] = location_features
        return result
    if subject and (not predicate or reference):
        try:
            result = world_data.vector_search(subject, reference, distance, location_features[0],
                radius_m=2000, predicate=predicate or "within_distance")
        except world_data.WorldDataError as exc:
            return _notice(query, "vectors", str(exc),
                status="needs_clarification" if exc.status_code in {400, 422} else "unavailable", **interpretation,
                provider_status_code=exc.status_code)
        result = dict(result)
        result.update(type="FeatureCollection", status="completed", route="vectors", result_kind="vector_features",
            dataset_id=None, export_url=None, execution_mode="mapped_vectors")
        for name in ("features", "uncertain_candidates", "context_features", "tile_candidates"):
            result.setdefault(name, [])
        result["context_features"] = [location_features[0], *result["context_features"]]
        result.setdefault("total_matches", len(result["features"]))
        result.setdefault("total_uncertain", len(result["uncertain_candidates"]))
        result["interpretation"] = {"original_query": query, "method": "mapped-vector spatial calculation",
            **interpretation, "search_radius_m": 2000,
            "scope_note": "Bounded 2 km sample around the resolved locality coordinate, not a citywide search."}
        result["message"] = f"Found {result['total_matches']} mapped {subject} in a 2 km sample around {location_text}."
        result["claim_scope"] = "Mapped vector evidence in a bounded 2 km area; no visual inference, citywide coverage or completeness guarantee. " + result.get("claim_scope", "")
        return result

    # Even a point inside a prepared bundle must not lose a named-place
    # proximity constraint. Only a declared area-name suffix routes locally.
    if _CLAUSE.search(search_text):
        result = _notice(query, route, "This request includes a named reference or additional conditions that the bounded worldwide executor cannot preserve. Specify supported mapped layers and one explicit metric relationship.", **interpretation)
    else:
        areas = ", ".join(str(detail.get("name", identifier)) for identifier, detail in registry.items()) or "none"
        result = _notice(query, "visual", f"No prepared imagery suitable for visual detection covers {location_text}. Available local imagery areas: {areas}. The basemap is for context and is not analyzed for this request.", status="unavailable", **interpretation)
        if predicate and not reference:
            result["message"] += " The named reference also needs resolution before its spatial relationship can be evaluated."
            result["interpretation"]["unresolved_conditions"] = ["named spatial reference"]
    result["context_features"] = location_features
    return result


def status_info(registry: dict, default_dataset: str) -> dict:
    return {"version": "2.0", "mode": "unified_discovery", "default_dataset_id": default_dataset,
        "providers": world_data.status(), "local_areas": [
            {key: deepcopy(value) for key, value in detail.items() if key != "path"}
            for detail in registry.values()],
        "scope": "Worldwide mapped-place lookup and bounded mapped-vector search; visual AI runs only against registered local imagery.",
        "limitations": ["Nearby needs an explicit radius.", "No worldwide rooftop-object imagery or detection claim.",
            "Mapped-vector search samples a disclosed 2 km area around a resolved locality, not a whole city."]}
