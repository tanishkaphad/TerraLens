"""Submitted place lookup and bounded, source-backed OpenStreetMap analysis.

No imagery or model inference is performed here. A mapped feature and an empty
OSM response are not evidence of visual detection or real-world completeness.
"""
from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import threading
import time
from typing import Any
from urllib.parse import urlparse

from pyproj import CRS, Geod, Transformer
import requests
from shapely.geometry import LineString, Point, Polygon, mapping
from shapely.ops import polygonize_full, transform, unary_union
from shapely.strtree import STRtree


class WorldDataError(ValueError):
    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.status_code = status_code


PLACE_ENDPOINT = "https://photon.komoot.io/api/"
OVERPASS_ENDPOINT = "https://overpass-api.de/api/interpreter"
MAX_RADIUS_M = 2000.0
MAX_DISTANCE_M = 2000.0
MAX_RESULTS = 100
MAX_ELEMENTS = 20000
MAX_RESPONSE_BYTES = 20 * 1024 * 1024
_CACHE: OrderedDict[str, dict] = OrderedDict()
_REQUEST_LOCK = threading.Lock()
_LAST_REQUEST: dict[str, float] = {}
_CACHE_TTL = {"places": 24 * 3600, "vectors": 3600}
_REQUEST_INTERVAL = {"places": 1.0, "vectors": 2.0}
_GEOD = Geod(ellps="WGS84")
_OSM_TYPES = {"N": "node", "W": "way", "R": "relation", "node": "node", "way": "way", "relation": "relation"}
_ALIASES = {
    "building": "buildings", "buildings": "buildings", "building footprints": "buildings",
    "road": "roads", "roads": "roads", "college": "colleges", "colleges": "colleges",
    "universities": "colleges", "university": "colleges", "river": "rivers", "rivers": "rivers",
    "forest": "forests", "forests": "forests", "woods": "forests", "woodland": "forests",
}
_ROAD_VALUES = {
    "motorway", "trunk", "primary", "secondary", "tertiary", "unclassified", "residential",
    "service", "living_street", "pedestrian", "track", "road", "motorway_link", "trunk_link",
    "primary_link", "secondary_link", "tertiary_link",
}


def status() -> dict:
    """Configuration status only; avoids a request and never returns credentials."""
    return {"place_provider": "Photon / OpenStreetMap", "place_lookup_configured": True,
        "place_availability": "not_checked", "place_lookup_mode": "submitted_queries_only",
        "vector_provider": "OpenStreetMap / Overpass", "vector_availability": "not_checked",
        "maximum_search_radius_m": MAX_RADIUS_M, "maximum_distance_m": MAX_DISTANCE_M,
        "maximum_returned_features": MAX_RESULTS,
        "coverage_note": "Provider-dependent mapped coverage. Public services do not guarantee availability."}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _text(value: Any, maximum: int = 300) -> str:
    return str(value or "").strip()[:maximum]


def _endpoint(variable: str, default: str) -> str:
    value = os.getenv(variable, default).strip()
    parts = urlparse(value)
    if parts.scheme not in {"http", "https"} or not parts.hostname or parts.username or parts.password:
        raise WorldDataError("The geographic data provider is not configured correctly.", 503)
    return value


def _cache_path(key: str) -> Path:
    pipeline = Path(__file__).resolve().parents[1]
    root = Path(os.getenv("WORLD_CACHE_DIR", "data/world-cache")).expanduser()
    if not root.is_absolute():
        root = pipeline / root
    return root / (key + ".json")


def _cached_json(provider: str, url: str, *, params: dict | None = None, data: dict | None = None) -> tuple[dict, str, bool]:
    """One serialized request at a time, bounded responses, and timestamped snapshots."""
    key = hashlib.sha256(json.dumps([provider, url, params, data], sort_keys=True).encode()).hexdigest()
    with _REQUEST_LOCK:
        cached = _CACHE.get(key)
        if cached is None:
            try:
                path = _cache_path(key)
                if path.stat().st_size <= MAX_RESPONSE_BYTES:
                    cached = json.loads(path.read_text())
            except (OSError, ValueError, TypeError):
                cached = None
        if isinstance(cached, dict) and isinstance(cached.get("payload"), dict):
            saved = cached.get("saved_at", 0)
            if isinstance(saved, (float, int)) and 0 <= time.time() - saved < _CACHE_TTL[provider]:
                _CACHE[key] = cached
                _CACHE.move_to_end(key)
                return deepcopy(cached["payload"]), str(cached.get("retrieved_at", "unknown")), True
        delay = _REQUEST_INTERVAL[provider] - (time.monotonic() - _LAST_REQUEST.get(provider, 0))
        if delay > 0:
            time.sleep(delay)
        _LAST_REQUEST[provider] = time.monotonic()
        try:
            response = requests.request("POST" if data else "GET", url, params=params, data=data,
                headers={"User-Agent": "GeoVerify-X/2.0 (bounded submitted geographic search)", "Accept": "application/json"},
                timeout=(10, 55), allow_redirects=False, stream=True)
            with response:
                if response.status_code == 429:
                    raise WorldDataError("The geographic data provider is busy. Please try again later.", 503)
                if response.status_code != 200:
                    raise WorldDataError("The geographic data provider could not complete this request.", 503)
                chunks, size = [], 0
                for chunk in response.iter_content(chunk_size=65536):
                    size += len(chunk)
                    if size > MAX_RESPONSE_BYTES:
                        raise WorldDataError("This area contains too much mapped data. Choose a smaller search radius.", 422)
                    chunks.append(chunk)
                payload = json.loads(b"".join(chunks))
        except WorldDataError:
            raise
        except (requests.RequestException, ValueError, TypeError):
            # Provider exception strings can contain URLs and API keys.
            raise WorldDataError("The geographic data provider is unavailable. Please try again later.", 503) from None
        if not isinstance(payload, dict):
            raise WorldDataError("The geographic data provider returned an invalid response.", 503)
        expected = "features" if provider == "places" else "elements"
        if not isinstance(payload.get(expected), list) or provider == "vectors" and payload.get("remark"):
            raise WorldDataError("The geographic data query did not finish. Please try again later.", 503)
        retrieved_at = _now()
        entry = {"payload": payload, "retrieved_at": retrieved_at, "saved_at": time.time()}
        _CACHE[key] = entry
        _CACHE.move_to_end(key)
        while len(_CACHE) > 128:
            _CACHE.popitem(last=False)
        try:
            path = _cache_path(key)
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(".tmp")
            temporary.write_text(json.dumps(entry, ensure_ascii=False))
            temporary.replace(path)
        except OSError:
            pass  # In-memory caching still works on read-only installations.
        return deepcopy(payload), retrieved_at, False


def _identity(raw_type: Any, raw_id: Any) -> tuple[str | None, int | None]:
    kind = _OSM_TYPES.get(str(raw_type))
    # bool is an integer in Python, but is not an OSM identifier.
    identifier = str(raw_id)
    if kind is None or isinstance(raw_id, bool) or not re.fullmatch(r"[1-9][0-9]{0,19}", identifier):
        return None, None
    return kind, int(identifier)


def _source_url(kind: str | None, identifier: int | None) -> str:
    return f"https://www.openstreetmap.org/{kind}/{identifier}" if kind and identifier else "https://www.openstreetmap.org/"


def _point_coordinates(geometry: Any) -> tuple[float, float]:
    if not isinstance(geometry, dict) or geometry.get("type") != "Point":
        raise WorldDataError("Choose a place with valid geographic coordinates.")
    values = geometry.get("coordinates")
    if not isinstance(values, (list, tuple)) or len(values) != 2 or any(isinstance(v, bool) for v in values):
        raise WorldDataError("Choose a place with valid geographic coordinates.")
    try:
        lon, lat = (float(v) for v in values)
    except (ValueError, TypeError):
        raise WorldDataError("Choose a place with valid geographic coordinates.") from None
    if not math.isfinite(lon) or not math.isfinite(lat) or not -180 <= lon <= 180 or not -90 < lat < 90:
        raise WorldDataError("Choose a place with valid geographic coordinates.")
    return lon, lat


def _place_bbox(feature: dict, properties: dict) -> list[float] | None:
    bbox = feature.get("bbox")
    # Photon extent is [west, north, east, south], not a GeoJSON bbox.
    extent = properties.get("extent")
    if bbox is None and isinstance(extent, (list, tuple)) and len(extent) == 4:
        bbox = [extent[0], extent[3], extent[2], extent[1]]
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4 or any(isinstance(v, bool) for v in bbox):
        return None
    try:
        result = [float(v) for v in bbox]
    except (ValueError, TypeError):
        return None
    west, south, east, north = result
    if all(math.isfinite(v) for v in result) and -180 <= west <= east <= 180 and -90 <= south <= north <= 90:
        return result
    return None


def lookup_places(text: str, limit: int = 6) -> list[dict]:
    """Look up submitted names; Photon handles its own spelling tolerance."""
    if not isinstance(text, str) or not 2 <= len(text.strip()) <= 300 or any(ord(c) < 32 for c in text):
        raise WorldDataError("Enter a place name between 2 and 300 characters.")
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 10:
        raise WorldDataError("Place lookup limit must be between 1 and 10.")
    query = re.sub(r"^(?:find|locate|show(?: me)?|in)\s+", "", text.strip(), flags=re.I)
    query = re.sub(r"\s+in\s+", ", ", query, flags=re.I)
    query = re.sub(r"\s+", " ", query).strip()
    if len(query) < 2:
        raise WorldDataError("Enter a named place to search.")
    payload, retrieved_at, cached = _cached_json("places", _endpoint("PHOTON_URL", PLACE_ENDPOINT),
        params={"q": query, "limit": limit, "lang": "en"})
    if not isinstance(payload.get("features"), list):
        raise WorldDataError("The place provider returned an invalid response.", 503)
    results, seen = [], set()
    for raw in payload["features"][:100]:
        if not isinstance(raw, dict) or not isinstance(raw.get("properties"), dict):
            continue
        props = raw["properties"]
        try:
            lon, lat = _point_coordinates(raw.get("geometry"))
        except WorldDataError:
            continue
        kind, identifier = _identity(props.get("osm_type"), props.get("osm_id"))
        name = _text(props.get("name") or props.get("street"))
        if not name:
            continue
        dedupe = (kind, identifier) if identifier else (name.casefold(), lon, lat)
        if dedupe in seen:
            continue
        seen.add(dedupe)
        parts = [name]
        for field in ("street", "city", "district", "state", "country"):
            value = _text(props.get(field))
            if value and value.casefold() not in {part.casefold() for part in parts}:
                parts.append(value)
        bbox = _place_bbox(raw, props)
        properties = {"name": name, "display_name": ", ".join(parts), "source": "OpenStreetMap via Photon",
            "source_url": _source_url(kind, identifier), "attribution": "OpenStreetMap contributors, ODbL 1.0",
            "evidence_kind": "mapped_place", "evidence_status": "mapped_place", "geometry_role": "geocoded_place_point",
            "osm_type": kind, "osm_id": identifier, "osm_key": _text(props.get("osm_key")),
            "osm_value": _text(props.get("osm_value")), "place_type": _text(props.get("type") or props.get("osm_value")),
            "retrieved_at": retrieved_at, "source_date": None, "source_date_status": "not_reported",
            "cache_hit": cached, "bbox": bbox, "conditions": [{"condition": "place_index", "status": "pass",
                "note": "Named place returned by the geographic index; this is not visual detection."}]}
        for field in ("city", "district", "state", "country", "postcode", "street", "housenumber"):
            if props.get(field):
                properties[field] = _text(props[field])
        result = {"type": "Feature", "id": f"osm:{kind}:{identifier}" if identifier else "place:" + hashlib.sha256(json.dumps(dedupe).encode()).hexdigest()[:16],
            "geometry": {"type": "Point", "coordinates": [lon, lat]}, "properties": properties}
        if bbox:
            result["bbox"] = bbox
        results.append(result)
        if len(results) >= limit:
            break
    return results


def _layer(value: str) -> str:
    if not isinstance(value, str) or value.strip().casefold() not in _ALIASES:
        raise WorldDataError("Mapped search supports buildings, roads, colleges, rivers and forests. Additional attributes need an explicit supported query.")
    return _ALIASES[value.strip().casefold()]


def _selectors(layer: str, radius: float, lat: float, lon: float) -> list[str]:
    area = f"(around:{radius:.2f},{lat:.7f},{lon:.7f})"
    if layer == "buildings":
        return [f'way["building"]["building"!="no"]{area};', f'relation["type"="multipolygon"]["building"]["building"!="no"]{area};']
    if layer == "roads":
        values = "|".join(sorted(_ROAD_VALUES))
        return [f'way["highway"~"^({values})$"]{area};']
    if layer == "colleges":
        return [f'nwr["amenity"~"^(college|university)$"]{area};']
    if layer == "rivers":
        return [f'way["waterway"="river"]{area};', f'way["water"="river"]{area};',
            f'relation["type"="multipolygon"]["water"="river"]{area};']
    return [f'way["landuse"="forest"]{area};', f'way["natural"="wood"]{area};',
        f'relation["type"="multipolygon"]["landuse"="forest"]{area};', f'relation["type"="multipolygon"]["natural"="wood"]{area};']


def _matches_layer(tags: dict, layer: str) -> bool:
    if layer == "buildings":
        return bool(tags.get("building")) and tags["building"] != "no"
    if layer == "roads":
        return tags.get("highway") in _ROAD_VALUES
    if layer == "colleges":
        return tags.get("amenity") in {"college", "university"}
    if layer == "rivers":
        return tags.get("waterway") == "river" or tags.get("water") == "river"
    return tags.get("landuse") == "forest" or tags.get("natural") == "wood"


def _coordinates(raw: Any) -> list[tuple[float, float]]:
    if not isinstance(raw, list):
        return []
    points = []
    for item in raw:
        if not isinstance(item, dict):
            return []
        try:
            points.append(_point_coordinates({"type": "Point", "coordinates": [item["lon"], item["lat"]]}))
        except (KeyError, WorldDataError):
            return []
    return points


def _geometry(element: dict):
    """Keep actual geometry, never substitute an Overpass center for a footprint."""
    kind = element.get("type")
    tags = element.get("tags", {})
    area = bool(tags.get("building")) or tags.get("landuse") == "forest" or tags.get("natural") == "wood" or tags.get("water") == "river" or tags.get("amenity") in {"college", "university"}
    if kind == "node":
        try:
            return Point(_point_coordinates({"type": "Point", "coordinates": [element["lon"], element["lat"]]}))
        except (KeyError, WorldDataError):
            return None
    if kind == "way":
        coordinates = _coordinates(element.get("geometry"))
        if area:
            if len(coordinates) < 4 or coordinates[0] != coordinates[-1] or tags.get("area") == "no":
                return None
            geometry = Polygon(coordinates)
        else:
            if len(coordinates) < 2:
                return None
            geometry = LineString(coordinates)
    elif kind == "relation" and tags.get("type") == "multipolygon":
        outer, inner = [], []
        members = element.get("members")
        if not isinstance(members, list):
            return None
        for member in members:
            if not isinstance(member, dict):
                return None
            if member.get("type") != "way":
                continue
            coordinates = _coordinates(member.get("geometry"))
            if len(coordinates) < 2:
                return None
            if member.get("role") == "inner":
                inner.append(LineString(coordinates))
            elif member.get("role") in {"outer", ""}:
                outer.append(LineString(coordinates))
        if not outer:
            return None
        # Polygonizing joined segments preserves building boundaries and holes.
        closed, cuts, dangles, invalid = polygonize_full(unary_union(outer))
        if not cuts.is_empty or not dangles.is_empty or not invalid.is_empty:
            return None
        polygons = list(closed.geoms)
        if not polygons:
            return None
        geometry = unary_union(polygons)
        if inner:
            closed, cuts, dangles, invalid = polygonize_full(unary_union(inner))
            if not cuts.is_empty or not dangles.is_empty or not invalid.is_empty:
                return None
            holes = list(closed.geoms)
            if not holes:
                return None
            holes_geometry = unary_union(holes)
            if not geometry.covers(holes_geometry):
                return None
            geometry = geometry.difference(holes_geometry)
    else:
        return None
    return geometry if not geometry.is_empty and geometry.is_valid else None


def _metric_crs(lon: float, lat: float) -> CRS:
    if -80 <= lat <= 84:
        zone = min(60, max(1, int((lon + 180) // 6) + 1))
        return CRS.from_epsg((32600 if lat >= 0 else 32700) + zone)
    return CRS.from_proj4(f"+proj=aeqd +lat_0={lat} +lon_0={lon} +datum=WGS84 +units=m +no_defs")


def _radius_bbox(lon: float, lat: float, radius: float) -> list[float]:
    points = [_GEOD.fwd(lon, lat, angle, radius)[:2] for angle in (0, 90, 180, 270)]
    # GeoJSON permits west > east for a bbox crossing the antimeridian.
    return [points[3][0], points[2][1], points[1][0], points[0][1]]


def _feature(element: dict, geometry, layer: str, retrieved_at: str, source_date: str | None) -> dict:
    kind, identifier = _identity(element["type"], element["id"])
    tags = {str(k)[:100]: _text(v, 500) for k, v in element.get("tags", {}).items()}
    identity = f"osm:{kind}:{identifier}"
    bbox = [float(v) for v in geometry.bounds]
    return {"type": "Feature", "id": identity, "bbox": bbox, "geometry": mapping(geometry), "properties": {
        "id": identity, "name": tags.get("name") or layer.rstrip("s").capitalize(), "subject_layer": layer,
        "layer": layer, "role": "match", "evidence_kind": "mapped_vector", "geometry_role": "source_vector_feature",
        "evidence_status": "supported_by_available_evidence", "source": "OpenStreetMap via Overpass",
        "source_url": _source_url(kind, identifier), "attribution": "OpenStreetMap contributors, ODbL 1.0",
        "osm_type": kind, "osm_id": identifier, "retrieved_at": retrieved_at, "source_date": source_date,
        "vector_extraction_date": retrieved_at, "source_date_status": "known" if source_date else "not_reported",
        "source_attributes": tags, "subject_attributes": tags, "bbox": bbox, "conditions": [],
        "observations": [], "evidence_geometries": [], "imagery_coverage_tile_ids": [], "imagery_coverage_dates": []}}


def vector_search(subject: str, reference: str | None, distance_m: float | None, location: dict,
        radius_m: float = 2000, *, predicate: str = "within_distance") -> dict:
    """Analyze mapped geometry in a local circle, with a reference halo for proximity."""
    started = time.perf_counter()
    subject_layer = _layer(subject)
    reference_layer = _layer(reference) if reference else None
    if predicate not in {"within_distance", "intersects"}:
        raise WorldDataError("Mapped spatial search supports proximity and intersection.")
    try:
        if isinstance(radius_m, bool) or isinstance(distance_m, bool):
            raise ValueError
        radius = float(radius_m)
        distance = float(0 if distance_m is None else distance_m)
    except (TypeError, ValueError):
        raise WorldDataError("Use a valid search radius and distance in metres.") from None
    if not math.isfinite(radius) or not 100 <= radius <= MAX_RADIUS_M:
        raise WorldDataError("Choose a local search radius between 100 and 2000 metres.")
    if not math.isfinite(distance) or not 0 <= distance <= MAX_DISTANCE_M:
        raise WorldDataError("Mapped proximity distance must be between 0 and 2000 metres.")
    if reference_layer and predicate == "within_distance" and distance_m is None:
        raise WorldDataError("Choose an explicit proximity distance in metres.")
    if not isinstance(location, dict):
        raise WorldDataError("Choose a named place before searching mapped features.")
    lon, lat = _point_coordinates(location.get("geometry", location))
    if predicate == "intersects":
        distance = 0.0
    statements = _selectors(subject_layer, radius, lat, lon)
    if reference_layer:
        statements.extend(_selectors(reference_layer, radius + distance, lat, lon))
    statements = list(dict.fromkeys(statements))
    query = "[out:json][timeout:45][maxsize:67108864];(" + "".join(statements) + f");out geom {MAX_ELEMENTS};"
    payload, retrieved_at, cached = _cached_json("vectors", _endpoint("OVERPASS_URL", OVERPASS_ENDPOINT), data={"data": query})
    if payload.get("remark") or not isinstance(payload.get("elements"), list):
        # Overpass remarks can indicate a timeout/partial response; never treat it as complete.
        raise WorldDataError("The mapped-data query did not finish. Please choose a smaller search radius or try again later.", 503)
    elements = payload["elements"]
    osm_metadata = payload.get("osm3s") or {}
    source_date = _text(osm_metadata.get("timestamp_osm_base"), 60) or None if isinstance(osm_metadata, dict) else None
    crs = _metric_crs(lon, lat)
    project = Transformer.from_crs("EPSG:4326", crs, always_xy=True).transform
    center_metric = transform(project, Point(lon, lat))
    aoi = center_metric.buffer(radius, quad_segs=64)
    parsed, invalid, seen = [], 0, set()
    for element in elements[:MAX_ELEMENTS]:
        if not isinstance(element, dict) or not isinstance(element.get("tags"), dict):
            continue
        kind, identifier = _identity(element.get("type"), element.get("id"))
        if identifier is None or (kind, identifier) in seen:
            continue
        element = {**element, "tags": {str(k)[:100]: _text(v, 500) for k, v in element["tags"].items()}}
        seen.add((kind, identifier))
        geometry = _geometry(element)
        if geometry is None:
            invalid += 1
            continue
        metric = transform(project, geometry)
        if metric.is_empty or not metric.is_valid or not all(math.isfinite(v) for v in metric.bounds):
            invalid += 1
            continue
        parsed.append((element, geometry, metric))
    # Building/forest point tags are usable map labels, never footprints/forest patches.
    def usable(record, layer):
        element, geometry, _metric = record
        if not _matches_layer(element["tags"], layer):
            return False
        return layer not in {"buildings", "forests"} or geometry.geom_type in {"Polygon", "MultiPolygon"}

    subjects = [record for record in parsed if usable(record, subject_layer) and record[2].intersects(aoi)]
    references = [record for record in parsed if usable(record, reference_layer)] if reference_layer else []
    reference_geometries = [record[2] for record in references]
    index = STRtree(reference_geometries) if references else None
    matches, uncertain, excluded = [], [], 0
    for element, geometry, metric in subjects:
        feature = _feature(element, geometry, subject_layer, retrieved_at, source_date)
        conditions = [{"condition": "vector_source", "status": "pass", "subject_layer": subject_layer,
            "feature_id": feature["id"], "source": "OpenStreetMap", "source_date": source_date,
            "note": "The supplied mapped geometry is used; no visual detection is claimed."}]
        state = "pass"
        if reference_layer:
            candidate_positions = index.query(metric.buffer(distance) if distance else metric) if index is not None else []
            witnesses = []
            for position in candidate_positions:
                record = references[int(position)]
                ref_element, ref_geometry, ref_metric = record
                # A feature cannot witness its own proximity to another feature of the same family.
                if element["type"] == ref_element["type"] and element["id"] == ref_element["id"]:
                    continue
                separation = float(metric.distance(ref_metric))
                passes = metric.intersects(ref_metric) if predicate == "intersects" else separation <= distance
                if passes:
                    kind, identifier = _identity(ref_element["type"], ref_element["id"])
                    witnesses.append({"id": f"osm:{kind}:{identifier}", "distance_m": round(separation, 3), "status": "pass",
                        "source_url": _source_url(kind, identifier), "geometry_role": "source_vector_feature"})
            witnesses.sort(key=lambda item: (item["distance_m"], item["id"]))
            witness_count = len(witnesses)
            # One actual witness proves this existential relation. Keep its geometry
            # in the exported context, including when the subject page is capped.
            witnesses = witnesses[:1]
            # A nonmatch cannot establish absence because OSM has unreported omissions.
            state = "pass" if witnesses else "unknown"
            conditions.append({"condition": predicate, "status": state, "reference_layer": reference_layer,
                "radius_m": distance, "distance_m": witnesses[0]["distance_m"] if witnesses else None,
                "distance_basis": "source_geometry_to_source_geometry", "reference_feature_ids": [w["id"] for w in witnesses],
                "reference_candidates": witnesses, "witnesses": witnesses,
                "witness_count": witness_count, "witnesses_truncated": witness_count > len(witnesses),
                "witness_selection": "nearest_qualifying_mapped_geometry",
                "reference_coverage_status": "mapped_snapshot_completeness_unknown",
                "note": "Calculated from complete available feature geometry in projected metres." if witnesses else
                    "No qualifying mapped reference was available in this snapshot; real-world absence is not established."})
        feature["properties"]["conditions"] = conditions
        if state == "pass":
            matches.append(feature)
        else:
            feature["properties"].update({"role": "uncertain_candidate", "evidence_status": "unknown",
                "reason": "The requested relationship has no qualifying witness in the available mapped snapshot."})
            uncertain.append(feature)
    matches.sort(key=lambda f: f["id"])
    uncertain.sort(key=lambda f: f["id"])
    bbox = _radius_bbox(lon, lat, radius)
    label = _text(location.get("properties", {}).get("display_name") or location.get("properties", {}).get("name")) or "Selected place"
    kind, identifier = _identity(location.get("properties", {}).get("osm_type"), location.get("properties", {}).get("osm_id"))
    safe_location = {"type": "Feature", "geometry": {"type": "Point", "coordinates": [lon, lat]}, "properties": {
        "name": _text(location.get("properties", {}).get("name")) or label, "display_name": label,
        "evidence_kind": "mapped_place", "osm_type": kind, "osm_id": identifier,
        "source_url": _source_url(kind, identifier)}}
    uncertain_limit = MAX_RESULTS - min(MAX_RESULTS, len(matches))
    returned_matches, returned_uncertain = matches[:MAX_RESULTS], uncertain[:uncertain_limit]
    witness_ids = {identifier for feature in returned_matches + returned_uncertain
        for condition in feature["properties"]["conditions"] for identifier in condition.get("reference_feature_ids", [])}
    context = []
    for element, geometry, _metric in references:
        kind, identifier = _identity(element["type"], element["id"])
        if f"osm:{kind}:{identifier}" in witness_ids:
            feature = _feature(element, geometry, reference_layer, retrieved_at, source_date)
            feature["properties"]["role"] = "reference"
            context.append(feature)
    truncated = len(matches) > MAX_RESULTS
    source_truncated = len(elements) >= MAX_ELEMENTS
    scope = {"center": [lon, lat], "radius_m": radius, "reference_radius_m": radius + distance if reference_layer else None,
        "bbox": bbox, "area_name": label, "type": "bounded_local_sample", "citywide": False,
        "subject_selection": "source_geometry_intersects_local_aoi",
        "reference_coverage_complete": False, "coverage_status": "mapped_snapshot_completeness_unknown",
        "source_truncated": source_truncated, "invalid_geometries_skipped": invalid,
        "note": f"Local mapped sample within {radius:g} metres of {label}; this does not cover the entire city or establish completeness."}
    result = {"route": "vector", "evidence_kind": "mapped_vector", "result_kind": "vector_features",
        "mode": "mapped", "dataset_version": "osm-live:" + retrieved_at, "analysis_crs": crs.to_string(),
        "plan": {"query_type": "vectors", "subject_layer": subject_layer, "reference_layer": reference_layer,
            "predicate": predicate if reference_layer else None, "distance_m": distance if reference_layer else None, "attributes": []},
        "features": returned_matches, "uncertain_candidates": returned_uncertain, "tile_candidates": [],
        "context_features": context, "total_matches": len(matches), "total_uncertain": len(uncertain),
        "objects_evaluated": len(subjects), "objects_excluded": excluded, "truncated": truncated,
        "uncertain_truncated": len(uncertain) > uncertain_limit, "source_truncated": source_truncated,
        "context_truncated": False, "page": 0, "page_size": MAX_RESULTS,
        "tiles_considered": 0, "tiles_processed": 0, "tiles_available": 0, "text_similarity_enabled": False,
        "text_encoder_status": "not_required_for_vector_query", "coverage": scope, "scope": scope, "bbox": bbox,
        "location": safe_location, "retrieved_at": retrieved_at, "source_date": source_date, "cache_hit": cached,
        "provenance": {"source": "OpenStreetMap via Overpass", "source_url": "https://www.openstreetmap.org/",
            "attribution": "OpenStreetMap contributors, ODbL 1.0", "retrieved_at": retrieved_at,
            "source_date": source_date, "snapshot_sha256": hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest(),
            "provider": "Overpass", "query_scope": scope, "source_feature_count": len(elements),
            "subject_feature_count": len(subjects), "reference_feature_count": len(references)},
        "export_scope": {"kind": "bounded_returned_subset", "maximum_subject_features": MAX_RESULTS,
            "returned_matches": len(returned_matches), "returned_uncertain": len(returned_uncertain),
            "all_returned_relationship_witnesses_included": True},
        "claim_scope": "Mapped geometries in a bounded local OSM snapshot. No visual inference or citywide or real-world completeness claim.",
        "timings_ms": {"total": round((time.perf_counter() - started) * 1000, 2)}}
    return result
