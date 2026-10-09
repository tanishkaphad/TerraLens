"""Configured, schema-described vector providers for the durable search pipeline.

Provider selectors are operator-owned data. User plans select catalog layers;
they never contain executable Overpass statements or arbitrary network URLs.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
import os
import threading
from pathlib import Path

from pyproj import CRS, Transformer
from shapely.geometry import LineString, Polygon, box, mapping, shape
from shapely import transform

from . import world_data


class SourceError(ValueError):
    pass


def metric_crs(geometry) -> CRS:
    west, south, east, north = geometry.bounds
    if east - west > 6 or north - south > 6 or south < -80 or north > 84:
        raise SourceError("This extent needs a regional or geodesic provider partition method; local metric partitioning is unavailable.")
    center = geometry.representative_point()
    return CRS.from_proj4(f"+proj=aeqd +lat_0={center.y} +lon_0={center.x} +datum=WGS84 +units=m +no_defs")


def partition_geometry(geometry: dict, cell_size_m: float) -> list[dict]:
    """Cover the exact area with metric cells, including irregular boundaries."""
    area = shape(geometry)
    if area.is_empty or not area.is_valid or area.geom_type not in {"Polygon", "MultiPolygon"}:
        raise SourceError("Provider extraction requires a valid polygon search area.")
    if not math.isfinite(cell_size_m) or cell_size_m <= 0:
        raise SourceError("Provider partition size must be positive metres.")
    crs = metric_crs(area)
    project = Transformer.from_crs("EPSG:4326", crs, always_xy=True).transform
    projected = transform(area, project, interleaved=False)
    metric_west, metric_south, metric_east, metric_north = projected.bounds
    columns = max(1, math.ceil((metric_east - metric_west) / cell_size_m))
    rows = max(1, math.ceil((metric_north - metric_south) / cell_size_m))
    west, south, east, north = area.bounds
    width, height = (east - west) / columns, (north - south) / rows
    result = []
    for row in range(rows):
        for column in range(columns):
            # Geographic clipping preserves the exact declared polygon. Back-
            # transforming straight projected edges can introduce narrow gaps.
            cell = box(west + column * width, south + row * height,
                       east if column == columns - 1 else west + (column + 1) * width,
                       north if row == rows - 1 else south + (row + 1) * height)
            clipped = area.intersection(cell)
            if clipped.is_empty or clipped.area <= 0:
                continue
            result.append({"id": f"{row}-{column}", "geometry": mapping(clipped),
                           "analysis_crs": crs.to_string(), "nominal_partition_size_m": cell_size_m})
    return result


def buffered_geometry(geometry: dict, margin_m: float) -> dict:
    if margin_m == 0:
        return deepcopy(geometry)
    area = shape(geometry)
    crs = metric_crs(area)
    project = Transformer.from_crs("EPSG:4326", crs, always_xy=True).transform
    inverse = Transformer.from_crs(crs, "EPSG:4326", always_xy=True).transform
    return mapping(transform(transform(area, project, interleaved=False).buffer(margin_m), inverse, interleaved=False))


class ConfiguredOSMSource:
    """Optional catalog layers loaded from a server-side JSON configuration.

    Each layer has fields, geometry_type and selectors [{field, value?}].
    Multiple selectors mean AND; alternatives are explicitly separate layers.
    No category is built into this adapter.
    """
    def __init__(self, path: Path | None = None):
        configured = os.getenv("SEARCH_OSM_CATALOG")
        self.path = Path(path or configured).expanduser().resolve() if path or configured else None
        self._layers: dict = {}
        self._lock = threading.RLock()
        self.revision = "unconfigured"
        self.refresh()

    def refresh(self) -> None:
        with self._lock:
            self._refresh_unlocked()

    def _refresh_unlocked(self) -> None:
        if self.path is None:
            return
        try:
            content = self.path.read_bytes()
            raw = json.loads(content)
            layers = raw.get("layers", {})
            if not isinstance(layers, dict):
                raise ValueError()
            for identifier, layer in layers.items():
                if not isinstance(identifier, str) or not identifier or not isinstance(layer, dict):
                    raise ValueError()
                if layer.get("geometry_type") not in {"point", "line", "polygon", "mixed"}:
                    raise ValueError()
                selectors = layer.get("selectors")
                if not isinstance(selectors, list) or not selectors:
                    raise ValueError()
                if not isinstance(layer.get("fields", []), list):
                    raise ValueError()
                for selector in selectors:
                    if not isinstance(selector, dict) or set(selector) - {"field", "value"}:
                        raise ValueError()
                    if not isinstance(selector.get("field"), str) or not selector["field"]:
                        raise ValueError()
                    if "value" in selector and not isinstance(selector["value"], str):
                        raise ValueError()
                size = float(layer.get("partition_size_m", os.getenv("SEARCH_PARTITION_SIZE_M", "2000")))
                if not math.isfinite(size) or size <= 0:
                    raise ValueError()
            self._layers = deepcopy(layers)
            self.revision = hashlib.sha256(content).hexdigest()
        except (OSError, ValueError, TypeError, KeyError):
            raise SourceError("The configured OSM layer catalog is unreadable or invalid.") from None

    def describe(self) -> dict:
        with self._lock:
            self._refresh_unlocked()
            layers, revision = deepcopy(self._layers), self.revision
        return {f"osm:{key}": {**value, "dataset_id": "osm", "layer_id": key,
                "provider": "OpenStreetMap / Overpass", "catalog_version": revision,
                "coverage_status": "provider_dependent", "reference_coverage_complete": False,
                "source": "OpenStreetMap", "attribution": "OpenStreetMap contributors, ODbL 1.0"}
                for key, value in layers.items()}

    def extract(self, layer_id: str, geometry: dict, *, catalog_version: str) -> tuple[list[dict], dict]:
        with self._lock:
            self._refresh_unlocked()
            if catalog_version != self.revision:
                raise SourceError("Provider configuration changed during this run; create a new run to use it.")
            if layer_id not in self._layers:
                raise SourceError("The requested provider layer is not registered.")
            layer = deepcopy(self._layers[layer_id])
        area = shape(geometry)
        west, south, east, north = area.bounds
        selectors = "".join("[" + json.dumps(item["field"]) +
            ("=" + json.dumps(item["value"]) if "value" in item else "") + "]" for item in layer["selectors"])
        # The provider retrieves complete member geometry, not an artificial centre.
        statement = f"[out:json][timeout:45];nwr{selectors}({south:.9f},{west:.9f},{north:.9f},{east:.9f});out geom;"
        payload, retrieved_at, cached = world_data._cached_json("vectors",
            world_data._endpoint("OVERPASS_URL", world_data.OVERPASS_ENDPOINT), data={"data": statement})
        features, omitted = [], 0
        for element in payload["elements"]:
            if not isinstance(element, dict):
                omitted += 1
                continue
            tags = element.get("tags", {})
            if not isinstance(tags, dict) or not all(item["field"] in tags and
                    ("value" not in item or tags[item["field"]] == item["value"]) for item in layer["selectors"]):
                continue
            kind = element.get("type")
            expected = layer["geometry_type"]
            if kind == "way":
                coordinates = world_data._coordinates(element.get("geometry"))
                if expected == "polygon":
                    geom = Polygon(coordinates) if len(coordinates) >= 4 and coordinates[0] == coordinates[-1] else None
                else:
                    geom = LineString(coordinates) if len(coordinates) >= 2 else None
            else:
                adjusted = deepcopy(element)
                if kind == "relation" and tags.get("type") == "boundary":
                    adjusted["tags"]["type"] = "multipolygon"
                geom = world_data._geometry(adjusted)
            if geom is None or geom.is_empty or not geom.is_valid:
                omitted += 1
                continue
            if not geom.intersects(area):
                continue
            if expected == "point" and geom.geom_type != "Point" or expected == "line" and geom.geom_type not in {"LineString", "MultiLineString"}:
                omitted += 1
                continue
            if expected == "polygon" and geom.geom_type not in {"Polygon", "MultiPolygon"}:
                omitted += 1
                continue
            identifier = f"osm:{kind}:{element.get('id')}"
            features.append({"type": "Feature", "id": identifier, "geometry": mapping(geom), "properties": {
                **deepcopy(tags), "source_attributes": deepcopy(tags), "source_feature_id": identifier,
                "source": "OpenStreetMap / Overpass", "source_url": f"https://www.openstreetmap.org/{kind}/{element.get('id')}",
                "geometry_role": "source_vector_feature", "dataset_id": "osm", "layer_id": layer_id,
                "retrieved_at": retrieved_at, "snapshot_date": payload.get("osm3s", {}).get("timestamp_osm_base"),
                "reference_coverage_complete": False, "attribution": "OpenStreetMap contributors, ODbL 1.0"}})
        return features, {"geometry": deepcopy(geometry), "source_enumeration": "incomplete_geometry" if omitted else "response_complete",
            "data_coverage": "map_completeness_unknown", "records_received": len(features), "omitted_geometry": omitted,
            "snapshot_date": payload.get("osm3s", {}).get("timestamp_osm_base"), "retrieved_at": retrieved_at,
            "snapshot_consistency": "per_partition_timestamps", "cache_hit": cached}


def place_boundary(osm_type: str, osm_id: int) -> dict:
    if osm_type not in {"way", "relation"} or isinstance(osm_id, bool) or osm_id <= 0:
        raise SourceError("Choose an OSM way or relation with a polygon boundary.")
    payload, retrieved_at, _cached = world_data._cached_json("vectors",
        world_data._endpoint("OVERPASS_URL", world_data.OVERPASS_ENDPOINT),
        data={"data": f"[out:json][timeout:45];{osm_type}({osm_id});out geom;"})
    for element in payload["elements"]:
        if element.get("type") != osm_type or element.get("id") != osm_id:
            continue
        adjusted = deepcopy(element)
        tags = adjusted.setdefault("tags", {})
        if osm_type == "relation":
            tags["type"] = "multipolygon"
            geom = world_data._geometry(adjusted)
        else:
            coords = world_data._coordinates(element.get("geometry"))
            geom = Polygon(coords) if len(coords) >= 4 and coords[0] == coords[-1] else None
        if geom is None or geom.is_empty or not geom.is_valid or geom.geom_type not in {"Polygon", "MultiPolygon"}:
            break
        return {"type": "Feature", "id": f"osm:{osm_type}:{osm_id}", "geometry": mapping(geom),
            "properties": {"source": "OpenStreetMap", "source_attributes": element.get("tags", {}),
                "retrieved_at": retrieved_at, "source_url": f"https://www.openstreetmap.org/{osm_type}/{osm_id}"}}
    raise SourceError("This place has no usable polygon boundary in the provider response; supply an explicit area.")
