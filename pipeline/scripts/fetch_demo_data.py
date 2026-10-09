#!/usr/bin/env python3
"""Fetch a bounded, genuine Sacramento NAIP/OSM source corpus.

No inferred labels are added. Existing completed source files are verified and
reused by default; choose a new output directory to change the scene.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import ssl
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

STAC = "https://planetarycomputer.microsoft.com/api/stac/v1"
SIGN = "https://planetarycomputer.microsoft.com/api/sas/v1/sign"
DEFAULT_ITEM = "ca_m_3812129_sw_10_060_20220621"
MAX_JSON_BYTES = 24 * 1024 * 1024


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def write_json(path: Path, value: dict) -> None:
    if path.exists():
        raise RuntimeError(f"Refusing to overwrite existing source artifact: {path}")
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def request_json(url: str, data: bytes | None = None) -> dict:
    import certifi

    request = urllib.request.Request(
        url, data=data,
        headers={"User-Agent": "TerraLens bounded GeoAI research demo/1.0"},
    )
    context = ssl.create_default_context(cafile=certifi.where())
    with urllib.request.urlopen(request, context=context, timeout=45) as response:
        raw = response.read(MAX_JSON_BYTES + 1)
    if len(raw) > MAX_JSON_BYTES:
        raise RuntimeError("Public source response exceeded the bounded JSON limit.")
    return json.loads(raw)


def projected_bounds(center: tuple[float, float], half_width: float, crs) -> tuple:
    from pyproj import Transformer

    x, y = Transformer.from_crs(4326, crs, always_xy=True).transform(*center)
    return x - half_width, y - half_width, x + half_width, y + half_width


def geographic_bounds(projected: tuple, crs) -> list[float]:
    from rasterio.warp import transform_bounds

    return list(transform_bounds(crs, "EPSG:4326", *projected))


def verify_existing(output: Path, args) -> bool:
    image = output / "sacramento-naip-rgb.tif"
    if not image.exists():
        return False
    import rasterio
    from pyproj import Transformer

    meta_path = image.with_suffix(image.suffix + ".metadata.json")
    required = [meta_path, output / "naip-item.json", output / "osm-response.json",
                output / "osm-query.txt", output / "preview.jpg"]
    required += [output / "vectors" / f"{layer}.geojson{suffix}"
                 for layer in ("buildings", "roads", "waterways")
                 for suffix in ("", ".metadata.json")]
    if not all(path.is_file() for path in required):
        raise RuntimeError("Existing source output is incomplete; choose a new --output directory.")
    meta = json.loads(meta_path.read_text())
    if meta.get("source_item_id") != args.item_id or digest(image) != meta.get("sha256"):
        raise RuntimeError("Existing imagery item/hash differs; choose a new --output directory.")
    with rasterio.open(image) as src:
        requested = projected_bounds((args.longitude, args.latitude), args.size_m / 2, src.crs)
        if any(abs(a - b) > 2 for a, b in zip(src.bounds, requested)):
            raise RuntimeError("Existing crop differs from requested bounds; choose a new --output directory.")
        if src.count != 3 or max(src.width, src.height) > 2048:
            raise RuntimeError("Existing crop is not a bounded RGB scene.")
        outer = projected_bounds((args.longitude, args.latitude), args.size_m / 2 + 525, src.crs)
        expected_osm = geographic_bounds(outer, src.crs)
    for layer in ("buildings", "roads", "waterways"):
        path = output / "vectors" / f"{layer}.geojson"
        metadata = json.loads(path.with_suffix(path.suffix + ".metadata.json").read_text())
        if digest(path) != metadata.get("sha256"):
            raise RuntimeError(f"Existing {layer} vector hash differs; choose a new output directory.")
        bounds = metadata.get("coverage_bbox_wgs84", [])
        if len(bounds) != 4 or any(abs(a - b) > 0.00001 for a, b in zip(bounds, expected_osm)):
            raise RuntimeError("Existing OSM extent differs from requested scene/margin.")
    return True


def fetch_imagery(output: Path, args) -> tuple[dict, list[float]]:
    import rasterio
    from rasterio.windows import from_bounds, bounds as window_bounds
    from PIL import Image

    item = request_json(f"{STAC}/collections/naip/items/{urllib.parse.quote(args.item_id, safe='')}")
    write_json(output / "naip-item.json", item)
    asset = item["assets"]["image"]["href"]
    signed = request_json(SIGN + "?" + urllib.parse.urlencode({"href": asset}))["href"]
    print("Reading only the requested RGB raster window.", flush=True)
    with rasterio.Env(GDAL_HTTP_TIMEOUT="40", GDAL_HTTP_MAX_RETRY="1",
                      GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
                      CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif"):
        with rasterio.open(signed) as src:
            if not src.crs or not src.crs.is_projected:
                raise RuntimeError("NAIP source must have a projected, metre-based CRS.")
            from pyproj import CRS
            if any(abs(axis.unit_conversion_factor - 1) > 1e-9 for axis in CRS(src.crs).axis_info):
                raise RuntimeError("NAIP source CRS axes must use metres.")
            crop = projected_bounds((args.longitude, args.latitude), args.size_m / 2, src.crs)
            if not (src.bounds.left <= crop[0] < crop[2] <= src.bounds.right
                    and src.bounds.bottom <= crop[1] < crop[3] <= src.bounds.top):
                raise RuntimeError("Requested crop is outside the selected NAIP item.")
            window = from_bounds(*crop, src.transform).round_offsets().round_lengths()
            if max(window.width, window.height) > 2048 or min(window.width, window.height) < 1:
                raise RuntimeError("Crop would exceed 2048 pixels; reduce --size-m.")
            array = src.read([1, 2, 3], window=window)
            profile = src.profile.copy()
            profile.update(count=3, width=array.shape[2], height=array.shape[1],
                           transform=src.window_transform(window), compress="deflate")
            path = output / "sacramento-naip-rgb.tif"
            with rasterio.open(path, "w", **profile) as dest:
                dest.write(array)
            Image.fromarray(array.transpose(1, 2, 0)).save(output / "preview.jpg", quality=92)
            bounds = geographic_bounds(window_bounds(window, src.transform), src.crs)
            osm_bounds = geographic_bounds(projected_bounds(
                (args.longitude, args.latitude), args.size_m / 2 + 525, src.crs), src.crs)
            crs_name = src.crs.to_string()
            pixel_size = [abs(src.transform.a), abs(src.transform.e)]
    metadata = {
        "source_name": "USDA NAIP via Microsoft Planetary Computer",
        "source": "https://planetarycomputer.microsoft.com/dataset/naip",
        "source_url": "https://planetarycomputer.microsoft.com/dataset/naip",
        "source_item_id": item["id"], "asset_url": asset,
        "acquisition_date": item["properties"].get("datetime"),
        "crs": crs_name, "bounds_wgs84": bounds,
        "width": int(array.shape[2]), "height": int(array.shape[1]),
        "pixel_size_m": pixel_size, "gsd_m": item["properties"].get("gsd"),
        "bands": ["red", "green", "blue"],
        "license": "USDA NAIP public domain; review dataset-specific terms before redistribution",
        "attribution": "USDA Farm Service Agency / NAIP; hosted by Microsoft Planetary Computer",
        "sha256": digest(path), "profile": {"limits": {"maximum_distance_m": 500}},
        "crop_request": {"center_wgs84": [args.longitude, args.latitude], "size_m": args.size_m},
    }
    write_json(path.with_suffix(path.suffix + ".metadata.json"), metadata)
    return metadata, osm_bounds


def osm_geometry(element: dict, layer: str):
    from shapely.geometry import LineString, Polygon
    from shapely.ops import polygonize, unary_union

    def points(value):
        return [(point["lon"], point["lat"]) for point in value if "lon" in point and "lat" in point]

    if element["type"] == "way":
        coordinates = points(element.get("geometry", []))
        if len(coordinates) < 2:
            raise ValueError("missing way geometry")
        if layer == "buildings":
            if len(coordinates) < 4 or coordinates[0] != coordinates[-1]:
                raise ValueError("unclosed building geometry")
            return Polygon(coordinates)
        return LineString(coordinates)
    if element.get("tags", {}).get("type") not in ("multipolygon", "boundary"):
        raise ValueError("non-polygon relation is unsupported")
    outer, inner = [], []
    for member in element.get("members", []):
        coordinates = points(member.get("geometry", []))
        if len(coordinates) >= 2:
            (inner if member.get("role") == "inner" else outer).append(LineString(coordinates))
    if not outer:
        raise ValueError("missing relation outer geometry")
    geometry = unary_union(list(polygonize(unary_union(outer))))
    if inner:
        geometry = geometry.difference(unary_union(list(polygonize(unary_union(inner)))))
    return geometry


def fetch_osm(output: Path, bounds: list[float]) -> None:
    from shapely.geometry import mapping

    west, south, east, north = bounds
    area = f"({south},{west},{north},{east})"
    query = ("[out:json][timeout:25];("
             + "".join(f"way[{tag}]{area};" for tag in ("building", "highway", "waterway"))
             + "".join(f"relation[{tag}]{area};" for tag in ("building", "waterway"))
             + ");out body geom;")
    raw, endpoint = None, None
    for endpoint in ("https://overpass-api.de/api/interpreter", "https://overpass.kumi.systems/api/interpreter"):
        try:
            raw = request_json(endpoint, urllib.parse.urlencode({"data": query}).encode())
            if raw.get("remark"):
                raise RuntimeError(raw["remark"])
            break
        except Exception as error:
            print(f"Bounded Overpass request failed at {endpoint}: {error}", flush=True)
            raw = None
    if raw is None:
        raise RuntimeError("Both bounded Overpass requests failed; no vector data was fabricated.")
    write_json(output / "osm-response.json", raw)
    (output / "osm-query.txt").write_text(query, encoding="utf-8")
    layers = {name: [] for name in ("buildings", "roads", "waterways")}
    skipped = []
    seen = set()
    for element in raw.get("elements", []):
        tags = element.get("tags", {})
        layer = ("buildings" if "building" in tags else "roads" if "highway" in tags
                 else "waterways" if "waterway" in tags else None)
        if not layer:
            continue
        identity = f'osm/{element["type"]}/{element["id"]}'
        if identity in seen:
            continue
        seen.add(identity)
        try:
            geometry = osm_geometry(element, layer)
            if geometry.is_empty or not geometry.is_valid:
                raise ValueError("empty or invalid geometry")
        except (ValueError, KeyError) as error:
            skipped.append({"type": element["type"], "id": element["id"], "reason": str(error)})
            continue
        properties = dict(tags)
        properties.update({"id": identity, "osm_type": element["type"], "osm_id": element["id"],
                           "source_name": "OpenStreetMap",
                           "source_url": f'https://www.openstreetmap.org/{element["type"]}/{element["id"]}'})
        layers[layer].append({"type": "Feature", "id": identity,
                              "properties": properties, "geometry": mapping(geometry)})
    retrieved = datetime.now(timezone.utc).isoformat()
    vector_dir = output / "vectors"
    vector_dir.mkdir()
    for layer, features in layers.items():
        path = vector_dir / f"{layer}.geojson"
        write_json(path, {"type": "FeatureCollection", "features": features})
        write_json(path.with_suffix(path.suffix + ".metadata.json"), {
            "source_name": "OpenStreetMap", "source": "https://www.openstreetmap.org",
            "source_url": endpoint, "retrieved_at": retrieved, "extraction_date": retrieved,
            "source_date": raw.get("osm3s", {}).get("timestamp_osm_base"),
            "license": "Open Data Commons Open Database License (ODbL) 1.0",
            "attribution": "OpenStreetMap contributors", "crs": "EPSG:4326",
            "coverage_bbox_wgs84": bounds, "bbox_wgs84": bounds,
            "coverage_margin_m": 525, "maximum_query_radius_m": 500,
            "feature_count": len(features), "sha256": digest(path),
            "raw_response": "../osm-response.json", "query": "../osm-query.txt",
            "query_scope": "OSM tagged ways and building/waterway polygon relations; not a completeness certification",
            "excluded_invalid_or_unclosed_features": skipped,
            "relation_policy": "Polygonize outer way members, subtract inner members; skip unsupported non-polygon relations",
        })
        print(f"{layer}: {len(features)} source features", flush=True)
    print(f"Skipped {len(skipped)} unsupported/invalid source features; details retained in sidecars.", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/source-fast"))
    parser.add_argument("--item-id", default=DEFAULT_ITEM)
    parser.add_argument("--longitude", type=float, default=-121.4815)
    parser.add_argument("--latitude", type=float, default=38.5485)
    parser.add_argument("--size-m", type=float, default=600)
    parser.add_argument("--skip-existing", action="store_true", default=True,
                        help="Verify and reuse completed source output (default); never overwrite raw source files.")
    args = parser.parse_args()
    if not all(math.isfinite(value) for value in (args.longitude, args.latitude, args.size_m)):
        parser.error("Scene coordinates and size must be finite.")
    if not (-180 <= args.longitude <= 180 and -90 < args.latitude < 90 and 100 <= args.size_m <= 1200):
        parser.error("Use valid coordinates and a crop between 100 and 1200 metres wide.")
    output = args.output.resolve()
    if verify_existing(output, args):
        print("Verified existing bounded sources; no network requests or source writes performed.")
    else:
        if output.exists() and any(output.iterdir()):
            raise RuntimeError("Output directory contains incomplete artifacts; choose a new --output directory.")
        output.mkdir(parents=True, exist_ok=True)
        _, bounds = fetch_imagery(output, args)
        fetch_osm(output, bounds)
    print(f"Imagery: {output / 'sacramento-naip-rgb.tif'}")
    print(f"Imagery metadata: {output / 'sacramento-naip-rgb.tif.metadata.json'}")
    for layer in ("buildings", "roads", "waterways"):
        path = output / "vectors" / f"{layer}.geojson"
        feature_count = json.loads(path.with_suffix(path.suffix + ".metadata.json").read_text())["feature_count"]
        print(f"Vector {layer}: {path} ({feature_count} features)")
    print("Zero returned waterways does not establish absence; leave an empty layer unconfigured.")
    print("NAIP imagery dates and OSM snapshot dates differ; both are retained without a contemporaneity claim.")


if __name__ == "__main__":
    main()
