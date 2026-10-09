"""Bounded SpaceNet acquisition logic; fixtures are synthetic and offline."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from rasterio.warp import transform_bounds
from shapely.geometry import box, mapping

MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts/fetch_spacenet_sample.py"
SPEC = importlib.util.spec_from_file_location("fetch_spacenet_sample", MODULE_PATH)
sample = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(sample)


def feature_collection(geometry=None, *, label_id=1):
    return json.dumps({"type": "FeatureCollection", "features": [{"type": "Feature",
        "geometry": geometry or mapping(box(3, 0.9, 3.00001, 0.90001)),
        "properties": {"BuildingId": label_id, "Area": 12}}]}).encode()


def raster_fixture(path, *, zero=False):
    transform = from_origin(500000, 100000, 0.3, 0.3)
    rgb = np.zeros((3, 50, 50), dtype=np.uint16) if zero else np.arange(7500, dtype=np.uint16).reshape(3, 50, 50)
    with rasterio.open(path, "w", driver="GTiff", width=50, height=50, count=3, dtype="uint16",
                       crs="EPSG:32631", transform=transform, nodata=0) as writer:
        writer.write(rgb)
    extent = transform_bounds("EPSG:32631", "EPSG:4326", 500000, 99985, 500015, 100000)
    return [{"geometry": mapping(box(*extent))}]


def test_listing_pairs_only_canonical_nonempty_annotation_candidates():
    raw = b'''<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">
        <Contents><Key>spacenet/SN1_buildings/train/geojson/Geo_AOI_1_RIO_img1.geojson</Key><Size>217</Size></Contents>
        <Contents><Key>spacenet/SN1_buildings/train/geojson/Geo_AOI_1_RIO_img100.geojson</Key><Size>1600</Size><LastModified>2025-01-01</LastModified></Contents>
        <Contents><Key>spacenet/SN1_buildings/train/geojson/untrusted.tif</Key><Size>2000</Size></Contents>
        </ListBucketResult>'''
    entries = sample.candidate_entries(raw)
    assert len(entries) == 1
    assert entries[0]["image_id"] == "img100"
    assert entries[0]["raster_key"] == "spacenet/SN1_buildings/train/3band/3band_AOI_1_RIO_img100.tif"
    assert "acquisition_date" not in entries[0]


def test_annotations_preserve_source_geometry_and_label_identity_without_use_claims():
    features = sample.annotations(feature_collection(), "img100")
    assert features[0]["properties"]["source_feature_id"] == "spacenet:img100:1"
    assert features[0]["properties"]["evidence_type"] == "dataset_annotation"
    assert features[0]["properties"]["annotation_properties"]["BuildingId"] == 1
    assert "building:use" not in features[0]["properties"]


@pytest.mark.parametrize("document", [
    {"type": "FeatureCollection", "features": []},
    {"type": "FeatureCollection", "features": [{"type": "Feature", "geometry": None}]},
    {"type": "FeatureCollection", "features": [{"type": "Feature", "geometry": {"type": "Polygon",
        "coordinates": [[[3, 0], [4, 1], [4, 0], [3, 1], [3, 0]]]}}]},
])
def test_empty_missing_and_invalid_annotations_are_rejected(document):
    with pytest.raises(ValueError):
        sample.annotations(json.dumps(document).encode(), "img1")


def test_source_raster_georeference_uint16_and_unknown_date_are_preserved(tmp_path):
    path = tmp_path / "source.tif"
    features = raster_fixture(path)
    inspection = sample.inspect_raster(path, features)
    assert inspection["dtype"] == "uint16"
    assert inspection["crs"] == "EPSG:32631"
    assert inspection["width"] == inspection["height"] == 50
    assert inspection["acquisition_date"] is None
    assert inspection["acquisition_date_status"] == "unknown"
    metadata = sample.imagery_metadata({"raster_key": "raster.tif", "image_id": "img1"}, inspection, path,
        "2026-10-09T00:00:00Z", {"http_last_modified": "2025-01-01"})
    assert metadata["license"] == "CC BY-SA 4.0"
    assert metadata["acquisition_date"] is None
    assert metadata["download_headers"]["http_last_modified"] == "2025-01-01"


def test_nodata_only_raster_is_rejected(tmp_path):
    path = tmp_path / "empty.tif"
    features = raster_fixture(path, zero=True)
    with pytest.raises(ValueError, match="no usable RGB"):
        sample.inspect_raster(path, features)


def test_unaligned_annotation_is_rejected(tmp_path):
    path = tmp_path / "source.tif"
    raster_fixture(path)
    with pytest.raises(ValueError, match="do not align"):
        sample.inspect_raster(path, [{"geometry": mapping(box(100, 10, 101, 11))}])


def test_existing_source_files_are_not_overwritten(tmp_path):
    path = tmp_path / "source.json"
    path.write_text("original")
    with pytest.raises(RuntimeError, match="overwrite"):
        sample.write_json(path, {"replacement": True})
    assert path.read_text() == "original"


def test_preparation_uses_bounded_local_models_and_building_annotation_association(tmp_path):
    args = SimpleNamespace(checkpoint=tmp_path / "sky.safetensors", owl_model=tmp_path / "owl",
        count=3, bundle_output=tmp_path / "bundle")
    command = sample.preparation_command(tmp_path, {"imagery": [{"file": "rgb.tif", "metadata": "rgb.json"}],
        "buildings": {"file": "buildings.geojson", "metadata": "buildings.json"}}, args)
    assert command[command.index("--max-tiles") + 1] == "3"
    assert command[command.index("--associate") + 1] == "buildings=buildings"
    assert command[command.index("--concept") + 1] == "buildings=buildings"
    assert command[command.index("--owl-revision") + 1] == sample.OWL_REVISION
    assert "CC BY-SA 4.0" in command[command.index("--imagery-attribution") + 1]
    assert "roads" not in " ".join(command)
