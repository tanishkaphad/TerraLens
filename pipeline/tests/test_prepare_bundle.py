"""Preparation geometry and provenance checks; no model loading or downloads."""
from __future__ import annotations

import importlib.util
import json
import pickle
from pathlib import Path
from unittest.mock import patch

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from affine import Affine
from pyproj import Transformer
from rasterio.io import MemoryFile
from rasterio.windows import Window
from shapely.geometry import Polygon, mapping


script_path = Path(__file__).resolve().parents[1] / "scripts" / "prepare_bundle.py"
spec = importlib.util.spec_from_file_location("prepare_bundle", script_path)
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)


def test_uint8_rgb_is_not_percentile_altered():
    pixels = np.array([[[12, 75], [150, 200]], [[24, 76], [180, 220]], [[48, 100], [240, 255]]], dtype=np.uint8)
    with MemoryFile() as memory:
        with memory.open(driver="GTiff", width=2, height=2, count=3, dtype="uint8", crs="EPSG:4326", transform=Affine(0.001, 0, -122, 0, -0.001, 38)) as dataset:
            dataset.write(pixels)
            np.testing.assert_array_equal(prepare.rgb_image(dataset, Window(0, 0, 2, 2)), np.moveaxis(pixels, 0, -1))


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16])
def test_scalar_all_nodata_mask_stays_black(dtype):
    class MaskedDataset:
        count, name = 3, "all-nodata"

        def read(self, indexes, *, window, masked):
            return np.ma.array(np.full((3, 2, 2), 200, dtype=dtype), mask=np.True_)

    np.testing.assert_array_equal(prepare.rgb_image(MaskedDataset(), Window(0, 0, 2, 2)), np.zeros((2, 2, 3), dtype=np.uint8))


def test_nodata_is_not_brightened_by_high_depth_stretch():
    class Dataset:
        count, name = 3, "negative-values"

        def read(self, indexes, *, window, masked):
            values = np.array([[[-100, -80], [-60, 0]]] * 3, dtype=np.int16)
            masks = np.array([[[False, False], [False, True]]] * 3)
            return np.ma.array(values, mask=masks)

    image = prepare.rgb_image(Dataset(), Window(0, 0, 2, 2))
    np.testing.assert_array_equal(image[1, 1], [0, 0, 0])
    assert image[1, 0, 0] == 255


def test_rotated_window_pixels_use_offset_once_and_keep_corner_order():
    parent = Affine(0.001, 0.0002, -122, 0.0001, -0.001, 38)
    window = Window(10, 20, 50, 40)
    tile_transform = rasterio.windows.transform(window, parent)
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:4326", always_xy=True)
    geometry, coordinates = prepare.pixel_box_geometry([1, 2, 4, 5], tile_transform, transformer)
    expected = [parent * point for point in [(11, 22), (14, 22), (14, 25), (11, 25)]]
    np.testing.assert_allclose(coordinates, expected)
    np.testing.assert_allclose(list(geometry.exterior.coords)[:4], expected)
    assert coordinates[0][1] > coordinates[3][1]


@pytest.mark.parametrize("length,expected", [(1000, [0]), (1024, [0]), (1100, [0, 76]), (2800, [0, 896, 1776]), (2816, [0, 896, 1792])])
def test_window_origins_cover_source_without_tiny_final_strips(length, expected):
    origins = prepare.tile_origins(length, 1024, 128)
    assert origins == expected
    intervals = [(origin, min(origin + 1024, length)) for origin in origins]
    assert intervals[0][0] == 0 and intervals[-1][1] == length
    assert all(right == left + min(length, 1024) for left, right in intervals)
    assert all(intervals[index][0] <= intervals[index - 1][1] for index in range(1, len(intervals)))


def test_bounds_only_mapping_requires_explicit_orientation_affirmation():
    class Dataset:
        crs, width, height, name = None, 400, 200, "skyscript.jpg"

    metadata = {"box": [-122.0, 37.9, -121.8, 38.0]}
    with pytest.raises(ValueError, match="bounds-georeference"):
        prepare.raster_georeference(Dataset(), metadata, False)
    crs, transform, georef = prepare.raster_georeference(Dataset(), metadata, True)
    assert crs.to_epsg() == 4326
    np.testing.assert_allclose(transform * (0, 0), [-122.0, 38.0])
    np.testing.assert_allclose(transform * (400, 200), [-121.8, 37.9])
    assert georef["registration_status"] == "bounds_approximation"


def test_dates_preserve_source_precision_and_creation_time_is_not_acquisition():
    assert prepare.acquisition_metadata({"time": [2019, 5, 2, 9, 45]}, {}) == ("2019-05-02T09:45", "sidecar")
    assert prepare.acquisition_metadata({}, {"TIFFTAG_DATETIME": "2026:10:09 10:00:00"}) == (None, "unknown")
    assert prepare.acquisition_metadata({}, {"DATE_ACQUIRED": "2022-06-04"}) == ("2022-06-04", "raster_acquisition_tag")
    assert prepare.acquisition_metadata({"acquisition_date": "2022-06"}, {}) == ("2022-06", "sidecar")
    with pytest.raises(ValueError):
        prepare.acquisition_metadata({"time": [2019, 13, 2, 9, 45]}, {})


def test_sky_script_primitive_pickle_and_json_keep_tags_as_annotations(tmp_path):
    metadata = {"box": (-122, 37.9, -121.8, 38), "time": (2019, 5, 2, 9, 45), "center_tags": {"building": "commercial"}, "surrounding_tags": [{"highway": "residential"}]}
    sidecar = tmp_path / "source.pickle"
    sidecar.write_bytes(pickle.dumps(metadata))
    assert prepare.read_metadata(sidecar) == metadata
    sidecar.write_bytes(pickle.dumps(Path("/tmp/unsupported-global")))
    with pytest.raises(ValueError, match="unsupported executable global"):
        prepare.read_metadata(sidecar)


def test_association_is_explicit_and_class_specific():
    concepts = prepare.parse_concepts(["panels=solar panels", "bare=bare ground"])
    assert concepts["panels"]["association_layer"] is None
    prepare.apply_associations(["panels=buildings"], concepts, {"buildings": object()})
    assert concepts["panels"]["association_layer"] == "buildings"
    assert concepts["bare"]["association_layer"] is None
    with pytest.raises(ValueError, match="configured concept"):
        prepare.apply_associations(["bare=roads"], concepts, {"buildings": object()})


def detection(identifier, *, date=None, source="same-source", concept="panels", score=0.8, shift=0):
    geometry = Polygon([[-122 + shift, 38], [-121.999 + shift, 38], [-121.999 + shift, 37.999], [-122 + shift, 37.999]])
    return {"type": "Feature", "id": identifier, "geometry": mapping(geometry), "properties": {
        "detection_id": identifier, "tile_id": f"tile-{identifier}", "concept_id": concept, "score": score,
        "acquisition_date": date, "source_sha256": source, "pixel_box": [0, 0, 10, 10]}}


def test_overlap_dedup_retains_strongest_observation_and_all_provenance():
    original = [detection("low", date="2022-06-04", score=0.7), detection("high", date="2022-06-04", source="adjacent-raster", score=0.9, shift=0.0001)]
    result = prepare.deduplicate_detections(original)
    assert len(result) == 1 and result[0]["id"] == "high"
    assert {record["detection_id"] for record in result[0]["properties"]["observations"]} == {"low", "high"}
    assert all(record["geometry"]["type"] == "Polygon" for record in result[0]["properties"]["observations"])
    assert "observations" not in original[0]["properties"]


def test_different_epochs_classes_and_unknown_sources_are_never_merged():
    assert len(prepare.deduplicate_detections([detection("a", date="2021-06-04"), detection("b", date="2022-06-04")])) == 2
    assert len(prepare.deduplicate_detections([detection("a"), detection("b", concept="road")])) == 2
    assert len(prepare.deduplicate_detections([detection("a"), detection("b", source="unknown-other-source")])) == 2
    assert len(prepare.deduplicate_detections([detection("a"), detection("b")])) == 1


def test_explicit_epoch_cannot_override_different_acquisition_dates():
    one, two = detection("a", date="2021-06-04"), detection("b", date="2022-06-04")
    one["properties"]["acquisition_epoch"] = two["properties"]["acquisition_epoch"] = "summer"
    assert len(prepare.deduplicate_detections([one, two])) == 2


@pytest.mark.parametrize("partial_date", ["2022", "2022-06"])
def test_partial_date_dedup_is_source_specific_unless_epoch_is_explicit(partial_date):
    one, two = detection("a", date=partial_date), detection("b", date=partial_date, source="other-source")
    assert len(prepare.deduplicate_detections([one, two])) == 2
    one["properties"]["acquisition_epoch"] = two["properties"]["acquisition_epoch"] = "verified-exposure"
    assert len(prepare.deduplicate_detections([one, two])) == 1
    two["properties"]["acquisition_date"] = "2021"
    assert len(prepare.deduplicate_detections([one, two])) == 2


def test_scene_profile_limits_use_most_restrictive_valid_distance():
    assert prepare.profile_limits([{}, {"profile": {"limits": {"maximum_distance_m": 500}}},
                                    {"profile": {"limits": {"maximum_distance_m": 250}}}]) == {"maximum_distance_m": 250}
    assert prepare.profile_limits([{}]) == {"maximum_distance_m": 5000}
    for value in (0, 5001, float("nan"), float("inf"), True):
        with pytest.raises(ValueError, match="maximum_distance_m"):
            prepare.profile_limits([{"profile": {"limits": {"maximum_distance_m": value}}}])


def test_bundle_version_tracks_evidence_encoder_detector_concepts_and_limits():
    baseline = ({"tiles.json": "tile-sha", "embeddings.npy": "embedding-sha"}, "encoder-sha", "detector-revision",
                {"panels": {"prompt": "solar panels", "association_layer": "buildings"}}, {"maximum_distance_m": 500})
    version = prepare.bundle_version(*baseline)
    reordered_artifacts = {"embeddings.npy": "embedding-sha", "tiles.json": "tile-sha"}
    assert prepare.bundle_version(reordered_artifacts, *baseline[1:]) == version
    changes = [
        ({"tiles.json": "different-sha", "embeddings.npy": "embedding-sha"}, *baseline[1:]),
        (baseline[0], "other-encoder-sha", *baseline[2:]),
        (*baseline[:2], "other-detector-revision", *baseline[3:]),
        (*baseline[:3], {"panels": {"prompt": "solar panels", "association_layer": None}}, baseline[4]),
        (*baseline[:4], {"maximum_distance_m": 250}),
    ]
    assert all(prepare.bundle_version(*change) != version for change in changes)


def test_bounded_preparation_preserves_pixels_dates_dims_and_digests(tmp_path):
    raster = tmp_path / "aerial.tif"
    with rasterio.open(raster, "w", driver="GTiff", width=1200, height=900, count=3, dtype="uint8", crs="EPSG:4326", transform=Affine(0.00001, 0, -122, 0, -0.00001, 38)) as dataset:
        dataset.write(np.full((3, 900, 1200), 128, dtype=np.uint8))
    vector = tmp_path / "buildings.geojson"
    gpd.GeoDataFrame({"osm_id": ["way/1"], "building": ["commercial"]}, geometry=[Polygon([[-122, 38], [-121.999, 38], [-121.999, 37.999], [-122, 37.999]])], crs="EPSG:4326").to_file(vector, driver="GeoJSON")
    checkpoint = tmp_path / "checkpoint.bin"
    checkpoint.write_bytes(b"unit-test-only-checkpoint")
    metadata = tmp_path / "aerial.metadata.json"
    metadata.write_text(json.dumps({"time": [2022, 6, 4, 12, 30], "center_tags": {"solar_panels": "yes"}, "source": "unit test", "profile": {"limits": {"maximum_distance_m": 500}}}))
    output = tmp_path / "bundle"
    calls = []

    def fake_detector(image, *, candidate_labels, threshold):
        calls.append((image.size, candidate_labels))
        return []  # Source solar tags must never create detector evidence.

    arguments = ["--area-name", "Test", "--imagery", str(raster), "--vectors", f"buildings={vector}",
                 "--concept", "panels=solar panels", "--associate", "panels=buildings", "--checkpoint", str(checkpoint),
                 "--metadata", f"{raster}={metadata}", "--max-tiles", "1", "--output", str(output)]
    with patch.object(prepare, "load_inference", return_value=(lambda image: np.array([1, 0], dtype=np.float32), fake_detector, "test-revision", {})):
        assert prepare.main(arguments) == 0
        with pytest.raises(SystemExit):
            prepare.main(arguments)  # Completed bundles cannot be overwritten.
    assert calls == [((1024, 900), ["solar panels"])]
    manifest, tiles = json.loads((output / "manifest.json").read_text()), json.loads((output / "tiles.json").read_text())
    tile = tiles[0]
    assert (tile["width"], tile["height"], tile["source_width"], tile["source_height"]) == (1024, 900, 1200, 900)
    assert (tile["preview_width"], tile["preview_height"]) == (800, 703)
    assert tile["acquisition_date"] == "2022-06-04T12:30"
    assert tile["source_annotation"]["evidence_type"] == "source_annotation"
    assert json.loads((output / "detections.geojson").read_text())["features"] == []
    assert manifest["coverage"]["preparation_truncated"] is True
    assert manifest["coverage"]["source_window_count"] == 2
    assert manifest["limits"]["maximum_distance_m"] == 500
    assert manifest["dataset_version"] == prepare.bundle_version(manifest["artifact_sha256"], manifest["image_encoder"]["checkpoint_sha256"],
        manifest["object_detector"]["revision"], manifest["supported_concepts"], manifest["limits"])
    assert manifest["vector_layers"]["buildings"]["extraction_date_status"] == "unknown"
    for relative, digest in manifest["artifact_sha256"].items():
        assert prepare.sha256_file(output / relative) == digest
    np.testing.assert_array_equal(np.load(output / "embeddings.npy"), [[1, 0]])
