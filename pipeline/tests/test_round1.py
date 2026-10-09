"""Round 1 contract over frozen visual evidence and a recorded OSM snapshot."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from fastapi.testclient import TestClient
import numpy as np
from shapely.geometry import shape

from terralens import api


PIPELINE_ROOT = Path(__file__).resolve().parents[1]
BUNDLE_ROOT = PIPELINE_ROOT / "data" / "spacenet-round1"
SKYCLIP_CHECKPOINT = PIPELINE_ROOT / "data" / "models" / "SkyCLIP_ViT_B32_top50pct.safetensors"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_round1_visual_spatial_search_has_reviewable_source_evidence(monkeypatch):
    """Exercise the public route without replacing the parser, encoder, bundle, or GIS engine."""
    assert BUNDLE_ROOT.is_dir()
    assert SKYCLIP_CHECKPOINT.is_file()
    monkeypatch.setenv("ENCODER_CHECKPOINT", str(SKYCLIP_CHECKPOINT))
    api.get_bundle.cache_clear()

    manifest = json.loads((BUNDLE_ROOT / "manifest.json").read_text())
    cached = json.loads((BUNDLE_ROOT / "detections.geojson").read_text())
    annotations = json.loads((BUNDLE_ROOT / "vectors" / "buildings.geojson").read_text())
    roads = json.loads((BUNDLE_ROOT / "vectors" / "roads.geojson").read_text())
    tiles = json.loads((BUNDLE_ROOT / "tiles.json").read_text())
    cached_by_id = {feature["properties"]["detection_id"]: feature for feature in cached["features"]}
    annotations_by_id = {feature["properties"]["source_feature_id"]: feature for feature in annotations["features"]}
    roads_by_id = {str(feature["properties"]["osm_id"]): feature for feature in roads["features"]}
    tiles_by_id = {tile["tile_id"]: tile for tile in tiles}

    try:
        client = TestClient(api.app)
        response = client.post("/api/query", json={
            "query": "Find buildings within 50 metres of roads",
            "dataset_id": "spacenet",
            "mode": "catalog_scan",
            "page_size": 100,
        })
        assert response.status_code == 200, response.text
        result = response.json()

        assert result["dataset_id"] == "spacenet"
        assert result["dataset_version"] == manifest["dataset_version"]
        assert result["plan"] == {
            "query_type": "objects",
            "concept_id": "buildings",
            "visual_query": "buildings",
            "reference_layer": "roads",
            "predicate": "within_distance",
            "distance_m": 50.0,
            "attributes": [],
            "unsupported_conditions": [],
            "subject_layer": None,
            "relations": [],
        }
        assert result["result_kind"] == "grounded_objects"
        assert result["evidence_kind"] == "cached_model_detection"
        assert result["text_similarity_enabled"] is True
        assert result["text_encoder_status"] == "ready"
        assert result["objects_evaluated"] == 19
        assert result["total_matches"] == 2
        assert result["total_uncertain"] == 17
        assert result["objects_excluded"] == 0
        assert result["truncated"] is False
        assert set(feature["id"] for feature in result["features"]) == {
            "buildings:spacenet:img1026:116",
            "buildings:spacenet:img1026:96",
        }

        evidence = [*result["features"], *result["uncertain_candidates"]]
        assert len(evidence) == len(cached["features"]) == 19
        assert max(evidence, key=lambda feature: feature["properties"]["visual_score"])["properties"]["role"] == "uncertain_candidate"
        assert "no guarantee" in result["claim_scope"]
        contexts_by_id = {feature["id"]: feature for feature in result["context_features"]}
        assert contexts_by_id

        for feature in result["features"]:
            properties = feature["properties"]
            assert properties["role"] == "match"
            assert properties["evidence_status"] == "supported_by_available_evidence"
            assert properties["association"] == "unique"
            assert properties["geometry_role"] == "associated_vector_footprint"
            assert properties["imagery_date"] is None
            assert properties["imagery_date_status"] == "unknown"
            assert math.isfinite(properties["retrieval_similarity"])
            assert -1 <= properties["retrieval_similarity"] <= 1

            visual, association, spatial = properties["conditions"]
            assert visual["condition"] == "visual_detection"
            assert visual["status"] == "pass"
            assert visual["score_meaning"] == "uncalibrated_model_score"
            assert visual["model_revision"] == manifest["object_detector"]["revision"]
            assert association == {
                "condition": "feature_association",
                "status": "pass",
                "layer": "buildings",
                "feature_id": properties["id"],
                "candidate_feature_ids": [properties["id"]],
                "association": "unique",
            }
            assert spatial["condition"] == "within_distance"
            assert spatial["status"] == "pass"
            assert spatial["reference_layer"] == "roads"
            assert spatial["radius_m"] == 50.0
            assert 0 <= spatial["distance_m"] <= 50
            assert spatial["distance_basis"] == "associated_vector_footprint"
            assert spatial["reference_coverage_status"] == "bounded_osm_snapshot_partial"
            assert spatial["reference_feature_ids"]
            for road_id in spatial["reference_feature_ids"]:
                context = contexts_by_id[f"roads:{road_id}"]
                raw_road = roads_by_id[road_id]
                assert context["properties"]["role"] == "reference"
                assert context["properties"]["attribution"] == "OpenStreetMap contributors"
                assert context["properties"]["source_attributes"]["source_url"] == (
                    f"https://www.openstreetmap.org/way/{road_id}"
                )
                assert shape(context["geometry"]).equals(shape(raw_road["geometry"]))

            annotation = annotations_by_id[properties["id"]]
            raw_detection = cached_by_id[properties["detection_id"]]
            tile = tiles_by_id[properties["tile_id"]]
            assert shape(feature["geometry"]).equals(shape(annotation["geometry"]))
            assert len(properties["evidence_geometries"]) == 1
            detection_geometry = properties["evidence_geometries"][0]
            assert detection_geometry["properties"]["role"] == "visual_detection"
            assert detection_geometry["properties"]["geometry_role"] == "visual_detection_envelope"
            assert shape(detection_geometry["geometry"]).equals(shape(raw_detection["geometry"]))
            assert properties["pixel_box"] == raw_detection["properties"]["pixel_box"]
            x0, y0, x1, y1 = properties["pixel_box"]
            assert 0 <= x0 < x1 <= tile["width"]
            assert 0 <= y0 < y1 <= tile["height"]
            assert properties["source_sha256"] == tile["source_sha256"]
            assert properties["asset_url"].endswith("?dataset_id=spacenet")

        for feature in result["uncertain_candidates"]:
            properties = feature["properties"]
            assert properties["role"] == "uncertain_candidate"
            assert properties["evidence_status"] == "unknown"
            assert properties["association"] == "unmatched"
            assert properties["geometry_role"] == "visual_detection_envelope"
            assert properties["conditions"][0]["score_meaning"] == "uncalibrated_model_score"
            assert properties["conditions"][1]["status"] == "unknown"
            assert properties["conditions"][1]["candidate_feature_ids"] == []
            spatial = properties["conditions"][2]
            assert spatial["condition"] == "within_distance"
            assert spatial["status"] == "unknown"
            assert spatial["reference_layer"] == "roads"
            assert spatial["radius_m"] == 50.0
            assert spatial["distance_basis"] == "visual_detection_envelope"
            if spatial["distance_m"] is not None:
                assert 0 <= spatial["distance_m"] <= 50

        provenance = result["provenance"]
        assert provenance["coverage"]["registration_status"] == "unverified"
        layer_metadata = provenance["vector_layers"]["buildings"]["metadata"]
        assert layer_metadata["feature_count"] == 457
        assert layer_metadata["coverage_status"] == "partial_training_labels"
        assert layer_metadata["reference_coverage_complete"] is False
        roads_provenance = provenance["vector_layers"]["roads"]
        assert roads_provenance["source"] == "https://www.openstreetmap.org"
        assert roads_provenance["source_url"] == "https://overpass-api.de/api/interpreter"
        assert roads_provenance["source_date"] == "2026-10-09T07:13:19Z"
        assert roads_provenance["coverage_status"] == "bounded_osm_snapshot_partial"
        assert roads_provenance["reference_coverage_complete"] is False
        assert roads_provenance["maximum_query_radius_m"] == 50.0
        assert roads_provenance["metadata"]["feature_count"] == 56
        assert roads_provenance["metadata"]["study_intersection_feature_count"] == 47
        assert roads_provenance["metadata"]["halo_only_feature_count"] == 9
        assert roads_provenance["metadata"]["temporal_alignment_status"] == "unknown"
        assert roads_provenance["metadata"]["sha256"] == _sha256(
            BUNDLE_ROOT / "vectors" / "roads.geojson"
        )
        assert roads_provenance["metadata"]["raw_response_sha256"] == _sha256(
            BUNDLE_ROOT / "provenance" / "osm-response.json"
        )
        assert roads_provenance["metadata"]["query_sha256"] == _sha256(
            BUNDLE_ROOT / "provenance" / "osm-query.txt"
        )
        assert provenance["image_encoder"]["embedding_dimension"] == 512
        assert _sha256(SKYCLIP_CHECKPOINT) == provenance["image_encoder"]["checkpoint_sha256"]
        owl_checkpoint = (PIPELINE_ROOT / "data" / "models" /
                          f"owlv2-{provenance['object_detector']['revision']}" / "model.safetensors")
        assert _sha256(owl_checkpoint) == provenance["object_detector"]["model_files_sha256"]["model.safetensors"]
        embeddings = np.load(BUNDLE_ROOT / "embeddings.npy", allow_pickle=False)
        assert embeddings.shape == (3, 512)
        assert np.allclose(np.linalg.norm(embeddings, axis=1), 1.0, atol=0.02)
        for relative, expected in provenance["artifact_sha256"].items():
            assert _sha256(BUNDLE_ROOT / relative) == expected

        export_response = client.post("/api/export", json={
            "plan": result["plan"],
            "dataset_id": "spacenet",
            "dataset_version": result["dataset_version"],
            "mode": "catalog_scan",
            "include_uncertain": True,
        })
        assert export_response.status_code == 200, export_response.text
        assert export_response.headers["content-type"].startswith("application/geo+json")
        assert "attachment" in export_response.headers["content-disposition"]
        exported = export_response.json()
        assert exported["type"] == "FeatureCollection"
        assert exported["metadata"]["plan"] == result["plan"]
        assert exported["metadata"]["dataset_version"] == result["dataset_version"]
        assert exported["metadata"]["total_matches"] == 2
        assert exported["metadata"]["total_uncertain"] == 17
        assert exported["metadata"]["include_uncertain"] is True
        exported_by_id = {feature["id"]: feature for feature in exported["features"]}
        for subject in evidence:
            properties = subject["properties"]
            saved_subject = exported_by_id[subject["id"]]
            assert saved_subject["properties"]["source_sha256"] == properties["source_sha256"]
            assert saved_subject["properties"]["source_image"] == properties["source_image"]
            assert saved_subject["properties"]["imagery_date"] is None
            assert saved_subject["properties"]["imagery_date_status"] == "unknown"
            detection_id = f"detection:{properties['detection_id']}"
            raw_detection = exported_by_id[detection_id]
            assert raw_detection["properties"]["role"] == "visual_detection"
            assert shape(raw_detection["geometry"]).equals(
                shape(cached_by_id[properties["detection_id"]]["geometry"])
            )
            spatial = properties["conditions"][2]
            for road_id in spatial["reference_feature_ids"]:
                saved_reference = exported_by_id[f"roads:{road_id}"]
                assert saved_reference["properties"]["role"] == "reference"
                assert saved_reference["properties"]["source_attributes"]["source_url"] == (
                    f"https://www.openstreetmap.org/way/{road_id}"
                )
                assert saved_reference["properties"]["vector_extraction_date"] == (
                    roads_provenance["extraction_date"]
                )
        assert "population" not in json.dumps(exported).casefold()

        map_response = client.get("/api/map", params={
            "dataset_id": "spacenet", "tile_limit": 3, "vector_limit": 5000,
        })
        assert map_response.status_code == 200, map_response.text
        mapped = map_response.json()
        assert mapped["dataset_version"] == result["dataset_version"]
        assert mapped["coverage"] == result["coverage"]
        assert mapped["counts"]["tiles"] == {"available": 3, "returned": 3, "truncated": False}
        assert mapped["counts"]["vector_layers"]["buildings"] == {
            "available": 457, "returned": 457, "truncated": False,
        }
        assert mapped["counts"]["vector_layers"]["roads"] == {
            "available": 56, "returned": 56, "truncated": False,
        }
        assert len(mapped["tiles"]) == 3
        assert len(mapped["vector_layers"]["buildings"]["features"]) == 457
        assert len(mapped["vector_layers"]["roads"]["features"]) == 56
        assert all(feature["properties"]["role"] == "base_vector"
                   for feature in mapped["vector_layers"]["buildings"]["features"])
        assert all(feature["properties"]["role"] == "base_vector"
                   for feature in mapped["vector_layers"]["roads"]["features"])

        preview = mapped["tiles"][0]
        asset = client.get(preview["asset_url"])
        assert asset.status_code == 200
        assert asset.headers["content-type"] == "image/jpeg"
        expected_preview_hash = manifest["artifact_sha256"][tiles_by_id[preview["tile_id"]]["preview"]]
        assert hashlib.sha256(asset.content).hexdigest() == expected_preview_hash
    finally:
        api.get_bundle.cache_clear()
