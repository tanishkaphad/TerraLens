# Prototype API validation

Checked 2026-10-09T07:47:01.431487+00:00 with a fresh local process using the launcher’s bundle and matching SkyCLIP checkpoint.

All checks below passed through FastAPI TestClient against the real local imagery, vector files, embeddings and cached detections. External language-model credentials were disabled for this process.

| Dataset | Readiness | Tiles | Cached model candidates | Search layers | Maximum radius |
| --- | --- | ---: | ---: | --- | ---: |
| spacenet | ready; SkyCLIP retrieval ready | 3 | 19 | buildings, roads | 50 m |
| demo | ready; SkyCLIP retrieval ready | 4 | 152 | buildings, roads | 500 m |

| Query | Evidence / execution | Supported matches | Review candidates | Scene tiles |
| --- | --- | ---: | ---: | ---: |
| Find buildings within 20 metres of roads (spacenet) | cached_model_detection / catalog_scan | 2 | 17 | 0 |
| Make the radius 10 metres (spacenet) | cached_model_detection / catalog_scan | 1 | 18 | 0 |
| Find mapped buildings within 20 metres of roads (spacenet) | mapped_feature / catalog_scan | 434 | 23 | 0 |
| Find mapped roads (spacenet) | mapped_feature / catalog_scan | 34 | 0 | 0 |
| Find satellite imagery of densely built neighbourhoods (spacenet) | image_similarity / explore | 0 | 0 | 3 |
| Find swimming pools within 100 metres of roads (demo) | cached_model_detection / catalog_scan | 4 | 0 | 0 |

The 10 m follow-up preserved every previous plan field except the radius. Mapped building and road results used source vectors without visual-inference claims; scene retrieval ran real SkyCLIP text encoding and returned tile candidates without object locations. Both datasets served geographic image quadrilaterals, vector layers and image assets.

Searches used a page size of one to exercise paging. Complete GeoJSON exports retained every supported and review record plus associated observation and reference geometry:

| Export | Total GeoJSON features | Supported records | Review records | Complete |
| --- | ---: | ---: | ---: | --- |
| [spacenet-buildings20m.geojson](../pipeline/data/validation/prototype/spacenet-buildings20m.geojson) | 82 | 2 | 17 | Yes; truncated=false |
| [spacenet-scene.geojson](../pipeline/data/validation/prototype/spacenet-scene.geojson) | 3 | 0 | 0 | Yes; truncated=false |
| [naip-pools100m.geojson](../pipeline/data/validation/prototype/naip-pools100m.geojson) | 16 | 4 | 0 | Yes; truncated=false |

All exported geometries passed Shapely validity checks. A stale dataset-version export correctly returned HTTP 409. The raw counts, readiness, timings and export hashes are in [api-check.json](../pipeline/data/validation/prototype/api-check.json).

The parent process completed the full deterministic test suite: **240 passed**. After adding regression coverage for “Change the distance to …” and preparing the conditional Phoenix dataset registration, the full suite passed again: **242 passed in 5.53 seconds**.

These checks establish local API/model execution and complete exports. They do not establish independent visual accuracy, detector precision/recall, current real-world conditions, public hosting or live object grounding. Object searches in this run used precomputed OWLv2 candidates; scene results are image similarity, and model scores are uncalibrated.

## Phoenix default and all five UI examples

Checked 2026-10-09T07:53:11.230455+00:00. The launcher and dataset API now select **Phoenix / Encanto** by default when its completed bundle is present. SpaceNet and Sacramento remain available.

The new sample contains 1,000 × 1,000 RGB pixels covering 300 × 300 metres, at 0.3 m ground sampling distance, acquired 17 September 2023. Four image tiles and 127 deduplicated precomputed model candidates loaded with real SkyCLIP retrieval ready. The OSM extract contains 1,548 building and 666 road features; viewport/search subsets are smaller. Its database timestamp is **15 July 2026, 15:22:01 UTC**, and it was retrieved **9 October 2026, 07:49:29 UTC**. These different imagery/vector dates do not establish current conditions.

| Current UI example | Evidence | Supported matches | Review candidates | Scene tiles |
| --- | --- | ---: | ---: | ---: |
| Find swimming pools within 100 metres of roads | cached_model_detection | 25 | 0 | 0 |
| Find building roofs within 50 metres of roads | cached_model_detection | 56 | 31 | 0 |
| Find mapped buildings within 20 metres of roads | mapped_feature | 107 | 0 | 0 |
| Find mapped roads | mapped_feature | 31 | 0 | 0 |
| Find satellite imagery of densely built neighbourhoods | image_similarity | 0 | 0 | 4 |

All five current UI examples returned HTTP 200. An additional mapped-house query with `building=house`, a 100 m radius and `highway=residential` also passed with 86 supported records and no review records. Their complete GeoJSON exports passed geometry validity checks and retained all supported and review records. The native browser attachment endpoint for the pool query returned the same features as the full POST export; this verifies the endpoint response, not a completed browser download. All four source image assets were served successfully.

| Phoenix export | GeoJSON features | Supported records | Review records |
| --- | ---: | ---: | ---: |
| [phoenix-pools100m.geojson](../pipeline/data/validation/prototype/phoenix-pools100m.geojson) | 123 | 25 | 0 |
| [phoenix-roofs50m.geojson](../pipeline/data/validation/prototype/phoenix-roofs50m.geojson) | 294 | 56 | 31 |
| [phoenix-houses100m.geojson](../pipeline/data/validation/prototype/phoenix-houses100m.geojson) | 103 | 86 | 0 |
| [phoenix-mapped-buildings20m.geojson](../pipeline/data/validation/prototype/phoenix-mapped-buildings20m.geojson) | 143 | 107 | 0 |
| [phoenix-scene.geojson](../pipeline/data/validation/prototype/phoenix-scene.geojson) | 4 | 0 | 0 |
| [phoenix-mapped-roads.geojson](../pipeline/data/validation/prototype/phoenix-mapped-roads.geojson) | 31 | 31 | 0 |

The repaired “Change the distance to 10 metres” wording also passed against SpaceNet: 1 supported and 18 review records, preserving every other plan field.

Detailed counts, source dates, readiness and export hashes are recorded in [phoenix-api-check.json](../pipeline/data/validation/prototype/phoenix-api-check.json). Independent registration and visual accuracy remain unverified; no live object grounding, public hosting or current-conditions claim is made.

The separate browser check on `http://127.0.0.1:8010` confirmed the Phoenix pool search showed **25 supported candidates at 100 m**. Typing **“Change the distance to 10 metres”** updated the displayed count to **19**, while preserving the swimming-pool concept. This verifies the visible search and conversational update, independently of the API checks above. Browser download completion was not established; the native attachment API response was verified.
