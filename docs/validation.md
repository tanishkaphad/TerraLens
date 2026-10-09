# Initial implementation checks — 9 October 2026

This records the first local development run, rather than an independently annotated accuracy benchmark or a public deployment check.

## Real model and source execution

The corpus is the 600-metre Sacramento NAIP scene documented in [datasets.md](datasets.md), paired with the actual 2026 OSM snapshot. Four overlapping 640-pixel windows were processed on CPU using official SkyCLIP ViT-B/32 top50pct and pinned OWLv2 base-patch16-ensemble. RGB values, transforms, original detector boxes, acquisition dates and model identities were retained.

SkyCLIP's official training checkpoint required a model-only conversion for the current weights-only loader. No model tensors were changed. The converted SHA-256 is `ae8a2553865b0d74063a03f5d639d3d88d9a7375361f62103d6e55c11b3fbd2a`. Text and image embeddings use that same checkpoint; the text probe produces a finite 512-dimensional vector.

At the uncalibrated 0.1 screening cutoff, inference emitted 201 raw observations. Geographic overlap deduplication retained **152 detections: 148 building roofs and 4 swimming pools**. This is a count of model output, not a count of correctly identified real objects. Every merged observation retains its own tile, pixel box and geometry.

The original diagnostic bundle was preserved as `data/demo-legacy-v1`. Its artifact-identical copy now has the content fingerprint `terralens-evidence-v1-448261eff535fc7b860d7a540606af4fb770b1ff04d88e3720ea5e6c9d75aa78`. Preparation derives dataset versions from artifact/model/configuration fingerprints, so a changed bundle invalidates stale query/export plans. The copy changes only manifest identity; model inference was not rerun.

## Actual API checks

`scripts/verify_demo.py` checked a running local FastAPI service with genuine source/model artifacts. Its final [machine-readable receipt](verification-receipt.json) is included here; full GeoJSON outputs are kept in ignored `pipeline/data/validation/`.

| Request | Observed result | One local request duration |
| --- | --- | --- |
| `/ready` including encoder load/probe | Ready; text ranking and cached object/spatial search available | 4.250 s |
| Bounded imagery/vector map | 4 tiles, 1,079 roads, 2,972 buildings; no display truncation | 1.001 s |
| Building roofs within 100 m of roads | 69 matches; 79 association-review candidates; 148 detections evaluated | 1.036 s |
| Swimming pools within 100 m of roads | 4 model candidates passed available spatial evidence; 4 evaluated | 0.065 s |
| Visual-only neighbourhood exploration | 4 image candidates; zero grounded objects asserted | 0.027 s |
| Unsupported pothole request | Rejected with 422 | 0.001 s |
| Export using a different dataset version | Rejected with 409 | < 0.001 s |

These are individual local timings, not p50/p95, public latency or a promise for another host. The OpenAI parser was not exercised; this run used the conservative parser without credentials.

Exact-plan complete exports include model observations and related vector evidence. Export with review candidates produced 421 total GeoJSON features for the roof query and 16 for the pool query. Feature totals include evidence/reference geometry and are not additional matched objects. Radius-only refinement to 25 metres preserved the concept and other plan fields.

## Deterministic and browser checks

**54 tests passed** in the Python 3.12 local environment. The tests cover pixel transforms, nodata/RGB preservation, edge-aligned tiling, date-aware deduplication, input integrity, typed query semantics, reference attributes, failure/unknown precedence, exact-plan pagination, version guards and complete exports. They use declared test fixtures and model substitutes where appropriate; those results do not replace real inference or visual evaluation.

JavaScript syntax was checked. The local browser rendered genuine aerial imagery and OSM outlines. A pool proximity search displayed its results and source evidence, including the original image box, raw score, measured distance, 2022 imagery date and 2026 vector snapshot. Image search displayed four explicitly unverified image candidates. Native attachment download completed through `GET /api/export`, and the downloaded GeoJSON was checked separately from the API receipt. The initial Blob download attempt timed out; it was replaced with the native attachment route. No browser console errors were observed in these checks.

## Outstanding acceptance gates

- Independently review detections and label positive/negative scenes before publishing precision, recall or compound-query accuracy. The observed pool output misses visibly present pools; the cutoff has not been tuned or calibrated.
- Expand beyond this tiny development scene and establish held-out geographic evaluation.
- Add supported solar/road/forest concepts only after actual observability and grounding checks. Commercial-use and open-drain queries remain unsupported by this sample's available evidence.
- Check registration at independent landmarks and estimate positional allowances before treating sensitivity values as physical bounds.
- Measure host memory, cold starts and latency distributions on the intended deployment.
- Verify a public URL, restart persistence and a fresh public-browser search/export flow. The current working URL is local only.
