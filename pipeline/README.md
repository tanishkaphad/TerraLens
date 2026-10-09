# TerraLens pipeline

An interactive aerial map and same-origin FastAPI service implementing the core visual/spatial search in [workflow.md](../workflow.md). It uses actual precomputed SkyCLIP image embeddings and OWLv2 object detections, then runs live text ranking and metric spatial checks. No live object-grounding claim is made.

## Install and acquire models

Run from `pipeline/` with Python 3.12:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
python scripts/fetch_assets.py
python scripts/convert_checkpoint.py
```

`fetch_assets.py` fetches MapLibre GL JS 5.5.0 with its licence, the official SkyCLIP ViT-B/32 top50pct checkpoint, and OWLv2 at commit `cfd3195ba4ea9592eec887ded089f4c08eff231d`. Downloads include source URLs and SHA-256 records. The SkyCLIP archive and training checkpoint together consume about 3.3 GB; conversion produces a 605 MB model-only safetensors file. OWLv2 adds about 620 MB. Model files stay under ignored `data/models/`.

The conversion accepts only the pinned official legacy checkpoint hash and narrowly allowlists its NumPy scalar metadata using `weights_only=True`. It preserves model tensors and removes training state. Both preparation and serving must use the converted **same** checkpoint; an identity mismatch disables text-image ranking.

`requirements-lock.txt` records the measured macOS/Python 3.12 environment, including development tools. Other platforms need their own runtime check; the lock is not proof of Linux/GPU compatibility.

## Acquire real demo data

```bash
python scripts/fetch_demo_data.py
```

Read [dataset notes](../docs/datasets.md) before using the extract. The sample is a bounded Sacramento scene, not Pune data. The acquisition script preserves the STAC item, georeferenced RGB crop, source metadata, actual OSM elements and extraction date. OSM is queried beyond the imagery by 525 metres for proximity checks up to 500 metres. Missing OSM features remain a coverage limitation.

Prepare that exact sample with `python scripts/prepare_demo.py --device cpu`. It uses four overlapping 640-pixel windows, the pinned models and explicit roof/building associations. The output defaults to `data/demo`; use a new `--output` for another run. The more general preparation command below is for other source data.

## Prepare a frozen evidence bundle

The general preparation interface accepts georeferenced RGB rasters and real vector files. It does not download models. Explicit associations apply only to appropriate classes; a pool must not automatically become a building.

```bash
python scripts/prepare_bundle.py \
  --area-name "Your verified area" \
  --imagery /path/to/aerial.tif \
  --metadata /path/to/aerial.tif=/path/to/aerial.metadata.json \
  --vectors roads=/path/to/roads.geojson \
  --vectors buildings=/path/to/buildings.geojson \
  --vector-metadata roads=/path/to/roads.metadata.json \
  --vector-metadata buildings=/path/to/buildings.metadata.json \
  --concept building_roofs="building roofs" \
  --associate building_roofs=buildings \
  --concept swimming_pools="swimming pools" \
  --checkpoint data/models/SkyCLIP_ViT_B32_top50pct.safetensors \
  --owl-model data/models/owlv2-cfd3195ba4ea9592eec887ded089f4c08eff231d \
  --owl-revision cfd3195ba4ea9592eec887ded089f4c08eff231d \
  --tile-size 1024 --overlap 128 --max-tiles 12 --device cpu \
  --imagery-attribution "USDA NAIP / Microsoft Planetary Computer" \
  --vector-attribution "© OpenStreetMap contributors, ODbL 1.0" \
  --output data/demo
```

Use the actual paths emitted by source acquisition. Output must be new or empty. Manifest is written last and source/artifact hashes are checked. Each detection retains original pixel bounds, window transform, model prompt/revision, inference time and source date. Deduplication preserves every observation and avoids merging conflicting dates. A screening threshold is an uncalibrated model cutoff, not a correctness probability.

SkyScript JSON or primitive-only pickle sidecars can provide `box`, `time`, `center_tags` and `surrounding_tags`. Tags stay source annotations. Bounds-only imagery requires `--bounds-georeference` after independently checking north-up orientation.

Set `profile.limits.maximum_distance_m` in imagery metadata to the distance supported by the reference-data margin. Preparation takes the minimum declared limit across sources. Without a profile, the general tool's 5,000-metre cap does not establish that the source has sufficient border coverage.

## Serve and verify

```bash
python scripts/run_demo.py
# In another terminal, using the same virtual environment:
python scripts/verify_demo.py
```

Open `http://127.0.0.1:8000`, or pass `--port 8010` when that port is occupied. The launcher uses `data/phoenix-demo` when prepared, then the SpaceNet sample (`spacenet-round1`, `spacenet-grounded`, `spacenet-urban`, `spacenet-sample`), falling back to `data/demo`, and the converted SkyCLIP checkpoint. The interface can switch between registered prepared datasets. `BUNDLE_DIR` and `ENCODER_CHECKPOINT` can override the defaults. `/health` reports liveness; `/ready` loads and probes the matching encoder and distinguishes full capability from degraded cached spatial search.

The verification script records real API execution, counts, latency and complete exports under `data/validation/`. It is an integration check, **not** an annotated visual-accuracy benchmark. Run `pytest -q` for deterministic geometry/semantic tests.

Optional backend-only environment variables `OPENAI_API_KEY` and `OPENAI_MODEL` enable schema-constrained language interpretation. Without them, supported templates include `Find swimming pools within 100 metres of roads`, `Find building roofs intersecting roads`, and `Make the radius 25 metres`. Unsupported clauses are rejected; they are never silently dropped.

## API and evidence

| Endpoint | Purpose |
| --- | --- |
| `GET /health`, `GET /ready` | Process liveness and actual model/data capability |
| `GET /api/capabilities` | Supported classes, fields, predicates, limits and provenance |
| `GET /api/map` | Bounded imagery quadrilaterals and real vector layers |
| `POST /api/query` | Query plus optional previous plan, mode and pagination |
| `POST /api/export`, `GET /api/export` | Complete GeoJSON using the executed plan and dataset version; GET supports native browser attachments |
| `GET /api/assets/{tile_id}` | Locally served source preview |

`catalog_scan` evaluates all cached detections of a supported concept before paging. `explore` ranks a bounded set of images; a visual-only query returns `tile_candidates` with no object-location claim. Query results separate matches from candidates needing review. A definite failed clause excludes a record even if another clause is unknown.

Distances use a local metric projection and the declared geometry role: a uniquely associated footprint or detector envelope. Road data represents mapped centreline geometry. Neither a detector box nor an OSM centreline is a precise physical boundary. Missing acquisition dates, attributes, associations or map data stay explicit.

GeoJSON export is complete for the selected execution mode and includes observation geometry, reference features and provenance; it is not restricted to the visible page. Including review candidates is an explicit UI option. No hosted deployment or independent accuracy result should be inferred from local checks.
