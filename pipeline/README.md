# TerraLens pipeline

This is the runnable search pipeline described in [`../workflow.md`](../workflow.md). It serves a same-origin search page and API. The repository contains no imagery, vector extracts, model weights or fabricated search results. Create a real data bundle before the service reports ready.

## Build an evidence bundle

Use georeferenced RGB GeoTIFFs and vector files you are allowed to process. Vector layers can be GeoPackage, GeoJSON or another format supported by GeoPandas. Supply a separate `--vectors` argument for each layer. Use `name=path#layer` for a named GeoPackage layer.

Download or provide a local OpenCLIP checkpoint compatible with the model architecture. The image embeddings and online text encoder must use the exact same checkpoint. The object detector downloads its weights from Hugging Face on first use; use `--owl-revision` to pin a commit.

From this directory:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/prepare_bundle.py \
  --area-name "Your verified area" \
  --imagery /path/to/georeferenced-aerial.tif \
  --vectors roads=/path/to/osm.gpkg#roads \
  --vectors buildings=/path/to/osm.gpkg#buildings \
  --concept solar_panels="solar panels" \
  --checkpoint /path/to/openclip-checkpoint.bin \
  --output data/demo
```

This offline step writes a versioned manifest, WGS84 tile and detection geometry, preview crops, normalized image embeddings and vector files under `data/demo`. The data directory is git-ignored because imagery and model outputs need their own size and redistribution review.

## Run the service

```bash
export BUNDLE_DIR=data/demo
export ENCODER_CHECKPOINT=/path/to/openclip-checkpoint.bin
# Optional natural-language parser. Keep these values on the backend.
export OPENAI_API_KEY=your-backend-secret
export OPENAI_MODEL=your-enabled-model
uvicorn terralens.api:app --reload
```

Open `http://127.0.0.1:8000`. `/health` reports process liveness; `/ready` checks that the data bundle can be loaded. If `OPENAI_API_KEY` and `OPENAI_MODEL` are absent, a conservative parser supports a concept already listed in the bundle plus a named vector layer and an explicit distance or intersection. A radius-only follow-up edits the previous validated plan. Ambiguous or unavailable attributes are rejected.

Without `ENCODER_CHECKPOINT`, catalog scans still run over the precomputed detections but text-image similarity ranking is disabled; exploration mode requires the matching local checkpoint. This distinction appears in the response.

## API

- `GET /health`
- `GET /ready`
- `GET /api/capabilities`
- `POST /api/query` with `{"query":"Find solar panels within 200 metres of roads"}`
- `GET /api/assets/{tile_id}` for the original tile preview

Query results are limited to the frozen detections and vector features in the configured bundle. They are not a completeness guarantee about the underlying municipality.
