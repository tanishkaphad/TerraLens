# GEOAI backend implementation

The revised workflow is implemented through a dynamic source catalog, validated entity/evidence plans, durable work units and saved result revisions. The primary entry point is `POST /api/search`. It returns a job identifier and URLs for progress, result pages, the full map and export.

The backend accepts new registered areas, layer schemas and visual descriptions without adding city branches, object whitelists or prepared question templates. Operational settings control worker count and batch size; they do not cap the result set or silently change the requested area.

## Start

From `TerraLens`:

```bash
pipeline/.venv/bin/python pipeline/scripts/run_search.py --port 8020
```

API documentation: `http://127.0.0.1:8020/docs`. Catalog: `http://127.0.0.1:8020/api/catalog`.

The launcher loads server configuration without selecting a city or requiring an imagery model for vector-only searches. The existing sample launcher and older routes remain available for compatibility. The former synchronous `/api/search` route moved to `/api/legacy/search`; the existing frontend still uses its earlier discovery workflow. Connecting that frontend to the new job API is separate work.

## Source registration

`BUNDLE_CATALOG_ROOTS` contains filesystem roots separated by the operating system path separator. The default root is `pipeline/data`. `BUNDLE_DIR` can add one explicitly configured source. Catalog reads discover additions and invalidate changed sources by file identity, even when a declared version has not changed.

A vector-only source needs a `manifest.json` and its actual vector files. For example, the following schema represents an administrator-added source; its names are illustrative configuration values, not built-in categories:

```json
{
  "dataset_id": "registered-area",
  "dataset_version": "source-version",
  "area_name": "Name from the source",
  "vector_layers": {
    "assets": {
      "file": "assets.geojson",
      "fields": ["source_id", "asset_type"],
      "source": "Source attribution",
      "extraction_date": "2026-10-09",
      "reference_coverage_complete": false
    }
  }
}
```

Prepared imagery bundles retain their rasters/windows, embeddings, detections, transforms and model metadata. The catalog reports actual fields, observed values, source counts, null/empty geometry omissions, dates, footprints and capabilities. It preserves valid vectors when an independent image index is unavailable. A layer's feature footprint does not certify geographic completeness.

`SEARCH_OSM_CATALOG` optionally points to a server-owned JSON file declaring OSM layers with fields, geometry types and tag selectors. This adapter contains no fixed facility or infrastructure categories. Explicit polygon scopes are partitioned; reference extraction includes requested proximity margins. Provider timeouts, missing geometry and database snapshot dates are retained. OSM completeness and consistent snapshots across partitions are not assumed. This adapter has deterministic mocked tests; a live run of the new partitioned adapter has not yet been recorded.

## Search API

| Endpoint | Behavior |
| --- | --- |
| `GET /api/catalog` | Sources, schemas, installed capabilities, policies and configured interpreter status |
| `GET /api/search/schema` | Structured builder's plan schema |
| `POST /api/search/plan` | Interpret/validate a question or structured plan without starting execution |
| `POST /api/search` or `/api/search/runs` | Save a plan and start background execution; HTTP 202 |
| `GET /api/search/runs` | Saved runs, optionally filtered by status |
| `GET /api/search/runs/{id}` | Progress, groups, coverage, saved plan and aggregate output |
| `GET /api/search/runs/{id}/units` | Per-unit status/checkpoints/coverage; worker lease tokens are hidden |
| `GET /api/search/runs/{id}/results` | Revision-bound cursor pages of persisted records |
| `GET /api/search/runs/{id}/map` | Complete saved map collection, independent of the list page |
| `GET /api/search/runs/{id}/export` | Complete saved GeoJSON; no provider or model rerun |
| `POST /api/search/runs/{id}/pause` | Revoke active work leases and preserve evidence/checkpoints |
| `POST /api/search/runs/{id}/resume` | Continue unfinished/failed work using the saved plan and source versions |
| `POST /api/search/runs/{id}/cancel` | End the run while retaining its saved evidence |
| `GET /api/catalog/places` | Provider-ranked identity suggestions with incomplete-enumeration notice |
| `GET /api/catalog/places/{type}/{id}/boundary` | Actual supported OSM polygon boundary; no substituted search circle |

Natural-language requests contain `query`, an explicit `scope` when selected, and optionally `parent_run_id`. Your teammate owns Gemini interpretation. The backend accepts their structured plans through the same API; the complete contract is at `/api/search/schema`. The optional language adapter is disabled by default (`SEARCH_PARSER_ENABLED=0`) and makes no provider request unless explicitly enabled. A disabled or unavailable adapter returns a clear planning response and preserves structured-plan execution. No fixed sentence fallback is used.

```json
{
  "query": "The user's complete geographic question",
  "scope": {
    "mode": "all_loaded",
    "dataset_ids": ["ID returned by the catalog"]
  }
}
```

Use `scope.mode=geometry` plus the selected GeoJSON polygon for a geographic area. Named unresolved areas require boundary resolution or clarification. `all_loaded` means explicitly loaded local evidence and does not automatically trigger unbounded external acquisition.

A structured request contains `plan` instead of `query`, using the schema endpoint. Entities specify runtime dataset/layer IDs or an open visual description. Clauses specify attribute, visual, spatial or temporal requirements; the Boolean expression preserves AND/OR/NOT grouping. A source attribute that is missing remains unknown. Unsupported processors remain explicit conditions.

Follow-up questions reference the preceding run. An explicit new structured plan creates a linked run. New inference and joins use the full relevant input catalog rather than filtering only previous matches.

## Worker and evidence pipeline

1. Discover sources and validate the complete request against actual schemas.
2. Freeze plan, source identities, policies, scope and work dependencies.
3. Enumerate local vectors or partition configured external extraction. Ground every selected image window for the requested concept; rank scene queries over the full selected scene set.
4. Commit each unit's evidence and provisional candidates durably. Missing processors, unsuitable dates and geographic gaps are recorded separately.
5. Evaluate the entity graph after its evidence dependencies finish. Spatial indexing conservatively removes impossible pairs; every qualifying or unresolved witness is retained.
6. Reclassify stable candidate IDs using pass/fail/unknown decisions. Save the new revision without rewriting earlier snapshots.
7. Serve cursor pages, the complete map and GeoJSON from the same saved revision.

SQLite transactional claims, lease tokens and heartbeats prevent duplicate work commitments. Checkpoints, results and failed units survive restarts. Expired orphan claims become resumable; valid workers retain their claims. Resource budgets pause work and expose remaining units. Source changes stop affected units rather than mixing new evidence into the old run.

The processor uses the installed open model and pins its actual files, prompt, image pixels, transform and screening policy in the cache identity. It scans uncached descriptions directly without requiring a high similarity ranking or using the old four-tile/200-box preview caps. Empty detections do not prove absence. Model findings and scene similarity remain distinguishable.

Optional association policies are declared in source manifests or `SEARCH_ASSOCIATION_POLICIES`. The installed `unique_overlap` processor checks exact feature identities, accepted geometry roles, overlap, competing footprints and configured date/registration requirements. Missing policies or ambiguous association remain unknown. Association is labelled as a geometric heuristic with its policy's validation state.

Counts and grouping use the complete saved entity set. Attribute groups preserve missing values. Geographic groups use an explicitly reported representative-point rule and preserve unassigned/ambiguous boundaries. Additional statistical measurements, temporal change and network analysis need suitable registered processor evidence; they are not fabricated from a count or single RGB image.

## Result semantics

Visible groups are `matches`, `needs_review`, `visual_candidates`, `related_results` and `coverage_gaps`. `excluded` and internal `evidence` records can be requested explicitly. A model-based match means its conditions pass under the recorded evidence policy; it is not a human-certified object.

`revision`, repeated `group`, `cursor` and `page_size` select pages. There is no semantic result cap. `output.sort_by` is a literal feature-property key; `-` selects descending order. Missing/non-scalar values sort last and stable IDs break ties. Cursors pin the run, revision, groups and sort policy.

`execution_complete`, `has_coverage_gaps`, `data_coverage_complete`, `complete` and `partial` remain distinct. Finished work cannot establish perfect model recall or a complete real-world inventory. Requested footprints and per-unit coverage geometries remain inspectable. Image/map epochs, assumed positional uncertainty, coarse detector boxes and unavailable field evidence remain visible in decisions and exports.

Map and exports include every selected saved result. Per-result properties contain all relevant witness features/geometry and observation provenance. The default map/export omits internal evidence and evaluated exclusions, which remain available through explicit group selection. GeoJSON coordinates are validated WGS84 longitude/latitude.

## Operational configuration

| Setting | Purpose |
| --- | --- |
| `SEARCH_DB` | Persistent search database; default `data/search.sqlite3` |
| `SEARCH_WORKERS` | Concurrent work-unit threads; default 4 |
| `SEARCH_MAX_PENDING` | Bounded queued-run admission; default 32 |
| `SEARCH_VECTOR_CHUNK_SIZE` | Vector work-unit size; default 1000 |
| `SEARCH_UNIT_BUDGET` | Optional work-unit budget before pausing |
| `SEARCH_TIME_BUDGET_SECONDS` | Optional runtime budget before pausing |
| `SEARCH_PARTITION_SIZE_M` | Nominal connected-vector partition size; preserves the original polygon |
| `SEARCH_GROUNDING_CACHE_DIR` | Persistent prompt/source/model-specific inference cache |
| `SEARCH_ASSOCIATION_POLICIES` | Optional server-owned JSON policy catalog |
| `SEARCH_PARSER_ENABLED` | Optional language bridge; default 0 for teammate integration |
| `SEARCH_PARSER_SAMPLE_VALUES` | Metadata sample size for interpretation; execution sees all records |
| `SEARCH_PARSER_MODEL`, `SEARCH_PARSER_BASE_URL`, `SEARCH_PARSER_API_KEY` | Optional structured-interpreter overrides |

Gemini integration is deferred to the teammate. They can submit structured plans directly, or enable the optional adapter with `SEARCH_PARSER_ENABLED=1`. Existing provider credentials/settings are unchanged. Secrets remain on the backend. `ENCODER_CHECKPOINT` must match the prepared image embeddings for scene ranking. Grounding can process every image window without loading the text-retrieval encoder.

## Recorded validation

- All 406 tests passed on 9 October 2026. The deterministic suite covers dynamic sources and concepts, Boolean/negation semantics, shared witnesses, historical unknowns, conservative spatial indexing, association ambiguity, source mutation, concurrent claims, lease recovery, sorting, more than 100 results, complete pagination/map/export and grouping.
- A real stored-data spatial run found **484 mapped building matches**. A seven-item page, complete map and saved export retained the same complete subject set. **3,839 witness distances** passed an independent calculation in EPSG:32610.
- A separate actual installed-model scan processed **all four imagery windows** and produced **18 model findings**, with complete saved evidence and no coverage gaps reported for the registered windows. These are model outputs, not independently annotated accuracy results.
- Evidence receipts and GeoJSON live under `pipeline/data/validation/search-backend/`. They are local generated artifacts and are not committed source fixtures.

Run the suite from `pipeline` with `.venv/bin/python -m pytest -q`. A checked-in HTTP verifier compares every result page with the complete saved map/export:

```bash
.venv/bin/python scripts/verify_search_backend.py --dataset CATALOG_ID --layer LAYER_ID
```

Add `--query "..."` to exercise the configured live interpreter. Gemini was excluded from this delivery at the user's request. An earlier diagnostic received HTTP 429 from the configured provider; no successful live language-parser claim is made. HTTP receipts and deterministic checks are assessed separately. Public deployment, independent visual accuracy and uninterrupted external-provider availability remain unverified.
