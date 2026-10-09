# GeoVerify-X — GEOAI-05 implementation workflow

Updated 9 October 2026. Work stopped at the user's request. This file records the current implementation and the remaining integration work; it does not claim that unfinished components are deployed.

## Objective

Build general-purpose, evidence-backed geospatial intelligence. Rooftops, solar panels, drainage and roads are examples, not a fixed query catalog. Users should describe geographic objects, visible conditions, attributes and spatial relationships in natural language. The engine must preserve all requested conditions or explain the missing evidence, tools or clarification.

OpenAI is the main planned conversational layer. SkyCLIP, OWLv2 and deterministic GIS operations establish the underlying evidence. Language models cannot supply missing geographic facts.

## Dataset decision

The organizers supplied only the SkyScript and SpaceNet reference links, not a separate municipal dataset.

1. **Primary implemented sample: SpaceNet 1 Rio.** Three georeferenced training images (`img1026`, `img1071`, `img1027`) and 457 building annotations are prepared locally. The grounded bundle contains 19 deduplicated model candidates from 25 raw detections. Annotations and detections are distinct evidence types. These counts do not establish detector accuracy.
2. **SkyScript / SkyCLIP:** use the official SkyCLIP ViT-B/32 checkpoint for text-image retrieval. This does not mean the complete SkyScript dataset has been downloaded or evaluated.
3. **Additional implemented sample: Sacramento NAIP plus OSM.** Four overlapping image tiles with real road/building vectors provide another geographic test area. The imagery and OSM extraction have different dates, which remain visible.
4. External imagery providers, Pune coverage, multispectral processing and temporal datasets are optional extensions. They are not prerequisites replacing the supplied references, and no working Pune dataset is claimed.

SpaceNet acquisition dates are unknown in the downloaded metadata. Its sampled training annotations are partial coverage, without a municipal reference halo. Neither sample supports worldwide or current-condition claims.

## Selected pipeline

```text
Question or follow-up
  -> OpenAI intent and workflow proposal
  -> registered dataset and tool selection
  -> typed workflow validation + explicit distance checks
  -> separate semantic review of the proposed conditions
  -> bounded background execution
       -> mapped vector / attribute search
       -> SkyCLIP scene retrieval
       -> cached or live OWLv2 object grounding
       -> pixel-to-geographic transformation
       -> object association and metric spatial joins
       -> optional count, area or length aggregation
  -> per-condition evidence verification
  -> supported / uncertain / excluded records
  -> map, evidence, tables/charts and run-bound GeoJSON export
```

Scene similarity is not an object detection. A detection box is not an exact object boundary. Commercial use, legality, temporal change and absence require appropriate independent evidence. A successful tool call is not proof that the user's intent was preserved.

## Current progress

Last pushed implementation baseline: `bf4c215459bc85f700fe99c32bbe08b200e4b12e` on `main`. The additions below remain local and uncommitted at this stop point.

| Component | Status at stop |
| --- | --- |
| FastAPI and MapLibre application | Existing same-origin application, map, evidence assets, dataset switching and complete GeoJSON export exercised locally |
| Reference-first data loading | SpaceNet sample is the default; additional NAIP/OSM sample remains available |
| Existing search engine | Vector attributes, cached detections, SkyCLIP retrieval, bounded live OWLv2 grounding and multiple spatial clauses implemented |
| Existing baseline verification | 115 tests passed before this round of changes; earlier real model, local API, browser and export checks completed |
| `verification.py` | New immutable evidence receipts, conservative geometry intervals, witness tracking and uncertain-result adjustment implemented; 22 focused tests and 42 combined verification/engine tests reported passing |
| `workflow.py` | Initial typed workflow executor added for search, spatial filtering, aggregation and coverage inspection; correctness review and dedicated workflow tests unfinished |
| `assistant.py` | OpenAI Responses planning, strict schema, numeric checks, semantic review and evidence-derived response assembly added; not yet connected to API routes or tested against OpenAI |
| `jobs.py` | Background-job and conversation persistence implementation started; integration and final validation unfinished |
| Conversation UI | Local UI changes for assistant/expert views, workflow steps, polling, artifacts and receipts added; full browser validation against new endpoints unfinished |
| Server configuration | `.env.example` and server-only dotenv loader added; startup integration unfinished |
| Deployment | Public deployment and durable hosting remain unverified; no new public URL is claimed |

**OpenAI blocker:** no usable OpenAI API key was available in the project or environment. Configuration code cannot create an account credential. Live conversation testing remains blocked until a valid server-side key is supplied. Never place it in source control or frontend code.

## Differentiation informed by research

Natural-language GIS orchestration, remote-sensing conversation and visual grounding already exist. Do not claim these combinations alone as novel.

[GriidAI](https://www.griidai.com/index.html) publishes conversational vector/raster workflows, joins, buffers and imagery analysis. Its public material does not establish the absence of stronger internal verification. Treat per-result evidence fingerprints, calibrated uncertainty, clause-preservation benchmarks and independently measured object-level geographic accuracy as **not demonstrated in the inspected public material**, not verified product weaknesses.

The comparison also considered [OpenGeoAgent](https://github.com/opengeos/GeoAgent), [GeoGPT](https://doi.org/10.1016/j.jag.2024.103976), [GeoChat](https://github.com/mbzuai-oryx/GeoChat), [EarthGPT](https://github.com/wivizhang/EarthGPT), [ASKTERRA](https://askterra.io/help/) and [UnivEARTH](https://openreview.net/forum?id=nkLCUM5ROh). This is a targeted review, not proof of scientific priority.

Implement and measure these distinctions:

- **Condition preservation:** every requested clause maps to execution steps; validate datasets, attributes, dependencies, units and available tools before execution. A model-based semantic review adds scrutiny but is not a correctness proof.
- **Conservative geographic verification:** use distance bounds for uncertain detection envelopes; distinguish exact vector geometry from visual candidate boxes. Preserve unknown registration error rather than inventing a numeric allowance.
- **Traceable evidence:** bind source geometry, model identity, prompts, transformations and clause witnesses to a content receipt. A hash detects content changes; it is neither a digital signature nor an accuracy certificate.
- **Honest coverage:** distinguish missing evidence, partial reference coverage and uncertain candidates from supported findings. No detection does not prove absence.
- **Reusable workflows:** registry-based operations and typed dependencies support additional datasets, models and analyses without hardcoding a single infrastructure category.

## Resume order — not executed at this stop

1. Finish workflow correctness review and meaningful tests, especially envelope joins, uncertain reference objects, self-matches, geographic scope, aggregation and dropped conditions.
2. Connect assistant, workflow and job modules to FastAPI; add status, chat, job polling and saved-run export endpoints. Load server configuration during startup.
3. Complete durable conversation/job handling, bounded concurrency, input/resource limits and request quotas. Keep credentials server-side and avoid arbitrary generated code or SQL execution.
4. Integrate and test UI polling, follow-ups, dataset changes, evidence display, error handling and exact-run exports.
5. Run the full suite and real SpaceNet/NAIP workflows. Separately test live OpenAI planning once credentials exist; mocked tests are not live API validation.
6. Measure model memory and latency, then provision hosting with the prepared evidence and pinned model files. Verify the public URL, restart persistence and complete deployed workflow.
7. Commit and push the reviewed implementation only when requested work resumes. No commit, push or deployment was performed as part of this stop-and-update request.

## Evaluation boundaries

Measure intent preservation, unsupported-request rejection, spatial boundary decisions, evidence/export consistency, runtime and resource use. Detector precision/recall and calibration require independently labelled evaluation data; software tests and a functioning map do not establish them. General-purpose architecture does not imply that every geographic question is answerable with the current two small samples.
