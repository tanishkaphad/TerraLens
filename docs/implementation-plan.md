# GeoVerify-X: End-to-End Implementation & Open-Source Deployment Plan

This document outlines the complete phased roadmap to resolve all remaining, hardcoded, and fragile components in GeoVerify-X, integrate **Groq** (hosting open-weights LLMs such as Llama 3.3 70B / Qwen 2.5 32B) for a **GIS-only** chatbot, and deliver an open-source, production-ready deployment.

---

## Architecture Summary & Current Gaps

```
┌────────────────────────────────────────────────────────────────────────┐
│                        GeoVerify-X (TerraLens)                        │
├────────────────────────────────────────────────────────────────────────┤
│                                                                        │
│   Frontend (Web UI)                                                    │
│   ├── Explorer / MapLibre (index.html, app.js, discovery.css)          │
│   └── Conversation Tab [CURRENTLY HIDDEN & DISABLED IN HTML]          │
│                                │                                       │
│                                ▼                                       │
│   FastAPI Service Layer                                                │
│   ├── Legacy Routes: /api/discover, /api/query, /api/capabilities      │
│   ├── Durable Search Backend: /api/search, /api/search/runs/*          │
│   └── Chat / Jobs Queue: /api/chat, /api/jobs/{id}                     │
│                                │                                       │
│                                ▼                                       │
│   LLM Planning & Parsing Engine                                        │
│   ├── assistant.py  (Supports only Gemini, OpenAI, xAI -> Needs GROQ) │
│   ├── planner.py    (Hardcoded OpenAI client -> Needs unified GROQ)    │
│   └── search_plan.py (Structured parser with SEARCH_PARSER_ENABLED=1)  │
│                                │                                       │
│                                ▼                                       │
│   Evidence & Spatial Execution Engine                                  │
│   ├── Local Imagery AI: SkyCLIP (ViT-B/32) + OWLv2 Live Grounding     │
│   ├── OSM Vectors: Overpass QL [5 HARDCODED LAYERS, 2KM CIRCLE CAP]    │
│   ├── Places: Photon Geocoder (Komoot)                                │
│   └── GIS Operators: Shapely, GeoPandas, PyProj (Metric EPSG/UTM)     │
│                                                                        │
└────────────────────────────────────────────────────────────────────────┘
```

---

## Phase 1: Groq Open-Source LLM Integration

### Objective
Integrate Groq API (`https://api.groq.com/openai/v1`) using high-speed open-source models (default: `llama-3.3-70b-versatile`) across the assistant, search planner, and legacy query planner.

### Detailed Steps:
1. **Extend Provider Registry in [`pipeline/terralens/assistant.py`](file:///c:/Users/tanee/TerraLens/pipeline/terralens/assistant.py)**:
   - Add `"groq"` entry to `PROVIDERS`:
     ```python
     "groq": {
         "key": "GROQ_API_KEY",
         "model": "GROQ_MODEL",
         "default_model": "llama-3.3-70b-versatile",
         "base_url": "https://api.groq.com/openai/v1",
     }
     ```
   - Update `status()` function to recognize `groq` and report its configuration status.
2. **Schema Sanitization for Groq Function Calling**:
   - Groq supports tool/function calling via the OpenAI-compatible endpoint, but does not support OpenAI's `strict: True` parameter and can choke on unresolved `$defs`/`$ref` in recursive schemas.
   - Generalize `_gemini_schema(schema)` into a provider-agnostic `_sanitize_tool_schema(schema, provider)` that:
     - Inlines and dereferences all `$defs` and `$ref`.
     - Flattens nullable `anyOf` constructs into `{... "nullable": True}`.
     - Strips `strict: True` and unsupported keywords when `provider in {"gemini", "groq"}`.
3. **Unify `planner.py` with Configured LLM Provider**:
   - Refactor [`planner.py:L706-L708`](file:///c:/Users/tanee/TerraLens/pipeline/terralens/planner.py#L706-L708) to stop hardcoding `OPENAI_API_KEY` and instead use the configured provider from `assistant.status()`, allowing Groq to generate query plans seamlessly.
4. **Enable Search Plan Parsing with Groq**:
   - Ensure [`search_plan.py:L413`](file:///c:/Users/tanee/TerraLens/pipeline/terralens/search_plan.py#L413) respects Groq when `SEARCH_PARSER_ENABLED=1`.
5. **Environment Configuration**:
   - Update `pipeline/.env.example` with:
     ```env
     LLM_PROVIDER=groq
     GROQ_API_KEY=gsk_...
     GROQ_MODEL=llama-3.3-70b-versatile
     ```

---

## Phase 2: GIS-Only Chatbot Guardrails & Strict Geospatial Scope

### Objective
Restrict the chatbot strictly to GIS domain workflows, geographic evidence discovery, spatial relationship verification, and data explanation, refusing generic non-geospatial requests.

### Detailed Steps:
1. **Update System Prompt in [`assistant.py`](file:///c:/Users/tanee/TerraLens/pipeline/terralens/assistant.py)**:
   - Explicitly instruct the model:
     - You are **GeoVerify-X**, a strict geospatial intelligence assistant.
     - **Scope Boundary**: Answer ONLY queries concerning geography, spatial topology (distance, intersection, proximity), satellite/aerial imagery, map layers, coordinates, and GIS data evidence.
     - **Out-of-Scope Requests**: If the user asks general chat, coding, creative writing, or non-GIS questions, reject politely:
       *"GeoVerify-X is dedicated to geospatial evidence discovery. Please ask a geographic question involving places, mapped infrastructure, or satellite imagery."*
     - Never invent coordinates, bounding boxes, or detection counts.
2. **Support Two-Mode Responses**:
   - **Mode A (Execution Workflow)**: Structured plan with operations (`search`, `spatial_filter`, `aggregate`, `coverage`).
   - **Mode B (Geospatial Explanation / Direct GIS Q&A)**:
     - When the user asks about dataset provenance (e.g., *"What is NAIP imagery?"*, *"What years does SpaceNet cover?"*, *"Why were no buildings found within 20m of roads?"*), respond directly with evidence citations rather than forcing a failing multi-step query.
3. **Resilient Intent Verification & Clarification**:
   - In [`workflow.py`](file:///c:/Users/tanee/TerraLens/pipeline/terralens/workflow.py) and [`intent.py`](file:///c:/Users/tanee/TerraLens/pipeline/terralens/intent.py), gracefully return `needs_input` with clear prompt hints when a location needs a metric distance or specific city, rather than throwing uncaught `WorkflowError`.

---

## Phase 3: Unhide & Activate the Chatbot in the Frontend

### Objective
Expose the conversation interface in the UI, enabling users to seamlessly toggle between the standard Search mode and the GIS Chatbot mode.

### Detailed Steps:
1. **Unhide Mode Tabs in [`index.html:L31`](file:///c:/Users/tanee/TerraLens/pipeline/terralens/static/index.html#L31)**:
   - Remove `hidden aria-hidden="true"` on `<div class="mode-tabs">`.
   - Remove `disabled` attribute from `<button id="conversationMode">`.
2. **Wire State & Status in [`app.js`](file:///c:/Users/tanee/TerraLens/pipeline/terralens/static/app.js)**:
   - In `loadCapabilities()`, inspect `data.conversation`:
     - If configured (Groq API key present), enable the Conversation button with tooltip: *"Ask geographic questions and follow up through Groq"*.
     - Update status pill to display active provider: *"Groq (Llama 3.3) Ready"*.
3. **Enhance Conversation Thread Rendering**:
   - Ensure cards for `map`, `table`, and `chart` render cleanly.
   - Add a "View on Map" shortcut that immediately highlights generated features on MapLibre.
   - Display a "Download Evidence GeoJSON" button for every completed assistant turn.
4. **Style Polish**:
   - Adjust `app.css` and `discovery.css` for smooth message bubbles, loading spinners, and clear markdown rendering for GIS explanations.

---

## Phase 4: Bridge Frontend to the Durable Search Backend

### Objective
Migrate the frontend from legacy synchronous `/api/discover` to the new persistent, catalog-driven `/api/search` pipeline.

### Detailed Steps:
1. **Catalog Sync**:
   - Update UI initialization to fetch `/api/catalog` (supported sources, installed capabilities, and vector layer schemas) alongside `/api/datasets`.
2. **Run Submission & Polling**:
   - In `app.js`, update search submission to post to `POST /api/search`.
   - Poll `/api/search/runs/{id}` to display real-time work unit progress and partial candidate streams.
3. **Paginated Map & GeoJSON Rendering**:
   - Fetch complete map collections via `/api/search/runs/{id}/map`.
   - Wire export button directly to `/api/search/runs/{id}/export`.
   - Support cursor-based pagination (`/api/search/runs/{id}/results?cursor=...`).

---

## Phase 5: Dynamic OSM Layers & Actual Boundary Polygons

### Objective
Remove hardcoded 5-layer limits in `world_data.py` and replace circular buffer approximations with real administrative polygon boundaries.

### Detailed Steps:
1. **Dynamic Overpass Tag Resolution**:
   - Refactor [`world_data.py:L272-L286`](file:///c:/Users/tanee/TerraLens/pipeline/terralens/world_data.py#L272-L286) to support configurable OSM tag selectors rather than the 5 hardcoded aliases (`buildings`, `roads`, `colleges`, `rivers`, `forests`).
   - Allow user queries like *"Find hospitals within 500m of fire stations in Pune"* by mapping `hospital` -> `amenity=hospital` and `fire station` -> `amenity=fire_station`.
2. **Real Administrative Boundary Geometry**:
   - Replace the `center_metric.buffer(2000)` circular search in `world_data.py` with actual OSM boundary polygons retrieved via [`search_sources.py:place_boundary`](file:///c:/Users/tanee/TerraLens/pipeline/terralens/search_sources.py#L16).
   - If a boundary is unavailable, warn the user explicitly that a local radial sample is being evaluated.

---

## Phase 6: Local Data Ingestion, Verification Suite & Production Deployment

### Objective
Verify end-to-end functionality locally and deploy containerized GeoVerify-X to cloud platforms (Docker / Render / VPS).

### Detailed Steps:
1. **Local Data & Checkpoints Preparation**:
   - Run preparation scripts:
     ```bash
     python pipeline/scripts/fetch_assets.py
     python pipeline/scripts/fetch_demo_data.py
     python pipeline/scripts/prepare_demo.py
     ```
2. **Automated Test Validation**:
   - Run the 400+ unit test suite:
     ```bash
     pytest pipeline/tests -v
     ```
   - Execute the verification script:
     ```bash
     python pipeline/scripts/verify_search_backend.py
     ```
3. **Containerization & Deployment Configuration**:
   - Update [`Dockerfile`](file:///c:/Users/tanee/TerraLens/Dockerfile) to ensure dependencies for Groq (`openai>=1.50.0`, `requests`, `shapely`, `geopandas`, `pyproj`, `fastapi`, `uvicorn`) are locked.
   - Update [`render.yaml`](file:///c:/Users/tanee/TerraLens/render.yaml):
     ```yaml
     envVars:
       - key: LLM_PROVIDER
         value: groq
       - key: GROQ_MODEL
         value: llama-3.3-70b-versatile
       - key: GROQ_API_KEY
         sync: false
     ```
4. **Health Check & Production Acceptance**:
   - Verify `/health` returns `200 OK`.
   - Verify `/api/assistant/status` reports `configured: true`, `provider: "groq"`.
   - Execute end-to-end search and chat workflows in the browser.

---

## Phase Milestones Summary

| Phase | Milestone Name | Key Deliverables | Verification Metric |
| :---: | :--- | :--- | :--- |
| **1** | **Groq LLM Integration** | `assistant.py`, `planner.py`, schema dereferencing, `.env` | Successful planning call via Groq API |
| **2** | **GIS-Only Guardrails** | Strict GIS prompt, non-geospatial rejections, GIS Q&A | Rejects general chit-chat, answers GIS questions |
| **3** | **Chatbot UI Activation** | Unhide mode-tabs in `index.html`, `app.js` chat handler | Conversation tab clickable & responsive in UI |
| **4** | **Durable Search Bridge** | UI calls `POST /api/search`, run polling, cursor pages | Seamless streaming results & export from runs |
| **5** | **Dynamic Vector & Polygon** | Overpass tag mapping, OSM boundary polygons | Searches beyond 5 fixed layers; polygon extents |
| **6** | **Open-Source Deployment**| Dockerfile, render.yaml, full pytest suite | Zero failed tests, clean deployment receipt |
