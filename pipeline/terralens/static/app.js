"use strict";

const $ = id => document.getElementById(id);
const state = {
  map: null, mapData: null, capabilities: null, datasets: [], datasetId: null,
  previousPlan: null, groundingRunId: null, groundingThreshold: 0.1,
  last: null, request: null, imagery: [], vectorIds: [], selected: null,
  selectedControl: null, taskCatalog: [], loadSequence: 0, searchSequence: 0,
};

const MODE_UI = {
  catalog_scan: {
    help: "Search cached detections and supplied map records with spatial conditions.",
    placeholder: "Find swimming pools within 100 metres of roads",
    label: "Geographic evidence question", button: "Search prepared evidence",
    examples: "Try a prepared-evidence query",
  },
  explore: {
    help: "Rank whole aerial images by scene similarity. Results do not identify or locate individual objects.",
    placeholder: "Find tree-lined residential streets",
    label: "Describe the scene to discover", button: "Discover images",
    examples: "Try an image-discovery query",
  },
  ground: {
    help: "Run a fresh object detector on a small number of top-ranked image tiles. This can take longer.",
    placeholder: "Find trees within 100 metres of roads",
    label: "Describe the objects to detect", button: "Detect in selected tiles",
    examples: "Try an on-demand detection query",
  },
};

const FALLBACK_TASKS = [
  {query: "Find buildings with building=house within 100 metres of roads with highway=residential", mode: "catalog_scan", evidence_kind: "mapped_feature"},
  {query: "Find swimming pools within 100 metres of roads", mode: "catalog_scan", evidence_kind: "prepared_model_detection"},
  {query: "Find tree-lined residential streets", mode: "explore", evidence_kind: "image_tile"},
  {query: "Find large parking lots beside commercial buildings", mode: "explore", evidence_kind: "image_tile"},
  {query: "Find trees within 100 metres of roads", mode: "ground", evidence_kind: "live_model_detection"},
];

const fc = features => ({type: "FeatureCollection", features: features || []});
const text = (tag, value, className) => {
  const element = document.createElement(tag);
  element.textContent = value ?? "Unknown";
  if (className) element.className = className;
  return element;
};
const nice = value => String(value || "Unknown").replaceAll("_", " ");
const number = value => Number.isFinite(value) ? value.toFixed(3) : "Not available";
const currentMode = () => document.querySelector('input[name="mode"]:checked')?.value || "catalog_scan";

function status(message, error = false) {
  $("status").textContent = message;
  $("status").className = "status" + (error ? " error" : "");
  $("status").setAttribute("role", error ? "alert" : "status");
}

async function request(url, body) {
  const response = await fetch(url, body ? {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(body)} : {});
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    const detail = data.detail;
    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail || "Request failed"));
  }
  return data;
}

function datasetUrl(path) {
  if (!state.datasetId) return path;
  return `${path}${path.includes("?") ? "&" : "?"}dataset_id=${encodeURIComponent(state.datasetId)}`;
}

function boundsOf(features) {
  const points = [];
  function walk(value) {
    if (typeof value?.[0] === "number") points.push(value);
    else if (Array.isArray(value)) value.forEach(walk);
  }
  features.forEach(feature => walk(feature.geometry?.coordinates));
  return points.length ? points.reduce((bounds, point) => [
    Math.min(bounds[0], point[0]), Math.min(bounds[1], point[1]),
    Math.max(bounds[2], point[0]), Math.max(bounds[3], point[1]),
  ], [180, 90, -180, -90]) : null;
}

function fit(bounds, padding = 60) {
  if (bounds && state.map) state.map.fitBounds([[bounds[0], bounds[1]], [bounds[2], bounds[3]]], {padding, maxZoom: 18, duration: 650});
}

function source(id, features) {
  state.map?.getSource(id)?.setData(fc(features));
}

function addGeometryLayers(sourceId, color, dashed = false) {
  const map = state.map;
  map.addSource(sourceId, {type: "geojson", data: fc([])});
  map.addLayer({id: sourceId + "-fill", type: "fill", source: sourceId, filter: ["==", ["geometry-type"], "Polygon"], paint: {"fill-color": color, "fill-opacity": 0.2}});
  map.addLayer({id: sourceId + "-line", type: "line", source: sourceId, paint: {"line-color": color, "line-width": 2, ...(dashed ? {"line-dasharray": [2, 2]} : {})}});
  map.addLayer({id: sourceId + "-point", type: "circle", source: sourceId, filter: ["==", ["geometry-type"], "Point"], paint: {"circle-color": color, "circle-radius": 5, "circle-stroke-width": 1, "circle-stroke-color": "#fff"}});
}

function resetLayerPicker() {
  const label = document.createElement("label");
  const input = document.createElement("input");
  input.type = "checkbox";
  input.id = "imageryToggle";
  input.checked = true;
  label.append(input, document.createTextNode(" Aerial imagery"));
  $("layers").replaceChildren(label);
}

function destroyMap() {
  if (state.map) state.map.remove();
  state.map = null;
  state.mapData = null;
  state.imagery = [];
  state.vectorIds = [];
  resetLayerPicker();
}

async function loadMap() {
  if (!window.maplibregl) throw new Error("The local map renderer is missing. Run the asset download step.");
  const data = await request(datasetUrl("/api/map?tile_limit=64&vector_limit=5000"));
  state.mapData = data;
  const attribution = Object.values(data.attribution || {}).filter(value => typeof value === "string").join(" · ");
  const map = state.map = new maplibregl.Map({
    container: "map",
    style: {version: 8, sources: {}, layers: [{id: "background", type: "background", paint: {"background-color": "#dfe5d7"}}]},
    center: [0, 0], zoom: 1, attributionControl: false,
  });
  map.addControl(new maplibregl.NavigationControl({showCompass: false}), "top-right");
  map.addControl(new maplibregl.ScaleControl({unit: "metric"}), "bottom-left");
  map.addControl(new maplibregl.AttributionControl({customAttribution: attribution, compact: true}), "bottom-right");
  await new Promise(resolve => map.on("load", resolve));

  for (const tile of data.tiles || []) {
    if (!tile.coordinates || tile.coordinates.length !== 4) continue;
    const id = "image-" + tile.tile_id;
    map.addSource(id, {type: "image", url: tile.asset_url, coordinates: tile.coordinates});
    map.addLayer({id, type: "raster", source: id, paint: {"raster-fade-duration": 0}});
    state.imagery.push(id);
  }

  const colors = {roads: "#efdcb0", buildings: "#f6faf0", drains: "#72b8dd", waterways: "#72b8dd"};
  for (const [name, collection] of Object.entries(data.vector_layers || {})) {
    const id = "base-" + name;
    map.addSource(id, {type: "geojson", data: collection});
    map.addLayer({id, type: "line", source: id, paint: {"line-color": colors[name] || "#e9eac4", "line-width": name === "buildings" ? 1 : 1.6, "line-opacity": 0.72}});
    state.vectorIds.push(id);
    const label = document.createElement("label");
    const input = document.createElement("input");
    input.type = "checkbox";
    input.checked = true;
    input.addEventListener("change", () => map.setLayoutProperty(id, "visibility", input.checked ? "visible" : "none"));
    label.append(input, document.createTextNode(" " + nice(name)));
    $("layers").append(label);
  }

  addGeometryLayers("matches", "#75d480");
  addGeometryLayers("uncertain", "#edbd52", true);
  addGeometryLayers("context", "#65bcee");
  addGeometryLayers("observations", "#ffad51", true);
  addGeometryLayers("selection", "#f5f76c");
  for (const layer of ["matches", "uncertain"]) {
    for (const suffix of ["-fill", "-line", "-point"]) {
      const id = layer + suffix;
      map.on("click", id, event => {
        const identifier = event.features?.[0]?.id ?? event.features?.[0]?.properties?.id;
        const all = [...(state.last?.features || []), ...(state.last?.tile_candidates || []), ...(state.last?.uncertain_candidates || [])];
        const feature = all.find(item => String(featureKey(item)) === String(identifier));
        if (feature) selectFeature(feature);
      });
      map.on("mouseenter", id, () => { map.getCanvas().style.cursor = "pointer"; });
      map.on("mouseleave", id, () => { map.getCanvas().style.cursor = ""; });
    }
  }

  $("imageryToggle").addEventListener("change", () => state.imagery.forEach(id => map.setLayoutProperty(id, "visibility", $("imageryToggle").checked ? "visible" : "none")));
  $("fitArea").onclick = () => fit(data.coverage?.bbox_wgs84 || boundsOf(data.tiles || []));
  fit(data.coverage?.bbox_wgs84 || boundsOf(data.tiles || []), 50);
  const counts = data.counts?.tiles || {};
  const truncated = counts.truncated || Object.values(data.counts?.vector_layers || {}).some(count => count.truncated);
  $("mapNote").textContent = `${(data.tiles || []).length} aerial tiles · ${data.coverage?.acquisition_dates?.join(", ") || "acquisition date unknown"}${truncated ? " · map display is limited" : ""}. Study evidence may differ from conditions today.`;
  if (state.last) renderResults(state.last);
}

function normalizeMode(value, evidenceKind) {
  const mode = String(value || "").toLowerCase();
  const kind = String(evidenceKind || "").toLowerCase();
  if (["ground", "detect", "detection", "live_grounding"].includes(mode) || kind === "live_model_detection") return "ground";
  if (["explore", "image", "imagery", "tile"].includes(mode) || ["image_tile", "tile_retrieval"].includes(kind)) return "explore";
  return "catalog_scan";
}

function normalizeTaskCatalog(raw) {
  if (!Array.isArray(raw)) return [];
  return raw.flatMap(item => {
    if (typeof item === "string") return [{query: item, mode: "catalog_scan", evidence_kind: "prepared_evidence"}];
    const queries = Array.isArray(item.examples) ? item.examples : [item.query || item.example];
    return queries.filter(Boolean).map(query => ({query: String(query), label: item.label ? String(item.label) : null, mode: normalizeMode(item.mode, item.evidence_kind), evidence_kind: item.evidence_kind || null}));
  });
}

function reasonText(value) {
  if (!value) return "required local model assets are not ready";
  if (typeof value === "string") return nice(value);
  if (Array.isArray(value)) return value.map(reasonText).join("; ");
  return value.reason || value.message || JSON.stringify(value);
}

function configureGrounding(c) {
  const grounding = c.grounding || {};
  const available = c.live_grounding === true;
  const modeInput = document.querySelector('input[name="mode"][value="ground"]');
  modeInput.dataset.capDisabled = String(!available);
  modeInput.disabled = !available;
  $("groundMode").classList.toggle("unavailable", !available);
  const maximum = Math.max(1, Math.min(4, Number(grounding.max_tiles || c.ground_limit || 4)));
  const select = $("groundingTileLimit");
  const selected = Math.min(maximum, Number(select.value) || 2);
  select.replaceChildren();
  for (let count = 1; count <= maximum; count += 1) {
    const option = document.createElement("option");
    option.value = String(count);
    option.textContent = `${count} ${count === 1 ? "tile" : "tiles"}`;
    option.selected = count === selected;
    select.append(option);
  }
  $("groundLimitHelp").textContent = `A smaller run is faster. Detection covers only these ranked tiles; this dataset allows up to ${maximum}.`;
  return available ? null : `Detect objects is unavailable for this dataset: ${reasonText(grounding.reason || c.retrieval?.reason || c.retrieval?.error || c.retrieval?.status || c.unavailable_reasons?.ground)}.`;
}

function configureExploration(c) {
  const available = c.text_similarity_enabled !== false;
  const modeInput = document.querySelector('input[name="mode"][value="explore"]');
  modeInput.dataset.capDisabled = String(!available);
  modeInput.disabled = !available;
  $("exploreMode").classList.toggle("unavailable", !available);
  return available ? null : `Image discovery is unavailable for this dataset: ${reasonText(c.retrieval?.reason || c.retrieval?.error || c.retrieval?.status || c.unavailable_reasons?.explore)}.`;
}

async function loadCapabilities() {
  const c = state.capabilities = await request(datasetUrl("/api/capabilities"));
  $("area").textContent = c.area_name || "Configured study area";
  state.taskCatalog = normalizeTaskCatalog(c.task_catalog);
  const notices = [configureGrounding(c), configureExploration(c)].filter(Boolean);
  $("modeAvailability").hidden = notices.length === 0;
  $("modeAvailability").textContent = notices.join(" ");
  if (document.querySelector('input[name="mode"]:checked')?.disabled) document.querySelector('input[name="mode"][value="catalog_scan"]').checked = true;
  updateModeUi(currentMode());
  return c;
}

async function loadDatasets() {
  let payload;
  try {
    payload = await request("/api/datasets");
  } catch (error) {
    payload = {datasets: [{id: "", name: "Configured dataset", default: true}], default_dataset_id: ""};
  }
  const datasets = Array.isArray(payload.datasets) && payload.datasets.length ? payload.datasets : [{id: "", name: "Configured dataset", default: true}];
  state.datasets = datasets;
  const select = $("datasetSelect");
  select.replaceChildren();
  for (const dataset of datasets) {
    const option = document.createElement("option");
    option.value = dataset.id || "";
    option.textContent = dataset.name || dataset.id || "Configured dataset";
    option.title = dataset.source || "";
    select.append(option);
  }
  const fallback = datasets.find(dataset => dataset.default)?.id ?? datasets[0]?.id ?? "";
  state.datasetId = payload.default_dataset_id ?? fallback;
  select.value = state.datasetId || "";
  $("datasetControl").hidden = false;
}

function capabilityStatus(c) {
  const messages = [`${c.coverage?.tile_count ?? "?"} prepared tiles loaded.`];
  if (c.text_similarity_enabled) messages.push("Image discovery is ready.");
  if (c.live_grounding) {
    const loaded = ["ready", "loaded"].includes(c.grounding?.status);
    messages.push(loaded ? "Bounded on-demand detection is loaded." : "Bounded on-demand detection is configured; its local model loads on the first run.");
  }
  return messages.join(" ");
}

async function loadDataset(datasetId) {
  const sequence = ++state.loadSequence;
  state.searchSequence += 1;
  state.datasetId = datasetId || null;
  $("datasetSelect").disabled = true;
  resetSearchState({clearQuery: true});
  destroyMap();
  $("area").textContent = "Loading study area";
  $("mapNote").textContent = "Loading local imagery and map features…";
  status("Loading the selected dataset and its capabilities…");
  const checks = await Promise.allSettled([loadCapabilities(), loadMap()]);
  if (sequence !== state.loadSequence) return;
  $("datasetSelect").disabled = false;
  const failures = checks.filter(result => result.status === "rejected");
  if (failures.length) {
    status(failures.map(result => result.reason.message).join(" "), true);
    if (checks[1].status === "rejected") $("mapNote").textContent = "A real, prepared evidence bundle is required to display this dataset.";
  } else status(capabilityStatus(checks[0].value));
}

function modeTasks(mode) {
  const configured = state.taskCatalog.filter(task => task.mode === mode);
  const fallback = FALLBACK_TASKS.filter(task => task.mode === mode);
  const seen = new Set();
  return (configured.length ? configured : fallback).filter(task => {
    const key = task.query.toLowerCase();
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  }).slice(0, 3);
}

function clearConversationContext() {
  state.previousPlan = null;
  state.groundingRunId = null;
  $("followup").hidden = true;
}

function renderExamples(mode) {
  $("examples").replaceChildren();
  for (const task of modeTasks(mode)) {
    const button = text("button", task.label || task.query, "example");
    button.type = "button";
    button.dataset.evidenceKind = task.evidence_kind || "";
    button.addEventListener("click", () => {
      clearConversationContext();
      $("query").value = task.query;
      $("query").focus();
    });
    $("examples").append(button);
  }
}

function updateModeUi(mode) {
  const ui = MODE_UI[mode] || MODE_UI.catalog_scan;
  $("modeHelp").textContent = ui.help;
  $("query").placeholder = modeTasks(mode)[0]?.query || ui.placeholder;
  $("queryLabel").textContent = ui.label;
  $("searchButtonLabel").textContent = ui.button;
  $("examplesLabel").textContent = ui.examples;
  $("groundControls").hidden = mode !== "ground";
  renderExamples(mode);
}

function emptyMessage(profile) {
  if (profile === "tile") return "No image candidates were returned for this scene description.";
  if (profile === "mapped") return "No supplied map records satisfy every condition. This does not establish real-world absence or map completeness.";
  if (profile === "live") return "No detector observations passed the threshold in the analyzed tiles. The bounded run does not establish absence across the full study area.";
  return "No prepared detections satisfy every condition. This does not establish real-world absence.";
}

function showEmpty(message = "Choose a workflow and enter a question to inspect geographic evidence.") {
  const root = text("div", "", "empty");
  const shape = text("div", "", "empty-shape");
  shape.setAttribute("aria-hidden", "true");
  root.append(shape, text("strong", "A question becomes a place."), text("p", message));
  $("results").replaceChildren(root);
}

function clearResultSources() {
  for (const id of ["matches", "uncertain", "context", "selection", "observations"]) source(id, []);
}

function resetSearchState({clearQuery = false} = {}) {
  clearConversationContext();
  state.last = null;
  state.request = null;
  state.selected = null;
  if (clearQuery) $("query").value = "";
  $("planPanel").hidden = true;
  $("download").hidden = true;
  $("exportOptions").hidden = true;
  $("includeReview").checked = false;
  $("pagination").hidden = true;
  $("scope").textContent = "";
  $("executionMeta").hidden = true;
  $("executionMeta").textContent = "";
  $("resultsTitle").textContent = "Your evidence appears here";
  $("evidence").hidden = true;
  clearResultSources();
  showEmpty();
}

function featureKey(feature) {
  const properties = feature?.properties || {};
  return feature?.id ?? properties.id ?? properties.detection_id ?? properties.tile_id;
}

function evidenceKind(feature, data = state.last) {
  const properties = feature?.properties || {};
  const declared = properties.evidence_kind || properties.evidence_type;
  if (declared) return declared;
  if (data?.result_kind === "tile_retrieval") return "image_tile";
  if (data?.mode === "ground" || state.request?.mode === "ground") return "live_model_detection";
  if (["mapped_features", "vector_features", "mapped_records"].includes(data?.result_kind)) return "mapped_feature";
  return "prepared_model_detection";
}

function resultProfile(data, features = []) {
  if (data.result_kind === "tile_retrieval") return "tile";
  const kinds = new Set(features.map(feature => evidenceKind(feature, data)));
  if (kinds.has("live_model_detection") || data.mode === "ground") return "live";
  if (kinds.has("mapped_feature") || ["mapped_features", "vector_features", "mapped_records"].includes(data.result_kind)) return "mapped";
  return "prepared";
}

function featureTitle(feature, profile) {
  const properties = feature.properties || {};
  if (profile === "tile") return "Image candidate";
  if (profile === "mapped") return nice(properties.display_name || properties.name || properties.visual_concept || properties.feature_type || properties.subject_layer || properties.layer || "Mapped record");
  return nice(properties.visual_concept || properties.concept_id || state.last?.plan?.visual_query || "Detected object");
}

function conditionDistance(properties) {
  return (properties.conditions || []).find(condition => Number.isFinite(condition.distance_m));
}

function vectorDate(properties) {
  const layer = properties.subject_layer || properties.layer || properties.source_layer || (properties.conditions || []).find(condition => condition.reference_layer)?.reference_layer;
  return properties.vector_extraction_date || state.capabilities?.vector_layers?.[layer]?.extraction_date;
}

function cardMeta(properties, profile) {
  const distance = conditionDistance(properties);
  const prefix = Number.isFinite(distance?.distance_m) ? `${distance.distance_m.toFixed(1)} m to ${nice(distance.reference_layer || "reference")} · ` : "";
  if (profile === "mapped") return `${prefix}${nice(properties.subject_layer || properties.layer || properties.source_layer || "Map layer")} · snapshot ${vectorDate(properties) || "date unknown"}`;
  return `${prefix}Imagery ${properties.imagery_date || properties.acquisition_date || "date unknown"}`;
}

function badgeFor(profile, uncertain) {
  if (uncertain) return ["Needs review", "badge review"];
  if (profile === "tile") return ["Whole-image similarity", "badge tile"];
  if (profile === "mapped") return ["Mapped source record", "badge mapped"];
  if (profile === "live") return ["On-demand detector observation", "badge live"];
  return ["Prepared detector evidence", "badge"];
}

function renderExecutionMeta(data, profile) {
  const element = $("executionMeta");
  if (profile !== "live" || (!data.grounding_run_id && !data.grounding)) {
    element.hidden = true;
    element.textContent = "";
    return;
  }
  const grounding = data.grounding || {};
  const parts = [];
  if (data.grounding_run_id) parts.push(`Run ${String(data.grounding_run_id).slice(0, 12)}`);
  if (Number.isFinite(grounding.tiles_processed)) parts.push(`${grounding.tiles_processed}${Number.isFinite(grounding.tiles_available) ? ` of ${grounding.tiles_available}` : ""} ranked tiles analyzed`);
  if (Number.isFinite(grounding.threshold)) parts.push(`threshold ${grounding.threshold.toFixed(3)}`);
  if (grounding.cache_hit) parts.push("saved observations reused");
  element.textContent = parts.join(" · ");
  element.hidden = parts.length === 0;
}

function renderResults(data) {
  const tiles = data.result_kind === "tile_retrieval";
  const features = tiles ? (data.tile_candidates || []) : (data.features || []);
  const uncertain = data.uncertain_candidates || [];
  const profile = resultProfile(data, [...features, ...uncertain]);
  const reviewCount = data.total_uncertain ?? uncertain.length;
  const total = tiles ? (data.total_tile_candidates ?? features.length) : (data.total_matches ?? features.length);
  const noun = profile === "tile" ? "image candidates" : profile === "mapped" ? "mapped records" : profile === "live" ? "on-demand detections" : "prepared matches";
  $("resultsTitle").textContent = `${total} ${noun}${reviewCount ? ` · ${reviewCount} need review` : ""}`;
  $("scope").textContent = data.claim_scope || "";
  renderExecutionMeta(data, profile);
  $("results").replaceChildren();
  if (!features.length && !uncertain.length) showEmpty(emptyMessage(profile));

  for (const [items, isUncertain] of [[features, false], [uncertain, true]]) {
    for (const feature of items) {
      const properties = feature.properties || {};
      const itemProfile = resultProfile(data, [feature]);
      const card = document.createElement("button");
      card.type = "button";
      card.className = "result";
      card.dataset.id = String(featureKey(feature));
      card.setAttribute("aria-pressed", "false");
      if (properties.asset_url && itemProfile !== "mapped") {
        const image = document.createElement("img");
        image.src = properties.asset_url;
        image.alt = itemProfile === "tile" ? "Aerial image candidate" : "Aerial source tile for this observation";
        image.loading = "lazy";
        card.append(image);
      } else {
        const marker = text("span", "", `result-marker ${itemProfile}`);
        marker.setAttribute("aria-hidden", "true");
        card.append(marker);
      }
      const content = text("div", "", "result-content");
      content.append(text("span", isUncertain ? `Needs review: ${featureTitle(feature, itemProfile)}` : featureTitle(feature, itemProfile), "result-title"));
      content.append(text("span", cardMeta(properties, itemProfile), "result-meta"));
      const [badgeText, badgeClass] = badgeFor(itemProfile, isUncertain);
      content.append(text("span", badgeText, badgeClass));
      card.append(content);
      card.addEventListener("click", () => selectFeature(feature, card));
      $("results").append(card);
    }
  }

  source("matches", features);
  source("uncertain", uncertain);
  source("context", data.context_features || []);
  source("selection", []);
  source("observations", []);
  $("evidence").hidden = true;
  $("matchLegend").textContent = profile === "tile" ? "Image candidates" : profile === "mapped" ? "Mapped records" : profile === "live" ? "On-demand detections" : "Prepared matches";
  $("download").hidden = false;
  $("exportOptions").hidden = !reviewCount;
  const pagingTotal = tiles ? (data.total_tile_candidates ?? features.length) : Math.max(data.total_matches || 0, reviewCount);
  const size = state.request?.page_size || 20;
  $("pagination").hidden = pagingTotal <= size;
  $("prevPage").disabled = data.page === 0;
  $("nextPage").disabled = (data.page + 1) * size >= pagingTotal;
  $("pageLabel").textContent = `${data.page + 1} / ${Math.max(1, Math.ceil(pagingTotal / size))}`;
  $("sourceNote").textContent = profile === "mapped"
    ? "Mapped results describe the supplied vector snapshot. Missing records do not establish real-world absence or map completeness."
    : profile === "live"
      ? "On-demand results cover only the ranked tiles analyzed in this run. They do not establish completeness across the wider study area."
      : profile === "tile"
        ? "Image discovery ranks whole tiles by similarity. It does not verify an object or its exact location."
        : "Results describe the loaded prepared evidence. Missing detections or map tags do not establish real-world absence.";
}

function planTaskLabel(data, profile) {
  if (profile === "tile") return "Whole-image discovery";
  if (profile === "mapped") return "Mapped-record search";
  if (profile === "live") return "Bounded on-demand detection";
  return "Prepared-evidence search";
}

function predicateText(plan) {
  if (plan.predicate === "within_distance") return `Within ${plan.distance_m} metres of ${nice(plan.reference_layer)}`;
  if (plan.predicate === "intersects") return `Intersects ${nice(plan.reference_layer)}`;
  return "No spatial relationship requested";
}

function renderPlan(plan, data) {
  const profile = resultProfile(data, [...(data.features || []), ...(data.tile_candidates || []), ...(data.uncertain_candidates || [])]);
  const rows = [["Task", planTaskLabel(data, profile)], ["Looking for", plan.visual_query || nice(plan.concept_id)], ["Spatial clause", predicateText(plan)]];
  if (plan.attributes?.length) rows.push(["Attribute clauses", plan.attributes.map(attribute => `${nice(attribute.role || "feature")} ${attribute.field} ${nice(attribute.operator)} ${attribute.value}`).join("; ")]);
  for (const [index, relation] of (plan.relations || []).entries()) {
    const relationText = relation.predicate === "within_distance"
      ? `Within ${relation.distance_m} metres of ${nice(relation.reference_layer)}`
      : `Intersects ${nice(relation.reference_layer)}`;
    const filters = (relation.attributes || []).map(attribute => `${attribute.field} ${nice(attribute.operator)} ${attribute.value}`).join("; ");
    rows.push([`AND clause ${index + 2}`, filters ? `${relationText}; ${filters}` : relationText]);
  }
  if (profile === "live") {
    const count = data.grounding?.tiles_processed ?? state.request?.grounding_tile_limit;
    rows.push(["Detection scope", count ? `${count} top-ranked image ${count === 1 ? "tile" : "tiles"}` : "Bounded ranked image tiles"]);
  }
  const root = $("planSummary");
  root.replaceChildren();
  for (const [label, value] of rows) root.append(text("dt", label), text("dd", value));
  $("plan").textContent = JSON.stringify(plan, null, 2);
  $("planPanel").hidden = false;
}

function conditionExplanation(condition) {
  if (condition.reason) return condition.reason;
  if (condition.condition === "vector_source") return `Supplied ${nice(condition.subject_layer)} record ${condition.feature_id || ""} from vector snapshot ${condition.vector_extraction_date || "date unknown"}.`;
  if (condition.condition === "visual_detection") return `Model observation of ${nice(condition.visual_query || condition.concept_id || state.last?.plan?.visual_query || "requested object")} · score ${number(condition.score)}.`;
  if (condition.condition === "feature_association") return `${nice(condition.association)} association with ${nice(condition.layer)}${condition.feature_id ? " · " + condition.feature_id : ""}.`;
  if (condition.field) {
    const requested = condition.expected ?? condition.value ?? "requested value";
    return `${nice(condition.role || "feature")} ${condition.field}: observed ${condition.observed ?? "unknown"}; requested ${nice(condition.operator || "equals")} ${requested}.`;
  }
  const reference = condition.reference_layer || condition.layer || "reference geometry";
  return `${Number.isFinite(condition.distance_m) ? condition.distance_m.toFixed(2) + " metres to " : "Checked against "}${nice(reference)} on the supplied geometry.`;
}

function closeEvidence({restoreFocus = true} = {}) {
  $("evidence").hidden = true;
  source("selection", []);
  source("observations", []);
  if (restoreFocus && state.selectedControl?.isConnected) state.selectedControl.focus();
  state.selectedControl = null;
}

function selectFeature(feature, control = null) {
  const properties = feature.properties || {};
  const profile = resultProfile(state.last, [feature]);
  state.selected = feature;
  state.selectedControl = control || document.activeElement;
  document.querySelectorAll(".result").forEach(card => {
    const selected = card.dataset.id === String(featureKey(feature));
    card.classList.toggle("selected", selected);
    card.setAttribute("aria-pressed", String(selected));
  });
  source("selection", [feature]);
  source("observations", properties.evidence_geometries || []);
  fit(boundsOf([feature]), 80);

  const kindLabel = profile === "tile" ? "Image discovery candidate" : profile === "mapped" ? "Mapped source record" : profile === "live" ? "On-demand detector evidence" : "Prepared detector evidence";
  const root = $("evidenceBody");
  const heading = text("h2", featureTitle(feature, profile));
  heading.id = "evidenceTitle";
  root.replaceChildren(text("div", kindLabel, "eyebrow"), heading);

  const observation = properties.observations?.[0] || {};
  const assetUrl = properties.asset_url || observation.asset_url;
  const box = properties.pixel_box || observation.pixel_box;
  const width = properties.original_width || observation.original_width || properties.width;
  const height = properties.original_height || observation.original_height || properties.height;
  if (assetUrl && profile !== "mapped") {
    const frame = text("div", "", "evidence-image");
    const image = document.createElement("img");
    image.src = assetUrl;
    image.alt = profile === "tile" ? "Aerial image candidate" : "Aerial source tile with the detector observation outlined when a box is available";
    frame.append(image);
    if (box?.length === 4 && width > 0 && height > 0) {
      const overlay = text("div", "", "pixel-box");
      overlay.style.left = (box[0] / width * 100) + "%";
      overlay.style.top = (box[1] / height * 100) + "%";
      overlay.style.width = ((box[2] - box[0]) / width * 100) + "%";
      overlay.style.height = ((box[3] - box[1]) / height * 100) + "%";
      frame.append(overlay);
    }
    root.append(frame);
  }

  const layer = properties.subject_layer || properties.layer || properties.source_layer || properties.association_layer;
  const commonBounds = boundsOf([feature])?.map(value => value.toFixed(6)).join(", ");
  let fields;
  if (profile === "mapped") {
    fields = [["Map layer", nice(layer || "Mapped feature")], ["Vector snapshot", vectorDate(properties) || "Unknown"], ["Feature ID", featureKey(feature)], ["Geometry", nice(properties.geometry_role || feature.geometry?.type)], ["Source", properties.source_name || properties.source || "Supplied vector data"], ["Imagery coverage dates", properties.imagery_coverage_dates?.join(", ")], ["Bounds W,S,E,N", commonBounds]];
  } else if (profile === "tile") {
    fields = [["Imagery date", properties.imagery_date || properties.acquisition_date || "Unknown"], ["Image similarity", number(properties.retrieval_similarity)], ["Source tile", properties.tile_id], ["Bounds W,S,E,N", commonBounds]];
  } else {
    fields = [["Imagery date", properties.imagery_date || properties.acquisition_date || observation.imagery_date || "Unknown"], ["Vector snapshot", vectorDate(properties) || "Not used"], ["Geometry", nice(properties.geometry_role || "visual detection envelope")], ["Detector score", number(properties.visual_score ?? properties.score)], ["Image similarity", number(properties.retrieval_similarity)], ["Association", nice(properties.association?.status || properties.association || "Not applicable")], ["Source tile", properties.tile_id || observation.tile_id], ["Detection run", profile === "live" ? state.last?.grounding_run_id : null], ["Model", properties.model_id || observation.model_id || state.last?.grounding?.model_id], ["Bounds W,S,E,N", commonBounds]];
  }
  const definitionList = document.createElement("dl");
  for (const [label, value] of fields) {
    if (value === null || value === undefined || value === "") continue;
    definitionList.append(text("dt", label), text("dd", value));
  }
  root.append(definitionList);

  const explanation = profile === "mapped"
    ? "This result comes from the supplied map or vector snapshot. It has no detector score or image crop."
    : profile === "tile"
      ? "Similarity ranks whole images. This candidate has no verified object location or spatial relationship."
      : profile === "live"
        ? "This detector observation was produced within the ranked tiles analyzed for this run. Scores are uncalibrated model outputs, and the bounded run does not establish wider-area completeness."
        : "This detector observation was prepared before the search. Scores are uncalibrated model outputs; the outline shows the original model observation when available.";
  root.append(text("p", explanation, "evidence-note"));

  if (properties.conditions?.length) {
    const section = text("div", "", "conditions");
    section.append(text("h3", "Evidence checks"));
    for (const condition of properties.conditions) {
      const div = text("div", "", `condition ${condition.status || "unknown"}`);
      div.append(text("strong", `${nice(condition.condition || condition.predicate || condition.field || "Evidence check")}: ${nice(condition.status)}`), text("p", conditionExplanation(condition)));
      section.append(div);
    }
    root.append(section);
  }
  const details = document.createElement("details");
  details.append(text("summary", "Full evidence record"), text("pre", JSON.stringify(properties, null, 2)));
  root.append(details);
  $("evidence").hidden = false;
  $("evidence").focus({preventScroll: true});
  if (window.matchMedia("(max-width:760px)").matches) $("evidence").scrollIntoView({behavior: "smooth", block: "nearest"});
}

function setBusy(busy, mode = currentMode(), reuseGrounding = false) {
  $("searchButton").disabled = busy;
  $("searchForm").setAttribute("aria-busy", String(busy));
  $("results").setAttribute("aria-busy", String(busy));
  $("query").readOnly = busy;
  $("groundingTileLimit").disabled = busy;
  $("datasetSelect").disabled = busy;
  document.querySelectorAll('input[name="mode"]').forEach(input => { input.disabled = busy || input.dataset.capDisabled === "true"; });
  $("searchButtonLabel").textContent = busy ? (mode === "ground" ? (reuseGrounding ? "Rechecking evidence…" : "Detecting objects…") : mode === "explore" ? "Ranking images…" : "Checking evidence…") : MODE_UI[mode].button;
}

// The server validates the edit; this only controls reuse of the displayed run.
const radiusFollowup = query => /^\s*(?:(?:please\s+)?(?:make|change|set|increase|decrease)\s+(?:it|this|that|the\s+radius|radius|distance)\s*(?:to\s*)?|(?:use|try)\s+|(?:radius|distance)\s*(?:to|=)?\s*)-?\d+(?:\.\d+)?\s*(?:kilometres?|kilometers?|km|metres?|meters?|m)(?:\s+instead)?[.!?]?\s*$/i.test(query);

async function search(page = null) {
  const query = $("query").value.trim();
  if (page === null && query.length < 3) return;
  const mode = page !== null ? state.request.mode : currentMode();
  const reuseGrounding = mode === "ground" && Boolean(state.groundingRunId) && (page !== null ||
    (radiusFollowup(query) && Number($("groundingTileLimit").value) === state.request?.grounding_tile_limit));
  const sequence = ++state.searchSequence;
  setBusy(true, mode, reuseGrounding);
  status(mode === "ground" ? (reuseGrounding ? "Rechecking spatial conditions using saved detections…" : "Ranking imagery, then running bounded on-demand detection. This can take longer…") : mode === "explore" ? "Ranking whole images by scene similarity…" : "Interpreting the question and checking prepared geographic evidence…");
  try {
    let body;
    if (page !== null) {
      body = {plan: state.last.plan, dataset_version: state.last.dataset_version, dataset_id: state.datasetId, mode, page, page_size: state.request.page_size};
      if (mode === "ground") {
        body.grounding_tile_limit = state.request.grounding_tile_limit;
        body.grounding_threshold = state.request.grounding_threshold;
        body.grounding_run_id = state.groundingRunId || state.last.grounding_run_id;
      }
    } else {
      body = {query, previous_plan: state.previousPlan, dataset_id: state.datasetId, mode, page: 0, page_size: 20};
      if (mode === "ground") {
        body.grounding_tile_limit = Number($("groundingTileLimit").value) || 2;
        body.grounding_threshold = state.groundingThreshold;
        if (reuseGrounding) body.grounding_run_id = state.groundingRunId;
      }
    }
    if (!body.dataset_id) delete body.dataset_id;
    const data = await request("/api/query", body);
    if (sequence !== state.searchSequence) return;
    state.last = data;
    state.groundingRunId = mode === "ground" ? (data.grounding_run_id || body.grounding_run_id || null) : null;
    state.request = {...body, grounding_run_id: state.groundingRunId};
    state.previousPlan = data.plan;
    renderPlan(data.plan, data);
    $("followup").hidden = false;
    renderResults(data);
    const profile = resultProfile(data, [...(data.features || []), ...(data.tile_candidates || [])]);
    if (profile === "tile") status(`${data.tiles_processed} of ${data.tiles_available} image tiles ranked. Inspect candidates before making object claims.`);
    else if (profile === "mapped") status(`${data.total_matches ?? data.features?.length ?? 0} mapped records satisfy the supplied clauses. These results have no detector or image-crop claim.`);
    else if (profile === "live") status(`${data.grounding?.tiles_processed ?? data.tiles_processed} ranked tiles analyzed on demand. Reuse this saved run for follow-ups, paging, and export.`);
    else status(`${data.tiles_processed} of ${data.tiles_available} prepared tiles checked. Results use cached detections and supplied spatial evidence.`);
  } catch (error) {
    if (sequence === state.searchSequence) status(error.message, true);
  } finally {
    if (sequence === state.searchSequence) setBusy(false, mode);
  }
}

$("searchForm").addEventListener("submit", event => { event.preventDefault(); search(); });
$("prevPage").addEventListener("click", () => search(Math.max(0, state.last.page - 1)));
$("nextPage").addEventListener("click", () => search(state.last.page + 1));
$("reset").addEventListener("click", () => { clearConversationContext(); status("Follow-up context cleared. Enter a new question."); });
document.querySelectorAll('input[name="mode"]').forEach(input => input.addEventListener("change", () => {
  state.searchSequence += 1;
  clearConversationContext();
  state.last = null;
  state.request = null;
  $("planPanel").hidden = true;
  $("download").hidden = true;
  $("exportOptions").hidden = true;
  $("includeReview").checked = false;
  $("pagination").hidden = true;
  $("scope").textContent = "";
  $("executionMeta").hidden = true;
  $("resultsTitle").textContent = "Your evidence appears here";
  closeEvidence({restoreFocus: false});
  clearResultSources();
  showEmpty({catalog_scan: "Enter a question to search prepared geographic evidence.", explore: "Describe a scene to discover relevant imagery.", ground: "Describe the objects to detect in selected imagery."}[input.value]);
  updateModeUi(input.value);
}));
$("closeEvidence").addEventListener("click", () => closeEvidence());
document.addEventListener("keydown", event => { if (event.key === "Escape" && !$("evidence").hidden) closeEvidence(); });
$("datasetSelect").addEventListener("change", event => loadDataset(event.target.value));
$("download").addEventListener("click", () => {
  if (!state.last) return;
  const params = new URLSearchParams({plan: JSON.stringify(state.last.plan), mode: state.request.mode, dataset_version: state.last.dataset_version, include_uncertain: String($("includeReview").checked)});
  if (state.datasetId) params.set("dataset_id", state.datasetId);
  const displayedRunId = state.groundingRunId || state.last.grounding_run_id;
  if (state.request.mode === "ground" && displayedRunId) params.set("grounding_run_id", displayedRunId);
  const anchor = document.createElement("a");
  anchor.href = "/api/export?" + params.toString();
  anchor.download = "terralens-evidence.geojson";
  document.body.append(anchor);
  anchor.click();
  anchor.remove();
  status(state.request.mode === "ground" ? "GeoJSON download requested from the saved detection run; no new inference was requested." : "GeoJSON download requested for the selected execution mode and its supporting evidence.");
});

async function start() {
  await loadDatasets();
  await loadDataset(state.datasetId);
}

start();
