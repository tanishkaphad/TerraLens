"use strict";

const state = {
  map: null,
  mapData: null,
  capabilities: null,
  ready: false,
  busy: false,
  result: null,
  records: [],
  selectedId: null,
  areaBounds: null,
  imageryLayers: [],
  searchController: null,
  busyTimer: null,
  busyIndex: 0,
  datasets: [],
  datasetId: null,
  vectorLayers: [],
  basemapMode: "street",
  mode: "search",
  assistant: null,
  conversationId: null,
  exportUrl: null,
  infoTrigger: null,
};

const $ = id => document.getElementById(id);
const collection = features => ({type: "FeatureCollection", features});
const isCount = value => Number.isInteger(value) && value >= 0;
const finite = value => (typeof value === "number" || typeof value === "string" && value.trim() !== "") && Number.isFinite(Number(value));
const displayNumber = value => finite(value) ? Number(value).toFixed(3) : "Not available";
const pretty = value => String(value || "Unknown").replaceAll("_", " ").replace(/\b\w/g, character => character.toUpperCase());
const safeClass = value => String(value || "unknown").toLowerCase().replace(/[^a-z0-9_-]+/g, "-");

function node(tag, value = "", className = "") {
  const element = document.createElement(tag);
  if (className) element.className = className;
  if (value !== "") element.textContent = String(value);
  return element;
}

async function requestJson(url, options = {}) {
  const response = await fetch(url, options);
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    const detail = typeof data.detail === "string" ? data.detail
      : Array.isArray(data.detail) ? data.detail.slice(0, 2).map(item => `${item.loc?.slice(-1)?.[0] || "request"}: ${item.msg || "invalid value"}`).join("; ")
        : data.detail?.message;
    throw new Error(detail || `Request failed (${response.status}).`);
  }
  return data;
}

function safeAssetUrl(value) {
  if (typeof value !== "string" || !value) return null;
  try {
    const url = new URL(value, window.location.origin);
    if (url.origin !== window.location.origin || !url.pathname.startsWith("/api/assets/")) return null;
    return url.href;
  } catch (_) {
    return null;
  }
}

function safeSourceUrl(value) {
  if (typeof value !== "string" || !value) return null;
  try {
    const url = new URL(value);
    const hosts = ["openstreetmap.org", "nominatim.openstreetmap.org", "maptiler.com", "maptiler.cloud", "overpass-api.de", "planetarycomputer.microsoft.com", "usgs.gov", "usda.gov", "esa.int", "earth-search.aws.element84.com"];
    return url.protocol === "https:" && hosts.some(host => url.hostname === host || url.hostname.endsWith(`.${host}`)) ? url.href : null;
  } catch (_) { return null; }
}

function setStatus(message, type = "") {
  $("searchStatus").textContent = message;
  $("searchStatus").className = `search-status${type ? ` ${type}` : ""}`;
}

function setBusy(busy) {
  state.busy = busy;
  document.body.classList.toggle("is-searching", busy);
  $("searchForm").setAttribute("aria-busy", String(busy));
  $("query").readOnly = busy;
  $("query").disabled = !state.ready;
  $("datasetSelect").disabled = busy || !state.ready;
  $("expertMode").disabled = busy;
  $("conversationMode").disabled = busy || !state.assistant?.configured;
  document.querySelectorAll(".example-query").forEach(button => { button.disabled = busy || !state.ready; });
  updateSubmit();
}

function startBusyStatus() {
  const messages = ["Understanding your question", "Finding the right geographic sources", "Checking available evidence", "Preparing map results"];
  state.busyIndex = 0;
  setStatus(messages[0]);
  window.clearInterval(state.busyTimer);
  state.busyTimer = window.setInterval(() => {
    state.busyIndex = Math.min(state.busyIndex + 1, messages.length - 1);
    setStatus(messages[state.busyIndex]);
  }, 3500);
}

function stopBusyStatus() {
  window.clearInterval(state.busyTimer);
  state.busyTimer = null;
}

function updateSubmit() {
  const valid = $("query").value.trim().length >= 3;
  $("searchButton").disabled = !state.ready || state.busy || !valid;
}

function featureBounds(features) {
  const points = [];
  const walk = value => {
    if (Array.isArray(value) && typeof value[0] === "number" && typeof value[1] === "number") points.push(value);
    else if (Array.isArray(value)) value.forEach(walk);
  };
  for (const feature of features) walk(feature?.geometry?.coordinates);
  if (!points.length) return null;
  return points.reduce((bounds, point) => [
    Math.min(bounds[0], point[0]),
    Math.min(bounds[1], point[1]),
    Math.max(bounds[2], point[0]),
    Math.max(bounds[3], point[1]),
  ], [180, 90, -180, -90]);
}

function outerRings(geometry) {
  if (!geometry || !Array.isArray(geometry.coordinates)) return [];
  if (geometry.type === "Polygon") return geometry.coordinates[0] ? [geometry.coordinates[0]] : [];
  if (geometry.type === "MultiPolygon") return geometry.coordinates.map(polygon => polygon[0]).filter(Boolean);
  return [];
}

function centroidOfGeometry(geometry) {
  const rings = outerRings(geometry);
  let best = null;
  for (const ring of rings) {
    if (!ring.length) continue;
    const originX = Number(ring[0][0]);
    const originY = Number(ring[0][1]);
    let twiceArea = 0;
    let longitude = 0;
    let latitude = 0;
    for (let index = 0; index < ring.length - 1; index += 1) {
      const x1 = Number(ring[index][0]) - originX;
      const y1 = Number(ring[index][1]) - originY;
      const x2 = Number(ring[index + 1][0]) - originX;
      const y2 = Number(ring[index + 1][1]) - originY;
      const cross = x1 * y2 - x2 * y1;
      twiceArea += cross;
      longitude += (x1 + x2) * cross;
      latitude += (y1 + y2) * cross;
    }
    if (Math.abs(twiceArea) < 1e-14) continue;
    const candidate = {area: Math.abs(twiceArea), coordinates: [originX + longitude / (3 * twiceArea), originY + latitude / (3 * twiceArea)]};
    if (!best || candidate.area > best.area) best = candidate;
  }
  if (best) return best.coordinates;
  const bounds = featureBounds([{geometry}]);
  return bounds ? [(bounds[0] + bounds[2]) / 2, (bounds[1] + bounds[3]) / 2] : null;
}

function observationGeometry(feature) {
  const geometries = Array.isArray(feature.properties?.evidence_geometries) ? feature.properties.evidence_geometries : [];
  return geometries.find(item => item?.properties?.role === "visual_detection")?.geometry || geometries.find(item => item?.geometry)?.geometry || feature.geometry;
}

function executionMode(data) {
  if (data?.execution_mode) return data.execution_mode;
  if (data?.mode) return data.mode;
  if (data?.result_kind === "tile_retrieval") return "explore";
  return "catalog_scan";
}

function resultProfile(data, feature = null) {
  const kind = String(feature?.properties?.evidence_kind || data?.evidence_kind || "").toLowerCase();
  const resultKind = String(data?.result_kind || "").toLowerCase();
  if (resultKind === "place_lookup" || kind === "mapped_place" || data?.route === "places") return "place";
  if (resultKind === "tile_retrieval" || executionMode(data) === "explore" || ["image_tile", "image_similarity", "tile_retrieval"].includes(kind)) return "scene";
  if (data?.route === "vectors" || ["vector_features", "mapped_features", "mapped_records"].includes(resultKind) || kind === "mapped_feature") return "mapped";
  if (executionMode(data) === "ground" || kind === "live_model_detection") return "live";
  return "cached";
}

function resultItems(data) {
  return resultProfile(data) === "scene"
    ? (Array.isArray(data.tile_candidates) ? data.tile_candidates : [])
    : (Array.isArray(data.features) ? data.features : []);
}

function conceptName(data, feature) {
  const properties = feature?.properties || {};
  return properties.name || properties.display_name || properties.visual_concept || properties.concept_id || data?.plan?.visual_query || data?.plan?.concept_id || null;
}

function fitBounds(bounds, focused = false) {
  if (!state.map || !bounds) return;
  const size = state.map.getContainer().getBoundingClientRect();
  const padding = Math.max(20, Math.min(65, size.width * .08, size.height * .08));
  state.map.fitBounds([[bounds[0], bounds[1]], [bounds[2], bounds[3]]], {padding, maxZoom: focused ? 19 : 17, duration: 750});
}

function sourceData(id, features) {
  state.map?.getSource(id)?.setData(collection(features));
}

function mapId(value) {
  return String(value).replace(/[^a-zA-Z0-9_-]/g, "-");
}

function areaLabel(value) {
  const label = String(value || "");
  if (/\brio\b/i.test(label)) return "RIO";
  if (/\bsacramento\b/i.test(label)) return "SACRAMENTO";
  if (/\bphoenix\b/i.test(label)) return "PHOENIX";
  return label || "STUDY AREA";
}

function datasetDetail(datasetId = state.datasetId) {
  return state.datasets.find(dataset => dataset.id === datasetId) || null;
}

function renderDatasetIdentity(datasetId, areaName = null) {
  const detail = datasetDetail(datasetId);
  $("datasetName").textContent = detail?.source || detail?.name || datasetId || "LOCAL EVIDENCE";
  if (areaName || detail?.name) $("areaName").textContent = areaLabel(areaName || detail.name);
}

function setImageryFocus(active) {
  for (const id of state.imageryLayers) {
    if (state.map?.getLayer(id)) state.map.setPaintProperty(id, "raster-opacity", .95);
  }
}

function destroyMap() {
  if (state.map) state.map.remove();
  state.map = null;
  state.mapData = null;
  state.areaBounds = null;
  state.imageryLayers = [];
  state.vectorLayers = [];
}

function addPolygonLayers(sourceId, prefix, colors, options = {}) {
  state.map.addSource(sourceId, {type: "geojson", data: collection([])});
  state.map.addLayer({
    id: `${prefix}-fill`, type: "fill", source: sourceId,
    paint: {"fill-color": colors.fill, "fill-opacity": options.opacity ?? .18},
  });
  state.map.addLayer({id: `${prefix}-point`, type: "circle", source: sourceId,
    filter: ["in", ["geometry-type"], ["literal", ["Point", "MultiPoint"]]],
    paint: {"circle-color": colors.fill, "circle-radius": 6, "circle-stroke-color": colors.line, "circle-stroke-width": 2}});
  state.map.addLayer({
    id: `${prefix}-line`, type: "line", source: sourceId,
    paint: {"line-color": colors.line, "line-width": options.width ?? 1.8, "line-opacity": options.lineOpacity ?? .9,
      ...(options.dash ? {"line-dasharray": options.dash} : {})},
  });
}

function addResultSources() {
  addPolygonLayers("context-results", "context-results", {fill: "#508d94", line: "#c2efef"}, {opacity: .09, width: 2, lineOpacity: .76});
  addPolygonLayers("scene-results", "scene-results", {fill: "#317582", line: "#a2f1e6"}, {opacity: .1, width: 2});
  addPolygonLayers("linked-results", "linked-results", {fill: "#489968", line: "#b6f5bd"}, {opacity: .23, width: 2.2});
  addPolygonLayers("linked-observations", "linked-observations", {fill: "#8de9bc", line: "#e5f9d7"}, {opacity: .06, width: 1.5, dash: [2, 1.6]});
  addPolygonLayers("review-results", "review-results", {fill: "#e4b361", line: "#ffc875"}, {opacity: .13, width: 1.7, dash: [2.4, 1.6]});
  addPolygonLayers("selected-result", "selected-result", {fill: "#fcfbeb", line: "#fcfbeb"}, {opacity: .1, width: 3});

  for (const layerId of ["scene-results-fill", "linked-results-fill", "review-results-fill", "scene-results-point", "linked-results-point", "review-results-point"]) {
    state.map.on("click", layerId, event => {
      const id = event.features?.[0]?.properties?._ui_id;
      if (id) selectRecord(id, {moveMap: false});
    });
    state.map.on("mouseenter", layerId, () => { state.map.getCanvas().style.cursor = "pointer"; });
    state.map.on("mouseleave", layerId, () => { state.map.getCanvas().style.cursor = ""; });
  }
}

async function loadMap(datasetId) {
  if (!window.maplibregl) throw new Error("The local map renderer is unavailable.");
  const data = await requestJson(`/api/map?dataset_id=${encodeURIComponent(datasetId)}&tile_limit=64&vector_limit=5000`);
  $("mapLoading").hidden = false;
  destroyMap();
  state.mapData = data;
  state.areaBounds = featureBounds(data.tiles || []) || (Array.isArray(data.coverage?.bbox_wgs84) ? data.coverage.bbox_wgs84 : null);

  const attribution = [...new Set(Object.values(data.attribution || {}).filter(value => typeof value === "string"))].join(" · ");
  const map = state.map = new maplibregl.Map({
    container: "map",
    center: [73.8567, 18.5204],
    zoom: 5,
    attributionControl: false,
    style: {version: 8, sources: {}, layers: [{id: "background", type: "background", paint: {"background-color": "#e5e9e1"}}]},
  });
  map.addControl(new maplibregl.NavigationControl({showCompass: false, visualizePitch: false}), "bottom-right");
  map.addControl(new maplibregl.ScaleControl({unit: "metric", maxWidth: 90}), "bottom-left");
  map.addControl(new maplibregl.AttributionControl({compact: true, customAttribution: attribution}), "bottom-right");
  await new Promise(resolve => map.once("load", resolve));

  map.addSource("satellite-basemap", {type: "raster", tiles: ["https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"], tileSize: 256, maxzoom: 19,
    attribution: "Sources: Esri, Maxar, Earthstar Geographics, GIS User Community"});
  map.addLayer({id: "satellite-basemap", type: "raster", source: "satellite-basemap", paint: {"raster-fade-duration": 150}});
  map.addSource("street-basemap", {type: "raster", tiles: ["https://tile.openstreetmap.org/{z}/{x}/{y}.png"], tileSize: 256, maxzoom: 19,
    attribution: "© OpenStreetMap contributors"});
  map.addLayer({id: "street-basemap", type: "raster", source: "street-basemap", layout: {visibility: "none"}, paint: {"raster-fade-duration": 150}});
  setBasemap(state.basemapMode);

  for (const [index, tile] of (data.tiles || []).entries()) {
    if (!Array.isArray(tile.coordinates) || tile.coordinates.length !== 4) continue;
    const id = `imagery-${mapId(tile.tile_id || index)}`;
    map.addSource(id, {type: "image", url: tile.asset_url, coordinates: tile.coordinates});
    map.addLayer({
      id, type: "raster", source: id,
      paint: {"raster-opacity": .95, "raster-saturation": -.08, "raster-contrast": .03, "raster-brightness-min": .02, "raster-brightness-max": 1, "raster-fade-duration": 0},
    });
    state.imageryLayers.push(id);
  }

  for (const [layerName, layer] of Object.entries(data.vector_layers || {})) {
    const sourceId = `reference-${mapId(layerName)}`;
    map.addSource(sourceId, {type: "geojson", data: layer});
    map.addLayer({id: `${sourceId}-fill`, type: "fill", source: sourceId,
      paint: {"fill-color": "#4c929a", "fill-opacity": .06}});
    map.addLayer({id: `${sourceId}-line`, type: "line", source: sourceId,
      paint: {"line-color": "#a6e4e9", "line-width": 1, "line-opacity": .58}});
    state.vectorLayers.push(`${sourceId}-fill`, `${sourceId}-line`);
  }

  map.addSource("study-boundary", {type: "geojson", data: collection((data.tiles || []).filter(tile => tile.geometry).map(tile => ({type: "Feature", geometry: tile.geometry, properties: {tile_id: tile.tile_id}})))});
  map.addLayer({id: "study-boundary", type: "line", source: "study-boundary", paint: {"line-color": "#f6eabf", "line-width": 2, "line-opacity": .85, "line-dasharray": [3, 2]}});

  addResultSources();
  applyLayerVisibility();
  const updateCoordinates = position => { $("mapCoordinates").textContent = `${Math.abs(position.lat).toFixed(5)}° ${position.lat < 0 ? "S" : "N"}  /  ${Math.abs(position.lng).toFixed(5)}° ${position.lng < 0 ? "W" : "E"}`; };
  updateCoordinates(map.getCenter());
  map.on("moveend", () => updateCoordinates(map.getCenter()));
  map.on("mousemove", event => updateCoordinates(event.lngLat));
  map.once("idle", () => { $("mapLoading").hidden = true; });
  map.on("sourcedata", event => { if (event.sourceId?.startsWith("imagery-") && event.isSourceLoaded) $("mapLoading").hidden = true; });
  $("fitArea").disabled = false;
  const tileCount = data.counts?.tiles?.available ?? data.tiles?.length ?? 0;
  const footprintCount = data.counts?.vector_layers?.buildings?.available;
  $("mapMeta").textContent = `WORLDWIDE PLACES · ${tileCount} ANALYSIS ${tileCount === 1 ? "TILE" : "TILES"} AVAILABLE`;
  $("mapScope").textContent = "Place search is worldwide. Visual analysis uses prepared imagery coverage.";
  renderDatasetIdentity(datasetId, data.area_name);
}

async function loadCapabilities(datasetId) {
  const data = await requestJson(`/api/capabilities?dataset_id=${encodeURIComponent(datasetId)}`);
  state.capabilities = data;
  renderExamples(data);
  state.assistant = data.conversation || state.assistant;
  const configured = state.assistant?.configured;
  const provider = state.assistant?.provider || "groq";
  const model = state.assistant?.model || "";
  const modelLabel = model ? ` (${model})` : "";
  $("conversationMode").disabled = !configured;
  $("conversationMode").title = configured
    ? `Ask geospatial questions and follow up \u00b7 ${provider.charAt(0).toUpperCase() + provider.slice(1)}${modelLabel} ready`
    : `GIS conversation not configured \u2014 add ${provider.toUpperCase()}_API_KEY to pipeline/.env. Evidence search still available.`;
  if (data.area_name) renderDatasetIdentity(datasetId, data.area_name);
}

async function loadDatasets() {
  const payload = await requestJson("/api/datasets");
  state.datasets = Array.isArray(payload.datasets) ? payload.datasets : [];
  if (!state.datasets.length) throw new Error("No prepared evidence datasets are available.");
  const select = $("datasetSelect");
  select.replaceChildren();
  state.datasets.forEach(dataset => { const option = node("option", dataset.name || dataset.id); option.value = dataset.id; select.append(option); });
  const requested = payload.default_dataset_id;
  return state.datasets.some(dataset => dataset.id === requested) ? requested : state.datasets.find(dataset => dataset.default)?.id || state.datasets[0].id;
}

async function loadDataset(datasetId) {
  if (!datasetId) throw new Error("The selected evidence source is unavailable.");
  if (state.datasetId === datasetId && state.map && state.capabilities) return;
  await Promise.all([loadCapabilities(datasetId), loadMap(datasetId)]);
  state.datasetId = datasetId;
  $("datasetSelect").value = datasetId;
  renderDatasetIdentity(datasetId, state.mapData?.area_name || state.capabilities?.area_name);
  renderSources();
}

function cloneForMap(feature, id, bucket) {
  return {
    ...feature,
    properties: {...(feature.properties || {}), _ui_id: id, _ui_bucket: bucket},
  };
}

function observationGeometries(feature, id) {
  const evidence = Array.isArray(feature.properties?.evidence_geometries) ? feature.properties.evidence_geometries : [];
  return evidence.filter(item => item?.geometry).map(item => cloneForMap(item, id, "observation"));
}

function normalizeResults(data) {
  const profile = resultProfile(data);
  const linked = resultItems(data);
  const review = Array.isArray(data.uncertain_candidates) ? data.uncertain_candidates : [];
  state.records = [
    ...linked.map((feature, index) => ({id: `${profile}-${index}`, feature, supported: profile !== "scene", profile: resultProfile(data, feature), index})),
    ...review.map((feature, index) => ({id: `review-${index}`, feature, supported: false, profile: resultProfile(data, feature), index: linked.length + index})),
  ];
  return {linked, review, profile};
}

function displayResults(data, query) {
  const {linked, review, profile} = normalizeResults(data);
  const linkedCount = profile === "scene"
    ? (isCount(data.total_tile_candidates) ? data.total_tile_candidates : linked.length)
    : (isCount(data.total_matches) ? data.total_matches : linked.length);
  const reviewCount = profile === "scene" ? 0 : (isCount(data.total_uncertain) ? data.total_uncertain : review.length);
  const candidateCount = linkedCount + reviewCount;
  const footprintLinks = state.records.filter(record => record.feature.properties?.association === "unique").length;

  $("candidateCount").textContent = String(candidateCount);
  $("candidateLabel").textContent = profile === "place" ? "POSSIBLE PLACES" : profile === "scene" ? "IMAGE SCENES" : profile === "mapped" ? "MAPPED RECORDS" : profile === "live" ? "MODEL OBSERVATIONS" : "AI CANDIDATES";
  $("linkCount").textContent = String(profile === "scene" ? (data.tiles_processed ?? linked.length) : profile === "cached" ? footprintLinks : linkedCount);
  $("linkLabel").textContent = profile === "place" ? "SOURCE RECORDS" : profile === "scene" ? "TILES SEARCHED" : profile === "cached" ? "FOOTPRINT LINKS" : "GEOGRAPHIC MATCHES";
  $("unknownCount").textContent = String(reviewCount);
  $("unknownLabel").textContent = profile === "place" ? "NEED REVIEW" : profile === "scene" ? "OBJECT CLAIMS" : "NEED REVIEW";
  $("resultsTitle").textContent = query;
  const scope = data.claim_scope || "Results reflect returned geographic sources and available imagery; no completeness claim.";
  const returnedCount = linked.length + review.length;
  $("scopeNote").textContent = `${scope}${candidateCount > returnedCount ? ` Showing ${returnedCount} of ${candidateCount} ${profile === "mapped" ? "evaluated" : "returned"} records.` : ""}`;
  $("exportResults").disabled = !data.plan && !state.exportUrl;
  $("resultsEmpty").hidden = true;
  const notice = ["unavailable", "needs_clarification"].includes(data.status);
  $("resultMetrics").hidden = notice;
  $("closeResults").hidden = false;
  $("scopeNote").hidden = false;
  $("discoveryStatus").replaceChildren();
  $("discoveryStatus").hidden = !data.message;
  $("discoveryStatus").className = `discovery-status ${safeClass(data.status || "completed")}`;
  if (data.message) {
    const label = notice ? (data.status === "needs_clarification" ? "A LITTLE MORE DETAIL" : "EVIDENCE AVAILABILITY") : profile === "place" ? "PLACE SEARCH" : profile === "mapped" ? "GEOGRAPHIC SEARCH" : "IMAGERY SEARCH";
    $("discoveryStatus").append(node("span", label, "discovery-status-label"), node("p", data.message));
  }

  const list = $("resultList");
  list.replaceChildren();
  if (!state.records.length) {
    const empty = node("div", "", "empty-results");
    empty.append(node("strong", notice ? (data.status === "needs_clarification" ? "Add a detail to continue" : "This search needs suitable data") : "No matches returned"), node("p", notice ? "Adjust your question or try an example with available evidence." : "This does not establish real-world absence. Review the source coverage or try a different search."));
    if (data.status === "unavailable") {
      const suggestion = node("button", "Try supported imagery in Phoenix", "notice-action");
      suggestion.type = "button";
      suggestion.addEventListener("click", () => { $("query").value = "Find swimming pools within 100 metres of roads in Phoenix"; updateSubmit(); search(); });
      empty.append(suggestion);
    }
    list.append(empty);
  } else {
    state.records.forEach((record, index) => list.append(resultCard(record, index)));
  }

  sourceData("scene-results", profile === "scene" ? linked.map((feature, index) => cloneForMap(feature, `scene-${index}`, "scene")) : []);
  sourceData("linked-results", profile === "scene" ? [] : linked.map((feature, index) => cloneForMap(feature, `${profile}-${index}`, "linked")));
  sourceData("linked-observations", ["cached", "live"].includes(profile) ? linked.flatMap((feature, index) => observationGeometries(feature, `${profile}-${index}`)) : []);
  sourceData("review-results", review.map((feature, index) => cloneForMap(feature, `review-${index}`, "review")));
  sourceData("context-results", Array.isArray(data.context_features) ? data.context_features : []);
  sourceData("selected-result", []);
  setImageryFocus(true);

  $("resultsPanel").hidden = false;
  document.body.classList.add("has-results");
  closeEvidence({restoreFocus: false});
  window.setTimeout(() => state.map?.resize(), 350);
  fitBounds(featureBounds([...linked, ...review]) || featureBounds(data.context_features || []));
  const location = data.interpretation?.location_text || data.interpretation?.location;
  $("mapAreaLabel").textContent = typeof location === "string" ? location : location?.name || (profile === "place" ? "Worldwide place results" : data.area_name || "Geographic results");
  $("mapMeta").textContent = notice ? "SEARCH NEEDS MORE EVIDENCE" : profile === "place" ? `${linkedCount} POSSIBLE PLACE ${linkedCount === 1 ? "MATCH" : "MATCHES"}` : profile === "mapped" ? `${linkedCount} GEOGRAPHIC MATCHES` : `${candidateCount} IMAGERY CANDIDATES`;
  $("mapScope").textContent = profile === "place" ? "Mapped place records · inspect the name, address and source." : "Map imagery is context. Results show the evidence actually searched.";
  if (notice) setStatus(data.status === "needs_clarification" ? "Add the requested detail to continue" : "Imagery or geographic data unavailable for this request");
  else if (profile === "place") setStatus(`${candidateCount} possible place ${candidateCount === 1 ? "match" : "matches"} · inspect their details`);
  else if (profile === "scene") setStatus(`${candidateCount} image scenes · no object claim`);
  else if (profile === "mapped") setStatus(`${candidateCount} mapped records`);
  else if (profile === "live") setStatus(`${candidateCount} bounded model observations`);
  else setStatus(`${candidateCount} AI candidates · ${footprintLinks} footprint ${footprintLinks === 1 ? "link" : "links"}`);
}

function resultCard(record, index) {
  const properties = record.feature.properties || {};
  const profile = record.profile;
  const button = node("button", "", "result-card");
  button.type = "button";
  button.dataset.recordId = record.id;
  button.setAttribute("aria-pressed", "false");
  button.style.animationDelay = `${Math.min(index, 12) * 28}ms`;

  const thumbnail = node("span", "", "result-thumb");
  const asset = safeAssetUrl(properties.asset_url);
  if (asset) {
    const image = document.createElement("img");
    image.src = asset;
    image.alt = "";
    image.loading = "lazy";
    thumbnail.append(image);
  } else if (["place", "mapped"].includes(profile)) {
    thumbnail.classList.add("location-thumb");
    thumbnail.textContent = String(index + 1).padStart(2, "0");
  } else thumbnail.classList.add("no-image");

  const copy = node("span", "", "result-copy");
  const sequence = String(index + 1).padStart(2, "0");
  const concept = conceptName(state.result, record.feature);
  const title = profile === "place" ? (concept || `Possible place ${sequence}`) : profile === "scene" ? `Image scene ${sequence}`
    : profile === "mapped" ? `${concept || pretty(properties.layer || "Mapped record")} ${concept ? "" : sequence}`.trim()
      : `${pretty(concept || "Model observation")} ${sequence}`;
  copy.append(node("span", title, "result-title"));
  const subtitle = profile === "place" ? (properties.display_name || properties.address || "Mapped location · inspect identity") : profile === "scene"
    ? (finite(properties.retrieval_similarity) ? `Whole-image similarity ${Number(properties.retrieval_similarity).toFixed(3)}` : "Whole-image scene candidate")
    : profile === "mapped" ? `Mapped source record${properties.layer ? ` · ${pretty(properties.layer)}` : ""}`
      : `${finite(properties.visual_score) ? `Model score ${Number(properties.visual_score).toFixed(3)}` : "Model score unavailable"}${properties.association === "unique" ? " · footprint linked" : ""}`;
  copy.append(node("span", subtitle, "result-subtitle"));
  const badgeText = profile === "place" ? "POSSIBLE MATCH" : profile === "scene" ? "SCENE ONLY" : profile === "mapped" ? "MAPPED" : record.supported ? "SUPPORTED" : "REVIEW";
  const badge = node("span", badgeText, `result-badge${record.supported || ["mapped", "place"].includes(profile) ? " linked" : ""}${profile === "scene" ? " scene" : ""}`);
  button.append(thumbnail, copy, badge);
  button.addEventListener("click", () => selectRecord(record.id));
  return button;
}

function addDefinition(list, label, value) {
  if (value === undefined || value === null || value === "") return;
  list.append(node("dt", label), node("dd", value));
}

function overlayPath(geometry, tile) {
  const corners = Array.isArray(tile?.coordinates) ? tile.coordinates : [];
  if (corners.length !== 4) return "";
  const west = Math.min(...corners.map(point => Number(point[0])));
  const east = Math.max(...corners.map(point => Number(point[0])));
  const south = Math.min(...corners.map(point => Number(point[1])));
  const north = Math.max(...corners.map(point => Number(point[1])));
  if (!(east > west && north > south)) return "";
  return outerRings(geometry).map(ring => ring.map((point, index) => {
    const x = (Number(point[0]) - west) / (east - west) * 100;
    const y = (north - Number(point[1])) / (north - south) * 100;
    return `${index ? "L" : "M"}${x.toFixed(4)},${y.toFixed(4)}`;
  }).join(" ") + " Z").join(" ");
}

function footprintOverlay(feature, properties) {
  if (properties.association !== "unique") return null;
  const tile = (state.mapData?.tiles || []).find(item => item.tile_id === properties.tile_id);
  const pathData = overlayPath(feature.geometry, tile);
  if (!pathData) return null;
  const namespace = "http://www.w3.org/2000/svg";
  const svg = document.createElementNS(namespace, "svg");
  svg.classList.add("footprint-overlay");
  svg.setAttribute("viewBox", "0 0 100 100");
  svg.setAttribute("preserveAspectRatio", "none");
  svg.setAttribute("aria-hidden", "true");
  const path = document.createElementNS(namespace, "path");
  path.setAttribute("d", pathData);
  path.setAttribute("fill-rule", "evenodd");
  svg.append(path);
  return svg;
}

function evidenceVisual(record) {
  const feature = record.feature;
  const properties = feature.properties || {};
  const asset = safeAssetUrl(properties.asset_url);
  if (!asset) return node("div", "", "evidence-placeholder");
  const tile = (state.mapData?.tiles || []).find(item => item.tile_id === properties.tile_id);
  const width = Number(properties.original_width || properties.width || tile?.width);
  const height = Number(properties.original_height || properties.height || tile?.height);
  const visual = node("div", "", "evidence-visual");
  if (width > 0 && height > 0) visual.style.aspectRatio = `${width} / ${height}`;
  const image = document.createElement("img");
  image.className = "evidence-image";
  image.src = asset;
  image.alt = record.profile === "scene" ? "Source tile for the selected image scene." : "Source tile for the selected model observation.";
  visual.append(image);
  const footprint = ["cached", "live"].includes(record.profile) ? footprintOverlay(feature, properties) : null;
  if (footprint) visual.append(footprint);
  const box = properties.pixel_box;
  if (["cached", "live"].includes(record.profile) && Array.isArray(box) && box.length === 4 && width > 0 && height > 0) {
    const marker = node("span", "", `detection-box${properties.association === "unique" ? " linked" : ""}`);
    const left = Math.max(0, Math.min(100, Number(box[0]) / width * 100));
    const top = Math.max(0, Math.min(100, Number(box[1]) / height * 100));
    marker.style.left = `${left}%`;
    marker.style.top = `${top}%`;
    marker.style.width = `${Math.max(1, Math.min(100 - left, (Number(box[2]) - Number(box[0])) / width * 100))}%`;
    marker.style.height = `${Math.max(1, Math.min(100 - top, (Number(box[3]) - Number(box[1])) / height * 100))}%`;
    visual.append(marker);
  }
  return visual;
}

function conditionCopy(condition) {
  if (condition.condition === "visual_detection") {
    const score = finite(condition.score) ? `Score ${Number(condition.score).toFixed(3)}` : "Score unavailable";
    const threshold = finite(condition.screening_threshold) ? ` · preparation threshold ${Number(condition.screening_threshold).toFixed(3)}` : "";
    return `${score}${threshold}`;
  }
  if (condition.condition === "feature_association") {
    return condition.association === "unique"
      ? "Uniquely associated with one supplied reference footprint."
      : "No unique supplied footprint association was established.";
  }
  if (finite(condition.distance_m) || finite(condition.radius_m)) {
    const witness = finite(condition.distance_m) ? `${Number(condition.distance_m).toFixed(2)} m measured` : "Witness distance unavailable";
    const radius = finite(condition.radius_m) ? ` · ${Number(condition.radius_m).toFixed(1)} m requested radius` : "";
    const verificationWitness = condition.verification?.witnesses?.[0];
    const interval = verificationWitness?.distance_interval_m || condition.reference_candidates?.[0]?.distance_interval_m;
    const range = Array.isArray(interval) && interval.length === 2 && interval.every(finite)
      ? ` · verified interval ${Number(interval[0]).toFixed(2)}–${Number(interval[1]).toFixed(2)} m` : "";
    return `${witness}${radius}${range}`;
  }
  return condition.reason || condition.note || pretty(condition.status);
}

function renderEvidence(record) {
  const feature = record.feature;
  const properties = feature.properties || {};
  const profile = record.profile;
  const root = $("evidenceBody");
  const associationLinked = properties.association === "unique";
  root.replaceChildren();
  if (["scene", "cached", "live"].includes(profile)) root.append(evidenceVisual(record));
  const label = profile === "place" ? "GEOGRAPHIC PLACE RECORD" : profile === "scene" ? "WHOLE-IMAGE SCENE"
    : profile === "mapped" ? "MAPPED SOURCE RECORD"
      : profile === "live" ? "BOUNDED MODEL OBSERVATION"
        : "PRECOMPUTED MODEL CANDIDATE";
  root.append(node("p", label, "evidence-label"));
  const concept = conceptName(state.result, feature);
  const titleText = profile === "place" ? (concept || "Possible place match") : profile === "scene" ? `Image scene ${String(record.index + 1).padStart(2, "0")}`
    : profile === "mapped" ? pretty(concept || properties.layer || "Mapped record")
      : pretty(concept || "Model observation");
  const title = node("h2", titleText);
  title.id = "evidenceTitle";
  root.append(title);
  const summary = profile === "place" ? "A possible match from a geographic place index. Check its name, address and source to confirm this is the place you meant." : profile === "scene"
    ? "This candidate ranks an entire source image. It does not locate or verify an object."
    : profile === "mapped" ? "This geometry comes from geographic source records. Any distance or intersection checks are shown below."
      : associationLinked
        ? `The model observation is linked to one supplied reference footprint${record.supported ? " and passes the returned evidence checks" : ", with another check still unresolved"}.`
        : record.supported ? "The bounded model observation passes the returned evidence checks in this sample."
          : "The model observation remains unresolved and needs review.";
  root.append(node("p", summary, "evidence-summary"));
  const statusText = profile === "place" ? "POSSIBLE PLACE MATCH" : profile === "scene" ? "SCENE CANDIDATE" : profile === "mapped" ? "MAPPED RECORD" : record.supported ? "SUPPORTED IN SAMPLE" : "NEEDS REVIEW";
  root.append(node("span", statusText, `evidence-status${record.supported || profile === "mapped" ? " linked" : ""}${profile === "scene" ? " scene" : ""}`));

  const details = document.createElement("dl");
  details.className = "evidence-grid";
  const geographic = ["cached", "live"].includes(profile) ? observationGeometry(feature) : feature.geometry;
  const bounds = featureBounds([{geometry: geographic}]);
  const centroid = centroidOfGeometry(geographic);
  addDefinition(details, "Evidence", profile === "place" ? "Mapped place identity and location" : profile === "scene" ? "Whole-image similarity"
    : profile === "mapped" ? "Supplied mapped feature"
      : profile === "live" ? "On-demand OWLv2 observation" : "Precomputed OWLv2 observation");
  addDefinition(details, "Association", ["cached", "live"].includes(profile) ? (associationLinked ? "Unique reference footprint link" : properties.association ? pretty(properties.association) : "Not requested") : null);
  addDefinition(details, "Model score", ["cached", "live"].includes(profile) && finite(properties.visual_score) ? `${Number(properties.visual_score).toFixed(3)} · uncalibrated` : null);
  addDefinition(details, "Image similarity", profile === "scene" && finite(properties.retrieval_similarity) ? `${Number(properties.retrieval_similarity).toFixed(3)} · whole image` : null);
  addDefinition(details, "Pixel box", ["cached", "live"].includes(profile) && Array.isArray(properties.pixel_box) ? `[${properties.pixel_box.map(value => Number(value).toFixed(1)).join(", ")}] px` : null);
  const suppliedBounds = Array.isArray(properties.bbox) && properties.bbox.length === 4 ? properties.bbox.map(Number) : bounds;
  addDefinition(details, profile === "scene" ? "Scene bounds" : "Geographic bbox", suppliedBounds?.every(Number.isFinite) ? `W ${suppliedBounds[0].toFixed(6)} · S ${suppliedBounds[1].toFixed(6)} · E ${suppliedBounds[2].toFixed(6)} · N ${suppliedBounds[3].toFixed(6)}` : null);
  addDefinition(details, profile === "place" ? "Coordinates" : "Centroid", centroid ? `${centroid[1].toFixed(6)} lat · ${centroid[0].toFixed(6)} lon` : null);
  addDefinition(details, "Address", profile === "place" ? properties.display_name || properties.address : null);
  addDefinition(details, "Source", properties.source || properties.source_name);
  addDefinition(details, "Retrieved", properties.retrieved_at);
  addDefinition(details, "Footprint", associationLinked ? (properties.id || feature.id) : null);
  addDefinition(details, "Layer", profile === "mapped" ? (properties.subject_layer || properties.layer || properties.source_layer) : null);
  addDefinition(details, "Source tile", properties.tile_id || properties.tile_ids?.[0]);
  addDefinition(details, "Image date", ["cached", "live", "scene"].includes(profile) ? properties.imagery_date || "Unknown" : null);
  addDefinition(details, "Model revision", ["cached", "live"].includes(profile) && properties.model_revision ? String(properties.model_revision).slice(0, 12) : null);
  addDefinition(details, "Evidence receipt", properties.evidence_receipt?.id || properties.receipt_id);
  root.append(details);
  const sourceUrl = safeSourceUrl(properties.source_url);
  if (sourceUrl) {
    const link = node("a", "Open geographic source", "source-link");
    link.href = sourceUrl; link.target = "_blank"; link.rel = "noopener noreferrer";
    root.append(link);
  }

  const conditions = Array.isArray(properties.conditions) ? properties.conditions : [];
  if (conditions.length) {
    const section = node("section", "", "conditions");
    section.append(node("h3", "EVIDENCE CHECKS"));
    for (const condition of conditions) {
      const card = node("div", "", `condition ${safeClass(condition.status)}`);
      card.append(node("strong", `${pretty(condition.condition)} · ${pretty(condition.status)}`));
      card.append(node("span", conditionCopy(condition)));
      section.append(card);
    }
    root.append(section);
  }
  const disclaimer = profile === "place" ? "This is a mapped identity candidate. A matching place name is not an AI detection or proof that the institution is currently operating." : profile === "scene"
    ? "Whole-image similarity retrieves scenes only. It does not establish that a requested object exists or identify its location."
    : profile === "mapped" ? "Mapped records describe the supplied snapshot. Missing records do not establish real-world absence or complete coverage."
      : "Scores are uncalibrated model outputs. These observations and any footprint links describe a bounded evidence sample, not a verified object count or complete real-world coverage.";
  root.append(node("p", disclaimer, "evidence-disclaimer"));
}

function selectRecord(id, {moveMap = true} = {}) {
  const record = state.records.find(item => item.id === id);
  if (!record) return;
  state.selectedId = id;
  document.querySelectorAll(".result-card").forEach(button => button.setAttribute("aria-pressed", String(button.dataset.recordId === id)));
  const selected = [cloneForMap(record.feature, id, "selected"), ...observationGeometries(record.feature, id)];
  sourceData("selected-result", selected);
  renderEvidence(record);
  $("evidencePanel").hidden = false;
  document.body.classList.add("detail-open");
  const overlay = window.matchMedia("(max-width: 1120px)").matches;
  $("evidencePanel").setAttribute("aria-modal", String(overlay));
  $("resultsPanel").inert = overlay;
  $("searchCluster").inert = overlay;
  document.querySelector(".topbar").inert = overlay;
  document.querySelector(".map-toolbar").inert = overlay;
  document.querySelector(".app-rail").inert = overlay;
  $("map").inert = overlay;
  $("evidencePanel").focus({preventScroll: true});
  if (moveMap) fitBounds(featureBounds(selected), true);
}

function closeEvidence({restoreFocus = true} = {}) {
  const selected = state.selectedId;
  state.selectedId = null;
  sourceData("selected-result", []);
  $("evidencePanel").hidden = true;
  document.body.classList.remove("detail-open");
  $("evidencePanel").setAttribute("aria-modal", "false");
  $("resultsPanel").inert = false;
  $("searchCluster").inert = false;
  document.querySelector(".topbar").inert = false;
  document.querySelector(".map-toolbar").inert = false;
  document.querySelector(".app-rail").inert = false;
  $("map").inert = false;
  document.querySelectorAll(".result-card").forEach(button => button.setAttribute("aria-pressed", "false"));
  if (restoreFocus && selected) document.querySelector(`[data-record-id="${CSS.escape(selected)}"]`)?.focus();
}

function closeResults() {
  closeEvidence({restoreFocus: false});
  $("resultsPanel").hidden = false;
  $("resultsEmpty").hidden = false;
  $("resultMetrics").hidden = true;
  $("closeResults").hidden = true;
  $("scopeNote").hidden = true;
  $("discoveryStatus").hidden = true;
  $("resultsTitle").textContent = "Evidence, not guesses.";
  $("resultList").replaceChildren();
  $("exportResults").disabled = true;
  $("exampleSection").hidden = false;
  document.body.classList.remove("has-results");
  for (const id of ["scene-results", "linked-results", "linked-observations", "review-results", "context-results", "selected-result"]) sourceData(id, []);
  setImageryFocus(false);
  state.records = [];
  state.result = null;
  state.exportUrl = null;
  setStatus("Ready to search places and geographic patterns");
  window.setTimeout(() => state.map?.resize(), 350);
  $("query").focus();
}

async function search() {
  const query = $("query").value.trim();
  if (!state.ready || state.busy || query.length < 3) return;
  state.searchController?.abort();
  const controller = new AbortController();
  state.searchController = controller;
  setBusy(true);
  startBusyStatus();
  closeEvidence({restoreFocus: false});
  appendMessage("user", query);
  $("exampleSection").hidden = true;
  try {
    const radiusFollowup = /^(?:(?:and\s+)?(?:within|under|less than)(?:\b|(?=[\d.+-]))|(?:make|change|set|increase|decrease|use|try)\s+(?:(?:it|this|that)\b|(?:the\s+)?(?:radius|distance)(?:\b|(?=[\d.+-]))))/i.test(query);
    const localResult = state.result?.dataset_id && state.datasets.some(dataset => dataset.id === state.result.dataset_id);
    if (radiusFollowup && state.result && !localResult) {
      stopBusyStatus();
      const clarification = {
        type: "FeatureCollection", status: "needs_clarification", route: state.result.route,
        result_kind: "coverage_notice", features: [], uncertain_candidates: [], context_features: [],
        message: "Please restate the full question with its place and new distance. This keeps your search in the intended area.",
        claim_scope: "No new geographic search was executed.",
      };
      state.result = clarification; state.exportUrl = null;
      displayResults(clarification, query);
      appendMessage("assistant", clarification.message);
      return;
    }
    const followup = radiusFollowup && localResult && state.result?.plan;
    const body = {query, dataset_id: state.datasetId};
    if (followup) body.previous_plan = state.result.plan;
    const data = await requestJson("/api/discover", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(body),
      signal: controller.signal,
    });
    stopBusyStatus();
    const resultDataset = data.dataset_id || state.datasetId;
    if (resultDataset !== state.datasetId) {
      setStatus("Loading the selected evidence source");
      await loadDataset(resultDataset);
    }
    state.result = data;
    state.exportUrl = data.export_url || null;
    displayResults(data, query);
    appendMessage("assistant", resultSummary(data));
    if (!["needs_clarification", "unavailable"].includes(data.status)) $("query").value = "";
  } catch (error) {
    stopBusyStatus();
    if (error.name !== "AbortError") { setStatus(error.message, "error"); appendMessage("assistant", error.message, "message-error"); }
  } finally {
    if (state.searchController === controller) state.searchController = null;
    setBusy(false);
  }
}

async function exportResults() {
  const result = state.result;
  if ((!result?.plan && !state.exportUrl) || $("exportResults").disabled) return;
  $("exportResults").disabled = true;
  setStatus("Preparing complete GeoJSON");
  try {
    const body = {
      plan: result.plan,
      dataset_version: result.dataset_version,
      dataset_id: result.dataset_id || state.datasetId,
      mode: executionMode(result),
      include_uncertain: true,
      page: 0,
      page_size: 100,
    };
    if (result.grounding_run_id) body.grounding_run_id = result.grounding_run_id;
    const response = state.exportUrl ? await fetch(state.exportUrl) : await fetch("/api/export", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(body),
    });
    if (!response.ok) {
      const data = await response.json().catch(() => ({}));
      const detail = typeof data.detail === "string" ? data.detail
        : Array.isArray(data.detail) ? data.detail.slice(0, 2).map(item => item.msg || "Invalid export field").join("; ") : null;
      throw new Error(detail || `Export failed (${response.status}).`);
    }
    const blob = await response.blob();
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    const exportArea = result.dataset_id || result.interpretation?.location_text || result.location?.name || result.route || "geographic";
    const datasetSlug = String(exportArea).replace(/[^a-z0-9_-]+/gi, "-").toLowerCase().slice(0, 80);
    anchor.download = `geoverify-x-${datasetSlug}-evidence.geojson`;
    document.body.append(anchor);
    anchor.click();
    anchor.remove();
    window.setTimeout(() => URL.revokeObjectURL(url), 1000);
    setStatus("GeoJSON downloaded");
  } catch (error) {
    setStatus(error.message, "error");
  } finally {
    $("exportResults").disabled = !state.result?.plan && !state.exportUrl;
  }
}

function applyLayerVisibility() {
  for (const [ids, visible] of [[state.imageryLayers, $("imageryToggle").checked], [state.vectorLayers, $("vectorsToggle").checked]]) {
    for (const id of ids) if (state.map?.getLayer(id)) state.map.setLayoutProperty(id, "visibility", visible ? "visible" : "none");
  }
}

function setBasemap(mode) {
  state.basemapMode = mode === "street" ? "street" : "satellite";
  for (const type of ["satellite", "street"]) {
    if (state.map?.getLayer(`${type}-basemap`)) state.map.setLayoutProperty(`${type}-basemap`, "visibility", type === state.basemapMode ? "visible" : "none");
    const button = $(`${type}Basemap`);
    button.classList.toggle("active", type === state.basemapMode);
    button.setAttribute("aria-pressed", String(type === state.basemapMode));
  }
}

function toggleExpandedMap(force) {
  const expanded = typeof force === "boolean" ? force : !document.body.classList.contains("map-expanded");
  document.body.classList.toggle("map-expanded", expanded);
  $("expandMap").setAttribute("aria-pressed", String(expanded));
  $("expandMap").setAttribute("aria-label", expanded ? "Exit expanded map" : "Expand map");
  $("expandMap").title = expanded ? "Exit expanded map (Escape)" : "Expand map";
  window.requestAnimationFrame(() => state.map?.resize());
  window.setTimeout(() => state.map?.resize(), 300);
}

function renderExamples() {
  const choices = [
    {query: "Find Sinhagad College in Pune", hint: "Worldwide places"},
    {query: "Find buildings within 300 metres of a river in Pune", hint: "Geographic relationships"},
    {query: "Find rooftop solar panels within 500 metres of Pune Municipal Corporation in Pune", hint: "Checks imagery availability"},
    {query: "Find swimming pools within 100 metres of roads in Phoenix", hint: "Supported visual search"},
    {query: "Find satellite imagery of densely built neighbourhoods in Phoenix", hint: "Image scene discovery"},
  ];
  const root = $("exampleList");
  root.replaceChildren();
  for (const task of choices) {
    const button = node("button", "", "example-query");
    button.type = "button";
    const copy = node("span", "", "example-copy");
    copy.append(node("span", task.query), node("span", task.hint, "example-hint"));
    button.append(copy, node("span", "→", "example-arrow"));
    button.addEventListener("click", () => { $("query").value = task.query; updateSubmit(); search(); });
    root.append(button);
  }
  $("query").placeholder = "Find Sinhagad College in Pune";
}

function appendMessage(role, message, className = "") {
  const article = node("article", "", `conversation-message chat-message ${role} message-${role}${className ? ` ${className}${className === "message-error" ? " error" : ""}` : ""}`);
  article.append(node("span", role === "user" ? "YOU" : "GEOVERIFY-X", "message-author message-role"), node("p", message));
  const root = $("conversationThread");
  root.append(article);
  root.scrollTop = root.scrollHeight;
  return article;
}

function resultSummary(data) {
  if (data.message) return data.message;
  const profile = resultProfile(data);
  const count = isCount(data.total_matches) ? data.total_matches : resultItems(data).length;
  const review = isCount(data.total_uncertain) ? data.total_uncertain : (data.uncertain_candidates || []).length;
  if (profile === "scene") {
    const scenes = isCount(data.total_tile_candidates) ? data.total_tile_candidates : resultItems(data).length;
    return `${scenes} image ${scenes === 1 ? "scene" : "scenes"} ranked by whole-image similarity. Select a scene to inspect the imagery. This does not locate or verify individual objects.`;
  }
  if (profile === "place") return `${count} possible place ${count === 1 ? "match" : "matches"}. Inspect the name, address and geographic source to choose the location you meant.`;
  if (profile === "mapped") return `${count} mapped ${count === 1 ? "record" : "records"} matched the search${review ? `; ${review} need review` : ""}. These findings describe the supplied vector snapshot. Select one to see its coordinates and source.`;
  return `${count} model ${count === 1 ? "candidate passes" : "candidates pass"} the returned evidence checks${review ? `; ${review} need review` : ""}. Select a finding to inspect the source image, geographic box and spatial checks. This is a sample, not a complete object inventory.`;
}

function setMode(mode) {
  if (state.busy || (mode === "conversation" && !state.assistant?.configured)) return;
  state.mode = mode;
  for (const [id, selected] of [["expertMode", mode === "search"], ["conversationMode", mode === "conversation"]]) {
    $(id).classList.toggle("active", selected);
    $(id).setAttribute("aria-pressed", String(selected));
  }
  $("composerHint").textContent = mode === "conversation" ? "Ask a question or follow up" : "Use a place, object or distance";
  $("query").maxLength = mode === "conversation" ? 2000 : 500;
  setStatus(mode === "conversation" ? "Conversation ready · evidence checks remain visible" : `${state.mapData?.tiles?.length || 0} source tiles ready`);
  $("query").focus();
}

async function submitConversation() {
  const query = $("query").value.trim();
  if (!state.ready || state.busy || query.length < 3) return;
  state.searchController?.abort();
  const controller = new AbortController();
  state.searchController = controller;
  setBusy(true);
  startBusyStatus();
  appendMessage("user", query);
  $("exampleSection").hidden = true;
  try {
    await conversationSearch(query, controller);
    $("query").value = "";
  } catch (error) {
    stopBusyStatus();
    if (error.name !== "AbortError") { setStatus(error.message, "error"); appendMessage("assistant", error.message, "message-error"); }
  } finally {
    if (state.searchController === controller) state.searchController = null;
    setBusy(false);
  }
}

async function conversationSearch(query, controller) {
  const body = {message: query, dataset_id: state.datasetId};
  if (state.conversationId) body.conversation_id = state.conversationId;
  let job = await requestJson("/api/chat", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(body), signal: controller.signal});
  state.conversationId = job.conversation_id;
  stopBusyStatus();
  for (let attempts = 0; attempts < 180; attempts += 1) {
    if (controller.signal.aborted) throw new DOMException("Search interrupted", "AbortError");
    if (["succeeded", "failed"].includes(job.status)) break;
    setStatus(job.stage?.label || "Checking geographic evidence");
    await new Promise(resolve => window.setTimeout(resolve, 1500));
    job = await requestJson(`/api/jobs/${encodeURIComponent(job.job_id)}?conversation_id=${encodeURIComponent(state.conversationId)}`, {signal: controller.signal});
  }
  if (job.status === "failed") throw new Error(job.message || "The conversation workflow could not complete.");
  if (job.status !== "succeeded") throw new Error("The workflow is taking longer than expected. The submitted job may still be running.");
  const response = job.result || {};
  const message = appendMessage("assistant", response.message || "The evidence workflow has completed.");
  const artifacts = Array.isArray(response.artifacts) ? response.artifacts : [];
  const maps = artifacts.filter(artifact => artifact.kind === "map" && artifact.result);
  const showMap = async artifact => {
    const data = artifact.result;
    if (data.dataset_id && data.dataset_id !== state.datasetId) await loadDataset(data.dataset_id);
    state.result = data;
    state.exportUrl = response.export_url || null;
    displayResults(data, artifact.title || query);
  };
  if (maps.length) await showMap(maps[maps.length - 1]);
  for (const artifact of artifacts) {
    const card = node("section", "", "artifact-card");
    card.append(node("strong", artifact.title || "Evidence artifact"));
    if (artifact.kind === "map" && maps.length > 1) {
      const button = node("button", "View on map", "artifact-action"); button.type = "button";
      button.addEventListener("click", () => showMap(artifact).catch(error => setStatus(error.message, "error")));
      card.append(button);
    } else if (artifact.kind === "table" || artifact.kind === "chart") {
      const table = node("table", "", "artifact-table");
      for (const row of (artifact.rows || []).slice(0, 20)) {
        const tr = node("tr");
        Object.values(row).forEach(value => tr.append(node("td", typeof value === "object" ? JSON.stringify(value) : String(value))));
        table.append(tr);
      }
      card.append(table);
    } else continue;
    message.append(card);
  }
  if (response.receipt?.id) message.append(node("span", `Evidence receipt · ${response.receipt.id}`, "message-receipt"));
  if (!maps.length) setStatus(response.status === "completed" ? "Evidence workflow completed" : "More evidence or clarification needed");
  $("conversationThread").scrollTop = $("conversationThread").scrollHeight;
}

function renderSources() {
  const root = $("sourceBody");
  const capabilities = state.capabilities || {};
  const provenance = capabilities.provenance || {};
  const coverage = capabilities.coverage || provenance.coverage || {};
  root.replaceChildren(node("p", datasetDetail()?.name || capabilities.area_name || "Prepared evidence dataset", "source-name"));
  const details = node("dl", "", "evidence-grid");
  addDefinition(details, "Image tiles", state.mapData?.tiles?.length || coverage.tile_count || 0);
  addDefinition(details, "Imagery dates", Array.isArray(coverage.acquisition_dates) && coverage.acquisition_dates.length ? coverage.acquisition_dates.join(", ") : "Unknown in supplied metadata");
  addDefinition(details, "Model observations", Object.keys(capabilities.supported_concepts || {}).map(pretty).join(", ") || "No cached concepts");
  addDefinition(details, "Scene retrieval", capabilities.text_similarity_enabled ? "SkyCLIP text–image retrieval available" : "Unavailable in this library");
  addDefinition(details, "Dataset version", capabilities.dataset_version);
  root.append(details);
  root.append(node("h3", "Reference layers"));
  for (const [id, layer] of Object.entries(capabilities.vector_layers || {})) {
    const item = node("section", "", "source-item");
    item.append(node("strong", pretty(id)), node("p", [layer.provenance?.source_name || layer.source || layer.source_name, layer.source_date || layer.provenance?.source_date, layer.license || layer.provenance?.license].filter(Boolean).join(" · ")));
    if (layer.coverage_note || layer.provenance?.coverage_note) item.append(node("p", layer.coverage_note || layer.provenance.coverage_note));
    root.append(item);
  }
  const attribution = Object.entries(provenance.attribution || state.mapData?.attribution || {}).map(([kind, value]) => `${pretty(kind)}: ${value}`).join("\n");
  if (attribution) root.append(node("p", attribution, "source-attribution"));
  root.append(node("p", "Aerial imagery, annotations and vector snapshots may have different dates. Partial coverage and uncertain detections remain visible; no result establishes real-world absence.", "info-note"));
}

function closeInfo({restoreFocus = true} = {}) {
  $("sourcePanel").hidden = true; $("helpPanel").hidden = true;
  document.querySelector(".workspace").inert = false;
  document.querySelector(".app-rail").inert = false;
  document.body.classList.remove("info-open");
  if (restoreFocus) state.infoTrigger?.focus();
  state.infoTrigger = null;
}

function openInfo(id, trigger) {
  closeInfo({restoreFocus: false}); closeEvidence({restoreFocus: false});
  state.infoTrigger = trigger;
  $(id).hidden = false;
  document.querySelector(".workspace").inert = true;
  document.querySelector(".app-rail").inert = true;
  document.body.classList.add("info-open");
  $(id).focus({preventScroll: true});
}

async function changeDataset() {
  const previous = state.datasetId;
  const next = $("datasetSelect").value;
  if (state.busy || !next || next === previous) return;
  closeResults();
  state.conversationId = null;
  $("conversationThread").replaceChildren();
  $("exampleSection").hidden = false;
  setBusy(true); setStatus("Loading the selected study area");
  try { await loadDataset(next); fitBounds(state.areaBounds); $("mapAreaLabel").textContent = state.mapData?.area_name || "Analysis imagery coverage"; setStatus("Imagery library loaded. Place search remains worldwide."); }
  catch (error) { $("datasetSelect").value = previous; setStatus(error.message, "error"); }
  finally { setBusy(false); }
}

function trapDialogFocus(event) {
  if (event.key !== "Tab" || event.currentTarget.getAttribute("aria-modal") !== "true") return;
  const items = [...event.currentTarget.querySelectorAll('button:not(:disabled), a[href], input:not(:disabled), textarea:not(:disabled), select:not(:disabled), [tabindex="0"]')];
  const first = items[0], last = items[items.length - 1];
  if (!items.length) { event.preventDefault(); return; }
  if (event.shiftKey && (document.activeElement === first || document.activeElement === event.currentTarget)) { event.preventDefault(); last.focus(); }
  else if (!event.shiftKey && (document.activeElement === last || document.activeElement === event.currentTarget)) { event.preventDefault(); first.focus(); }
}

async function start() {
  setStatus("Loading geographic evidence");
  try {
    const defaultDataset = await loadDatasets();
    await loadDataset(defaultDataset);
    state.ready = true;
    $("datasetSelect").disabled = false;
    $("engineStatus").replaceChildren(node("span"), document.createTextNode("Search ready"));
    document.body.classList.add("is-ready");
    $("query").disabled = false;
    setStatus("Ready to search places and geographic patterns");
    updateSubmit();
  } catch (error) {
    state.ready = false;
    $("datasetName").textContent = "EVIDENCE UNAVAILABLE";
    $("areaName").textContent = "NOT READY";
    $("mapLoading").hidden = true;
    $("engineStatus").replaceChildren(node("span"), document.createTextNode("Evidence unavailable"));
    setStatus(error.message, "error");
  }
}

$("query").addEventListener("input", updateSubmit);
$("query").addEventListener("keydown", event => { if (event.key === "Enter" && !event.shiftKey && !event.isComposing) { event.preventDefault(); state.mode === "conversation" ? submitConversation() : search(); } });
$("searchForm").addEventListener("submit", event => { event.preventDefault(); state.mode === "conversation" ? submitConversation() : search(); });
$("fitArea").addEventListener("click", () => fitBounds(state.areaBounds));
$("closeResults").addEventListener("click", closeResults);
$("exportResults").addEventListener("click", exportResults);
$("closeEvidence").addEventListener("click", () => closeEvidence());
$("datasetSelect").addEventListener("change", changeDataset);
$("imageryToggle").addEventListener("change", applyLayerVisibility);
$("vectorsToggle").addEventListener("change", applyLayerVisibility);
$("satelliteBasemap").addEventListener("click", () => setBasemap("satellite"));
$("streetBasemap").addEventListener("click", () => setBasemap("street"));
$("expandMap").addEventListener("click", () => toggleExpandedMap());
$("expertMode").addEventListener("click", () => setMode("search"));
$("conversationMode").addEventListener("click", () => setMode("conversation"));
$("sourcesNav").addEventListener("click", () => openInfo("sourcePanel", $("sourcesNav")));
$("helpNav").addEventListener("click", () => openInfo("helpPanel", $("helpNav")));
$("explorerNav").addEventListener("click", () => { closeInfo(); closeEvidence(); $("query").focus(); });
$("closeSources").addEventListener("click", () => closeInfo());
$("closeHelp").addEventListener("click", () => closeInfo());
for (const id of ["evidencePanel", "sourcePanel", "helpPanel"]) $(id).addEventListener("keydown", trapDialogFocus);
document.addEventListener("keydown", event => {
  if (event.key !== "Escape") return;
  if (!$("sourcePanel").hidden || !$("helpPanel").hidden) closeInfo();
  else if (!$("evidencePanel").hidden) closeEvidence();
  else if (document.body.classList.contains("map-expanded")) { toggleExpandedMap(false); $("expandMap").focus(); }
});
window.addEventListener("resize", () => state.map?.resize());
window.addEventListener("beforeunload", () => state.searchController?.abort());

start();
