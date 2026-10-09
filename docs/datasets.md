# TerraLens dataset audit and prepared prototype corpora

Dataset roles are different: SkyScript supports language–image retrieval, SpaceNet 1 supports building extraction, and NAIP supplies small real aerial scenes paired with bounded OpenStreetMap snapshots. The current launcher selects the sharper Phoenix sample first when its prepared bundle exists. None supplies ground truth for the complete challenge example about commercial roofs, solar panels and open drains.

## Prepared prototype datasets — checked 9 October 2026

The fastest suitable choice is the existing georeferenced imagery and prepared evidence. Both samples already include genuine SkyCLIP image embeddings, OWLv2 model candidates, original raster transforms, image previews, vector geometries and source provenance. Starting the local interface does not require another imagery or model download.

| Sample | Prepared bundle | Real imagery | Vector evidence | Cached model candidates | Maximum distance |
| --- | --- | --- | --- | --- | --- |
| Phoenix / Encanto — launcher default | `pipeline/data/phoenix-demo` | Four overlapping windows from one 1,000 × 1,000 RGB NAIP crop at 0.3 m per pixel; EPSG:26912 | 1,548 OSM buildings and 666 road ways | 102 roof candidates and 25 pool candidates after deduplication; 203 raw candidates | 500 m |
| SpaceNet 1 Rio | `pipeline/data/spacenet-round1` | Three RGB training tiles, each 438 × 406 pixels, approximately 200 m per side; EPSG:4326 | 457 SpaceNet building annotations and 56 OSM road ways | 19 building candidates after deduplication; 25 raw candidates | 50 m |
| Sacramento / Land Park | `pipeline/data/demo` | Four overlapping windows from one 1,000 × 1,000 RGB NAIP crop at 0.6 m per pixel; EPSG:26910 | 2,972 OSM buildings and 1,079 road ways | 148 roof candidates and 4 pool candidates after deduplication; 201 raw candidates | 500 m |

The SpaceNet source tiles are `img1026`, `img1071` and `img1027`, with 163, 153 and 141 building annotations respectively. They were selected for annotation-rich demonstration coverage; they are not a representative or held-out accuracy benchmark. The three tiles do not form continuous municipal coverage. Acquisition dates remain unknown in their downloaded metadata. The Rio OSM road snapshot records `2026-10-09T07:13:19Z`; reference coverage is explicitly partial.

The check opened the existing source GeoTIFFs and prepared JPEG previews, confirmed valid nonempty vector geometry, and found finite normalized embedding matrices of shape `3 × 512` and `4 × 512`. The new Phoenix bundle also passed the application's manifest, asset and embedding validation after running real SkyCLIP and OWLv2 preparation on four windows. The local model weight hashes match the prepared manifests. These are file and provenance checks; independent detector accuracy and image-to-map registration remain unmeasured. SpaceNet annotations remain distinct from model detections, and a large detection box is a coarse candidate rather than an exact individual footprint.

With the existing project environment, start the prototype from the repository root:

```bash
pipeline/.venv/bin/python pipeline/scripts/run_demo.py
```

Open `http://127.0.0.1:8000`. The launcher selects `phoenix-demo` when present, then the available SpaceNet bundle, and finally the Sacramento bundle. `BUNDLE_DIR` and `ENCODER_CHECKPOINT` can override those defaults. All three available areas remain selectable in the interface.

Five suggested Phoenix searches:

1. “Find swimming pools within 100 metres of roads.”
2. “Find building roofs within 50 metres of roads.”
3. “Find mapped buildings within 20 metres of roads.”
4. “Find mapped roads.”
5. “Find satellite imagery of densely built neighbourhoods.”

The first two inspect visual model candidates and spatial evidence. The next two use mapped vector records. The fifth ranks source scenes by text similarity; it does not establish individual object locations. A radius-only follow-up such as “Change the distance to 10 metres” can refine an existing proximity plan. For the Rio sample, keep distance at or below 50 m.

Optional advanced attribute query: “Find mapped buildings with building=house within 100 metres of roads with highway=residential.” It joins explicit OSM tags; it does not infer residential use from roof appearance.

SkyCLIP ranks scene candidates for free-text visual descriptions. Cached grounding covers only the declared concepts above; broader object grounding requires the bounded OWLv2 execution path and its own review. These samples do not provide verified solar, commercial-use, open-drain or historical forest-clearing evidence. Preserve unsupported conditions and explain the missing evidence.

### Why Phoenix is the sharper prototype choice

The Phoenix source item is [`az_m_3311240_nw_12_030_20230917_20240304`](https://planetarycomputer.microsoft.com/api/stac/v1/collections/naip/items/az_m_3311240_nw_12_030_20230917_20240304), acquired **17 September 2023**. Its 0.3 m pixels have twice the linear sampling density of the 0.6 m Sacramento source. The crop covers 300 m × 300 m around longitude `-112.095`, latitude `33.485`. Original-resolution visual inspection shows clear roofs, roads and pools. Sharper input does not establish better measured detector accuracy.

The source raster is `pipeline/data/source-phoenix-naip-2023/phoenix-naip-rgb.tif`, with WGS84 bounds `[-112.0966328325119, 33.48363527395297, -112.09337014634835, 33.48636920995804]` and SHA-256 `ce69c1969c76ed6f92c6df0b7bea2a8a7135f00b159bff10abcecae76f4cb95b`. Only a bounded RGB window was fetched; source files including vectors and raw responses total about 8.6 MB. This is genuinely different imagery from the same USDA NAIP provider, rather than a claim of a new imagery provider.

The OSM response was retrieved from `https://overpass.private.coffee/api/interpreter` on **9 October 2026**, after the first two public endpoints failed. Its recorded database timestamp is **15 July 2026**, not the retrieval date. Both dates are retained. The extract includes a 525 m margin and has zero returned waterways. The image and vector snapshots have different dates, the map's completeness is unknown, and open-drain proximity remains unsupported. Imagery attribution is **USDA Farm Service Agency / NAIP; hosted by Microsoft Planetary Computer**; vectors are **OpenStreetMap contributors**, under [ODbL 1.0](https://www.openstreetmap.org/copyright).

## 1. The two challenge references

### SkyScript / SkyCLIP

The [official SkyScript repository](https://github.com/wangzhecheng/SkyScript) describes 5.2 million image–text pairs. Images are JPG crops retrieved around OSM focus objects across several imagery sources and acquisition years. They are distributed in six large image archives, with separate metadata archives and caption CSVs. This is a dispersed retrieval dataset; geographic continuity across one municipality must be checked before treating a subset as an atlas.

Each image filename encodes its focus object ID, imagery source alias and year. Its metadata dictionary includes:

| Field | Meaning | TerraLens use |
| --- | --- | --- |
| `box` | `(west, south, east, north)` in longitude/latitude | Preserve extent; validate coordinate mapping before geographic grounding |
| `time` | Acquisition `(year, month, day, hour, minute)` | Retain the source date |
| `center_tags` | Visually groundable tags of the focus object | Attribute evidence only; focus object need not be centered |
| `surrounding_tags` | Tags of surrounding objects | Context; insufficient to assign an attribute to a specific detection |
| `title` / `title_multi_objects` | Captions from focus-object or multi-object tags | Retrieval reference, with OSM-derived caption provenance |

The [source mapping](https://github.com/wangzhecheng/SkyScript/blob/main/image_sources.py) includes NAIP, European orthophotos, SkySat, Sentinel-2 and Landsat. Resolution and permitted reuse therefore depend on the underlying source. The repository's MIT software licence does not establish one universal imagery licence.

Use the published SkyCLIP checkpoint for the image and text towers together. SkyScript captions or OSM tags must not become fabricated detector outputs. Because captions are derived from OSM, evaluating against those same tags can reward label overlap without proving independent visual recognition. Preserve a manually reviewed evaluation set separately.

### SpaceNet 1

The [SpaceNet 1 page](https://spacenet.ai/spacenet-buildings-dataset-v1/) provides Rio imagery with building footprints. Its processed tiles cover approximately 200 m × 200 m; the advertised training RGB download is 23 GB and the testing imagery download is 7.9 GB. The page lists three-band imagery, eight-band imagery and separate training building GeoJSON archives under `spacenet/SN1_buildings/`.

This is useful for footprint extraction and image-to-building association. Its building labels do not establish solar installations, commercial use, open drainage, unpaved roads, historical forest clearing or pothole severity. Holdout imagery without building labels cannot be treated as a locally verified benchmark. The page declares **Creative Commons Attribution-ShareAlike 4.0** for the dataset; retain its attribution and applicable share-alike obligations when distributing derivatives. Avoid downloading multi-gigabyte archives for the first implementation.

## 2. Additional NAIP corpus and provenance

The working area is a **600 m residential scene in Sacramento, California**. This is historical real aerial imagery, with genuine OSM features downloaded separately. It is a small development area, not Pune coverage or a current municipal survey.

| Item | Recorded value |
| --- | --- |
| Raster | `pipeline/data/source-fast/sacramento-naip-rgb.tif` |
| Source | USDA NAIP hosted by Microsoft Planetary Computer |
| STAC item | `ca_m_3812129_sw_10_060_20220621` |
| Acquisition time | `2022-06-21T16:00:00Z` as recorded by the STAC item |
| Image dimensions | 1,000 × 1,000 pixels; RGB bands |
| Resolution | 0.6 m per pixel; 600 m per side in the source projection |
| Raster CRS | EPSG:26910, NAD83 / UTM zone 10N; metre units |
| WGS84 extent | `[-121.48500088022256, 38.54575482525348, -121.47800407524632, 38.55124944984303]` |
| SHA-256 | `44a3087c1228020ff5538c6227cc0c38213b4a9116f6f018f6972f2b5e0744f2` |
| OSM database timestamp | `2026-10-09T04:51:14Z` |
| OSM retrieval time | `2026-10-09T04:52:30.927128+00:00` |
| Retrieved vectors | 1,079 roads; 2,972 buildings; 0 waterways |
| Vector extent | `[-121.4911196333497, 38.540944536177925, -121.47187841955424, 38.55605475402251]` |
| Border allowance | 525 m outside the scene for a maximum 500 m proximity query |

The [Planetary Computer NAIP collection](https://planetarycomputer.microsoft.com/api/stac/v1/collections/naip) documents four-band RGB/near-infrared source COGs and includes a Public Domain licence link. The local crop retains RGB only. Attribute imagery to **USDA Farm Service Agency / NAIP; hosted by Microsoft Planetary Computer**. Keep the original item and asset references with the crop; the collection's generic licence field and licence link should be read together before wider redistribution.

Attribute vector data to **OpenStreetMap contributors**, under [ODbL 1.0](https://www.openstreetmap.org/copyright). Source IDs, tags, geometry and snapshot details remain in the vector files and sidecars. The [OSM API policy](https://operations.osmfoundation.org/policies/api/) identifies Overpass as a read-only alternative; TerraLens uses a bounded Overpass query rather than bulk downloading basemap tiles.

The 2022 imagery and 2026 map snapshot are **not contemporaneous**. A new or changed building in OSM may disagree with historical imagery. Retain this mismatch in result evidence and avoid claiming that image observations establish current conditions. The vector allowance reduces missing nearby references at the image edge; it does not establish OSM completeness.

## 3. Files and supported evidence

The source directory contains the raster, `preview.jpg`, `naip-item.json`, `search.json`, raster metadata sidecar, raw `osm-response.json`, `osm-query.txt`, and roads/buildings/waterways GeoJSON layers with individual metadata sidecars. Sidecars preserve source URLs, dates, CRS, extents, attribution, checksums and query scope. The raster sidecar sets `profile.limits.maximum_distance_m` to `500`.

Visible roofs and backyard pools are reasonable first concepts to evaluate at original resolution. Model inference and manual review must establish which detections are usable; the presence of real imagery is not a measured detector result. A 2 m object spans only about 3.3 pixels at this resolution, so small solar panels or road damage cannot be assumed observable.

The waterways file is genuinely empty for the chosen query. Its absence does not prove that no drain exists. **Open-drain proximity remains unavailable for this corpus.** Commercial use also requires explicit attributable tags or independent records; roof size is insufficient. Historical forest clearing needs paired dates and suitable change evidence.

Use retained raster transforms for pixel-to-map conversion and reproject both detections and vectors into the local metre CRS for distances. Preserve WGS84 longitude/latitude for GeoJSON export. Basemap display is independent of the evidence raster and must not be presented as its acquisition date or detector input.
