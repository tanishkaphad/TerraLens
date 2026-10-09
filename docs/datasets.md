# TerraLens dataset audit and first working corpus

Dataset roles are different: SkyScript supports language–image retrieval, SpaceNet 1 supports building extraction, and the first TerraLens corpus uses a small real NAIP raster paired with a bounded OpenStreetMap snapshot. None supplies ground truth for the complete challenge example about commercial roofs, solar panels and open drains.

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

## 2. Actual initial corpus

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
