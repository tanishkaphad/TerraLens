# GeoVerify-X

The revised GEOAI backend now provides a dynamic catalog and durable concurrent search pipeline at `POST /api/search`. It preserves complete result revisions for pagination, map delivery and GeoJSON export. Start it with `pipeline/.venv/bin/python pipeline/scripts/run_search.py --port 8020`; see the [backend implementation guide](docs/backend-pipeline.md) for source registration, API contracts, recorded checks and capability limits. The existing interface below uses the earlier discovery routes.

Search the Earth Beyond the Map. Natural-language discovery of named places, mapped infrastructure and imagery evidence for the GEOAI 05 challenge.

The core flow is **question → intent routing → place lookup, vector analysis or visual search → evidence checks → map and GeoJSON**. Named places use a worldwide place index. Spatial queries use bounded, actual map geometry. Visual discovery uses the prepared SkyCLIP/OWLv2 imagery libraries. Object detections, mapped footprints and reference features remain separate evidence; scene similarity remains a candidate.

## Run the prototype

One search box routes requests automatically. The world map is available for context; imagery analysis covers the registered Phoenix, SpaceNet 1 Rio and Sacramento libraries. Phoenix uses NAIP imagery from September 2023 at 0.3 m per pixel. Dashed outlines identify analysis coverage. Pune rooftop imagery is not prepared, so requests for rooftop solar there explain the missing evidence rather than producing matches.

With the existing environment and prepared data, start it directly:

```bash
pipeline/.venv/bin/python pipeline/scripts/run_demo.py --port 8010
```

Open [the local prototype](http://127.0.0.1:8010). Try:

1. Find Sinhagad College in Pune.
2. Find buildings within 300 metres of a river in Pune.
3. Find rooftop solar panels within 500 metres of Pune Municipal Corporation in Pune.
4. Find swimming pools within 100 metres of roads in Phoenix.
5. Find satellite imagery of densely built neighbourhoods in Phoenix.

Place lookup depends on the configured geocoder and its map index. Vector searches cover an explicitly reported local sample around the resolved location, with a reference margin for distance checks; they do not establish municipality-wide completeness. Multiple place matches remain visible. “Near” requires an explicit distance before a proximity search executes.

The prototype uses Photon for submitted place queries, with caching and serialized requests. Its public demo service permits reasonable use without an availability guarantee; configure a private instance or a suitable provider for sustained traffic. The app does not perform per-keystroke autocomplete. [Photon service documentation](https://github.com/komoot/photon)

See [pipeline/README.md](pipeline/README.md) for source acquisition, local model preparation and startup. The service hosts its own interactive MapLibre interface and API; the internal Python package remains `terralens`. Imagery, vector snapshots, model weights and generated evidence bundles stay outside Git under `pipeline/data/`.

```bash
cd pipeline
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
python scripts/fetch_assets.py
python scripts/convert_checkpoint.py
python scripts/fetch_demo_data.py
python scripts/prepare_demo.py
python scripts/run_demo.py
```

The default parser works without an API key for declared concepts, explicit distances, intersections and radius-only follow-ups. An optional schema-constrained language-model parser runs on the backend; it never executes model-generated SQL or code.

## Project references

- [workflow.md](workflow.md): problem analysis, research gap, selected architecture, evidence semantics and eight-hour delivery gates.
- [docs/datasets.md](docs/datasets.md): dataset roles, source provenance and limitations.
- [pipeline/tests](pipeline/tests): geometry, query semantics, evidence handling and export checks.

Model detections and retrieval scores require local accuracy evaluation. A working API or successful spatial check does not establish detection accuracy, municipality-wide coverage or current real-world conditions. Public deployment remains a separate validation milestone.

Code uses the existing [MIT licence](LICENSE). Data, model and MapLibre notices have their own terms and attribution.
