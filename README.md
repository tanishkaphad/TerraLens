# TerraLens

Natural-language search over aerial imagery and OpenStreetMap features for the GEOAI 05 challenge.

The core flow is **question → validated query plan → visual evidence → geographic association → metric spatial check → map and GeoJSON**. TerraLens preserves object detections, mapped footprints and reference features as separate evidence. Whole-image similarity is displayed as a candidate, rather than a verified object location.

## Run the prototype

The prepared local prototype includes a three-panel discovery workspace, dataset switching, natural-language search and radius follow-ups, image/vector controls, result evidence and complete GeoJSON export. It opens with sharper Phoenix NAIP imagery from September 2023 at 0.3 m per pixel; SpaceNet 1 Rio and Sacramento NAIP/OSM remain available. A continuous satellite or street basemap provides wider geographic context, with an expanded-map view. Dashed outlines identify the analysis imagery; searches use the prepared evidence inside that coverage.

With the existing environment and prepared data, start it directly:

```bash
pipeline/.venv/bin/python pipeline/scripts/run_demo.py --port 8010
```

Open [the local prototype](http://127.0.0.1:8010). Try “Find swimming pools within 100 metres of roads”, then “Change the distance to 50 metres”. Five clickable examples are available for each study area. The Search mode works without a conversation API key. Conversation mode is enabled only when the backend provider is configured; configuration alone does not verify the provider connection.

See [pipeline/README.md](pipeline/README.md) for source acquisition, local model preparation and startup. The service hosts its own interactive MapLibre interface and API. Imagery, vector snapshots, model weights and generated evidence bundles are stored outside Git under `pipeline/data/`.

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
