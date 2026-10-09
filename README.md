# TerraLens

Natural-language search over aerial imagery and OpenStreetMap features for the GEOAI 05 challenge.

The core flow is **question → validated query plan → visual evidence → geographic association → metric spatial check → map and GeoJSON**. TerraLens preserves object detections, mapped footprints and reference features as separate evidence. Whole-image similarity is displayed as a candidate, rather than a verified object location.

## Run the prototype

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
