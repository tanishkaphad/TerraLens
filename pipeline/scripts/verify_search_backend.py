#!/usr/bin/env python3
"""Exercise the HTTP search backend using its actual runtime source catalog."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import requests


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8020")
    parser.add_argument("--dataset", required=True, help="An ID returned by /api/catalog.")
    parser.add_argument("--layer", help="A mapped layer ID from the selected source; omitted for --query.")
    parser.add_argument("--query", help="Exercise the configured natural-language parser with this question.")
    parser.add_argument("--direct-parser", action="store_true", help="Diagnose the configured parser directly with sanitized failure metadata.")
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "data/validation/search-backend/http")
    args = parser.parse_args()
    if args.direct_parser:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from terralens.configuration import load_configuration
        from terralens.search_api import get_search_service
        from terralens.search_plan import SearchPlanError
        load_configuration()
        service = get_search_service()
        try:
            plan = service.plan(args.query, scope={"mode": "all_loaded", "dataset_ids": [args.dataset]})
            print(json.dumps({"status": plan.status, "plan": plan.model_dump()}, indent=2))
            return 0
        except SearchPlanError as exc:
            cause = exc.__cause__
            print(json.dumps({"code": exc.code, "details": exc.details,
                "failure_type": type(cause).__name__ if cause else None,
                "provider_status": getattr(cause, "status_code", None),
                "provider_code": getattr(cause, "code", None)}, indent=2))
            return 1
        finally:
            service.runner.shutdown()
    base = args.base_url.rstrip("/")
    session = requests.Session()
    catalog = session.get(base + "/api/catalog", timeout=55).json()
    if args.dataset not in catalog["datasets"]:
        raise ValueError("Choose a dataset from the current runtime catalog.")
    scope = {"mode": "all_loaded", "dataset_ids": [args.dataset]}
    if args.query:
        body = {"query": args.query, "scope": scope}
        planned = session.post(base + "/api/search/plan", json=body, timeout=100)
        if planned.status_code != 200:
            print(json.dumps({"planning_http_status": planned.status_code, "response": planned.json()}, indent=2))
            return 1
        body = {"plan": planned.json()["plan"]}
    else:
        if not args.layer or args.layer not in catalog["datasets"][args.dataset]["layers"]:
            raise ValueError("Choose a registered mapped layer or provide --query.")
        body = {"plan": {"original_query": "HTTP verification of all mapped records in an explicit source",
            "scope": scope, "entities": [{"id": "subject", "description": args.layer, "kind": "mapped",
                "dataset_id": args.dataset, "layer_id": args.layer}], "subject_entity_ids": ["subject"], "clauses": []}}
    created = session.post(base + "/api/search", json=body, timeout=55)
    created.raise_for_status()
    run = created.json()
    deadline = time.monotonic() + args.timeout
    while time.monotonic() < deadline:
        run = session.get(base + run["status_url"], timeout=55).json()
        if run["status"] not in {"queued", "running"}:
            break
        time.sleep(.2)
    if run["status"] != "completed":
        print(json.dumps({"status": run["status"], "coverage": run["coverage"]}, indent=2))
        return 1
    page = session.get(base + run["results_url"], params={"revision": run["revision"], "page_size": 7}, timeout=55).json()
    ids, first_page_size = [], len(page["records"])
    while True:
        ids.extend(record["id"] for record in page["records"])
        if not page["next_cursor"]:
            break
        page = session.get(base + run["results_url"], params={"cursor": page["next_cursor"], "page_size": 7}, timeout=55).json()
    exported = session.get(base + run["export_url"], params={"revision": run["revision"]}, timeout=55).json()
    mapped = session.get(base + run["map_url"], params={"revision": run["revision"]}, timeout=55).json()
    export_ids = {item["properties"]["result_id"] for item in exported["features"]}
    if len(ids) != len(set(ids)) or set(ids) != export_ids or len(mapped["features"]) != len(ids):
        raise AssertionError("Result pages, complete map and saved export disagree.")
    args.output.mkdir(parents=True, exist_ok=True)
    receipt = {"validation_kind": "actual HTTP execution over registered source evidence", "run": run,
        "first_page_count": first_page_size, "all_paged_results": len(ids), "map_feature_count": len(mapped["features"]),
        "export_feature_count": len(exported["features"]), "independent_visual_accuracy": "not_measured"}
    (args.output / "receipt.json").write_text(json.dumps(receipt, indent=2))
    (args.output / "result.geojson").write_text(json.dumps(exported, indent=2))
    print(json.dumps({"status": run["status"], "run_id": run["run_id"], "first_page": first_page_size,
        "complete_results": len(ids), "export": len(exported["features"]), "receipt": str(args.output / "receipt.json")}, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except requests.RequestException:
        raise SystemExit("The backend HTTP request failed; check the local server and connection.") from None
