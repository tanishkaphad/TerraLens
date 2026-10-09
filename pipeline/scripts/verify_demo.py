#!/usr/bin/env python3
"""Exercise a running real-data demo; record capability/count/latency evidence.

This checks execution and provenance, not object accuracy or model recall.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import time
from datetime import datetime, timezone
import httpx

ROOT=Path(__file__).resolve().parents[1]

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url",default="http://127.0.0.1:8000")
    parser.add_argument("--output",type=Path,default=ROOT/"data/validation/api-check.json")
    args=parser.parse_args()
    report={"checked_at_utc":datetime.now(timezone.utc).isoformat(),"url":args.url,"scope":"Actual API/model execution; no annotated accuracy evaluation","checks":[]}
    with httpx.Client(base_url=args.url,timeout=180) as client:
        def call(method,path,body=None,expected=200):
            start=time.perf_counter();response=client.request(method,path,json=body)
            record={"path":path,"status":response.status_code,"elapsed_seconds":round(time.perf_counter()-start,3)}
            report["checks"].append(record)
            if response.status_code!=expected:
                raise AssertionError(f"{path}: {response.status_code} {response.text[:500]}")
            return response.json(),record
        health,_=call("GET","/health");assert health["status"]=="alive"
        ready,_=call("GET","/ready");assert ready["status"]=="ready",ready
        report["readiness"]=ready
        capabilities,_=call("GET","/api/capabilities")
        report["dataset_version"]=capabilities["dataset_version"]
        map_data,_=call("GET","/api/map?tile_limit=64&vector_limit=5000")
        assert map_data["tiles"] and map_data["vector_layers"]
        assert all(len(tile["coordinates"])==4 for tile in map_data["tiles"])
        asset=client.get(map_data["tiles"][0]["asset_url"])
        assert asset.status_code==200 and asset.headers["content-type"].startswith("image/")
        report["map"]=map_data["counts"]
        reference="roads" if "roads" in capabilities["vector_layers"] else next(iter(capabilities["vector_layers"]))
        results=[]
        for identifier,concept in capabilities["supported_concepts"].items():
            phrase=concept["aliases"][0] if concept.get("aliases") else concept["prompt"]
            query=f"Find {phrase} within 100 metres of {reference}"
            result,record=call("POST","/api/query",{"query":query,"mode":"catalog_scan","page_size":20})
            assert result["result_kind"]=="grounded_objects" and result["plan"]["distance_m"]==100
            assert result["text_similarity_enabled"] is True
            record.update(query=query,matches=result["total_matches"],uncertain=result["total_uncertain"],objects_evaluated=result["objects_evaluated"])
            results.append(result)
            for feature in result["features"]:
                assert all(condition["status"]=="pass" for condition in feature["properties"]["conditions"])
            exported,_=call("POST","/api/export",{"plan":result["plan"],"mode":"catalog_scan","dataset_version":result["dataset_version"],"include_uncertain":True})
            assert exported["type"]=="FeatureCollection" and exported["metadata"]["truncated"] is False
            assert exported["metadata"]["total_matches"]==result["total_matches"]
            args.output.parent.mkdir(parents=True,exist_ok=True)
            export_path=args.output.parent/f"{identifier}-evidence.geojson"
            export_path.write_text(json.dumps(exported,indent=2,allow_nan=False))
            record["export"]={"file":export_path.name,"features":len(exported["features"]),"sha256":hashlib.sha256(export_path.read_bytes()).hexdigest()}
        if results:
            original=results[0]["plan"]
            followup,_=call("POST","/api/query",{"query":"Make the radius 25 metres","previous_plan":original})
            assert followup["plan"]["distance_m"]==25 and followup["plan"]["concept_id"]==original["concept_id"]
            assert followup["plan"]["attributes"]==original["attributes"]
        image,record=call("POST","/api/query",{"query":"aerial residential neighbourhood","mode":"explore","page_size":20})
        assert image["result_kind"]=="tile_retrieval" and image["objects_evaluated"]==0
        record["image_candidates"]=image["total_tile_candidates"]
        call("POST","/api/query",{"query":f"Find potholes within 100 metres of {reference}"},expected=422)
        call("POST","/api/export",{"plan":results[0]["plan"],"dataset_version":"different-version"},expected=409)
    report["result"]="passed"
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False))
    print(json.dumps(report,indent=2))

if __name__=="__main__":main()
