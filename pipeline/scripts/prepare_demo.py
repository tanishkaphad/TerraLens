#!/usr/bin/env python3
"""Prepare the bounded Sacramento sample with the locally pinned open models."""
from __future__ import annotations
import argparse
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
REVISION="cfd3195ba4ea9592eec887ded089f4c08eff231d"

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir",type=Path,default=ROOT/"data/source-fast")
    parser.add_argument("--output",type=Path,default=ROOT/"data/demo")
    parser.add_argument("--device",choices=("auto","cpu","mps","cuda"),default="auto")
    args=parser.parse_args()
    source=args.source_dir.resolve();imagery=source/"sacramento-naip-rgb.tif"
    command=[sys.executable,str(ROOT/"scripts/prepare_bundle.py"),"--area-name","Sacramento / Land Park study area",
        "--imagery",str(imagery),"--metadata",f"{imagery}={imagery}.metadata.json",
        "--concept","building_roofs=building roofs","--associate","building_roofs=buildings",
        "--concept","swimming_pools=swimming pools",
        "--checkpoint",str(ROOT/"data/models/SkyCLIP_ViT_B32_top50pct.safetensors"),
        "--owl-model",str(ROOT/f"data/models/owlv2-{REVISION}"),"--owl-revision",REVISION,
        "--tile-size","640","--overlap","64","--max-tiles","16","--device",args.device,
        "--imagery-attribution","USDA Farm Service Agency / NAIP; Microsoft Planetary Computer",
        "--vector-attribution","© OpenStreetMap contributors, ODbL 1.0","--output",str(args.output.resolve())]
    for layer in ("roads","buildings"):
        path=source/"vectors"/f"{layer}.geojson"
        command.extend(["--vectors",f"{layer}={path}","--vector-metadata",f"{layer}={path}.metadata.json"])
    return subprocess.call(command,cwd=ROOT)

if __name__=="__main__":raise SystemExit(main())
