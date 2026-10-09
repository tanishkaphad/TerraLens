#!/usr/bin/env python3
"""Convert the pinned official SkyCLIP training checkpoint to plain safetensors.

No arbitrary pickle globals are allowed. The legacy file is accepted only at the
recorded official hash; NumPy scalar metadata has a narrowly scoped allowlist.
The output contains the model tensors only, without optimizer/training state.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_SHA256 = "69ba9fa7089322a83e9164fe515a222bcece2a715888db5749ab5ce4d1602313"

def digest(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""):h.update(chunk)
    return h.hexdigest()

def main():
    source=ROOT/"data/models/SkyCLIP_ViT_B32_top50pct.pt"
    target=source.with_suffix(".safetensors")
    if digest(source)!=EXPECTED_SHA256:
        raise ValueError("Source differs from the pinned official checkpoint; refusing conversion.")
    allowed=[(np._core.multiarray.scalar,"numpy.core.multiarray.scalar"),np.dtype,np.dtypes.Float64DType]
    with torch.serialization.safe_globals(allowed):
        checkpoint=torch.load(source,map_location="cpu",weights_only=True)
    state={key.removeprefix("module."):value.detach().cpu().contiguous() for key,value in checkpoint["state_dict"].items()}
    if not all(isinstance(value,torch.Tensor) and bool(torch.isfinite(value).all()) for value in state.values()):
        raise ValueError("Checkpoint contains invalid model tensors.")
    if not target.exists():
        save_file(state,str(target),metadata={"source_sha256":EXPECTED_SHA256,"model":"ViT-B-32","origin":"official SkyScript SkyCLIP B32 top50pct epoch20"})
    record={"file":str(target.relative_to(ROOT)),"sha256":digest(target),"bytes":target.stat().st_size,"source_file":str(source.relative_to(ROOT)),"source_sha256":EXPECTED_SHA256,"tensors":len(state),"transformation":"strip module prefix; preserve model tensors; discard training state"}
    metadata=ROOT/"data/asset-provenance.json"
    records=json.loads(metadata.read_text()) if metadata.exists() else []
    by_path={r["file"]:r for r in records+[record]}
    metadata.write_text(json.dumps(list(by_path.values()),indent=2))
    print(json.dumps(record))

if __name__=="__main__":main()
