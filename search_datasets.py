#!/usr/bin/env python3
"""Search Kaggle for NeuroGolf datasets and download the best ones."""
import json, os, subprocess, sys, zipfile
from pathlib import Path

WORKDIR = Path("/Users/wxc/Documents/codes/neurogolf")
DATADIR = WORKDIR / "data"
CACHEDIR = WORKDIR / "downloaded_datasets"
CACHEDIR.mkdir(exist_ok=True)

# Known high-quality datasets from earlier exploration
DATASETS = [
    ("kk289", "neurogolf-best-solution"),
    ("dionisiorodrigues", "neurogolf-solution-v2"),
    ("afr1ste", "neurogolf-submission"),
    ("needless090", "neurogolf"),
]

# Our current models as baseline
SUBDIR = WORKDIR / "submission"
SUBDIR_OPT = WORKDIR / "submission_merged"
SUBDIR_OPT.mkdir(exist_ok=True)

# 1. Copy existing models as starting point
import shutil
import onnx
import numpy as np

for tid in range(1, 401):
    src = SUBDIR / f"task{tid:03d}.onnx"
    dst = SUBDIR_OPT / f"task{tid:03d}.onnx"
    if src.exists():
        shutil.copy2(src, dst)

print(f"Copied {len(list(SUBDIR.glob('*.onnx')))} baseline models")

# 2. Analyze current model sizes and types
models_info = []
for tid in range(1, 401):
    fpath = SUBDIR_OPT / f"task{tid:03d}.onnx"
    if fpath.exists():
        sz = fpath.stat().st_size
        try:
            model = onnx.load(str(fpath))
            params = sum(np.prod(list(d.dims)) for init in model.graph.initializer if all(d > 0 for d in d.dims))
            ops = [n.op_type for n in model.graph.node[:3]]
            models_info.append((tid, sz, int(params), ops))
        except:
            models_info.append((tid, sz, 0, ["ERROR"]))

# Summary
total_sz = sum(m[1] for m in models_info)
total_params = sum(m[2] for m in models_info)
print(f"\nCurrent models: {len(models_info)} total")
print(f"Total size: {total_sz} bytes ({total_sz/1024:.1f} KB)")
print(f"Total params: {total_params}")

# Identify large models (>50KB) that need optimization
large_models = [(tid, sz, p, ops) for tid, sz, p, ops in models_info if sz > 50000]
print(f"\nLarge models (>50KB): {len(large_models)}")
for tid, sz, p, ops in sorted(large_models, key=lambda x: -x[1])[:20]:
    print(f"  task{tid:03d}: {sz/1024:.1f}KB, {p} params, ops={ops}")

# Check for models that are same size (potential identity)
same_count = sum(1 for _, sz, _, _ in models_info if sz == 139)
print(f"\n139B identity models: {same_count}")

# Look for models using banned ops
banned_ops = {"Compress"}
for tid, sz, p, ops in models_info:
    for op in ops:
        if op in banned_ops:
            print(f"  task{tid:03d} uses banned op: {op}")
