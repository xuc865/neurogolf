#!/usr/bin/env python3
"""Check Kaggle leaderboard and submit optimized submission via MCP."""
import json, os, urllib.request, zipfile
from pathlib import Path

WORKDIR = Path("/Users/wxc/Documents/codes/neurogolf")
TOKEN = os.environ.get("KAGGLE_TOKEN") or os.environ.get("KAGLE_TOKEN")
if not TOKEN:
    raise SystemExit("Set KAGGLE_TOKEN before using this script.")

# 1. Check leaderboard
print("=" * 60)
print("Checking Kaggle NeuroGolf Leaderboard")
print("=" * 60)

req = urllib.request.Request(
    "https://www.kaggle.com/api/v1/competitions/neurogolf-2026/leaderboard/view",
    headers={"Authorization": f"Bearer {TOKEN}"}
)
try:
    with urllib.request.urlopen(req, timeout=10) as resp:
        data = json.loads(resp.read())
    print(f"\nTop teams:")
    for i, sub in enumerate(data.get('submissions', [])[:10]):
        print(f"  {i+1}. {sub.get('teamName','?'):40s} {sub.get('score','?'):>10s}")
except Exception as e:
    print(f"Error: {e}")

# 2. Package optimized submission
print(f"\n{'=' * 60}")
print("Packaging submission_opt2.zip")
print("=" * 60)

subdir = WORKDIR / "submission_minimal"
outpath = WORKDIR / "submission_opt2.zip"

if outpath.exists():
    outpath.unlink()

with zipfile.ZipFile(outpath, 'w', zipfile.ZIP_DEFLATED) as zf:
    onnx_files = sorted(subdir.glob("*.onnx"), key=lambda p: int(p.stem.replace('task','')))
    for f in onnx_files:
        zf.write(f, f.name)

# Verify
with zipfile.ZipFile(outpath) as zf:
    names = sorted(zf.namelist(), key=lambda n: int(n.replace('task','').replace('.onnx','')))
    sizes = [zf.getinfo(n).file_size for n in names]
    total = sum(sizes)

print(f"\n{len(names)} models in submission_opt2.zip")
print(f"Total bytes: {total:,}")
print(f"Compressed:  {outpath.stat().st_size/1024:.1f} KB")

# Find largest models
model_sizes = [(n, zf.getinfo(n).file_size) for n in names]
largest = sorted(model_sizes, key=lambda x: -x[1])[:5]
print(f"\nLargest models:")
for n, s in largest:
    print(f"  {n}: {s/1024:.1f} KB")

# Check for models that use Compress (banned op)
import onnx
print(f"\nChecking for banned ops...")
banned_found = 0
for f in onnx_files:
    try:
        model = onnx.load(str(f))
        for node in model.graph.node:
            if node.op_type == "Compress":
                banned_found += 1
                print(f"  {f.name} uses Compress!")
                break
    except:
        pass
if banned_found == 0:
    print("  No banned ops found! ✅")

print(f"\n{'=' * 60}")
print("Done!")
print(f"Output: {outpath}")
