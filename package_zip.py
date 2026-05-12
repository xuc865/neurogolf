#!/usr/bin/env python3
"""Package submission_minimal as submission_opt2.zip"""
import zipfile, os, sys
from pathlib import Path

workdir = Path("/Users/wxc/Documents/codes/neurogolf")
subdir = workdir / "submission_minimal"
outpath = workdir / "submission_opt2.zip"

if outpath.exists():
    outpath.unlink()

with zipfile.ZipFile(outpath, 'w', zipfile.ZIP_DEFLATED) as zf:
    for onnx_file in sorted(subdir.glob("*.onnx"), key=lambda p: int(p.stem.replace('task',''))):
        zf.write(onnx_file, onnx_file.name)

# Verify
with zipfile.ZipFile(outpath) as zf:
    names = zf.namelist()
    print(f"{len(names)} files in submission_opt2.zip")
    sizes = [zf.getinfo(n).file_size for n in names]
    total = sum(sizes)
    print(f"Total size: {total} bytes ({total/1024:.1f} KB)")
    print(f"First: {names[0]}, Last: {names[-1]}")

print(f"\nOutput: {outpath}")
print(f"File size: {outpath.stat().st_size / 1024:.1f} KB")
