#!/usr/bin/env python3
"""Package the fixed submission as submission_opt2.zip and verify."""
import zipfile, os
from pathlib import Path

WORKDIR = Path("/Users/wxc/Documents/codes/neurogolf")
SUBDIR = WORKDIR / "submission_minimal"
OUT = WORKDIR / "submission_opt2.zip"

# Repackage
if OUT.exists():
    OUT.unlink()

with zipfile.ZipFile(OUT, 'w', zipfile.ZIP_DEFLATED) as zf:
    for f in sorted(SUBDIR.glob("*.onnx"), key=lambda p: int(p.stem.replace('task',''))):
        zf.write(f, f.name)

# Verify
with zipfile.ZipFile(OUT) as zf:
    names = sorted(zf.namelist())
    total_raw = sum(zf.getinfo(n).file_size for n in names)
    total_compressed = OUT.stat().st_size

print(f"{'='*60}")
print(f"submission_opt2.zip ready!")
print(f"{'='*60}")
print(f"Files:     {len(names)}")
print(f"Raw size:  {total_raw:,} bytes ({total_raw/1024:.1f} KB)")
print(f"ZIP size:  {total_compressed:,} bytes ({total_compressed/1024:.1f} KB)")

# Show size distribution
size_buckets = {"<1KB": 0, "1-10KB": 0, "10-50KB": 0, "50-100KB": 0, "100-500KB": 0, ">500KB": 0}
for n in names:
    sz = zf.getinfo(n).file_size
    if sz < 1024: size_buckets["<1KB"] += 1
    elif sz < 10240: size_buckets["1-10KB"] += 1
    elif sz < 51200: size_buckets["10-50KB"] += 1
    elif sz < 102400: size_buckets["50-100KB"] += 1
    elif sz < 524288: size_buckets["100-500KB"] += 1
    else: size_buckets[">500KB"] += 1

print(f"\nSize distribution:")
for k, v in size_buckets.items():
    print(f"  {k}: {v} files")

# Show largest 10
largest = sorted([(n, zf.getinfo(n).file_size) for n in names], key=lambda x: -x[1])[:10]
print(f"\nLargest 10:")
for n, s in largest:
    print(f"  {n}: {s/1024:.1f} KB")
