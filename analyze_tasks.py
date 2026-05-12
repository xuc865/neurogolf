#!/usr/bin/env python3
"""Analyze all 400 tasks to understand structure and patterns."""
import json, glob, os

DATA_DIR = "/Users/wxc/workspace/neurogolf-2026/data"
os.chdir(DATA_DIR)

files = sorted(glob.glob("task*.json"))
print(f"Total: {len(files)} tasks\n")

# Categorize by relationship
same_size = 0
diff_size = 0
color_remap = 0  # same size, same positions, just color change
identity = 0
complex_map = {}

for f in files:
    with open(f) as fh:
        d = json.load(fh)
    train = d.get("train", [])
    if not train:
        complex_map[f] = "no_train"
        continue
    
    h0, w0 = len(train[0]["input"]), len(train[0]["input"][0])
    h1, w1 = len(train[0]["output"]), len(train[0]["output"][0])
    
    all_same_size = True
    all_pixel_maps_consistent = True
    pixel_map = {}
    
    for ex in train:
        inp = ex["input"]
        out = ex["output"]
        hi, wi = len(inp), len(inp[0])
        ho, wo = len(out), len(out[0])
        if hi != ho or wi != wo:
            all_same_size = False
            break
        for r in range(hi):
            for c in range(wi):
                ci, co = inp[r][c], out[r][c]
                if (r, c) not in pixel_map:
                    pixel_map[(r, c)] = {}
                if ci not in pixel_map[(r, c)]:
                    pixel_map[(r, c)][ci] = co
                elif pixel_map[(r, c)][ci] != co:
                    all_pixel_maps_consistent = False
                    
    if all_same_size:
        same_size += 1
        # Check if position-independent color remap
        all_positions_same = True
        ref_map = None
        for pos, pm in pixel_map.items():
            if ref_map is None:
                ref_map = pm
            elif pm != ref_map:
                all_positions_same = False
                break
        
        if all_positions_same and ref_map:
            is_identity = all(k == v for k, v in ref_map.items())
            if is_identity:
                identity += 1
            else:
                color_remap += 1
        else:
            complex_map[f] = "positional"
    else:
        diff_size += 1
        complex_map[f] = f"size_change {h0}x{w0}->{h1}x{w1}"

print(f"Same size: {same_size}")
print(f"  Identity: {identity}")
print(f"  Color remap: {color_remap}")
print(f"  Positional changes: {len([v for v in complex_map.values() if 'positional' in str(v)])}")
print(f"Different size: {diff_size}")
print(f"\nComplex tasks sample:")
for f in list(complex_map.keys())[:10]:
    print(f"  {f}: {complex_map[f]}")
