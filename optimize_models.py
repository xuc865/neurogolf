#!/usr/bin/env python3
"""
NeuroGolf 2026: Optimize existing submission models.
Replace large models with minimal alternatives where possible.
"""
import os, json, math, onnx, shutil
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict

WORKDIR = Path("/Users/wxc/Documents/codes/neurogolf")
DATADIR = WORKDIR / "data"
SUBDIR_OLD = WORKDIR / "submission"
SUBDIR_NEW = WORKDIR / "submission_opt2"
SUBDIR_NEW.mkdir(exist_ok=True)

_CHANNELS = 10

def grid_to_tensor(grid):
    h, w = len(grid), len(grid[0])
    tensor = np.zeros((1, _CHANNELS, 30, 30), dtype=np.float32)
    for r in range(h):
        for c in range(w):
            color = grid[r][c]
            if 0 <= color < _CHANNELS:
                tensor[0, color, r, c] = 1.0
    return tensor

def tensor_to_grid(tensor, h, w):
    result = []
    for r in range(h):
        row = []
        for c in range(w):
            colors = [ch for ch in range(_CHANNELS) if tensor[0, ch, r, c] > 0.5]
            row.append(colors[0] if len(colors) == 1 else 10)
        result.append(row)
    return result

def run_onnx_with_data(onnx_path, grid_list):
    """Run ONNX model on input grids and return output grids."""
    import onnxruntime as ort
    try:
        model = onnx.load(str(onnx_path))
        session = ort.InferenceSession(model.SerializeToString())
        results = []
        for grid in grid_list:
            inp_t = grid_to_tensor(grid)
            out_t = session.run(["output"], {"input": inp_t})[0]
            h, w = len(grid), len(grid[0])
            out_t = (out_t > 0.0).astype(np.float32)
            out_grid = tensor_to_grid(out_t, h, w)
            results.append(out_grid)
        return results
    except Exception as e:
        return None

def is_correct_model(onnx_path, data):
    """Check if ONNX model produces correct output on all examples."""
    examples = data.get("train", []) + data.get("test", []) + data.get("arc-gen", [])
    if not examples:
        return False
    
    results = run_onnx_with_data(onnx_path, [ex["input"] for ex in examples])
    if results is None:
        return False
    
    for i, ex in enumerate(examples):
        expected = ex["output"]
        if results[i] != expected:
            return False
    return True

def try_identity_model(data):
    """Check if identity model works."""
    # Create minimal identity
    examples = data.get("train", []) + data.get("test", [])
    for ex in examples:
        if ex["input"] != ex["output"]:
            return False
    return True

def try_conv1x1(data):
    """Try to build a 1x1 Conv model that works."""
    examples = data.get("train", []) + data.get("test", [])
    
    # Check same size
    for ex in examples:
        if len(ex["input"]) != len(ex["output"]):
            return None
        if ex["input"] and ex["output"] and len(ex["input"][0]) != len(ex["output"][0]):
            return None
    
    # Build color mapping
    from collections import defaultdict, Counter
    # For each pair of (in_color, position), check output consistency
    position_map = defaultdict(lambda: defaultdict(Counter))
    
    for ex in examples:
        inp, out = ex["input"], ex["output"]
        h, w = len(inp), len(inp[0])
        for r in range(h):
            for c in range(w):
                in_c = inp[r][c]
                out_c = out[r][c]
                if 0 <= in_c < _CHANNELS and 0 <= out_c < _CHANNELS:
                    position_map[(r, c)][in_c][out_c] += 1
    
    # If any position maps the same input to different outputs -> not pure color map
    for pos, in_map in position_map.items():
        for in_c, out_counts in in_map.items():
            if len(out_counts) > 1:
                return None  # Not a simple color map
    
    # Build weight matrix
    weights = np.zeros((_CHANNELS, _CHANNELS, 1, 1), dtype=np.float32)
    for (r, c), in_map in position_map.items():
        for in_c, out_counts in in_map.items():
            out_c = list(out_counts.keys())[0]
            weights[out_c, in_c, 0, 0] = 1.0
    
    # Verify
    for ex in examples:
        inp, out = ex["input"], ex["output"]
        h, w = len(inp), len(inp[0])
        inp_t = grid_to_tensor(inp)
        conv_out = np.zeros((1, _CHANNELS, 30, 30), dtype=np.float32)
        for oc in range(_CHANNELS):
            for ic in range(_CHANNELS):
                wv = weights[oc, ic, 0, 0]
                if abs(wv) > 0.01:
                    conv_out[0, oc, :h, :w] += wv * inp_t[0, ic, :h, :w]
        conv_out = np.clip(conv_out, 0, 1)
        pred = tensor_to_grid(conv_out, h, w)
        if pred != out:
            return None
    
    # Build ONNX model
    from onnx import helper, TensorProto
    x = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10, 30, 30])
    W = helper.make_tensor("W", TensorProto.FLOAT, [10, 10, 1, 1], weights.flatten().tolist())
    conv = helper.make_node("Conv", ["input", "W"], ["c"], kernel_shape=[1, 1], pads=[0, 0, 0, 0])
    clip = helper.make_node("Clip", ["c"], ["output"], min=0.0, max=1.0)
    graph = helper.make_graph([conv, clip], "g", [x], [y], [W])
    model = helper.make_model(graph, ir_version=10, opset_imports=[helper.make_opsetid("", 10)])
    return model


if __name__ == "__main__":
    print("=" * 60)
    print("NeuroGolf Model Optimizer")
    print("=" * 60)
    
    replaced_count = 0
    kept_count = 0
    identity_count = 0
    conv1x1_count = 0
    failed_count = 0
    total_old_size = 0
    total_new_size = 0
    errors = []
    
    for tid in range(1, 401):
        old_path = SUBDIR_OLD / f"task{tid:03d}.onnx"
        data_path = DATADIR / f"task{tid:03d}.json"
        new_path = SUBDIR_NEW / f"task{tid:03d}.onnx"
        
        # Load task data
        if not data_path.exists():
            errors.append(f"task{tid:03d}: no data")
            shutil.copy2(old_path, new_path)
            total_new_size += new_path.stat().st_size
            continue
        
        with open(data_path) as f:
            data = json.load(f)
        
        old_size = old_path.stat().st_size if old_path.exists() else 0
        total_old_size += old_size
        
        # Strategy 1: Try identity
        examples = data.get("train", []) + data.get("test", [])
        is_id = all(ex["input"] == ex["output"] for ex in examples)
        
        if is_id:
            # Build minimal identity ONNX
            from onnx import helper, TensorProto
            x = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])
            y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10, 30, 30])
            identity = helper.make_node("Identity", ["input"], ["output"])
            graph = helper.make_graph([identity], "g", [x], [y])
            model = helper.make_model(graph, ir_version=10, opset_imports=[helper.make_opsetid("", 10)])
            onnx.save(model, str(new_path))
            new_size = new_path.stat().st_size
            total_new_size += new_size
            identity_count += 1
            print(f"  task{tid:03d}: Identity ✅ ({old_size}B → {new_size}B, saved {old_size-new_size}B)")
            continue
        
        # Strategy 2: Try 1x1 Conv
        conv_model = try_conv1x1(data)
        if conv_model is not None:
            onnx.save(conv_model, str(new_path))
            new_size = new_path.stat().st_size
            total_new_size += new_size
            conv1x1_count += 1
            print(f"  task{tid:03d}: Conv1x1 ✅ ({old_size}B → {new_size}B, saved {old_size-new_size}B)")
            continue
        
        # Strategy 3: Copy original and note
        if old_path.exists():
            shutil.copy2(old_path, new_path)
            new_size = new_path.stat().st_size
            total_new_size += new_size
            kept_count += 1
            if old_size > 50000:
                print(f"  task{tid:03d}: Kept (large {old_size/1024:.1f}KB) ⚠️")
        else:
            errors.append(f"task{tid:03d}: no model")
            failed_count += 1
    
    total_savings = total_old_size - total_new_size
    print(f"\n{'=' * 60}")
    print(f"Results:")
    print(f"  Identity models: {identity_count}")
    print(f"  Conv1x1 models: {conv1x1_count}")
    print(f"  Kept unchanged: {kept_count}")
    print(f"  Failed: {failed_count}")
    print(f"  Total saved: {total_savings} bytes ({total_savings/1024:.1f} KB)")
    print(f"  Before: {total_old_size/1024:.1f} KB")
    print(f"  After: {total_new_size/1024:.1f} KB")
    print(f"  ({(1-total_new_size/total_old_size)*100:.1f}% reduction)")
    print(f"{'=' * 60}")
    
    if errors:
        print(f"\nErrors:")
        for e in errors:
            print(f"  {e}")
