#!/usr/bin/env python3
"""
NeuroGolf 2026: Verify all ONNX models locally and rebuild minimal submission.
"""
import os, json, math, shutil
import numpy as np
import onnx
from onnx import helper, TensorProto
from pathlib import Path

WORKDIR = Path("/Users/wxc/Documents/codes/neurogolf")
DATADIR = WORKDIR / "data"
SUBDIR_CURRENT = WORKDIR / "submission"
SUBDIR_MINIMAL = WORKDIR / "submission_minimal"
SUBDIR_MINIMAL.mkdir(exist_ok=True)

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

def make_identity_model():
    x = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10, 30, 30])
    identity = helper.make_node("Identity", ["input"], ["output"])
    graph = helper.make_graph([identity], "g", [x], [y])
    return helper.make_model(graph, ir_version=10, opset_imports=[helper.make_opsetid("", 10)])

def build_conv1x1_from_data(data):
    """Build a 1x1 Conv model from training data examples."""
    examples = data.get("train", []) + data.get("test", [])
    
    # Check same size
    for ex in examples:
        inp, out = ex["input"], ex["output"]
        if len(inp) != len(out) or (inp and out and len(inp[0]) != len(out[0])):
            return None
    
    # Build color mapping from examples
    # For each position, track what input color maps to what output
    from collections import defaultdict, Counter
    pos_color_map = defaultdict(lambda: defaultdict(Counter))
    
    for ex in examples:
        inp, out = ex["input"], ex["output"]
        h, w = len(inp), len(inp[0])
        for r in range(h):
            for c in range(w):
                in_c = inp[r][c]
                out_c = out[r][c]
                if 0 <= in_c < _CHANNELS and 0 <= out_c < _CHANNELS:
                    pos_color_map[(r, c)][in_c][out_c] += 1
    
    # Check if any position has non-deterministic mapping
    for pos, in_map in pos_color_map.items():
        for in_c, out_counts in in_map.items():
            if len(out_counts) > 1:
                return None
    
    # Build weight matrix as indicator
    weights = np.zeros((_CHANNELS, _CHANNELS, 1, 1), dtype=np.float32)
    for (r, c), in_map in pos_color_map.items():
        for in_c, out_counts in in_map.items():
            out_c = list(out_counts.keys())[0]
            weights[out_c, in_c, 0, 0] = 1.0
    
    # Verify against all examples
    for ex in examples:
        inp, out = ex["input"], ex["output"]
        h, w = len(inp), len(inp[0])
        inp_t = grid_to_tensor(inp)
        conv_out = np.zeros_like(inp_t)
        for oc in range(_CHANNELS):
            for ic in range(_CHANNELS):
                wv = weights[oc, ic, 0, 0]
                if abs(wv) > 0.01:
                    conv_out[0, oc, :h, :w] += wv * inp_t[0, ic, :h, :w]
        conv_out = np.clip(conv_out, 0, 1)
        pred = tensor_to_grid(conv_out, h, w)
        if pred != out:
            return None
    
    # Build ONNX
    x = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10, 30, 30])
    W = helper.make_tensor("W", TensorProto.FLOAT, [10, 10, 1, 1], weights.flatten().tolist())
    conv = helper.make_node("Conv", ["input", "W"], ["c"], kernel_shape=[1, 1], pads=[0, 0, 0, 0])
    clip = helper.make_node("Clip", ["c"], ["output"], min=0.0, max=1.0)
    graph = helper.make_graph([conv, clip], "g", [x], [y], [W])
    return helper.make_model(graph, ir_version=10, opset_imports=[helper.make_opsetid("", 10)])

def run_model_check(model, data):
    """Verify ONNX model against training+test examples."""
    import onnxruntime as ort
    examples = data.get("train", []) + data.get("test", [])
    if not examples:
        return False
    
    try:
        session = ort.InferenceSession(model.SerializeToString())
    except:
        return False
    
    for ex in examples:
        inp_t = grid_to_tensor(ex["input"])
        try:
            out_t = session.run(["output"], {"input": inp_t})[0]
        except:
            return False
        
        out_t = (out_t > 0.0).astype(np.float32)
        h, w = len(ex["output"]), len(ex["output"][0])
        pred = tensor_to_grid(out_t, h, w)
        if pred != ex["output"]:
            return False
    return True


if __name__ == "__main__":
    print("=" * 60)
    print("NeuroGolf Model Verification & Minimization")
    print("=" * 60)
    
    stats = {
        "pass_original": 0,
        "pass_minimal": 0,
        "identity_works": 0,
        "conv1x1_works": 0,
        "fallback_original": 0,
        "fallback_identity": 0,
    }
    
    total_old_size = 0
    total_new_size = 0
    
    for tid in range(1, 401):
        old_path = SUBDIR_CURRENT / f"task{tid:03d}.onnx"
        data_path = DATADIR / f"task{tid:03d}.json"
        new_path = SUBDIR_MINIMAL / f"task{tid:03d}.onnx"
        
        if not data_path.exists():
            shutil.copy2(old_path, new_path)
            continue
        
        with open(data_path) as f:
            data = json.load(f)
        
        old_size = old_path.stat().st_size if old_path.exists() else 0
        total_old_size += old_size
        
        # Load original model
        try:
            orig_model = onnx.load(str(old_path))
        except:
            orig_model = None
        
        # Check if original passes
        orig_pass = False
        if orig_model is not None:
            orig_pass = run_model_check(orig_model, data)
            if orig_pass:
                stats["pass_original"] += 1
        
        # Try identity
        identity_model = make_identity_model()
        id_pass = run_model_check(identity_model, data)
        
        if id_pass:
            onnx.save(identity_model, str(new_path))
            stats["identity_works"] += 1
            new_size = new_path.stat().st_size
            total_new_size += new_size
            print(f"  task{tid:03d}: Identity ✅ ({old_size}B → {new_size}B)")
            continue
        
        # Try Conv1x1
        conv_model = build_conv1x1_from_data(data)
        if conv_model is not None:
            conv_pass = run_model_check(conv_model, data)
            if conv_pass:
                onnx.save(conv_model, str(new_path))
                stats["conv1x1_works"] += 1
                new_size = new_path.stat().st_size
                total_new_size += new_size
                print(f"  task{tid:03d}: Conv1x1 ✅ ({old_size}B → {new_size}B)")
                continue
        
        # Keep original if it passes
        if orig_pass:
            shutil.copy2(old_path, new_path)
            stats["fallback_original"] += 1
            new_size = new_path.stat().st_size
            total_new_size += new_size
            print(f"  task{tid:03d}: Original (kept) ({old_size}B)")
        else:
            # Original failed - use identity as best guess
            onnx.save(identity_model, str(new_path))
            stats["fallback_identity"] += 1
            new_size = new_path.stat().st_size
            total_new_size += new_size
            print(f"  task{tid:03d}: ORIG FAILED → Identity ⚠️")
    
    print(f"\n{'=' * 60}")
    print("Verification Results:")
    print(f"  Original models passing:     {stats['pass_original']}/400")
    print(f"  Identity models working:     {stats['identity_works']}/400")
    print(f"  Conv1x1 models working:      {stats['conv1x1_works']}/400")
    print(f"  Kept original (passing):     {stats['fallback_original']}/400")
    print(f"  Fallback to identity:        {stats['fallback_identity']}/400")
    print(f"\nSize:")
    print(f"  Before: {total_old_size/1024:.1f} KB")
    print(f"  After:  {total_new_size/1024:.1f} KB")
    print(f"  Saved:  {(total_old_size-total_new_size)/1024:.1f} KB")
    print(f"{'=' * 60}")
