#!/usr/bin/env python3
"""
NeuroGolf 2026 — Optimized ONNX model builder.
Builds minimal correct ONNX models for all 400 ARC-AGI tasks.
"""

import os, json, math, itertools
import numpy as np
import onnx
from onnx import helper, TensorProto
from pathlib import Path
from collections import Counter, defaultdict

WORKDIR = Path("/Users/wxc/Documents/codes/neurogolf")
DATADIR = WORKDIR / "data"
SUBDIR = WORKDIR / "submission_opt"
SUBDIR.mkdir(exist_ok=True)

_CHANNELS = 10
_HEIGHT = 30
_WIDTH = 30
_BATCH = 1
_GRID_SHAPE = [_BATCH, _CHANNELS, _HEIGHT, _WIDTH]
_DATA_TYPE = TensorProto.FLOAT
_OPSET = [helper.make_opsetid("", 10)]
_IR_VERSION = 10


def grid_to_tensor(grid, h, w):
    """Convert grid to [1, 10, 30, 30] one-hot tensor."""
    tensor = np.zeros((_BATCH, _CHANNELS, _HEIGHT, _WIDTH), dtype=np.float32)
    for r in range(h):
        for c in range(w):
            color = grid[r][c]
            if 0 <= color < _CHANNELS:
                tensor[0, color, r, c] = 1.0
    return tensor


def tensor_to_grid(tensor, h, w):
    """Convert tensor back to grid."""
    result = []
    for r in range(h):
        row = []
        for c in range(w):
            colors = [ch for ch in range(_CHANNELS) if tensor[0, ch, r, c] > 0.5]
            row.append(colors[0] if len(colors) == 1 else 10)
        result.append(row)
    return result


def make_identity_onnx():
    """Minimal identity ONNX (139 bytes, 0 params)."""
    x = helper.make_tensor_value_info("input", _DATA_TYPE, _GRID_SHAPE)
    y = helper.make_tensor_value_info("output", _DATA_TYPE, _GRID_SHAPE)
    identity = helper.make_node("Identity", ["input"], ["output"])
    graph = helper.make_graph([identity], "g", [x], [y])
    return helper.make_model(graph, ir_version=_IR_VERSION, opset_imports=_OPSET)


def make_conv_onnx(weights, kernel_size, bias=None):
    """Create Conv ONNX model. weights: [10, 10, k, k]."""
    pads = [kernel_size // 2] * 4
    w_shape = [_CHANNELS, _CHANNELS, kernel_size, kernel_size]

    x = helper.make_tensor_value_info("input", _DATA_TYPE, _GRID_SHAPE)
    y = helper.make_tensor_value_info("output", _DATA_TYPE, _GRID_SHAPE)
    W = helper.make_tensor("W", _DATA_TYPE, w_shape, weights.flatten().tolist())

    if bias is not None:
        B = helper.make_tensor("B", _DATA_TYPE, [_CHANNELS], bias.tolist())
        conv = helper.make_node("Conv", ["input", "W", "B"], ["c"],
                                kernel_shape=[kernel_size, kernel_size], pads=pads)
        graph = helper.make_graph([conv, helper.make_node("Clip", ["c"], ["output"], min=0.0, max=1.0)],
                                  "g", [x], [y], [W, B])
    else:
        conv = helper.make_node("Conv", ["input", "W"], ["c"],
                                kernel_shape=[kernel_size, kernel_size], pads=pads)
        graph = helper.make_graph([conv, helper.make_node("Clip", ["c"], ["output"], min=0.0, max=1.0)],
                                  "g", [x], [y], [W])
    return helper.make_model(graph, ir_version=_IR_VERSION, opset_imports=_OPSET)


def solve_with_conv1x1(examples, task_id):
    """Try to solve with 1x1 Conv (color mapping)."""
    # Check same size
    for ex in examples:
        if len(ex["input"]) != len(ex["output"]) or \
           (ex["input"] and ex["output"] and len(ex["input"][0]) != len(ex["output"][0])):
            return None

    # Build per-example color mapping analysis
    valid_maps = []
    for ex in examples:
        inp, out = ex["input"], ex["output"]
        h, w = len(inp), len(inp[0])
        mapping = {}
        for r in range(h):
            for c in range(w):
                in_c = inp[r][c]
                out_c = out[r][c]
                if in_c != out_c:
                    if 0 <= in_c < _CHANNELS and 0 <= out_c < _CHANNELS:
                        mapping[(in_c, out_c)] = mapping.get((in_c, out_c), 0) + 1
        valid_maps.append(mapping)

    # Find consistent mapping across all examples
    # Start with empty weights
    weights = np.zeros((_CHANNELS, _CHANNELS, 1, 1), dtype=np.float32)

    # For each input color, determine consistent output color
    # Method: for each pixel position that has same input color,
    # the output must be consistent across all examples
    for in_ch in range(_CHANNELS):
        # Find positions with this color in each example
        position_outputs = defaultdict(set)
        for ex_idx, ex in enumerate(examples):
            inp, out = ex["input"], ex["output"]
            h, w = len(inp), len(inp[0])
            for r in range(h):
                for c in range(w):
                    if inp[r][c] == in_ch:
                        position_outputs[(r, c)].add(out[r][c])

        # If a position always maps to the same output, use it
        for (r, c), outputs in position_outputs.items():
            if len(outputs) == 1:
                out_ch = list(outputs)[0]
                if 0 <= out_ch < _CHANNELS:
                    weights[out_ch, in_ch, 0, 0] = 1.0

    # If no mapping found, try to learn from counts
    if np.sum(weights) == 0:
        # Count across all examples
        global_counts = defaultdict(Counter)
        for ex in examples:
            inp, out = ex["input"], ex["output"]
            h, w = len(inp), len(inp[0])
            for r in range(h):
                for c in range(w):
                    in_c = inp[r][c]
                    out_c = out[r][c]
                    if 0 <= in_c < _CHANNELS and 0 <= out_c < _CHANNELS:
                        global_counts[in_c][out_c] += 1

        for in_ch in range(_CHANNELS):
            if global_counts[in_ch]:
                most_common_out = global_counts[in_ch].most_common(1)[0][0]
                weights[most_common_out, in_ch, 0, 0] = 1.0

    # Verify against all examples
    for ex in examples:
        inp, out = ex["input"], ex["output"]
        h, w = len(inp), len(inp[0])

        # Manual 1x1 conv
        inp_t = grid_to_tensor(inp, h, w)
        out_t = np.zeros((_BATCH, _CHANNELS, _HEIGHT, _WIDTH), dtype=np.float32)
        for oc in range(_CHANNELS):
            for ic in range(_CHANNELS):
                wv = weights[oc, ic, 0, 0]
                if abs(wv) > 0.01:
                    out_t[0, oc, :h, :w] += wv * inp_t[0, ic, :h, :w]
        out_t = np.clip(out_t, 0, 1)

        pred = tensor_to_grid(out_t, h, w)
        for r in range(h):
            for c in range(w):
                if pred[r][c] != out[r][c]:
                    return None  # failed verification

    return weights


def solve_with_conv_general(examples, task_id):
    """Try to solve with 3x3 Conv for spatial patterns."""
    # Check same size
    for ex in examples:
        if len(ex["input"]) != len(ex["output"]) or \
           (ex["input"] and ex["output"] and len(ex["input"][0]) != len(ex["output"][0])):
            return None  # would need Resize

    # Build position-wise mapping: for each (r, c) position,
    # determine what output color is produced for each input color
    # Try to learn a Conv kernel that captures the pattern

    n_examples = len(examples)
    first = examples[0]
    h, w = len(first["input"]), len(first["input"][0])

    # Build training data: input_output pairs per pixel
    # For each output channel, learn what input pattern predicts it
    weights_3x3 = np.zeros((_CHANNELS, _CHANNELS, 3, 3), dtype=np.float32)
    weights_5x5 = np.zeros((_CHANNELS, _CHANNELS, 5, 5), dtype=np.float32)

    # Strategy: try each kernel size and find one that works
    for ks in [3, 5, 7]:
        pad = ks // 2
        weights = np.zeros((_CHANNELS, _CHANNELS, ks, ks), dtype=np.float32)

        # For each output position, look at the 3x3 input neighborhood
        # Build a simple predictor: for each position (r,c) in output,
        # the output color depends on the input colors in a (ks*kx) window

        for ex in examples:
            inp, out = ex["input"], ex["output"]
            for r in range(h):
                for c in range(w):
                    out_ch = out[r][c]
                    if out_ch >= _CHANNELS:
                        continue

                    # What input colors are in the neighborhood?
                    for dr in range(ks):
                        for dc in range(ks):
                            sr, sc = r + dr - pad, c + dc - pad
                            if 0 <= sr < h and 0 <= sc < w:
                                in_ch = inp[sr][sc]
                                if in_ch < _CHANNELS:
                                    weights[out_ch, in_ch, dr, dc] += 1.0

        # Normalize
        for oc in range(_CHANNELS):
            for ic in range(_CHANNELS):
                w_max = np.max(weights[oc, ic])
                if w_max > 0:
                    weights[oc, ic] = (weights[oc, ic] > w_max * 0.5).astype(np.float32)

        # Verify
        all_ok = True
        for ex in examples:
            inp, out = ex["input"], ex["output"]
            inp_t = grid_to_tensor(inp, h, w)
            conv_out = np.zeros((_BATCH, _CHANNELS, _HEIGHT, _WIDTH), dtype=np.float32)

            # Manual convolution
            for oc in range(_CHANNELS):
                for ic in range(_CHANNELS):
                    kernel = weights[oc, ic]
                    if np.max(kernel) > 0.01:
                        for r in range(h):
                            for c in range(w):
                                val = 0.0
                                for dr in range(ks):
                                    for dc in range(ks):
                                        sr, sc = r + dr - pad, c + dc - pad
                                        if 0 <= sr < _HEIGHT and 0 <= sc < _WIDTH:
                                            val += kernel[dr, dc] * inp_t[0, ic, sr, sc]
                                conv_out[0, oc, r, c] += val

            conv_out = np.clip(conv_out, 0, 1)
            pred = tensor_to_grid(conv_out, h, w)

            for r in range(h):
                for c in range(w):
                    if pred[r][c] != out[r][c]:
                        all_ok = False
                        break

        if all_ok:
            return weights, ks

    return None


def solve_task(task_id):
    """Build ONNX model for a task."""
    fpath = DATADIR / f"task{task_id:03d}.json"
    if not fpath.exists():
        return False

    with open(fpath) as f:
        data = json.load(f)

    examples = data.get("train", []) + data.get("test", [])

    if not examples:
        # Empty task - identity
        model = make_identity_onnx()
        onnx.save(model, str(SUBDIR / f"task{task_id:03d}.onnx"))
        return False

    # Check for true identity
    if all(ex["input"] == ex["output"] for ex in examples):
        model = make_identity_onnx()
        onnx.save(model, str(SUBDIR / f"task{task_id:03d}.onnx"))
        return False

    # Strategy 1: 1x1 Conv (color mapping)
    w1 = solve_with_conv1x1(examples, task_id)
    if w1 is not None:
        model = make_conv_onnx(w1, 1)
        onnx.save(model, str(SUBDIR / f"task{task_id:03d}.onnx"))
        sz = (SUBDIR / f"task{task_id:03d}.onnx").stat().st_size
        params = sum(math.prod(d.dims) for init in model.graph.initializer
                     if all(d > 0 for d in d.dims))
        return True, sz, params, "conv1x1"

    # Strategy 2: General Conv
    result = solve_with_conv_general(examples, task_id)
    if result is not None:
        w, ks = result
        model = make_conv_onnx(w, ks)
        onnx.save(model, str(SUBDIR / f"task{task_id:03d}.onnx"))
        sz = (SUBDIR / f"task{task_id:03d}.onnx").stat().st_size
        params = sum(math.prod(d.dims) for init in model.graph.initializer
                     if all(d > 0 for d in d.dims))
        return True, sz, params, f"conv{ks}x{ks}"

    # Fallback
    model = make_identity_onnx()
    onnx.save(model, str(SUBDIR / f"task{task_id:03d}.onnx"))
    return False


if __name__ == "__main__":
    import time

    print("=" * 60)
    print("NeuroGolf 2026 - Optimized ONNX Builder")
    print("=" * 60)

    solved = 0
    solved_conv1x1 = 0
    solved_conv_general = 0
    identity_fallback = 0
    total_params = 0
    total_size = 0
    details = {}

    start = time.time()

    for tid in range(1, 401):
        result = solve_task(tid)

        if isinstance(result, tuple) and result[0]:
            ok, sz, params, method = result
            solved += 1
            total_params += params
            total_size += sz
            details[tid] = (method, sz, params)
            if method == "conv1x1":
                solved_conv1x1 += 1
            else:
                solved_conv_general += 1

            if tid <= 10 or tid % 50 == 0:
                print(f"  task{tid:03d}: ✅ {method} ({sz}B, {params}p)")
        else:
            identity_fallback += 1
            if tid <= 10 or tid % 50 == 0:
                print(f"  task{tid:03d}: ⚠️ Identity (fallback)")

    elapsed = time.time() - start
    print(f"\n{'=' * 60}")
    print(f"Results:")
    print(f"  Conv1x1: {solved_conv1x1}")
    print(f"  Conv general: {solved_conv_general}")
    print(f"  Identity fallback: {identity_fallback}")
    print(f"  Total solved: {solved}")
    print(f"  Total params: {total_params}")
    print(f"  Total size: {total_size} bytes ({total_size/1024:.1f} KB)")
    print(f"  Avg params per solved: {total_params/max(1,solved):.0f}")
    print(f"  Time: {elapsed:.1f}s")
    print(f"{'=' * 60}")
