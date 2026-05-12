#!/usr/bin/env python3
"""
Smart NeuroGolf solver - handles multiple ARC-AGI patterns.
Creates ONNX models with appropriate kernels for each task type.
"""
import json, os, math, itertools, numpy as np
import onnx
import onnx.helper

BASE = "/Users/wxc/workspace/neurogolf-2026"
DATA_DIR = f"{BASE}/data"
OUTPUT_DIR = f"{BASE}/models"
os.makedirs(OUTPUT_DIR, exist_ok=True)

_CH = 10; _H = 30; _W = 30
_GRID = [1, _CH, _H, _W]
_FLOAT = onnx.TensorProto.FLOAT
_IR = 10
_OPS = [onnx.helper.make_opsetid("", 10)]


def load_task(tn):
    with open(f"{DATA_DIR}/task{tn:03d}.json") as f:
        return json.load(f)


def make_conv(weights_10x10xk, k=1):
    """weights: 10-out x 10-in x k x k list of floats, or a 4D list."""
    ws = [_CH, _CH, k, k]
    pads = [k//2]*4
    flat = [float(weights_10x10xk[o][i][r][c]) for o in range(10) for i in range(10) for r in range(k) for c in range(k)]
    x = onnx.helper.make_tensor_value_info("input", _FLOAT, _GRID)
    y = onnx.helper.make_tensor_value_info("output", _FLOAT, _GRID)
    w = onnx.helper.make_tensor("W", _FLOAT, ws, flat)
    node = onnx.helper.make_node("Conv", ["input", "W"], ["output"], kernel_shape=[k,k], pads=pads)
    g = onnx.helper.make_graph([node], "graph", [x], [y], [w])
    return onnx.helper.make_model(g, ir_version=_IR, opset_imports=_OPS)


def identity_weights(k=1):
    """Create identity kernel."""
    w = [[[[0.0]*k for _ in range(k)] for _ in range(10)] for _ in range(10)]
    for c in range(10):
        w[c][c][k//2][k//2] = 1.0
    return w


def color_map_weights(color_map, k=1):
    """Color remap: map input color i to output color color_map[i]."""
    w = [[[[0.0]*k for _ in range(k)] for _ in range(10)] for _ in range(10)]
    for i in range(10):
        o = color_map.get(i, i)
        w[o][i][k//2][k//2] = 1.0
    return w


def solve_pixel_wise(task_data):
    """Solve tasks where transformation is position-independent (pixel-wise)."""
    train = task_data.get('train', [])
    if not train:
        return None
    
    h, w = len(train[0]['input']), len(train[0]['input'][0])
    ho, wo = len(train[0]['output']), len(train[0]['output'][0])
    if h != ho or w != wo:
        return None
    
    # Check per-pixel mapping
    mapping = {}
    for ex in train:
        inp, out = ex['input'], ex['output']
        for r in range(len(inp)):
            for c in range(len(inp[0])):
                ci, co = inp[r][c], out[r][c]
                if ci not in mapping:
                    mapping[ci] = co
                elif mapping[ci] != co:
                    return None  # Not pixel-wise
    
    if not mapping:
        return None
    
    is_id = all(mapping.get(k, k) == k for k in range(10))
    if is_id:
        return make_conv(identity_weights(1), 1)
    else:
        return make_conv(color_map_weights(mapping, 1), 1)


def solve_crop_black(task_data):
    """Detect and solve tasks that crop black (0) borders."""
    train = task_data.get('train', [])
    if not train:
        return None
    
    for ex in train:
        inp = ex['input']
        out = ex['output']
        h, w = len(inp), len(inp[0])
        ho, wo = len(out), len(out[0])
        
        # Check if output is a subgrid of input inside black borders
        if ho > h or wo > w:
            continue
        
        # Find the non-black subgrid in input
        rows = [r for r in range(h) if any(inp[r][c] != 0 for c in range(w))]
        cols = [c for c in range(w) if any(inp[r][c] != 0 for r in range(h))]
        
        if rows and cols:
            r_min, r_max = min(rows), max(rows)+1
            c_min, c_max = min(cols), max(cols)+1
            crop_h, crop_w = r_max - r_min, c_max - c_min
            
            if crop_h == ho and crop_w == wo:
                # Check if cropped input == output
                cropped = [row[c_min:c_max] for row in inp[r_min:r_max]]
                if cropped == out:
                    return ('crop_black', r_min, c_min, r_max, c_max)
    
    return None


def solve_color_recolor(task_data):
    """Detect color recolor (all pixels of one color become another)."""
    train = task_data.get('train', [])
    if not train:
        return None
    
    # Look for tasks where specific colors are recolored
    mapping = {}
    for ex in train:
        inp, out = ex['input'], ex['output']
        h, w = len(inp), len(inp[0])
        ho, wo = len(out), len(out[0])
        if h != ho or w != wo:
            return None
        
        for r in range(h):
            for c in range(w):
                ci, co = inp[r][c], out[r][c]
                if ci not in mapping:
                    mapping[ci] = co
                elif mapping[ci] != co:
                    # Check if this is position-dependent
                    pass
    
    return None


def solve_by_example(task_data):
    """Try to solve by directly implementing the pattern from examples."""
    train = task_data.get('train', [])
    if not train:
        return None
    
    # Check for same-size tasks
    first_in = len(train[0]['input']), len(train[0]['input'][0])
    first_out = len(train[0]['output']), len(train[0]['output'][0])
    
    # Variable sizes - need a general approach
    all_same_in = all((len(ex['input']), len(ex['input'][0])) == first_in for ex in train)
    all_same_out = all((len(ex['output']), len(ex['output'][0])) == first_out for ex in train)
    
    if not all_same_in or not all_same_out:
        return None
    
    h, w = first_in
    ho, wo = first_out
    
    if h == ho and w == wo:
        return solve_same_size_grid(task_data, h, w)
    
    return None


def solve_same_size_grid(task_data, h, w):
    """Solve same-size grid task by creating a convolution that implements the pattern."""
    train = task_data.get('train', [])
    
    # Step 1: Check if it's a simple uniform transformation
    # A uniform transform means: same input color always maps to same output color
    # (already handled by solve_pixel_wise)
    
    # Step 2: Build a per-position lookup table
    # If position (r,c) with color ci always maps to co
    pos_maps = {}
    consistent = True
    for ex in train:
        inp, out = ex['input'], ex['output']
        hi, wi = len(inp), len(inp[0])
        ho, wo = len(out), len(out[0])
        
        for r in range(min(hi, ho)):
            for c in range(min(wi, wo)):
                ci, co = inp[r][c], out[r][c]
                key = (r, c, ci)
                if key not in pos_maps:
                    pos_maps[key] = co
                elif pos_maps[key] != co:
                    consistent = False
    
    if consistent and len(pos_maps) > 0:
        # Position-specific color mapping
        # Can be implemented as a 30x30 lookup per color
        # But in ONNX, this is hardest without Gather/Scatter
        # Best we can do is a large kernel conv or identity as fallback
        return None  # Too complex for simple conv
    
    # Step 3: Try to learn a 3x3 conv kernel from training data
    # Build I/O pairs as numpy arrays
    xs = []
    ys = []
    for ex in train:
        inp, out = ex['input'], ex['output']
        inp_np = np.zeros((1, 10, 30, 30), dtype=np.float32)
        out_np = np.zeros((1, 10, 30, 30), dtype=np.float32)
        hi, wi = len(inp), len(inp[0])
        ho, wo = len(out), len(out[0])
        for r in range(hi):
            for c in range(wi):
                inp_np[0][inp[r][c]][r][c] = 1.0
        for r in range(ho):
            for c in range(wo):
                out_np[0][out[r][c]][r][c] = 1.0
        xs.append(inp_np)
        ys.append(out_np)
    
    X = np.concatenate(xs, axis=0) if xs else None
    Y = np.concatenate(ys, axis=0) if ys else None
    
    if X is None or Y is None:
        return None
    
    # Try 1x1 conv that minimizes MSE
    # weight shape: [10, 10, 1, 1]
    n = len(xs)
    # For each position, solve: output_channel = W[o][i] * input_channel
    # Since each pixel has exactly one 1, the optimal weight is:
    # W[o][i] = mean over pixels where input=i of (output=o?)
    # For a 1x1 conv, the output at position (r,c) is sum_i(W[o][i] * X[i][r][c])
    # Since X has exactly one 1 per position, W[o][i] should be 1 when i maps to o
    
    # Actually, let me use the consistent mapping approach
    # Build a position-dependent mapping
    pos_color_map = {}
    for ex in train:
        inp, out = ex['input'], ex['output']
        for r in range(len(inp)):
            for c in range(len(inp[0])):
                key = (r, c)
                ci = inp[r][c]
                co = out[r][c]
                if key not in pos_color_map:
                    pos_color_map[key] = {}
                if ci not in pos_color_map[key]:
                    pos_color_map[key][ci] = co
                elif pos_color_map[key][ci] != co:
                    pass  # Inconsistent
    
    # Build weight function from position-dependent mapping
    # Use a 1x1 conv but with position-specific weights
    # Actually 1x1 conv can't do position-specific... we need 3x3 or larger
    # or multiple layers
    
    return None


def build_all():
    """Build best-effort ONNX models for all tasks."""
    stats = {'color_remap': 0, 'identity': 0, 'complex_identity': 0, 'failed': 0}
    
    for tn in range(1, 401):
        try:
            data = load_task(tn)
            
            # Try easy solvers first
            model = solve_pixel_wise(data)
            if model:
                stat = 'color_remap'
            else:
                # Fallback to identity
                model = make_conv(identity_weights(1), 1)
                stat = 'complex_identity'
            
            onnx.save(model, f"{OUTPUT_DIR}/task{tn:03d}.onnx")
            stats[stat] = stats.get(stat, 0) + 1
            
        except Exception as e:
            stats['failed'] = stats.get('failed', 0) + 1
            try:
                model = make_conv(identity_weights(1), 1)
                onnx.save(model, f"{OUTPUT_DIR}/task{tn:03d}.onnx")
            except:
                pass
    
    print(f"Stats: {stats}")


if __name__ == "__main__":
    build_all()
    
    # Verify a few
    import onnxruntime as ort
    
    for tn in [1, 5, 10, 16, 50, 100, 150, 200, 267]:
        try:
            data = load_task(tn)
            session = ort.InferenceSession(f"{OUTPUT_DIR}/task{tn:03d}.onnx")
            correct = 0
            total = 0
            for ex in data.get('train', []):
                inp, out = ex['input'], ex['output']
                h, w = len(inp), len(inp[0])
                ho, wo = len(out), len(out[0])
                if max(h, w, ho, wo) > 30:
                    continue
                inp_np = np.zeros((1, 10, 30, 30), dtype=np.float32)
                out_np = np.zeros((1, 10, 30, 30), dtype=np.float32)
                for r in range(h):
                    for c in range(w):
                        inp_np[0][inp[r][c]][r][c] = 1.0
                for r in range(ho):
                    for c in range(wo):
                        out_np[0][out[r][c]][r][c] = 1.0
                result = session.run(["output"], {"input": inp_np})
                pred = (result[0] > 0.0).astype(float)
                if np.array_equal(pred[:,:,:ho,:wo], out_np[:,:,:ho,:wo]):
                    correct += 1
                total += 1
            status = "✓" if correct == total else "✗"
            print(f"  {status} task{tn:03d}: {correct}/{total}")
        except Exception as e:
            print(f"  ! task{tn:03d}: {e}")
