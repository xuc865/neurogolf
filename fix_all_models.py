"""Fix model compatibility issues for Kaggle."""
import onnx
from onnx import helper, TensorProto, numpy_helper
import onnxruntime as ort
import json
import numpy as np

SUBMISSION_DIR = '/Users/wxc/Documents/codes/neurogolf/submission'

# ==============================
# 1) Fix task101: downgrade opset 24 -> 11
# ==============================
print("=== Fixing task101 (opset 24 -> 11) ===")
model = onnx.load(f'{SUBMISSION_DIR}/task101.onnx')
model.opset_import.pop(0)  # Remove opset 24
# Keep opset 11
for imp in model.opset_import:
    if imp.domain == '' or imp.domain == 'ai.onnx':
        imp.version = 11
# Also fix IR version
model.ir_version = 7  # onnx IR v7 is widely supported
onnx.save(model, f'{SUBMISSION_DIR}/task101.onnx')
print("  Saved with opset 11")

# Verify it loads
sess = ort.InferenceSession(f'{SUBMISSION_DIR}/task101.onnx')
print(f"  ✅ Loads OK: {sess.get_inputs()[0].shape} -> {sess.get_outputs()[0].shape}")

# Verify it still predicts correctly
def grid_to_onehot(grid, max_colors=10, max_h=30, max_w=30):
    h, w = len(grid), len(grid[0])
    oh = np.zeros((max_colors, max_h, max_w), dtype=np.float32)
    for r in range(h):
        for c in range(w):
            oh[grid[r][c], r, c] = 1.0
    return oh[np.newaxis, :, :, :]

with open(f'{SUBMISSION_DIR}/../data/task101.json') as f:
    task = json.load(f)
for i, pair in enumerate(task['train']):
    inp = pair['input']
    out = np.array(pair['output'])
    h, w = len(inp), len(inp[0])
    oh = grid_to_onehot(inp)
    result = sess.run(None, {'input': oh})
    pred = np.argmax(result[0][0, :, :h, :w], axis=0)
    assert np.all(pred == out), f"train {i} failed"
print("  ✅ All predictions correct")

# ==============================
# 2) Fix task096: replace ArgMax
# ==============================
print("\n=== Fixing task096 (ArgMax -> manual) ===")
# Analyze task096 pattern
with open(f'{SUBMISSION_DIR}/../data/task096.json') as f:
    task = json.load(f)
# Check first train pair to understand
t0 = task['train'][0]
inp, out = np.array(t0['input']), np.array(t0['output'])
print(f"  Pattern: {inp.shape} -> {out.shape}")
# Check if it's just identity or has a simple transformation
if np.array_equal(inp, out):
    print("  Simple identity! Building identity model")
    # Build identity model
    X = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 10, 30, 30])
    Y = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 10, 30, 30])
    identity = helper.make_node('Identity', ['input'], ['output'])
    graph = helper.make_graph([identity], 'task096', [X], [Y])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 11)])
    model.ir_version = 7
    onnx.save(model, f'{SUBMISSION_DIR}/task096.onnx')
else:
    # Need to find what the transformation is
    diffs = []
    for i, pair in enumerate(task['train']):
        inp = np.array(pair['input'])
        out = np.array(pair['output'])
        diff = (inp != out).sum()
        print(f"  Train {i}: {inp.shape} -> {out.shape}, {diff} cells differ")
        diffs.append(diff)
    print(f"  Train shapes: {[p['input'] for p in task['train']]}")
    
    # Check the test pair
    for i, pair in enumerate(task['test']):
        inp = np.array(pair['input'])
        out = np.array(pair['output'])
        print(f"  Test {i}: {inp.shape} -> {out.shape}")

# ==============================
# 3) Fix task118: replace Equal
# ==============================
print("\n=== Fixing task118 (Equal -> manual) ===")
with open(f'{SUBMISSION_DIR}/../data/task118.json') as f:
    task = json.load(f)
for i, pair in enumerate(task['train']):
    inp = np.array(pair['input'])
    out = np.array(pair['output'])
    print(f"  Train {i}: {inp.shape} -> {out.shape}, diff={(inp!=out).sum()}")

# ==============================
# 4) Fix task266: IR version 13 -> 7
# ==============================
print("\n=== Fixing task266 (IR v13 -> v7) ===")
model = onnx.load(f'{SUBMISSION_DIR}/task266.onnx')
model.ir_version = 7
# Also fix opset to 11
for imp in model.opset_import:
    if imp.domain == '' or imp.domain == 'ai.onnx':
        imp.version = 11
onnx.save(model, f'{SUBMISSION_DIR}/task266.onnx')
# Verify
sess = ort.InferenceSession(f'{SUBMISSION_DIR}/task266.onnx')
print(f"  ✅ Loads OK")

# Final verification of all 4
print("\n=== Final verification ===")
for tid in ['096', '101', '118', '266']:
    try:
        sess = ort.InferenceSession(f'{SUBMISSION_DIR}/task{tid}.onnx')
        print(f"  task{tid}: ✅")
    except Exception as e:
        print(f"  task{tid}: ❌ {str(e)[:80]}")
