import onnx
import onnxruntime as ort
import json
import numpy as np

# 1) Load the existing model
model = onnx.load('/Users/wxc/Documents/codes/neurogolf/submission/task101.onnx')
print('IR version:', model.ir_version)
for imp in model.opset_import:
    print(f'  Opset: domain={imp.domain}, version={imp.version}')
print('Graph inputs:')
for inp in model.graph.input:
    dims = [d.dim_value for d in inp.type.tensor_type.shape.dim]
    print(f'  {inp.name}: {dims}')
print('Graph outputs:')
for out in model.graph.output:
    dims = [d.dim_value for d in out.type.tensor_type.shape.dim]
    print(f'  {out.name}: {dims}')
print()
print('Nodes:')
for node in model.graph.node:
    print(f'  {node.op_type}: {node.input} -> {node.output}')
print(f'Total nodes: {len(model.graph.node)}')

# 2) Run inference with the existing model
with open('/Users/wxc/Documents/codes/neurogolf/data/task101.json') as f:
    task = json.load(f)

# Convert grid to one-hot
def grid_to_onehot(grid, max_colors=10, max_h=30, max_w=30):
    h, w = len(grid), len(grid[0])
    oh = np.zeros((max_colors, max_h, max_w), dtype=np.float32)
    for r in range(h):
        for c in range(w):
            oh[grid[r][c], r, c] = 1.0
    return oh[np.newaxis, :, :, :]  # [1, C, H, W]
    # Actually, neurogolf expects [1, C, H, W] but ONNX input might be different
    # Let's try both

def onehot_to_grid(onehot, shape=None):
    # onehot: [1, C, H, W] or [C, H, W]
    if onehot.ndim == 4:
        oh = onehot[0]
    else:
        oh = onehot
    oh = oh[:, :oh.shape[1], :oh.shape[2]]  # Trim to non-padded
    grid_val = np.argmax(oh, axis=0)
    return grid_val.tolist()

# Try inference
sess = ort.InferenceSession('/Users/wxc/Documents/codes/neurogolf/submission/task101.onnx')
input_name = sess.get_inputs()[0].name
print(f'\nInput name: {input_name}')
print(f'Input shape: {sess.get_inputs()[0].shape}')
print(f'Output shape: {sess.get_outputs()[0].shape}')

for i, pair in enumerate(task['train']):
    inp = pair['input']
    out = pair['output']
    h, w = len(inp), len(inp[0])
    
    oh = grid_to_onehot(inp)
    # Try running
    result = sess.run(None, {input_name: oh})
    pred = result[0]
    pred_grid = onehot_to_grid(pred)
    
    # Compare
    match = pred_grid == out
    correct = np.sum(match) / (len(out) * len(out[0]))
    print(f'\nTrain {i}: h={h}, w={w}, accuracy={correct*100:.1f}%')
    
    if correct < 1.0:
        print('  Predicted vs Expected (first few rows):')
        for r in range(min(5, len(out))):
            pred_row = ''.join(str(c) for c in pred_grid[r][:20])
            exp_row = ''.join(str(c) for c in out[r][:20])
            diff = ''.join('X' if pred_grid[r][c] != out[r][c] else '.' for c in range(min(20, len(out[0]))))
            print(f'  P: {pred_row}')
            print(f'  E: {exp_row}')
            print(f'  D: {diff}')
    else:
        print('  ✅ Perfect match!')
