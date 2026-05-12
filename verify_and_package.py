"""Verify predictions and package."""
import onnxruntime as ort
import json
import numpy as np
import zipfile, os

SUBMISSION_DIR = '/Users/wxc/Documents/codes/neurogolf/submission'

# 1) Verify task101 predictions
sess = ort.InferenceSession(f'{SUBMISSION_DIR}/task101.onnx')

with open(f'{SUBMISSION_DIR}/../data/task101.json') as f:
    task = json.load(f)

def g2oh(grid, h, w):
    oh = np.zeros((10, 30, 30), dtype=np.float32)
    for r in range(h):
        for c in range(w):
            oh[grid[r][c], r, c] = 1.0
    return oh[np.newaxis]

print("=== task101 prediction verification ===")
all_ok = True
for phase in ['train', 'test']:
    for i, pair in enumerate(task[phase]):
        inp = pair['input']
        out = np.array(pair['output'])
        h, w = len(inp), len(inp[0])
        res = sess.run(None, {'input': g2oh(inp, h, w)})
        pred = np.argmax(res[0][0, :, :h, :w], axis=0)
        ok = np.all(pred == out)
        print(f'  {phase.capitalize()} {i}: {"✅" if ok else "❌"}')
        if not ok: all_ok = False

if all_ok:
    print('  🎉 All predictions correct!')
else:
    print('  ❌ Some predictions wrong!')

# 2) Re-package submission.zip from the submission/ directory
#    The other 399 models are from the clean restore
print('\n=== Packaging submission.zip ===')
zip_path = '/Users/wxc/Documents/codes/neurogolf/submission.zip'
with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
    onnx_files = sorted([f for f in os.listdir(SUBMISSION_DIR) if f.endswith('.onnx')])
    if len(onnx_files) != 400:
        print(f'⚠️ Expected 400 models, found {len(onnx_files)}')
    for f in onnx_files:
        file_path = os.path.join(SUBMISSION_DIR, f)
        zf.write(file_path, f)
    print(f'✅ {len(onnx_files)} models packaged')
    print(f'   Size: {os.path.getsize(zip_path)/1024:.1f} KB')

# 3) Quick verify task101 model structure is clean now
import onnx
m = onnx.load(f'{SUBMISSION_DIR}/task101.onnx')
print(f'\n=== task101 final state ===')
print(f'  Opset: {[imp.version for imp in m.opset_import if not imp.domain][0]}')
print(f'  IR: {m.ir_version}')
print(f'  Nodes: {len(m.graph.node)}')
print(f'  Inputs: {[i.name for i in m.graph.input]}')
print(f'  Duplicate outputs: {len([n for n,c in __import__("collections").Counter(n for node in m.graph.node for n in node.output if n).items() if c>1])}')

# Check for empty node names
empty_names = sum(1 for n in m.graph.node if not n.name)
print(f'  Empty node names: {empty_names}')
