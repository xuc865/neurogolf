"""Final cleanup and packaging."""
import onnx
from onnx import helper
import os, zipfile, shutil

SUBMISSION_DIR = '/Users/wxc/Documents/codes/neurogolf/submission'

# 1) Remove task000.onnx if it exists
task000 = os.path.join(SUBMISSION_DIR, 'task000.onnx')
if os.path.exists(task000):
    os.remove(task000)
    print('Removed task000.onnx')

# 2) Fix task101: clean up unused tensors and empty node name
model_path = os.path.join(SUBMISSION_DIR, 'task101.onnx')
model = onnx.load(model_path)

# Find and remove unused initializers
# Build set of all referenced tensor names
referenced = set()
for node in model.graph.node:
    for inp in node.input:
        if inp: referenced.add(inp)
    for out in node.output:
        if out: referenced.add(out)

# Graph inputs/outputs
for inp in model.graph.input:
    referenced.add(inp.name)
for out in model.graph.output:
    referenced.add(out.name)

# Remove unused initializers
orig_init = len(model.graph.initializer)
new_initializers = [init for init in model.graph.initializer if init.name in referenced]
removed = orig_init - len(new_initializers)
print(f'Removed {removed} unused initializers from task101')

# Rebuild graph without unused initializers
new_graph = helper.make_graph(
    nodes=list(model.graph.node),
    name=model.graph.name,
    inputs=list(model.graph.input),
    outputs=list(model.graph.output),
    initializer=new_initializers,
    value_info=list(model.graph.value_info)
)

# Fix empty node name
for node in new_graph.node:
    if not node.name:
        node.name = f'node_{id(node)}'

new_model = helper.make_model(
    new_graph,
    ir_version=10,
    opset_imports=[helper.make_opsetid('', 11)]
)

# Validate
onnx.checker.check_model(new_model, full_check=True)
print('✅ ONNX checker passed')

# Save
onnx.save(new_model, model_path)

# 3) Count final models
onnx_files = sorted([f for f in os.listdir(SUBMISSION_DIR) if f.endswith('.onnx')])
print(f'\nTotal models: {len(onnx_files)}')

# 4) Verify no duplicate outputs
m = onnx.load(model_path)
from collections import Counter
output_counts = Counter()
for node in m.graph.node:
    for out in node.output:
        if out: output_counts[out] += 1
dups = {n: c for n, c in output_counts.items() if c > 1}
if dups:
    print(f'❌ Still has {len(dups)} duplicate outputs!')
else:
    print('✅ No duplicate outputs')

# Check initializer overlap
init_names = {init.name for init in m.graph.initializer}
input_names = {inp.name for inp in m.graph.input}
overlap = input_names & init_names
if overlap:
    print(f'❌ Input/init overlap: {overlap}')
else:
    print('✅ No input/initializer overlap')

# Check empty names
empty = sum(1 for n in m.graph.node if not n.name)
if empty:
    print(f'⚠️ {empty} empty node names')
else:
    print('✅ No empty node names')

# 5) Package final submission.zip
zip_path = '/Users/wxc/Documents/codes/neurogolf/submission.zip'
with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
    for f in onnx_files:
        file_path = os.path.join(SUBMISSION_DIR, f)
        zf.write(file_path, f)
    print(f'\n✅ submission.zip: {len(onnx_files)} models, {os.path.getsize(zip_path)/1024:.1f} KB')

# 6) Final prediction verification
import onnxruntime as ort
import json, numpy as np
sess = ort.InferenceSession(model_path)
with open(os.path.join(SUBMISSION_DIR, '..', 'data', 'task101.json')) as f:
    task = json.load(f)
def g2oh(grid, h, w):
    oh = np.zeros((10, 30, 30), dtype=np.float32)
    for r in range(h):
        for c in range(w):
            oh[grid[r][c], r, c] = 1.0
    return oh[np.newaxis]
all_ok = True
for phase in ['train', 'test']:
    for i, pair in enumerate(task[phase]):
        inp = pair['input']; out = np.array(pair['output'])
        h, w = len(inp), len(inp[0])
        pred = np.argmax(sess.run(None, {'input': g2oh(inp, h, w)})[0][0, :, :h, :w], axis=0)
        if not np.all(pred == out): all_ok = False
print(f'{"🎉 task101 predictions correct!" if all_ok else "❌ task101 wrong!"}')
