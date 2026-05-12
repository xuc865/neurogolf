"""Restore clean models from submission.zip and fix only task101."""
import onnx, os, shutil
from onnx import helper
import onnxruntime as ort

ZIP = '/Users/wxc/Documents/codes/neurogolf/submission.zip'
TARGET = '/Users/wxc/Documents/codes/neurogolf/submission'
workdir = '/tmp/neurogolf_restore'

# 1. Re-extract clean models from the intact submission.zip
import subprocess
subprocess.run(['unzip', '-o', ZIP, '-d', workdir], capture_output=True)

# 2. Copy all clean models back
files = sorted([f for f in os.listdir(workdir) if f.endswith('.onnx')])
print(f'Extracted {len(files)} clean models')

for f in files:
    shutil.copy2(os.path.join(workdir, f), os.path.join(TARGET, f))

# 3. Check task101's current state
m = onnx.load(os.path.join(TARGET, 'task101.onnx'))
for imp in m.opset_import:
    if not imp.domain: print(f'task101 restore: opset={imp.version}, IR={m.ir_version}')

# 4. Set IR version to 10 and opset to 20 (well within Kaggle's support range)
print('\nFixing task101: opset 11 -> 11, IR 8 -> 10...')
m.ir_version = 10
while m.opset_import:
    m.opset_import.pop()
m.opset_import.append(helper.make_opsetid('', 11))

# Also need to remove unused initializers from graph inputs
# Check for input/initializer overlap
init_names = {init.name for init in m.graph.initializer}
input_names = {inp.name for inp in m.graph.input}
overlap = input_names & init_names
if overlap:
    print(f'  Removing {len(overlap)} inputs that overlap initializers')
    new_inputs = [inp for inp in m.graph.input if inp.name not in init_names]
    m = helper.make_model(
        helper.make_graph(
            list(m.graph.node), m.graph.name, new_inputs,
            list(m.graph.output), list(m.graph.initializer)
        ),
        ir_version=10,
        opset_imports=[helper.make_opsetid('', 11)]
    )

onnx.save(m, os.path.join(TARGET, 'task101.onnx'))
print('  Saved task101 with opset=11, IR=10')

# 5. Verify ALL models load
print(f'\nVerifying all {len(files)} models...')
bad = []
for f in sorted(files):
    path = os.path.join(TARGET, f)
    try:
        sess = ort.InferenceSession(path)
        inp = [d.dim_value for d in sess.get_inputs()[0].shape]
        out = [d.dim_value for d in sess.get_outputs()[0].shape]
        if inp != [1, 10, 30, 30] or out != [1, 10, 30, 30]:
            bad.append((f, f'shape {inp}->{out}'))
    except Exception as e:
        bad.append((f, str(e)[:60]))

if bad:
    print(f'\n❌ {len(bad)} bad models:')
    for name, err in bad[:10]:
        print(f'  {name}: {err}')
else:
    print(f'\n✅ All {len(files)} models load with correct shapes!')

# 6. Specifically verify task101 predicts correctly
print('\nVerifying task101 predictions...')
sess = ort.InferenceSession(os.path.join(TARGET, 'task101.onnx'))
import json, numpy as np
with open('/Users/wxc/Documents/codes/neurogolf/data/task101.json') as f:
    task = json.load(f)

def g2oh(grid, h, w):
    oh = np.zeros((10, 30, 30), dtype=np.float32)
    for r in range(h):
        for c in range(w):
            oh[grid[r][c], r, c] = 1.0
    return oh[np.newaxis]

all_ok = True
for i, pair in enumerate(task['train']):
    inp = pair['input']
    out = np.array(pair['output'])
    h, w = len(inp), len(inp[0])
    res = sess.run(None, {'input': g2oh(inp, h, w)})
    pred = np.argmax(res[0][0, :, :h, :w], axis=0)
    ok = np.all(pred == out)
    print(f'  Train {i}: {"✅" if ok else "❌"}')
    if not ok: all_ok = False

for i, pair in enumerate(task['test']):
    inp = pair['input']
    out = np.array(pair['output'])
    h, w = len(inp), len(inp[0])
    res = sess.run(None, {'input': g2oh(inp, h, w)})
    pred = np.argmax(res[0][0, :, :h, :w], axis=0)
    ok = np.all(pred == out)
    print(f'  Test {i}:  {"✅" if ok else "❌"}')
    if not ok: all_ok = False

if all_ok:
    print('  🎉 task101 predictions all correct!')
