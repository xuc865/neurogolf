import onnxruntime as ort
import json
import numpy as np

with open('/Users/wxc/Documents/codes/neurogolf/data/task101.json') as f:
    task = json.load(f)

def grid_to_onehot(grid, max_colors=10, max_h=30, max_w=30):
    h, w = len(grid), len(grid[0])
    oh = np.zeros((max_colors, max_h, max_w), dtype=np.float32)
    for r in range(h):
        for c in range(w):
            oh[grid[r][c], r, c] = 1.0
    return oh[np.newaxis, :, :, :]  # [1, 10, 30, 30]

def onehot_to_grid(onehot, h, w):
    # onehot: [1, 10, 30, 30]
    oh = onehot[0, :, :h, :w]  # crop to actual h,w first
    grid_val = np.argmax(oh, axis=0)
    return grid_val  # [h, w]

sess = ort.InferenceSession('/Users/wxc/Documents/codes/neurogolf/submission/task101.onnx')
input_name = sess.get_inputs()[0].name

all_correct = True
for i, pair in enumerate(task['train']):
    inp = pair['input']
    out = np.array(pair['output'])
    h, w = len(inp), len(inp[0])
    
    oh = grid_to_onehot(inp)
    result = sess.run(None, {input_name: oh})
    pred_grid = onehot_to_grid(result[0], h, w)
    
    match = np.all(pred_grid == out)
    if match:
        print(f'Train {i}: ✅ PASS (h={h}, w={w})')
    else:
        all_correct = False
        errors = (pred_grid != out).sum()
        total = h * w
        print(f'Train {i}: ❌ FAIL - {errors}/{total} cells wrong')
        # Show first mismatch rows
        for r in range(h):
            pr = ''.join(str(int(c)) for c in pred_grid[r])
            ex = ''.join(str(int(c)) for c in out[r])
            if pr != ex:
                diff = ''.join('X' if pred_grid[r][c]!=out[r][c] else '.' for c in range(w))
                print(f'  Row {r}: P={pr}')
                print(f'  Row {r}: E={ex}')
                print(f'  Row {r}: D={diff}')

# Also test the test pair
for i, pair in enumerate(task['test']):
    inp = pair['input']
    out = np.array(pair['output'])
    h, w = len(inp), len(inp[0])
    
    oh = grid_to_onehot(inp)
    result = sess.run(None, {input_name: oh})
    pred_grid = onehot_to_grid(result[0], h, w)
    
    match = np.all(pred_grid == out)
    if match:
        print(f'Test {i}:  ✅ PASS (h={h}, w={w})')
    else:
        all_correct = False
        errors = (pred_grid != out).sum()
        total = h * w
        print(f'Test {i}:  ❌ FAIL - {errors}/{total} cells wrong')
        for r in range(h):
            pr = ''.join(str(int(c)) for c in pred_grid[r])
            ex = ''.join(str(int(c)) for c in out[r])
            if pr != ex:
                print(f'  Row {r}: P={pr}')
                print(f'  Row {r}: E={ex}')

if all_correct:
    print('\n🎉 All train+test pass! Model is correct.')
else:
    print(f'\n❌ Some tests failed.')
