import json, numpy as np

DATA = '/Users/wxc/Documents/codes/neurogolf/data'

for tid in ['096', '118']:
    print(f"{'='*60}")
    print(f"=== Task {tid} ===")
    print(f"{'='*60}")
    with open(f'{DATA}/task{tid}.json') as f:
        task = json.load(f)
    
    for i, pair in enumerate(task['train']):
        inp = np.array(pair['input'])
        out = np.array(pair['output'])
        h_in, w_in = inp.shape
        h_out, w_out = out.shape
        print(f'\nTrain {i}: {h_in}x{w_in} -> {h_out}x{w_out}')
        print('Input:')
        for r in range(h_in):
            print('  ', ''.join(str(int(c)) for c in inp[r]))
        print('Output:')
        for r in range(h_out):
            print('  ', ''.join(str(int(c)) for c in out[r]))
    
    for i, pair in enumerate(task['test']):
        inp = np.array(pair['input'])
        out = np.array(pair['output'])
        h_in, w_in = inp.shape
        h_out, w_out = out.shape
        print(f'\nTest {i}: {h_in}x{w_in} -> {h_out}x{w_out}')
        print('Input:')
        for r in range(h_in):
            print('  ', ''.join(str(int(c)) for c in inp[r]))
        print('Expected Output:')
        for r in range(h_out):
            print('  ', ''.join(str(int(c)) for c in out[r]))
