import json

with open('/Users/wxc/Documents/codes/neurogolf/data/task101.json') as f:
    task = json.load(f)

print('Task ID:', task.get('task_id'))
print('Train pairs:', len(task.get('train', [])))
print('Test pairs:', len(task.get('test', [])))
print()

for i, pair in enumerate(task.get('train', [])):
    inp = pair['input']
    out = pair['output']
    print(f'--- Train {i} ---')
    print(f'Input:  {len(inp)}x{len(inp[0])}')
    print('Input grid:')
    for row in inp:
        print('  ', ''.join(str(c) for c in row))
    print(f'Output: {len(out)}x{len(out[0])}')
    print('Output grid:')
    for row in out:
        print('  ', ''.join(str(c) for c in row))
    print()

for i, pair in enumerate(task.get('test', [])):
    inp = pair['input']
    out = pair['output']
    print(f'--- Test {i} ---')
    print(f'Input:  {len(inp)}x{len(inp[0])}')
    print('Input grid:')
    for row in inp:
        print('  ', ''.join(str(c) for c in row))
    print(f'Output: {len(out)}x{len(out[0])}')
    print('Output grid:')
    for row in out:
        print('  ', ''.join(str(c) for c in row))
    print()
