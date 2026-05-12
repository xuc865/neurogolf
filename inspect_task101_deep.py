"""Deep inspect task101.onnx for Kaggle compatibility issues."""
import onnx
import onnxruntime as ort

model = onnx.load('/Users/wxc/Documents/codes/neurogolf/submission/task101.onnx')

# Check ops
ops = set()
for n in model.graph.node:
    ops.add(n.op_type)
print(f'Ops used ({len(ops)}): {sorted(ops)}')

# Check opset
for imp in model.opset_import:
    print(f'Domain: {imp.domain}, version: {imp.version}')
print(f'IR: {model.ir_version}')

# Check for banned ops
banned = {'LOOP', 'SCAN', 'NONZERO', 'UNIQUE', 'SCRIPT', 'FUNCTION', 'COMPRESS'}
used_banned = ops & banned
if used_banned:
    print(f'⚠️ Banned ops found: {used_banned}')
else:
    print('✅ No banned ops')

# Check initializer overlap
init_names = {init.name for init in model.graph.initializer}
input_names = {inp.name for inp in model.graph.input}
overlap = input_names & init_names
if overlap:
    print(f'⚠️ Input/initializer overlap: {overlap}')
else:
    print('✅ No input/initializer overlap')

# Check subgraphs
for n in model.graph.node:
    for attr in n.attribute:
        if attr.type in (onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS):
            print(f'⚠️ Subgraph found in node {n.name}')
            break

# Check functions
if model.functions:
    print(f'⚠️ Custom functions: {len(model.functions)}')
else:
    print('✅ No custom functions')

# Check custom domains
for imp in model.opset_import:
    if imp.domain not in ('', 'ai.onnx'):
        print(f'⚠️ Custom domain: {imp.domain}')

# Check for duplicate node names
from collections import Counter
node_names = [n.name for n in model.graph.node]
dups = [n for n, c in Counter(node_names).items() if c > 1]
if dups:
    print(f'⚠️ Duplicate node names: {dups[:5]}')
else:
    print('✅ No duplicate node names')

# Check model size
import os
size = os.path.getsize('/Users/wxc/Documents/codes/neurogolf/submission/task101.onnx')
print(f'Model size: {size} bytes ({size/1024:.1f} KB)')
print(f'Max allowed: {1.44*1024*1024:.0f} bytes ({1.44:.1f} MB)')
if size <= 1.44*1024*1024:
    print('✅ Under file limit')

# Check ONNX checker
try:
    onnx.checker.check_model(model, full_check=True)
    print('✅ ONNX checker passed')
except Exception as e:
    print(f'⚠️ ONNX checker: {str(e)[:100]}')

# Check shape inference
try:
    inferred = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    print('✅ Shape inference OK')
except Exception as e:
    print(f'⚠️ Shape inference: {str(e)[:100]}')
