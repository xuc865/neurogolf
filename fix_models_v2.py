"""Fix all 4 failing models to use opset 10 compatible operations."""
import onnx
from onnx import helper, TensorProto, numpy_helper
import onnxruntime as ort
import json
import numpy as np
import glob

SUBMISSION_DIR = '/Users/wxc/Documents/codes/neurogolf/submission'
DATA_DIR = '/Users/wxc/Documents/codes/neurogolf/data'

def grid_to_onehot(grid, max_colors=10, max_h=30, max_w=30):
    h, w = len(grid), len(grid[0])
    oh = np.zeros((max_colors, max_h, max_w), dtype=np.float32)
    for r in range(h):
        for c in range(w):
            oh[grid[r][c], r, c] = 1.0
    return oh[np.newaxis, :, :, :]

def make_identity_model(task_id, opset=10):
    """Build a simple identity model with proper opset."""
    X = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 10, 30, 30])
    Y = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 10, 30, 30])
    identity = helper.make_node('Identity', ['input'], ['output'])
    graph = helper.make_graph([identity], f'task{task_id}', [X], [Y])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', opset)])
    model.ir_version = 10
    return model

# ==================================================
# Step 1: Try setting opset to 10 for all 4 models
# ==================================================
print("=== Step 1: Convert to opset 10 and check ===")
for tid in ['101', '266', '096', '118']:
    path = f'{SUBMISSION_DIR}/task{tid}.onnx'
    try:
        model = onnx.load(path)
        
        # Set opset to 10 and IR version to 10
        new_imports = []
        for imp in model.opset_import:
            if imp.domain == '' or imp.domain == 'ai.onnx':
                new_imports.append(helper.make_opsetid('', 10))
            else:
                new_imports.append(imp)
        
        # Clear opset imports and add the fixed ones
        while model.opset_import:
            model.opset_import.pop()
        for imp in new_imports:
            model.opset_import.append(imp)
        
        model.ir_version = 10
        
        # Save temp copy
        temp_path = path + '.temp'
        onnx.save(model, temp_path)
        
        # Try loading
        try:
            sess = ort.InferenceSession(temp_path)
            print(f"  task{tid}: ✅ Opset 10 works!")
        except Exception as e:
            print(f"  task{tid}: ❌ {str(e)[:100]}")
            
            # Check what Slice ops are being used
            print(f"     Checking Slice nodes...")
            slice_nodes = [n for n in model.graph.node if n.op_type == 'Slice']
            for sn in slice_nodes[:3]:
                print(f"     Slice inputs: {sn.input}, outputs: {sn.output}")
                for attr in sn.attribute:
                    print(f"       attr: {attr.name}")
            
    except Exception as e:
        print(f"  task{tid}: ❌ Load failed: {str(e)[:50]}")

print()

# ==================================================
# Step 2: For models that fail, rebuild from scratch
# ==================================================
print("=== Step 2: Rebuild failing models ===")

# For task096 and task118, let me check if the ONNX model files 
# are actually different from ones we can build from the task data

# Let's check if task096 is just a pattern extraction that needs a trainable net
# First, let's see the original model structure to understand what it does
print("\n--- task096 original model ---")
model = onnx.load(f'{SUBMISSION_DIR}/task096.onnx')
for n in model.graph.node[:20]:
    print(f"  {n.op_type}: {n.input} -> {n.output}")

print("\n--- task118 original model ---")
model = onnx.load(f'{SUBMISSION_DIR}/task118.onnx')
for n in model.graph.node[:20]:
    print(f"  {n.op_type}: {n.input} -> {n.output}")

print("\n--- task266 original model ---")
model = onnx.load(f'{SUBMISSION_DIR}/task266.onnx')
for n in model.graph.node[:20]:
    print(f"  {n.op_type}: {n.input} -> {n.output}")
