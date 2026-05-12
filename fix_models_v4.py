"""Fix models: keep original opset but fix IR version. 
Also find the original opset for each model and try to match it."""
import onnx
from onnx import helper
import onnxruntime as ort
import json
import numpy as np
import glob

SUBMISSION_DIR = '/Users/wxc/Documents/codes/neurogolf/submission'

def inspect_model(path):
    model = onnx.load(path)
    opsets = {}
    for imp in model.opset_import:
        opsets[imp.domain] = imp.version
    return {
        'ir_version': model.ir_version,
        'opset_imports': opsets,
        'num_nodes': len(model.graph.node),
        'num_inputs': len(model.graph.input),
        'input_names': [i.name for i in model.graph.input],
        'op_types': list(set(n.op_type for n in model.graph.node))
    }

# Inspect the 3 failing models
for tid in ['096', '101', '118']:
    info = inspect_model(f'{SUBMISSION_DIR}/task{tid}.onnx')
    print(f"\n=== Task {tid} ===")
    print(f"  IR version: {info['ir_version']}")
    print(f"  Opsets: {info['opset_imports']}")
    print(f"  Nodes: {info['num_nodes']}")
    print(f"  Inputs: {info['input_names']}")
    
# Strategy: for each model, find the ORIGINAL opset version it was created with
# and save it with that opset + IR v10
print("\n\n=== Fixing models ===")
for tid in ['096', '101', '118']:
    path = f'{SUBMISSION_DIR}/task{tid}.onnx'
    model = onnx.load(path)
    
    # Find the original ONNX domain opset
    original_opset = None
    for imp in model.opset_import:
        if imp.domain == '' or imp.domain == 'ai.onnx':
            original_opset = imp.version
    
    # Set IR to 10 (Kaggle supports up to 10)
    model.ir_version = 10
    
    # Keep opset as-is (it was already set correctly for the operators used)
    # Just ensure the opset is valid (between 1-21 for Kaggle)
    
    onnx.save(model, path)
    
    # Test
    try:
        sess = ort.InferenceSession(path)
        print(f"  task{tid}: ✅ (opset={original_opset}, IR=10)")
    except Exception as e:
        print(f"  task{tid}: ❌ {str(e)[:100]}")
        # Try older approach: save as opset-agnostic
        print(f"     Trying alternate fix...")
        
        # The model might have been created with opset 24 for 101, 11 for 096/118
        # Let me check what ops are used and find the minimum opset that supports them
        
        ops_used = set(n.op_type for n in model.graph.node)
        print(f"     Ops: {sorted(ops_used)}")
        
        # For task101: uses Slice with 5 inputs (dynamic) - needs opset >= 11
        # For task096: uses ArgMax, Equal - needs opset >= 11
        # For task118: uses Equal, GatherND, Where - needs opset >= 11
        
        # Try opset 11
        while model.opset_import:
            model.opset_import.pop()
        model.opset_import.append(helper.make_opsetid('', 11))
        onnx.save(path.replace('.onnx', '_v11.onnx'), path)  # Save back
        
        try:
            sess = ort.InferenceSession(path)
            print(f"     task{tid}: ✅ (opset=11, IR=10)")
        except Exception as e:
            print(f"     task{tid}: ❌ opset 11 also fails: {str(e)[:80]}")
            
            # Try opset 20  
            while model.opset_import:
                model.opset_import.pop()
            model.opset_import.append(helper.make_opsetid('', 20))
            onnx.save(model, path)
            
            try:
                sess = ort.InferenceSession(path)
                print(f"     task{tid}: ✅ (opset=20, IR=10)")
            except Exception as e2:
                print(f"     task{tid}: ❌ all failed: {str(e2)[:80]}")

# Final check
print("\n=== Final verification ===")
for tid in ['096', '101', '118', '266']:
    try:
        sess = ort.InferenceSession(f'{SUBMISSION_DIR}/task{tid}.onnx')
        print(f"  task{tid}: ✅")
    except Exception as e:
        print(f"  task{tid}: ❌ {str(e)[:80]}")
