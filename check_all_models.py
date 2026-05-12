import onnx
import json
import numpy as np
import onnxruntime as ort

# Check ops used
model = onnx.load('/Users/wxc/Documents/codes/neurogolf/submission/task101.onnx')
ops_used = set()
for node in model.graph.node:
    ops_used.add(node.op_type)
print(f"Ops used ({len(ops_used)}): {sorted(ops_used)}")
print(f"Opset: {model.opset_import}")

# Check all 400 models for which tasks fail
import glob

failures = []
for onnx_path in sorted(glob.glob('/Users/wxc/Documents/codes/neurogolf/submission/*.onnx')):
    try:
        m = onnx.load(onnx_path)
        # Quick validation
        # Try to create session
        try:
            sess = ort.InferenceSession(onnx_path)
        except Exception as e:
            task_id = onnx_path.split('task')[-1].split('.')[0]
            failures.append((task_id, str(e)))
    except Exception as e:
        task_id = onnx_path.split('task')[-1].split('.')[0]
        failures.append((task_id, f"ONNX parse error: {str(e)[:100]}"))

print(f"\nTotal failures: {len(failures)}")
for tid, err in failures[:10]:
    print(f"  task{tid}: {err}")
