#!/usr/bin/env python3
"""Build the best possible submission:
- For tasks where we have proper solutions: use those
- For color-remap tasks (5): proper 1x1 conv
- For all others: identity 1x1 conv (gives minimum 1 point each)
"""
import json, os, sys
import onnx
import onnx.helper
import numpy as np
import onnxruntime as ort
import zipfile

DATA_DIR = "/Users/wxc/workspace/neurogolf-2026/data"
OUTPUT_DIR = "/Users/wxc/workspace/neurogolf-2026/models_v4"
MODELS_V3 = "/Users/wxc/workspace/neurogolf-2026/models_v3"
os.makedirs(OUTPUT_DIR, exist_ok=True)

_CH, _H, _W = 10, 30, 30
_DT = onnx.TensorProto.FLOAT
_IR, _OPS = 10, [onnx.helper.make_opsetid("", 10)]

def make_identity():
    w = [1.0 if o == i else 0.0 for o in range(_CH) for i in range(_CH) for _ in range(1)]
    x = onnx.helper.make_tensor_value_info("input", _DT, [1, _CH, _H, _W])
    y = onnx.helper.make_tensor_value_info("output", _DT, [1, _CH, _H, _W])
    w_t = onnx.helper.make_tensor("W", _DT, [_CH, _CH, 1, 1], w)
    node = onnx.helper.make_node("Conv", ["input", "W"], ["output"],
                                 kernel_shape=[1, 1], pads=[0, 0, 0, 0])
    graph = onnx.helper.make_graph([node], "graph", [x], [y], [w_t])
    return onnx.helper.make_model(graph, ir_version=_IR, opset_imports=_OPS)

def verify(task_num, model):
    try:
        task_data = json.load(open(f"{DATA_DIR}/task{task_num:03d}.json"))
        session = ort.InferenceSession(model.SerializeToString())
        examples = task_data.get("train", []) + task_data.get("test", [])
        for ex in examples:
            inp_g, out_g = ex["input"], ex["output"]
            hi, wi = len(inp_g), len(inp_g[0])
            ho, wo = len(out_g), len(out_g[0])
            if max(hi, wi, ho, wo) > 30: continue
            inp_np = np.zeros((1, _CH, _H, _W), dtype=np.float32)
            for r in range(hi):
                for c in range(wi):
                    inp_np[0][inp_g[r][c]][r][c] = 1.0
            result = session.run(["output"], {"input": inp_np})
            pred = result[0][0]
            for r in range(ho):
                for c in range(wo):
                    exp_c = out_g[r][c]
                    act = [ch for ch in range(_CH) if pred[ch][r][c] > 0.5]
                    if len(act) != 1 or act[0] != exp_c:
                        return False
        return True
    except:
        return False

# Build all models
identity = make_identity()
passed = 0
total = 0

for tn in range(1, 401):
    # Check if we have a working model from earlier versions
    working_model = None
    
    # Try v3 models first
    old_path = f"{MODELS_V3}/task{tn:03d}.onnx"
    if os.path.exists(old_path):
        try:
            model = onnx.load(old_path)
            if verify(tn, model):
                working_model = model
        except:
            pass
    
    if working_model is None:
        # Use identity
        working_model = identity
    
    out_path = f"{OUTPUT_DIR}/task{tn:03d}.onnx"
    onnx.save(working_model, out_path)
    total += 1
    if verify(tn, working_model):
        passed += 1

# Create submission.zip
zip_path = "/Users/wxc/workspace/neurogolf-2026/submission_best.zip"
with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
    for tn in range(1, 401):
        zf.write(f"{OUTPUT_DIR}/task{tn:03d}.onnx", f"task{tn:03d}.onnx")

print(f"\n=== Results ===")
print(f"Total: {total}")
print(f"Passing: {passed}")
print(f"Submission: {zip_path}")

# File sizes
sizes = []
for tn in range(1, 401):
    sz = os.path.getsize(f"{OUTPUT_DIR}/task{tn:03d}.onnx")
    sizes.append(sz)
print(f"Avg size: {sum(sizes)/len(sizes):.0f} bytes")
print(f"Zip size: {os.path.getsize(zip_path)} bytes")
print(f"Non-identity files: {len([s for s in sizes if s != 572])}")
