#!/usr/bin/env python3
"""Build best models: keep afr1ste models, supplement from others."""
import os, sys, shutil, warnings, json
os.environ["PYTHONWARNINGS"] = "ignore"
import numpy as np
import onnxruntime as ort
ort.set_default_logger_severity(3)

BASE = "/Users/wxc/workspace/neurogolf-2026"
data_dir = os.path.join(BASE, "data")
sub_dir = os.path.join(BASE, "submission")
out_dir = os.path.join(BASE, "best_models")
os.makedirs(out_dir, exist_ok=True)
_CH = 10; _H = 30; _W = 30

def verify_model(model_path, task_path):
    with open(task_path) as f:
        task = json.load(f)
    try:
        session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    except:
        return False, 0, 0
    examples = task.get("train", []) + task.get("test", []) + task.get("arc-gen", [])
    correct = 0; total = 0
    for ex in examples:
        inp = np.zeros((1, _CH, _H, _W), dtype=np.float32)
        exp = np.zeros((1, _CH, _H, _W), dtype=np.float32)
        grid = ex["input"]; out = ex["output"]
        h, w = len(grid), len(grid[0]); ho, wo = len(out), len(out[0])
        if max(h, w, ho, wo) > 30: continue
        for r in range(h):
            for c in range(w): inp[0][grid[r][c]][r][c] = 1.0
        for r in range(ho):
            for c in range(wo): exp[0][out[r][c]][r][c] = 1.0
        try:
            res = session.run(["output"], {"input": inp})
        except:
            return False, correct, total
        pred = (res[0] > 0.0).astype(float)
        if np.array_equal(pred[0, :, :ho, :wo], exp[0, :, :ho, :wo]):
            correct += 1
        total += 1
    return correct == total, correct, total

# Verify all current (kushalasriijjada) models
print("=== Current (kushalasriijjada) models ===")
passing = []
for i in range(1, 401):
    m = os.path.join(BASE, f"task{i:03d}.onnx")
    t = os.path.join(data_dir, f"task{i:03d}.json")
    if not os.path.exists(m): continue
    ok, c, tot = verify_model(m, t)
    if ok: passing.append(i)
print(f"Passing: {len(passing)}/400")
remaining = [i for i in range(1,401) if i not in passing]
print(f"Remaining: {remaining}")

# Copy passing models to out_dir
for i in passing:
    src = os.path.join(BASE, f"task{i:03d}.onnx")
    dst = os.path.join(out_dir, f"task{i:03d}.onnx")
    shutil.copy2(src, dst)

# Now check submission/ directory models for remaining
if remaining:
    print(f"\n=== submission/ for remaining ===")
    for i in remaining:
        m = os.path.join(sub_dir, f"task{i:03d}.onnx")
        t = os.path.join(data_dir, f"task{i:03d}.json")
        if os.path.exists(m):
            ok, c, tot = verify_model(m, t)
            if ok:
                shutil.copy2(m, os.path.join(out_dir, f"task{i:03d}.onnx"))
                print(f"  [+] task{i:03d} fixed from submission/")
            else:
                print(f"  [-] task{i:03d} FAIL ({c}/{tot})")
        else:
            print(f"  [-] task{i:03d} NOT FOUND")

# Check remaining after submission
remaining2 = [i for i in remaining if not os.path.exists(os.path.join(out_dir, f"task{i:03d}.onnx"))]
print(f"\nStill remaining: {remaining2}")

# For the remaining tasks, try to build simple identity model
if remaining2:
    print(f"\n=== Building identity models for remaining ===")
    import onnx
    for i in remaining2:
        w = [1.0 if o == i_ch else 0.0 for o in range(_CH) for i_ch in range(_CH) for _ in range(1)]
        x = onnx.helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, [1, _CH, _H, _W])
        y = onnx.helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [1, _CH, _H, _W])
        wt = onnx.helper.make_tensor("W", onnx.TensorProto.FLOAT, [_CH, _CH, 1, 1], w)
        nd = onnx.helper.make_node("Conv", ["input","W"], ["output"], kernel_shape=[1,1], pads=[0,0,0,0])
        model = onnx.helper.make_model(onnx.helper.make_graph([nd], "g", [x], [y], [wt]),
                                        ir_version=10, opset_imports=[onnx.helper.make_opsetid("", 10)])
        dst = os.path.join(out_dir, f"task{i:03d}.onnx")
        onnx.save(model, dst)
        print(f"  Built identity model for task{i:03d}")

print(f"\n=== Final count: {len(os.listdir(out_dir))} models in best_models/ ===")
