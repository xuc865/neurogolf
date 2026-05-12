#!/usr/bin/env python3
"""Build final submission with all models, submit via MCP."""
import json, os, zipfile
import onnx, onnx.helper, onnxruntime as ort
import numpy as np

DATA_DIR = "/Users/wxc/workspace/neurogolf-2026/data"
MODEL_DIR = "/Users/wxc/workspace/neurogolf-2026/models_final"
os.makedirs(MODEL_DIR, exist_ok=True)

_CH, _H, _W = 10, 30, 30

def load_task(n):
    return json.load(open(f"{DATA_DIR}/task{n:03d}.json"))

def mk_conv(w, k):
    pads = [k//2]*4
    x = onnx.helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, [1, _CH, _H, _W])
    y = onnx.helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [1, _CH, _H, _W])
    wt = onnx.helper.make_tensor("W", onnx.TensorProto.FLOAT, [_CH, _CH, k, k], w)
    nd = onnx.helper.make_node("Conv", ["input","W"], ["output"], kernel_shape=[k,k], pads=pads)
    return onnx.helper.make_model(onnx.helper.make_graph([nd], "g", [x], [y], [wt]),
                                  ir_version=10, opset_imports=[onnx.helper.make_opsetid("", 10)])

def verify_model(model, td):
    try:
        session = ort.InferenceSession(model.SerializeToString())
    except:
        return False
    for ex in td.get("train", []) + td.get("test", []):
        ig, og = ex["input"], ex["output"]
        hi, wi = len(ig), len(ig[0]); ho, wo = len(og), len(og[0])
        if max(hi, wi, ho, wo) > 30: continue
        inp = np.zeros((1, _CH, _H, _W), dtype=np.float32)
        for r in range(hi):
            for c in range(wi):
                inp[0][ig[r][c]][r][c] = 1.0
        try:
            res = session.run(["output"], {"input": inp})
        except:
            return False
        pred = res[0][0]
        for r in range(ho):
            for c in range(wo):
                exp = og[r][c]
                act = [ch for ch in range(_CH) if pred[ch][r][c] > 0.5]
                if len(act) != 1 or act[0] != exp:
                    return False
    return True

IDENTITY = mk_conv([1.0 if o == i else 0.0 for o in range(_CH) for i in range(_CH) for _ in range(1)], 1)
solved = 0; total = 0; stats = {}

for tn in range(1, 401):
    td = load_task(tn)
    train = td.get("train", [])
    if not train:
        onnx.save(IDENTITY, f"{MODEL_DIR}/task{tn:03d}.onnx")
        continue
    
    hi, wi = len(train[0]["input"]), len(train[0]["input"][0])
    ho, wo = len(train[0]["output"]), len(train[0]["output"][0])
    
    best = IDENTITY
    best_type = "identity"
    
    # 1) Check identity
    is_id = all(len(e["input"]) == len(e["output"]) and
                all(a == b for a, b in zip(e["input"], e["output"])) for e in train)
    if is_id and verify_model(IDENTITY, td):
        best_type = "identity"
    else:
        # 2) Try color remap (1x1 conv, position-independent)
        mapping = {}
        ok = True
        for ex in train:
            inp, out = ex["input"], ex["output"]
            if len(inp) != hi or len(inp[0]) != wi or len(out) != ho or len(out[0]) != wo:
                ok = False; break
            if hi == ho and wi == wo:
                for r in range(hi):
                    for c in range(wi):
                        ci, co = inp[r][c], out[r][c]
                        if ci not in mapping: mapping[ci] = co
                        elif mapping[ci] != co: ok = False
        if ok and mapping:
            w = []
            for o in range(_CH):
                for i in range(_CH):
                    for _ in range(1):
                        if i in mapping: w.append(1.0 if mapping[i] == o else 0.0)
                        else: w.append(1.0 if o == i else 0.0)
            m = mk_conv(w, 1)
            if verify_model(m, td):
                best = m; best_type = "remap"
        
        # 3) Try larger kernels for same-size tasks
        if hi == ho and wi == wo and best_type == "identity":
            same_sizes = all(len(e["input"]) == hi and len(e["input"][0]) == wi and
                            len(e["output"]) == hi and len(e["output"][0]) == wi for e in train)
            if same_sizes:
                for k in [3, 5, 7, 9, 11, 13, 15, 19, 23, 29]:
                    off = list(range(-k//2+1, k//2+1))
                    signal = [[[[False for _ in off] for _ in off] for _ in range(_CH)] for _ in range(_CH)]
                    noise = [[[[False for _ in off] for _ in off] for _ in range(_CH)] for _ in range(_CH)]
                    for ex in train:
                        inp, out = ex["input"], ex["output"]
                        for r in range(hi):
                            for c in range(wi):
                                oc = out[r][c]
                                for di, dr in enumerate(off):
                                    for dj, dc in enumerate(off):
                                        ir, ic = r+dr, c+dc
                                        if 0 <= ir < hi and 0 <= ic < wi:
                                            icl = inp[ir][ic]
                                            signal[oc][icl][di][dj] = True
                                            for o2 in range(_CH):
                                                if o2 != oc: noise[o2][icl][di][dj] = True
                    w = []
                    for o in range(_CH):
                        for i in range(_CH):
                            for di in range(k):
                                for dj in range(k):
                                    sig = signal[o][i][di][dj]
                                    noi = noise[o][i][di][dj]
                                    if sig and not noi: w.append(1.0)
                                    elif noi and not sig: w.append(-1.0)
                                    else: w.append(0.0)
                    m = mk_conv(w, k)
                    if verify_model(m, td):
                        best = m; best_type = f"conv{k}"
                        break
    
    onnx.save(best, f"{MODEL_DIR}/task{tn:03d}.onnx")
    total += 1
    stats[best_type] = stats.get(best_type, 0) + 1
    if verify_model(best, td): solved += 1
    if tn % 50 == 0:
        print(f"  [{tn}/400] solved: {solved}/{total}")
    if best_type not in ("identity", "remap"):
        print(f"  [{tn:03d}] {best_type} ✓")

print(f"\n=== Results ===")
print(f"Total: {total}, Pass train+test: {solved}")
print(f"Stats: {json.dumps(stats, indent=2)}")

# Create zip
zip_path = "/Users/wxc/workspace/neurogolf-2026/submission_final.zip"
with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
    for tn in range(1, 401):
        zf.write(f"{MODEL_DIR}/task{tn:03d}.onnx", f"task{tn:03d}.onnx")
print(f"\nSubmission zip: {zip_path} ({os.path.getsize(zip_path)} bytes)")

# Count non-identity
non_id = sum(1 for tn in range(1, 401) if os.path.getsize(f"{MODEL_DIR}/task{tn:03d}.onnx") != 572)
print(f"Non-identity files: {non_id}")
