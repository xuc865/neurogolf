#!/usr/bin/env python3
"""验证所有 ONNX 模型在本地数据上的表现。"""
import json, os, sys
import onnx, onnxruntime as ort
import numpy as np

DATA_DIR = "/Users/wxc/workspace/neurogolf-2026/data"
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
SUB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "submission")
_CH, _H, _W = 10, 30, 30

def verify_task(task_num, model_bytes):
    """返回 (passed, total) 通过示例数和总示例数。"""
    session = ort.InferenceSession(model_bytes)
    td = json.load(open(f"{DATA_DIR}/task{task_num:03d}.json"))
    examples = td.get("train", []) + td.get("test", [])
    ok, total_ex = 0, 0
    for ex in examples:
        ig, og = ex["input"], ex["output"]
        hi, wi, ho, wo = len(ig), len(ig[0]), len(og), len(og[0])
        if max(hi, wi, ho, wo) > 30:
            continue
        total_ex += 1
        inp = np.zeros((1, _CH, _H, _W), dtype=np.float32)
        for r in range(hi):
            for c in range(wi):
                inp[0][ig[r][c]][r][c] = 1.0
        res = session.run(["output"], {"input": inp})[0][0]
        match = all(
            sum(1 for ch in range(_CH) if res[ch][r][c] > 0.5) == 1
            and [ch for ch in range(_CH) if res[ch][r][c] > 0.5][0] == og[r][c]
            for r in range(ho) for c in range(wo)
        )
        if match:
            ok += 1
    return ok, total_ex

# 从命令行参数或默认路径读取模型
model_files = sys.argv[1:] if len(sys.argv) > 1 else sorted(os.listdir(SUB_DIR))

passed = total = 0
failures = []

for mf in model_files:
    # 提取任务编号
    if mf.endswith('.onnx'):
        tn = int(os.path.basename(mf).replace('task', '').replace('.onnx', ''))
        model_path = mf if os.path.exists(mf) else os.path.join(SUB_DIR, mf)
        if not os.path.exists(model_path):
            print(f"MISSING: {mf}")
            continue
        try:
            model = onnx.load(model_path)
            p, t = verify_task(tn, model.SerializeToString())
            total += 1
            sz = os.path.getsize(model_path)
            if p == t and t > 0:
                passed += 1
                if total <= 5 or total % 100 == 0:
                    print(f"  OK: task{tn:03d} ({p}/{t}, {sz}B)")
            else:
                failures.append((tn, p, t))
                print(f"  FAIL: task{tn:03d} ({p}/{t}, {sz}B)")
        except Exception as e:
            failures.append((tn, 0, 0))
            print(f"  ERROR: task{tn:03d} - {e}")

print(f"\n=== 验证结果 ===")
print(f"总计: {total}")
print(f"通过: {passed}")
print(f"失败: {len(failures)}")
if failures:
    print(f"失败任务编号: {[f[0] for f in failures[:20]]}")
