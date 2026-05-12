#!/usr/bin/env python3
"""Verify ONNX models against training data."""
import json, os, numpy as np
import onnxruntime as ort

BASE = "/Users/wxc/workspace/neurogolf-2026"
DATA_DIR = f"{BASE}/data"
MODEL_DIR = f"{BASE}/models"


def verify_task(task_num):
    """Verify a single task's ONNX model."""
    try:
        with open(f"{DATA_DIR}/task{task_num:03d}.json") as f:
            task_data = json.load(f)
        
        model_path = f"{MODEL_DIR}/task{task_num:03d}.onnx"
        if not os.path.exists(model_path):
            return None, "no_model"
        
        session = ort.InferenceSession(model_path)
        
        correct = 0
        total = 0
        details = []
        
        for ex in task_data.get('train', []) + task_data.get('test', []):
            inp_grid = ex['input']
            out_grid = ex['output']
            h, w = len(inp_grid), len(inp_grid[0])
            ho, wo = len(out_grid), len(out_grid[0])
            
            if max(h, w, ho, wo) > 30:
                continue
            
            inp_np = np.zeros((1, 10, 30, 30), dtype=np.float32)
            out_expected = np.zeros((1, 10, 30, 30), dtype=np.float32)
            
            for r in range(h):
                for c in range(w):
                    inp_np[0][inp_grid[r][c]][r][c] = 1.0
            for r in range(ho):
                for c in range(wo):
                    out_expected[0][out_grid[r][c]][r][c] = 1.0
            
            result = session.run(["output"], {"input": inp_np})
            pred = (result[0] > 0.0).astype(float)
            
            # Compare only within the output grid area
            expected_slice = out_expected[:, :, :ho, :wo]
            pred_slice = pred[:, :, :ho, :wo]
            
            if np.array_equal(pred_slice, expected_slice):
                correct += 1
            else:
                # Debug - show differences
                diffs = np.where(pred_slice != expected_slice)
                details.append(f"  Example {total}: {h}x{w}->{ho}x{wo}, diffs at {len(diffs[0])} positions")
            
            total += 1
        
        return correct, total, details
    except Exception as e:
        return None, None, [str(e)]


if __name__ == "__main__":
    print("Verifying tasks...")
    
    # Test a few specific tasks
    for tn in [1, 5, 10, 50, 100, 150, 200]:
        result = verify_task(tn)
        if result[0] is not None:
            correct, total, details = result
            status = "✓" if correct == total else "✗"
            print(f"  {status} task{tn:03d}: {correct}/{total} correct")
            for d in details:
                print(f"    {d}")
    
    # Scan all tasks and report stats
    print("\n=== Scan all 400 tasks ===")
    results = {}
    for tn in range(1, 401):
        r = verify_task(tn)
        if r[0] is not None:
            correct, total, _ = r
            if correct == total and total > 0:
                results[tn] = "CORRECT"
            elif correct == 0:
                results[tn] = "ALL_WRONG"
            else:
                results[tn] = f"PARTIAL({correct}/{total})"
        else:
            results[tn] = "ERROR"
    
    correct_count = sum(1 for v in results.values() if v == "CORRECT")
    wrong_count = sum(1 for v in results.values() if v == "ALL_WRONG")
    partial_count = sum(1 for v in results.values() if "PARTIAL" in str(v))
    
    print(f"  Correct: {correct_count}/400")
    print(f"  All wrong: {wrong_count}/400")
    print(f"  Partial: {partial_count}/400")
