"""Fix all model compatibility issues for Kaggle (opset 20, IR 10)."""
import onnx
from onnx import helper
import onnxruntime as ort
import json
import numpy as np
import glob

SUBMISSION_DIR = '/Users/wxc/Documents/codes/neurogolf/submission'
DATA_DIR = '/Users/wxc/Documents/codes/neurogolf/data'

def fix_model(path, target_opset=20, target_ir=10):
    """Set opset to a supported version and IR to 10."""
    try:
        model = onnx.load(path)
        
        # Fix opset
        new_imports = []
        has_onnx = False
        for imp in model.opset_import:
            if imp.domain == '' or imp.domain == 'ai.onnx':
                new_imports.append(helper.make_opsetid('', target_opset))
                has_onnx = True
            else:
                new_imports.append(imp)
        if not has_onnx:
            new_imports.append(helper.make_opsetid('', target_opset))
        
        while model.opset_import:
            model.opset_import.pop()
        for imp in new_imports:
            model.opset_import.append(imp)
        
        # Fix IR version
        model.ir_version = target_ir
        
        # Fix producer name
        model.producer_name = 'neurogolf-fix'
        
        onnx.save(model, path)
        
        # Verify
        try:
            sess = ort.InferenceSession(path)
            return True, "OK"
        except Exception as e:
            return False, str(e)[:150]
    except Exception as e:
        return False, f"Load: {str(e)[:100]}"

# Fix specific models
problems = ['096', '101', '118', '266']
print("=== Fixing 4 problem models ===")
for tid in problems:
    path = f'{SUBMISSION_DIR}/task{tid}.onnx'
    ok, msg = fix_model(path)
    print(f"  task{tid}: {'✅' if ok else '❌'} {msg}")

# Also check if there are other models with similar issues
print("\n=== Scanning all models ===")
all_paths = sorted(glob.glob(f'{SUBMISSION_DIR}/*.onnx'))
bad_models = []
for path in all_paths:
    try:
        model = onnx.load(path)
        # Check IR version
        if model.ir_version > 10:
            bad_models.append((path.split('task')[-1].split('.')[0], f'IR v{model.ir_version}'))
            continue
        # Check opset
        for imp in model.opset_import:
            if (imp.domain == '' or imp.domain == 'ai.onnx') and imp.version > 21:
                bad_models.append((path.split('task')[-1].split('.')[0], f'opset v{imp.version}'))
                continue
    except:
        pass

print(f"Found {len(bad_models)} additional bad models:")
for tid, reason in bad_models[:10]:
    print(f"  task{tid}: {reason}")

# Fix them too
for tid, reason in bad_models:
    fix_model(f'{SUBMISSION_DIR}/task{tid}.onnx')

# Final verification - load all models
print("\n=== Final verification ===")
failures = []
for tid in sorted(set(problems + [b[0] for b in bad_models])):
    try:
        sess = ort.InferenceSession(f'{SUBMISSION_DIR}/task{tid}.onnx')
        print(f"  task{tid}: ✅")
    except Exception as e:
        failures.append(tid)
        print(f"  task{tid}: ❌ {str(e)[:80]}")

if failures:
    print(f"\n❌ {len(failures)} models still fail: {failures}")
else:
    print(f"\n🎉 All models load successfully!")
