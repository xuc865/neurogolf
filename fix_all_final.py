"""Fix: Save all models with opset 11 (correct for the ops used)."""
import onnx
from onnx import helper
import onnxruntime as ort
import glob

SUBMISSION_DIR = '/Users/wxc/Documents/codes/neurogolf/submission'

print("=== Fixing model opset versions ===")
for path in sorted(glob.glob(f'{SUBMISSION_DIR}/task*.onnx')):
    tid = path.split('task')[-1].split('.')[0]
    model = onnx.load(path)
    
    # Set opset to 11 (supports all ops in these models including ArgMax, Equal, GatherND, etc.)
    while model.opset_import:
        model.opset_import.pop()
    model.opset_import.append(helper.make_opsetid('', 11))
    
    # Fix IR version
    model.ir_version = 10
    
    # Remove producer name that might confuse Kaggle
    model.producer_name = ''
    
    onnx.save(model, path)

print("=== Verifying all models ===")
failures = []
for path in sorted(glob.glob(f'{SUBMISSION_DIR}/task*.onnx')):
    tid = path.split('task')[-1].split('.')[0]
    try:
        sess = ort.InferenceSession(path)
        inp_shape = [d.dim_value for d in sess.get_inputs()[0].shape]
        out_shape = [d.dim_value for d in sess.get_outputs()[0].shape]
        if inp_shape == [1, 10, 30, 30] and out_shape == [1, 10, 30, 30]:
            pass  # OK
        else:
            print(f"  task{tid}: ❌ Wrong shapes {inp_shape} -> {out_shape}")
            failures.append(tid)
    except Exception as e:
        print(f"  task{tid}: ❌ {str(e)[:100]}")
        failures.append(tid)

if failures:
    print(f"\n❌ {len(failures)} failures: {failures}")
else:
    print(f"✅ All {len(list(glob.glob(f'{SUBMISSION_DIR}/task*.onnx')))} models load OK!")

# Specifically verify the 4 problem models
print("\n=== Problem models detail ===")
for tid in ['096', '101', '118', '266']:
    path = f'{SUBMISSION_DIR}/task{tid}.onnx'
    try:
        sess = ort.InferenceSession(path)
        print(f"  task{tid}: ✅ Shape {sess.get_inputs()[0].shape} -> {sess.get_outputs()[0].shape}")
    except Exception as e:
        print(f"  task{tid}: ❌ {str(e)[:80]}")
