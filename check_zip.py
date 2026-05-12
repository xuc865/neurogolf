import subprocess, os

workdir = '/tmp/neurogolf_restore'
os.makedirs(workdir, exist_ok=True)
os.chdir(workdir)

# Extract submission.zip
subprocess.run(['unzip', '-o', '/Users/wxc/Documents/codes/neurogolf/submission.zip'], 
               capture_output=True, text=True)

# Check all models
files = sorted([f for f in os.listdir(workdir) if f.endswith('.onnx')])
print(f'Total models extracted: {len(files)}')

# Check task101 from the zip
import onnx
m = onnx.load('task101.onnx')
for imp in m.opset_import:
    if not imp.domain: print(f'task101 opset: {imp.version}')
print(f'task101 IR: {m.ir_version}')

# Check task001 from the zip
m1 = onnx.load('task001.onnx')
for imp in m1.opset_import:
    if not imp.domain: print(f'task001 opset: {imp.version}')
print(f'task001 IR: {m1.ir_version}')
