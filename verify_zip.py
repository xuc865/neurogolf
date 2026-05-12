"""验证 submission.zip"""
import zipfile

z = zipfile.ZipFile('/Users/wxc/Documents/codes/neurogolf/submission.zip')
names = sorted(z.namelist())
print(f'Total files: {len(names)}')
print(f'First: {names[0]}')
print(f'Last: {names[-1]}')
print(f'Any non-onnx? {[n for n in names if not n.endswith(".onnx")]}')

# 检查是否 1-indexed
expected = [f'task{str(i).zfill(3)}.onnx' for i in range(1, 401)]
missing = [f for f in expected if f not in names]
extra = [f for f in names if f not in expected]
print(f'\nMissing: {len(missing)}')
if missing:
    print(f'  {missing[:10]}...')
print(f'Extra: {len(extra)}')
if extra:
    print(f'  {extra[:10]}...')

print(f'\n✅ ZIP 验证 {"通过" if len(names)==400 and not missing and not extra else "失败"}')
