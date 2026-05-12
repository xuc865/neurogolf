"""测试 Kaggle token 是否有效"""
import json
import os

# 读取 kaggle.json
with open(os.path.expanduser('~/.kaggle/kaggle.json')) as f:
    config = json.load(f)

token = config.get('key', '')
print(f"Token prefix: {token[:10]}...")
print(f"Token length: {len(token)}")

# 测试 API
import urllib.request
import urllib.error

# 测试 Kaggle API
url = "https://www.kaggle.com/api/v1/competitions/neurogolf-2026/data?format=json"
req = urllib.request.Request(url)
req.add_header('Authorization', f'Bearer {token}')

try:
    resp = urllib.request.urlopen(req)
    print(f"\n✅ Bearer token 有效！")
    print(f"Response: {resp.read(200)}")
except urllib.error.HTTPError as e:
    print(f"\n❌ Bearer token 无效: {e.code} {e.reason}")
    print(f"Response: {e.read(200)}")
except Exception as e:
    print(f"\n❌ 错误: {e}")
