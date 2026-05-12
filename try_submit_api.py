"""尝试用 Kaggle API key 通过 HTTP 直接提交"""
import json
import os
import base64
import urllib.request
import urllib.error

# 读取 kaggle.json
with open(os.path.expanduser('~/.kaggle/kaggle.json')) as f:
    config = json.load(f)

username = config['username']
key = config['key']

# Basic Auth header
credentials = base64.b64encode(f"{username}:{key}".encode()).decode()
print(f"Username: {username}")
print(f"Key prefix: {key[:10]}...")

# 尝试获取上传 URL
url = "https://www.kaggle.com/api/v1/competitions.CompetitionApiService/StartSubmissionUpload"
headers = {
    'Authorization': f'Basic {credentials}',
    'Content-Type': 'application/json',
}

data = json.dumps({
    "competitionName": "neurogolf-2026",
    "fileName": "submission.zip",
    "contentLength": os.path.getsize("/Users/wxc/Documents/codes/neurogolf/submission.zip")
}).encode()

req = urllib.request.Request(url, data=data, headers=headers, method='POST')

try:
    resp = urllib.request.urlopen(req)
    print(f"\n✅ 成功获取上传 URL！")
    print(f"Response: {resp.read().decode()[:500]}")
except urllib.error.HTTPError as e:
    print(f"\n❌ 失败: {e.code} {e.reason}")
    print(f"Response: {e.read().decode()[:500]}")
except Exception as e:
    print(f"\n❌ 错误: {e}")
