#!/bin/bash
# Submit via Kaggle MCP with correct tool schemas
KAGLE_TOKEN="${KAGLE_TOKEN:?Set KAGLE_TOKEN before running this script}"
ZIP_PATH="/Users/wxc/Documents/codes/neurogolf/submission.zip"
FILE_SIZE=$(stat -f%z "$ZIP_PATH")
LAST_MODIFIED=$(stat -f%m "$ZIP_PATH")

echo "=== Step 1: Start upload ==="
UPLOAD_RESP=$(curl -s -X POST "https://www.kaggle.com/mcp" \
  -H "Authorization: Bearer $KAGLE_TOKEN" \
  -H "Content-Type: application/json" \
  -d "{
    \"jsonrpc\": \"2.0\",
    \"id\": 1,
    \"method\": \"tools/call\",
    \"params\": {
      \"name\": \"start_competition_submission_upload\",
      \"arguments\": {
        \"request\": {
          \"competitionName\": \"neurogolf-2026\",
          \"fileName\": \"submission.zip\",
          \"contentLength\": $FILE_SIZE,
          \"lastModifiedEpochSeconds\": $LAST_MODIFIED
        }
      }
    }
  }")
echo "Upload response:"
echo "$UPLOAD_RESP" | python3 -c "
import sys, json
data = json.loads('\n'.join([l for l in sys.stdin.read().split('\n') if l.startswith('data: ')]).replace('data: ',''))
result = data.get('result', data)
content = result.get('content', [])
for c in content:
    txt = c.get('text', '')
    try:
        parsed = json.loads(txt)
        print(json.dumps(parsed, indent=2))
    except:
        print(txt)
" 2>/dev/null
echo ""
echo "Raw response:"
echo "$UPLOAD_RESP"
