#!/bin/bash
# Submit via Kaggle MCP with cookie persistence
KAGLE_TOKEN="${KAGLE_TOKEN:?Set KAGLE_TOKEN before running this script}"
ZIP_PATH="/Users/wxc/Documents/codes/neurogolf/submission.zip"
FILE_SIZE=$(stat -f%z "$ZIP_PATH")
LAST_MODIFIED=$(stat -f%m "$ZIP_PATH")

echo "=== Step 1: Authorize + capture session ==="
AUTH_HEADERS=$(curl -s -D - -X POST "https://www.kaggle.com/mcp" \
  -H "Authorization: Bearer $KAGLE_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "id": 1,
    "method": "tools/call",
    "params": {
      "name": "authorize",
      "arguments": {}
    }
  }' 2>&1)

echo "$AUTH_HEADERS"

# Extract cookies
CSRF=$(echo "$AUTH_HEADERS" | grep -i "set-cookie.*XSRF-TOKEN" | sed 's/.*XSRF-TOKEN=\([^;]*\).*/\1/')
SESSION=$(echo "$AUTH_HEADERS" | grep -i "set-cookie.*ka_sessionid" | sed 's/.*ka_sessionid=\([^;]*\).*/\1/')
CLIENT=$(echo "$AUTH_HEADERS" | grep -i "set-cookie.*CLIENT-TOKEN" | sed 's/.*CLIENT-TOKEN=\([^;]*\).*/\1/')

echo ""
echo "CSRF: $CSRF"
echo "Session: $SESSION"

# Wait a moment
sleep 1

echo ""
echo "=== Step 2: Start upload ==="
UPLOAD_RESP=$(curl -s -X POST "https://www.kaggle.com/mcp" \
  -H "Authorization: Bearer $KAGLE_TOKEN" \
  -H "Content-Type: application/json" \
  -H "Cookie: XSRF-TOKEN=$CSRF; ka_sessionid=$SESSION; CLIENT-TOKEN=$CLIENT" \
  -d "{
    \"jsonrpc\": \"2.0\",
    \"id\": 2,
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
echo "$UPLOAD_RESP"
