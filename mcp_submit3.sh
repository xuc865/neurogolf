#!/bin/bash
# Submit via Kaggle MCP - authorize first, then submit
KAGLE_TOKEN="${KAGLE_TOKEN:?Set KAGLE_TOKEN before running this script}"
ZIP_PATH="/Users/wxc/Documents/codes/neurogolf/submission.zip"
FILE_SIZE=$(stat -f%z "$ZIP_PATH")
LAST_MODIFIED=$(stat -f%m "$ZIP_PATH")

# Session file to persist cookies
COOKIE_JAR="/tmp/kaggle_mcp_cookies.txt"

echo "=== Step 1: Authorize ==="
AUTH_RESP=$(curl -s -c "$COOKIE_JAR" -b "$COOKIE_JAR" \
  -X POST "https://www.kaggle.com/mcp" \
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
  }')
echo "Auth: $AUTH_RESP"

# Extract session cookies for next call
echo ""
echo "=== Step 2: Start upload ==="
UPLOAD_RESP=$(curl -s -c "$COOKIE_JAR" -b "$COOKIE_JAR" \
  -X POST "https://www.kaggle.com/mcp" \
  -H "Authorization: Bearer $KAGLE_TOKEN" \
  -H "Content-Type: application/json" \
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
