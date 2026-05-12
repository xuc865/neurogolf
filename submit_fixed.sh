#!/bin/bash
# Submit fixed submission.zip to Kaggle NeuroGolf 2026
set -e

KAGLE_TOKEN="${KAGLE_TOKEN:?Set KAGLE_TOKEN before running this script}"
ZIP_PATH="/Users/wxc/Documents/codes/neurogolf/submission.zip"
FILE_SIZE=$(stat -f%z "$ZIP_PATH")

echo "=== Step 1: Start upload (file size: $FILE_SIZE bytes) ==="
RESPONSE=$(curl -s -X POST "https://www.kaggle.com/api/v1/competitions/neurogolf-2026/submissions/upload" \
  -H "Authorization: Bearer $KAGLE_TOKEN" \
  -H "Content-Type: application/json" \
  -d "{\"contentLength\": $FILE_SIZE, \"fileName\": \"submission.zip\", \"contentType\": \"application/zip\"}")

UPLOAD_URL=$(echo "$RESPONSE" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('createPresignedUrl',''))")
UPLOAD_TOKEN=$(echo "$RESPONSE" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('uploadToken',''))")

if [ -z "$UPLOAD_URL" ] || [ -z "$UPLOAD_TOKEN" ]; then
    echo "ERROR: Failed to get upload URL/token"
    echo "$RESPONSE" | python3 -m json.tool
    exit 1
fi
echo "Got upload URL and token"

echo ""
echo "=== Step 2: Upload file to GCS ==="
UPLOAD_RESP=$(curl -s -w "\n%{http_code}" -X PUT \
  -H "Content-Type: application/zip" \
  -H "X-Upload-Content-Length: $FILE_SIZE" \
  --upload-file "$ZIP_PATH" \
  "$UPLOAD_URL")
HTTP_CODE=$(echo "$UPLOAD_RESP" | tail -1)
echo "HTTP status: $HTTP_CODE"

if [ "$HTTP_CODE" != "200" ] && [ "$HTTP_CODE" != "201" ]; then
    echo "Upload failed"
    exit 1
fi
echo "Upload successful!"

echo ""
echo "=== Step 3: Submit to competition ==="
SUBMIT_RESP=$(curl -s -X POST "https://www.kaggle.com/api/v1/competitions/neurogolf-2026/submissions" \
  -H "Authorization: Bearer $KAGLE_TOKEN" \
  -H "Content-Type: application/json" \
  -d "{\"competition\": \"neurogolf-2026\", \"uploadToken\": \"$UPLOAD_TOKEN\"}")
echo "$SUBMIT_RESP" | python3 -m json.tool
