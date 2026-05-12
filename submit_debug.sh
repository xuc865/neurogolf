#!/bin/bash
KAGLE_TOKEN="${KAGLE_TOKEN:?Set KAGLE_TOKEN before running this script}"
ZIP_PATH="/Users/wxc/Documents/codes/neurogolf/submission.zip"
FILE_SIZE=$(stat -f%z "$ZIP_PATH")

echo "=== Step 1: Start upload ==="
curl -v -X POST "https://www.kaggle.com/api/v1/competitions/neurogolf-2026/submissions/upload" \
  -H "Authorization: Bearer $KAGLE_TOKEN" \
  -H "Content-Type: application/json" \
  -d "{\"contentLength\": $FILE_SIZE, \"fileName\": \"submission.zip\", \"contentType\": \"application/zip\"}" 2>&1
