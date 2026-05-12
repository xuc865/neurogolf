#!/bin/bash
# Try correct Kaggle API endpoints
KAGLE_TOKEN="${KAGLE_TOKEN:?Set KAGLE_TOKEN before running this script}"

echo "=== Try 1: POST /api/v1/competitions/submissions/upload ==="
curl -s -X POST "https://api.kaggle.com/api/v1/competitions/submissions/upload" \
  -H "Authorization: Bearer $KAGLE_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"competitionName": "neurogolf-2026", "fileName": "submission.zip", "contentLength": 958506}' 2>&1
echo ""

echo ""
echo "=== Try 2: POST /api/v1/competitions/submissions/upload with competition in URL ==="
curl -s -X POST "https://api.kaggle.com/api/v1/competitions/submissions/upload?competition=neurogolf-2026" \
  -H "Authorization: Bearer $KAGLE_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"fileName": "submission.zip", "contentLength": 958506}' 2>&1
echo ""

echo ""
echo "=== Try 3: Use Kaggle's web upload API ==="
curl -s -X POST "https://www.kaggle.com/api/v1/competitions/neurogolf-2026/submissions" \
  -H "Authorization: Bearer $KAGLE_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{}' 2>&1 | head -c 200
echo ""
