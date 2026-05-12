#!/bin/bash
# Submit via Kaggle MCP SSE endpoint
KAGLE_TOKEN="${KAGLE_TOKEN:?Set KAGLE_TOKEN before running this script}"

# Try the Kaggle MCP HTTP endpoint
echo "=== Submitting via Kaggle MCP ==="

# First, let's test if the SSE endpoint works
curl -s -X POST "https://www.kaggle.com/mcp" \
  -H "Authorization: Bearer $KAGLE_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "id": 1,
    "method": "tools/call",
    "params": {
      "name": "start_competition_submission_upload",
      "arguments": {
        "competition": "neurogolf-2026",
        "contentLength": 958506,
        "fileName": "submission.zip",
        "contentType": "application/zip"
      }
    }
  }' 2>&1
echo ""
