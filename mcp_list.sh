#!/bin/bash
# Kaggle MCP - list tools and authorize
KAGLE_TOKEN="${KAGLE_TOKEN:?Set KAGLE_TOKEN before running this script}"

echo "=== Step 1: Authorize ==="
curl -s -X POST "https://www.kaggle.com/mcp" \
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
  }' 2>&1
echo ""

echo ""
echo "=== Step 2: List tools ==="
curl -s -X POST "https://www.kaggle.com/mcp" \
  -H "Authorization: Bearer $KAGLE_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "id": 2,
    "method": "tools/list",
    "params": {}
  }' 2>&1 | python3 -c "
import sys, json
data = json.load(sys.stdin)
for tool in data.get('result', {}).get('tools', []):
    print(f\"  {tool['name']}: {tool.get('description', '')[:100]}\")
    if tool['name'] in ['start_competition_submission_upload', 'submit_to_competition']:
        print(f\"    Input schema: {json.dumps(tool.get('inputSchema', {}), indent=4)[:300]}\")
" 2>&1
echo ""
