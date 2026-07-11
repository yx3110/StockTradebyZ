# API Calls

Use this reference to call the production Yixin OpenAPI gateway. All requests use:

```http
Content-Type: application/json
Accept: application/json
X-API-KEY: <product-subscription-key>
```

One product subscription key calls all public APIs. Do not include real API keys in examples. Use the `YIXIN_API_KEY` environment variable or load the key from the user's private file.

## Load The Key

Example shell pattern:

```bash
YIXIN_API_KEY="$(jq -r '.api_key' ~/.config/yixin-api/api-key.json)"
```

## search API

Endpoint:

```text
POST https://openapi.billionsintelligence.com/api/v2/search
```

Use the path without a trailing slash.

Request body:

```json
{
  "query": "宁德时代最新业绩",
  "source": "report",
  "search_mode": "advanced",
  "count": 10,
  "time_range": "past 1 month"
}
```

Fields:

| Field | Required | Notes |
| --- | --- | --- |
| `query` | yes | Search keywords or natural-language question. |
| `source` | no | `web`, `academic`, `image`, `video`, `announcement`, `report`, or `expert`. Defaults to `web`. |
| `search_mode` | no | `fast`, `advanced`, or `expert`. Defaults to `fast`. |
| `count` | no | Maximum results, `1` to `50`. |
| `timeout` | no | Per-engine timeout in seconds, `1` to `120`. |
| `time_range` | no | Examples: `past 3 days`, `past 1 month`, `from 2025-01-01 to 2025-06-30`. |

curl:

```bash
YIXIN_API_KEY="$(jq -r '.api_key' ~/.config/yixin-api/api-key.json)"

curl -sS --fail-with-body \
  -X POST "https://openapi.billionsintelligence.com/api/v2/search" \
  -H "Content-Type: application/json" \
  -H "Accept: application/json" \
  -H "X-API-KEY: ${YIXIN_API_KEY}" \
  --data-binary @- <<'JSON'
{
  "query": "宁德时代最新业绩",
  "source": "report",
  "search_mode": "advanced",
  "count": 10,
  "time_range": "past 1 month"
}
JSON
```

Python:

```python
import json
import os
import urllib.error
import urllib.request

url = "https://openapi.billionsintelligence.com/api/v2/search"
api_key = os.environ["YIXIN_API_KEY"]
payload = {
    "query": "宁德时代最新业绩",
    "source": "report",
    "search_mode": "advanced",
    "count": 10,
    "time_range": "past 1 month",
}

request = urllib.request.Request(
    url,
    data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
    method="POST",
    headers={
        "Content-Type": "application/json",
        "Accept": "application/json",
        "X-API-KEY": api_key,
    },
)

try:
    with urllib.request.urlopen(request, timeout=120) as response:
        print(response.status)
        print(response.read().decode("utf-8", errors="replace"))
except urllib.error.HTTPError as exc:
    body = exc.read().decode("utf-8", errors="replace")
    if exc.code == 429:
        raise SystemExit("额度已用完，请联系销售升级：https://www.billionsintelligence.com") from exc
    print(exc.code)
    print(body)
    raise
```

## FinData fin_db API

Endpoint:

```text
POST https://openapi.billionsintelligence.com/api/v1/fin_db
```

Request body:

```json
{
  "query": "美国2025年平均每日成交金额是多少",
  "data_sources": ["auto"]
}
```

Fields:

| Field | Required | Notes |
| --- | --- | --- |
| `query` | yes | Natural-language financial data question. |
| `data_sources` | no | `auto`, `A股财务行情数据库`, `海外财务行情数据库`, or `宏观行业数据库`. Use string or array form. |

curl:

```bash
YIXIN_API_KEY="$(jq -r '.api_key' ~/.config/yixin-api/api-key.json)"

curl -sS --fail-with-body \
  -X POST "https://openapi.billionsintelligence.com/api/v1/fin_db" \
  -H "Content-Type: application/json" \
  -H "Accept: application/json" \
  -H "X-API-KEY: ${YIXIN_API_KEY}" \
  --data-binary @- <<'JSON'
{
  "query": "美国2025年平均每日成交金额是多少",
  "data_sources": ["auto"]
}
JSON
```

Python:

```python
import json
import os
import urllib.error
import urllib.request

url = "https://openapi.billionsintelligence.com/api/v1/fin_db"
api_key = os.environ["YIXIN_API_KEY"]
payload = {
    "query": "美国2025年平均每日成交金额是多少",
    "data_sources": ["auto"],
}

request = urllib.request.Request(
    url,
    data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
    method="POST",
    headers={
        "Content-Type": "application/json",
        "Accept": "application/json",
        "X-API-KEY": api_key,
    },
)

try:
    with urllib.request.urlopen(request, timeout=60) as response:
        print(response.status)
        print(response.read().decode("utf-8", errors="replace"))
except urllib.error.HTTPError as exc:
    body = exc.read().decode("utf-8", errors="replace")
    if exc.code == 429:
        raise SystemExit("额度已用完，请联系销售升级：https://www.billionsintelligence.com") from exc
    print(exc.code)
    print(body)
    raise
```

## Status Handling

| Status | Meaning | User guidance |
| --- | --- | --- |
| `200` | Request reached the API. | Check the JSON body's `success`, `result`, and `error` fields. |
| `400` | Invalid request body or empty required field. | Fix JSON and required parameters. |
| `401` | Missing or invalid API key. | Confirm `X-API-KEY` and that the key was copied correctly. |
| `403` | Key is revoked or the product subscription is inactive. | Check subscription status in the portal; renew the key or re-subscribe. |
| `429` | Rate limit or quota exceeded. | `额度已用完，请联系销售升级：https://www.billionsintelligence.com` |
| `5xx` | Gateway or upstream service failure. | Retry later and preserve request/response context for support. |
