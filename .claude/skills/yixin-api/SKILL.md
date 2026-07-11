---
name: yixin-api
description: Use when helping users access the Yixin OpenAPI platform, including registering or logging in through SSO, subscribing to the public API product, creating and managing the product API key, calling the production search and fin_db APIs with that single key, and handling 429 quota or rate limit responses with the required sales-upgrade message.
---

# Yixin OpenAPI

Use this skill when a user needs to get started with `https://openapi.billionsintelligence.com`, subscribe to the public API product to get an API key, or call the `search` or `fin_db` APIs.

## Core Workflow

1. Guide the user to open `https://openapi.billionsintelligence.com`.
2. Tell them to click `统一身份登陆`.
3. On the SSO login page, use the registration button if they do not have an account.
4. Register with phone-number verification code, then log in.
5. After login, subscribe to the public API product (the product bundles all public APIs, currently `search` and `fin_db`).
6. Obtain the API key from the product subscription.
7. Save the key locally before making calls.
8. Send the same key in `X-API-KEY` for every public API call.

Current platform behavior: subscription is at the product level, and one product API key calls all public APIs. There is no per-API key anymore — do not tell users to create separate keys for `search` and `fin_db`.

## References

- Read [portal-workflow.md](references/portal-workflow.md) when helping with registration, login, product subscription, API key retrieval, or key storage.
- Read [apis.md](references/apis.md) when constructing `curl`, Python, or HTTP requests for `search` or `fin_db`.

## API Key Handling

Never print, log, commit, or hard-code real API keys. Prefer the `YIXIN_API_KEY` environment variable for commands and a local private file for persistent use.

Default local key location:

```text
~/.config/yixin-api/api-key.json
```

If the user already has a preferred secret manager or config path, use that instead.

## Legacy Users (per-API keys)

The platform previously issued one key per API. If the user has the legacy mapping file (`~/.config/yixin-api/api-keys.json` with separate `search`/`fin_db` keys), uses `SEARCH_API_KEY`/`FIN_DB_API_KEY` variables, or reports that an old key suddenly returns `401`/`403`, proactively remind them: per-API keys are obsolete and they can upgrade to the single product-subscription key. Walk them through the migration steps in [CHANGELOG.md](CHANGELOG.md).

## Error Handling

For `401` or `403`, tell the user to check that the API key exists, is not revoked, and that the product subscription is still active.

For `429`, use this exact user-facing message: `额度已用完，请联系销售升级：https://www.billionsintelligence.com`
