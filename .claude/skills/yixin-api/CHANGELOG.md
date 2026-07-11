# Changelog

## 2026-06-11 — Product-subscription API key (breaking change for legacy users)

The Yixin OpenAPI platform switched from per-API keys to a single public API product subscription:

- **Before**: each API (`search`, `fin_db`) required its own subscription and its own API key. A `search` key could not call `fin_db`, and vice versa.
- **Now**: subscribe once to the public API product; the subscription yields **one API key that calls all public APIs** (`search` and `fin_db`).

### Migration guide for existing users

1. Log in to `https://openapi.billionsintelligence.com` and subscribe to the public API product (if your account predates this change, your old per-API subscriptions do not carry over automatically — create the product subscription in the portal).
2. Copy the API key from the product subscription page.
3. Replace the legacy key file `~/.config/yixin-api/api-keys.json` (per-API `search`/`fin_db` fields) with the new single-key file:

   ```bash
   mkdir -p ~/.config/yixin-api
   cat > ~/.config/yixin-api/api-key.json <<'JSON'
   {
     "api_key": "<product-subscription-key>"
   }
   JSON
   chmod 600 ~/.config/yixin-api/api-key.json
   rm -f ~/.config/yixin-api/api-keys.json
   ```

4. Update scripts: use the `YIXIN_API_KEY` environment variable (replacing `SEARCH_API_KEY` / `FIN_DB_API_KEY`) and send it in `X-API-KEY` for every call.

Old per-API keys are obsolete. If a request with an old key returns `401`/`403`, that is expected — switch to the product subscription key.

### Skill changes in this release

- `SKILL.md`: workflow rewritten around product subscription; removed the one-key-per-API rule; added a legacy-user upgrade reminder.
- `references/portal-workflow.md`: per-API key creation replaced by product subscription and key retrieval; key file simplified to `api-key.json`.
- `references/apis.md`: all examples use the single `YIXIN_API_KEY`; `403` guidance now points to subscription status instead of key-to-API binding.

## Earlier

- Initial release: SSO registration/login, per-API key workflow (one key per API), `search` and `fin_db` call references, 429 sales-upgrade message.
