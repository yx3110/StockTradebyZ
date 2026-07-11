# Portal Workflow

Use this reference for onboarding, login, public API product subscription, API key retrieval, and key storage.

## Register And Log In

1. Open `https://openapi.billionsintelligence.com`.
2. Click `统一身份登陆`.
3. On the SSO login page, click the registration button if the user has no account.
4. Register with phone number and SMS verification code.
5. After registration, log in through the same SSO page.

If the user cannot receive the SMS code, ask them to verify the phone number and contact platform support. Do not invent alternate registration channels.

## Subscribe To The Public API Product

After login, guide the user to the public API product in the portal and subscribe to it. The product bundles all public APIs:

| API | Portal/API display intent | Gateway path |
| --- | --- | --- |
| `search` | search | `/api/v2/search` |
| `fin_db` | FinData 数据库问数 API | `/api/v1/fin_db` |

Use the portal's current product and API list as the source of truth for the exact visible display names.

Current behavior: subscription is at the product level only. One subscription yields one API key, and that key calls every API in the product. There is no per-API subscription or per-API key — do not guide users to create separate keys for `search` and `fin_db`.

## Get The API Key

1. Subscribe to the public API product.
2. Open the subscription detail to view or copy the API key.
3. If the key is lost or compromised, renew/revoke it from the same subscription page; the old key stops working.

## Save The Key

Prefer a local private file outside the repository:

```text
~/.config/yixin-api/api-key.json
```

Recommended shape:

```json
{
  "api_key": "<product-subscription-key>"
}
```

Set strict local permissions when creating this file:

```bash
mkdir -p ~/.config/yixin-api
chmod 700 ~/.config/yixin-api
chmod 600 ~/.config/yixin-api/api-key.json
```

If the user uses a secret manager, store the key there instead.

Legacy note: earlier platform versions issued one key per API and this skill recommended `~/.config/yixin-api/api-keys.json` with separate `search`/`fin_db` fields. Those per-API keys are obsolete — guide the user to subscribe to the product and replace the old file with the single-key shape above.

## Use The Key

Before making a request:

1. Load the key from the file or the `YIXIN_API_KEY` environment variable.
2. Send it as `X-API-KEY` — the same key works for both `search` and `fin_db`.

If the user has no key, guide them back to the portal to subscribe to the public API product.

## User-Friendly Limit Handling

If the gateway returns `429`, use this exact user-facing message:

```text
额度已用完，请联系销售升级：https://www.billionsintelligence.com
```
