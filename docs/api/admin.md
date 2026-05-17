---
title: Admin API
---

# Admin API Reference

The gateway ships a built-in REST API under `/admin/api/` that powers the
[Admin Panel](../gateway/admin-panel.md) and can be consumed by scripts or
external tooling.  All endpoints return JSON unless otherwise noted.

!!! tip "Base path"
    Every endpoint below is relative to the gateway's listen address, e.g.
    `http://localhost:8765`.

---

## Authentication

### Check auth requirement

Check whether the admin panel requires a password.

```
GET /admin/api/auth-check
```

**Response**

```json
{"requires_auth": true}
```

| Field | Type | Description |
|-------|------|-------------|
| `requires_auth` | `bool` | `true` when `server.admin_password` is set in the config. |

---

### Login

Validate the admin password and obtain a session token.  Rate-limited to
5 failures per IP; the IP is locked out for 5 minutes after the threshold.

```
POST /admin/api/login
```

**Request body**

```json
{"password": "my-admin-password"}
```

**Success response**

```json
{"ok": true, "token": "<session-token>"}
```

**Error responses**

| Status | Body | Condition |
|--------|------|-----------|
| 400 | `{"error": "Admin password not configured"}` | No `server.admin_password` in config |
| 400 | `{"error": "Invalid JSON body"}` | Malformed request body |
| 401 | `{"error": "Invalid password"}` | Wrong password |
| 429 | `{"error": "Too many failed attempts. Try again in Ns."}` | Rate-limited |

---

## Configuration

### Get config

Return the full gateway configuration with masked API keys.

```
GET /admin/api/config
```

**Response**

```json
{
  "config_path": "/home/user/.config/llm-rosetta-gateway/config.jsonc",
  "providers": {
    "openai": {
      "api_key": "sk-a***abcd",
      "base_url": "https://api.openai.com/v1",
      "type": "openai_chat"
    }
  },
  "models": {
    "gpt-4o": {
      "provider": "openai",
      "capabilities": ["text", "vision", "tools"]
    }
  },
  "server": {"proxy": "http://proxy:8080"},
  "credential_visible": false,
  "known_provider_types": ["openai_chat", "openai_responses", "anthropic", "google"],
  "registered_shims": [
    {
      "name": "openai",
      "base": "openai_chat",
      "logo": "https://...",
      "default_base_url": "https://api.openai.com/v1",
      "default_api_key_env": "OPENAI_API_KEY"
    }
  ]
}
```

!!! note "Key masking"
    Literal API keys are masked (`sk-a***abcd`).  Environment variable
    placeholders like `${OPENAI_API_KEY}` are returned as-is.

---

### Add / update provider

Create a new provider or update an existing one.  Supports renaming
(with automatic model reference updates).

```
PUT /admin/api/config/providers/<name>
```

**Request body**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `api_key` | `string` | Yes* | Provider API key.  Omit or send empty to keep the existing key when editing. |
| `base_url` | `string` | Yes | Provider base URL. |
| `type` | `string` | No | API standard or shim name (e.g. `openai_chat`, `anthropic`). |
| `proxy` | `string` | No | Per-provider proxy URL. |
| `rename_from` | `string` | No | Original name when renaming — old entry is removed and model refs updated. |

**Success response**

```json
{
  "ok": true,
  "provider": "openai",
  "providers": ["openai", "anthropic"]
}
```

**Errors**: 400 (missing fields, invalid JSON), 404 (rename source not found), 409 (rename target exists), 500 (disk / reload failure).

---

### Delete provider

```
DELETE /admin/api/config/providers/<name>?cascade=true
```

| Query param | Type | Default | Description |
|-------------|------|---------|-------------|
| `cascade` | `"true"` \| `"1"` | — | Also delete models that reference this provider. |

**Success response**

```json
{
  "ok": true,
  "deleted": "openai",
  "providers": ["anthropic"],
  "cascade_deleted_models": ["gpt-4o"]
}
```

**Errors**: 404 (not found), 409 (models still reference this provider and `cascade` not set).

---

### Toggle provider

Enable or disable a provider without removing it.

```
POST /admin/api/config/providers/<name>/toggle
```

**Response**

```json
{"ok": true, "provider": "openai", "enabled": false}
```

---

### Add / update model

```
PUT /admin/api/config/models/<name>
```

**Request body**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `provider` | `string` | Yes | Target provider name (must exist). |
| `capabilities` | `string[]` | No | List of capabilities.  Defaults to `["text"]`. |
| `upstream_model` | `string` | No | Upstream model ID if different from the gateway-facing name. |
| `rename_from` | `string` | No | Original name when renaming. |

**Success response**

```json
{
  "ok": true,
  "model": "gpt-4o",
  "provider": "openai",
  "capabilities": ["text", "vision", "tools"],
  "models": {"gpt-4o": "openai"}
}
```

---

### Delete model

```
DELETE /admin/api/config/models/<name>
```

**Response**

```json
{"ok": true, "deleted": "gpt-4o", "models": {}}
```

---

### Update server settings

Currently supports the global proxy setting.

```
PUT /admin/api/config/server
```

**Request body**

```json
{"proxy": "http://proxy:8080"}
```

Send an empty string (`""`) to remove the proxy.

**Response**

```json
{"ok": true, "server": {"proxy": "http://proxy:8080"}}
```

---

### Reload config

Force a hot-reload of the config from disk, without restarting the gateway.

```
POST /admin/api/config/reload
```

**Response**

```json
{
  "ok": true,
  "providers": ["openai", "anthropic"],
  "models": {"gpt-4o": "openai"}
}
```

---

### Fetch upstream models

Query a provider's `/v1/models` endpoint to discover available models.

```
GET /admin/api/config/providers/<name>/models
```

**Response**

```json
{
  "provider": "openai",
  "api_standard": "openai_chat",
  "models": ["gpt-4o", "gpt-4o-mini", "o3-mini"]
}
```

**Errors**: 404 (provider not found), 502 (upstream connection or HTTP error).

---

### Bulk-add models

Add multiple models for a provider in a single call.

```
POST /admin/api/config/models
```

**Request body**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `provider` | `string` | Yes | Target provider name. |
| `models` | `string[]` | Yes | List of model IDs to add. |
| `prefix` | `string` | No | Prefix prepended to each model name.  When used, the original ID is stored as `upstream_model`. |
| `capabilities` | `string[]` | No | Capabilities for all added models.  Defaults to `["text", "vision", "tools"]`. |

**Response**

```json
{
  "ok": true,
  "added": ["my/gpt-4o", "my/gpt-4o-mini"],
  "skipped": ["my/o3-mini"],
  "models": {"my/gpt-4o": "openai", "my/gpt-4o-mini": "openai"}
}
```

---

### Reveal provider API key

Return the unmasked API key for a provider.  Only available when
`server.credential_visible` is `true` in the config.

```
GET /admin/api/config/providers/<name>/key
```

**Response**

```json
{"api_key": "sk-abc123..."}
```

**Errors**: 403 (credential visibility disabled), 404 (provider not found).

---

## API Key Management

### List keys

Return all gateway API keys with masked values.

```
GET /admin/api/keys
```

**Response**

```json
{
  "keys": [
    {
      "id": "a1b2c3d4",
      "key": "rsk-***abcd",
      "label": "production",
      "created": "2025-01-15T10:30:00+00:00"
    }
  ]
}
```

!!! note "Legacy single key"
    If the config uses the older `server.api_key` field instead of
    `server.api_keys`, it is returned as a synthetic entry with
    `id: "default"`.

---

### Create key

Generate a new gateway API key (or register a manually specified one).

```
POST /admin/api/keys
```

**Request body**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `label` | `string` | No | Human-readable label. |
| `key` | `string` | No | Manual key value.  If omitted, a random `rsk-<hex>` key is generated. |

**Response**

```json
{
  "ok": true,
  "key": {
    "id": "e5f6a7b8",
    "key": "rsk-0123456789abcdef0123456789abcdef",
    "label": "staging",
    "created": "2025-01-15T12:00:00+00:00"
  }
}
```

!!! warning
    The full key value is returned **only once** in this response.
    Subsequent `GET /admin/api/keys` calls return masked values.

---

### Update key label

```
PUT /admin/api/keys/<key_id>
```

**Request body**

```json
{"label": "new-label"}
```

**Response**

```json
{"ok": true, "id": "a1b2c3d4", "label": "new-label"}
```

---

### Delete key

```
DELETE /admin/api/keys/<key_id>
```

**Response**

```json
{"ok": true, "deleted": "a1b2c3d4"}
```

---

### Reveal key

Return the unmasked value of a specific API key.  Requires
`server.credential_visible: true`.

```
GET /admin/api/keys/<key_id>/reveal
```

**Response**

```json
{"key": "rsk-0123456789abcdef0123456789abcdef"}
```

**Errors**: 403 (credential visibility disabled), 404 (key not found).

---

### Get internal token

Return the ephemeral internal token used by the admin panel to make test
requests through the gateway.

```
GET /admin/api/internal-token
```

**Response**

```json
{"token": "internal-abc123"}
```

---

## Observability

### Metrics snapshot

Return real-time gateway metrics including counters, breakdowns, and a
rolling time-series.

```
GET /admin/api/metrics?seconds=60
```

| Query param | Type | Default | Range | Description |
|-------------|------|---------|-------|-------------|
| `seconds` | `int` | `60` | 1–300 | Length of the time-series window. |

**Response**

```json
{
  "uptime_seconds": 3600.5,
  "total_requests": 1234,
  "total_errors": 12,
  "total_streams": 456,
  "error_rate": 0.0097,
  "active_streams": 3,
  "by_model": {"gpt-4o": 800, "claude-sonnet-4-20250514": 434},
  "by_source_provider": {"openai_chat": 1000, "anthropic": 234},
  "by_target_provider": {"openai": 800, "anthropic": 434},
  "by_status_code": {"200": 1222, "500": 12},
  "series": [
    {"t": -59, "count": 2, "avg_ms": 450.12, "errors": 0},
    {"t": -58, "count": 0, "avg_ms": 0, "errors": 0}
  ]
}
```

| Series field | Description |
|-------------|-------------|
| `t` | Negative offset in seconds (e.g. `-59` = 59 seconds ago). |
| `count` | Requests in that second. |
| `avg_ms` | Average latency (ms). |
| `errors` | Number of error responses (status >= 400). |

---

### Request log

Return paginated, filtered request log entries (newest first).

```
GET /admin/api/requests?limit=50&offset=0&model=gpt-4o&provider=openai&status=ok
```

| Query param | Type | Default | Description |
|-------------|------|---------|-------------|
| `limit` | `int` | `50` | Page size. |
| `offset` | `int` | `0` | Number of entries to skip. |
| `model` | `string` | — | Filter by model name. |
| `provider` | `string` | — | Filter by target provider. |
| `status` | `"ok"` \| `"error"` | — | `ok` = 2xx/3xx, `error` = 4xx/5xx. |

**Response**

```json
{
  "entries": [
    {
      "id": "a1b2c3d4e5f6...",
      "timestamp": "2025-01-15T12:00:00+00:00",
      "model": "gpt-4o",
      "source_provider": "openai_chat",
      "target_provider": "openai",
      "is_stream": false,
      "status_code": 200,
      "duration_ms": 1234.56,
      "api_key_label": "production"
    }
  ],
  "total": 100
}
```

---

### Clear request log

```
DELETE /admin/api/requests
```

**Response**

```json
{"ok": true}
```

---

### Network diagnostics

Run IP geolocation and Google connectivity checks through the gateway's
configured proxy (if any).

```
GET /admin/api/diagnostics/network
```

**Response**

```json
{
  "proxy": "http://proxy:8080",
  "ip": {
    "ok": true,
    "ip": "203.0.113.42",
    "country": "United States",
    "city": "San Francisco",
    "isp": "Example ISP"
  },
  "google": {
    "ok": true,
    "status": 204
  }
}
```

The `proxy` field is present only when a global proxy is configured.
Each sub-object includes `"ok": false` and an `"error"` message on failure.

---

## Model Testing

The admin API runs model tests asynchronously.  `POST` starts a task,
then poll `GET` for the result.

### Start test

```
POST /admin/api/test
```

**Request body**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `endpoint` | `string` | Yes | Gateway endpoint to call, e.g. `/v1/chat/completions`. |
| `payload` | `object` | Yes | Request body forwarded to the endpoint. |

**Response**

```json
{"task_id": "a1b2c3d4e5f6"}
```

The test request is routed through the gateway itself using an ephemeral
internal token, so it exercises the full proxy pipeline.

---

### Poll test result

```
GET /admin/api/test/<task_id>
```

**Response (pending)**

```json
{"status": "pending"}
```

**Response (completed)**

```json
{
  "status": "done",
  "status_code": 200,
  "body": {"id": "chatcmpl-...", "choices": [...]}
}
```

**Response (error)**

```json
{"status": "error", "error": "Connection refused"}
```

| Status value | Meaning |
|-------------|---------|
| `pending` | Task is still running. |
| `done` | Upstream responded; check `status_code` and `body`. |
| `error` | Exception during the request. |
| `cancelled` | Task was cancelled via `DELETE`. |

Tasks are automatically cleaned up after 5 minutes.

---

### Cancel test

```
DELETE /admin/api/test/<task_id>
```

**Response**

```json
{"ok": true}
```

---

## Common error format

All endpoints use a consistent error envelope:

```json
{"error": "Human-readable error message"}
```

Endpoints that write to disk may return a partial-success response when
the config was saved but the hot-reload failed:

```json
{
  "error": "Config saved but reload failed: ...",
  "saved": true,
  "reloaded": false
}
```
