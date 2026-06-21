---
title: Configuration
---

# Configuration

This page covers the gateway's configuration file format in detail.

## Providers

Each provider entry requires an `api_key`, `base_url`, and optionally a `type` specifying the API standard:

```jsonc
"providers": {
  "my-openai":   { "type": "openai_chat",      "api_key": "sk-...",     "base_url": "https://api.openai.com/v1" },
  "my-anthropic": { "type": "anthropic",        "api_key": "sk-ant-...", "base_url": "https://api.anthropic.com" },
  "my-google":   { "type": "google",            "api_key": "AIza...",    "base_url": "https://generativelanguage.googleapis.com" }
}
```

Provider names are user-defined strings (e.g. `"my-openai"`, `"prod-claude"`). The `type` field specifies which API standard to use.

Available types: `openai_chat`, `openai_responses`, `anthropic`, `google`.

### Using Shims

Instead of `type`, you can use a `shim` field to reference a registered provider shim. A shim is a lightweight identity card that declares which base API standard a provider uses, along with connection defaults and field-level transforms.

```jsonc
"providers": {
  "my-deepseek":   { "shim": "deepseek",   "api_key": "${DEEPSEEK_API_KEY}" },
  "my-volcengine": { "shim": "volcengine",  "api_key": "${VOLCENGINE_API_KEY}", "base_url": "https://ark.cn-beijing.volces.com/api/v3" }
}
```

When `shim` is specified:

- The **base type** is resolved automatically (e.g. `deepseek` → `openai_chat`)
- **Default `base_url`** and **`api_key` env var** are populated from the shim if not set in config
- **Field-level transforms** are applied during request/response conversion (e.g. Volcengine's shim strips `logprobs` and `top_logprobs` fields that its API does not support)

Built-in shims: `openai`, `openai_responses`, `anthropic`, `google`, `deepseek`, `volcengine`.

You can also register custom shims programmatically via `register_shim()`.

### Image Count Limits

Shims support `max_images` and `max_images_pattern` fields to enforce per-model image count limits:

| Field | Type | Description |
|-------|------|-------------|
| `max_images` | int | Maximum number of images allowed per request |
| `max_images_pattern` | str | Regex applied to the model name; the limit is only enforced for models whose names match the pattern |

When a request exceeds the limit, the oldest images are replaced with a text placeholder. If `max_images_pattern` is set, only matching model names are subject to the limit — others pass through unchanged.

**Example — built-in Argo OpenAI shim:**

The built-in Argo OpenAI shim declares `max_images: 50` with `max_images_pattern: "^(gpt|o\d)"`. This means:

- GPT and o-series models: truncated to 50 images
- Gemini and Claude models routed through the same provider: pass through unchanged

You can declare equivalent limits in a custom shim registered via `register_shim()`.

!!! tip "Resolution priority"
    The provider type resolution order is: `shim` → `type` → provider name (fallback).

!!! note "Backward compatibility"
    If both `shim` and `type` are omitted, the provider name itself is used as the type. This means configs using the old format (where provider names were `openai_chat`, `anthropic`, etc.) continue to work without changes.

### Enabling / Disabling Providers

Each provider supports an `enabled` field (default `true`). Disabled providers and their associated models are silently excluded from routing:

```jsonc
"my-openai": { "type": "openai_chat", "api_key": "sk-...", "base_url": "https://api.openai.com/v1", "enabled": false }
```

This is useful for temporarily taking a provider offline without deleting its configuration. The [admin panel](admin-panel.md) provides toggle switches for this.

### API Key Rotation

Each provider supports multiple API keys via comma-separated values. The gateway rotates through them in round-robin order:

```jsonc
"my-openai": { "type": "openai_chat", "api_key": "sk-key1,sk-key2,sk-key3", "base_url": "https://api.openai.com/v1" }
```

### Environment Variable Substitution

API keys support `${ENV_VAR}` syntax — values are read from environment variables at startup:

```jsonc
"my-openai": { "type": "openai_chat", "api_key": "${OPENAI_API_KEY}", "base_url": "https://api.openai.com/v1" }
```

### Per-Provider Proxy

Individual providers can use a specific proxy:

```jsonc
"my-anthropic": { "type": "anthropic", "api_key": "sk-ant-...", "base_url": "https://api.anthropic.com", "proxy": "http://proxy:8080" }
```

## Proxy Configuration

A global proxy can be set in the `server` section and applies to all providers unless overridden per-provider:

```jsonc
{
  "server": {
    "host": "0.0.0.0",
    "port": 8765,
    "proxy": "http://proxy.example.com:8080"
  }
}
```

Both HTTP and SOCKS5 proxies are supported:

```jsonc
// HTTP proxy
"proxy": "http://proxy.example.com:8080"

// SOCKS5 proxy (no auth)
"proxy": "socks5://proxy.example.com:1080"

// SOCKS5 proxy (with username/password)
"proxy": "socks5://username:password@proxy.example.com:1080"
```

The CLI `--proxy` flag overrides the config-level proxy for all providers.

## Unix Domain Socket

The gateway can listen on a Unix domain socket instead of TCP. This is useful for shared multi-user hosts (e.g. HPC login nodes) where `127.0.0.1` still exposes the service to all local users:

```jsonc
{
  "server": {
    "socket": "/run/user/1000/rosetta.sock"
  }
}
```

Or via CLI:

```bash
llm-rosetta-gateway --socket /run/user/$(id -u)/rosetta.sock
```

When `socket` is set, `host` and `port` are ignored. The socket file is:

- Created with **owner-only permissions** (`0600`) — other users on the host cannot connect
- **Automatically removed** on shutdown
- **Stale sockets cleaned up** on startup (if a previous instance crashed)

Combined with SSH `LocalForward`, this locks down the entire access chain end-to-end.

## Model Routing

The `models` section maps model names to providers:

```jsonc
"models": {
  "gpt-4o": "my-openai",
  "claude-sonnet-4-20250514": "my-anthropic",
  "gemini-2.0-flash": "my-google"
}
```

When a request arrives with `"model": "claude-sonnet-4-20250514"`, the gateway looks up `my-anthropic` and forwards accordingly.

### Model Capabilities

Models can optionally declare capabilities using the dict format:

```jsonc
"models": {
  "gpt-4o": { "provider": "my-openai", "capabilities": ["text", "vision", "tools"] },
  "gemini-2.0-flash": { "provider": "my-google", "capabilities": ["text", "tools"] }
}
```

Available capabilities: `text`, `vision`, `tools`, `embedding`, `reasoning`. If not specified, defaults to `["text"]`. Note that `embedding` is mutually exclusive with `vision`/`tools`, and `reasoning` is mutually exclusive with `embedding`.

Capabilities are displayed in the [admin panel](admin-panel.md) and can be edited there.

## Gateway API Key

Protect AI request endpoints with a gateway-level API key:

```jsonc
{
  "server": {
    "host": "0.0.0.0",
    "port": 8765,
    "api_key": "my-secret-gateway-key"
  }
}
```

When configured, all `/v1/*` endpoints require authentication using the format native to each API standard:

| API Standard | Credential Format |
|-------------|-------------------|
| OpenAI Chat / Responses | `Authorization: Bearer <key>` |
| Anthropic | `x-api-key: <key>` |
| Google GenAI | `x-goog-api-key: <key>` or `?key=<key>` query param |

The API key also supports `${ENV_VAR}` substitution:

```jsonc
"api_key": "${GATEWAY_API_KEY}"
```

!!! note "Admin panel"
    The admin panel (`/admin/*`) does **not** require the gateway API key. You can protect it with the built-in `admin_password` option (see below), or use a reverse proxy (e.g. Caddy with `basicauth`, Nginx with `auth_basic`).

When no `api_key` is configured, all requests pass through without authentication (backward compatible).

## Admin Panel Security

### `admin_password`

Optional. When set, the admin panel (`/admin/*`) requires a password login before granting access. Sessions are tracked with HMAC-based tokens, so no external session store is needed.

Supports `${ENV_VAR}` substitution:

```jsonc
{
  "server": {
    "admin_password": "${ADMIN_PASSWORD}"
  }
}
```

!!! tip
    If you expose the gateway publicly, setting `admin_password` is strongly recommended to prevent unauthorized access to provider configuration and request logs.

!!! warning "Unresolved placeholders"
    If `admin_password` contains an unresolved `${ENV_VAR}` placeholder (because the environment variable was not set at startup), the gateway **refuses to start** and logs a clear error. This prevents accidentally using the literal string `${ADMIN_PASSWORD}` as the password.

### `credential_visible`

Boolean, default `true`. When set to `false`, API key values are hidden across the admin UI — the copy and view controls are disabled. This is useful when the gateway is shared among multiple users and you want to prevent API keys from being read directly from the panel.

```jsonc
{
  "server": {
    "credential_visible": false
  }
}
```

!!! note
    This setting controls UI visibility only. The keys are still used by the gateway for upstream requests; they are simply not surfaced in the admin interface.

### `admin_cors_origins`

List of allowed origins for cross-origin requests to the admin API (`/admin/api/*`). By default (empty list), no `Access-Control-Allow-Origin` header is sent — only same-origin requests are permitted.

To allow a specific origin:

```jsonc
{
  "server": {
    "admin_cors_origins": ["https://my-dashboard.example.com"]
  }
}
```

!!! note
    CORS tightening applies to `/admin/api/*` endpoints only. The `/v1/*` proxy endpoints are unaffected.

## Debug Options

```jsonc
{
  "debug": {
    "verbose": true,       // Enable DEBUG-level logging
    "log_bodies": true     // Log full request/response bodies
  }
}
```

These can also be set via environment variables: `LLM_ROSETTA_VERBOSE=1`, `LLM_ROSETTA_LOG_BODIES=1`.

## Request Tracing

Every proxy request is assigned an `X-Request-ID` header. If the incoming request already carries this header, its value is preserved; otherwise a new UUID is generated. The header is:

- Forwarded to the upstream provider
- Included in all response headers (including error responses)
- Logged with a `[request_id]` prefix for end-to-end traceability

No configuration is required — request ID propagation is always active.

## Health Check Endpoints

The gateway exposes three health check endpoints:

| Endpoint | HTTP status | Description |
|----------|------------|-------------|
| `/health` | Always 200 | Gateway status: uptime, request counts, errors in the last hour, and per-provider health |
| `/health/live` | Always 200 | Kubernetes liveness probe — confirms the process is running |
| `/health/ready` | 200 / 503 | Kubernetes readiness probe — 503 when any provider is degraded |

Example `/health` response:

```json
{
  "status": "ok",
  "uptime": 3600.5,
  "requests_total": 1234,
  "errors_last_hour": 2,
  "providers": {
    "openai-prod":    { "status": "ok" },
    "anthropic-prod": { "status": "ok" }
  }
}
```

The `status` field is `"ok"` when all providers are healthy, or `"degraded"` when one or more providers are experiencing errors.

## Full Example

```jsonc
{
  "providers": {
    "openai-prod":    { "type": "openai_chat",      "api_key": "${OPENAI_API_KEY}",    "base_url": "https://api.openai.com/v1" },
    "openai-resp":    { "type": "openai_responses",  "api_key": "${OPENAI_API_KEY}",    "base_url": "https://api.openai.com/v1" },
    "anthropic-prod": { "type": "anthropic",         "api_key": "${ANTHROPIC_API_KEY}",  "base_url": "https://api.anthropic.com" },
    "google-prod":    { "type": "google",            "api_key": "${GOOGLE_API_KEY}",     "base_url": "https://generativelanguage.googleapis.com" },
    // Shim-based providers — base_url and transforms resolved automatically
    "deepseek":       { "shim": "deepseek",          "api_key": "${DEEPSEEK_API_KEY}" },
    "volcengine":     { "shim": "volcengine",         "api_key": "${VOLCENGINE_API_KEY}", "base_url": "https://ark.cn-beijing.volces.com/api/v3" }
  },
  "models": {
    "gpt-4o":                     { "provider": "openai-prod",    "capabilities": ["text", "vision", "tools"] },
    "claude-sonnet-4-20250514":   { "provider": "anthropic-prod", "capabilities": ["text", "vision", "tools"] },
    "gemini-2.0-flash":           { "provider": "google-prod",    "capabilities": ["text", "tools"] },
    "deepseek-r1":                { "provider": "deepseek",       "capabilities": ["text", "tools"] }
  },
  "server": {
    "host": "0.0.0.0",
    "port": 8765,
    "api_key": "${GATEWAY_API_KEY}",
    "admin_password": "${ADMIN_PASSWORD}",
    "credential_visible": false,
    "admin_cors_origins": []
  }
}
```
