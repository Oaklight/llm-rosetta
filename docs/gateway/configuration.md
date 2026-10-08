---
title: Configuration
---

# Configuration

This page covers the gateway's configuration file format in detail.

!!! tip "Configuration reload"
    The gateway does **not** watch the config file for changes. To apply changes, either restart the gateway process or use the [Admin Panel](admin-panel.md) / [Admin API](../api/admin.md) — changes made through the admin interface are hot-reloaded without restart and persisted back to the config file.

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

Available types: `openai_chat`, `openai_responses`, `anthropic`, `google`, `google_interactions`.

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

### Dynamic Token Refresh (`token_command`)

For providers that use short-lived tokens (e.g. ALCF Inference Service with Globus OAuth), the gateway can run an external command to obtain and periodically refresh the API key:

```jsonc
"my-alcf": {
  "provider": "alcf--sophia",
  "token_command": ["python3", "/scripts/alcf-token.py"],
  "token_refresh_interval": 3600
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `token_command` | `list[str]` | — | Command argv; stdout is captured as the API key |
| `token_refresh_interval` | `int` | `3600` | Seconds between scheduled refreshes (minimum 60) |

When `token_command` is set:

1. The command runs **at startup** to seed the initial API key
2. A background task re-runs it every `token_refresh_interval` seconds
3. On upstream **401 responses**, the gateway triggers an immediate out-of-cycle refresh (debounced to avoid storms)
4. The command's stdout is used as the new key — comma-separated output is supported for multi-key round-robin

!!! note "`token_command` and `api_key` are mutually exclusive"
    If both are specified, the gateway refuses to start. Use one or the other.

#### ALCF example with Docker

ALCF providers use Globus OAuth tokens managed by `scripts/alcf-token.py`. When running in Docker, mount the token file and script into the container:

```yaml
# docker-compose.yaml
volumes:
  - ./config:/config
  - ~/.globus:/home/appuser/.globus              # Globus token file (rw for refresh)
  - ./scripts/alcf-token.py:/scripts/alcf-token.py:ro  # Refresh script
```

Setup:

1. **Login on the host** (one-time): `python3 scripts/alcf-token.py --login`
2. **Start the container** with the volume mounts above
3. **Add the provider** via the admin panel or `config.jsonc` with `token_command: ["python3", "/scripts/alcf-token.py"]`

The container's `appuser` (uid 1000) needs read-write access to `~/.globus` so the refresh token can be updated in place.

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

### Routing Strategy

Multi-provider model entries support a `strategy` field to control how
the gateway selects among providers:

```jsonc
"models": {
  "gpt-4o": {
    "providers": [
      {"name": "openai-prod", "weight": 5},
      {"name": "openai-backup", "weight": 1}
    ],
    "strategy": "weighted_round_robin"
  }
}
```

| Strategy | Description |
|----------|-------------|
| `weighted_round_robin` | **(default)** Smooth nginx-style interleaving based on weights. Weights [5, 1] yield the sequence A A A A A B, not burst-then-switch. |
| `affinity_round_robin` | SHA-256 hash of client identity (API key or IP) selects a deterministic preferred provider for cache locality. Falls back to weighted round-robin when no identity is available or only one provider is configured. |

The strategy is set per model.  Single-provider models ignore it.

## Gateway API Keys

Gateway API keys are managed through the **admin panel** — there is no
config-file setting for API keys.

Generate, rotate, and delete keys at `/admin` → **Keys** tab, or via the
[Admin API](../api/admin.md#api-keys). All keys are stored in a SQLite
keystore (`keys.db`).

When keys exist, all `/v1/*` endpoints require authentication using the
format native to each API standard:

| API Standard | Credential Format |
|-------------|-------------------|
| OpenAI Chat / Responses | `Authorization: Bearer <key>` |
| Anthropic | `x-api-key: <key>` |
| Google GenAI | `x-goog-api-key: <key>` or `?key=<key>` query param |

!!! note "Admin panel"
    The admin panel (`/admin/*`) does **not** require a gateway API key. You can protect it with the built-in `admin_password` option (see below), or use a reverse proxy (e.g. Caddy with `basicauth`, Nginx with `auth_basic`).

When no keys are configured, behavior depends on `open_on_no_keys` (default: `false` — all `/v1/*` requests are blocked).

## Admin Panel Security

### `admin_password`

Optional. When set, the admin panel (`/admin/*`) requires a password login before granting access. Sessions are tracked server-side in an in-memory session store.

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

!!! info "Session-based auth vs. internal token"
    Browser authentication uses **session-based cookies** (HttpOnly +
    SameSite=Lax, 30-minute inactivity timeout).  The `X-Admin-Token`
    header is a separate mechanism for programmatic API access — it
    validates against the internal proxy token via HMAC comparison, not
    the admin password.

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

## Rate Limiting

The gateway supports per-client rate limiting with single or multiple sliding windows. Configure in the `server` section:

```jsonc
{
  "server": {
    "rate_limit": "10/m"        // Single window: 10 requests per minute
  }
}
```

Multi-window rate limiting uses comma-separated window specs:

```jsonc
{
  "server": {
    "rate_limit": "10/m, 100/h"  // 10 per minute AND 100 per hour
  }
}
```

| Window suffix | Meaning |
|--------------|---------|
| `/s` | Per second |
| `/m` | Per minute |
| `/h` | Per hour |

Each client (identified by API key or IP) is tracked independently. When any window is exhausted, the request receives a 429 response with `Retry-After` header.

Rate limit state is visible in the admin panel via `GET /admin/api/rate-limits`.

## Routing Loop Detection

When multiple gateway instances are chained (e.g. a campus gateway forwarding to a cloud gateway), routing loops can occur. The gateway detects these via a hop-count header:

```jsonc
{
  "server": {
    "max_hops": 4              // Default: 4
  }
}
```

Each gateway increments the `X-Rosetta-Hops` header. When the count exceeds `max_hops`, the request is rejected with a 508 (Loop Detected) response.

## Soft Error Detection

Some upstream providers return HTTP 200 with an error body (e.g. rate limit HTML pages, JSON error objects). The gateway can detect these via shim-configured regex patterns and re-wrap them as proper error responses:

```jsonc
// In a provider shim YAML:
soft_error_patterns:
  - pattern: "rate limit exceeded"
    status: 429
  - pattern: "internal server error"
    status: 500
```

When a 200 response body matches a pattern, it is re-wrapped with the configured status code and proper error envelope.

## Fidelity Verification

The gateway verifies conversion fidelity for **same-format routes**
(e.g. OpenAI → OpenAI via different providers) by comparing the
original request/response body against the post-round-trip converted
body.

- **Same-format shadow diffs run always** — no configuration needed.
  When source and target use the same API format, the gateway
  automatically diffs the original vs. converted request and response.
- **Critical-severity diffs** trigger automatic error dumps, visible
  in the admin panel's Error Dumps section.
- Fidelity diff results are stored in the request's profile data
  under the `"fidelity"` key.

Cross-format routes do not run fidelity diffs because the source and
target formats are inherently different.

## Deferred Startup

By default, the gateway defers blocking startup work (provider connectivity checks, model list fetches) to background tasks. The gateway starts accepting requests immediately while these tasks complete in the background. Startup progress is visible in the admin panel.

## Error Dump Retention

Error dumps are capped by `error_dump_cap` in the `server` section:

```jsonc
{
  "server": {
    "error_dump_cap": 500      // Default: 500
  }
}
```

!!! warning "Deprecated key"
    The previous `error_max` key is deprecated and emits a warning at startup. Use `error_dump_cap` instead.

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

## Embedding Providers

The gateway can proxy `/v1/embeddings` requests with cross-format IR conversion (OpenAI ↔ Cohere ↔ Jina ↔ Voyage). Configure embedding providers and models separately from chat providers:

```jsonc
{
  "embedding_providers": {
    "jina-embed": { "type": "jina", "api_key": "${JINA_API_KEY}", "base_url": "https://api.jina.ai" },
    "voyage-embed": { "type": "voyage", "api_key": "${VOYAGE_API_KEY}", "base_url": "https://api.voyageai.com" }
  },
  "embedding_models": {
    "jina-embeddings-v3": "jina-embed",
    "voyage-3-large": "voyage-embed"
  },
  "default_embedding_format": "openai"  // Source format when auto-detect fails
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `embedding_providers` | `dict` | `{}` | Provider configs for embedding upstreams (same format as `providers`) |
| `embedding_models` | `dict` | `{}` | Model → provider mapping for embedding requests |
| `default_embedding_format` | `str` | `"openai"` | Source format fallback. Options: `openai`, `cohere`, `jina`, `voyage` |

When `embedding_providers` is not configured, `/v1/embeddings` falls back to passthrough mode (forwarded to the chat provider without IR conversion).

## Rerank Providers

The gateway can proxy `/v1/rerank` and `/v2/rerank` requests with cross-format IR conversion (Jina ↔ Cohere ↔ Voyage):

```jsonc
{
  "rerank_providers": {
    "jina-rerank": { "type": "jina", "api_key": "${JINA_API_KEY}", "base_url": "https://api.jina.ai" },
    "cohere-rerank": { "type": "cohere", "api_key": "${COHERE_API_KEY}", "base_url": "https://api.cohere.com" }
  },
  "rerank_models": {
    "jina-reranker-v2-base-multilingual": "jina-rerank",
    "rerank-v3.5": "cohere-rerank"
  },
  "default_rerank_format": "jina"  // Source format when auto-detect fails
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `rerank_providers` | `dict` | `{}` | Provider configs for rerank upstreams |
| `rerank_models` | `dict` | `{}` | Model → provider mapping for rerank requests |
| `default_rerank_format` | `str` | `"jina"` | Source format fallback. Options: `jina`, `cohere`, `voyage` |

The `/v2/rerank` endpoint auto-detects Cohere source format from the URL path.

## Decision Providers

The gateway can proxy `/v1/decision` and `/v1/systemone` requests for probabilistic structured decision models:

```jsonc
{
  "decision_providers": {
    "jev-prod": { "type": "typesafe_decision", "api_key": "${JEV_API_KEY}", "base_url": "https://api.typesafe.ai" }
  },
  "decision_models": {
    "jev-1": "jev-prod"
  },
  "default_decision_format": "typesafe_decision"
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `decision_providers` | `dict` | `{}` | Provider configs for decision upstreams |
| `decision_models` | `dict` | `{}` | Model → provider mapping for decision requests |
| `default_decision_format` | `str` | `"typesafe_decision"` | Source format fallback |

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
