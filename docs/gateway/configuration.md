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

!!! note "Backward compatibility"
    If `type` is omitted, the provider name itself is used as the type. This means configs using the old format (where provider names were `openai_chat`, `anthropic`, etc.) continue to work without changes.

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

Available capabilities: `text`, `vision`, `tools`. If not specified, defaults to `["text"]`.

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
    The admin panel (`/admin/*`) does **not** require the gateway API key. If you need to protect the admin panel, use a reverse proxy (e.g. Caddy with `basicauth`, Nginx with `auth_basic`).

When no `api_key` is configured, all requests pass through without authentication (backward compatible).

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

## Full Example

```jsonc
{
  "providers": {
    "openai-prod":    { "type": "openai_chat",      "api_key": "${OPENAI_API_KEY}",    "base_url": "https://api.openai.com/v1" },
    "openai-resp":    { "type": "openai_responses",  "api_key": "${OPENAI_API_KEY}",    "base_url": "https://api.openai.com/v1" },
    "anthropic-prod": { "type": "anthropic",         "api_key": "${ANTHROPIC_API_KEY}",  "base_url": "https://api.anthropic.com" },
    "google-prod":    { "type": "google",            "api_key": "${GOOGLE_API_KEY}",     "base_url": "https://generativelanguage.googleapis.com" }
  },
  "models": {
    "gpt-4o":                     { "provider": "openai-prod",    "capabilities": ["text", "vision", "tools"] },
    "claude-sonnet-4-20250514":   { "provider": "anthropic-prod", "capabilities": ["text", "vision", "tools"] },
    "gemini-2.0-flash":           { "provider": "google-prod",    "capabilities": ["text", "tools"] }
  },
  "server": {
    "host": "0.0.0.0",
    "port": 8765,
    "api_key": "${GATEWAY_API_KEY}"
  }
}
```
