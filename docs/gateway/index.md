---
title: Gateway
---

# Gateway

The LLM-Rosetta Gateway is an HTTP proxy that translates between LLM provider API formats in real time. Send requests in any supported format — the gateway converts and forwards them to the configured upstream provider.

```text
Client (OpenAI format) ──→ Gateway ──→ Anthropic API
Client (Anthropic format) ──→ Gateway ──→ OpenAI API
Client (Google format) ──→ Gateway ──→ Any provider
```

## Installation

```bash
pip install "llm-rosetta[gateway]"
```

This installs the gateway dependencies: [Starlette](https://www.starlette.io/), [uvicorn](https://www.uvicorn.org/), and [httpx](https://www.python-httpx.org/).

## Quick Start

### 1. Create a config file

Create a `config.jsonc` (JSON with comments):

```jsonc
{
  "providers": {
    "openai_chat":      { "api_key": "${OPENAI_API_KEY}",    "base_url": "https://api.openai.com/v1" },
    "openai_responses": { "api_key": "${OPENAI_API_KEY}",    "base_url": "https://api.openai.com/v1" },
    "anthropic":        { "api_key": "${ANTHROPIC_API_KEY}",  "base_url": "https://api.anthropic.com" },
    "google":           { "api_key": "${GOOGLE_API_KEY}",     "base_url": "https://generativelanguage.googleapis.com" }
  },
  "models": {
    "gpt-4o": "openai_chat",
    "gpt-4o-mini": "openai_chat",
    "claude-sonnet-4-20250514": "anthropic",
    "claude-3-5-haiku-20241022": "anthropic",
    "gemini-2.0-flash": "google",
    "gemini-2.5-pro": "google"
  },
  "server": {
    "host": "0.0.0.0",
    "port": 8765
  }
}
```

API keys support `${ENV_VAR}` syntax — values are read from environment variables at startup.

### 2. Start the gateway

```bash
# CLI command (after pip install)
llm-rosetta-gateway

# Or specify a config file explicitly
llm-rosetta-gateway --config /path/to/config.jsonc

# Or as a Python module
python -m llm_rosetta.gateway
```

The gateway auto-discovers config files at these locations (first match wins):

1. `./config.jsonc` (current directory)
2. `~/.config/llm-rosetta-gateway/config.jsonc`
3. `~/.llm-rosetta-gateway/config.jsonc`

You can also bootstrap a config file using the `add` subcommands:

```bash
# Add providers with sensible defaults
llm-rosetta-gateway add provider openai_chat
llm-rosetta-gateway add provider anthropic
llm-rosetta-gateway add provider google

# Add model routing entries
llm-rosetta-gateway add model gpt-4o --provider openai_chat
llm-rosetta-gateway add model claude-sonnet-4-20250514 --provider anthropic
llm-rosetta-gateway add model gemini-2.0-flash --provider google
```

### 3. Send requests

Use any provider's format — the gateway routes based on the model name:

```bash
# Send OpenAI-format request, routed to Anthropic
curl http://localhost:8765/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-sonnet-4-20250514",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'

# Send Anthropic-format request, routed to OpenAI
curl http://localhost:8765/v1/messages \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4o-mini",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 100
  }'
```

## Endpoints

| Path | Source Format | Description |
|------|-------------|-------------|
| `POST /v1/chat/completions` | OpenAI Chat | Drop-in for OpenAI SDK |
| `POST /v1/messages` | Anthropic | Drop-in for Anthropic SDK |
| `POST /v1/responses` | OpenAI Responses | Drop-in for OpenAI Responses SDK |
| `POST /v1beta/models/{model}:generateContent` | Google GenAI | Drop-in for Google REST API |
| `POST /v1beta/models/{model}:streamGenerateContent` | Google GenAI (streaming) | Drop-in for Google streaming |
| `GET /health` | — | Health check |

The endpoint path determines the source format — no auto-detection needed.

## Streaming

Streaming is supported for all provider combinations. Request streaming the same way you would with the native API:

```bash
# OpenAI-format streaming, routed to any provider
curl http://localhost:8765/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-sonnet-4-20250514",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": true
  }'
```

The gateway converts SSE chunks in real time between provider formats.

## Configuration

### Providers

Each provider entry needs an `api_key` and `base_url`:

```jsonc
"providers": {
  "openai_chat":      { "api_key": "sk-...",     "base_url": "https://api.openai.com/v1" },
  "anthropic":        { "api_key": "sk-ant-...", "base_url": "https://api.anthropic.com" },
  "google":           { "api_key": "AIza...",    "base_url": "https://generativelanguage.googleapis.com" }
}
```

Provider keys must be one of: `openai_chat`, `openai_responses`, `anthropic`, `google`.

#### API Key Rotation

Each provider supports multiple API keys via comma-separated values. The gateway rotates through them in round-robin order:

```jsonc
"openai_chat": { "api_key": "sk-key1,sk-key2,sk-key3", "base_url": "https://api.openai.com/v1" }
```

#### Per-Provider Proxy

Individual providers can use a specific proxy:

```jsonc
"anthropic": { "api_key": "sk-ant-...", "base_url": "https://api.anthropic.com", "proxy": "http://proxy:8080" }
```

### Proxy Configuration

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

The CLI `--proxy` flag overrides the config-level proxy for all providers.

### Model Routing

The `models` section maps model names to providers:

```jsonc
"models": {
  "gpt-4o": "openai_chat",
  "claude-sonnet-4-20250514": "anthropic",
  "gemini-2.0-flash": "google"
}
```

When a request arrives with `"model": "claude-sonnet-4-20250514"`, the gateway looks up `anthropic` and forwards accordingly.

### CLI Options

```
llm-rosetta-gateway [OPTIONS] [COMMAND]

Options:
  --config, -c PATH    Config file path (auto-discovered if omitted)
  --version, -V        Show version and exit
  --no-banner          Suppress the startup banner
  --edit, -e           Open config file in $EDITOR for editing
  --host HOST          Override server host
  --port PORT          Override server port
  --proxy URL          HTTP/SOCKS proxy URL for all upstream requests
  --log-level LEVEL    Log level: debug, info, warning, error (default: info)

Commands:
  add provider <name>  Add a provider entry to config
    --api-key KEY        API key or ${ENV_VAR} placeholder
    --base-url URL       Provider base URL (auto-filled for known providers)

  add model <name>     Add a model routing entry to config
    --provider NAME      Target provider name
```

#### Config auto-discovery

When `--config` is not specified, the gateway searches these paths in order:

1. `./config.jsonc` — current working directory
2. `~/.config/llm-rosetta-gateway/config.jsonc` — XDG standard location
3. `~/.llm-rosetta-gateway/config.jsonc` — dotfile convention

## Programmatic Usage

The gateway can also be used as a library:

```python
from llm_rosetta.gateway import create_app, GatewayConfig, load_config

# Load config and create ASGI app
raw = load_config("config.jsonc")
config = GatewayConfig(raw)
app = create_app(config)

# Mount in your own ASGI application, or run with any ASGI server
import uvicorn
uvicorn.run(app, host="0.0.0.0", port=8765)
```

## How It Works

The gateway uses LLM-Rosetta's converter pipeline:

```text
1. Incoming request (source format)
2. source_converter.request_from_provider() → IR Request
3. Look up model → target provider
4. target_converter.request_to_provider() → target format
5. Forward to upstream API
6. target_converter.response_from_provider() → IR Response
7. source_converter.response_to_provider() → source format
8. Return to client
```

For streaming, the same pipeline operates at the SSE chunk level using `stream_response_from_provider()` and `stream_response_to_provider()` with `StreamContext` for stateful conversion.
