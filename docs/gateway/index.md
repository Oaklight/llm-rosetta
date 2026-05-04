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

The gateway has **zero external runtime dependencies** — it uses vendored [zerodep](https://github.com/Oaklight/zerodep) `httpserver` and `httpclient` modules (stdlib-only, single-file).

## Quick Start

### 1. Create a config file

Create a `config.jsonc` (JSON with comments):

```jsonc
{
  "providers": {
    "my-openai":    { "type": "openai_chat",      "api_key": "${OPENAI_API_KEY}",    "base_url": "https://api.openai.com/v1" },
    "my-anthropic": { "type": "anthropic",         "api_key": "${ANTHROPIC_API_KEY}",  "base_url": "https://api.anthropic.com" },
    "my-google":    { "type": "google",            "api_key": "${GOOGLE_API_KEY}",     "base_url": "https://generativelanguage.googleapis.com" }
  },
  "models": {
    "gpt-4o": "my-openai",
    "gpt-4o-mini": "my-openai",
    "claude-sonnet-4-20250514": "my-anthropic",
    "gemini-2.0-flash": "my-google"
  },
  "server": {
    "host": "0.0.0.0",
    "port": 8765
  }
}
```

Provider names are user-defined strings. The `type` field specifies the API standard (`openai_chat`, `openai_responses`, `anthropic`, `google`). See [Configuration](configuration.md) for full details.

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

You can also bootstrap a config file using the `init` or `add` subcommands. See [CLI Reference](cli.md) for all options.

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
| `GET /v1/models` | OpenAI / Anthropic | List configured models (compatible with both SDKs) |
| `GET /v1beta/models` | Google GenAI | List configured models (Google SDK format) |
| `GET /health` | — | Health check |
| `GET /admin/` | — | [Admin panel](admin-panel.md) (web UI) |

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

## Authentication

Protect AI endpoints with a gateway API key in the `server` config:

```jsonc
"server": { "api_key": "my-secret-key" }
```

Requests must provide the key in the format native to each API standard (Bearer token, `x-api-key` header, etc.). See [Configuration — Gateway API Key](configuration.md#gateway-api-key) for details.

## Docker Deployment

The gateway is available as a pre-built image on DockerHub:

```bash
# Pull from DockerHub and run
docker pull oaklight/llm-rosetta-gateway:latest
docker run -p 8765:8765 -v /path/to/config:/config oaklight/llm-rosetta-gateway

# Or use Docker Compose (see docker/docker-compose.yaml)
cd docker && docker compose up -d
```

To build from source:

```bash
# Build with Makefile (uses local wheel if available, otherwise PyPI)
make build-docker

# Or build manually
docker build -t llm-rosetta-gateway .
```

Set `PUID`/`PGID` environment variables to match your host user's UID/GID. See `docker/docker-compose.yaml` for the full configuration example.

## Admin Panel

The gateway includes a built-in web admin panel at `/admin/` for managing configuration, monitoring real-time metrics, and viewing request logs. See the [Admin Panel](admin-panel.md) page for details.

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
