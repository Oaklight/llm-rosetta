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

You can also bootstrap a config file using the `init` or `add` subcommands:

```bash
# Create a template config at ~/.config/llm-rosetta-gateway/config.jsonc
llm-rosetta-gateway init

# Or build up a config incrementally with add subcommands
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
| `GET /v1/models` | OpenAI / Anthropic | List configured models (compatible with both SDKs) |
| `GET /v1beta/models` | Google GenAI | List configured models (Google SDK format) |
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
  init                 Create a template config.jsonc at ~/.config/llm-rosetta-gateway/

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

## CLI Integration

The gateway is a drop-in backend for popular AI coding CLI tools. Each tool speaks a different API format — the gateway handles the translation automatically.

### Claude Code

Claude Code uses the Anthropic Messages API (`/v1/messages`).

```bash
export ANTHROPIC_BASE_URL=http://localhost:8765
export ANTHROPIC_API_KEY=your-key  # or any placeholder
export CLAUDE_CODE_SKIP_ANTHROPIC_AUTH=1
claude --model claude-sonnet-4-20250514
```

Or in `~/.claude/settings.json`:

```json
{
  "env": {
    "ANTHROPIC_MODEL": "claude-sonnet-4-20250514",
    "ANTHROPIC_BASE_URL": "http://localhost:8765",
    "CLAUDE_CODE_SKIP_ANTHROPIC_AUTH": "1"
  }
}
```

**Supported**: chat, multi-turn, images, tool calls, streaming ✅

### Kilo Code

Kilo Code uses the OpenAI Chat Completions API (`/v1/chat/completions`).

In `~/.config/kilo/kilo.jsonc`, add a custom provider:

```jsonc
{
  "provider": {
    "rosetta": {
      "api": "openai",
      "name": "Rosetta Gateway",
      "models": {
        "claude-sonnet-4-20250514": {
          "name": "Claude Sonnet 4",
          "attachment": true,
          "tool_call": true,
          "cost": { "input": 0, "output": 0 },
          "limit": { "context": 200000, "output": 8192 }
        }
        // Add more models as needed
      },
      "options": {
        "apiKey": "your-key",
        "baseURL": "http://localhost:8765/v1"
      }
    }
  }
}
```

Then use: `kilo --model rosetta/claude-sonnet-4-20250514`

**Supported**: chat, multi-turn, tool calls, streaming ✅

### OpenAI Codex CLI

Codex CLI uses the OpenAI Responses API (`/v1/responses`).

Create `~/.codex/config.toml`:

```toml
model = "gpt-4o"
model_provider = "rosetta"

[model_providers.rosetta]
name = "Rosetta Gateway"
base_url = "http://localhost:8765/v1"
env_key = "ROSETTA_API_KEY"
wire_api = "responses"
```

Then:

```bash
export ROSETTA_API_KEY=your-key
codex "your prompt here"
```

**Supported**: chat, multi-turn, tool calls, streaming ✓

### Ollama

[Ollama](https://ollama.com/) (v0.13+) exposes OpenAI-compatible endpoints locally, making it a natural fit as both an upstream provider and a client target for the gateway.

#### Using Ollama as an upstream provider

Point a gateway provider at your local Ollama instance:

```jsonc
"providers": {
  "openai_chat": { "api_key": "ollama", "base_url": "http://localhost:11434/v1" }
},
"models": {
  "llama3.2": "openai_chat",
  "qwen3:8b": "openai_chat"
}
```

Then any client (Anthropic SDK, Google SDK, etc.) can query local Ollama models through the gateway with automatic format conversion.

#### Using Ollama as a client

Ollama v0.13+ supports three API formats that the gateway can serve:

| Ollama Endpoint | Gateway Route | Converter |
|---|---|---|
| `/v1/chat/completions` | Same | `openai_chat` |
| `/v1/responses` | Same | `openai_responses` (v0.13.3+) |
| `/v1/messages` | Same | `anthropic` (v0.14.0+) |

This means tools built on Ollama's OpenAI-compatible layer can use the gateway to reach cloud providers (Anthropic, Google, etc.) without code changes — just point the base URL at the gateway.

### Gemini CLI

Gemini CLI uses the Google GenAI API (`/v1beta/models/...`).

=== "Config Files (Recommended)"

    **`~/.gemini/.env`** — Gemini CLI auto-reads this file on startup:

    ```bash
    GEMINI_API_KEY=your-key
    GOOGLE_GEMINI_BASE_URL=http://localhost:8765
    ```

    **`~/.gemini/settings.json`** — set auth mode and default model:

    ```json
    {
        "model": {
            "name": "gemini-2.5-pro"
        },
        "security": {
            "auth": {
                "selectedType": "gemini-api-key"
            }
        }
    }
    ```

    With both files configured, just run `gemini` — no extra flags needed.

=== "Environment Variables"

    ```bash
    export GOOGLE_GEMINI_BASE_URL=http://localhost:8765
    export GEMINI_API_KEY=your-key
    gemini -m gemini-2.5-pro -p "your prompt here"
    ```

!!! tip "Bearer token authentication"
    If your upstream proxy expects Bearer token auth (e.g., OneAPI), add to `~/.gemini/.env`:

    ```bash
    GEMINI_API_KEY_AUTH_MECHANISM=bearer
    ```

    This sends the API key as a `Bearer` token in the `Authorization` header instead of as a query parameter.

!!! note "TTY requirement"
    Gemini CLI requires a TTY even in headless mode (`-p`). When running from scripts or non-interactive shells, wrap with `script`:

    ```bash
    script -qec 'gemini -m gemini-2.5-pro -p "your prompt"' /dev/null
    ```

!!! note "Network dependencies"
    Gemini CLI makes outbound connections to `github.com` and `play.googleapis.com` during startup. These must be reachable (directly or via proxy) for the CLI to function.

**Supported**: chat, streaming ✅

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
