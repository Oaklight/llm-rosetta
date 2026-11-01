---
title: Gateway Quick Start
---

# Gateway Quick Start

Get a format-translating HTTP proxy running in under 5 minutes. The gateway is the fourth level of control: it owns the provider call, routing, keys and admin console, and keeps the upstream configuration in your deployment.

## 1. Install

```bash
pip install llm-rosetta-gateway
```

`pip install "llm-rosetta[gateway]"` also works as a transitional alias that installs both distributions.

Or use Docker (Alpine binary, ~21 MB):

```bash
docker pull oaklight/llm-rosetta-gateway:latest
```

See [`docker/docker-compose.yaml`](https://github.com/Oaklight/llm-rosetta/blob/master/docker/docker-compose.yaml) for a ready-to-use Compose file with volume mounts, environment variables, and ALCF token passthrough.

## 2. Create a config file

Create a `config.jsonc` (JSON with comments):

```jsonc
{
  "providers": {
    "my-openai":    { "type": "chat_completions",      "api_key": "${OPENAI_API_KEY}",    "base_url": "https://api.openai.com/v1" },
    "my-anthropic": { "type": "messages",         "api_key": "${ANTHROPIC_API_KEY}",  "base_url": "https://api.anthropic.com" },
    "my-google":    { "type": "google_generate",            "api_key": "${GOOGLE_API_KEY}",     "base_url": "https://generativelanguage.googleapis.com" }
  },
  "models": {
    "gpt-4o": "my-openai",
    "gpt-4o-mini": "my-openai",
    "claude-sonnet-4-20250514": "my-anthropic",
    "gemini-2.0-flash": "my-google"
  },
  "server": {
    "host": "127.0.0.1",
    "port": 8765,
    // Local demo only. For a shared deployment, bind 0.0.0.0 behind your
    // own network controls, create a gateway key in /admin, and send it
    // in the format's native auth header.
    "open_on_no_keys": true
  }
}
```

Provider names are user-defined strings. The `type` field specifies the API standard (`chat_completions`, `open_responses`, `messages`, `google_generate`, `google_interactions`).

## 3. Start the gateway

```bash
llm-rosetta-gateway init     # writes a template config
llm-rosetta-gateway          # serves on :8765
```

The gateway auto-discovers config files at these locations (first match wins):

1. `./config.jsonc` (current directory)
2. `~/.config/llm-rosetta-gateway/config.jsonc`
3. `~/.llm-rosetta-gateway/config.jsonc`

You can also specify a config file explicitly:

```bash
llm-rosetta-gateway --config /path/to/config.jsonc
```

## 4. Send requests

Use any provider's format — the gateway routes based on the model name:

```bash
# Send OpenAI-format request, routed to Anthropic
curl http://localhost:8765/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-sonnet-4-20250514",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

Streaming works too — just add `"stream": true`.

## Next Steps

- [Configuration](../gateway/configuration.md) — full config reference
- [CLI Reference](../gateway/cli.md) — all CLI options and subcommands
- [CLI Integrations](../gateway/cli-integrations.md) — use with Claude Code, Codex, OpenCode and Kilo Code
- [Three Levels and a Toolbox](../index.md#control-levels) — where the gateway sits
- [Admin Panel](../gateway/admin-panel.md) — web UI for config and monitoring
