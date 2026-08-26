---
title: Gateway Quick Start
---

# Gateway Quick Start

Get a format-translating HTTP proxy running in under 5 minutes.

## 1. Install

```bash
pip install "llm-rosetta[gateway]"
```

Or use Docker (Alpine binary, ~21 MB):

```bash
docker pull oaklight/llm-rosetta-gateway:latest
```

Or download a standalone binary from [GitHub Releases](https://github.com/Oaklight/llm-rosetta/releases) — no Python needed.

## 2. Create a config file

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

Provider names are user-defined strings. The `type` field specifies the API standard (`openai_chat`, `openai_responses`, `anthropic`, `google`).

## 3. Start the gateway

```bash
llm-rosetta-gateway
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
- [CLI Integrations](../gateway/cli-integrations.md) — use with Claude Code, Codex, Gemini CLI
- [Admin Panel](../gateway/admin-panel.md) — web UI for config and monitoring
