---
title: CLI Integrations
---

# CLI Integrations

The gateway is a drop-in backend for popular AI coding CLI tools. Each tool speaks a different API format — the gateway handles the translation automatically.

## Claude Code

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

## Kilo Code

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

## OpenAI Codex CLI

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

## Ollama

[Ollama](https://ollama.com/) (v0.13+) exposes OpenAI-compatible endpoints locally, making it a natural fit as both an upstream provider and a client target for the gateway.

### Using Ollama as an upstream provider

Point a gateway provider at your local Ollama instance:

```jsonc
"providers": {
  "local-ollama": { "type": "openai_chat", "api_key": "ollama", "base_url": "http://localhost:11434/v1" }
},
"models": {
  "llama3.2": "local-ollama",
  "qwen3:8b": "local-ollama"
}
```

Then any client (Anthropic SDK, Google SDK, etc.) can query local Ollama models through the gateway with automatic format conversion.

### Using Ollama as a client

Ollama v0.13+ supports three API formats that the gateway can serve:

| Ollama Endpoint | Gateway Route | Converter |
|---|---|---|
| `/v1/chat/completions` | Same | `openai_chat` |
| `/v1/responses` | Same | `openai_responses` (v0.13.3+) |
| `/v1/messages` | Same | `anthropic` (v0.14.0+) |

This means tools built on Ollama's OpenAI-compatible layer can use the gateway to reach cloud providers (Anthropic, Google, etc.) without code changes — just point the base URL at the gateway.

## Gemini CLI

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
