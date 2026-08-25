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

## Antigravity CLI (agy)

[Antigravity CLI](https://antigravity.google/docs/cli/install/) is Google's agentic coding CLI (successor to Gemini CLI). It uses the Google GenAI API (`/v1beta/models/...`).

**Install:**

```bash
curl -fsSL https://antigravity.google/cli/install.sh | bash
```

**Configure** `~/.gemini/antigravity-cli/settings.json`:

```json
{
    "modelProvider": "gemini",
    "model": {
        "name": "gemini-3.5-flash"
    }
}
```

The model name can be **any model** configured on your gateway — including non-Google models like `claude-sonnet-4-6`, `deepseek-v4-flash`, or `gpt-5-nano`. The gateway handles cross-format conversion transparently.

**Set environment variables:**

```bash
export GEMINI_API_KEY=your-gateway-api-key
export GOOGLE_GEMINI_BASE_URL=http://localhost:8765
```

**Run:**

```bash
agy
```

Or in headless mode:

```bash
agy -p "your prompt here"
```

!!! warning "Planner model required"
    agy internally uses a hardcoded planner model (`gemini-3.1-pro-preview`) alongside your chosen model. This **cannot be overridden** via `settings.json` or CLI flags — there is no user-facing configuration for it. Your gateway **must** have this model configured, or the planner will fail before your main model runs.

    Add it as an alias in your gateway config pointing to any capable provider:

    ```json
    "gemini-3.1-pro-preview": {
        "provider": "YourProvider",
        "capabilities": ["text", "vision", "tools", "reasoning"],
        "upstream_model": "any-capable-model"
    }
    ```

!!! tip "Using non-Google models"
    This is the key use case for llm-rosetta with agy: you can run agy against Claude, DeepSeek, GPT, or any other provider. The gateway converts between Google GenAI format and the target provider's format automatically.

    Tested combinations:

    | Model | Provider | Text | Vision | Tools |
    |-------|----------|:----:|:------:|:-----:|
    | `claude-sonnet-4-6` | Anthropic | ✅ | ✅ | ✅ |
    | `deepseek-v4-flash` | DeepSeek | ✅ | ✅ | ✅ |
    | `gemini-3.5-flash` | Google | ✅ | ✅ | ✅ |

**Supported**: chat, streaming, vision, tool use ✅

---

## Gemini CLI (Discontinued)

!!! warning "Discontinued"
    Gemini CLI was discontinued on June 18, 2026 and replaced by [Antigravity CLI (agy)](#antigravity-cli-agy). The configuration below is preserved for reference only.

??? note "Legacy Gemini CLI configuration"

    Gemini CLI used the Google GenAI API (`/v1beta/models/...`).

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
