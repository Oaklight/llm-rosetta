---
title: CLI Integrations
---

# CLI Integrations

The gateway is a drop-in backend for popular AI coding CLI tools. Each tool speaks a different API format — the gateway handles the translation automatically.

!!! note "Prerequisites"
    All examples assume the gateway is running at `http://localhost:8765`. Replace with your actual host and port.

## Claude Code

[Claude Code](https://docs.anthropic.com/en/docs/claude-code/overview) connects via the **Anthropic Messages API** (`/v1/messages`).

=== "Config File (Recommended)"

    ```bash
    claude config set -g apiKeyHelper "echo 'your-api-key'"
    claude config set -g env.ANTHROPIC_BASE_URL "http://localhost:8765"
    claude config set -g env.CLAUDE_CODE_SKIP_ANTHROPIC_AUTH "1"
    ```

    This writes to `~/.claude/settings.json`:

    ```json
    {
        "apiKeyHelper": "echo 'your-api-key'",
        "env": {
            "ANTHROPIC_BASE_URL": "http://localhost:8765",
            "CLAUDE_CODE_SKIP_ANTHROPIC_AUTH": "1"
        }
    }
    ```

=== "Environment Variables"

    ```bash
    export ANTHROPIC_BASE_URL="http://localhost:8765"
    export ANTHROPIC_API_KEY="your-api-key"
    export CLAUDE_CODE_SKIP_ANTHROPIC_AUTH=1
    claude
    ```

!!! important
    - `CLAUDE_CODE_SKIP_ANTHROPIC_AUTH=1` is **required** — it skips Anthropic's default authentication flow
    - Set `ANTHROPIC_BASE_URL` to the proxy root (e.g., `http://localhost:8765`), **not** `http://localhost:8765/v1/messages` — Claude Code appends the path automatically

**Supported**: chat, multi-turn, images, tool calls, streaming ✅

---

## Codex CLI (OpenAI)

[Codex CLI](https://github.com/openai/codex) connects via the **OpenAI Responses API** (`/v1/responses`).

=== "Config File (Recommended)"

    Create `~/.codex/config.toml`:

    ```toml
    model = "gpt-5-nano"
    model_provider = "rosetta"

    [model_providers.rosetta]
    name = "Rosetta Gateway"
    base_url = "http://localhost:8765/v1"
    env_key = "ROSETTA_API_KEY"
    wire_api = "responses"
    ```

    Then set the API key in your shell profile:

    ```bash
    export ROSETTA_API_KEY="your-api-key"
    ```

=== "Environment Variables"

    ```bash
    export OPENAI_BASE_URL="http://localhost:8765/v1"
    export OPENAI_API_KEY="your-api-key"
    codex "your prompt here"
    ```

!!! note
    Codex CLI uses the **Responses API** wire format by default. When using the config file approach, `wire_api = "responses"` makes this explicit.

**Supported**: chat, multi-turn, tool calls, streaming ✅

---

## Aider

[Aider](https://aider.chat/) supports both OpenAI and Anthropic backends.

=== "OpenAI Mode"

    ```bash
    export OPENAI_API_BASE="http://localhost:8765/v1"
    export OPENAI_API_KEY="your-api-key"
    aider --model gpt-5-nano
    ```

=== "Anthropic Mode"

    ```bash
    export ANTHROPIC_BASE_URL="http://localhost:8765"
    export ANTHROPIC_API_KEY="your-api-key"
    aider --model claude-sonnet-4-6
    ```

!!! tip
    You can add these to your `.aider.conf.yml` or shell profile for persistence.

**Supported**: chat, multi-turn, tool calls, streaming ✅

---

## Antigravity CLI (agy)

[Antigravity CLI](https://antigravity.google/docs/cli/install/) is Google's agentic coding CLI (successor to Gemini CLI). It uses the **Google GenAI API** (`/v1beta/models/...`).

=== "Config Files (Recommended)"

    **1. `~/.gemini/antigravity-cli/settings.json`**:

    ```json
    {
        "modelProvider": "gemini",
        "model": {
            "name": "gemini-3.5-flash"
        }
    }
    ```

    **2. Environment variables** (add to shell profile):

    ```bash
    export GEMINI_API_KEY="your-api-key"
    export GOOGLE_GEMINI_BASE_URL="http://localhost:8765"
    ```

    Then just run `agy`.

=== "Environment Variables"

    ```bash
    GOOGLE_GEMINI_BASE_URL=http://localhost:8765 \
    GEMINI_API_KEY=your-api-key \
    agy
    ```

The model name can be **any model** configured on your gateway — including non-Google models like `claude-sonnet-4-6`, `deepseek-v4-flash`, or `gpt-5-nano`. The gateway handles cross-format conversion transparently.

!!! warning "Planner model required"
    agy internally uses a hardcoded planner model (`gemini-3.1-pro-preview`) alongside your chosen model. This **cannot be overridden** — your gateway must have this model configured.

!!! tip "Using non-Google models"
    This is the key use case: run agy against Claude, DeepSeek, GPT, or any other provider. The gateway converts between Google GenAI format and the target provider's format automatically.

**Supported**: chat, streaming, vision, tool calls ✅

---

## OpenCode

[OpenCode](https://github.com/opencode-ai/opencode) supports OpenAI-compatible endpoints.

=== "Config File (Recommended)"

    Add a custom provider in `~/.config/opencode/opencode.json`:

    ```json
    {
        "provider": {
            "rosetta": {
                "npm": "@ai-sdk/openai-compatible",
                "name": "Rosetta Gateway",
                "options": {
                    "baseURL": "http://localhost:8765/v1",
                    "apiKey": "your-api-key"
                },
                "models": {
                    "gpt-5-nano": { "name": "GPT-5 Nano" },
                    "claude-sonnet-4-6": { "name": "Claude Sonnet 4.6" }
                }
            }
        }
    }
    ```

=== "Environment Variables"

    ```bash
    export OPENAI_BASE_URL="http://localhost:8765/v1"
    export OPENAI_API_KEY="your-api-key"
    opencode
    ```

**Supported**: chat, multi-turn, tool calls, streaming ✅

---

## Kilo Code

[Kilo Code](https://kilocode.ai/) (VS Code extension) uses the OpenAI Chat Completions API (`/v1/chat/completions`).

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
      },
      "options": {
        "apiKey": "your-api-key",
        "baseURL": "http://localhost:8765/v1"
      }
    }
  }
}
```

Then use: `kilo --model rosetta/claude-sonnet-4-20250514`

**Supported**: chat, multi-turn, tool calls, streaming ✅

---

## Ollama

[Ollama](https://ollama.com/) (v0.13+) exposes OpenAI-compatible endpoints locally, making it a natural fit as both an upstream provider and a client target.

### As an upstream provider

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

### As a client

Ollama v0.13+ supports three API formats that the gateway serves:

| Ollama Endpoint | Gateway Route | Converter |
|---|---|---|
| `/v1/chat/completions` | Same | `openai_chat` |
| `/v1/responses` | Same | `openai_responses` (v0.13.3+) |
| `/v1/messages` | Same | `anthropic` (v0.14.0+) |

---

## Generic SDKs

### OpenAI Python SDK

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8765/v1",
    api_key="your-api-key",
)

response = client.chat.completions.create(
    model="claude-sonnet-4-6",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(response.choices[0].message.content)
```

### Anthropic Python SDK

```python
import anthropic

client = anthropic.Anthropic(
    base_url="http://localhost:8765",
    api_key="your-api-key",
)

message = client.messages.create(
    model="gpt-5-nano",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Hello!"}],
)
print(message.content[0].text)
```

---

## Gemini CLI (Discontinued)

!!! warning "Discontinued"
    Gemini CLI was discontinued on June 18, 2026 and replaced by [Antigravity CLI (agy)](#antigravity-cli-agy). The configuration below is preserved for reference only.

??? note "Legacy Gemini CLI configuration"

    **`~/.gemini/.env`**:

    ```bash
    GEMINI_API_KEY=your-api-key
    GOOGLE_GEMINI_BASE_URL=http://localhost:8765
    ```

    **`~/.gemini/settings.json`**:

    ```json
    {
        "model": { "name": "gemini-2.5-pro" },
        "security": { "auth": { "selectedType": "gemini-api-key" } }
    }
    ```

---

## Summary

| Tool | API Format | Base URL Env Var | Value |
|------|-----------|-----------------|-------|
| Claude Code | Anthropic | `ANTHROPIC_BASE_URL` | `http://localhost:8765` |
| Codex CLI | OpenAI Responses | `OPENAI_BASE_URL` | `http://localhost:8765/v1` |
| Aider (OpenAI) | OpenAI | `OPENAI_API_BASE` | `http://localhost:8765/v1` |
| Aider (Anthropic) | Anthropic | `ANTHROPIC_BASE_URL` | `http://localhost:8765` |
| agy | Google GenAI | `GOOGLE_GEMINI_BASE_URL` | `http://localhost:8765` |
| OpenCode | OpenAI | `OPENAI_BASE_URL` | `http://localhost:8765/v1` |
| Kilo Code | OpenAI | — (config file) | `http://localhost:8765/v1` |
| OpenAI SDK | OpenAI | `OPENAI_BASE_URL` | `http://localhost:8765/v1` |
| Anthropic SDK | Anthropic | `ANTHROPIC_BASE_URL` | `http://localhost:8765` |
