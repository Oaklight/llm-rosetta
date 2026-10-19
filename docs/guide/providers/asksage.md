# AskSage

[AskSage](https://asksage.ai) is a government-focused AI platform that provides standard-format API endpoints for OpenAI, Anthropic, and Google Gemini models. LLM-Rosetta supports AskSage through four shims — one per API format — using a single API key.

!!! warning "Ships as a standalone package"
    Since 0.14.0 the four AskSage shims are no longer built into LLM-Rosetta; they
    ship in the
    [llm-rosetta-shim-asksage](https://github.com/Oaklight/llm-rosetta-shim-asksage)
    package. Declare it under `shims.remotes` and the names below keep working.
    **Settings → Shim Sources** offers the same entry as a one-click preset.

    ```jsonc
    "shims": {
      "remotes": [{
        "url": "https://github.com/Oaklight/llm-rosetta-shim-asksage",
        "path": "src/llm_rosetta_shim_asksage/providers",
        "token_env": "GITHUB_TOKEN"      // the repository is private
      }]
    }
    ```

## Endpoints

| Shim Name | Base Type | Default Base URL | API Key Env |
|:---|:---|:---|:---|
| `asksage--chat_completions` | `chat_completions` | `https://api.asksage.ai/server/openai/v1` | `ASKSAGE_API_KEY` |
| `asksage--open_responses` | `open_responses` | `https://api.asksage.ai/server/openai/v1` | `ASKSAGE_API_KEY` |
| `asksage--messages` | `messages` | `https://api.asksage.ai/server/anthropic` | `ASKSAGE_API_KEY` |
| `asksage--google_generate` | `google_generate` | `https://api.asksage.ai/server/google/v1beta` | `ASKSAGE_API_KEY` |

## Authentication

AskSage's provider-compatible endpoints accept the API key directly — no token exchange required.

- **OpenAI endpoints**: standard `Authorization: Bearer <key>` header
- **Anthropic endpoint**: standard `x-api-key: <key>` header
- **Gemini endpoint**: uses `x-access-tokens: <key>` header (non-standard — handled by `connection.auth_header` override)

## Transforms

| Shim | Type | Transform | Purpose |
|:---|:---|:---|:---|
| `asksage--chat_completions` | Post-IR | `rename_field("max_tokens", "max_completion_tokens")` | AskSage rejects `max_tokens` on OpenAI endpoints |
| `asksage--open_responses` | Post-IR | `rename_field("max_tokens", "max_completion_tokens")` | Same |

The Anthropic and Gemini shims require no transforms — AskSage uses standard formats for these.

## Available Models

Each endpoint exposes a different subset of models:

**OpenAI endpoints**: GPT models (`gpt-4.1-*`, `gpt-5-*`, `gpt-6-*`, `gpt-o3`, `gpt-o4-mini`), Bedrock models (`aws-bedrock-*`)

**Anthropic endpoint**: Claude models (`claude-haiku-4-5`, `claude-sonnet-4-5`, `claude-opus-*`)

**Gemini endpoint**: Gemini models (`gemini-2.5-flash`, `gemini-2.5-pro`, `gemini-3-flash`, `gemini-3-pro`)

!!! note "No Gemini model list endpoint"
    The Gemini endpoint does not provide a `/models` list API. You need to specify model names directly in the gateway config.

## Instance-specific Base URLs

AskSage deployments use instance-specific hostnames. The default `api.asksage.ai` is the commercial instance. For other instances (e.g., Argonne National Lab), override the base URL in your gateway config:

```jsonc
{
  "providers": {
    "asksage--chat_completions": {
      "api_key": "${ASKSAGE_API_KEY}",
      "base_url": "https://api.asksage.anl.gov/server/openai/v1"
    }
  }
}
```

## Gateway Configuration

```jsonc
{
  "providers": {
    "asksage--chat_completions": {
      "api_key": "${ASKSAGE_API_KEY}"
    },
    "asksage--open_responses": {
      "api_key": "${ASKSAGE_API_KEY}"
    },
    "asksage--messages": {
      "api_key": "${ASKSAGE_API_KEY}"
    },
    "asksage--google_generate": {
      "api_key": "${ASKSAGE_API_KEY}"
    }
  },
  "models": {
    "gpt-4.1-nano": "asksage--chat_completions",
    "gpt-4.1-mini": "asksage--chat_completions",
    "claude-haiku-4-5": "asksage--messages",
    "claude-sonnet-4-5": "asksage--messages",
    "gemini-2.5-flash": "asksage--google_generate",
    "gemini-3-flash": "asksage--google_generate"
  }
}
```
