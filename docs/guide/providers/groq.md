---
title: Groq
---

# Groq

Groq serves open-weight models on its own inference hardware through an
OpenAI-compatible endpoint.

## Endpoint

| Shim Name | Base Type | Default Base URL | API Key Env |
|:---|:---|:---|:---|
| `groq--chat_completions` | `chat_completions` | `https://api.groq.com/openai/v1` | `GROQ_API_KEY` |

## Transforms

| Type | Transform | Purpose |
|:---|:---|:---|
| Pre-IR | `rename_reasoning()` | Renames `message`/`delta` `reasoning` → `reasoning_content` |
| IR | `hoist_late_system_messages()` | Moves late system messages to conversation start |

The rename is a no-op on a response that carries no reasoning, so the same
shim serves the non-reasoning models. `effort_field` is `reasoning_effort`.

!!! note
    The `reasoning` field name follows Groq's documented `openai/gpt-oss-*`
    response shape. It has not been probed live.

## Rate limits

The free plan gives the chat models 30 RPM, 1,000 RPD, 8,000 TPM and
200,000 TPD. Exact limits are per organisation and are read from the account
limits page, not from this shim.

## Gateway configuration

```jsonc
{
  "providers": {
    "groq": {
      "type": "groq--chat_completions",
      "api_key": "${GROQ_API_KEY}"
    }
  }
}
```
