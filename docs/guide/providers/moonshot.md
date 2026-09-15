# Moonshot

Moonshot (Kimi) is supported through the OpenAI Chat Completions compatible API.

## Endpoints

| Shim Name | Base Type | Default Base URL | API Key Env |
|:---|:---|:---|:---|
| `moonshot` | `openai_chat` | `https://api.moonshot.cn/v1` | `MOONSHOT_API_KEY` |

## Transforms

| Type | Transform | Purpose |
|:---|:---|:---|
| Post-IR | `strip_fields("logprobs", "top_logprobs", "logit_bias", "seed")` | Removes unsupported request fields |
| IR | `hoist_late_system_messages()` | Moves late system messages to conversation start |

## Gateway Configuration

```jsonc
{
  "providers": {
    "moonshot": {
      "shim": "moonshot",
      "api_key": "${MOONSHOT_API_KEY}"
    }
  }
}
```
