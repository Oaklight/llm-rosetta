# xAI

xAI (Grok) is supported through the OpenAI Chat Completions compatible API.

## Endpoints

| Shim Name | Base Type | Default Base URL | API Key Env |
|:---|:---|:---|:---|
| `xai` | `openai_chat` | `https://api.x.ai/v1` | `XAI_API_KEY` |

## Reasoning Support

| Field | Value |
|:---|:---|
| Effort field | `reasoning_effort` |
| Effort range | `minimal` → `xhigh` |

!!! note "Extended Effort Range"
    xAI supports an extended effort range up to `xhigh`, which goes beyond the standard `high` ceiling used by most other providers.

## Transforms

| Type | Transform | Purpose |
|:---|:---|:---|
| Post-IR | `strip_fields("logit_bias")` | Removes unsupported request field |
| IR | `hoist_late_system_messages()` | Moves late system messages to conversation start |

## Gateway Configuration

```jsonc
{
  "providers": {
    "xai": {
      "shim": "xai",
      "api_key": "${XAI_API_KEY}"
    }
  }
}
```
