# DeepSeek

DeepSeek is supported through the OpenAI Chat Completions compatible API, with specific handling for its R1 reasoning model's thinking behavior.

## Endpoints

| Shim Name | Base Type | Default Base URL | API Key Env |
|:---|:---|:---|:---|
| `deepseek` | `openai_chat` | `https://api.deepseek.com` | `DEEPSEEK_API_KEY` |

## Reasoning Support

| Field | Value |
|:---|:---|
| Effort field | `reasoning_effort` |
| Effort range | `low` → `max` |

**Thinking modes:**

| IR Mode | DeepSeek Value |
|:---|:---|
| `enabled` | `enabled` |
| `disabled` | `disabled` |

!!! warning "No Auto Mode"
    DeepSeek R1 requires explicit enable/disable for thinking — there is no `auto` mode. When the IR requests `mode: auto`, only the effort level is sent without a thinking block.

## Transforms

| Type | Transform | Purpose |
|:---|:---|:---|
| Post-IR | `strip_fields("n", "logit_bias", "seed")` | Removes unsupported request fields |
| IR | `hoist_late_system_messages()` | Moves late system messages to conversation start |

## Gateway Configuration

```jsonc
{
  "providers": {
    "deepseek": {
      "shim": "deepseek",
      "api_key": "${DEEPSEEK_API_KEY}"
    }
  }
}
```
