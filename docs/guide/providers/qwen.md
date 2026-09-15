# Qwen

Qwen (Alibaba Cloud / DashScope) is supported through the OpenAI Chat Completions compatible API.

## Endpoints

| Shim Name | Base Type | Default Base URL | API Key Env |
|:---|:---|:---|:---|
| `qwen` | `openai_chat` | `https://dashscope.aliyuncs.com/compatible-mode/v1` | `DASHSCOPE_API_KEY` |

## Transforms

| Type | Transform | Purpose |
|:---|:---|:---|
| Post-IR | `strip_fields("frequency_penalty", "logit_bias")` | Removes unsupported request fields |
| IR | `hoist_late_system_messages()` | Moves late system messages to conversation start |

## Gateway Configuration

```jsonc
{
  "providers": {
    "qwen": {
      "shim": "qwen",
      "api_key": "${DASHSCOPE_API_KEY}"
    }
  }
}
```
