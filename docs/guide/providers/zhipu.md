# Zhipu

Zhipu (GLM / BigModel) is supported through the OpenAI Chat Completions compatible API.

## Endpoints

| Shim Name | Base Type | Default Base URL | API Key Env |
|:---|:---|:---|:---|
| `zhipu` | `openai_chat` | `https://open.bigmodel.cn/api/paas/v4` | `ZHIPU_API_KEY` |

## Transforms

| Type | Transform | Purpose |
|:---|:---|:---|
| Post-IR | `strip_fields("n", "presence_penalty", "frequency_penalty", "logprobs", "top_logprobs", "logit_bias", "seed")` | Removes unsupported request fields |
| IR | `hoist_late_system_messages()` | Moves late system messages to conversation start |

## Gateway Configuration

```jsonc
{
  "providers": {
    "zhipu": {
      "shim": "zhipu",
      "api_key": "${ZHIPU_API_KEY}"
    }
  }
}
```
