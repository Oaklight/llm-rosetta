---
title: ModelScope
---

# ModelScope

ModelScope (魔搭) is Alibaba's model hub. Its inference gateway speaks the
OpenAI Chat Completions shape and is free for registered users.

## Endpoint

| Shim Name | Base Type | Default Base URL | API Key Env |
|:---|:---|:---|:---|
| `modelscope--chat_completions` | `chat_completions` | `https://api-inference.modelscope.cn/v1` | `MODELSCOPE_API_KEY` |

## Model roster

`GET /v1/models` answers the OpenAI shape with 38 entries and no type field,
so the whole catalogue is offered to the picker.

## Access and quota

Access needs an Alibaba Cloud account bound to the ModelScope account, plus
real-name verification. The quota is dynamic: 2,000 requests/day per user in
total, capped at 500 per model, with concurrency limited dynamically.

## Gateway configuration

```jsonc
{
  "providers": {
    "modelscope": {
      "type": "modelscope--chat_completions",
      "api_key": "${MODELSCOPE_API_KEY}"
    }
  }
}
```
