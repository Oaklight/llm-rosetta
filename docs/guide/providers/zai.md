---
title: Z.ai
---

# Z.ai

Z.ai is Zhipu AI's international platform. It serves the same GLM model
family as the domestic [Zhipu](zhipu.md) lane, on separate hosts and with a
separate credential, and it is the lane to use from outside mainland China.

## Endpoints

| Shim Name | Base Type | Default Base URL | API Key Env |
|:---|:---|:---|:---|
| `zai--chat_completions` | `chat_completions` | `https://api.z.ai/api/paas/v4` | `ZAI_API_KEY` |
| `zai--messages` | `messages` | `https://api.z.ai/api/anthropic` | `ZAI_API_KEY` |
| `zai--embedding` | `embeddings` (`openai`) | `https://api.z.ai/api/paas/v4` | `ZAI_API_KEY` |
| `zai--rerank` | `rerank` (`cohere`) | `https://api.z.ai/api/paas/v4` | `ZAI_API_KEY` |

One key signs every lane.

## What each lane declares

- **Chat Completions** clamps `temperature` to `[0, 1]`: a value of `2` is a
  400 upstream, so the conversion clamps it and reports the change under the
  IR path. `top_p` is not constrained — unlike the domestic lane, `0` is
  accepted. `n`, `seed` and `logprobs` are accepted, so no field stripping is
  declared.
- **Messages** is Anthropic-compatible: it returns `thinking` blocks that
  carry a `signature`, and accepts `thinking` and `output_config.effort`.
  There is no adaptive mode, so `auto` is not mapped.
- **Embeddings** use the OpenAI request and response shape (`embedding-3`
  answers a float list).
- **Rerank** uses the Cohere shape; the answer carries
  `results[].index/relevance_score` and no `document`.

## Gateway configuration

```jsonc
{
  "providers": {
    "zai": {
      "type": "zai--chat_completions",
      "api_key": "${ZAI_API_KEY}"
    },
    "zai-claude": {
      "type": "zai--messages",
      "api_key": "${ZAI_API_KEY}"
    }
  }
}
```

## Free tier

Some GLM models are priced free on Z.ai (`glm-4.7-flash`, `glm-4.6v-flash`).
A free lane can answer `429` while the service is overloaded, so retry rather
than treat it as a quota exhaustion.
