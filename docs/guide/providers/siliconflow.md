---
title: SiliconFlow
---

# SiliconFlow

SiliconFlow (硅基流动) hosts open-weight models on one credential, with chat,
embeddings and rerank on the same host. A slice of the catalogue is
permanently free.

## Endpoints

| Shim Name | Base Type | Default Base URL | API Key Env |
|:---|:---|:---|:---|
| `siliconflow--chat_completions` | `chat_completions` | `https://api.siliconflow.cn/v1` | `SILICONFLOW_API_KEY` |
| `siliconflow--embedding` | `embeddings` (`openai`) | `https://api.siliconflow.cn/v1` | `SILICONFLOW_API_KEY` |
| `siliconflow--rerank` | `rerank` (`cohere`) | `https://api.siliconflow.cn/v1` | `SILICONFLOW_API_KEY` |

## The typed model catalogue

The plain `GET /v1/models` carries no type field, so a picker cannot tell a
chat model from an image one. SiliconFlow serves a typed catalogue on a
second host:

```
https://busy-bear.siliconflow.cn/api/v1/playground/comprehensive/all
```

Every entry carries `type`, `subType` and a `status` of `normal` or
`disable`. Measured at the time of writing: 201 entries, 91 active, of which
61 are `subType: chat`, 8 `embedding` and 6 `reranker`.

The shim owns that URL as an absolute `models_path` on each leaf, and each
leaf ships a `model_list_transform` that keeps only its own active slice. The
model id is `modelName` — the string the endpoint accepts.

## Measured endpoints

- Chat answers the OpenAI shape.
- Embeddings (`BAAI/bge-m3`) answer `data[].embedding` as a float list.
- Rerank (`BAAI/bge-reranker-v2-m3`) takes the Cohere request shape and
  answers `results[].index/relevance_score`.

## Gateway configuration

```jsonc
{
  "providers": {
    "sf": {
      "type": "siliconflow--chat_completions",
      "api_key": "${SILICONFLOW_API_KEY}"
    },
    "sf-rerank": {
      "type": "siliconflow--rerank",
      "api_key": "${SILICONFLOW_API_KEY}"
    }
  }
}
```

## Access

Free models need no credit card but do need identity verification with
mainland-Chinese documents. International users must contact support.
