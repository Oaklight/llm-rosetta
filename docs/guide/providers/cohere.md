---
title: Cohere
---

# Cohere

Cohere serves a native API and an OpenAI-compatible one. The native v2
shapes are used for embeddings and rerank; chat goes through the
compatibility endpoint, which speaks the OpenAI Chat Completions shape.

## Endpoints

| Shim Name | Base Type | Default Base URL | API Key Env |
|:---|:---|:---|:---|
| `cohere--chat_completions` | `chat_completions` | `https://api.cohere.com/compatibility/v1` | `COHERE_API_KEY` |
| `cohere--embedding` | `embeddings` (`cohere`) | `https://api.cohere.com` | `COHERE_API_KEY` |
| `cohere--rerank` | `rerank` (`cohere`) | `https://api.cohere.com` | `COHERE_API_KEY` |

The chat leaf overrides the vendor manifest's native `base_url`; the
credential is declared once, on the manifest, and every lane inherits it.

## Notes

- Chat and tool calls answer on both `api.cohere.com` and `api.cohere.ai`;
  the shim keeps `.com` so every lane shares one host.
- `/compatibility/v1/models` lists the same catalogue as the native API. The
  model-list hook parses `name` rather than `id`, because that is the field
  Cohere's own listing uses.
- Embeddings require `input_type`; the shim supplies `retrieval_document` as
  the default and a client's own `task_type` still wins.

## Gateway configuration

```jsonc
{
  "providers": {
    "cohere": {
      "type": "cohere--chat_completions",
      "api_key": "${COHERE_API_KEY}"
    },
    "cohere-embed": {
      "type": "cohere--embedding",
      "api_key": "${COHERE_API_KEY}"
    }
  }
}
```
