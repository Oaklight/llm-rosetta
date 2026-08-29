---
title: Rerank API Formats
---

# Rerank API Formats

Rerank APIs allow you to re-order a list of documents by relevance to a query. Unlike chat completions which have a de facto standard (OpenAI Chat), rerank APIs have **no unified standard** — each provider defines its own request/response structure.

LLM-Rosetta identifies **3 format families** that cover all major rerank providers. Each provider's format is handled by a dedicated converter, enabling cross-provider rerank conversion through the IR.

## Overview

| Format Family | Result Field | Document Text | Usage Location | Converter Class |
|--------------|-------------|---------------|---------------|-----------------|
| Jina | `results` | Included by default | Top-level `usage` | `JinaRerankConverter` |
| Cohere | `results` | Not included | `meta.tokens` / `meta.billed_units` | `CohereRerankConverter` |
| Voyage | `data` | Optional | Top-level `usage` | `VoyageRerankConverter` |

## Provider Lineage

Each format family is used by multiple providers and platforms:

| Format Family | Providers |
|--------------|-----------|
| **Jina** | Jina AI, GPUStack, vLLM (`/v1/rerank`), Xinference, llama-box |
| **Cohere** | Cohere, Siliconflow, OpenVINO Model Server, vLLM (`/v2/rerank`) |
| **Voyage** | Voyage AI, MongoDB Atlas (Voyage backend) |

!!! note
    vLLM exposes both `/v1/rerank` (Jina-compatible) and `/v2/rerank` (Cohere-compatible) simultaneously for the same model.

## Request Format

All three families share a nearly identical request structure:

```json
{
  "model": "model-name",
  "query": "search query",
  "documents": ["document 1", "document 2", "..."],
  "top_n": 3
}
```

Key differences between families:

| Field | Jina | Cohere | Voyage |
|-------|------|--------|--------|
| Top-N results | `top_n` | `top_n` | `top_k` |
| Return documents | `return_documents` (default: `true`) | — | `return_documents` (default: `false`) |
| Truncation limit | — | `max_tokens_per_doc` (default: 4096) | `truncation` (bool, default: `true`) |

## Response Formats

### Jina Format

Used by: Jina AI, GPUStack, vLLM, Xinference, llama-box

```json
{
  "model": "jina-reranker-v2-base-multilingual",
  "object": "list",
  "usage": {
    "total_tokens": 54
  },
  "results": [
    {
      "index": 0,
      "relevance_score": 0.8397,
      "document": "Paris is the capital of France."
    },
    {
      "index": 2,
      "relevance_score": 0.1645,
      "document": "The Eiffel Tower is in Paris."
    }
  ]
}
```

**Characteristics:**

- Results in `results` array, sorted by relevance (descending)
- Document text returned as **plain string** (not wrapped in `{text: "..."}`)
- Usage at top level with `total_tokens` (some providers also report `prompt_tokens`)
- Has `object: "list"` field

### Cohere Format

Used by: Cohere (v2 API), Siliconflow (variant), OpenVINO Model Server

```json
{
  "id": "c317b8b2-d572-4725-af60-cfb856aa28c8",
  "results": [
    {
      "index": 0,
      "relevance_score": 0.8923
    },
    {
      "index": 2,
      "relevance_score": 0.2516
    }
  ],
  "meta": {
    "api_version": {
      "version": "2"
    },
    "billed_units": {
      "search_units": 1
    }
  }
}
```

**Characteristics:**

- Results in `results` array, **no document text** by default
- Has unique `id` field
- Usage in `meta.billed_units` (billing-oriented: `search_units`)
- Cohere v4+ also provides `meta.tokens` with `input_tokens`, `output_tokens`, `cached_tokens`

#### Siliconflow Variant

Siliconflow follows the Cohere structure but with richer usage reporting:

```json
{
  "id": "019fea4450327dbea799b2175a8cc34c",
  "results": [
    {
      "index": 0,
      "document": null,
      "relevance_score": 0.9998
    }
  ],
  "meta": {
    "tokens": {
      "input_tokens": 54,
      "output_tokens": 0,
      "image_tokens": 0
    },
    "billed_units": {
      "input_tokens": 54,
      "output_tokens": 0,
      "search_units": 0,
      "classifications": 0
    }
  }
}
```

Notable differences from Cohere:

- Includes `document: null` explicitly (Cohere omits the field entirely)
- `meta.tokens` always populated (Cohere v3 only has `billed_units`)
- `meta.billed_units` includes token counts alongside `search_units`

### Voyage Format

Used by: Voyage AI, MongoDB Atlas

```json
{
  "object": "list",
  "data": [
    {
      "relevance_score": 0.7188,
      "index": 0,
      "document": "Paris is the capital of France."
    },
    {
      "relevance_score": 0.4980,
      "index": 2,
      "document": "The Eiffel Tower is in Paris."
    }
  ],
  "model": "rerank-2-lite",
  "usage": {
    "total_tokens": 32
  }
}
```

**Characteristics:**

- Results in **`data`** array (not `results`) — mirrors OpenAI Embeddings response structure
- Document text returned as plain string when `return_documents: true`
- Usage at top level with `total_tokens` only
- Has `object: "list"` and `model` at top level

## IR Mapping

LLM-Rosetta normalizes all three formats into a unified IR:

| Provider Field | IR Field | Notes |
|----------------|----------|-------|
| `results` / `data` | `results` | Canonical name is `results` |
| `top_n` / `top_k` | `top_n` | Canonical name is `top_n` |
| `document` (string) / `document.text` | `document.text` | Normalized to `RerankDocument(text=...)` |
| `usage.total_tokens` / `meta.tokens.input_tokens` | `usage.total_tokens` | Token-based, billing fields dropped |

### Usage Normalization

| Provider | `total_tokens` source | `prompt_tokens` source | Dropped |
|----------|----------------------|----------------------|---------|
| Jina | `usage.total_tokens` | `usage.prompt_tokens` | — |
| Voyage | `usage.total_tokens` | — | — |
| Cohere v3 | — | — | `meta.billed_units.search_units` |
| Cohere v4 | `meta.tokens.input_tokens` (derived) | `meta.tokens.input_tokens` | `meta.billed_units.*` |
| Siliconflow | `meta.tokens.input_tokens` (derived) | `meta.tokens.input_tokens` | `meta.billed_units.*` |

## Cross-Provider Conversion

With the IR as hub, any format can be converted to any other:

```
Jina request ──→ IR ──→ Cohere request
                   └──→ Voyage request

Cohere response ──→ IR ──→ Jina response
                      └──→ Voyage response
```

Information loss boundaries:

- **Jina → Cohere**: document text is dropped (Cohere doesn't include documents in responses)
- **Cohere → Jina/Voyage**: no document text available to include
- **Usage**: Cohere v3's `search_units` is not convertible to token counts

### Library Usage

```python
from llm_rosetta.converters.rerank.jina import JinaRerankConverter
from llm_rosetta.converters.rerank.cohere import CohereRerankConverter

jina_conv = JinaRerankConverter()
cohere_conv = CohereRerankConverter()

# Jina rerank request
jina_request = {
    "model": "jina-reranker-v2-base-multilingual",
    "query": "What is deep learning?",
    "documents": ["Neural networks...", "Decision trees...", "Transformers..."],
    "top_n": 2,
}

# Jina → IR → Cohere
ir_request = jina_conv.request_from_provider(jina_request)
cohere_request, warnings = cohere_conv.request_to_provider(ir_request)
```
