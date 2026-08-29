---
title: Embedding API Formats
---

# Embedding API Formats

Embedding APIs convert text into dense vector representations for semantic search, clustering, and classification. Unlike rerank APIs where no standard exists, embedding APIs have a **de facto standard** set by OpenAI — but significant divergences remain, especially from Google GenAI and Cohere.

LLM-Rosetta identifies **5 format families** that cover all major embedding providers. Each provider's format is handled by a dedicated converter, enabling cross-provider embedding conversion through the IR.

## Overview

| Format Family | Input Field | Task Type Field | Dimensions Field | Response Vectors | Encoding Formats |
|--------------|-------------|-----------------|-----------------|-----------------|-----------------|
| OpenAI | `input` | — | `dimensions` | `data[].embedding` | `float`, `base64` |
| Google GenAI | `content` (Content/Parts) | `taskType` | `outputDimensionality` | `embedding.values` | `float` only |
| Cohere | `texts` | `input_type` | `output_dimension` | `embeddings.{type}[][]` | `float`, `base64`, `int8`, `uint8`, `binary`, `ubinary` |
| Voyage | `input` | `input_type` | `output_dimension` | `data[].embedding` | `float`, `base64`, `int8`, `uint8`, `binary`, `ubinary` |
| Jina | `input` | `task` | `dimensions` | `data[].embedding` | `float`, `base64`, `binary`, `ubinary` |

## Provider Lineage

| Format Family | Providers |
|--------------|-----------|
| **OpenAI** | OpenAI, Azure OpenAI, DeepSeek, Together AI, Fireworks, most OpenAI-compatible platforms |
| **Google GenAI** | Google Gemini, Vertex AI |
| **Cohere** | Cohere (v2 API) |
| **Voyage** | Voyage AI |
| **Jina** | Jina AI |

## Request Format

### OpenAI (de facto standard)

```json
{
  "model": "text-embedding-3-small",
  "input": ["hello world", "another text"],
  "encoding_format": "float",
  "dimensions": 512,
  "user": "user-1234"
}
```

### Google GenAI

Single text — `embedContent`:

```json
{
  "content": {"parts": [{"text": "hello world"}]},
  "taskType": "RETRIEVAL_DOCUMENT",
  "title": "Document Title"
}
```

Batch — `batchEmbedContents`:

```json
{
  "requests": [
    {"content": {"parts": [{"text": "text 1"}]}, "taskType": "RETRIEVAL_QUERY"},
    {"content": {"parts": [{"text": "text 2"}]}, "taskType": "RETRIEVAL_QUERY"}
  ]
}
```

### Cohere

```json
{
  "model": "embed-v4.0",
  "texts": ["hello world", "another text"],
  "input_type": "search_document",
  "embedding_types": ["float"],
  "truncate": "END",
  "output_dimension": 1024
}
```

### Voyage

```json
{
  "model": "voyage-4-large",
  "input": ["hello world", "another text"],
  "input_type": "document",
  "output_dtype": "float",
  "encoding_format": null,
  "output_dimension": 1024,
  "truncation": true
}
```

### Jina

```json
{
  "model": "jina-embeddings-v5-omni-small",
  "input": ["hello world", "another text"],
  "task": "retrieval.passage",
  "embedding_type": "float",
  "dimensions": 512,
  "normalized": true
}
```

### Key Request Differences

| Field | OpenAI | Google | Cohere | Voyage | Jina |
|-------|--------|--------|--------|--------|------|
| Input | `input: string[]` | `content: {parts}` | `texts: string[]` | `input: string[]` | `input: string[]` |
| Task type | — | `taskType` | `input_type` (required) | `input_type` | `task` |
| Dimensions | `dimensions` | `outputDimensionality` | `output_dimension` | `output_dimension` | `dimensions` |
| Truncation | — | `autoTruncate` (bool) | `truncate` (enum) | `truncation` (bool) | `truncate` (bool) |

## Response Formats

### OpenAI Format

Used by: OpenAI, Azure OpenAI, DeepSeek, Together AI, most OpenAI-compatible platforms

```json
{
  "object": "list",
  "data": [
    {"object": "embedding", "embedding": [0.0023, -0.009, ...], "index": 0},
    {"object": "embedding", "embedding": [0.0051,  0.012, ...], "index": 1}
  ],
  "model": "text-embedding-3-small",
  "usage": {"prompt_tokens": 8, "total_tokens": 8}
}
```

### Google GenAI Format

Single text response:

```json
{
  "embedding": {"values": [0.123, -0.456, ...]},
  "metadata": {"promptTokenCount": 5}
}
```

Batch response:

```json
{
  "embeddings": [
    {"values": [0.123, -0.456, ...]},
    {"values": [0.789, -0.012, ...]}
  ]
}
```

**Characteristics:**

- No `object`/`data`/`index` wrapper — vectors directly in `embedding.values`
- Model name in URL path, not in response body
- Only supports float output, no encoding format control

### Cohere Format

```json
{
  "id": "da6e531f-54c6-4a73-bf92-f60566d8d753",
  "embeddings": {
    "float": [[0.016, -0.008, ...], [0.031, 0.045, ...]]
  },
  "texts": ["hello world", "another text"],
  "meta": {
    "billed_units": {"input_tokens": 10.0},
    "tokens": {"input_tokens": 10.0, "output_tokens": 0.0}
  }
}
```

**Characteristics:**

- Vectors in `embeddings.{type}[][]` — a **dict of lists** keyed by embedding type
- Can request multiple embedding types simultaneously (e.g. `["float", "int8"]`)
- No per-vector `index` field — order matches input order
- Has unique `id` field and `texts` echo

### Voyage Format

```json
{
  "object": "list",
  "data": [
    {"object": "embedding", "embedding": [-0.0167, 0.0269, ...], "index": 0}
  ],
  "model": "voyage-4-large",
  "usage": {"total_tokens": 8}
}
```

**Characteristics:**

- OpenAI-compatible response structure
- Separates `output_dtype` (data type) from `encoding_format` (serialization) — orthogonal axes
- Usage only reports `total_tokens` (no `prompt_tokens`)

### Jina Format

```json
{
  "model": "jina-embeddings-v5-omni-small",
  "object": "list",
  "usage": {"total_tokens": 15, "prompt_tokens": 10},
  "data": [
    {"object": "embedding", "embedding": [0.123, -0.456, ...], "index": 0}
  ]
}
```

**Characteristics:**

- OpenAI-compatible response structure
- Supports multimodal inputs (images, video, audio, PDF) — not yet in IR
- Supports multi-vector (ColBERT) output with `return_multivector` — not yet in IR
- Has unique `late_chunking` and `normalized` options

## Task Type Normalization

Each provider uses different names for the same semantic concepts. The IR normalizes them:

| IR Canonical | OpenAI | Google | Cohere | Voyage | Jina |
|-------------|--------|--------|--------|--------|------|
| `retrieval_query` | — | `RETRIEVAL_QUERY` | `search_query` | `query` | `retrieval.query` |
| `retrieval_document` | — | `RETRIEVAL_DOCUMENT` | `search_document` | `document` | `retrieval.passage` |
| `semantic_similarity` | — | `SEMANTIC_SIMILARITY` | — | — | `text-matching` |
| `classification` | — | `CLASSIFICATION` | `classification` | — | `classification` |
| `clustering` | — | `CLUSTERING` | `clustering` | — | `clustering` |
| `question_answering` | — | `QUESTION_ANSWERING` | — | — | — |
| `fact_verification` | — | `FACT_VERIFICATION` | — | — | — |
| `code_retrieval_query` | — | `CODE_RETRIEVAL_QUERY` | — | — | `code.query` |
| `code_retrieval_document` | — | — | — | — | `code.passage` |

!!! note
    OpenAI has no task type concept — the field is simply omitted from the IR when converting from/to OpenAI format.

## Encoding Format Normalization

| IR Canonical | OpenAI | Google | Cohere | Voyage | Jina |
|-------------|--------|--------|--------|--------|------|
| `float` | `float` | (always) | `float` | `float` | `float` |
| `base64` | `base64` | — | `base64` | `base64` | `base64` |
| `int8` | — | — | `int8` | `int8` | — |
| `uint8` | — | — | `uint8` | `uint8` | — |
| `binary` | — | — | `binary` | `binary` | `binary` |
| `ubinary` | — | — | `ubinary` | `ubinary` | `ubinary` |

!!! note
    Voyage separates `output_dtype` (data type of values) from `encoding_format` (serialization format). The IR merges these into a single `encoding_format` axis. Cohere's ability to request multiple embedding types simultaneously is provider-specific and handled via `provider_extensions`.

## IR Mapping

LLM-Rosetta normalizes all five formats into a unified IR:

| Provider Field | IR Field | Notes |
|----------------|----------|-------|
| `input` / `texts` / `content.parts` | `input` | Normalized to `list[str]` |
| `taskType` / `input_type` / `task` | `task_type` | See task type normalization table |
| `dimensions` / `outputDimensionality` / `output_dimension` | `dimensions` | Integer |
| `encoding_format` / `embedding_types` / `output_dtype` / `embedding_type` | `encoding_format` | Single value from the IR enum |
| `truncation` / `truncate` / `autoTruncate` | `truncation` | Simplified to boolean |
| `data[].embedding` / `embedding.values` / `embeddings.{type}[][]` | `data[].embedding` | OpenAI-style list of items |

### Usage Normalization

| Provider | `total_tokens` source | `prompt_tokens` source | Dropped |
|----------|----------------------|----------------------|---------| 
| OpenAI | `usage.total_tokens` | `usage.prompt_tokens` | — |
| Google | `metadata.promptTokenCount` | `metadata.promptTokenCount` | — |
| Cohere | `meta.tokens.input_tokens` (derived) | `meta.tokens.input_tokens` | `meta.billed_units.*` |
| Voyage | `usage.total_tokens` | — | — |
| Jina | `usage.total_tokens` | `usage.prompt_tokens` | `image_tokens`, `audio_tokens`, `video_tokens` |

## Cross-Provider Conversion

With the IR as hub, any format can be converted to any other:

```
OpenAI request ──→ IR ──→ Google request
                     ├──→ Cohere request
                     ├──→ Voyage request
                     └──→ Jina request
```

Information loss boundaries:

- **Any → Google**: `encoding_format` dropped (Google only supports float)
- **Any → OpenAI**: `task_type` dropped (OpenAI has no task type concept)
- **Cohere multi-type → IR**: only one embedding type selected; others require `provider_extensions`
- **Usage**: Cohere's `billed_units` and Jina's per-modality token breakdown are not preserved in IR

### Library Usage

```python
from llm_rosetta.converters.embedding.openai import OpenAIEmbeddingConverter
from llm_rosetta.converters.embedding.jina import JinaEmbeddingConverter

openai_conv = OpenAIEmbeddingConverter()
jina_conv = JinaEmbeddingConverter()

# OpenAI embedding request
openai_request = {
    "model": "text-embedding-3-small",
    "input": ["Hello world", "Goodbye world"],
}

# OpenAI → IR → Jina
ir_request = openai_conv.request_from_provider(openai_request)
jina_request, warnings = jina_conv.request_to_provider(ir_request)
```

## Features Not Yet in IR

The following provider-specific features are out of scope for the current IR and can be passed through via `provider_extensions`:

- **Multimodal inputs**: Cohere images, Jina video/audio/PDF
- **Multi-vector / ColBERT**: Jina `return_multivector` (per-token embeddings)
- **Late chunking**: Jina `late_chunking` (cross-chunk context preservation)
- **Normalization control**: Jina `normalized` (L2-normalize output vectors)
- **Document title**: Google `title` (improves retrieval quality for `RETRIEVAL_DOCUMENT`)
- **Cohere multi-format**: requesting `["float", "int8"]` simultaneously
