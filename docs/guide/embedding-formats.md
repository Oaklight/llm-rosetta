---
title: Embedding API 格式
---

# Embedding API 格式

Embedding API 将文本转换为稠密向量表示，用于语义搜索、聚类和分类。与没有统一标准的 rerank API 不同，embedding API 有一个由 OpenAI 设立的**事实标准** —— 但仍然存在显著的分歧，尤其是 Google GenAI 和 Cohere。

LLM-Rosetta 识别了 **5 个格式族**，覆盖所有主流 embedding provider。每种格式由专用 converter 处理，通过 IR 实现跨 provider 的 embedding 转换。

## 概览

| 格式族 | 输入字段 | 任务类型字段 | 维度字段 | 响应向量 | 编码格式 |
|-------|---------|------------|---------|---------|---------|
| OpenAI | `input` | — | `dimensions` | `data[].embedding` | `float`, `base64` |
| Google GenAI | `content`（Content/Parts） | `taskType` | `outputDimensionality` | `embedding.values` | 仅 `float` |
| Cohere | `texts` | `input_type` | `output_dimension` | `embeddings.{type}[][]` | `float`, `base64`, `int8`, `uint8`, `binary`, `ubinary` |
| Voyage | `input` | `input_type` | `output_dimension` | `data[].embedding` | `float`, `base64`, `int8`, `uint8`, `binary`, `ubinary` |
| Jina | `input` | `task` | `dimensions` | `data[].embedding` | `float`, `base64`, `binary`, `ubinary` |

## Provider 族谱

| 格式族 | 使用者 |
|-------|-------|
| **OpenAI** | OpenAI, Azure OpenAI, DeepSeek, Together AI, Fireworks 及大多数 OpenAI 兼容平台 |
| **Google GenAI** | Google Gemini, Vertex AI |
| **Cohere** | Cohere（v2 API） |
| **Voyage** | Voyage AI |
| **Jina** | Jina AI |

## 请求格式

### OpenAI（事实标准）

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

单文本 — `embedContent`：

```json
{
  "content": {"parts": [{"text": "hello world"}]},
  "taskType": "RETRIEVAL_DOCUMENT",
  "title": "Document Title"
}
```

批量 — `batchEmbedContents`：

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

### 关键请求差异

| 字段 | OpenAI | Google | Cohere | Voyage | Jina |
|------|--------|--------|--------|--------|------|
| 输入 | `input: string[]` | `content: {parts}` | `texts: string[]` | `input: string[]` | `input: string[]` |
| 任务类型 | — | `taskType` | `input_type`（必需） | `input_type` | `task` |
| 维度 | `dimensions` | `outputDimensionality` | `output_dimension` | `output_dimension` | `dimensions` |
| 截断 | — | `autoTruncate`（bool） | `truncate`（枚举） | `truncation`（bool） | `truncate`（bool） |

## 响应格式

### OpenAI 格式

使用者：OpenAI, Azure OpenAI, DeepSeek, Together AI 及大多数 OpenAI 兼容平台

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

### Google GenAI 格式

单文本响应：

```json
{
  "embedding": {"values": [0.123, -0.456, ...]},
  "metadata": {"promptTokenCount": 5}
}
```

批量响应：

```json
{
  "embeddings": [
    {"values": [0.123, -0.456, ...]},
    {"values": [0.789, -0.012, ...]}
  ]
}
```

**特点：**

- 没有 `object`/`data`/`index` 包装 —— 向量直接在 `embedding.values` 中
- 模型名在 URL 路径中，不在响应体中
- 仅支持 float 输出，不支持编码格式控制

### Cohere 格式

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

**特点：**

- 向量在 `embeddings.{type}[][]` 中 —— 按 embedding 类型分组的 **dict-of-lists**
- 可以在一次请求中同时请求多种 embedding 类型（如 `["float", "int8"]`）
- 没有逐向量的 `index` 字段 —— 顺序与输入一致
- 有唯一的 `id` 字段和 `texts` 回显

### Voyage 格式

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

**特点：**

- 与 OpenAI 兼容的响应结构
- 将 `output_dtype`（数据类型）与 `encoding_format`（序列化格式）分离 —— 两个正交维度
- 用量仅报告 `total_tokens`（无 `prompt_tokens`）

### Jina 格式

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

**特点：**

- 与 OpenAI 兼容的响应结构
- 支持多模态输入（图像、视频、音频、PDF）—— 尚未纳入 IR
- 支持多向量（ColBERT）输出（`return_multivector`）—— 尚未纳入 IR
- 有独特的 `late_chunking` 和 `normalized` 选项

## 任务类型归一化

各 provider 对相同语义概念使用不同名称。IR 将它们归一化：

| IR 规范值 | OpenAI | Google | Cohere | Voyage | Jina |
|----------|--------|--------|--------|--------|------|
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
    OpenAI 没有任务类型的概念 —— 从/向 OpenAI 格式转换时，该字段直接省略。

## 编码格式归一化

| IR 规范值 | OpenAI | Google | Cohere | Voyage | Jina |
|----------|--------|--------|--------|--------|------|
| `float` | `float` | （固定） | `float` | `float` | `float` |
| `base64` | `base64` | — | `base64` | `base64` | `base64` |
| `int8` | — | — | `int8` | `int8` | — |
| `uint8` | — | — | `uint8` | `uint8` | — |
| `binary` | — | — | `binary` | `binary` | `binary` |
| `ubinary` | — | — | `ubinary` | `ubinary` | `ubinary` |

!!! note
    Voyage 将 `output_dtype`（值的数据类型）与 `encoding_format`（序列化格式）分离。IR 将这两个维度合并为单一的 `encoding_format` 轴。Cohere 支持在一次请求中同时请求多种 embedding 类型，这属于 provider 特有行为，通过 `provider_extensions` 处理。

## IR 映射

LLM-Rosetta 将五种格式归一化到统一的 IR：

| Provider 字段 | IR 字段 | 说明 |
|--------------|--------|------|
| `input` / `texts` / `content.parts` | `input` | 归一化为 `list[str]` |
| `taskType` / `input_type` / `task` | `task_type` | 见任务类型归一化表 |
| `dimensions` / `outputDimensionality` / `output_dimension` | `dimensions` | 整数 |
| `encoding_format` / `embedding_types` / `output_dtype` / `embedding_type` | `encoding_format` | IR 枚举中的单个值 |
| `truncation` / `truncate` / `autoTruncate` | `truncation` | 简化为布尔值 |
| `data[].embedding` / `embedding.values` / `embeddings.{type}[][]` | `data[].embedding` | OpenAI 风格的列表 |

### 用量归一化

| Provider | `total_tokens` 来源 | `prompt_tokens` 来源 | 丢弃的字段 |
|----------|-------------------|---------------------|-----------|
| OpenAI | `usage.total_tokens` | `usage.prompt_tokens` | — |
| Google | `metadata.promptTokenCount` | `metadata.promptTokenCount` | — |
| Cohere | `meta.tokens.input_tokens`（推导） | `meta.tokens.input_tokens` | `meta.billed_units.*` |
| Voyage | `usage.total_tokens` | — | — |
| Jina | `usage.total_tokens` | `usage.prompt_tokens` | `image_tokens`, `audio_tokens`, `video_tokens` |

## 跨 Provider 转换

以 IR 为枢纽，任意格式之间可以互相转换：

```
OpenAI 请求 ──→ IR ──→ Google 请求
                  ├──→ Cohere 请求
                  ├──→ Voyage 请求
                  └──→ Jina 请求
```

信息损失边界：

- **Any → Google**：`encoding_format` 被丢弃（Google 仅支持 float）
- **Any → OpenAI**：`task_type` 被丢弃（OpenAI 没有任务类型概念）
- **Cohere 多格式 → IR**：仅选择一种 embedding 类型；其他需要 `provider_extensions`
- **用量**：Cohere 的 `billed_units` 和 Jina 的多模态 token 细分不保留在 IR 中

## 尚未纳入 IR 的特性

以下 provider 特有功能不在当前 IR 范围内，可通过 `provider_extensions` 透传：

- **多模态输入**：Cohere 图像, Jina 视频/音频/PDF
- **多向量 / ColBERT**：Jina `return_multivector`（逐 token 嵌入）
- **延迟分块**：Jina `late_chunking`（跨块上下文保留）
- **归一化控制**：Jina `normalized`（L2 归一化输出向量）
- **文档标题**：Google `title`（提升 `RETRIEVAL_DOCUMENT` 检索质量）
- **Cohere 多格式**：同时请求 `["float", "int8"]`
