---
title: Rerank API 格式
---

# Rerank API 格式

Rerank API 根据查询对一组文档按相关性重排。和聊天补全不同，rerank **没有统一标准**——每家提供方的请求/响应结构各不相同。

LLM-Rosetta 识别出 **3 个格式族**，覆盖主流 rerank 提供方。每种格式有专用 converter，通过 IR 做跨提供方转换。

## 概览

| 格式族 | 结果字段 | 文档文本 | 用量位置 | Converter 类 |
|-------|---------|---------|---------|-------------|
| Jina | `results` | 默认包含 | 顶层 `usage` | `JinaRerankConverter` |
| Cohere | `results` | 不包含 | `meta.tokens` / `meta.billed_units` | `CohereRerankConverter` |
| Voyage | `data` | 可选 | 顶层 `usage` | `VoyageRerankConverter` |

## Provider 族谱

每个格式族被多个 provider 和平台使用：

| 格式族 | 使用者 |
|-------|-------|
| **Jina** | Jina AI, GPUStack, vLLM (`/v1/rerank`), Xinference, llama-box |
| **Cohere** | Cohere, Siliconflow, OpenVINO Model Server, vLLM (`/v2/rerank`) |
| **Voyage** | Voyage AI, MongoDB Atlas (Voyage 后端) |

!!! note
    vLLM 同时暴露 `/v1/rerank`（Jina 兼容）和 `/v2/rerank`（Cohere 兼容），使用同一个模型。

## 请求格式

三个格式族共享几乎相同的请求结构：

```json
{
  "model": "model-name",
  "query": "搜索查询",
  "documents": ["文档 1", "文档 2", "..."],
  "top_n": 3
}
```

格式族之间的关键差异：

| 字段 | Jina | Cohere | Voyage |
|------|------|--------|--------|
| Top-N 结果数 | `top_n` | `top_n` | `top_k` |
| 返回文档文本 | `return_documents`（默认：`true`） | — | `return_documents`（默认：`false`） |
| 截断限制 | — | `max_tokens_per_doc`（默认：4096） | `truncation`（bool，默认：`true`） |

## 响应格式

### Jina 格式

使用者：Jina AI, GPUStack, vLLM, Xinference, llama-box

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

**特点：**

- 结果在 `results` 数组中，按相关性降序排列
- 文档文本作为 **纯字符串** 返回（不是 `{text: "..."}`）
- 用量在顶层，包含 `total_tokens`（部分 provider 还报告 `prompt_tokens`）
- 包含 `object: "list"` 字段

### Cohere 格式

使用者：Cohere（v2 API）, Siliconflow（变体）, OpenVINO Model Server

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

**特点：**

- 结果在 `results` 数组中，默认**不包含文档文本**
- 有唯一的 `id` 字段
- 用量在 `meta.billed_units` 中（面向计费：`search_units`）
- Cohere v4+ 还提供 `meta.tokens`，包含 `input_tokens`, `output_tokens`, `cached_tokens`

#### Siliconflow 变体

Siliconflow 遵循 Cohere 结构，但有更丰富的用量报告：

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

与 Cohere 的主要差异：

- 显式包含 `document: null`（Cohere 完全省略该字段）
- `meta.tokens` 始终有值（Cohere v3 只有 `billed_units`）
- `meta.billed_units` 同时包含 token 计数和 `search_units`

### Voyage 格式

使用者：Voyage AI, MongoDB Atlas

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

**特点：**

- 结果在 **`data`** 数组中（不是 `results`）—— 与 OpenAI Embeddings 响应结构一致
- 当 `return_documents: true` 时，文档文本作为纯字符串返回
- 用量在顶层，只有 `total_tokens`
- 顶层包含 `object: "list"` 和 `model`

## IR 映射

LLM-Rosetta 将三种格式归一化到统一的 IR：

| Provider 字段 | IR 字段 | 说明 |
|--------------|--------|------|
| `results` / `data` | `results` | 统一名称为 `results` |
| `top_n` / `top_k` | `top_n` | 统一名称为 `top_n` |
| `document`（字符串）/ `document.text` | `document.text` | 归一化为 `RerankDocument(text=...)` |
| `usage.total_tokens` / `meta.tokens.input_tokens` | `usage.total_tokens` | 基于 token，计费字段丢弃 |

### 用量归一化

| Provider | `total_tokens` 来源 | `prompt_tokens` 来源 | 丢弃的字段 |
|----------|-------------------|---------------------|-----------|
| Jina | `usage.total_tokens` | `usage.prompt_tokens` | — |
| Voyage | `usage.total_tokens` | — | — |
| Cohere v3 | — | — | `meta.billed_units.search_units` |
| Cohere v4 | `meta.tokens.input_tokens`（推导） | `meta.tokens.input_tokens` | `meta.billed_units.*` |
| Siliconflow | `meta.tokens.input_tokens`（推导） | `meta.tokens.input_tokens` | `meta.billed_units.*` |

## 跨 Provider 转换

以 IR 为枢纽，任意格式之间可以互相转换：

```
Jina 请求 ──→ IR ──→ Cohere 请求
                └──→ Voyage 请求

Cohere 响应 ──→ IR ──→ Jina 响应
                  └──→ Voyage 响应
```

信息损失边界：

- **Jina → Cohere**：文档文本被丢弃（Cohere 响应中不包含文档）
- **Cohere → Jina/Voyage**：无文档文本可供包含
- **用量**：Cohere v3 的 `search_units` 无法转换为 token 数

### 库级用法

```python
from llm_rosetta.converters.rerank.jina import JinaRerankConverter
from llm_rosetta.converters.rerank.cohere import CohereRerankConverter

jina_conv = JinaRerankConverter()
cohere_conv = CohereRerankConverter()

# Jina rerank 请求
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
