---
title: 路线图
---

# 路线图

本页概述了计划中的功能以及欢迎社区贡献的方向。

## 当前状态

LLM-Rosetta v0.2.0 支持 4 种提供商 API 之间的双向转换：

| 提供商 | 格式 | 流式 | 工具调用 |
|-------|------|:----:|:------:|
| OpenAI Chat Completions | `openai_chat` | ✓ | ✓ |
| OpenAI Responses | `openai_responses` | ✓ | ✓ |
| Anthropic Messages | `anthropic` | ✓ | ✓ |
| Google GenAI | `google` | ✓ | ✓ |

[网关](gateway/index.md)提供这些格式之间任意组合的实时 HTTP 代理，已通过 [5 种 CLI 工具和 SDK 测试套件验证](gateway/validation.md)。

---

## 计划中的功能

### Open Responses 集成

!!! tip "状态：计划中"

[Open Responses](https://www.openresponses.org/) 是 OpenAI 于 2026 年 1 月发起的开源规范（Apache 2.0）。它将专有的 OpenAI Responses API 变为**厂商中立的标准**，并具有正式的可扩展性规则。

**重要性**：Open Responses 是 OpenAI Responses API 的**真超集**。已经在与 OpenAI Responses API 通信的客户端只需最小改动即可与 Open Responses 端点通信。主要采用者包括 OpenRouter、Hugging Face、Vercel、LM Studio、Ollama 和 vLLM。

**实现策略**：扩展现有的 `openai_responses` 转换器，而不是构建单独的转换器。差异很小：

| 功能 | 描述 |
|-----|------|
| Reasoning `content` 字段 | 开源模型的原始推理链（除了 `summary` 和 `encrypted_content`） |
| Slug 前缀扩展 | `implementor:type_name` 格式的项目、工具和事件（如 `openai:web_search_call`） |
| `allowed_tools` 字段 | 缓存友好的工具限制 |
| `OpenResponses-Version` 头部 | 规范版本控制机制 |
| 默认无状态 | 已兼容——llm-rosetta 不假设服务端状态 |

可以通过以下方式暴露：

- 标志参数：转换器上的 `output_format="open_responses"`
- 或轻量子类：`OpenResponsesConverter(OpenAIResponsesConverter)`
- 网关：通过 `OpenResponses-Version` 头部检测并相应路由

详细的模式比较请参见完整[分析文档](https://github.com/Oaklight/llm-rosetta/blob/master/analysis/openapi_specs_and_open_responses.md)。

### Ollama 提供商支持

!!! tip "状态：计划中"

[Ollama](https://ollama.com/) 是一款流行的本地 LLM 运行工具。它同时暴露原生 API 和 OpenAI 兼容 API。

**实现方式**：

- **OpenAI 兼容模式**：Ollama 的 `/v1/chat/completions` 端点已与 `openai_chat` 转换器兼容——无需新转换器，只需将网关配置指向 `http://localhost:11434/v1`
- **原生 Ollama API**：专用转换器可以支持 Ollama 特有功能（模型管理、嵌入等），但优先级较低，因为 OpenAI 兼容模式已覆盖主要用例

### LM Studio 提供商支持

!!! tip "状态：计划中 — [#42](https://github.com/Oaklight/llm-rosetta/issues/42)"

[LM Studio](https://lmstudio.ai/) 提供 OpenAI 兼容的本地推理。与 Ollama 类似，通过网关配置使用现有的 `openai_chat` 转换器即可工作。

### HuggingFace Inference API

!!! tip "状态：计划中 — [#40](https://github.com/Oaklight/llm-rosetta/issues/40)"

[HuggingFace Inference API](https://huggingface.co/docs/api-inference/) 支持多种模型格式。专用转换器将支持通过网关路由到 HuggingFace 托管的模型。

### 按提供商 SDK 列出模型

!!! note "状态：开放 — [#54](https://github.com/Oaklight/llm-rosetta/issues/54)"

扩展网关的模型列表端点，查询上游提供商并与本地配置的模型合并。

---

## 社区贡献

我们欢迎针对上述任何计划功能的 Pull Request。开始方式：

1. 查看 [Issue 跟踪器](https://github.com/Oaklight/llm-rosetta/issues) 中的开放问题
2. 阅读[核心概念](guide/concepts.md)指南，了解转换器架构
3. 参考现有转换器（如 `src/llm_rosetta/converters/openai_chat/`）作为模板
4. 提交前运行 `ruff check` 和 `uvx ty check`

对于较大的功能（Open Responses、新提供商），请先开 Issue 讨论方案。
