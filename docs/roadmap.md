---
title: 路线图
---

# 路线图

本页概述了当前功能状态以及欢迎社区贡献的方向。

## 当前状态

LLM-Rosetta v0.5.0 支持 5 种 API 标准之间的双向转换：

| 提供商 | 格式 | 流式 | 工具调用 |
|-------|------|:----:|:------:|
| OpenAI Chat Completions | `openai_chat` | ✓ | ✓ |
| OpenAI Responses | `openai_responses` | ✓ | ✓ |
| Open Responses | `open_responses` | ✓ | ✓ |
| Anthropic Messages | `anthropic` | ✓ | ✓ |
| Google GenAI | `google` | ✓ | ✓ |

[网关](gateway/index.md)提供这些格式之间任意组合的实时 HTTP 代理，已通过 [5 种 CLI 工具和 SDK 测试套件验证](gateway/validation.md)。网关还内置了[管理面板](gateway/admin-panel.md)，支持配置管理、指标监控和请求日志查看。

详见 [API 标准](guide/api-standards.md)了解各格式详情。

---

## 近期完成

### Open Responses 集成

!!! success "状态：已完成（v0.5.0）"

[Open Responses](https://www.openresponses.org/) 是 OpenAI 于 2026 年 1 月发起的开源规范（Apache 2.0）。它将专有的 OpenAI Responses API 变为**厂商中立的标准**，并具有正式的可扩展性规则。

**已实现内容**：

- `open_responses` 提供商类型 — 复用 `OpenAIResponsesConverter`，因为线上格式兼容
- 网关在存在 `OpenResponses-Version` 头部时自动转发到上游
- 自动检测可识别 Open Responses 请求体
- 可在网关配置中作为独立的提供商类型配置

主要采用者包括 OpenRouter、Hugging Face、Vercel、LM Studio、Ollama 和 vLLM。

详细的模式比较请参见完整[分析文档](https://github.com/Oaklight/llm-rosetta/blob/master/analysis/openapi_specs_and_open_responses.md)。

### Ollama 支持

!!! success "状态：已完成（v0.5.0）"

[Ollama](https://ollama.com/)（v0.13+）通过两种方式与网关配合使用：

- **作为上游提供商**：将网关提供商指向 `http://localhost:11434/v1`，使用 `openai_chat` 类型 — 无需新转换器
- **作为客户端**：Ollama 的 OpenAI 兼容端点（`/v1/chat/completions`、`/v1/responses`、`/v1/messages`）可以通过网关访问云提供商

配置示例详见 [CLI 工具集成 — Ollama](gateway/cli-integrations.md#ollama)。

---

## 计划中的功能

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

对于较大的功能（新提供商），请先开 Issue 讨论方案。
