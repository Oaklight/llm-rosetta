---
title: 路线图
---

# 路线图

本页概述了当前功能状态以及欢迎社区贡献的方向。

## 当前状态

LLM-Rosetta v0.9.0 支持 3 个 API 族的双向转换：

**Chat / Completions**（5 种标准）：

| 提供商 | 格式 | 流式 | 工具调用 |
|-------|------|:----:|:------:|
| OpenAI Chat Completions | `openai_chat` | ✓ | ✓ |
| OpenAI Responses | `openai_responses` | ✓ | ✓ |
| Open Responses | `open_responses` | ✓ | ✓ |
| Anthropic Messages | `anthropic` | ✓ | ✓ |
| Google GenAI | `google` | ✓ | ✓ |

**Embedding**（4 种格式）：OpenAI、Jina、Voyage、Cohere——支持基于 IR 的跨格式转换。

**Rerank**（3 种格式）：Jina、Cohere、Voyage——支持基于 IR 的跨格式转换。

[网关](gateway/index.md)提供**零运行时依赖**的实时 HTTP 代理，已通过 [5 种 CLI 工具和 SDK 测试套件验证](gateway/validation.md)。网关内置[管理面板](gateway/admin-panel.md)及完整的 [REST API](api/admin.md)。

**提供商 shim 层**支持通过声明式 YAML 文件添加新提供商——OpenAI 兼容提供商无需编写转换器代码。内置支持 16 个提供商。

详见 [API 标准](guide/api-standards.md)了解各格式详情。

---

!!! info "已完成的功能"
    已发布的功能详见[更新日志](changelog.md)。关键里程碑包括：声明式 shim 系统（v0.6.0）、零依赖网关（v0.6.0）、Embedding/Rerank IR 转换（v0.6.1+）、推理字段标准化（v0.8.1）、上游超时（v0.8.2）、多 API 模式提供商（v0.6.8）。

---

## 计划中的功能

### 转换器增强

#### 服务端工具类型映射

!!! tip "状态：计划中 — [#181](https://github.com/Oaklight/llm-rosetta/issues/181)"

跨提供商映射服务端工具类型（`web_search`、`code_execution`、`computer_use`），这些工具类型在部分提供商中存在但在其他提供商中不存在。

### Shim 系统

#### 每模型转换（ModelShim）

!!! tip "状态：计划中 — [#192](https://github.com/Oaklight/llm-rosetta/issues/192)"

恢复 `ModelShim` 以支持每模型的转换规则——同一提供商的不同模型可能需要不同的字段处理。

### 网关

#### 速率限制中间件

!!! tip "状态：计划中 — [#124](https://github.com/Oaklight/llm-rosetta/issues/124)"

基于令牌桶或滑动窗口的速率限制，按 API Key 或客户端 IP 限流。

#### 增强错误响应

!!! tip "状态：计划中 — [#123](https://github.com/Oaklight/llm-rosetta/issues/123)"

在网关错误响应中包含上游错误上下文，方便调试。

#### 每提供商费用追踪

!!! note "状态：开放 — [#131](https://github.com/Oaklight/llm-rosetta/issues/131)"

按提供商追踪 token 使用费用，在管理面板仪表盘中展示。

#### 故障转移与负载均衡

!!! note "状态：开放 — [#129](https://github.com/Oaklight/llm-rosetta/issues/129)"

主提供商不可用时自动故障转移到备用提供商，可选在多个提供商间负载均衡。

### 提供商支持

#### LM Studio

!!! warning "状态：推迟 — [#42](https://github.com/Oaklight/llm-rosetta/issues/42)"

[LM Studio](https://lmstudio.ai/) 提供 OpenAI 兼容的本地推理。通过网关配置使用现有的 `openai_chat` 转换器即可工作。因已可通过现有方式使用，优先级较低。

#### HuggingFace Inference API

!!! warning "状态：推迟 — [#40](https://github.com/Oaklight/llm-rosetta/issues/40)"

[HuggingFace Inference API](https://huggingface.co/docs/api-inference/) 支持多种模型格式。专用转换器或 shim 将支持通过网关路由到 HuggingFace 托管的模型。待社区需求确定后推进。

---

## 社区贡献

我们欢迎针对上述任何计划功能的 Pull Request。开始方式：

1. 查看 [Issue 跟踪器](https://github.com/Oaklight/llm-rosetta/issues) 中的开放问题
2. 阅读[核心概念](guide/concepts.md)指南，了解转换器架构
3. 参考现有转换器（如 `src/llm_rosetta/converters/openai_chat/`）作为模板
4. 对于新提供商，优先考虑创建 [shim](guide/shims.md)——通常就够了
5. 提交前运行 `pre-commit run --all-files`

对于较大的功能，请先开 Issue 讨论方案。
