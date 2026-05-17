---
title: 路线图
---

# 路线图

本页概述了当前功能状态以及欢迎社区贡献的方向。

## 当前状态

LLM-Rosetta v0.6.2 支持 5 种 API 标准之间的双向转换：

| 提供商 | 格式 | 流式 | 工具调用 | Embeddings |
|-------|------|:----:|:------:|:----------:|
| OpenAI Chat Completions | `openai_chat` | ✓ | ✓ | ✓ |
| OpenAI Responses | `openai_responses` | ✓ | ✓ | — |
| Open Responses | `open_responses` | ✓ | ✓ | — |
| Anthropic Messages | `anthropic` | ✓ | ✓ | — |
| Google GenAI | `google` | ✓ | ✓ | — |

[网关](gateway/index.md)提供**零运行时依赖**的实时 HTTP 代理，已通过 [5 种 CLI 工具和 SDK 测试套件验证](gateway/validation.md)。网关内置[管理面板](gateway/admin-panel.md)及完整的 [REST API](api/admin.md)。

**提供商 shim 层**支持通过声明式 YAML 文件添加新提供商——OpenAI 兼容提供商无需编写转换器代码。内置支持 14 个提供商。

详见 [API 标准](guide/api-standards.md)了解各格式详情。

---

## 近期完成

### 声明式提供商 Shim 系统

!!! success "状态：已完成（v0.6.0）"

提供商现在通过 `shims/providers/<name>/` 下的 `provider.yaml` + 可选 `transforms.py` 文件定义，导入时自动发现。三个可组合的转换原语——`strip_fields()`、`rename_field()`、`set_defaults()`——处理提供商 API 方言与其基础标准之间的字段差异。

新增 7 个内置 shim：xAI (Grok)、Qwen (DashScope)、Moonshot (Kimi)、MiniMax、Zhipu (GLM)、OpenRouter、Volcengine。网关代理管道在请求和响应路径上均应用 shim 转换。

### 零依赖网关

!!! success "状态：已完成（v0.6.0）"

使用 vendored 的零依赖 `httpserver` 和 `httpclient` 模块替换了 Starlette + uvicorn + httpx。`[gateway]` extra 现在零外部运行时依赖。

### Embeddings 透传

!!! success "状态：已完成（v0.6.1）"

`/v1/embeddings` 透传端点直接将嵌入请求代理到上游提供商，无需 IR 转换。`/v1/models` 响应现在包含 `api_standard` 和每模型 `capabilities` 字段。

### 管理面板增强

!!! success "状态：已完成（v0.6.1）"

- **从提供商获取**：查询上游 `/v1/models`，浏览并批量添加模型
- **模型能力**：`embedding` 和 `reasoning` 能力类型及专用测试模式
- **提供商 Logo**：shim 可声明 SVG logo 显示在管理卡片上
- **Admin API**：用于编程配置管理的完整 REST API

### SOCKS5 代理支持

!!! success "状态：已完成（v0.6.0）"

通过 vendored httpclient v0.4.0 支持完整的 SOCKS5 代理（RFC 1928/1929），包括用户名/密码认证。

---

## 计划中的功能

### 转换器增强

#### 服务端工具类型映射

!!! tip "状态：计划中 — [#181](https://github.com/Oaklight/llm-rosetta/issues/181)"

跨提供商映射服务端工具类型（`web_search`、`code_execution`、`computer_use`），这些工具类型在部分提供商中存在但在其他提供商中不存在。

#### Responses API 自定义工具类型

!!! tip "状态：计划中 — [#182](https://github.com/Oaklight/llm-rosetta/issues/182)"

在 IR 中处理 OpenAI Responses 的 `custom` 工具类型，支持提供商特定工具扩展的透传。

#### 推理字段标准化

!!! tip "状态：计划中 — [#185](https://github.com/Oaklight/llm-rosetta/issues/185)"

通过 shim 转换（而非逐提供商的转换器代码）标准化 OpenAI Chat 兼容提供商（如 DeepSeek、Qwen）的 `reasoning_content` / 思考字段。

### Shim 系统

#### 每模型转换（ModelShim）

!!! tip "状态：计划中 — [#192](https://github.com/Oaklight/llm-rosetta/issues/192)"

恢复 `ModelShim` 以支持每模型的转换规则——同一提供商的不同模型可能需要不同的字段处理。

#### 多 API 模式提供商

!!! note "状态：开放 — [#186](https://github.com/Oaklight/llm-rosetta/issues/186)"

支持同时暴露多种 API 标准的提供商（如同时提供 Chat Completions 和 Responses 端点的提供商）。

### 网关

#### 上游超时与熔断器

!!! tip "状态：计划中 — [#121](https://github.com/Oaklight/llm-rosetta/issues/121)"

可配置的每提供商超时和熔断器模式，优雅处理慢速或故障的上游。

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

!!! note "状态：开放 — [#42](https://github.com/Oaklight/llm-rosetta/issues/42)"

[LM Studio](https://lmstudio.ai/) 提供 OpenAI 兼容的本地推理。通过网关配置使用现有的 `openai_chat` 转换器即可工作。

#### HuggingFace Inference API

!!! note "状态：开放 — [#40](https://github.com/Oaklight/llm-rosetta/issues/40)"

[HuggingFace Inference API](https://huggingface.co/docs/api-inference/) 支持多种模型格式。专用转换器或 shim 将支持通过网关路由到 HuggingFace 托管的模型。

---

## 社区贡献

我们欢迎针对上述任何计划功能的 Pull Request。开始方式：

1. 查看 [Issue 跟踪器](https://github.com/Oaklight/llm-rosetta/issues) 中的开放问题
2. 阅读[核心概念](guide/concepts.md)指南，了解转换器架构
3. 参考现有转换器（如 `src/llm_rosetta/converters/openai_chat/`）作为模板
4. 对于新提供商，优先考虑创建 [shim](guide/shims.md)——通常就够了
5. 提交前运行 `pre-commit run --all-files`

对于较大的功能，请先开 Issue 讨论方案。
