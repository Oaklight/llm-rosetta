<p align="center">
  <img src="https://raw.githubusercontent.com/Oaklight/llm-rosetta/master/design/logo/out/rosetta-icon-github.svg" alt="LLM-Rosetta" width="96">
</p>

<h1 align="center">llm-rosetta</h1>

<p align="center">
  <a href="https://pypi.org/project/llm-rosetta/"><img src="https://img.shields.io/pypi/v/llm-rosetta?color=green" alt="PyPI"></a>
  <a href="https://pypi.org/project/llm-rosetta/"><img src="https://img.shields.io/pypi/pyversions/llm-rosetta" alt="Python"></a>
  <a href="https://github.com/Oaklight/llm-rosetta/actions/workflows/ci.yml"><img src="https://github.com/Oaklight/llm-rosetta/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
</p>

<p align="center">
  <a href="https://llm-rosetta.readthedocs.io/en/latest/getting-started/library-quickstart/"><img src="https://img.shields.io/badge/docs-English-2f6fdd" alt="Docs (English)"></a>
  <a href="https://llm-rosetta.readthedocs.io/zh-cn/latest/getting-started/library-quickstart/"><img src="https://img.shields.io/badge/docs-%E4%B8%AD%E6%96%87-2f6fdd" alt="Docs (中文)"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-green.svg" alt="MIT"></a>
</p>

<p align="center">
  <a href="https://github.com/Oaklight/llm-rosetta/blob/master/packages/llm-rosetta/README_en.md">English</a> | 中文
</p>

统一的 LLM API 适配器：通过中心化中间表示（IR），在 OpenAI Chat Completions、
Open Responses、Anthropic Messages、Google generateContent 和 Google
Interactions 之间双向转换。

这是**核心发行包**。在 Python 3.11+ 上没有第三方运行时依赖；Python 3.10 上也只
需要 `typing_extensions`，它用于向后移植较新的 typing 功能。它从不导入网关。

## 三种控制粒度与一个工具箱

| | 工具箱 | 原始转换器 | 管线 |
|---|---|---|---|
| **适用场景** | 转换单个 IR 值或工具 | 在自己的循环里转换单条链路 | 一次调用完成 source → IR → target |
| **入口** | `tool_ops`、`transforms`、`capabilities` | `ChatCompletionsConverter()` | `convert()` / `ConversionPipeline` |
| **调用提供方** | 你 | 你 | 你 |

网关发行包在管线之上再加一层——一个自托管的 HTTP 代理，负责调用提供方、路由、
密钥和管理面板。

三种粒度共用同一套 IR 和转换器，工具箱也基于同一套 IR。`tool_ops` 把单个工具
定义、选择、调用或结果在提供方方言与 IR 之间转换；`transforms` 和 `capabilities`
只执行一次字段变换或能力处理，不需要创建转换器。管线会解析两侧的 shim，因此目标
shim 的默认值可能让输出与原始转换器不同——Anthropic 默认的 `cache_control`
断点就是一例。

## 安装

```bash
pip install llm-rosetta
```

需要 **Python ≥ 3.10**。

## 快速开始

原始转换器——只转换单条链路，提供方调用由你完成：

```python
from llm_rosetta import ChatCompletionsConverter, MessagesConverter

openai_request = {
    "model": "gpt-4o",
    "messages": [{"role": "user", "content": "Hello!"}],
}

ir = ChatCompletionsConverter().request_from_provider(openai_request)
anthropic_request, warnings = MessagesConverter().request_to_provider(ir)

# 用你自己的客户端调用 Anthropic API 之后：
ir_response = MessagesConverter().response_from_provider(response)
```

管线——一次调用完成 source → IR → target，并处理 shim 解析：

```python
from llm_rosetta import convert, detect_provider

detect_provider(openai_request)                          # 'chat_completions'
anthropic_request = convert(openai_request, target_provider="messages")
```

流式转换是有状态的；原始转换器用一个 `StreamContext` 跨 chunk 保存状态。详见
[流式转换](https://llm-rosetta.readthedocs.io/zh-cn/latest/guide/streaming/)。

## 用 IR 作为 payload

IR 是类型化的、提供方中立的。你可以把对话历史、路由状态或评测数据保存在 IR 里，
只在调用边界把它渲染成目标提供方的方言。换提供方时改变的是渲染器，不是应用的数据
模型。因为 payload 始终保存在 IR 里，支持另一种 API 格式不会给应用增加格式专用
分支。

## 支持的格式

| 模型类型 | 格式 |
|----------|------|
| LLM（chat） | `chat_completions`、`open_responses`、`messages`、`google_generate`、`google_interactions` |
| Embeddings | `openai`、`cohere`、`jina`、`voyage`、`google_generate`、`minimax` |
| Rerank | `jina`、`cohere`、`voyage` |
| Decisions | `systemone`、`openai_decisions` |

每种格式都能与 IR 互转（请求和响应）；chat 格式还支持流式 chunk 转换。
`open_responses` 是提供方中立的 Open Responses 规范；OpenAI profile
（`openai--open_responses` shim）在此基础上增加 `resp_` ID、`store: true` 和
回显字段。

提供方差异由 39 个提供方 shim 声明，覆盖 19 个提供方前缀，转换器因此保持通用。

## 文档

- [完整 README](https://github.com/Oaklight/llm-rosetta/blob/master/README_zh.md)
- [文档](https://llm-rosetta.readthedocs.io/zh-cn/latest/)
- [网关发行包](https://pypi.org/project/llm-rosetta-gateway/)——增加 HTTP 代理、
  路由、密钥和管理面板

## 许可证

本项目使用 MIT 许可证——详见 [LICENSE](https://github.com/Oaklight/llm-rosetta/blob/master/LICENSE)。
