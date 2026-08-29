---
title: 首页
author: Oaklight
hide:
  - navigation
---

<div style="display: flex; align-items: center; gap: 1.5em; margin-bottom: 0.5em;">
  <img src="images/rosetta_stone.svg" alt="Rosetta Stone" style="width: 96px; flex-shrink: 0;">
  <div>
    <h1 style="margin: 0 0 0.2em 0;">LLM-Rosetta</h1>
    <p style="margin: 0; font-size: 1.1em; opacity: 0.85;">用于 LLM 提供方 API 之间的统一消息格式转换库。</p>
    <p style="margin: 0.4em 0 0 0;">
      <a href="https://pypi.org/project/llm-rosetta/"><img src="https://img.shields.io/pypi/v/llm-rosetta?color=green" alt="PyPI"></a>
      <a href="https://github.com/Oaklight/llm-rosetta/releases/latest"><img src="https://img.shields.io/github/v/release/Oaklight/llm-rosetta?color=green" alt="Release"></a>
      <a href="https://hub.docker.com/r/oaklight/llm-rosetta-gateway"><img src="https://img.shields.io/docker/v/oaklight/llm-rosetta-gateway?label=Docker&color=blue" alt="Docker"></a>
      <a href="https://github.com/Oaklight/llm-rosetta/blob/master/LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="MIT"></a>
      <a href="https://arxiv.org/abs/2604.09360"><img src="https://img.shields.io/badge/arXiv-2604.09360-b31b1b.svg" alt="arXiv"></a>
    </p>
  </div>
</div>

罗塞塔石碑让古代文字得以互译。LLM-Rosetta 做的是类似的事——在不兼容的 LLM API 格式之间充当翻译层，让你用任意格式发请求、对接任意提供方。

---

## 问题

各家 LLM 提供方的 API 格式互不相通——给 OpenAI 写的请求发不了 Anthropic，也发不了 Google。换一家就得改一遍集成代码；要同时支持多家，转换器数量会按 N² 增长。

**LLM-Rosetta** 用一套**中间表示（IR）**解决这个问题：每家提供方只需要实现一个到 IR 的转换器，转换器总数从 N² 降到 2N。

Provider A ↔ **IR** ↔ Provider B — 任何格式进，任何格式出。

---

## 两种使用方式

=== "作为库使用"

    在自己的代码里做格式转换，不需要起服务器：

    ```python
    from llm_rosetta import OpenAIChatConverter, AnthropicConverter

    openai_conv = OpenAIChatConverter()
    anthropic_conv = AnthropicConverter()

    # OpenAI 格式 → IR → Anthropic 格式
    ir_request = openai_conv.request_from_provider(openai_request)
    anthropic_request, warnings = anthropic_conv.request_to_provider(ir_request)
    ```

    ```bash
    pip install llm-rosetta
    ```

    [库快速开始 →](getting-started/library-quickstart.md){ .md-button }

=== "作为网关使用"

    起一个本地 HTTP 代理，请求进来时自动做格式转换：

    ```text
    客户端（Chat Completions）──→ 网关 ──→ Anthropic API
    客户端（Responses API）   ──→ 网关 ──→ Google API
    客户端（Open Responses）  ──→ 网关 ──→ 任意提供方
    客户端（Anthropic 格式）  ──→ 网关 ──→ OpenAI API
    客户端（Google 格式）     ──→ 网关 ──→ 任意提供方
    ```

    ```bash
    pip install "llm-rosetta[gateway]"
    llm-rosetta-gateway
    ```

    可直接作为 **Claude Code**、**Antigravity CLI (agy)**、**OpenAI Codex CLI**、**Kilo Code** 和 **Ollama** 的后端。详见 [CLI 工具集成](gateway/cli-integrations.md)。

    [网关快速开始 →](getting-started/gateway-quickstart.md){ .md-button }

---

## 支持的 API 标准

| 提供方 | API 标准 | ProviderType |
|--------|---------|:------------:|
| OpenAI | Chat Completions | `openai_chat` |
| OpenAI | Responses | `openai_responses` |
| Open Responses | 厂商中立标准 | `open_responses` |
| Anthropic | Messages | `anthropic` |
| Google | GenAI | `google` |

各格式的详细对比见 [API 标准](guide/api-standards.md)，完整的提供方支持矩阵见[提供方兼容性](guide/compatibility.md)。

---

## 核心特性

| | |
|---|---|
| **中枢辐射架构** | 一套 IR，解决 N² 转换问题 |
| **双向转换** | 请求、响应、消息都能双向转 |
| **流式支持** | 有状态地逐 chunk 转换流式数据 |
| **工具调用** | 统一的工具定义和调用，跨提供方通用 |
| **自动检测** | 根据请求结构自动识别来源格式 |
| **网关 + 管理面板** | HTTP 代理 + Web UI，管配置、看指标、查日志 |
| **类型安全** | 全量 TypedDict 注解 |

---

## 使用场景

**多提供方应用** — 应用在多家 LLM 之间自由切换，集成代码不用动。生产跑 OpenAI、测试跑 Claude，或者让用户自己选。

**AI 编程工具代理** — 一个网关同时服务 Claude Code、Antigravity CLI、Codex CLI 等工具，按模型路由到对应上游。

**本地模型访问** — 网关指向 Ollama 或 LM Studio，云 SDK 工具就能直接用本地模型，格式自动转。

**API 迁移** — 换提供方？只需要改路由配置，业务逻辑里的请求/响应处理不用重写。

---

## 文档目录

- **[快速入门](getting-started/installation.md)** — 安装、核心概念和快速开始
- **[指南](guide/converters.md)** — 转换器、IR 类型、提供方、流式处理
- **[网关](gateway/index.md)** — HTTP 代理、配置、CLI 工具集成、管理面板
- **[参考](api/index.md)** — API 文档、示例、贡献指南、更新日志
- **[贡献](contributing/guide.md)** — 如何贡献、风格指南、架构指南

## 引用

如果你在研究中用到了 LLM-Rosetta，欢迎引用：

```bibtex
@article{ding2026llm,
  title={LLM-Rosetta: A Hub-and-Spoke Intermediate Representation for Cross-Provider LLM API Translation},
  author={Ding, Peng},
  journal={arXiv preprint arXiv:2604.09360},
  year={2026}
}
```

## 许可证

MIT 许可证
