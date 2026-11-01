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
    <p style="margin: 0; font-size: 1.1em; opacity: 0.85;">面向 LLM 提供方 API 的自托管、格式中立翻译层。</p>
    <p style="margin: 0.4em 0 0 0;">
      <a href="https://pypi.org/project/llm-rosetta/"><img src="https://img.shields.io/pypi/v/llm-rosetta?color=green" alt="PyPI"></a>
      <a href="https://test.pypi.org/project/llm-rosetta/"><img src="https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Ftest.pypi.org%2Fpypi%2Fllm-rosetta%2Fjson&query=%24.info.version&label=TestPyPI&color=blue" alt="TestPyPI"></a>
      <a href="https://github.com/Oaklight/llm-rosetta/releases/latest"><img src="https://img.shields.io/github/v/release/Oaklight/llm-rosetta?color=green" alt="Release"></a>
      <a href="https://hub.docker.com/r/oaklight/llm-rosetta-gateway"><img src="https://img.shields.io/docker/v/oaklight/llm-rosetta-gateway?label=Docker&color=blue" alt="Docker"></a>
      <a href="https://github.com/Oaklight/llm-rosetta/blob/master/LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="MIT"></a>
      <a href="https://arxiv.org/abs/2604.09360"><img src="https://img.shields.io/badge/arXiv-2604.09360-b31b1b.svg" alt="arXiv"></a>
    </p>
  </div>
</div>

罗塞塔石碑让古代文字得以互译。LLM-Rosetta 做的是类似的事——让任意客户端用它自己的格式说话，抵达任意模型来源，同时把每个上游的控制权留在你手里。

---

## 问题

各家 LLM 提供方的 API 格式互不相通——给 OpenAI 写的请求发不了 Anthropic，也发不了 Google。换一家就得改一遍集成代码；两两支持 N 种格式需要 O(N²) 个适配器。

**LLM-Rosetta** 用一套**中间表示（IR）**解决这个问题：每种格式只需要一个与 IR 互转的双向转换器，总数降到 2N。

提供方 A ↔ **IR** ↔ 提供方 B——任意受支持的格式进，任意受支持的格式出。

---

<a id="control-levels"></a>

## 三种控制粒度与一个工具箱

| | 工具箱 | 原始转换器 | 管线 | 网关 |
|---|---|---|---|---|
| **适用场景** | 转换单个 IR 值或工具 | 在自己的循环里转换单条链路 | 一次调用完成 source → IR → target | 面向任意客户端的 HTTP 代理 |
| **入口** | `tool_ops`、`transforms`、`capabilities` | `ChatCompletionsConverter()` | `convert()` / `ConversionPipeline` | `llm-rosetta-gateway` |
| **调用提供方** | 你 | 你 | 你 | 网关 |
| **路由、密钥、管理面板** | — | — | — | 内置 |

三种粒度共用同一套 IR 和转换器，工具箱也基于同一套 IR。从需要的层级开始，向上迁移时不用重写翻译逻辑。
`tool_ops` 把单个工具定义、选择、调用或结果在提供方方言与 IR 之间转换；
`transforms` 和 `capabilities` 只执行一次字段变换或能力处理，不需要创建转换器。

Python 3.11+ 没有第三方运行时依赖；3.10 上也只需要 `typing_extensions`，即标准库
`typing` 模块的向后移植包（backport）。对多提供方网关来说这并不常见，也意味着同一条
管线可以在不同应用和部署之间搬动，不需要解析依赖图。

---

## 快速开始

=== "作为库使用"

    用原始转换器转换单条链路，或用管线一次走完 source → IR → target：

    ```python
    from llm_rosetta import ChatCompletionsConverter, MessagesConverter, convert

    openai_request = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "Hello!"}],
    }

    # 原始转换器：一次转换一条链路
    ir = ChatCompletionsConverter().request_from_provider(openai_request)
    anthropic_request, warnings = MessagesConverter().request_to_provider(ir)

    # 管线：一次调用完成 source → IR → target
    anthropic_request = convert(openai_request, target_provider="messages")
    ```

    ```bash
    pip install llm-rosetta
    ```

    [库快速开始 →](getting-started/library-quickstart.md){ .md-button }

=== "作为网关使用"

    起一个本地 HTTP 代理，在途中实时转换格式，并把每个上游、凭据和路由都留在
    你自己的部署里：

    ```mermaid
    flowchart LR
        C1[Chat Completions] --> GW((Gateway))
        C2[Responses API] --> GW
        C3[Open Responses] --> GW
        C4[Anthropic format] --> GW
        C5[Google format] --> GW
        GW --> P1[Anthropic API]
        GW --> P2[Google API]
        GW --> P3[OpenAI API]
        GW --> P4[Any provider]
    ```

    ```bash
    pip install llm-rosetta-gateway
    llm-rosetta-gateway init
    llm-rosetta-gateway
    ```

    可作为 **Claude Code**、**Antigravity CLI (agy)**、**OpenAI Codex CLI**、
    **Kilo Code** 和 **OpenCode** 的后端，也可以作为 **Ollama** 前面的转换代理。
    详见 [CLI 工具集成](gateway/cli-integrations.md)。

    [网关快速开始 →](getting-started/gateway-quickstart.md){ .md-button }

---

## 支持的格式

| 模型类型 | 格式 |
|----------|------|
| LLM（chat） | `chat_completions`、`open_responses`、`messages`、`google_generate`、`google_interactions` |
| Embeddings | `openai`、`cohere`、`jina`、`voyage`、`google_generate`、`minimax` |
| Rerank | `jina`、`cohere`、`voyage` |
| Decisions | `systemone`、`openai_decisions` |

每种格式都能与 IR 互转（请求和响应）；chat 格式还支持流式 chunk 转换。各格式的
详细对比见 [API 标准](guide/api-standards.md)，完整的提供方支持矩阵见
[提供方兼容性](guide/compatibility.md)。

---

## 核心特性

| | |
|---|---|
| **三种控制粒度 + 工具箱** | 工具箱、原始转换器、管线、网关，底层共用一套 IR |
| **双向转换** | 请求、响应、流式 chunk 都能双向转 |
| **IR 作为应用 payload** | 用提供方中立的形式保存历史和编排状态 |
| **Provider shim** | 39 个 shim 覆盖 19 个提供方前缀，声明字段、鉴权和 URL 差异 |
| **网关 + 管理面板** | HTTP 代理，带路由、密钥、日志、指标和管理界面 |
| **零第三方依赖** | Python 3.11+ 无依赖；3.10 仅需 `typing_extensions` |
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
