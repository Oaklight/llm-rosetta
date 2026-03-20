---
title: 网关验证报告
---

# 网关验证报告

本页记录了 LLM-Rosetta 网关与真实 CLI 工具和 SDK 测试套件的端到端验证结果，作为跨提供商兼容性的证明。

!!! info "最后更新：2026-03-20"
    使用 llm-rosetta v0.2.0（+ [#56](https://github.com/Oaklight/llm-rosetta/issues/56)–[#62](https://github.com/Oaklight/llm-rosetta/issues/62) 的未发布修复）

## CLI 工具兼容性

五款主流 AI 编程 CLI 工具通过网关进行了测试。每个工具使用不同的 API 格式——网关根据端点路径自动翻译。

| CLI 工具 | API 格式 | 源 → 目标 | 对话 | 流式 | 工具调用 | 多轮 |
|---------|---------|----------|:----:|:----:|:------:|:----:|
| [Codex CLI](https://github.com/openai/codex) | OpenAI Responses | `openai_responses` → `openai_responses` | ✓ | ✓ | ✓ | ✓ |
| [Kilo Code](https://kilocode.ai/) | OpenAI Chat | `openai_chat` → `openai_responses` | ✓ | ✓ | ✓ | ✓ |
| [OpenCode](https://opencode.ai/) | OpenAI Chat | `openai_chat` → `openai_responses` | ✓ | ✓ | ✓ | ✓ |
| [Claude Code](https://docs.anthropic.com/en/docs/claude-code) | Anthropic Messages | `anthropic` → `anthropic` | ✓ | ✓ | ✓ | ✓ |
| [Gemini CLI](https://github.com/google-gemini/gemini-cli) | Google GenAI | `google` → `google` | ✓ | ✓ | ✓ | ✓ |

### 测试详情

#### Codex CLI（OpenAI Responses API）

Codex 使用 OpenAI Responses API（`wire_api = "responses"`）。通过网关路由到上游 OpenAI，验证了多轮工具调用。

- **模型**：`gpt-4.1-nano`
- **测试**：提示列出目录内容 → 模型发起 `shell` 工具调用 → 工具结果返回 → 模型总结输出
- **结果**：2 轮成功完成，SSE 流式事件格式正确

#### Kilo Code（OpenAI Chat Completions）

Kilo 使用 OpenAI Chat Completions API。网关将 `openai_chat` 格式转换为 `openai_responses` 格式发送到上游。

- **模型**：`gpt-4.1-nano`（通过 `rosetta/gpt-4.1-nano`）
- **测试**：`kilo run --auto -m rosetta/gpt-4.1-nano "List the files in the current directory"`
- **结果**：第 1 轮（26 chunks，3.80s）→ 工具调用 `ls -la` → 第 2 轮（23 chunks，6.48s）→ 文本总结

#### OpenCode（OpenAI Chat Completions）

OpenCode 同样使用 OpenAI Chat Completions API，基于 Vercel AI SDK。

- **模型**：`gpt-4.1-nano`（通过 `rosetta/gpt-4.1-nano`）
- **测试**：`opencode run --model rosetta/gpt-4.1-nano "List the files in the current directory"`
- **结果**：第 1 轮（21 chunks，3.04s）→ `bash` 工具调用 → 第 2 轮（103 chunks，6.12s）→ 文件列表

#### Claude Code（Anthropic Messages API）

Claude Code 使用 Anthropic Messages API。通过 OpenRouter 作为上游 Anthropic 提供商进行测试。

- **模型**：`anthropic/claude-haiku-4.5`
- **结果**：对话、流式、工具调用和多轮均正常工作。需要有效的上游 API 密钥。

#### Gemini CLI（Google GenAI API）

Gemini CLI 使用 `@google/genai` JS SDK（v1.30.0）。修复 camelCase 工具定义解析（#61）和流式工具调用块格式（#62）后实现完全兼容。

- **模型**：`gemini-2.5-flash-lite`
- **配置**：`GOOGLE_GEMINI_BASE_URL=http://localhost:8765 GEMINI_API_KEY=dummy`
- **结果**：对话、流式和工具调用均正常工作。无头模式（`-p`）下工具调用正常退出；交互模式显示完整的工具调用往返。

---

## 集成测试套件（`tests/integration/`）

集成测试套件使用真实 API 调用验证所有四个转换器管道，包括官方 SDK 和直接 REST 调用。每个测试涵盖非流式、流式、工具调用、往返转换和多轮对话。

### 结果汇总

| 测试套件 | 测试数 | 结果 |
|---------|:-----:|:----:|
| Google GenAI SDK | 5 | **5/5** ✓ |
| Google GenAI REST | 5 | **5/5** ✓ |
| OpenAI Chat SDK | 7 | **7/7** ✓ |
| OpenAI Chat REST | 7 | **7/7** ✓ |
| OpenAI Responses SDK | 4 | **4/4** ✓ |
| OpenAI Responses REST | 4 | **4/4** ✓ |
| Anthropic SDK | 6 | **6/6** ✓ |
| Anthropic REST | 6 | **6/6** ✓ |
| **合计** | **44** | **44/44** ✓ |

### 各套件测试覆盖

| 测试 | OpenAI Chat | OpenAI Responses | Anthropic | Google GenAI |
|-----|:-----------:|:----------------:|:---------:|:------------:|
| 非流式基础文本 | ✓ | ✓ | ✓ | ✓ |
| 非流式图像 | ✓ | — | — | — |
| 非流式工具调用 | ✓ | ✓ | ✓ | ✓ |
| 流式文本 | ✓ | — | ✓ | — |
| 流式工具调用 | ✓ | — | ✓ | — |
| 请求往返 | ✓ | ✓ | ✓ | ✓ |
| 响应往返 | ✓ | ✓ | ✓ | ✓ |
| 多轮对话 | — | — | — | ✓ |

---

## SDK 测试套件（`llm_api_simple_tests`）

[llm_api_simple_tests](https://github.com/Oaklight/llm_api_simple_tests) 套件使用官方 SDK 为每个提供商运行 5 个标准化测试。所有 Anthropic 测试通过网关进行了跨提供商路由。

### 通过网关的 Anthropic SDK

**配置**：`ANTHROPIC_BASE_URL=http://localhost:8765`，模型 `anthropic/claude-3-haiku`

| 测试 | 描述 | 状态 |
|-----|------|:----:|
| `simple_query.py` | 单轮流式查询 | ✓ 通过 |
| `multi_round_chat.py` | 3 轮对话（斐波那契解释 → 代码 → 优化） | ✓ 通过 |
| `multi_round_function_calling.py` | 3 轮工具调用（天气 → 温度转换 → 对比） | ✓ 通过 |
| `multi_round_comprehensive.py` | 3 轮图像+工具调用（地标 → 天气 → 推荐） | ✓ 通过 |
| `multi_round_image.py` | 3 轮视觉对话（描述 → 定位 → 知识） | ✓ 通过 |

### 通过网关的 Google GenAI（curl）

通过 `curl` 直接对网关的 Google 端点进行了多轮工具调用测试。

| 轮次 | 请求 | 模型响应 | 状态 |
|:---:|------|--------|:----:|
| 1 | "What is 127 * 389?"，`calculator` 工具，`mode=ANY` | `functionCall: calculator({expression: "127 * 389"})` | ✓ |
| 2 | 工具结果 `49403`，"add 100 to that" | `functionCall: calculator({expression: "49403 + 100"})` | ✓ |
| 3 | 工具结果 `49503`，`mode=AUTO` | 文本："The result is 49503." | ✓ |

使用 `gemini-2.5-flash-lite` 和 `gemini-3.1-flash-lite-preview` 均测试通过。两个模型都正确返回了带有 `thoughtSignature` 的函数调用。

---

## 已测试的跨提供商路由组合

| 源格式 | 目标提供商 | 已验证 |
|-------|----------|:-----:|
| OpenAI Chat → | OpenAI Responses | ✓ |
| OpenAI Responses → | OpenAI Responses | ✓ |
| Anthropic → | Anthropic (OpenRouter) | ✓ |
| Google GenAI → | Google GenAI | ✓ |

---

## 验证期间发现并修复的 Bug

| Issue | 描述 | 修复 |
|-------|------|------|
| [#56](https://github.com/Oaklight/llm-rosetta/issues/56) | OpenAI Responses 流式：缺少 `id`/`object`/`model` 字段，事件顺序错误 | 在转换器中修复 |
| [#57](https://github.com/Oaklight/llm-rosetta/issues/57) | OpenAI Chat 流式：`tool_calls` 缺少 `index` 字段 | 在转换器中修复 |
| [#58](https://github.com/Oaklight/llm-rosetta/issues/58) | `stream_options`（Chat 专用字段）泄漏到 Responses API 请求中 | 从 Responses `ir_stream_config_to_p()` 中移除 |
| [#59](https://github.com/Oaklight/llm-rosetta/issues/59) | Google 转换器忽略 REST 格式请求中的 tools | 添加顶层字段回退 |
| [#61](https://github.com/Oaklight/llm-rosetta/issues/61) | Google camelCase `functionDeclarations` 未解析；仅提取第一个声明 | 支持双大小写；提取所有声明 |
| [#62](https://github.com/Oaklight/llm-rosetta/issues/62) | Google 流式工具调用被拆分为两个 chunk（仅名称 + 仅参数） | 延迟 `tool_call_start`，在 `tool_call_delta` 时发送完整的 `function_call` |

---

## 已知限制

- **Claude Code**：需要为配置的 Anthropic 提供商提供有效的上游 API 密钥。网关本身是透明的——认证错误从上游透传。
- **图像透传**：未测试跨提供商的图像路由（例如 OpenAI Chat → Google GenAI 带图像）。同提供商的图像支持正常（Anthropic SDK 测试确认视觉功能通过网关工作）。
- **Gemini CLI 无头模式**：在非交互式（`-p`）模式下，工具调用结果可能不会被 CLI 显示，但往返已成功完成。这是 Gemini CLI 的显示行为，非网关问题。
