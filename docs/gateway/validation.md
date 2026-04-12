---
title: 网关验证报告
---

# 网关验证报告

本页汇总了 LLM-Rosetta 网关（通过 [argo-proxy](https://github.com/Oaklight/argo-proxy)）与真实 CLI 工具、SDK 测试套件和跨格式路由的端到端验证结果。

!!! info "最后更新：2026-04-12"
    使用 llm-rosetta v0.5.0 和 argo-proxy v3.0.0b7

## 跨格式路由矩阵

argo-proxy 将 Claude 模型路由到原生 Anthropic 上游，其他模型（GPT、Gemini）路由到 OpenAI Chat 上游。网关自动在不同认证凭证格式之间转换（`Authorization: Bearer`、`x-api-key`、`x-goog-api-key`）。

### 文本生成（9/9 ✓）

| 客户端（API 格式） | Claude 模型 | GPT 模型 | Gemini 模型 |
|-------------------|:----------:|:--------:|:----------:|
| **Claude Code**（Anthropic） | ✓ 透传 | ✓ anthropic→openai_chat | ✓ anthropic→openai_chat |
| **Codex CLI**（OpenAI Responses） | ✓ responses→anthropic | ✓ 透传 | ✓ 透传 |
| **Gemini CLI**（Google GenAI） | ✓ google→anthropic | ✓ google→openai_chat | ✓ google→openai_chat |

### 图像理解（9/9 ✓）

| 客户端（图像方式） | Claude 模型 | GPT 模型 | Gemini 模型 |
|-------------------|:----------:|:--------:|:----------:|
| **Codex CLI**（`-i` 参数） | ✓ | ✓ | ✓ |
| **Claude Code**（Read 工具） | ✓ | ✓¹ | ✓ |
| **Gemini CLI**（read_file 工具） | ✓ | ✓ | ✓ |

¹ 需要 GPT-5.4+；GPT-4.1-nano 可能无法正确解读 Read 工具返回的图像结果。

可复现命令和详细测试步骤参见 [CLI 跨格式测试](validation-cli.md)。

---

## 集成测试汇总（22/22 ✓）

| 测试套件 | 测试数 | 结果 |
|---------|:-----:|:----:|
| Google GenAI SDK | 5 | **5/5** ✓ |
| Google GenAI REST | 6 | **6/6** ✓ |
| OpenAI Chat SDK | 5 | **5/5** ✓ |
| OpenAI Responses SDK | 3 | **3/3** ✓ |
| Anthropic REST | 3 | **3/3** ✓ |
| **合计** | **22** | **22/22** ✓ |

SDK 测试详情和 curl 验证参见 [SDK 与集成测试](validation-sdk.md)。

---

## 验证期间发现并修复的 Bug

| Issue | 描述 | 修复 |
|-------|------|------|
| [#56](https://github.com/Oaklight/llm-rosetta/issues/56) | OpenAI Responses 流式：缺少 `id`/`object`/`model` 字段 | 在转换器中修复 |
| [#57](https://github.com/Oaklight/llm-rosetta/issues/57) | OpenAI Chat 流式：`tool_calls` 缺少 `index` 字段 | 在转换器中修复 |
| [#58](https://github.com/Oaklight/llm-rosetta/issues/58) | `stream_options` 泄漏到 Responses API 请求中 | 从 `ir_stream_config_to_p()` 中移除 |
| [#59](https://github.com/Oaklight/llm-rosetta/issues/59) | Google 转换器忽略 REST 格式请求中的 tools | 添加顶层字段回退 |
| [#61](https://github.com/Oaklight/llm-rosetta/issues/61) | Google camelCase `functionDeclarations` 未解析 | 支持双大小写；提取所有声明 |
| [#62](https://github.com/Oaklight/llm-rosetta/issues/62) | Google 流式工具调用被拆分为两个 chunk | 延迟 `tool_call_start`，在 delta 时发送 |
| — | Anthropic `input_schema` 无参数工具缺少 `type` | 默认为 `{"type": "object"}` |

---

## 已知限制

- **Claude Code + 非 Claude 模型**：Claude Code 通过 Read 工具传递图像数据。部分模型（如 GPT-4.1-nano）可能无法正确解读工具结果中的 base64 图像。建议使用 GPT-5.4+ 以获得可靠的图像理解。
- **Gemini CLI 无头模式**：非交互式（`-p`）模式下 Gemini CLI 没有 `--image` 参数。可通过内置 `read_file` 工具读取工作区内的图片文件。
- **认证透传**：网关透传上游的认证错误。上游返回 401 表示 API 密钥对目标提供商无效，不是网关问题。
