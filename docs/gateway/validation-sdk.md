---
title: SDK 与集成测试
---

# SDK 与集成测试

本页记录了用于验证 LLM-Rosetta 转换器管道的自动化测试套件，使用真实 API 调用。

!!! info "最后更新：2026-04-07"
    使用 llm-rosetta v0.2.7（未发布）、argo-proxy v3.0.0b7

## 集成测试套件（`tests/integration/`）

集成测试套件使用官方 SDK 和直接 REST 调用验证所有四个转换器管道。每个测试涵盖非流式、流式、工具调用、往返转换和多轮对话。

### 运行测试

```bash
cd /path/to/llm-rosetta

# 设置代理端点和模型
export ARGO_PROXY_URL=http://localhost:44511
export MODEL=argo:gpt-4.1-nano  # 或其他可用模型

# 运行所有集成测试
python -m pytest tests/integration/ -v

# 运行特定测试套件
python -m pytest tests/integration/test_google_genai_sdk_e2e.py -v
python -m pytest tests/integration/test_openai_chat_sdk_e2e.py -v
python -m pytest tests/integration/test_openai_responses_rest_e2e.py -v
python -m pytest tests/integration/test_anthropic_rest_e2e.py -v
```

### 结果汇总

| 测试套件 | 测试数 | 结果 |
|---------|:-----:|:----:|
| OpenAI Chat SDK | 9 | **9/9** ✓ |
| OpenAI Responses SDK | 6 | **6/6** ✓ |
| Anthropic SDK | 8 | **8/8** ✓ |
| Google GenAI SDK | 7 | **7/7** ✓ |
| **合计** | **30** | **30/30** ✓ |

### 各套件测试覆盖

| 测试 | OpenAI Chat | OpenAI Responses | Anthropic | Google GenAI |
|-----|:-----------:|:----------------:|:---------:|:------------:|
| 非流式基础文本 | ✓ | ✓ | ✓ | ✓ |
| 非流式图像 | ✓ | — | — | — |
| 非流式工具调用 | ✓ | ✓ | ✓ | ✓ |
| 流式文本 | ✓ | — | ✓ | — |
| 流式工具调用 | ✓ | — | ✓ | — |
| 多模态工具结果 | ✓ | ✓ | ✓ | ✓ |
| 图片输入 + 工具调用 | ✓ | ✓ | ✓ | ✓ |
| 请求往返 | ✓ | ✓ | ✓ | ✓ |
| 响应往返 | ✓ | ✓ | ✓ | ✓ |
| 多轮对话 | — | — | — | ✓ |

---

## 同格式 CLI 验证

五款 CLI 工具使用其原生 API 格式通过网关进行了测试：

| CLI 工具 | API 格式 | 源 → 目标 | 对话 | 流式 | 工具调用 | 多轮 |
|---------|---------|----------|:----:|:----:|:------:|:----:|
| [Codex CLI](https://github.com/openai/codex) | OpenAI Responses | 透传 | ✓ | ✓ | ✓ | ✓ |
| [Kilo Code](https://kilocode.ai/) | OpenAI Chat | `openai_chat` → `openai_responses` | ✓ | ✓ | ✓ | ✓ |
| [OpenCode](https://opencode.ai/) | OpenAI Chat | `openai_chat` → `openai_responses` | ✓ | ✓ | ✓ | ✓ |
| [Claude Code](https://docs.anthropic.com/en/docs/claude-code) | Anthropic Messages | 透传 | ✓ | ✓ | ✓ | ✓ |
| [Gemini CLI](https://github.com/google-gemini/gemini-cli) | Google GenAI | `google` → `openai_chat` | ✓ | ✓ | ✓ | ✓ |

---

## SDK 测试套件（`llm_api_simple_tests`）

[llm_api_simple_tests](https://github.com/Oaklight/llm_api_simple_tests) 套件使用官方 SDK 为每个提供商运行 5 个标准化测试。

### 通过网关的 Anthropic SDK

**配置**：`ANTHROPIC_BASE_URL=http://localhost:8765`，模型 `anthropic/claude-3-haiku`

| 测试 | 描述 | 状态 |
|-----|------|:----:|
| `simple_query.py` | 单轮流式查询 | ✓ |
| `multi_round_chat.py` | 3 轮对话 | ✓ |
| `multi_round_function_calling.py` | 3 轮工具调用 | ✓ |
| `multi_round_comprehensive.py` | 3 轮图像+工具调用 | ✓ |
| `multi_round_image.py` | 3 轮视觉对话 | ✓ |

### 通过网关的 Google GenAI（curl）

通过 `curl` 直接对网关的 Google 端点进行多轮工具调用测试：

```bash
# 第 1 轮：函数调用
curl -s http://localhost:44511/v1beta/models/gemini-2.5-flash:generateContent \
  -H "Content-Type: application/json" \
  -H "x-goog-api-key: your_key" \
  -d '{
    "contents": [{"role": "user", "parts": [{"text": "What is 127 * 389?"}]}],
    "tools": [{"functionDeclarations": [{"name": "calculator", "description": "Calculate math", "parameters": {"type": "object", "properties": {"expression": {"type": "string"}}}}]}],
    "toolConfig": {"functionCallingConfig": {"mode": "ANY"}}
  }'
# 预期：functionCall，calculator({expression: "127 * 389"})

# 第 2 轮：发送工具结果，追问
curl -s http://localhost:44511/v1beta/models/gemini-2.5-flash:generateContent \
  -H "Content-Type: application/json" \
  -H "x-goog-api-key: your_key" \
  -d '{
    "contents": [
      {"role": "user", "parts": [{"text": "What is 127 * 389?"}]},
      {"role": "model", "parts": [{"functionCall": {"name": "calculator", "args": {"expression": "127 * 389"}}}]},
      {"role": "user", "parts": [{"functionResponse": {"name": "calculator", "response": {"result": 49403}}}]},
      {"role": "user", "parts": [{"text": "Now add 100 to that"}]}
    ],
    "tools": [{"functionDeclarations": [{"name": "calculator", "description": "Calculate math", "parameters": {"type": "object", "properties": {"expression": {"type": "string"}}}}]}],
    "toolConfig": {"functionCallingConfig": {"mode": "ANY"}}
  }'
# 预期：functionCall，calculator({expression: "49403 + 100"})
```

| 轮次 | 请求 | 模型响应 | 状态 |
|:---:|------|--------|:----:|
| 1 | "What is 127 * 389?" + `calculator` 工具 | `functionCall: calculator({expression: "127 * 389"})` | ✓ |
| 2 | 工具结果 `49403`，"add 100 to that" | `functionCall: calculator({expression: "49403 + 100"})` | ✓ |
| 3 | 工具结果 `49503`，mode=AUTO | 文本："The result is 49503." | ✓ |

使用 `gemini-2.5-flash-lite` 和 `gemini-3.1-flash-lite-preview` 均测试通过。
