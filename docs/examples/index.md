---
title: 示例
---

# 示例

LLM-Rosetta 包含全面的跨提供商对话示例。

## 可用示例

| 示例 | 描述 |
|------|------|
| [跨提供商对话](cross-provider.md) | 在不同提供商之间的多轮对话 |
| [工具调用](tool-calling.md) | 跨提供商的函数调用 |

## 独立 API 测试脚本

`llm_api_simple_tests/` 子模块包含直接使用官方提供商 SDK 的独立测试脚本（不依赖 LLM-Rosetta）。可用于独立验证提供商 API 或网关兼容性。

**支持的提供商：** OpenAI Chat、Anthropic、Google GenAI、OpenAI Responses

每个提供商有 5 个脚本：`simple_query`、`multi_round_chat`、`multi_round_image`、`multi_round_function_calling`、`multi_round_comprehensive`

```bash
cd llm_api_simple_tests
pip install -r requirements.txt

# 使用提供商特定的环境变量运行
BASE_URL=https://api.openai.com/v1 API_KEY=sk-... MODEL=gpt-4o \
  python scripts/openai_chat/simple_query.py

# 或在 .env 中设置 OPENAI_API_KEY、ANTHROPIC_API_KEY 等
```

详见 [`llm_api_simple_tests/README.md`](https://github.com/Oaklight/llm_api_simple_tests) 完整文档。

## 运行跨提供商示例

跨提供商示例脚本位于仓库的 `examples/` 目录中：

```bash
git clone https://github.com/Oaklight/llm-rosetta.git
cd llm-rosetta/examples

# 设置 API 密钥
cp .env.example .env
# 编辑 .env 填入您的 API 密钥

# 运行示例
python sdk_based/cross_openai_chat_anthropic.py
```

示例有两种形式：

- **基于 SDK**（`sdk_based/`）— 直接使用提供商 SDK
- **基于 REST**（`rest_based/`）— 使用原始 HTTP 请求
