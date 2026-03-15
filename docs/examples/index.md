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

## 运行示例

示例脚本位于仓库的 `examples/` 目录中：

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
