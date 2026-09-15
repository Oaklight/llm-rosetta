---
title: 提供方参考
---

# 提供方参考

各提供方的详细指南，涵盖 shim 配置、转换规则、认证方式和已知限制。

| 提供方 | Shim | 基础类型 | 备注 |
|:-------|:-----|:---------|:-----|
| [OpenAI](openai.md) | `openai` | `openai_chat` | Chat Completions API |
| [Anthropic](anthropic.md) | `anthropic` | `anthropic` | Messages API |
| [Google](google.md) | `google` | `google_generate` | generateContent API |
| [DeepSeek](deepseek.md) | `deepseek` | `openai_chat` | OpenAI 兼容 |
| [Moonshot](moonshot.md) | `moonshot` | `openai_chat` | OpenAI 兼容 |
| [Qwen](qwen.md) | `qwen` | `openai_chat` | OpenAI 兼容 |
| [xAI](xai.md) | `xai` | `openai_chat` | OpenAI 兼容 |
| [Zhipu](zhipu.md) | `zhipu` | `openai_chat` | OpenAI 兼容 |
| [Volcengine](volcengine.md) | `volcengine` | `openai_chat` | 字段转换 |
| [OpenRouter](openrouter.md) | `openrouter` | `openai_chat` | 多模型路由 |
| [MiniMax](minimax.md) | `minimax` | `openai_chat` | OpenAI 兼容 |
| [Argo](argo.md) | `argo-*` | `openai_chat` | 阿贡内部网关 |
| [ALCF](alcf.md) | `alcf--*` | `openai_chat` | Globus OAuth, token_command |

关于 shim 的通用概念和自定义 shim 注册，请参阅[提供方 Shims](../shims.md)。
