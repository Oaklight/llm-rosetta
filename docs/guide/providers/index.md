---
title: 提供方参考
---

# 提供方参考

各提供方的详细指南，涵盖 shim 配置、转换规则、认证方式和已知限制。

| 提供方 | Shim | 基础类型 | 备注 |
|:-------|:-----|:---------|:-----|
| [OpenAI](openai.md) | `openai--chat_completions` | `chat_completions` | Chat Completions API |
| [Anthropic](anthropic.md) | `anthropic--messages` | `messages` | Messages API |
| [Google](google.md) | `google--google_generate` | `google_generate` | generateContent API |
| [DeepSeek](deepseek.md) | `deepseek--chat_completions` | `chat_completions` | OpenAI 兼容 |
| [Moonshot](moonshot.md) | `moonshot` | `chat_completions` | OpenAI 兼容 |
| [Qwen](qwen.md) | `qwen` | `chat_completions` | OpenAI 兼容 |
| [xAI](xai.md) | `xai` | `chat_completions` | OpenAI 兼容 |
| [Zhipu](zhipu.md) | `zhipu` | `chat_completions` | OpenAI 兼容 |
| [Volcengine](volcengine.md) | `volcengine--chat_completions` | `chat_completions` | 字段转换 |
| [OpenRouter](openrouter.md) | `openrouter--chat_completions` | `chat_completions` | 多模型路由 |
| [MiniMax](minimax.md) | `minimax--chat_completions` | `chat_completions` | OpenAI 兼容 |
| [Argo](argo.md) | `argo--*` | 多种 | 独立包；阿贡内部网关 |
| [AskSage](asksage.md) | `asksage--*` | 多种 | 独立包；多格式聚合器 |
| [ALCF](alcf.md) | `alcf_<cluster>--chat_completions` | `chat_completions` | 独立包；Globus OAuth, token_command |
| [Kilo](kilo.md) | `kilo--chat_completions` | `chat_completions` | 免费池，keyless |
| [OpenCode](opencode.md) | `opencode--*` | 多种 | Go 订阅 / Zen 余额 |
| [Plan (ChatGPT)](openai_chatgpt.md) | `openai_chatgpt--open_responses` | `open_responses` | ChatGPT 套餐通道；登录而非 API key |

Argo、AskSage、ALCF 与 OpenCode 免费通道的 shim 作为独立包发布 —— 在使用其名称之前，请通过 `shims.presets`（或 **Settings → Shim Sources**）声明对应源。参见[独立 Shim 包](../shims.md#standalone-shim-packages)。

关于 shim 的通用概念和自定义 shim 注册，请参阅[提供方 Shims](../shims.md)。
