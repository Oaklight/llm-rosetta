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
| [Argo](argo.md) | † `argo--*` | `chat_completions` | 阿贡内部网关 |
| [AskSage](asksage.md) | `asksage--*` | 多种 | 独立包；多格式聚合器 |
| [ALCF](alcf.md) | `alcf_<cluster>--chat_completions` | `chat_completions` | 独立包；Globus OAuth, token_command |
| [Kilo](kilo.md) | `kilo--chat_completions` | `chat_completions` | 免费池，keyless |
| [OpenCode](opencode.md) | `opencode--*` | 多种 | Go 订阅 / Zen 余额 |

† Argo 的 shim [计划于 0.16.0 移除](../shims.md#shims-scheduled-for-removal)，届时将迁移到独立的 shim 包。在此之前它们继续可用，名称也保持不变。ALCF 与 AskSage 的 shim 已作为独立包发布 —— 在使用其名称之前先声明对应源（**Settings → Shim Sources** 提供预设）。

关于 shim 的通用概念和自定义 shim 注册，请参阅[提供方 Shims](../shims.md)。
