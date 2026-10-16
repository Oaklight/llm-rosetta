---
title: Provider Reference
---

# Provider Reference

Per-provider guides covering shim configuration, transforms, authentication, and known limitations.

| Provider | Shim | Base Type | Notes |
|:---------|:-----|:----------|:------|
| [OpenAI](openai.md) | `openai--chat_completions` | `chat_completions` | Chat Completions API |
| [Anthropic](anthropic.md) | `anthropic--messages` | `messages` | Messages API |
| [Google](google.md) | `google--google_generate` | `google_generate` | generateContent API |
| [DeepSeek](deepseek.md) | `deepseek--chat_completions` | `chat_completions` | OpenAI-compatible |
| [Moonshot](moonshot.md) | `moonshot` | `chat_completions` | OpenAI-compatible |
| [Qwen](qwen.md) | `qwen` | `chat_completions` | OpenAI-compatible |
| [xAI](xai.md) | `xai` | `chat_completions` | OpenAI-compatible |
| [Zhipu](zhipu.md) | `zhipu` | `chat_completions` | OpenAI-compatible |
| [Volcengine](volcengine.md) | `volcengine--chat_completions` | `chat_completions` | Field transforms |
| [OpenRouter](openrouter.md) | `openrouter--chat_completions` | `chat_completions` | Multi-model router |
| [MiniMax](minimax.md) | `minimax--chat_completions` | `chat_completions` | OpenAI-compatible |
| [Argo](argo.md) | † `argo--*` | `chat_completions` | Argonne internal gateway |
| [AskSage](asksage.md) | † `asksage--*` | multiple | Multi-format aggregator |
| [ALCF](alcf.md) | † `alcf_<cluster>--chat_completions` | `chat_completions` | Globus OAuth, token_command |
| [Kilo](kilo.md) | `kilo--chat_completions` | `chat_completions` | Free pool, keyless |
| [OpenCode](opencode.md) | `opencode--*` | multiple | Go subscription / Zen balance |

† The Argo, AskSage and ALCF shims are [scheduled for removal](../shims.md#shims-scheduled-for-removal) in 0.16.0, when they move to standalone shim packages. They keep working, and keep their names, until then.

For general shim concepts and custom shim registration, see [Provider Shims](../shims.md).
