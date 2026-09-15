---
title: Provider Reference
---

# Provider Reference

Per-provider guides covering shim configuration, transforms, authentication, and known limitations.

| Provider | Shim | Base Type | Notes |
|:---------|:-----|:----------|:------|
| [OpenAI](openai.md) | `openai` | `openai_chat` | Chat Completions API |
| [Anthropic](anthropic.md) | `anthropic` | `anthropic` | Messages API |
| [Google](google.md) | `google` | `google_generate` | generateContent API |
| [DeepSeek](deepseek.md) | `deepseek` | `openai_chat` | OpenAI-compatible |
| [Moonshot](moonshot.md) | `moonshot` | `openai_chat` | OpenAI-compatible |
| [Qwen](qwen.md) | `qwen` | `openai_chat` | OpenAI-compatible |
| [xAI](xai.md) | `xai` | `openai_chat` | OpenAI-compatible |
| [Zhipu](zhipu.md) | `zhipu` | `openai_chat` | OpenAI-compatible |
| [Volcengine](volcengine.md) | `volcengine` | `openai_chat` | Field transforms |
| [OpenRouter](openrouter.md) | `openrouter` | `openai_chat` | Multi-model router |
| [MiniMax](minimax.md) | `minimax` | `openai_chat` | OpenAI-compatible |
| [Argo](argo.md) | `argo-*` | `openai_chat` | UIUC research proxy |
| [ALCF](alcf.md) | `alcf--*` | `openai_chat` | Globus OAuth, token_command |

For general shim concepts and custom shim registration, see [Provider Shims](../shims.md).
