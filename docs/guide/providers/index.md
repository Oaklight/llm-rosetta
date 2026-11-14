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
| [Z.ai](zai.md) | `zai--*` | multiple | GLM international platform; Chat, Messages, embeddings, rerank |
| [Cohere](cohere.md) | `cohere--*` | multiple | Native embeddings and rerank; chat through the compatibility endpoint |
| [Groq](groq.md) | `groq--chat_completions` | `chat_completions` | OpenAI-compatible, LPU inference |
| [NVIDIA NIM](nvidia.md) | `nvidia--chat_completions` | `chat_completions` | OpenAI-compatible; prototyping licence |
| [SiliconFlow](siliconflow.md) | `siliconflow--*` | multiple | Typed catalogue; chat, embeddings, rerank |
| [ModelScope](modelscope.md) | `modelscope--chat_completions` | `chat_completions` | OpenAI-compatible; Alibaba Cloud account |
| [Argo](argo.md) | `argo--*` | multiple | Standalone package; Argonne internal gateway |
| [AskSage](asksage.md) | `asksage--*` | multiple | Standalone package; multi-format aggregator |
| [ALCF](alcf.md) | `alcf_<cluster>--chat_completions` | `chat_completions` | Standalone package; Globus OAuth, token_command |
| [Kilo](kilo.md) | `kilo--chat_completions` | `chat_completions` | Free pool, keyless |
| [OpenCode](opencode.md) | `opencode--*` | multiple | Go subscription / Zen balance |
| [Open Design](opendesign.md) | `opendesign--*` | multiple | Standalone package; model gateway |
| [Plan (ChatGPT)](openai_chatgpt.md) | `openai_chatgpt--open_responses` | `open_responses` | ChatGPT plan route; sign-in instead of an API key |

The Argo, AskSage, ALCF, Open Design and OpenCode free-lane shims ship as standalone packages — declare their source with `shims.presets` (or **Settings → Shim Sources**) before using their names. See [Standalone shim packages](../shims.md#standalone-shim-packages).

For general shim concepts and custom shim registration, see [Provider Shims](../shims.md).
