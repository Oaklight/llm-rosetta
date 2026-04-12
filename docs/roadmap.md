---
title: Roadmap
---

# Roadmap

This page outlines the current feature status and areas where community contributions are welcome.

## Current Status

LLM-Rosetta v0.5.0 supports bidirectional conversion between 5 API standards:

| Provider | Format | Streaming | Tool Calls |
|----------|--------|:---------:|:----------:|
| OpenAI Chat Completions | `openai_chat` | ✓ | ✓ |
| OpenAI Responses | `openai_responses` | ✓ | ✓ |
| Open Responses | `open_responses` | ✓ | ✓ |
| Anthropic Messages | `anthropic` | ✓ | ✓ |
| Google GenAI | `google` | ✓ | ✓ |

The [Gateway](gateway/index.md) provides real-time HTTP proxying between any combination of these formats, verified with [5 CLI tools and SDK test suites](gateway/validation.md). The gateway also includes a built-in [admin panel](gateway/admin-panel.md) for configuration management, metrics monitoring, and request logging.

See [API Standards](guide/api-standards.md) for details on each format.

---

## Recently Completed

### Open Responses Integration

!!! success "Status: Done (v0.5.0)"

[Open Responses](https://www.openresponses.org/) is an open-source specification (Apache 2.0) initiated by OpenAI in January 2026. It turns the proprietary OpenAI Responses API into a **vendor-neutral standard** with formal extensibility rules.

**What was implemented**:

- `open_responses` provider type — reuses `OpenAIResponsesConverter` since the wire format is compatible
- Gateway forwards `OpenResponses-Version` header to upstream when present
- Auto-detection recognizes Open Responses request bodies
- Configurable as a distinct provider type in gateway config

Major adopters include OpenRouter, Hugging Face, Vercel, LM Studio, Ollama, and vLLM.

See the full [analysis](https://github.com/Oaklight/llm-rosetta/blob/master/analysis/openapi_specs_and_open_responses.md) for detailed schema comparisons.

### Ollama Support

!!! success "Status: Done (v0.5.0)"

[Ollama](https://ollama.com/) (v0.13+) works with the gateway in two ways:

- **As an upstream provider**: Point a gateway provider at `http://localhost:11434/v1` using the `openai_chat` type — no new converter needed
- **As a client**: Ollama's OpenAI-compatible endpoints (`/v1/chat/completions`, `/v1/responses`, `/v1/messages`) can target the gateway to reach cloud providers

See [CLI Integrations — Ollama](gateway/cli-integrations.md#ollama) for configuration examples.

---

## Planned Features

### LM Studio Provider Support

!!! tip "Status: Planned — [#42](https://github.com/Oaklight/llm-rosetta/issues/42)"

[LM Studio](https://lmstudio.ai/) provides OpenAI-compatible local inference. Similar to Ollama, it works with the existing `openai_chat` converter via gateway configuration.

### HuggingFace Inference API

!!! tip "Status: Planned — [#40](https://github.com/Oaklight/llm-rosetta/issues/40)"

[HuggingFace Inference API](https://huggingface.co/docs/api-inference/) supports multiple model formats. A dedicated converter would enable routing to HuggingFace-hosted models through the gateway.

### Model Listing per Provider SDK

!!! note "Status: Open — [#54](https://github.com/Oaklight/llm-rosetta/issues/54)"

Extend the gateway's model listing endpoints to query upstream providers and merge with locally configured models.

---

## Community Contributions

We welcome pull requests for any of the planned features above. Here's how to get started:

1. Check the [issue tracker](https://github.com/Oaklight/llm-rosetta/issues) for open issues
2. Read the [Core Concepts](guide/concepts.md) guide to understand the converter architecture
3. Look at existing converters (e.g., `src/llm_rosetta/converters/openai_chat/`) as templates
4. Run `ruff check` and `uvx ty check` before submitting

For larger features (new providers), please open an issue first to discuss the approach.
