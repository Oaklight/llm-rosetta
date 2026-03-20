---
title: Roadmap
---

# Roadmap

This page outlines planned features and areas where community contributions are welcome.

## Current Status

LLM-Rosetta v0.2.0 supports bidirectional conversion between 4 provider APIs:

| Provider | Format | Streaming | Tool Calls |
|----------|--------|:---------:|:----------:|
| OpenAI Chat Completions | `openai_chat` | :white_check_mark: | :white_check_mark: |
| OpenAI Responses | `openai_responses` | :white_check_mark: | :white_check_mark: |
| Anthropic Messages | `anthropic` | :white_check_mark: | :white_check_mark: |
| Google GenAI | `google` | :white_check_mark: | :white_check_mark: |

The [Gateway](gateway/index.md) provides real-time HTTP proxying between any combination of these formats, verified with [5 CLI tools and SDK test suites](gateway/validation.md).

---

## Planned Features

### Open Responses Integration

!!! tip "Status: Planned"

[Open Responses](https://www.openresponses.org/) is an open-source specification (Apache 2.0) initiated by OpenAI in January 2026. It turns the proprietary OpenAI Responses API into a **vendor-neutral standard** with formal extensibility rules.

**Why it matters**: Open Responses is a **proper superset** of the OpenAI Responses API. A client already talking to OpenAI's Responses API can talk to an Open Responses endpoint with minimal changes. Major adopters include OpenRouter, Hugging Face, Vercel, LM Studio, Ollama, and vLLM.

**Implementation strategy**: Extend the existing `openai_responses` converter rather than building a separate one. The delta is small:

| Feature | Description |
|---------|-------------|
| Reasoning `content` field | Raw reasoning traces from open-weight models (in addition to `summary` and `encrypted_content`) |
| Slug-prefixed extensions | `implementor:type_name` items, tools, and events (e.g., `openai:web_search_call`) |
| `allowed_tools` field | Cache-preserving tool restriction |
| `OpenResponses-Version` header | Spec versioning mechanism |
| Stateless default | Already compatible — llm-rosetta doesn't assume server-side state |

This could be exposed as:

- A flag: `output_format="open_responses"` on the converter
- Or a thin subclass: `OpenResponsesConverter(OpenAIResponsesConverter)`
- Gateway: detect via `OpenResponses-Version` header and route accordingly

See the full [analysis](https://github.com/Oaklight/llm-rosetta/blob/master/analysis/openapi_specs_and_open_responses.md) for detailed schema comparisons.

### Ollama Provider Support

!!! tip "Status: Planned"

[Ollama](https://ollama.com/) is a popular tool for running LLMs locally. It exposes both a native API and an OpenAI-compatible API.

**Implementation approach**:

- **OpenAI-compatible mode**: Ollama's `/v1/chat/completions` endpoint is already compatible with the `openai_chat` converter — no new converter needed, just gateway configuration pointing to `http://localhost:11434/v1`
- **Native Ollama API**: A dedicated converter could support Ollama-specific features (model management, embedding, etc.) but is lower priority since the OpenAI-compatible mode covers the primary use case

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

For larger features (Open Responses, new providers), please open an issue first to discuss the approach.
