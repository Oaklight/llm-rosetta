---
title: Roadmap
---

# Roadmap

This page outlines the current feature status and areas where community contributions are welcome.

## Current Status

LLM-Rosetta v0.9.0 supports bidirectional conversion across 3 API families:

**Chat / Completions** (5 standards):

| Provider | Format | Streaming | Tool Calls |
|----------|--------|:---------:|:----------:|
| OpenAI Chat Completions | `openai_chat` | ✓ | ✓ |
| OpenAI Responses | `openai_responses` | ✓ | ✓ |
| Open Responses | `open_responses` | ✓ | ✓ |
| Anthropic Messages | `anthropic` | ✓ | ✓ |
| Google GenAI | `google` | ✓ | ✓ |

**Embedding** (4 formats): OpenAI, Jina, Voyage, Cohere — with IR-based cross-format conversion.

**Rerank** (3 formats): Jina, Cohere, Voyage — with IR-based cross-format conversion.

The [Gateway](gateway/index.md) provides real-time HTTP proxying with a **zero-dependency runtime**, verified with [5 CLI tools and SDK test suites](gateway/validation.md). The gateway includes a built-in [Admin Panel](gateway/admin-panel.md) with a full [REST API](api/admin.md).

The **provider shim layer** enables adding new providers via declarative YAML files — no converter code needed for OpenAI-compatible providers. 16 providers are supported out of the box.

See [API Standards](guide/api-standards.md) for details on each format.

---

!!! info "Completed features"
    For features already shipped, see the [Changelog](changelog.md). Key milestones include: declarative shim system (v0.6.0), zero-dependency gateway (v0.6.0), embedding/rerank IR conversion (v0.6.1+), reasoning field normalization via shims (v0.8.1), upstream timeout (v0.8.2), and multi-API-mode providers (v0.6.8).

---

## Planned Features

### Converter Enhancements

#### Server-Side Tool Type Mapping

!!! tip "Status: Planned — [#181](https://github.com/Oaklight/llm-rosetta/issues/181)"

Cross-provider mapping for server-side tool types (`web_search`, `code_execution`, `computer_use`) that exist in some providers but not others.

### Shim System

#### Per-Model Transforms (ModelShim)

!!! tip "Status: Planned — [#192](https://github.com/Oaklight/llm-rosetta/issues/192)"

Restore `ModelShim` to enable per-model transform rules — different models from the same provider may need different field handling.

### Gateway

#### Rate Limiting Middleware

!!! tip "Status: Planned — [#124](https://github.com/Oaklight/llm-rosetta/issues/124)"

Token-bucket or sliding-window rate limiting per API key or per client IP.

#### Enhanced Error Responses

!!! tip "Status: Planned — [#123](https://github.com/Oaklight/llm-rosetta/issues/123)"

Include upstream error context in gateway error responses for easier debugging.

#### Cost Tracking per Provider

!!! note "Status: Open — [#131](https://github.com/Oaklight/llm-rosetta/issues/131)"

Track token usage costs per provider and surface them in the admin dashboard.

#### Fallback Chain & Load Balancing

!!! note "Status: Open — [#129](https://github.com/Oaklight/llm-rosetta/issues/129)"

Automatic failover to backup providers when the primary is unavailable, with optional load balancing across multiple providers.

### Provider Support

#### LM Studio

!!! warning "Status: Deferred — [#42](https://github.com/Oaklight/llm-rosetta/issues/42)"

[LM Studio](https://lmstudio.ai/) provides OpenAI-compatible local inference. Works with the existing `openai_chat` converter via gateway configuration. Low priority as it already works without dedicated support.

#### HuggingFace Inference API

!!! warning "Status: Deferred — [#40](https://github.com/Oaklight/llm-rosetta/issues/40)"

[HuggingFace Inference API](https://huggingface.co/docs/api-inference/) supports multiple model formats. A dedicated converter or shim would enable routing to HuggingFace-hosted models. Deferred pending community interest.

---

## Community Contributions

We welcome pull requests for any of the planned features above. Here's how to get started:

1. Check the [issue tracker](https://github.com/Oaklight/llm-rosetta/issues) for open issues
2. Read the [Core Concepts](guide/concepts.md) guide to understand the converter architecture
3. Look at existing converters (e.g., `src/llm_rosetta/converters/openai_chat/`) as templates
4. For new providers, consider creating a [shim](guide/shims.md) first — it's often enough
5. Run `pre-commit run --all-files` before submitting

For larger features, please open an issue first to discuss the approach.
