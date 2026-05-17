---
title: Roadmap
---

# Roadmap

This page outlines the current feature status and areas where community contributions are welcome.

## Current Status

LLM-Rosetta v0.6.2 supports bidirectional conversion between 5 API standards:

| Provider | Format | Streaming | Tool Calls | Embeddings |
|----------|--------|:---------:|:----------:|:----------:|
| OpenAI Chat Completions | `openai_chat` | ✓ | ✓ | ✓ |
| OpenAI Responses | `openai_responses` | ✓ | ✓ | — |
| Open Responses | `open_responses` | ✓ | ✓ | — |
| Anthropic Messages | `anthropic` | ✓ | ✓ | — |
| Google GenAI | `google` | ✓ | ✓ | — |

The [Gateway](gateway/index.md) provides real-time HTTP proxying with a **zero-dependency runtime**, verified with [5 CLI tools and SDK test suites](gateway/validation.md). The gateway includes a built-in [Admin Panel](gateway/admin-panel.md) with a full [REST API](api/admin.md).

The **provider shim layer** enables adding new providers via declarative YAML files — no converter code needed for OpenAI-compatible providers. 14 providers are supported out of the box.

See [API Standards](guide/api-standards.md) for details on each format.

---

## Recently Completed

### Declarative Provider Shim System

!!! success "Status: Done (v0.6.0)"

Providers are now defined as `provider.yaml` + optional `transforms.py` files under `shims/providers/<name>/`, automatically discovered at import time. Three composable transform primitives — `strip_fields()`, `rename_field()`, `set_defaults()` — handle field-level differences between a provider's API dialect and its base standard.

7 new built-in shims added: xAI (Grok), Qwen (DashScope), Moonshot (Kimi), MiniMax, Zhipu (GLM), OpenRouter, Volcengine. The gateway proxy pipeline applies shim transforms on both request and response paths.

### Zero-Dependency Gateway

!!! success "Status: Done (v0.6.0)"

Replaced Starlette + uvicorn + httpx with vendored zerodep `httpserver` and `httpclient` modules. The `[gateway]` extra now has zero external runtime dependencies.

### Embeddings Passthrough

!!! success "Status: Done (v0.6.1)"

`/v1/embeddings` passthrough endpoint proxies embedding requests directly to upstream providers without IR conversion. `/v1/models` response now includes `api_standard` and per-model `capabilities` fields.

### Admin Panel Enhancements

!!! success "Status: Done (v0.6.1)"

- **Fetch from Provider**: query upstream `/v1/models`, browse and bulk-add models
- **Model capabilities**: `embedding` and `reasoning` capability types with dedicated test modes
- **Provider logos**: shims can declare SVG logos displayed on admin cards
- **Admin API**: full REST API for programmatic configuration management

### SOCKS5 Proxy Support

!!! success "Status: Done (v0.6.0)"

Full SOCKS5 proxy support (RFC 1928/1929) via vendored httpclient v0.4.0, including username/password authentication.

---

## Planned Features

### Converter Enhancements

#### Server-Side Tool Type Mapping

!!! tip "Status: Planned — [#181](https://github.com/Oaklight/llm-rosetta/issues/181)"

Cross-provider mapping for server-side tool types (`web_search`, `code_execution`, `computer_use`) that exist in some providers but not others.

#### Custom Tool Type in Responses API

!!! tip "Status: Planned — [#182](https://github.com/Oaklight/llm-rosetta/issues/182)"

Handle the OpenAI Responses `custom` tool type in IR, enabling pass-through for provider-specific tool extensions.

#### Reasoning Field Normalization

!!! tip "Status: Planned — [#185](https://github.com/Oaklight/llm-rosetta/issues/185)"

Normalize `reasoning_content` / thinking fields across OpenAI Chat-compatible providers (e.g. DeepSeek, Qwen) via shim transforms instead of per-provider converter code.

### Shim System

#### Per-Model Transforms (ModelShim)

!!! tip "Status: Planned — [#192](https://github.com/Oaklight/llm-rosetta/issues/192)"

Restore `ModelShim` to enable per-model transform rules — different models from the same provider may need different field handling.

#### Multi-API-Mode Providers

!!! note "Status: Open — [#186](https://github.com/Oaklight/llm-rosetta/issues/186)"

Support providers that expose multiple API standards simultaneously (e.g. a provider offering both Chat Completions and Responses endpoints).

### Gateway

#### Upstream Timeout & Circuit Breaker

!!! tip "Status: Planned — [#121](https://github.com/Oaklight/llm-rosetta/issues/121)"

Configurable per-provider timeouts and circuit breaker pattern to handle slow or failing upstreams gracefully.

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

!!! note "Status: Open — [#42](https://github.com/Oaklight/llm-rosetta/issues/42)"

[LM Studio](https://lmstudio.ai/) provides OpenAI-compatible local inference. Works with the existing `openai_chat` converter via gateway configuration.

#### HuggingFace Inference API

!!! note "Status: Open — [#40](https://github.com/Oaklight/llm-rosetta/issues/40)"

[HuggingFace Inference API](https://huggingface.co/docs/api-inference/) supports multiple model formats. A dedicated converter or shim would enable routing to HuggingFace-hosted models.

---

## Community Contributions

We welcome pull requests for any of the planned features above. Here's how to get started:

1. Check the [issue tracker](https://github.com/Oaklight/llm-rosetta/issues) for open issues
2. Read the [Core Concepts](guide/concepts.md) guide to understand the converter architecture
3. Look at existing converters (e.g., `src/llm_rosetta/converters/openai_chat/`) as templates
4. For new providers, consider creating a [shim](guide/shims.md) first — it's often enough
5. Run `pre-commit run --all-files` before submitting

For larger features, please open an issue first to discuss the approach.
