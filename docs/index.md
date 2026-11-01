---
title: Home
author: Oaklight
hide:
  - navigation
---

<div style="display: flex; align-items: center; gap: 1.5em; margin-bottom: 0.5em;">
  <img src="images/rosetta_stone.svg" alt="Rosetta Stone" style="width: 96px; flex-shrink: 0;">
  <div>
    <h1 style="margin: 0 0 0.2em 0;">LLM-Rosetta</h1>
    <p style="margin: 0; font-size: 1.1em; opacity: 0.85;">A self-hosted, format-neutral translation layer for LLM provider APIs.</p>
    <p style="margin: 0.4em 0 0 0;">
      <a href="https://pypi.org/project/llm-rosetta/"><img src="https://img.shields.io/pypi/v/llm-rosetta?color=green" alt="PyPI"></a>
      <a href="https://test.pypi.org/project/llm-rosetta/"><img src="https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Ftest.pypi.org%2Fpypi%2Fllm-rosetta%2Fjson&query=%24.info.version&label=TestPyPI&color=blue" alt="TestPyPI"></a>
      <a href="https://github.com/Oaklight/llm-rosetta/releases/latest"><img src="https://img.shields.io/github/v/release/Oaklight/llm-rosetta?color=green" alt="Release"></a>
      <a href="https://hub.docker.com/r/oaklight/llm-rosetta-gateway"><img src="https://img.shields.io/docker/v/oaklight/llm-rosetta-gateway?label=Docker&color=blue" alt="Docker"></a>
      <a href="https://github.com/Oaklight/llm-rosetta/blob/master/LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="MIT"></a>
      <a href="https://arxiv.org/abs/2604.09360"><img src="https://img.shields.io/badge/arXiv-2604.09360-b31b1b.svg" alt="arXiv"></a>
    </p>
  </div>
</div>

Just as the Rosetta Stone enabled translation between ancient scripts, LLM-Rosetta bridges the gap between incompatible LLM provider APIs — letting any client speak its own format and reach any model source, while you keep control of every upstream.

---

## The Problem

Different LLM providers use incompatible API formats. A request that works with OpenAI won't work with Anthropic or Google. Switching providers means rewriting integration code. Supporting N formats pairwise takes O(N²) adapters.

**LLM-Rosetta** introduces a central **Intermediate Representation (IR)** as a hub. Each format only needs one bidirectional converter to and from the IR, reducing the total to 2N.

Provider A ↔ **IR** ↔ Provider B — any supported format in, any supported format out.

---

<a id="four-levels"></a>

## Four Levels of Control

| | Functions | Raw converter | Pipeline | Gateway |
|---|---|---|---|---|
| **Use case** | Convert one IR value or tool | Translate one leg inside your own loop | Source → IR → target in one call | HTTP proxy for any client |
| **Entry point** | `tool_ops`, `transforms`, `capabilities` | `ChatCompletionsConverter()` | `convert()` / `ConversionPipeline` | `llm-rosetta-gateway` |
| **Provider call** | You | You | You | The gateway |
| **Routing, keys, admin** | — | — | — | Built in |

All four share the same IR and converters. Start at the level you need and move up without rewriting the translation. `tool_ops` converts one tool definition, choice, call or result between a provider dialect and IR; `transforms` and `capabilities` apply one field or capability pass without building a converter.

Python 3.11+ needs no third-party runtime dependency; 3.10 adds only `typing_extensions`, a backport of the standard library's `typing` module. That is unusual for a multi-provider gateway, and it means the same pipeline moves between applications and deployments without a dependency graph to resolve.

---

## Quick Start

=== "As a Library"

    Convert one leg with a raw converter, or run the full source → IR → target
    path with the pipeline:

    ```python
    from llm_rosetta import ChatCompletionsConverter, MessagesConverter, convert

    openai_request = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "Hello!"}],
    }

    # Raw converter: one leg at a time
    ir = ChatCompletionsConverter().request_from_provider(openai_request)
    anthropic_request, warnings = MessagesConverter().request_to_provider(ir)

    # Pipeline: source → IR → target in one call
    anthropic_request = convert(openai_request, target_provider="messages")
    ```

    ```bash
    pip install llm-rosetta
    ```

    [Library Quick Start →](getting-started/library-quickstart.md){ .md-button }

=== "As a Gateway"

    Run a local HTTP proxy that translates between formats in real time, and
    keeps every upstream, credential and route in your deployment:

    ```mermaid
    flowchart LR
        C1[Chat Completions] --> GW((Gateway))
        C2[Responses API] --> GW
        C3[Open Responses] --> GW
        C4[Anthropic format] --> GW
        C5[Google format] --> GW
        GW --> P1[Anthropic API]
        GW --> P2[Google API]
        GW --> P3[OpenAI API]
        GW --> P4[Any provider]
    ```

    ```bash
    pip install llm-rosetta-gateway
    llm-rosetta-gateway init
    llm-rosetta-gateway
    ```

    A backend for **Claude Code**, **Antigravity CLI (agy)**, **OpenAI Codex CLI**,
    **Kilo Code** and **OpenCode**, and a translating proxy in front of **Ollama**.
    See [CLI Integrations](gateway/cli-integrations.md).

    [Gateway Quick Start →](getting-started/gateway-quickstart.md){ .md-button }

---

## Supported Formats

| Model type | Formats |
|------------|---------|
| LLM (chat) | `chat_completions`, `open_responses`, `messages`, `google_generate`, `google_interactions` |
| Embeddings | `openai`, `cohere`, `jina`, `voyage`, `google_generate`, `minimax` |
| Rerank | `jina`, `cohere`, `voyage` |
| Decisions | `systemone`, `openai_decisions` |

Every format converts both to and from the IR, for requests and responses; chat
formats also convert streamed chunks. See [API Standards](guide/api-standards.md)
for detailed comparisons and [Provider Compatibility](guide/compatibility.md) for
the provider matrix.

---

## Key Features

| | |
|---|---|
| **Four levels of control** | Functions, raw converter, pipeline, gateway — one IR underneath |
| **Bidirectional conversion** | Requests, responses and streamed chunks in both directions |
| **IR as the application payload** | Keep history and orchestration state provider-neutral |
| **Provider shims** | 39 shims across 18 providers declare field, auth and URL differences |
| **Gateway + admin console** | HTTP proxy with routing, keys, logs, metrics and an admin UI |
| **Zero third-party dependencies** | Nothing on Python 3.11+; `typing_extensions` on 3.10 only |
| **Type safe** | Full TypedDict annotations for all types |

---

## Use Cases

**Multi-provider applications** — Build apps that switch between LLM providers without changing integration code. Use OpenAI in production and Claude for testing, or let users choose their provider.

**AI coding tool proxy** — Run a single gateway serving Claude Code, Antigravity CLI, Codex CLI, and more, routing each to the right upstream.

**Local model access** — Point the gateway at Ollama or LM Studio to let cloud-SDK-based tools talk to local models with automatic format conversion.

**API migration** — Migrating providers? Convert existing request/response handling without rewriting business logic.

---

## Documentation

- **[Getting Started](getting-started/installation.md)** — Installation, core concepts, and quick starts
- **[Guide](guide/converters.md)** — Converters, IR types, providers, streaming
- **[Gateway](gateway/index.md)** — HTTP proxy, configuration, CLI integrations, admin panel
- **[Reference](api/index.md)** — API documentation, examples, contributing, changelog
- **[Contributing](contributing/guide.md)** — How to contribute, style guide, architecture

## Citation

If you use LLM-Rosetta in your research, please cite our paper:

```bibtex
@article{ding2026llm,
  title={LLM-Rosetta: A Hub-and-Spoke Intermediate Representation for Cross-Provider LLM API Translation},
  author={Ding, Peng},
  journal={arXiv preprint arXiv:2604.09360},
  year={2026}
}
```

## License

MIT License
