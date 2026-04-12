---
title: Home
author: Oaklight
hide:
  - navigation
---

<div style="display: flex; align-items: center; gap: 1.5em; margin-bottom: 0.5em;">
  <img src="images/rosetta_stone.jpg" alt="Rosetta Stone" style="width: 96px; border-radius: 10px; flex-shrink: 0;">
  <div>
    <h1 style="margin: 0 0 0.2em 0;">LLM-Rosetta</h1>
    <p style="margin: 0; font-size: 1.1em; opacity: 0.85;">A unified message format conversion library for LLM provider APIs.</p>
    <p style="margin: 0.4em 0 0 0;">
      <a href="https://pypi.org/project/llm-rosetta/"><img src="https://img.shields.io/pypi/v/llm-rosetta?color=green" alt="PyPI"></a>
      <a href="https://github.com/Oaklight/llm-rosetta/releases/latest"><img src="https://img.shields.io/github/v/release/Oaklight/llm-rosetta?color=green" alt="Release"></a>
      <a href="https://github.com/Oaklight/llm-rosetta/blob/master/LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="MIT"></a>
    </p>
  </div>
</div>

Just as the Rosetta Stone enabled translation between ancient scripts, LLM-Rosetta bridges the gap between incompatible LLM provider APIs — letting you speak any format and be understood by any provider.

---

## The Problem

Different LLM providers use incompatible API formats. A request that works with OpenAI won't work with Anthropic or Google. Switching providers means rewriting integration code. Supporting multiple providers means maintaining N² converters.

**LLM-Rosetta** introduces a central **Intermediate Representation (IR)** as a hub. Each provider only needs one converter to/from the IR, reducing the total from N² to 2N.

```text
                    ┌──────────────┐
  OpenAI Chat  ◄──►│              │
                    │              │
  OpenAI Resp  ◄──►│              │
                    │     IR       │
  Open Resp    ◄──►│ Intermediate │
                    │    Repr.     │
  Anthropic    ◄──►│              │
                    │              │
  Google GenAI ◄──►│              │
                    └──────────────┘
```

---

## Two Ways to Use

=== "As a Library"

    Convert between provider formats in your own code — no server needed:

    ```python
    from llm_rosetta import OpenAIChatConverter, AnthropicConverter

    openai_conv = OpenAIChatConverter()
    anthropic_conv = AnthropicConverter()

    # OpenAI format → IR → Anthropic format
    ir_request = openai_conv.request_from_provider(openai_request)
    anthropic_request, warnings = anthropic_conv.request_to_provider(ir_request)
    ```

    ```bash
    pip install llm-rosetta
    ```

=== "As a Gateway"

    Run a local HTTP proxy that translates between formats in real time:

    ```text
    Client (OpenAI format) ──→ Gateway ──→ Anthropic API
    Client (Anthropic format) ──→ Gateway ──→ Google API
    Client (Google format) ──→ Gateway ──→ Any provider
    ```

    ```bash
    pip install "llm-rosetta[gateway]"
    llm-rosetta-gateway
    ```

    Drop-in backend for **Claude Code**, **Gemini CLI**, **OpenAI Codex CLI**, **Kilo Code**, and **Ollama**. See [CLI Integrations](gateway/cli-integrations.md).

---

## Supported API Standards

| Provider | API Standard | ProviderType |
|----------|-------------|:------------:|
| OpenAI | Chat Completions | `openai_chat` |
| OpenAI | Responses | `openai_responses` |
| Open Responses | Vendor-neutral standard | `open_responses` |
| Anthropic | Messages | `anthropic` |
| Google | GenAI | `google` |

See [API Standards](guide/api-standards.md) for detailed format comparisons.

---

## Key Features

| | |
|---|---|
| **Hub-and-Spoke Architecture** | Central IR eliminates N² conversion problem |
| **Bidirectional Conversion** | Requests, responses, and messages in both directions |
| **Streaming Support** | Convert streaming chunks with stateful context |
| **Tool Calling** | Unified tool definition and call handling across providers |
| **Auto Detection** | Detect provider format from request structure |
| **Gateway + Admin Panel** | HTTP proxy with web UI for config, metrics, and logs |
| **Type Safe** | Full TypedDict annotations for all types |
| **Zero Overhead** | Pure dict transformations, no validation cost |

---

## Use Cases

**Multi-provider applications** — Build apps that switch between LLM providers without changing integration code. Use OpenAI in production and Claude for testing, or let users choose their provider.

**AI coding tool proxy** — Run a single gateway serving Claude Code, Gemini CLI, Codex CLI, and more, routing each to the right upstream.

**Local model access** — Point the gateway at Ollama or LM Studio to let cloud-SDK-based tools talk to local models with automatic format conversion.

**API migration** — Migrating providers? Convert existing request/response handling without rewriting business logic.

---

## Documentation

- **[Getting Started](getting-started/installation.md)** — Installation and first steps
- **[Guide](guide/concepts.md)** — Core concepts, converters, IR types, streaming
- **[API Standards](guide/api-standards.md)** — Detailed comparison of supported formats
- **[Gateway](gateway/index.md)** — HTTP proxy setup, configuration, CLI integrations
- **[Examples](examples/)** — Cross-provider conversations, tool calling
- **[API Reference](api/)** — Complete API documentation
- **[Changelog](changelog.md)** — Version history

## Citation

If you use LLM-Rosetta in your research, please cite our paper:

```bibtex
@article{ding2025llmrosetta,
  title={LLM-Rosetta: A Hub-and-Spoke Intermediate Representation for Cross-Provider LLM API Translation},
  author={Ding, Peng},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2025}
}
```

## License

MIT License
