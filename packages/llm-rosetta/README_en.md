<p align="center">
  <img src="https://raw.githubusercontent.com/Oaklight/llm-rosetta/master/design/logo/out/rosetta-icon-github.svg" alt="LLM-Rosetta" width="96">
</p>

<h1 align="center">llm-rosetta</h1>

<p align="center">
  <a href="https://pypi.org/project/llm-rosetta/"><img src="https://img.shields.io/pypi/v/llm-rosetta?color=green" alt="PyPI"></a>
  <a href="https://pypi.org/project/llm-rosetta/"><img src="https://img.shields.io/pypi/pyversions/llm-rosetta" alt="Python"></a>
  <a href="https://github.com/Oaklight/llm-rosetta/actions/workflows/ci.yml"><img src="https://github.com/Oaklight/llm-rosetta/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
</p>

<p align="center">
  <a href="https://llm-rosetta.readthedocs.io/en/latest/getting-started/library-quickstart/"><img src="https://img.shields.io/badge/docs-English-2f6fdd" alt="Docs (English)"></a>
  <a href="https://llm-rosetta.readthedocs.io/zh-cn/latest/getting-started/library-quickstart/"><img src="https://img.shields.io/badge/docs-%E4%B8%AD%E6%96%87-2f6fdd" alt="Docs (中文)"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-green.svg" alt="MIT"></a>
</p>

<p align="center">
  English | <a href="https://github.com/Oaklight/llm-rosetta/blob/master/packages/llm-rosetta/README_zh.md">中文</a>
</p>

Unified LLM API adapter: bidirectional conversion between OpenAI Chat
Completions, Open Responses, Anthropic Messages, Google generateContent and
Google Interactions through a central IR (Intermediate Representation).

This is the **core distribution**. On Python 3.11+ it has no third-party runtime
dependency; on Python 3.10, its only third-party runtime dependency is
`typing_extensions`, which backports newer typing features. It never imports the gateway.

## Three Levels and a Toolbox

| | Toolbox | Raw converter | Pipeline |
|---|---|---|---|
| **Use case** | Convert one IR value or tool | Translate one leg inside your own loop | Source → IR → target in one call |
| **Entry point** | `tool_ops`, `transforms`, `capabilities` | `ChatCompletionsConverter()` | `convert()` / `ConversionPipeline` |
| **Provider call** | You | You | You |

The gateway distribution adds the third level — a self-hosted HTTP proxy that
owns the provider call, routing, keys and admin console.

All three levels share the same IR and converters, and the toolbox operates on
that same IR. `tool_ops` converts one tool definition, choice, call or result
between a provider dialect and IR; `transforms` and `capabilities` apply a
single field transformation or capability pass without building a converter.
The pipeline resolves both shims, so a target shim's defaults can make its body
differ from the raw converter's — Anthropic's default `cache_control` breakpoint
is one example.

## Install

```bash
pip install llm-rosetta
```

Requires **Python ≥ 3.10**.

## Quick Start

Raw converter — translate one leg, then call the provider yourself:

```python
from llm_rosetta import ChatCompletionsConverter, MessagesConverter

openai_request = {
    "model": "gpt-4o",
    "messages": [{"role": "user", "content": "Hello!"}],
}

ir = ChatCompletionsConverter().request_from_provider(openai_request)
anthropic_request, warnings = MessagesConverter().request_to_provider(ir)

# After calling the Anthropic API with your own client:
ir_response = MessagesConverter().response_from_provider(response)
```

Pipeline — source → IR → target in one call, with shim resolution:

```python
from llm_rosetta import convert, detect_provider

detect_provider(openai_request)                          # 'chat_completions'
anthropic_request = convert(openai_request, target_provider="messages")
```

Streaming conversion is stateful; a raw converter keeps a `StreamContext` across
chunks. See [Streaming](https://llm-rosetta.readthedocs.io/en/latest/guide/streaming/).

## Use the IR as Your Payload

The IR is typed and provider-neutral. Keep conversation history, routing state
or evaluation data in IR, and render it to a provider's dialect only at the call
boundary. Switching providers changes the renderer, not your application's data
model. Because the payload stays in IR, supporting another API format does not
add a format-specific branch to your application.

## Supported Formats

| Model type | Formats |
|------------|---------|
| LLM (chat) | `chat_completions`, `open_responses`, `messages`, `google_generate`, `google_interactions` |
| Embeddings | `openai`, `cohere`, `jina`, `voyage`, `google_generate`, `minimax` |
| Rerank | `jina`, `cohere`, `voyage` |
| Decisions | `systemone`, `openai_decisions` |

Every format converts both to and from the IR, for requests and responses; chat
formats also convert streamed chunks. `open_responses` is the vendor-neutral
Open Responses spec; the OpenAI profile (`openai--open_responses` shim) adds its
`resp_` IDs, `store: true` and echo fields.

Vendor differences are declared by 39 provider shims across 19 vendor prefixes,
so the converters stay generic.

## Documentation

- [Full README](https://github.com/Oaklight/llm-rosetta#readme)
- [Documentation](https://llm-rosetta.readthedocs.io/en/latest/)
- [Gateway distribution](https://pypi.org/project/llm-rosetta-gateway/) — add
  the HTTP proxy, routing, keys and admin console

## License

MIT — see the [LICENSE](https://github.com/Oaklight/llm-rosetta/blob/master/LICENSE)
file for details.
