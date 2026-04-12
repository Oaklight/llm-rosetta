---
title: Gateway Validation
---

# Gateway Validation Report

This page summarizes end-to-end validation of the LLM-Rosetta Gateway (via [argo-proxy](https://github.com/Oaklight/argo-proxy)) with real-world CLI tools, SDK test suites, and cross-format routing.

!!! info "Last updated: 2026-04-12"
    Tested with llm-rosetta v0.5.0 and argo-proxy v3.0.0b7

## Cross-Format Routing Matrix

Argo-proxy routes Claude models to the native Anthropic upstream and all other models (GPT, Gemini) to the OpenAI Chat upstream. The gateway automatically translates between auth credential formats (`Authorization: Bearer`, `x-api-key`, `x-goog-api-key`).

### Text Generation (9/9 ✓)

| Client (API Format) | Claude Model | GPT Model | Gemini Model |
|---------------------|:------------:|:---------:|:------------:|
| **Claude Code** (Anthropic) | ✓ passthrough | ✓ anthropic→openai_chat | ✓ anthropic→openai_chat |
| **Codex CLI** (OpenAI Responses) | ✓ responses→anthropic | ✓ passthrough | ✓ passthrough |
| **Gemini CLI** (Google GenAI) | ✓ google→anthropic | ✓ google→openai_chat | ✓ google→openai_chat |

### Image Understanding (9/9 ✓)

| Client (Image Method) | Claude Model | GPT Model | Gemini Model |
|-----------------------|:------------:|:---------:|:------------:|
| **Codex CLI** (`-i` flag) | ✓ | ✓ | ✓ |
| **Claude Code** (Read tool) | ✓ | ✓¹ | ✓ |
| **Gemini CLI** (read_file tool) | ✓ | ✓ | ✓ |

¹ Requires GPT-5.4+; GPT-4.1-nano may fail to interpret Read tool image results.

For reproducible commands and detailed test procedures, see [CLI Cross-Format Testing](validation-cli.md).

---

## Integration Test Summary (22/22 ✓)

| Test Suite | Tests | Result |
|-----------|:-----:|:------:|
| Google GenAI SDK | 5 | **5/5** ✓ |
| Google GenAI REST | 6 | **6/6** ✓ |
| OpenAI Chat SDK | 5 | **5/5** ✓ |
| OpenAI Responses SDK | 3 | **3/3** ✓ |
| Anthropic REST | 3 | **3/3** ✓ |
| **Total** | **22** | **22/22** ✓ |

For SDK test details and curl-based validation, see [SDK & Integration Tests](validation-sdk.md).

---

## Bugs Found and Fixed During Validation

| Issue | Description | Fix |
|-------|-------------|-----|
| [#56](https://github.com/Oaklight/llm-rosetta/issues/56) | OpenAI Responses streaming: missing `id`/`object`/`model` fields | Fixed in converter |
| [#57](https://github.com/Oaklight/llm-rosetta/issues/57) | OpenAI Chat streaming: `tool_calls` missing `index` field | Fixed in converter |
| [#58](https://github.com/Oaklight/llm-rosetta/issues/58) | `stream_options` leaked into Responses API requests | Removed from `ir_stream_config_to_p()` |
| [#59](https://github.com/Oaklight/llm-rosetta/issues/59) | Google converter ignored tools in REST-format requests | Added fallback to top-level fields |
| [#61](https://github.com/Oaklight/llm-rosetta/issues/61) | Google camelCase `functionDeclarations` not parsed | Handle both casings; extract all |
| [#62](https://github.com/Oaklight/llm-rosetta/issues/62) | Google streaming tool calls split into two chunks | Defer `tool_call_start`, emit on delta |
| — | Anthropic `input_schema` missing `type` for no-param tools | Default to `{"type": "object"}` |

---

## Known Limitations

- **Claude Code + non-Claude models**: Claude Code passes image data via its Read tool result. Some models (e.g., GPT-4.1-nano) may not interpret base64 image content from tool results correctly. Use GPT-5.4+ for reliable image understanding.
- **Gemini CLI headless mode**: In non-interactive (`-p`) mode, Gemini CLI has no `--image` flag. Images can be read via the built-in `read_file` tool by referencing a file path within the workspace.
- **Auth transparency**: The gateway passes upstream auth errors through. A 401 from upstream means the API key is invalid for the target provider, not a gateway issue.
