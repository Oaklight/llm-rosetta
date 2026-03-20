---
title: Gateway Validation
---

# Gateway Validation Report

This page documents end-to-end validation of the LLM-Rosetta Gateway with real-world CLI tools and SDK test suites, serving as proof of cross-provider compatibility.

!!! info "Last updated: 2026-03-20"
    Tested with llm-rosetta v0.2.0 (+ unreleased fixes for [#56](https://github.com/Oaklight/llm-rosetta/issues/56)–[#62](https://github.com/Oaklight/llm-rosetta/issues/62))

## CLI Tool Compatibility

Five popular AI coding CLI tools were tested through the gateway. Each tool speaks a different API format — the gateway translates automatically based on the endpoint path.

| CLI Tool | API Format | Source → Target | Chat | Stream | Tool Calls | Multi-Round |
|----------|-----------|-----------------|:----:|:------:|:----------:|:-----------:|
| [Codex CLI](https://github.com/openai/codex) | OpenAI Responses | `openai_responses` → `openai_responses` | ✓ | ✓ | ✓ | ✓ |
| [Kilo Code](https://kilocode.ai/) | OpenAI Chat | `openai_chat` → `openai_responses` | ✓ | ✓ | ✓ | ✓ |
| [OpenCode](https://opencode.ai/) | OpenAI Chat | `openai_chat` → `openai_responses` | ✓ | ✓ | ✓ | ✓ |
| [Claude Code](https://docs.anthropic.com/en/docs/claude-code) | Anthropic Messages | `anthropic` → `anthropic` | ✓ | ✓ | ✓ | ✓ |
| [Gemini CLI](https://github.com/google-gemini/gemini-cli) | Google GenAI | `google` → `google` | ✓ | ✓ | ✓ | ✓ |

### Test Details

#### Codex CLI (OpenAI Responses API)

Codex uses the OpenAI Responses API with `wire_api = "responses"`. Multi-round tool calling was verified with the gateway routing to upstream OpenAI.

- **Model**: `gpt-4.1-nano`
- **Test**: Prompted to list directory contents → model issued `shell` tool call → tool result sent back → model summarized output
- **Result**: 2 rounds completed successfully, streaming SSE events correctly formatted

#### Kilo Code (OpenAI Chat Completions)

Kilo uses the OpenAI Chat Completions API. The gateway translates `openai_chat` → `openai_responses` format for upstream.

- **Model**: `gpt-4.1-nano` (via `rosetta/gpt-4.1-nano`)
- **Test**: `kilo run --auto -m rosetta/gpt-4.1-nano "List the files in the current directory"`
- **Result**: Round 1 (26 chunks, 3.80s) → tool call `ls -la` → Round 2 (23 chunks, 6.48s) → text summary

#### OpenCode (OpenAI Chat Completions)

OpenCode also uses the OpenAI Chat Completions API with the Vercel AI SDK.

- **Model**: `gpt-4.1-nano` (via `rosetta/gpt-4.1-nano`)
- **Test**: `opencode run --model rosetta/gpt-4.1-nano "List the files in the current directory"`
- **Result**: Round 1 (21 chunks, 3.04s) → `bash` tool call → Round 2 (103 chunks, 6.12s) → file listing

#### Claude Code (Anthropic Messages API)

Claude Code uses the Anthropic Messages API. Tested with OpenRouter as the upstream Anthropic provider.

- **Model**: `anthropic/claude-haiku-4.5`
- **Result**: Chat, streaming, tool calls, and multi-round all work. Requires a valid upstream API key.

#### Gemini CLI (Google GenAI API)

Gemini CLI uses the `@google/genai` JS SDK (v1.30.0). Full compatibility achieved after fixing camelCase tool definition parsing (#61) and streaming tool call chunk format (#62).

- **Model**: `gemini-2.5-flash-lite`
- **Configuration**: `GOOGLE_GEMINI_BASE_URL=http://localhost:8765 GEMINI_API_KEY=dummy`
- **Result**: Chat, streaming, and tool calling all work. Headless mode (`-p`) with tool calls exits cleanly; interactive mode displays full tool call round-trips.

---

## Integration Test Suite (`tests/integration/`)

The integration test suite validates all four converter pipelines with real API calls using both official SDKs and direct REST. Each test covers non-streaming, streaming, tool calls, round-trip conversions, and multi-turn conversations.

### Results Summary

| Test Suite | Tests | Result |
|-----------|:-----:|:------:|
| Google GenAI SDK | 5 | **5/5** ✓ |
| Google GenAI REST | 5 | **5/5** ✓ |
| OpenAI Chat SDK | 7 | **7/7** ✓ |
| OpenAI Chat REST | 7 | **7/7** ✓ |
| OpenAI Responses SDK | 4 | **4/4** ✓ |
| OpenAI Responses REST | 4 | **4/4** ✓ |
| Anthropic SDK | 6 | **6/6** ✓ |
| Anthropic REST | 6 | **6/6** ✓ |
| **Total** | **44** | **44/44** ✓ |

### Test Coverage Per Suite

Each SDK/REST test suite covers:

| Test | OpenAI Chat | OpenAI Responses | Anthropic | Google GenAI |
|------|:-----------:|:----------------:|:---------:|:------------:|
| Non-stream basic text | ✓ | ✓ | ✓ | ✓ |
| Non-stream with image | ✓ | — | — | — |
| Non-stream with tool calls | ✓ | ✓ | ✓ | ✓ |
| Streaming text | ✓ | — | ✓ | — |
| Streaming with tool calls | ✓ | — | ✓ | — |
| Request round-trip | ✓ | ✓ | ✓ | ✓ |
| Response round-trip | ✓ | ✓ | ✓ | ✓ |
| Multi-turn conversation | — | — | — | ✓ |

---

## SDK Test Suite (`llm_api_simple_tests`)

The [llm_api_simple_tests](https://github.com/Oaklight/llm_api_simple_tests) suite runs 5 standardized tests per provider using official SDKs. All Anthropic tests were run through the gateway with cross-provider routing.

### Anthropic SDK via Gateway

**Configuration**: `ANTHROPIC_BASE_URL=http://localhost:8765`, model `anthropic/claude-3-haiku`

| Test | Description | Status |
|------|-------------|:------:|
| `simple_query.py` | Single-turn streaming query | ✓ PASS |
| `multi_round_chat.py` | 3-round conversation (Fibonacci explanation → code → optimization) | ✓ PASS |
| `multi_round_function_calling.py` | 3-round tool calling (weather → temperature conversion → comparison) | ✓ PASS |
| `multi_round_comprehensive.py` | 3-round with image + tool calls (landmark → weather → recommendation) | ✓ PASS |
| `multi_round_image.py` | 3-round vision conversation (describe → locate → facts) | ✓ PASS |

### Google GenAI via Gateway (curl)

Multi-round tool calling was tested directly via `curl` against the gateway's Google endpoint.

| Round | Request | Model Response | Status |
|:-----:|---------|---------------|:------:|
| 1 | "What is 127 * 389?" with `calculator` tool, `mode=ANY` | `functionCall: calculator({expression: "127 * 389"})` | ✓ |
| 2 | Tool result `49403`, "add 100 to that" | `functionCall: calculator({expression: "49403 + 100"})` | ✓ |
| 3 | Tool result `49503`, `mode=AUTO` | Text: "The result is 49503." | ✓ |

Tested with both `gemini-2.5-flash-lite` and `gemini-3.1-flash-lite-preview`. Both models correctly returned function calls with `thoughtSignature` preserved.

---

## Cross-Provider Routing Combinations Tested

| Source Format | Target Provider | Verified |
|--------------|----------------|:--------:|
| OpenAI Chat → | OpenAI Responses | ✓ |
| OpenAI Responses → | OpenAI Responses | ✓ |
| Anthropic → | Anthropic (OpenRouter) | ✓ |
| Google GenAI → | Google GenAI | ✓ |

---

## Bugs Found and Fixed During Validation

| Issue | Description | Fix |
|-------|-------------|-----|
| [#56](https://github.com/Oaklight/llm-rosetta/issues/56) | OpenAI Responses streaming: missing `id`/`object`/`model` fields, incorrect event ordering | Fixed in converter |
| [#57](https://github.com/Oaklight/llm-rosetta/issues/57) | OpenAI Chat streaming: `tool_calls` missing `index` field | Fixed in converter |
| [#58](https://github.com/Oaklight/llm-rosetta/issues/58) | `stream_options` (Chat-only) leaked into Responses API requests | Removed from Responses `ir_stream_config_to_p()` |
| [#59](https://github.com/Oaklight/llm-rosetta/issues/59) | Google converter ignored tools in REST-format requests | Added fallback to top-level fields |
| [#61](https://github.com/Oaklight/llm-rosetta/issues/61) | Google camelCase `functionDeclarations` not parsed; only first declaration extracted | Handle both casings; extract all declarations |
| [#62](https://github.com/Oaklight/llm-rosetta/issues/62) | Google streaming tool calls split into two chunks (name-only + args-only) | Defer `tool_call_start`, emit complete `function_call` on `tool_call_delta` |

---

## Known Limitations

- **Claude Code**: Requires a valid upstream API key for the configured Anthropic provider. The gateway itself is transparent — auth errors are passed through from upstream.
- **Image passthrough**: Not tested for cross-provider image routing (e.g., OpenAI Chat → Google GenAI with images). Same-provider image support works (Anthropic SDK test confirms vision through the gateway).
- **Gemini CLI headless mode**: In non-interactive (`-p`) mode, tool call results may not be displayed by the CLI, though the round-trip completes successfully. This is a Gemini CLI display behavior, not a gateway issue.
