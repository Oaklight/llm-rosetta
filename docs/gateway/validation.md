---
title: Gateway Validation
---

# Gateway Validation Report

This page documents end-to-end validation of the LLM-Rosetta Gateway with real-world CLI tools and SDK test suites, serving as proof of cross-provider compatibility.

!!! info "Last updated: 2026-03-20"
    Tested with llm-rosetta v0.2.0 (+ unreleased fixes for [#56](https://github.com/Oaklight/llm-rosetta/issues/56)–[#59](https://github.com/Oaklight/llm-rosetta/issues/59))

## CLI Tool Compatibility

Five popular AI coding CLI tools were tested through the gateway. Each tool speaks a different API format — the gateway translates automatically based on the endpoint path.

| CLI Tool | API Format | Source → Target | Chat | Stream | Tool Calls | Multi-Round |
|----------|-----------|-----------------|:----:|:------:|:----------:|:-----------:|
| [Codex CLI](https://github.com/openai/codex) | OpenAI Responses | `openai_responses` → `openai_responses` | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: |
| [Kilo Code](https://kilocode.ai/) | OpenAI Chat | `openai_chat` → `openai_responses` | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: |
| [OpenCode](https://opencode.ai/) | OpenAI Chat | `openai_chat` → `openai_responses` | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: |
| [Claude Code](https://docs.anthropic.com/en/docs/claude-code) | Anthropic Messages | `anthropic` → `anthropic` | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: |
| [Gemini CLI](https://github.com/google-gemini/gemini-cli) | Google GenAI | `google` → `google` | :white_check_mark: | :warning: | — | — |

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

Gemini CLI uses the Google GenAI SDK. Basic chat works through the gateway, but the CLI's streaming protocol has compatibility nuances (returns empty responses in some cases).

- **Model**: `gemini-2.5-flash-lite`
- **Result**: Chat works. Streaming returns empty responses in CLI mode (likely JS SDK streaming protocol difference). Tool calling works via direct REST API calls (see below).

---

## SDK Test Suite (`llm_api_simple_tests`)

The [llm_api_simple_tests](https://github.com/Oaklight/llm_api_simple_tests) suite runs 5 standardized tests per provider using official SDKs. All Anthropic tests were run through the gateway with cross-provider routing.

### Anthropic SDK via Gateway

**Configuration**: `ANTHROPIC_BASE_URL=http://localhost:8765`, model `anthropic/claude-3-haiku`

| Test | Description | Status |
|------|-------------|:------:|
| `simple_query.py` | Single-turn streaming query | :white_check_mark: PASS |
| `multi_round_chat.py` | 3-round conversation (Fibonacci explanation → code → optimization) | :white_check_mark: PASS |
| `multi_round_function_calling.py` | 3-round tool calling (weather → temperature conversion → comparison) | :white_check_mark: PASS |
| `multi_round_comprehensive.py` | 3-round with image + tool calls (landmark → weather → recommendation) | :white_check_mark: PASS |
| `multi_round_image.py` | 3-round vision conversation (describe → locate → facts) | :white_check_mark: PASS |

### Google GenAI via Gateway (curl)

Multi-round tool calling was tested directly via `curl` against the gateway's Google endpoint.

| Round | Request | Model Response | Status |
|:-----:|---------|---------------|:------:|
| 1 | "What is 127 * 389?" with `calculator` tool, `mode=ANY` | `functionCall: calculator({expression: "127 * 389"})` | :white_check_mark: |
| 2 | Tool result `49403`, "add 100 to that" | `functionCall: calculator({expression: "49403 + 100"})` | :white_check_mark: |
| 3 | Tool result `49503`, `mode=AUTO` | Text: "The result is 49503." | :white_check_mark: |

Tested with both `gemini-2.5-flash-lite` and `gemini-3.1-flash-lite-preview`. Both models correctly returned function calls with `thoughtSignature` preserved.

---

## Cross-Provider Routing Combinations Tested

| Source Format | Target Provider | Verified |
|--------------|----------------|:--------:|
| OpenAI Chat → | OpenAI Responses | :white_check_mark: |
| OpenAI Responses → | OpenAI Responses | :white_check_mark: |
| Anthropic → | Anthropic (OpenRouter) | :white_check_mark: |
| Google GenAI → | Google GenAI | :white_check_mark: |

---

## Bugs Found and Fixed During Validation

| Issue | Description | Fix |
|-------|-------------|-----|
| [#56](https://github.com/Oaklight/llm-rosetta/issues/56) | OpenAI Responses streaming: missing `id`/`object`/`model` fields, incorrect event ordering | Fixed in converter |
| [#57](https://github.com/Oaklight/llm-rosetta/issues/57) | OpenAI Chat streaming: `tool_calls` missing `index` field | Fixed in converter |
| [#58](https://github.com/Oaklight/llm-rosetta/issues/58) | `stream_options` (Chat-only) leaked into Responses API requests | Removed from Responses `ir_stream_config_to_p()` |
| [#59](https://github.com/Oaklight/llm-rosetta/issues/59) | Google converter ignored tools in REST-format requests | Added fallback to top-level fields |

---

## Known Limitations

- **Gemini CLI**: Returns empty responses when streaming through the gateway — likely due to the Google GenAI JS SDK expecting a different streaming protocol than the gateway provides. Direct REST API calls work correctly.
- **Claude Code**: Requires a valid upstream API key for the configured Anthropic provider. The gateway itself is transparent — auth errors are passed through from upstream.
- **Image passthrough**: Not tested for cross-provider image routing (e.g., OpenAI Chat → Google GenAI with images). Same-provider image support works (Anthropic SDK test confirms vision through the gateway).
