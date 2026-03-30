---
title: Changelog
---

# Changelog

All notable changes to LLM-Rosetta are documented here. This project follows [Keep a Changelog](https://keepachangelog.com/) conventions.

## Unreleased

### Refactored

- **Extracted per-provider constants for reason mappings and magic values** (#64): Inline reason mapping dicts, SSE event type string literals, status-to-reason conditional logic, and ID generation patterns across all 4 converters (Anthropic, Google GenAI, OpenAI Chat, OpenAI Responses) are now centralized in per-provider `_constants.py` modules. Includes `AnthropicEventType` and `ResponsesEventType` classes for namespaced event type constants, `REASON_FROM_PROVIDER` / `REASON_TO_PROVIDER` dicts for each provider, and `generate_tool_call_id()` / `generate_message_id()` helper functions

### Fixed

- **Anthropic streaming missing `refusal` reason mapping**: The streaming `reason_map` was missing the `refusal` entry present in the non-streaming path, causing Anthropic refusal stop reasons to be silently dropped during streaming. Fixed as a side effect of the constants extraction (#64) — both paths now share the same `ANTHROPIC_REASON_FROM_PROVIDER` dict

### Added

- **Constants validation tests**: 39 new tests across 4 `test_constants.py` files verifying that all reason mapping values are valid IR finish reasons, mapping coverage is complete, event type constants are well-formed, and ID generation produces correct formats
- **Finish reason mapping test coverage**: 38 tests validating reason mapping correctness as a safety net for the constants refactoring

## v0.2.6 — 2026-03-29

### Fixed

- **Chat Completions tool message ordering after Responses API conversion** *([@caidao22](https://github.com/caidao22))*: Codex CLI interleaves `function_call_output` with other items (e.g. user warnings) in Responses API format — valid there since items match by `call_id`. But after IR → Chat Completions conversion, the interleaved messages break the OpenAI Chat API constraint that `role: "tool"` messages must immediately follow their `assistant` `tool_calls`, causing upstream 400 errors. Added `_reorder_tool_messages()` post-processing in `OpenAIChatMessageOps.ir_messages_to_p()` that groups tool responses back to their corresponding assistant messages
- **Orphaned `tool_choice`/`tool_config` stripped when no tools defined** *([@caidao22](https://github.com/caidao22))*: Codex context compaction can drop all tool definitions while keeping `tool_choice` (e.g. `"auto"`), causing upstream APIs to reject with *"tool_choice is set but no tools are provided"*. Added `strip_orphaned_tool_config()` in all four converters — part of the same Codex compaction fix family as `fix_orphaned_tool_calls_ir` (orphaned tool_call/result pairing) and `_reorder_tool_messages` (tool message ordering). Also extended `fix_orphaned_tool_calls_ir` to Google GenAI converter for completeness (#87)
- **Stream event ordering**: `UsageEvent` is now emitted before `FinishEvent` in all four provider converters (OpenAI Chat, OpenAI Responses, Anthropic, Google GenAI). Previously `FinishEvent` was processed first, causing `response.completed` to carry `output_tokens=0` — downstream consumers (e.g. Codex token tracking) saw stale usage data. For cross-chunk scenarios (OpenAI Chat sends `finish_reason` and `usage` in separate chunks), `FinishEvent` now defers `response.completed` to `StreamEndEvent` which merges any pending usage
- **Parallel tool calls merged into one in Anthropic/Google → Chat streaming**: Anthropic and Google GenAI `stream_response_from_provider` emitted `ToolCallStartEvent` and `ToolCallDeltaEvent` without `tool_call_index`. When routing to Chat Completions, all parallel tool calls defaulted to index 0, causing the client SDK to merge them into a single call. Anthropic now derives `tool_call_index` from `context._tool_call_order` position; Google computes it from registration order in context (#88, #89)
- **Missing `id` field on Responses `function_call` output**: Non-streaming `response_to_provider` was missing the `id` field on `function_call` output items. Streaming used a synthetic `fc_` prefix that could leak into IR via `p_tool_call_to_ir` fallback path. Unified both paths to use `call_id` directly as `id` (no prefix)
- **Responses streaming `item_id` and empty `tool_call_id` resolution** *([@caidao22](https://github.com/caidao22))*: Added `item_id` tracking to `StreamContext` (`tool_call_item_id_map`, bidirectional mapping). Responses `stream_response_to_provider` now emits `item.id` on `output_item.added` and `item_id` (not `call_id`) on `function_call_arguments.delta/done` events. Defense-in-depth: resolves empty `tool_call_id` by `tool_call_index` via context (#86)
- **Non-function tool names mangled with type prefix** *([@caidao22](https://github.com/caidao22))*: Non-function IR tool definitions (e.g. `type="custom"`, `name="apply_patch"`) were converted with a type prefix (`custom_apply_patch`), breaking tool_call matching since the client expects the original name. Both OpenAI Chat and Responses converters now use `ir_tool["name"]` directly (#84)

## v0.2.5 — 2026-03-23

### Fixed

- **Anthropic `input_schema` missing `type` for parameterless tools**: MCP tools with no parameters produce `input_schema: {}`, but Anthropic requires `"type"` to be present. Now defaults to `{"type": "object"}` when the schema dict lacks a `type` field — fixes `tools.0.custom.input_schema.type: Field required` errors when routing Google GenAI or OpenAI Responses tool calls to Anthropic upstream
- **Google GenAI camelCase field handling across the full converter stack**: Gemini CLI and the Google REST API use camelCase (`inlineData`, `fileData`, `mimeType`, `fileUri`, `functionCall`, `functionResponse`, `finishReason`, `usageMetadata`, `responseMimeType`, `responseSchema`, `thinkingConfig`, `maxOutputTokens`, `stopSequences`, etc.), but the converter only accepted snake_case. All P→IR methods in content_ops, config_ops, tool_ops, message_ops, and converter now accept both conventions; all IR→P methods now output camelCase for REST API compatibility
- **Image/audio/file data lost during Google→IR conversion**: `p_part_to_ir` checked for `inline_data` (snake_case) but Gemini CLI sends `inlineData` (camelCase) — binary content was silently dropped with a `不支持的Part类型` warning. Fixed by normalizing camelCase keys at the dispatch entry point
- **Cross-format image conversion failure (Google → OpenAI/Anthropic)**: Google's `p_image_to_ir` produces `ImagePart` with top-level `data` + `media_type` fields, but OpenAI Chat, Anthropic, and OpenAI Responses `ir_image_to_p` only checked `image_url` and nested `image_data` — threw `ValueError`. All three target converters now handle top-level fields as a fallback path (#68)
- **Google GenAI tool_call_id reconciliation**: Google `functionCall` has no ID field, so UUIDs are generated during P→IR. But Gemini CLI assigns its own IDs to `functionResponse` (format: `name_timestamp_index`), creating a mismatch. New `_reconcile_tool_call_ids` method matches tool results to tool calls by function name, fixing orphaned tool_call errors
- **tool_call_id exceeds OpenAI 40-character limit**: Generated IDs used `call_{name}_{8hex}` format — MCP tool names like `mcp_toolregistry-hub-server_datetime-now` produced 54-char IDs. Shortened to `call_{24hex}` (fixed 29 chars)
- **Google→IR role mapping for tool results**: `functionResponse` parts produced `role: "user"` IR messages, so `fix_orphaned_tool_calls_ir` (which checks `role: "tool"`) couldn't detect them. Now separates `functionResponse` into `role: "tool"` messages with explicit `"tool": "user"` in `_IR_TO_GOOGLE_ROLE`
- **Mixed content message ordering**: When a Google message contains both `functionResponse` and `inlineData`, the content parts were emitted before tool results, breaking OpenAI's required `assistant(tool_calls) → tool(response)` ordering. Tool results now precede content parts in the split
- **Google built-in tools (googleSearch, codeExecution)**: `p_tool_definition_to_ir` now returns `None` for tool entries without a `name` field; converter skips them instead of producing empty `function.name` errors
- **Gateway: Starlette `on_shutdown` deprecation**: Replaced deprecated `on_shutdown` parameter with `lifespan` async context manager — fixes compatibility with Starlette 0.38+ which removed `on_shutdown`/`on_startup`

### Added

- **StreamContext**: `get_tool_call_args()` and `get_pending_tool_calls()` methods for querying accumulated tool call state during streaming

### Changed

- **`BaseToolOps.p_tool_definition_to_ir` return type**: Now `ToolDefinition | list[ToolDefinition] | None` to support unconvertible tool entries

### Added (Documentation)

- **Provider & CLI Compatibility Matrix**: New guide page documenting real-world issues found during live integration testing with Gemini CLI, Claude Code, and OpenCode through format-converting proxies

## v0.2.4 — 2026-03-22

### Added

- **`fix_orphaned_tool_calls()` utilities**: Public functions in `converters/openai_chat/tool_ops.py`, `converters/openai_responses/tool_ops.py`, and `converters/anthropic/tool_ops.py` that detect mismatched tool calls/results and fix them bidirectionally — injecting synthetic placeholder results for orphaned calls **and** removing orphaned results without matching calls. OpenAI (Chat & Responses) and Anthropic strictly require this pairing (return 400 otherwise); only Google Gemini is lenient. Automatically applied at the IR level during `request_to_provider()` for all strict-pairing converters; emits `WARNING`-level log when orphaned tool calls or results are detected (#82, #84)

### Fixed

- **Anthropic→IR role normalization for `tool_result` messages**: Anthropic places `tool_result` blocks in `role: "user"` messages, but IR uses `role: "tool"` (like OpenAI). The Anthropic converter now normalizes pure `tool_result` user messages to `role: "tool"`, and splits mixed `tool_result` + text messages into separate `role: "tool"` and `role: "user"` IR messages. This fixes `fix_orphaned_tool_calls_ir()` failing to detect answered tool calls in cross-format conversions (e.g. Anthropic → OpenAI Chat) (#84)
- **OpenAI Responses→IR role normalization for `function_call_output` items**: `function_call_output` and `mcp_call_output` items were grouped into `role: "user"` IR messages, but IR uses `role: "tool"` for tool results. The Responses converter now groups these items into `role: "tool"` messages, fixing `fix_orphaned_tool_calls_ir()` failing to detect answered tool calls when converting Responses → other formats (e.g. Responses → OpenAI Chat) (#84)

### Added (Documentation)

- **Provider Dialect Differences guide**: New section in the Converters guide (EN + ZH) documenting tool schema sanitization, orphaned tool call handling, and Google camelCase/snake_case differences

## v0.2.3 — 2026-03-22

### Fixed

- **Tool schema sanitization applied to all converters**: `_sanitize_schema()` was previously only called in the OpenAI Chat converter. Google GenAI, OpenAI Responses, and Anthropic converters now also sanitize tool parameter schemas before sending to upstream, preventing rejections from strict endpoints like Vertex AI (#80)
- **Non-standard `ref` and `$schema` keywords stripped**: OpenCode's built-in tools use a bare `ref` field (without `$` prefix) and `$schema` at the top level, both rejected by Vertex AI. Added to the unsupported keywords blocklist (#80)
- **`$ref`/`$defs` resolved by inlining**: JSON Schema `$ref` references are now resolved by inlining the referenced definition from `$defs`/`definitions`, and both keys are removed from the output. Supports nested and chained references (#80)
- **Streaming tool call arguments not accumulated**: OpenAI Chat, Anthropic, and Google GenAI converters registered tool calls in `StreamContext` but never called `append_tool_call_args()` to accumulate argument deltas during streaming. This caused tool call arguments to arrive empty at upstream (e.g., MCP tools returning `'query' is a required property`). Only the OpenAI Responses converter was correct (#81)
- **OpenAI Chat streaming tool call ID resolution**: Delta-only chunks (carrying `index` but no `id`) produced an empty-string `tool_call_id`. Now resolves the effective ID from `StreamContext._tool_call_order` using the chunk index (#81)

### Changed

- **`sanitize_schema` extracted to `converters/base/tools.py`**: The schema sanitization utility (previously `_sanitize_schema` private to `openai_chat/tool_ops.py`) is now a public shared function in `converters/base/tools.py`, exported via `converters.base`. All 4 converter `tool_ops.py` files import from the shared location instead of cross-importing from `openai_chat` (#66)

## v0.2.2 — 2026-03-22

### Fixed

- **Missing `content_block_stop` in Anthropic SSE output**: When converting OpenAI Chat streaming responses to Anthropic SSE format, `content_block_stop` events were not emitted before `message_delta`, causing Claude Code to silently discard response content. The Anthropic converter now emits `content_block_stop` for any open content block when processing a `FinishEvent` (#77)
- **Upstream preflight chunk misinterpreted as stream end**: Argo API sends a preflight chunk with `choices: []` and empty `id`/`model` before actual content. The OpenAI Chat converter now only treats empty-choices chunks as stream-end after the stream has actually started (`context.is_started` guard) (#77)

## v0.2.1 — 2026-03-20

### Added

- **Gateway request/response body logging**: configurable debug logging with colorized output, body sanitization and truncation — enable via config (`"debug": {"verbose": true, "log_bodies": true}`), env vars (`LLM_ROSETTA_VERBOSE`, `LLM_ROSETTA_LOG_BODIES`), or `--verbose` CLI flag
- **Google `output_format="rest"` for `request_to_provider()`**: pass `output_format="rest"` to get a REST API–ready request body with `tools`/`tool_config` at top level and generation params wrapped in `generationConfig` — eliminates the need for manual SDK→REST fixups

### Changed

- **Gateway modularization**: split `app.py` (1057 lines) into `proxy.py` (proxy engine, SSE handling, upstream requests), `cli.py` (CLI entry point, argparse, subcommands), and a slimmed `app.py` (route handlers, app factory, ~210 lines)
- **Moved Google REST body fixup to core**: `_fixup_google_body()` logic moved from `gateway/proxy.py` into `GoogleGenAIConverter._to_rest_body()`, removing duplicated SDK→REST transforms from the gateway and all 6 REST examples

### Fixed

- OpenAI Responses streaming: added missing `id`/`object`/`model` fields to `response.completed`, `output_index`/`content_index` to text delta events, and proper lifecycle events (`output_item.added`, `content_part.added`, `content_part.done`, `output_item.done`) (#56)
- OpenAI Chat streaming: `tool_calls` entries now always include the required `index` field, defaulting to `0` when not explicitly provided by the upstream IR event (#57)
- OpenAI Chat streaming: usage-only chunk now includes `"choices": []` to satisfy clients that validate every `chat.completion.chunk` must contain a `choices` array (#55)
- `stream_options` (Chat Completions-only field) no longer leaks into OpenAI Responses API requests — the Responses converter's `ir_stream_config_to_p()` was incorrectly emitting `stream_options`, causing upstream rejection when Chat-format clients (Kilo, OpenCode) were proxied to the Responses API (#58)
- Google GenAI converter now handles tools and tool_config in REST-format requests (top-level fields) in addition to SDK format (`config.tools`) — previously only SDK format was recognized, silently stripping tool definitions from gateway-proxied requests (#59)
- Google camelCase `functionDeclarations` not parsed: `p_tool_definition_to_ir()` now handles both `functionDeclarations` (camelCase/REST) and `function_declarations` (snake_case/SDK), and extracts all declarations instead of only the first. Also added camelCase support for `functionCallingConfig`/`allowedFunctionNames` and `toolConfig` in request parsing — fixes Gemini CLI tool calling through the gateway (#61)
- Google streaming tool calls split into two chunks: `stream_response_to_provider()` now defers `tool_call_start` and emits the complete `function_call` (name + args) in a single chunk on `tool_call_delta`, matching the Google API's native format (#62)

## v0.2.0 — 2026-03-18

### Added

- **Standalone API test scripts** (`llm_api_simple_tests/`): 20 test scripts (5 per provider) using official SDKs directly, covering simple query, multi-round chat, image, function calling, and comprehensive scenarios — added as a git submodule from [Oaklight/llm_api_simple_tests](https://github.com/Oaklight/llm_api_simple_tests)
- **LLM-Rosetta Gateway**: REST gateway application for cross-provider HTTP proxying
- CLI entry point (`llm-rosetta-gateway`) and package structure for the gateway
- Gateway config auto-discovery at `./config.jsonc`, `~/.config/llm-rosetta-gateway/config.jsonc`, `~/.llm-rosetta-gateway/config.jsonc`
- `--edit` / `-e` flag to open config file in `$EDITOR` (falls back to nano/vi/vim)
- `--version` / `-V` flag showing current version
- ASCII art startup banner with `--no-banner` to suppress
- `add provider <name>` subcommand for adding provider entries to config (with `--api-key`, `--base-url` flags or interactive prompts; known providers auto-fill defaults)
- `add model <name>` subcommand for adding model routing entries (with `--provider` flag or interactive prompt)
- **Gateway providers module** (`providers.py`): centralized provider definitions with auth-header builders, URL templates, default base URLs, and API key env-var names
- **API key rotation**: round-robin `KeyRing` for comma-separated API keys per provider
- **Proxy support**: global `server.proxy` and per-provider `proxy` config for HTTP/SOCKS proxies; CLI `--proxy` flag overrides config
- Makefile `test-integration` target using `proxychains` (if available) for integration tests
- `init` subcommand to create a template `config.jsonc` at the XDG default location (`~/.config/llm-rosetta-gateway/`)
- **Model listing endpoints**: `GET /v1/models` (compatible with both OpenAI and Anthropic SDKs) and `GET /v1beta/models` (Google GenAI SDK format) — enables `client.models.list()` across all three SDKs (#54)

### Changed

- Bumped minimum Python to 3.10+; migrated to stdlib `typing` (removed `typing_extensions`)
- Applied `ruff` formatter across the entire codebase
- Updated Makefile with `lint`, `test`, and `build` targets
- Added `ty` (type checker) configuration
- Configured `ruff` lint rules (`E`, `F`, `UP`) in `pyproject.toml`; ignore `UP007` (Union syntax) and `E501` (line length)
- Modernized typing imports across `src/`, `tests/`, `examples/`, and `scripts/` — replaced `typing.Dict`, `List`, `Tuple`, `Optional`, `Type` with stdlib builtins

### Fixed

- Streaming crash with Anthropic provider when usage tokens are `null` — `TypeError: NoneType + int` in all converters (replaced `.get("*_tokens", 0)` with `.get("*_tokens") or 0`)
- Gateway provider `base_url` validation — fail early with clear error on config typos like `https:example.com` (missing `//`)
- Added `socksio` to gateway dependencies for SOCKS proxy support (`httpx[socks]`)
- Added missing `__init__.py` for `types` package
- Updated `git clone` URL from `llm-rosetta` to `llm-rosetta` in documentation
- Resolved all `ty` type checker diagnostics in `src/` (31 → 0):
    - Fixed `is_part_type()` TypeGuard narrowing — replaced with specific type guard functions (`is_text_part`, etc.)
    - Added missing TypedDict fields: `provider_metadata` on `TextPart`/`ReasoningPart`, `file_id` on `ImagePart`/`FilePart`
    - Fixed `IRRequest.messages` type from `Required[Message]` to `Required[Iterable[Message]]`
    - Used `cast()` to bridge `dict[str, Any]` intermediates to TypedDict return types
    - Fixed dict literal type inference conflicts in converter response builders
- Resolved all `ty` type checker diagnostics in `tests/` (1506 → 0):
    - Added `cast()` wrappers on dict literals passed to functions expecting TypedDict parameters (`GenerationConfig`, `IRRequest`, `IRResponse`, `ToolDefinition`, `ToolChoice`, etc.)
    - Narrowed `Message | ExtensionItem` union results with `cast(list[Any], ...)` or `cast(Message, ...)`
    - Converted `Iterable` content fields to `list` for subscript and `len()` access
    - Added `assert ... is not None` guards before subscripting optional return types
    - Fixed `FinishReason` from bare string to TypedDict form `{"reason": "stop"}`
    - Fixed `IRResponse.object` literal from `"chat.completion"` to `"response"`
- Resolved all `ruff` lint violations in `src/` and `tests/` (UP035 deprecated imports, F401 unused imports)
- Google `thought_signature` preservation through gateway round-trips — newer Google models require `thoughtSignature` echoed back in function call parts; the gateway now caches `provider_metadata` (including `thought_signature`) keyed by `tool_call_id` and re-injects it on subsequent requests for both streaming and non-streaming modes (#51)
- OpenAI Responses converter now handles all 3 `input` formats: bare string (`"input": "hello"`), shorthand list (`[{"role": "user", "content": "hi"}]`), and structured list — previously only the structured format was supported, causing the OpenAI Python SDK's shorthand items to be silently dropped and producing empty IR messages when cross-converting to Anthropic or Google providers

---

## 2026-03-15 — Rebrand to LLM-Rosetta

### Changed

- **Project renamed from LLM-Rosetta to LLM-Rosetta** across all code, docs, and configuration
- Package renamed from `llm-rosetta` to `llm_rosetta`; `pyproject.toml` updated accordingly
- Documentation fully rewritten with Zensical for both English (`docs_en`) and Chinese (`docs_zh`)
- README (EN/ZH) updated with new branding, badges, and `pyproject.toml` metadata

---

## 2026-03-06 — Streaming & StreamContext

### Added

- **`StreamContext`** for stateful stream chunk processing across all 4 providers
- `stream_response_from_provider()` and `stream_response_to_provider()` methods on all converters
- `accumulate_stream_to_assistant_message()` helper function
- Stream abstract methods (`stream_response_to_provider`, `stream_response_from_provider`) added to `BaseConverter`
- 4 new IR stream event types: `StreamStart`, `StreamEnd`, `ContentBlockStart`, `ContentBlockEnd`
- `ReasoningDeltaEvent` and `tool_call_index` field on IR stream types
- Cross-provider streaming examples for all provider pairs (SDK and REST variants)
- Local file cache and retry logic for image downloads in examples

### Changed

- Stream method signatures updated with optional `context` parameter
- Deprecated `from_provider` methods removed; `auto_detect` updated to new API
- Obsolete single-provider example scripts removed (replaced by cross-provider examples)
- `_normalize()` extracted to `BaseConverter` as a shared utility

### Fixed

- camelCase fallback for Google GenAI REST stream/response fields
- Anthropic stream converter: `thinking_delta`, `signature_delta`, `tool_call_id` handling
- OpenAI Chat stream converter: `reasoning_content`, empty string, `tool_call_index` handling
- Missing `__init__.py` for test package discovery
- `from_provider` calls in `google_genai_rest_e2e` integration test

---

## 2026-02-14 — Cross-Provider Examples & Stream Converters

### Added

- **Stream converters** for all 4 providers: OpenAI Chat, Anthropic, Google GenAI, OpenAI Responses
- Stream converter unit tests for all providers
- **6 cross-provider conversation examples** (SDK-based): OpenAI Chat ↔ Anthropic, OpenAI Chat ↔ Google GenAI, OpenAI Chat ↔ OpenAI Responses, Anthropic ↔ Google GenAI, Anthropic ↔ OpenAI Responses, Google GenAI ↔ OpenAI Responses
- Common resources module for cross-provider conversation examples
- Image URL to inline base64 conversion helpers for Google GenAI compatibility
- OpenAI Responses E2E integration tests (REST + SDK)
- Unit tests for OpenAI Responses Ops classes and converter
- Examples README in English and Chinese

### Changed

- **OpenAI Responses converter** restructured to Bottom-Up Ops Pattern
- Post-refactor cleanup: removed deprecated utils and empty directories

### Fixed

- Image URLs converted to inline base64 for Google GenAI provider compatibility

---

## 2026-02-13 — Bottom-Up Ops Architecture

### Added

- **Google GenAI converter** rebuilt with Bottom-Up Ops Pattern
- TypedDict replicas of **OpenAI Responses API** types
- TypedDict replicas of **Google GenAI SDK** types
- Google GenAI REST and SDK E2E integration tests
- Unit tests for `google_genai` converter Ops classes
- Anthropic SDK and REST E2E integration tests
- OpenAI Chat E2E tests split into SDK and REST versions
- **GitHub Actions** CI/CD workflows and Dependabot configuration

### Changed

- **Anthropic converter** redesigned with bottom-up Ops architecture
- Imports updated to use new `google_genai` converter module
- Old `google/` converter and legacy tests removed

---

## 2026-02-12 — Converter Redesign

### Added

- TypedDict replicas of **Anthropic SDK** types
- TypedDict replicas of **OpenAI Chat** types with backward compatibility and tests
- Legacy body converter design preserved as historical reference

### Changed

- **OpenAI Chat converter** redesigned with bottom-up Ops architecture
- Ruff lint errors fixed across entire codebase

---

## 2026-01-06 — Layered Architecture & Documentation

### Added

- English and Chinese documentation structures initialized (`docs_en`, `docs_zh`)
- Comprehensive error handling documentation
- OpenAI Chat Converter integration tests
- Comprehensive mock implementations for `BaseConverter` test class
- File handling functionality in base converter
- Provider-to-IR mapping documentation

### Changed

- Converter base refined with layered abstract template
- All 4 converters restructured with layered architecture (Anthropic, OpenAI Chat, OpenAI Responses, Google GenAI)
- Type annotations updated for IR content/part conversion methods
- IR type system reorganized and enhanced
- English translations added to code comments and docstrings

### Fixed

- Reasoning content field assertion corrected
- File content handling in OpenAI Chat Completions converter

---

## 2026-01-05 — Auto-Detection & Package Maturity

### Added

- **`detect_provider()`** for automatic provider format auto-detection
- **`convert()`** convenience function for one-step format conversion
- `developer` role support in message validation
- Comprehensive validation tests for `BaseConverter`, Anthropic, Google GenAI, and OpenAI converters
- Tool call and tool definition conversion tests
- pytest configuration and `pytest-cov` dependency
- Competitive analysis document

### Changed

- **Package renamed** from `llm-provider-converter` to `llm-rosetta`
- IR format usage standardized across all providers
- Message creation standardized using `Message` class in examples
- Test suite migrated from unittest to pytest
- Common logic extracted into shared utility modules

### Fixed

- Standalone tool calls without current message context in OpenAI Responses converter
- Google GenAI Pydantic model handling reordered for tuple compatibility
- OpenAI content handling logic simplified for single text parts

---

## 2026-01-04 — Examples & Packaging

### Added

- `pyproject.toml` for package configuration
- Multi-turn chat example with tool integration
- Anthropic handover in multi-turn chat example
- Google GenAI function calling in multi-turn chat example

### Changed

- Utility functions moved from converters to IR types module
- OpenAI Chat converter code formatting improved
- Deprecated multi-provider query and weather tool modules removed

---

## 2025-12-24 — Initial Implementation

### Added

- **IR type system**: intermediate representation types for messages, content parts, tools, configs, request/response
- **`BaseConverter`** abstract class for LLM provider conversion
- **`AnthropicConverter`**: bidirectional Anthropic Messages API conversion
- **`OpenAIChatConverter`**: bidirectional OpenAI Chat Completions API conversion
- **`OpenAIResponsesConverter`**: bidirectional OpenAI Responses API conversion
- **`GoogleGenAIConverter`**: bidirectional Google GenAI SDK format conversion
- Comprehensive test suites for all 4 converters
- Package initialization and exports
- Weather tool example with mock data

---

## 2025-12-09 — Research & Design

### Added

- Initial project structure
- LLM provider message typing schemas documentation and comparison
- Provider messages IR design documentation
- MCP support comparison across providers (OpenAI, Anthropic, Google)
- Google GenAI Interactions API type analysis
- Multi-provider query example function
- OpenAI Responses API support in query examples
