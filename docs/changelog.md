---
title: Changelog
---

# Changelog

All notable changes to LLM-Rosetta are documented here. This project follows [Keep a Changelog](https://keepachangelog.com/) conventions.

## [Unreleased]

## v0.12.0 — 2026-09-03

### Added

- **Namespace tool flattening** (PR [#626](https://github.com/Oaklight/llm-rosetta/pull/626)): flatten `type: "namespace"` tool containers (used by Codex) into individual IR tools with namespace metadata and cross-namespace dedup. Qualified naming keeps tool names within the 64-char limit.
- **`additional_tools` input item support** (PR [#623](https://github.com/Oaklight/llm-rosetta/pull/623)): extract tool definitions from Codex's Responses API `additional_tools` input items. Namespace tools are automatically expanded via the #626 pipeline.
- **Custom tool tri-state** (PR [#624](https://github.com/Oaklight/llm-rosetta/pull/624)): `supports_custom_tools` is now `None | bool` — `None` defers to shim default, explicit `False` forces downgrade even when the shim says `True`. Gateway config `supports_custom_tools: false` now works correctly.
- **Configurable `data_dir`** (PR [#619](https://github.com/Oaklight/llm-rosetta/pull/619)): gateway persistence storage location is now configurable via `--data-dir` CLI flag or `server.data_dir` in config. Keys DB and request log DB default to this directory.
- **Admin logo picker** (PR [#630](https://github.com/Oaklight/llm-rosetta/pull/630)): compact icon button with searchable dropdown. Curated popular provider icons shown first, full list fetched on-demand from jsdelivr (`@lobehub/icons-static-svg`). Supports custom URL fallback, keyboard navigation.
- **Admin layout editor** (PR [#629](https://github.com/Oaklight/llm-rosetta/pull/629)): drag-and-drop developer tool at `design/ui/layout-editor.html` for prototyping provider modal field layouts with preview mode and HTML export.
- **Provider `models_path` and `logo` config fields** (PR [#617](https://github.com/Oaklight/llm-rosetta/pull/617)): per-provider model listing endpoint path override and logo URL in gateway config and admin UI.
- **Reasoning capability enforcement** in pipeline: block `reasoning` tool choice when the upstream model does not declare reasoning capability.
- **Admin chart redesign** (PR [#600](https://github.com/Oaklight/llm-rosetta/pull/600)): replace line charts with step (throughput) + scatter (latency) for more accurate visualization.
- **Admin accessibility improvements** (PRs [#603](https://github.com/Oaklight/llm-rosetta/pull/603), [#606](https://github.com/Oaklight/llm-rosetta/pull/606)): keyboard navigation for model toggles, focus traps, ARIA attributes.

### Fixed

- **Anthropic streaming tool call binding** (PR [#628](https://github.com/Oaklight/llm-rosetta/pull/628)): use `chunk["index"]` to resolve tool call ID for `input_json_delta` instead of last-registered key. Fixes interleaved parallel tool calls producing incorrect tool IDs.
- **Admin hint popup clipping** (PR [#631](https://github.com/Oaklight/llm-rosetta/pull/631)): switch from CSS absolute positioning to JS-driven fixed positioning with smart above/below detection. Popups no longer clipped by modal `overflow-y: auto`.
- **`tool_call_id` sanitization** at provider output boundary: sanitize IDs that contain characters not accepted by downstream providers.
- **Defensive `model_list_transform`** when `internal_id` is absent in model config.
- **Admin error dump buttons** not responding to clicks.
- **Admin provider list view** redesigned as compact single-line rows.
- **Logging import** moved to module level to avoid repeated imports.

### Changed

- **Admin provider modal layout** rearranged: Logo + Provider Name row, API Key full width, Base URL + Models Listing Path row (69/31), Proxy URL + Timeout row (69/31). Hint text removed from Models Path and Timeout fields.
- **Shim `ReasoningCapability`** redesigned with `thinking_modes`, `effort_range`, and `visibility_modes` fields (PR [#614](https://github.com/Oaklight/llm-rosetta/pull/614)).
- **Argo shim** uses `argo:` prefix in model display names and shows upstream provider in fetch dialog.
- **Admin responsive layout** unified provider card markup with responsive list stages (PR [#612](https://github.com/Oaklight/llm-rosetta/pull/612)).

## v0.11.2 — 2026-08-30

### Added

- **Native `tool_search` passthrough** (PR [#593](https://github.com/Oaklight/llm-rosetta/pull/593)): vendor `sparse_search`, add `tool_search_mode` to shim schema, and support native `tool_search` passthrough in both streaming and non-streaming Responses API paths.
- **Provider connectivity test** (PR [#588](https://github.com/Oaklight/llm-rosetta/pull/588)): new `POST /admin/api/config/providers/<name>/test-connectivity` endpoint probes a provider's base URL and each configured endpoint (models, embedding, rerank) for reachability. Shows raw and normalized URLs to help diagnose double-prefix issues. "Test" button added to provider cards in the admin UI.
- **`.well-known/change-password` redirect** (PR [#579](https://github.com/Oaklight/llm-rosetta/pull/579)): `GET /.well-known/change-password` returns a 302 redirect to `/admin#change-password`, auto-opening the settings panel. Enables browser and password manager integration per the [web standard](https://web.dev/articles/change-password-url).

### Fixed

- **Duplicate chat finish events** (PR [#592](https://github.com/Oaklight/llm-rosetta/pull/592)): deduplicate `finish_reason` when upstream repeats it on the same choice index (e.g. OpenAI with `stream_options.include_usage`). Adds per-choice finish tracking to `StreamContext`. Fixes [#589](https://github.com/Oaklight/llm-rosetta/issues/589).
- **Embedding route `upstream_model` remapping** (PR [#586](https://github.com/Oaklight/llm-rosetta/pull/586)): the embedding-specific route now applies `upstream_model` name remapping from `model_upstream_names`, matching the chat fallback route. Previously model aliases (e.g. `argo:text-embedding-3-small` → `v3small`) were ignored, causing upstream 404s.
- **Double version prefix in embedding/rerank URLs** (PR [#588](https://github.com/Oaklight/llm-rosetta/pull/588)): `ProviderInfo` now auto-detects when `base_url` ends with a version segment (e.g. `/v1`) that would duplicate the `url_template` path start, and strips it. Fixes `base_url: "https://api.openai.com/v1"` + `embedding_path: "/v1/embeddings"` producing `/v1/v1/embeddings`.
- **Logo icon centering** (commit [f4c89e3](https://github.com/Oaklight/llm-rosetta/commit/f4c89e3)): center stone silhouette in icon SVGs, switch to transparent background with `prefers-color-scheme` media query for automatic dark/light theme adaptation.

### Changed

- **Unified endpoint URL construction** (PR [#588](https://github.com/Oaklight/llm-rosetta/pull/588)): embedding and rerank handlers now use `ProviderInfo.upstream_url()` (template-based) instead of ad hoc f-string concatenation, matching the chat path architecture.
- **Vendor updates** (PR [#594](https://github.com/Oaklight/llm-rosetta/pull/594)): httpclient 0.4.5→0.4.6, sse 0.3.2→0.3.3.

## v0.11.1 — 2026-08-29

### Fixed

- **OpenAI Responses reasoning encrypted state** (PR [#576](https://github.com/Oaklight/llm-rosetta/pull/576)): forced same-format Responses→Responses streaming conversion now preserves `encrypted_content` and the source reasoning item ID from `response.output_item.done`. Previously the IR round-trip dropped both, breaking clients that replay completed reasoning state (e.g. `store: false` / ZDR flows). Fixes [#575](https://github.com/Oaklight/llm-rosetta/issues/575).
- **Google GenAI reasoning config round-trip** (PRs [#582](https://github.com/Oaklight/llm-rosetta/pull/582), [#583](https://github.com/Oaklight/llm-rosetta/pull/583), [#584](https://github.com/Oaklight/llm-rosetta/pull/584)): parse `thinkingConfig` from REST `generationConfig` on inbound, map reasoning effort to `thinkingLevel`, forward `summary`/`include_thoughts` across all converters.
- **Responses API reasoning summary forwarding** (PR [#581](https://github.com/Oaklight/llm-rosetta/pull/581)): forward `reasoning.summary` in outbound Responses API requests.

### Changed

- **Gateway `create_app` composable** (PR [#578](https://github.com/Oaklight/llm-rosetta/pull/578)): refactored `create_app` via `GatewayExtensions` for extensibility.
- **IR `ReasoningDeltaEvent`** now declares `encrypted_content` and `provider_metadata` fields, consistent with `ToolCallStartEvent`.

## v0.11.0 — 2026-08-28

### Changed

- **Admin UI color system redesign** (PRs [#564](https://github.com/Oaklight/llm-rosetta/pull/564), [#566](https://github.com/Oaklight/llm-rosetta/pull/566), [#571](https://github.com/Oaklight/llm-rosetta/pull/571)): replaced the single light/dark theme with a two-scheme system — **Minimal** (Vercel-inspired pure B&W) and **Emerald** (Neon-inspired green accent), each with light and dark modes (4 total combinations). Architecture moved from JS-driven `THEMES` object to CSS-driven compound selectors (`[data-scheme][data-mode]`). Settings UI now has separate Scheme and Mode selectors.
- **Admin UI typography and consistency** (PR [#570](https://github.com/Oaklight/llm-rosetta/pull/570)): consolidated font sizes from 10 discrete steps to 7. Extracted `.btn-disabled` class to replace inline disabled styles. Normalized elevation model — removed box-shadow from settings sections, unified to border-hover pattern. Standardized form input font-family (mono for code inputs, sans for selects).
- **Admin UI responsive layout** (PR [#565](https://github.com/Oaklight/llm-rosetta/pull/565)): tab bar overflow scroll on mobile, table-scroll wrappers, content max-width 1400px, provider card grid 280px min for 4-column layout, settings popup padding reduced on mobile.
- **Admin UI CSS variable architecture** (PR [#571](https://github.com/Oaklight/llm-rosetta/pull/571)): migrated scheme-specific structural differences (table header typography, badge radius, chart bar colors) from CSS selector overrides to custom properties. Adding a new scheme now requires only one variable block in `base.css`.

### Fixed

- **Admin UI CSS bugs** (PR [#564](https://github.com/Oaklight/llm-rosetta/pull/564)): fixed undefined `var(--hover)` in fetch-models list, consolidated duplicate `.btn-danger` definitions, added `--purple` to dark theme, replaced all hardcoded hex/rgba colors with CSS variables and `color-mix()`.
- **Error dump coverage** (PRs [#572](https://github.com/Oaklight/llm-rosetta/pull/572), [#573](https://github.com/Oaklight/llm-rosetta/pull/573)): added `dump_error()` to 4 previously uninstrumented failure paths — request-phase conversion errors (400), non-streaming connection errors (502), response-phase conversion errors (502), and mid-stream error chunks. Extracted `DumpContext` dataclass for cleaner parameter passing.
- **Emoji empty-state icons replaced** with SVG line icons (bar-chart, camera, folder) matching the minimal design language.
- **Rosetta Stone SVG favicon** — replaced the emoji favicon (🔀) with the project's Rosetta Stone silhouette, served as both inline `<link rel="icon">` and server-side `/favicon.ico`.

- **Admin UI config path abbreviated** (PR [#565](https://github.com/Oaklight/llm-rosetta/pull/565)) — shows filename only with full path in tooltip. System clock now shows timezone abbreviation.
- **Admin UI toast centered** (PR [#565](https://github.com/Oaklight/llm-rosetta/pull/565)) — toast notifications now centered horizontally instead of pinned to bottom-right.
- **OpenAI Responses reasoning input lifecycle** (PR [#569](https://github.com/Oaklight/llm-rosetta/pull/569)): fixed output-only fields (`status: "completed"`, synthetic `rs_` IDs) leaking into Responses request input items. Unproven cross-format reasoning (from Chat/Anthropic/Google) is now omitted rather than assigned a fake identity. Proven Responses-origin reasoning preserves the real ID and summary but strips output-only status. Fixes [#568](https://github.com/Oaklight/llm-rosetta/issues/568).
- **Admin UI "Allowed Shims" hidden** (PR [#574](https://github.com/Oaklight/llm-rosetta/pull/574)): removed the confusing "Allowed Shims" column and modal input from the API keys page — "shim" is an internal concept. Backend storage unchanged (defaults to `["*"]`).

### Added

- **Admin UI accessibility** (PR [#565](https://github.com/Oaklight/llm-rosetta/pull/565)): ARIA roles (`tablist`/`tab`/`tabpanel`, `radiogroup`/`radio`, `dialog`), arrow key navigation for tabs and segmented controls, focus trap for modals and settings popup.
- **Rosetta Stone header logo** — SVG silhouette icon in the admin panel header.
- **GitHub repo icon** — `design/logo/out/rosetta-icon-github.svg` for GitHub repository settings.
- **Admin UI i18n** — added Scheme/Mode selector labels in English and Chinese.
- **Design demos** — theme comparison demos on `design/ui` branch (Minimal, Emerald, Vercel, Neon, Metallic styles).

## v0.10.0 — 2026-08-26

### Added

- **Standalone Nuitka binaries** (PR [#555](https://github.com/Oaklight/llm-rosetta/pull/555)): pre-compiled single-file executables for 6 platforms — linux-x86_64 (glibc + musl), linux-arm64 (glibc + musl), macOS arm64, and Windows x86_64. No Python runtime required. Includes pyinstrument profiling support.
- **Binary-based Docker images** (PR [#555](https://github.com/Oaklight/llm-rosetta/pull/555)): three image variants — `alpine` (musl binary, ~21 MB, default), `glibc` (busybox:glibc, ~25 MB), and `python` (pip-based, ~80 MB). Alpine variant tagged as `:latest` and `:<version>`.
- **Makefile build targets** (PR [#555](https://github.com/Oaklight/llm-rosetta/pull/555)): `build-binary`, `build-binary-musl`, `build-docker-alpine`, `build-docker-glibc`, `build-docker-python` for local and CI builds.

### Changed

- **Docker privilege model** (PR [#555](https://github.com/Oaklight/llm-rosetta/pull/555)): replaced su-exec/PUID/PGID with Docker-native `USER appuser` + `--user` flag. Use `docker run --user $(id -u):$(id -g)` for custom UID mapping.
- **Docker default image** — `:latest` now points to the Alpine binary image (~21 MB) instead of the Python-based image (~80 MB). The Python image is still available as `:<version>-python`.

## v0.9.0 — 2026-08-21

### Added

- **Rerank IR types and converter family** (PR [#506](https://github.com/Oaklight/llm-rosetta/pull/506)): 5 TypedDict IR types (`IRRerankRequest`, `IRRerankResponse`, `RerankDocument`, `RerankResultItem`, `RerankUsageInfo`) and 3 converters (`JinaRerankConverter`, `CohereRerankConverter`, `VoyageRerankConverter`) covering all major rerank API format families.
- **Embedding IR types and converter family** (PR [#510](https://github.com/Oaklight/llm-rosetta/pull/510)): 6 IR types (`IREmbeddingRequest`, `IREmbeddingResponse`, `EmbeddingItem`, `EmbeddingUsageInfo`, `EmbeddingTaskType`, `EmbeddingEncodingFormat`) and 4 converters (`OpenAIEmbeddingConverter`, `JinaEmbeddingConverter`, `VoyageEmbeddingConverter`, `CohereEmbeddingConverter`).
- **Gateway rerank proxy** (PRs [#511](https://github.com/Oaklight/llm-rosetta/pull/511), [#512](https://github.com/Oaklight/llm-rosetta/pull/512)): `/v1/rerank` and `/v2/rerank` endpoints with IR-based cross-format conversion. Config-driven routing via `rerank_providers`, `rerank_models`, `default_rerank_format`. `/v2/rerank` auto-detects Cohere source format.
- **Gateway embedding proxy with IR conversion** (PR [#517](https://github.com/Oaklight/llm-rosetta/pull/517)): convert `/v1/embeddings` from passthrough to IR-based conversion mode (OpenAI↔Cohere↔Jina↔Voyage). Backward compatible — configs without `embedding_providers` still use passthrough.
- **`UpstreamTimeoutError`** (PR [#513](https://github.com/Oaklight/llm-rosetta/pull/513)): distinguish upstream timeout (504) from connection error (502) across all transport methods. Rerank and embedding handlers return proper 504 on timeouts.
- **`x-rosetta-conversion: passthrough` header** — returned when response conversion fails, allowing clients to detect fallback to raw upstream format.
- **Rerank API format documentation** — bilingual (en/zh) documentation of Jina, Cohere, Siliconflow, and Voyage rerank API formats with provider lineage and IR mapping tables.
- **Embedding API format documentation** — bilingual (en/zh) documentation of OpenAI, Cohere, Jina, and Voyage embedding API formats.
- **ConversionPipeline passthrough mode** (PR [#520](https://github.com/Oaklight/llm-rosetta/pull/520)): `force_conversion` parameter (default `True`). When `False` and source == target, the pipeline skips IR round-trip — fixes information loss when proxying same-format traffic (e.g. Claude Code → gateway → Anthropic upstream).
- **Converter instance caching** (PR [#520](https://github.com/Oaklight/llm-rosetta/pull/520)): `get_converter_for_provider()` caches instances in a module-level dict. Eliminates per-request converter allocation.
- **Fidelity checker** (`fidelity.py`, PR [#520](https://github.com/Oaklight/llm-rosetta/pull/520)): compare original and round-tripped bodies to detect IR conversion loss. Two modes: `"critical"` (per-format field check, ~0.01ms) and `"full"` (recursive leaf-level diff). Wired into pipeline passthrough path via `fidelity_mode` parameter for background monitoring.
- **`StreamProcessorProtocol`** (PR [#520](https://github.com/Oaklight/llm-rosetta/pull/520)): shared `Protocol` for `StreamProcessor` and `PassthroughStreamProcessor` with terminal chunk detection.
- **Rerank source format auto-detection** (PR [#522](https://github.com/Oaklight/llm-rosetta/pull/522)) — detect Voyage format from `top_k` in request body; `/v2/rerank` implies Cohere.
- **Embedding source format auto-detection** (PR [#521](https://github.com/Oaklight/llm-rosetta/pull/521)) — detect embedding source format from request body fields (`input_type` → Cohere/Jina/Voyage; `encoding_format` candidates disambiguate).
- **Admin `disabled_tabs` parameter** (PR [#505](https://github.com/Oaklight/llm-rosetta/pull/505)): `setup_admin(disabled_tabs=["metrics"])` hides admin UI tabs at initialization time.
- **Shim `multimodal_tool_result` capability** (PRs [#523](https://github.com/Oaklight/llm-rosetta/pull/523), [#524](https://github.com/Oaklight/llm-rosetta/pull/524)): `ProviderShim` can now declare `multimodal_tool_result: true/false` in YAML to override the converter's class-level default. The flag is wired through `ConversionContext.options` in both `convert()` and `ConversionPipeline`. Chat converter threads it to `_convert_tool_result_with_packing` so multimodal content is preserved natively when the provider supports it.
- **Streaming refusal events** (PR [#528](https://github.com/Oaklight/llm-rosetta/pull/528)): `response.refusal.delta` / `response.refusal.done` SSE events in the OpenAI Responses converter. New `RefusalDeltaEvent` IR stream type with full p→ir→p round-trip support. Completes issue [#431](https://github.com/Oaklight/llm-rosetta/issues/431).

### Fixed

- **Google GenAI multimodal tool result handling** (PR [#525](https://github.com/Oaklight/llm-rosetta/pull/525)): structured content blocks (`list[ContentPart]`) in tool results are now preserved natively instead of being flattened via `json.dumps`/`str()`. Dict content uses `json.dumps` (not `str()` which produced invalid Python repr). A `_is_content_block_list` guard distinguishes typed content blocks from plain data lists.
- **Chat converter multimodal content loss** (PR [#524](https://github.com/Oaklight/llm-rosetta/pull/524)): `_do_request_to_provider` was not passing `supports_multimodal_tool_result` to `ir_messages_to_p`, so shim overrides had no effect on the real request path. Additionally, `_convert_tool_result_with_packing` always stripped images from tool messages even when the flag was True — images were packed but never injected back, causing silent content loss.
- **Test ordering flakiness** (PR [#523](https://github.com/Oaklight/llm-rosetta/pull/523)): `test_shims.py` fixture now saves/restores the global shim registry instead of clearing it, preventing cross-module test failures.

### Changed

- **Unified `convert()` and `ConversionPipeline`** (PR [#520](https://github.com/Oaklight/llm-rosetta/pull/520)): both now support dual-shim (source + target) transforms and response conversion. `ConversionPipeline` delegates to `convert()` internally, eliminating code divergence.
- **Unified outbound transport** (PR [#516](https://github.com/Oaklight/llm-rosetta/pull/516)): merge `send_request` and `send_passthrough` into a single `send(provider_info, url, body)` method. Move URL construction and stream-flag injection from transport to proxy layer. Net -68 lines.
- **Documentation restructured** — split into three top-level tabs: IR Type System, Library, Gateway. API Reference merged into each tab. Removed standalone API Reference tab.

## v0.8.2 — 2026-08-09

### Added

- **Per-provider/model timeout overrides** (PR [#502](https://github.com/Oaklight/llm-rosetta/pull/502)): configure upstream timeout at provider and model granularity instead of global-only. Resolution: `model.timeout > provider.timeout > server.upstream_timeout`. Admin UI timeout inputs in both provider and model modals.
- **Prompt cache preservation** (PR [#499](https://github.com/Oaklight/llm-rosetta/pull/499)): wire `hoist_late_system_messages` IR transform to all 15 provider shims. Mid-conversation system/developer messages are rewritten as user-role `[System: ...]` envelopes to keep the prompt cache prefix stable.
- **Per-provider hoist toggle** (PR [#499](https://github.com/Oaklight/llm-rosetta/pull/499)): `hoist_system_messages` boolean in gateway config, per-provider override via admin UI checkbox with (i) hint popup.
- **SQLite API key storage** (PR [#496](https://github.com/Oaklight/llm-rosetta/pull/496)): migrate API key storage from plaintext config to SQLite with hash-based validation.

### Changed

- **Complexity reduction** — refactor base converter helpers, Google GenAI converter, OpenAI Responses converter, gateway config, and SOCKS5 test handler to reduce cyclomatic complexity.
- **complexipy v6 pre-commit hook** — enable complexipy v6 pre-commit hook; bump to v6.2.0 with threshold raised to 30.
- **Config override resolution** (PR [#501](https://github.com/Oaklight/llm-rosetta/pull/501)): unify per-provider toggle resolution to Pattern C — `config.resolve()` uses shim defaults as fallback, downstream types simplified from `bool | None` to `bool`.

### Fixed

- **Hint popup hover** (PR [#499](https://github.com/Oaklight/llm-rosetta/pull/499)): admin UI hint popups are now hoverable for text selection (CSS `::before` bridge over the icon-popup gap).
- **Abort-path test doubles** (PR [#497](https://github.com/Oaklight/llm-rosetta/pull/497)): replace `_FakeContext`/`_FakeProcessor` with real `StreamContext` to eliminate interface drift risk.

### Removed

- **Backward-compat aliases** (PR [#492](https://github.com/Oaklight/llm-rosetta/pull/492)): remove `to_provider()` compat aliases from openai_responses and google_genai converters.
- **Dead compat shims** — remove dead backward-compat shims and compat aliases from base converter and converters.
- **Stale docs** — remove stale base converter READMEs.
- **Stray file** — remove stray `validate.py` from project root.

## v0.8.1 — 2026-08-07

### Bug Fixes

- **Custom tool grammar shape** (PR [#489](https://github.com/Oaklight/llm-rosetta/pull/489)): Fix grammar-constrained custom tools (e.g. Codex `apply_patch`) failing on Chat Completions upstreams. Responses API uses a flat `format` shape (`{type, syntax, definition}`); Chat Completions nests under `format.grammar`. Added bidirectional, idempotent shape converters at the Chat boundary.
- **Streaming null union members** (PR [#489](https://github.com/Oaklight/llm-rosetta/pull/489)): Fix `AttributeError` crash on streaming custom tool call deltas. Providers serialize all union members on each delta with inactive ones set to `null`; `dict.get("type", "function")` returns `None` for a present-but-null key. Type is now recovered from the populated payload, falling back to the context-registered type.

### Improvements

- **Public API for tool call order** (PR [#490](https://github.com/Oaklight/llm-rosetta/pull/490)): Replace direct `_tool_call_order` private access from all 4 converters with public methods on `StreamContext`: `tool_call_ids`, `tool_call_count`, `resolve_tool_call_id_by_index()`, `get_tool_call_index()`.
- **O(1) tool call index lookup** — `get_tool_call_index()` uses a reverse dict instead of linear `list.index()` scan.

## v0.8.0 — 2026-08-06

### Spec Compliance

Systematic pass across all four converters to ensure output matches official API specs. Driven by [llm-comply](https://github.com/Oaklight/llm-comply) compliance testing.

- **Anthropic usage fields** — emit all spec-required usage fields (`input_tokens`, `output_tokens`, `cache_creation_input_tokens`, `cache_read_input_tokens`). Always emit `stop_sequence` and `stop_details` on responses.
- **Anthropic caller, citations, container** — add `caller`, `citations`, and `container` fields to Anthropic response output.
- **Anthropic streaming `message_start.input_tokens`** ([#424](https://github.com/Oaklight/llm-rosetta/issues/424), PR [#425](https://github.com/Oaklight/llm-rosetta/pull/425)): Fix `message_start` always reporting `input_tokens=0` by swapping `UsageEvent`/`StreamStartEvent` emission order.
- **OpenAI Chat `logprobs`** — always emit `logprobs` on response choices (required nullable field).
- **OpenAI Chat `finish_reason`** — include `finish_reason: null` on all streaming chunks; map IR refusal reason to `stop` explicitly.
- **OpenAI Chat `annotations`** — nest annotations under `url_citation` and always emit the field.
- **OpenAI Responses `response.in_progress`** — emit `response.in_progress` streaming event per spec.
- **Google `responseId` / `modelVersion`** — include `responseId` and `modelVersion` in streaming chunks.
- **Google `ModalityTokenCount`** — filter non-standard modality values; normalize schema type case between Google and IR formats.
- **Google streaming usage** — flush pending usage on `stream_end` for cross-format streaming; extract `_build_stream_usage_metadata` helper.

### Cross-Format Refusal Handling

- **Complete refusal support** ([#429](https://github.com/Oaklight/llm-rosetta/issues/429)): All 4 converters handle refusal bidirectionally:
    - OpenAI Chat ([#430](https://github.com/Oaklight/llm-rosetta/issues/430)): `refusal` field always present (nullable), streaming `delta.refusal` accumulated.
    - Anthropic ([#432](https://github.com/Oaklight/llm-rosetta/issues/432)): Structured `stop_reason: "refusal"` + `stop_details` (category/explanation), including streaming.
    - Open Responses ([#431](https://github.com/Oaklight/llm-rosetta/issues/431)): `RefusalContent` type parsed and produced.
    - Google ([#433](https://github.com/Oaklight/llm-rosetta/issues/433)): `promptFeedback.blockReason` handling, missing `finishReason` values (`BLOCKLIST`, `PROHIBITED_CONTENT`, `SPII`, `IMAGE_SAFETY`), refusal round-trip via `_provider_metadata` marker.
- **IR RefusalPart** ([#427](https://github.com/Oaklight/llm-rosetta/pull/427)): `RefusalPart` added to `AssistantContentPart` union — refusal responses no longer fail IR validation with 502.

### Response Identity & Metadata

- **Shim-driven response ID prefix** ([#410](https://github.com/Oaklight/llm-rosetta/issues/410), PR [#420](https://github.com/Oaklight/llm-rosetta/pull/420)): `ProviderShim` declares `response_id_prefix` in `provider.yaml`. Converter strips source prefix on ingest, adds target prefix on output. Enables clean cross-format response ID mapping (e.g. `chatcmpl-xxx` ↔ `resp_xxx`). OpenAI (`chatcmpl-`), Anthropic (`msg_`), and OpenAI Responses (`resp_`) prefixes declared.
- **`completed_at` timestamp** ([#410](https://github.com/Oaklight/llm-rosetta/issues/410)): Responses converter sets `completed_at` to Unix timestamp on completed responses (previously always `null`).
- **Function call item ID preservation** — preserve `function_call` item ID across all conversion paths.
- **Strip `_provider_metadata` at HTTP boundary** ([#422](https://github.com/Oaklight/llm-rosetta/issues/422), PR [#423](https://github.com/Oaklight/llm-rosetta/pull/423)): Internal `_provider_metadata` fields no longer leak into outbound HTTP requests or downstream responses.

### Responses API Streaming

- **Unified `output_index`** ([#418](https://github.com/Oaklight/llm-rosetta/issues/418), PR [#419](https://github.com/Oaklight/llm-rosetta/pull/419)): Single monotonic counter on `OpenAIResponsesStreamContext` replaces fragmented per-handler calculation. All output item types allocate indices through `next_output_index()`.
- **Reasoning output items** ([#407](https://github.com/Oaklight/llm-rosetta/issues/407), [#408](https://github.com/Oaklight/llm-rosetta/issues/408), PR [#419](https://github.com/Oaklight/llm-rosetta/pull/419)): Non-streaming reasoning items always include `id` and `status` fields. Streaming reasoning has full lifecycle events. Deterministic reasoning ID generation via SHA-256 hash.
- **Reasoning item order** ([#437](https://github.com/Oaklight/llm-rosetta/issues/437), PR [#438](https://github.com/Oaklight/llm-rosetta/pull/438)): Fix `response_to_provider` output ordering — reasoning items now correctly precede message items.
- **Message phase preservation** ([#440](https://github.com/Oaklight/llm-rosetta/issues/440), PR [#441](https://github.com/Oaklight/llm-rosetta/pull/441)): Responses API `phase` field (`commentary`/`final_answer`) preserved through all conversion paths and streaming. Pipeline bridges phase across streaming contexts.
- **SSE `[DONE]` terminator** ([#409](https://github.com/Oaklight/llm-rosetta/issues/409)): Gateway emits `[DONE]` as the final SSE event after `response.completed`.

### IR & Architecture

- **Provider passthrough events** — new IR types for non-stream and streaming provider passthrough items, allowing converters to forward provider-specific data without loss.
- **Empty reasoning content** — OpenAI Chat converter preserves empty reasoning content instead of dropping it.
- **Multimodal tool result payload duplication** ([#480](https://github.com/Oaklight/llm-rosetta/issues/480), PR [#482](https://github.com/Oaklight/llm-rosetta/pull/482)): When targeting OpenAI Chat, images in tool results were emitted twice — once as a real image block in the synthetic user message, and once as inert base64 text via `json.dumps()` in the `role: "tool"` body — exactly doubling the payload and tripping upstream request-size limits. Packed blocks are now stripped from the tool message; blocks that fail to pack are retained.

### Gateway

- **CORS preflight auth bypass** (PR [#404](https://github.com/Oaklight/llm-rosetta/pull/404), [#405](https://github.com/Oaklight/llm-rosetta/pull/405)): Skip auth on CORS preflight requests; hardened to require both `Origin` + `Access-Control-Request-Method` headers.
- **CORS headers on auth errors** — add CORS headers to auth error responses.
- **Bearer token fallback** — accept `Bearer` token as fallback for all auth strategies.
- **Embedding request IDs** — align embedding request IDs with upstream format.
- **Terminal event on upstream stream abort** ([#481](https://github.com/Oaklight/llm-rosetta/issues/481), PR [#483](https://github.com/Oaklight/llm-rosetta/pull/483)): When an upstream connection dropped mid-stream, the gateway closed the SSE socket without any terminal event — clients waiting on one reported a bare `stream closed before response.completed` with no trace of the cause. The gateway now emits a format-appropriate terminal event carrying the upstream reason (`response.failed` + `[DONE]` for Responses, an error chunk + `[DONE]` for Chat, `event: error` for Anthropic, an error object for Google). Skipped when the stream already ended or the client disconnected. Adds `StreamProcessor.source_context`, plus `StreamContext.next_sequence_number` and `StreamContext.outbound_response_id`.
- **Structured JSON logging** (PR [#468](https://github.com/Oaklight/llm-rosetta/pull/468)): Configurable `debug.log_format` (`json`/`text`/`auto`). JSON mode emits one JSON object per line with UTC ISO 8601 timestamps, structured extras promoted to top-level keys via allowlist. `auto` resolves to `json` for non-TTY, `text` for interactive. Hot-reload via admin API.
- **In-band upstream error surfacing** (PR [#454](https://github.com/Oaklight/llm-rosetta/pull/454)): When an upstream reports a request error inside a 200 SSE stream (e.g. Argo sends `event: error` for over-limit tools), the error chunk was previously swallowed and the client received a successful but empty response. Now detects bare error envelopes, emits a format-appropriate error event, and stops the stream cleanly.
- **Admin custom tools toggle** (PR [#467](https://github.com/Oaklight/llm-rosetta/pull/467)): Expose `supports_custom_tools` checkbox in admin provider settings with (i) hint tooltip.
- **Configurable timeouts** (PR [#463](https://github.com/Oaklight/llm-rosetta/pull/463)): `server.upstream_timeout` and `server.read_timeout` config options (both default 300s).
- **Root redirect** (PR [#461](https://github.com/Oaklight/llm-rosetta/pull/461)): `server.root_redirect` config option for redirecting `GET /` to admin panel.
- **Anonymous access opt-in** — `server.open_on_no_keys` allows anonymous access when no API keys are configured.
- **Atomic config writes** — cross-platform file locking for concurrent config read-modify-write safety.
- **Admin modal polish** — CSS fixes, flatten hint tooltip, i18n alignment.


### Shims & Transforms

- **Auto-inject Anthropic cache breakpoints** ([#464](https://github.com/Oaklight/llm-rosetta/issues/464), [#465](https://github.com/Oaklight/llm-rosetta/issues/465), PR [#469](https://github.com/Oaklight/llm-rosetta/pull/469)): Cross-format requests (OpenAI/Gemini → Anthropic) that lack cache semantics now get up to 4 `cache_hint` breakpoints injected automatically via the `auto_cache_breakpoints` IR transform. Breakpoints placed on: last tool definition, system instruction tail, and last two user messages. Mounted on `argo--anthropic` and `openrouter--anthropic` shims. Two modes: `none_only` (default, skips if any hint exists) and `fill_gaps` (fills per-segment independently).
- **Custom tool downgrade for Chat upstreams** ([#460](https://github.com/Oaklight/llm-rosetta/issues/460), PR [#486](https://github.com/Oaklight/llm-rosetta/pull/486)): `supports_custom_tools` flag on `ProviderShim` (default `False`). When targeting a Chat upstream that doesn't support `{type: "custom"}` tool definitions, `enforce_custom_tools()` downgrades them to `{type: "function"}` with synthetic parameters on request, and `restore_custom_tool_calls()` restores the original type on response.

### Converters

- **Native custom tool support in OpenAI Chat** — `openai_chat` converter natively handles `custom` tool type, including `set_tool_call_type()` public API on `ConversionContext`.
- **`custom_tool_call_output` item type** — OpenAI Responses converter supports `custom_tool_call_output` input items.
- **BaseConverter template method refactor** — `BaseConverter` converted to template method pattern with `__init_subclass__` enforcement of `_PASSTHROUGH_RESTORE_KEY`.

### CI & Documentation

- **On-demand compliance testing** — add [llm-comply](https://github.com/Oaklight/llm-comply) GitHub Actions workflow for schema/spec-based compliance testing against the gateway.
- **Compliance Testing section in README** — link to llm-comply, hosted service, and CI workflow.

## v0.7.3 — 2026-07-25

### Added

- **Per-model enable/disable toggle** ([#382](https://github.com/Oaklight/llm-rosetta/pull/382)): Models can now be individually enabled or disabled via an ON/OFF pill toggle in the admin panel. Disabled models are excluded from routing (`_parse_models` skips `enabled: false`). Backend routes: `toggle_model`, `bulk_update_models` (batch enable/disable/delete).
- **Embedding test menu** ([#382](https://github.com/Oaklight/llm-rosetta/pull/382)): Embedding models now show dedicated test options — Embedding, Batch (array of texts), Matryoshka (user-specified dimensions), and Multimodal (image). Matryoshka uses a custom modal instead of a native `prompt()` dialog.
- **URL template admin UI** for provider and model cards — configure custom upstream URL templates directly from the admin panel.

### Changed

- **Model modal three-tab layout** ([#389](https://github.com/Oaklight/llm-rosetta/pull/389)): Redesigned model edit/add modal with tabbed interface — General (name+provider, segmented LLM/Embedding control, pill-style capability chips), Routing (URL template with expand-link for stream), Transforms (flatten system + reasoning config). Replaces long-scroll single-panel form.

- **Model table UI restyle** ([#382](https://github.com/Oaklight/llm-rosetta/pull/382)): Checkbox column for multi-select with bulk action bar (Enable/Disable/Delete). Clone and Delete moved into ⋯ dropdown menu. Test button unified across LLM and embedding models.
- **Atomic config writes** ([#387](https://github.com/Oaklight/llm-rosetta/pull/387)): `write_config` now uses `tempfile.mkstemp` + `fsync` + `os.replace` for crash-safe, cross-platform atomic writes. Removes all platform-specific locking code (`fcntl`/`msvcrt`). Readers never see a partially-written file.
- **Cross-process config serialization** ([#387](https://github.com/Oaklight/llm-rosetta/pull/387)): New `config_lock(path)` context manager using `.lock` sidecar files with `fcntl.flock` (Unix) / `msvcrt.locking` (Windows). Protects against multiple gateway instances sharing the same config file. All 14 admin route handlers wrapped to serialize read-modify-write cycles.

### Fixed

- **Windows compatibility** ([#381](https://github.com/Oaklight/llm-rosetta/pull/381)): Gateway no longer imports Unix-only `fcntl` module at top level.
- **Preserve upstream User-Agent header** ([#385](https://github.com/Oaklight/llm-rosetta/pull/385)): The gateway now passes through the client’s `User-Agent` header to upstream providers instead of dropping it.
- Filter null values from usage token details before IR validation.
- Align `test_auth` with `open_on_no_keys` behavior ([#388](https://github.com/Oaklight/llm-rosetta/pull/388)).

### Security

- **Block requests when no API keys configured** ([#383](https://github.com/Oaklight/llm-rosetta/pull/383)): When `api_keys` is empty and `open_on_no_keys` is not set, API requests are now rejected with 403 instead of silently passing through.

## v0.7.2 — 2026-07-20

### Added

- **`custom_head` injection for admin panel** ([#378](https://github.com/Oaklight/llm-rosetta/pull/378)): `setup_admin()` accepts an optional `custom_head` HTML fragment injected before `</head>`. Downstream projects can inject `<style>`/`<script>` tags to customize admin UI without modifying the reference `admin.html`. Cached per value — no per-request overhead.
- **`branding` dict for admin panel identity** ([#378](https://github.com/Oaklight/llm-rosetta/pull/378)): `setup_admin(..., branding={title, subtitle, version, links, attribution})` customizes the header, login screen, and settings footer. Serialized as `window.__branding` via `custom_head`; consumer script in `admin.html` patches the DOM. Element IDs: `brandTitle`, `brandLoginTitle`, `brandFooterName`, `brandFooterLinks`. Without branding, the default llm-rosetta identity is unchanged.

### Changed

- Bump vendored `httpclient` 0.4.4 → 0.4.5 — fixes fd leak where `close()` did not close `_async_writer`, preventing `__del__` from cleaning up leaked async streaming responses.
- **Extract `ConfigIO` protocol for admin panel config I/O** ([#376](https://github.com/Oaklight/llm-rosetta/pull/376)): Admin routes now use a `ConfigIO` protocol instead of importing `load_config`/`load_config_raw`/`write_config` directly. Default `JsoncConfigIO` implementation preserves existing behavior; downstream projects (e.g. argo-proxy) can supply alternative implementations via `setup_admin(..., config_io=...)`. Internal helpers `_get_config_path` and `_get_config_io` now raise descriptive `RuntimeError` on missing values instead of returning `None`, removing 16 redundant guard blocks across route handlers.
- Replace Unicode emoji (🔍) with inline SVG in the content capture table for consistent cross-platform rendering.

### Fixed

- Escape `</` in branding JSON serialization to prevent `<script>` tag breakout when branding values contain `</script>`.

## v0.7.1 — 2026-07-16

### Fixed

- **Tool schema sanitization for Anthropic and Google** ([#372](https://github.com/Oaklight/llm-rosetta/issues/372)): Anthropic rejects the OpenAPI `nullable` extension in tool parameter schemas (e.g. from Pydantic-generated JSON Schema). New `convert_nullable_to_type_array()` helper recursively converts `"nullable": true` to standard JSON Schema `"type": [T, "null"]`. Anthropic converter now strips `title` fields and converts `nullable` to type arrays; Google GenAI converter strips `title` (keeps `nullable` — Google supports it). Also handles the edge case where `nullable: true` appears alongside `anyOf`/`oneOf` without a `type` field.
- **`flatten_system` checkbox layout and i18n** in the gateway admin panel.

### Changed

- Bump vendored `validate` 0.6.0 → 0.6.1 (dataclass instance support).
- Restrict Dependabot to LLM SDK dependencies only.

### Added

- SDK type coverage scanner and manual CI workflow for tracking provider SDK type alignment.

## v0.7.0 — 2026-07-10

### Added

- **Anthropic `cache_control` preservation** ([#362](https://github.com/Oaklight/llm-rosetta/pull/362)): New `cache_hint` field on IR parts (`TextPart`, `ImagePart`, `FilePart`, `ReasoningPart`, `ToolCallPart`, `ToolResultPart`, `ToolDefinition`) enables round-tripping Anthropic's block-level `cache_control` through the IR pipeline. The Anthropic converter reads `cache_control` → `cache_hint` on ingest and writes it back on output; non-Anthropic converters silently ignore `cache_hint`, ensuring cross-format safety.
- **`flatten_system_content()` transform** ([#370](https://github.com/Oaklight/llm-rosetta/issues/370)): New body-level transform factory that flattens system message content arrays to plain strings. OpenAI Chat converter now outputs structured content arrays for system messages (preserving block boundaries for `cache_hint`); `flatten_system_content()` downgrades to plain strings for upstream compatibility. Per-model `flatten_system` gateway config with auto-detection for Gemini models. Admin panel toggle included.

### Fixed

- **OpenAI SDK 2.45+ compatibility**: Added `cache_write_tokens` field to `InputTokensDetails` (Responses API) and `PromptTokensDetails` (Chat Completions API) TypedDict replicas to match upstream SDK changes.

### Changed

- **Transform fields renamed** — `from_transforms` → `pre_ir_transforms`, `to_transforms` → `post_ir_transforms` on `ProviderShim`. Old names accepted as backward-compatible aliases in both constructor kwargs and `transforms.py` exports.
- **`system_instruction` unified to `list[TextPart]`** ([#364](https://github.com/Oaklight/llm-rosetta/issues/364)): The canonical IR form of `system_instruction` is now `list[TextPart]` instead of `str`. A single string `"You are helpful."` is represented as `[TextPart(type="text", text="You are helpful.")]`. This ensures consistent structure across all converters and enables block-level metadata (e.g. `cache_hint` for Anthropic prompt caching) to flow through the IR pipeline. All 4 converters updated. **Breaking**: code that reads `ir_request["system_instruction"]` as `str` must handle `list[TextPart]`.

## v0.7.0a1 — 2026-06-27

### Added

- **Hybrid profiling system** ([#339](https://github.com/Oaklight/llm-rosetta/pull/339)): Always-on `perf_counter` phase timing in `ConversionPipeline.profile` (source_to_ir_ms, ir_transforms_ms, ir_to_target_ms, etc.) plus on-demand per-request pyinstrument deep profiling via admin API. `DeepProfiler` context manager in `llm_rosetta.profiling`; new `[profiling]` optional dependency group. Admin endpoints: `POST /admin/api/profiling/enable`, `GET /admin/api/profiling/results`, `GET /admin/api/profiling/results/<index>`, `POST /admin/api/profiling/disable`, `DELETE /admin/api/profiling/results`
- **Profiling admin UI** ([#339](https://github.com/Oaklight/llm-rosetta/pull/339)): New "Profiling" section in admin dashboard with enable/disable controls, result listing, flamegraph download (single and bulk), and restart hint
- **Error dump capability** ([#341](https://github.com/Oaklight/llm-rosetta/issues/341)): Fire-and-forget error dump system that captures full request context on upstream/conversion failures. Image offload before hashing for content-based dedup, zlib compression, 10K entry cap with cascade pruning. Four trigger points covering upstream errors, stream header errors, stream chunk errors, and conversion errors. New functions `dump_error()`, `offload_images()`, `compute_body_hash()`, `compress_body()`, `decompress_body()` exported from `llm_rosetta.observability`
- **Metrics rebuild** ([#340](https://github.com/Oaklight/llm-rosetta/pull/340)): `POST /admin/api/metrics/rebuild` endpoint and "Rebuild Counters" button in admin dashboard. Reconstructs all metrics counters from request log history using batched iteration and atomic swap to avoid exposing half-rebuilt state
- **Observability package** ([#341](https://github.com/Oaklight/llm-rosetta/issues/341)): Extracted `MetricsCollector`, `RequestLog`, `RequestLogEntry`, `PersistenceManager`, and `ProfilerState` from `gateway/admin/` into a new top-level `llm_rosetta.observability` package. These modules are framework-agnostic and can be used by any LLM proxy consumer (e.g. argo-proxy) without depending on the gateway's config system or HTTP server. The `gateway/admin/` modules now re-export from `observability/` for full backward compatibility

### Fixed

- **Metrics breakdown by provider name** ([#340](https://github.com/Oaklight/llm-rosetta/pull/340)): Dashboard breakdown section now groups by provider display name instead of API type, which was merging all Anthropic-format providers into one row
- **Config file write safety**: `write_config()` now uses file locking for cross-process safety
- **Vendored httpserver updated to 0.2.1**: Returns proper HTTP error responses instead of silent disconnects on malformed requests
- **Vendored SSE updated to 0.3.2**: Uses constructor arguments for parser initialization instead of post-init mutation

### Changed

- **Dev tool versions pinned**: `ruff==0.15.20` and `ty==0.0.54` pinned in `[project.optional-dependencies]` to prevent CI drift from upstream tool releases

## v0.7.0a0 — 2026-06-25

### Added

- **ConversionPipeline class** ([#322](https://github.com/Oaklight/llm-rosetta/pull/332)): High-level orchestration class encapsulating the full Phase 1→2→4 conversion lifecycle. `convert_request()`, `convert_response()`, `create_stream_processor()` with `on_ir_ready` callbacks for metadata store integration. One-shot guard prevents accidental reuse
- **Routing layer** ([#323](https://github.com/Oaklight/llm-rosetta/pull/331)): `ResolvedRoute` frozen dataclass and `Router` protocol in the core library. `GatewayConfig.resolve()` consolidates model lookup, provider type, shim binding, capabilities, and reasoning overrides into a single typed result
- **Capabilities module** ([#335](https://github.com/Oaklight/llm-rosetta/pull/336)): `capabilities.py` with `enforce_reasoning()` (pre-IR) and `enforce_vision()` (post-IR) — platform-level capability enforcement separated from provider-specific shim transforms
- **IRTransform system** ([#330](https://github.com/Oaklight/llm-rosetta/pull/334)): `TransformContext` dataclass, `IRTransform` callable type, `apply_ir_transforms()` executor, and `_NamedIRTransform` wrapper. IR-level transforms are now declarative on `ProviderShim.ir_transforms`, separate from body-level `Transform`
- **IR transform factories**: `strip_non_vision_images()`, `truncate_images(max, pattern)`, `unwind_parallel_tool_calls(pattern)` — factory functions producing `IRTransform` callables
- **Message-level transform primitives** ([#328](https://github.com/Oaklight/llm-rosetta/pull/333)): `replace_message_field()`, `default_message_field()`, `strip_fields_for_model()` for nested field operations on `messages[]`
- **Transport layer** ([#321](https://github.com/Oaklight/llm-rosetta/pull/329)): `UpstreamTransport` protocol, `HttpTransport` implementation, `UpstreamResponse`/`UpstreamStream` types, `HttpClientPool`, `send_passthrough()` for non-conversion endpoints
- **`resolve_shim()` public function**: Promoted from private `_resolve_shim()` to public API on `provider_shim.py`

### Breaking Changes

- **`ProviderShim` fields removed**: `max_images`, `max_images_pattern`, `unwind_parallel_tool_calls`, `unwind_parallel_tool_calls_pattern` deleted — these capabilities are now declared via `ir_transforms` tuple using factory functions (`truncate_images()`, `unwind_parallel_tool_calls()`)
- **`apply_shim_to_ir()` behavior changed**: No longer hardcodes image/unwind operations; reads `shim.ir_transforms` declaratively. Renamed to `apply_ir_transforms()` (old name is a deprecated alias)
- **Gateway handler signatures changed**: `handle_non_streaming` and `handle_streaming` take `route: ResolvedRoute` instead of 6 separate parameters (`source_provider`, `target_provider`, `model`, `target_shim_name`, `reasoning_config_override`, `model_capabilities`)

### Refactored

- **Pipeline renamed** ([#330](https://github.com/Oaklight/llm-rosetta/pull/334)): `apply_shim_to_ir()` → `apply_ir_transforms()`, `setup_shim_context()` → `configure_context()`. Old names emit `DeprecationWarning`
- **Gateway proxy.py**: Handlers use `ConversionPipeline` internally. `_resolve_target_transforms`, `process_stream_chunk` deleted
- **Embeddings handler**: Uses `transport.send_passthrough()` instead of reaching into `HttpTransport._pool`. Migrated from `resolve_model()` to unified `resolve()` API and replaced inline telemetry with shared `_record_telemetry()`
- **Auth functions renamed**: `_openai_auth` → `openai_auth` etc. (dropped underscore, public API)
- **Removed `GatewayConfig.resolve_model()`**: Legacy 5-tuple API superseded by `resolve()` which returns `ResolvedRoute` + `ProviderInfo`. Duplicate `DEFAULT_CAPABILITIES` class variable removed

### Fixed

- **Restored legacy `converters/base/` import paths** ([#317](https://github.com/Oaklight/llm-rosetta/pull/317)): Backward-compatible shim modules at old paths (`converters.base.tools`, `.schema`, `.tool_content`, `.cache`)
- **`sanitize_schema` strips `exclusiveMinimum`/`exclusiveMaximum`** ([#337](https://github.com/Oaklight/llm-rosetta/pull/337)): Google GenAI API rejects JSON Schema draft 6+ numeric constraints in tool definitions
- **Stop emitting `reasoning.type` for OpenAI Responses API** ([#337](https://github.com/Oaklight/llm-rosetta/pull/337)): OpenAI and Volcengine Responses APIs reject `reasoning.type` — reasoning is controlled via `reasoning.effort` only. Historical bug from v0.6.8

## v0.6.12 — 2026-06-23

### Fixed

- **Restored legacy `converters/base/` import paths** ([#310](https://github.com/Oaklight/llm-rosetta/issues/310)): The v0.6.11 helpers/ reorganization unintentionally broke import paths that external callers relied on. `sanitize_schema`, `extract_part_ids`, `log_orphan_warnings`, `fix_orphaned_tool_calls_ir`, and `strip_orphaned_tool_config` are re-exported from `converters.base.tools` again, and compatibility shim modules at `converters.base.schema`, `converters.base.tool_content`, and `converters.base.cache` redirect to their new `helpers/` locations. The canonical import path remains `llm_rosetta.converters.base.helpers`; existing code importing from the old paths (e.g. `from llm_rosetta.converters.base.tools import sanitize_schema`) keeps working without changes. The cache singletons are shared across both paths

## v0.6.11 — 2026-06-21

### Added

- **Admin panel provider UX enhancements** ([#292](https://github.com/Oaklight/llm-rosetta/pull/292)): Three improvements to the provider tab:
    - **Multi-key entry list**: API key field auto-detects comma-separated keys (rotation) and switches to multiple `<input type="password">` entries. `+ Add key` button always visible for manual promotion. Eye toggle and copy button in unified footer
    - **Provider search bar**: appears when provider count exceeds 6, filters by name, type, and base URL
    - **Grid/list view toggle**: two icon buttons switch between card grid and compact single-column list view. Preference persisted in localStorage
- **Request ID propagation** ([#296](https://github.com/Oaklight/llm-rosetta/pull/296), [#122](https://github.com/Oaklight/llm-rosetta/issues/122)): Every proxy request generates or honours an `X-Request-ID` header. Propagated to upstream providers, included in all response headers (including error responses), and logged with `[request_id]` prefix for end-to-end traceability
- **Enhanced health check endpoints** ([#297](https://github.com/Oaklight/llm-rosetta/pull/297), [#127](https://github.com/Oaklight/llm-rosetta/issues/127)):
    - `/health` — returns uptime, request counts, errors in the last hour, and per-provider health snapshot (success rate, avg latency, last error). Always HTTP 200; `status` field shows `"ok"` or `"degraded"`
    - `/health/live` — always 200 (Kubernetes liveness probe)
    - `/health/ready` — 200 when all providers healthy, 503 when any provider is critically unhealthy (Kubernetes readiness probe)
- **CORS restriction on admin API** ([#294](https://github.com/Oaklight/llm-rosetta/pull/294), [#233](https://github.com/Oaklight/llm-rosetta/issues/233)): `/admin/api/*` endpoints no longer send `Access-Control-Allow-Origin: *`. New config option `server.admin_cors_origins` (list, default `[]`) allows explicit origin allow-listing. `/v1/*` proxy endpoints unchanged
- **Image count enforcement via shim** ([#301](https://github.com/Oaklight/llm-rosetta/pull/301), [#299](https://github.com/Oaklight/llm-rosetta/issues/299)): `ProviderShim` gains `max_images` and `max_images_pattern` fields. When set, images exceeding the limit are replaced with `[image omitted due to limit]` text placeholders (oldest first, most recent kept). Argo OpenAI shim declares `max_images: 50` with pattern `^(gpt|o\d)` — only GPT/o models are truncated; Gemini and Claude through the same provider pass through unaffected
- **Vision capability enforcement** ([#314](https://github.com/Oaklight/llm-rosetta/pull/314), [#313](https://github.com/Oaklight/llm-rosetta/issues/313)): Models without `vision` capability now have all images automatically stripped and replaced with `[image not available]` instead of being forwarded to upstream where they cause opaque errors (e.g. DeepSeek's "unknown variant `image_url`"). Gateway logs a warning with image count and model name
- **Unix domain socket support** ([#315](https://github.com/Oaklight/llm-rosetta/pull/315)): Gateway can listen on a Unix socket instead of TCP via `--socket/-S` CLI flag or `server.socket` config field. Enables secure deployments on shared multi-user hosts where `127.0.0.1` still exposes the service to all local users. Socket file is restricted to owner-only (`0600`) and cleaned up on shutdown
- **Parallel tool call unwind** ([#303](https://github.com/Oaklight/llm-rosetta/pull/303), [#300](https://github.com/Oaklight/llm-rosetta/issues/300)): `ProviderShim` gains `unwind_parallel_tool_calls` and `unwind_parallel_tool_calls_pattern` fields. When enabled, parallel tool calls (multiple `tool_call` parts in one assistant message) are unwound into sequential call-result pairs before forwarding. Argo OpenAI shim enables this with pattern `^gemini` — Gemini models through Argo get sequential pairs; GPT/o models pass through unchanged

### Changed

- **`converters/base/` reorganized into helpers/ subpackage** ([#311](https://github.com/Oaklight/llm-rosetta/pull/311), [#312](https://github.com/Oaklight/llm-rosetta/pull/312), [#310](https://github.com/Oaklight/llm-rosetta/issues/310)): Utility functions extracted from the flat `converters/base/` directory into `converters/base/helpers/`. Abstract base classes (the Ops pattern contract) stay at the top level; implementation utilities (`cache`, `schema`, `tool_orphan_fix`, `tool_content`, `tool_call_unwind`, `image_limit`, `reasoning`) move to `helpers/`. `tools.py` reduced from 428→185 lines (pure ABC). `reasoning_helpers.py` moved from `converters/` root. `orphan_fix.py` renamed to `tool_orphan_fix.py` for consistent `tool_*` prefix. `helpers/__init__.py` re-exports public functions

- **Retired Argo `_normalize_thinking` transform** ([#304](https://github.com/Oaklight/llm-rosetta/pull/304), [#192](https://github.com/Oaklight/llm-rosetta/issues/192)): Removed dead code from Argo Anthropic shim — the `_normalize_thinking` function, `_BUDGET_RATIO`, and `_ADAPTIVE_THINKING_MODELS` were replaced by declarative `reasoning.model_overrides` in `provider.yaml` but the code and 19 tests remained
- **Speculative extension types marked experimental** ([#302](https://github.com/Oaklight/llm-rosetta/pull/302), [#71](https://github.com/Oaklight/llm-rosetta/issues/71)): `SystemEvent`, `BatchMarker`, `SessionControl`, `ToolChainNode` moved from `types.ir.extensions` to `types.ir.extensions_experimental`. Old import path still works but emits `DeprecationWarning`. Types removed from default `types.ir` namespace; available via `from llm_rosetta.types.ir import experimental`
- **Admin panel i18n**: Chinese translation updated from "服务商" to "服务方" (more neutral for mixed commercial and self-hosted providers)
- **Request log timestamps** ([#298](https://github.com/Oaklight/llm-rosetta/pull/298)): Now show date and time (e.g. "06/19, 20:25:29") instead of time only

### Fixed

- **Admin panel auth flash** ([#291](https://github.com/Oaklight/llm-rosetta/pull/291)): Eliminated flash of unauthenticated content when `admin_password` is configured. Main UI is hidden via CSS (`body.auth-pending`) until the async auth check completes
- **Admin password unresolved env var** ([#293](https://github.com/Oaklight/llm-rosetta/pull/293)): Gateway now refuses to start if `admin_password` contains an unresolved `${...}` placeholder, preventing a predictable literal string from being used as the password
- **`is_image_part` type guard for OpenAI format** ([#306](https://github.com/Oaklight/llm-rosetta/pull/306)): `is_image_part()` now matches both `type: "image"` (IR canonical) and `type: "image_url"` (OpenAI format retained in IR), fixing image truncation being silently skipped for OpenAI-format requests
- **Tool result images counted for truncation** ([#308](https://github.com/Oaklight/llm-rosetta/pull/308), [#299](https://github.com/Oaklight/llm-rosetta/issues/299)): `truncate_images()` now scans images inside `tool_result.result` lists, not just direct message content. Fixes requests that had ≤50 images at IR level but exceeded 50 after the OpenAI Chat converter unpacked tool result images into synthetic user messages. Also optimized `deepcopy` to only copy affected messages instead of the entire conversation
- **Argo Gemini parallel tool call failures** ([#303](https://github.com/Oaklight/llm-rosetta/pull/303), [#300](https://github.com/Oaklight/llm-rosetta/issues/300)): All Gemini models through Argo gateway failed with "function response parts ≠ function call parts" when Claude Code made parallel tool calls. Root cause: Argo's internal OpenAI→Gemini conversion doesn't merge separate tool result messages into a single `functionResponse` Content block. Fixed by unwinding parallel tool calls to sequential pairs at IR level before forwarding

## v0.6.10 — 2026-06-18

### Added

- **Process-level conversion cache** ([#276](https://github.com/Oaklight/llm-rosetta/issues/276), [#279](https://github.com/Oaklight/llm-rosetta/pull/279), [#281](https://github.com/Oaklight/llm-rosetta/pull/281), [#283](https://github.com/Oaklight/llm-rosetta/pull/283)): Per-entry LRU cache with access-refreshed TTL (default 30 min) for tool conversion, schema sanitization, and IR validation. Eliminates repeated work for unchanged tool definitions and messages across conversation turns
    - **Hub-and-spoke architecture**: conversion caches (spokes) are converter-specific; IR validation cache (hub) is converter-agnostic and shared across all converters
    - **Per-entry caching**: individual tools and messages cached by content hash — partial tool changes only re-convert the changed entries, and cross-agent tool overlap shares cache entries
    - **Incremental message validation**: only newly appended messages are validated; previously-seen messages are skipped via the IR validation hub
    - **Mutation detection**: `check_integrity()` on test teardown catches accidental in-place mutation of cached objects; optional `verify=True` mode for runtime self-healing
    - **Benchmark**: 4.4× warm-path speedup (3250 µs → 527 µs local); 33% TTFB reduction in production (11.4 ms → 7.6 ms)
- **`validate_tools()`** ([#283](https://github.com/Oaklight/llm-rosetta/pull/283)): New standalone IR validation function for tool definition lists, symmetric with `validate_messages()`
- **OpenRouter Anthropic shim** ([#284](https://github.com/Oaklight/llm-rosetta/pull/284)): OpenRouter's Anthropic-compatible Messages endpoint is now a first-class provider type. The single `openrouter` shim is split into `openrouter--openai_chat` (Chat Completions) and `openrouter--anthropic` (Messages API), letting OpenRouter route Claude models through the native Anthropic format
- **Admin panel per-model reasoning override** ([#288](https://github.com/Oaklight/llm-rosetta/pull/288)): The model edit modal now displays the effective reasoning config (`thinking_type`, `budget_tokens_ratio`, `disabled_strategy`) with a source badge (provider / model_override / config) and inline editing. Overrides are persisted to `config.jsonc` and resolved at runtime with priority: config override > shim model_override > shim provider default
- **`budget_tokens_default_ratio` reasoning capability** ([#287](https://github.com/Oaklight/llm-rosetta/pull/287)): `ReasoningCapability` gains a `budget_tokens_default_ratio` field. When a provider requires `thinking.type=enabled` but the caller omits `budget_tokens`, a default is derived as `min(max(1024, max_tokens × ratio), max_tokens - 1)` instead of falling back to the unsupported `adaptive` type

### Changed

- **`_convert_tools_from_p` no longer abstract** ([#281](https://github.com/Oaklight/llm-rosetta/pull/281)): Default implementation in `BaseConverter` handles all providers (including Google's list/None return). Per-converter overrides removed — 90 lines of duplicated code eliminated
- **Complete Claude thinking model_overrides** ([#287](https://github.com/Oaklight/llm-rosetta/pull/287)): Added per-model thinking overrides for the Anthropic and Argo shims based on tested support matrices — Haiku 4.5 (`enabled`+budget), Opus 4.7/4.8 (`adaptive`-only), Sonnet 4 on Argo (`enabled`+budget)
- **Model "Clone" replaces "Copy"** ([#290](https://github.com/Oaklight/llm-rosetta/pull/290)): The model row's clone action now opens a prefilled model modal (provider, capabilities, upstream model, and effective reasoning config) with a blank name, matching the provider row's "Clone" behavior — instead of copying a YAML snippet to the clipboard. The model name in the table remains click-to-copy

### Fixed

- **Haiku 4.5 `adaptive` thinking 400 errors** ([#287](https://github.com/Oaklight/llm-rosetta/pull/287)): Haiku 4.5 supports extended thinking but only accepts `thinking.type=enabled` + `budget_tokens`, not `adaptive`. The previous fallback to `adaptive` when no budget was provided caused 400 errors on Anthropic Official, Argo, and OpenRouter. The new `budget_tokens_default_ratio` derives a budget instead
- **Haiku 4.5 `effort` parameter 400 errors** ([#289](https://github.com/Oaklight/llm-rosetta/pull/289)): The `effort` parameter (`output_config.effort`) is only supported on Opus 4.5/4.6/4.7/4.8 and Sonnet 4.6 — not Haiku. Anthropic Official rejected `reasoning_effort` on Haiku 4.5 with a 400. The Haiku model_override now sets `effort_field: none` to drop the unsupported field while keeping the working `thinking.type=enabled` + budget path
- **OpenRouter Anthropic reasoning effort field** ([#284](https://github.com/Oaklight/llm-rosetta/pull/284)): The `openrouter--anthropic` shim uses `output_config.effort` (Anthropic format) instead of the OpenAI Chat `reasoning_effort` field
- **`.env` secret leakage in Docker builds**: Docker build context no longer includes `.env` files, preventing API keys from being baked into image layers

## v0.6.9 — 2026-06-13

### Added

- **API key rotate**: New `POST /admin/api/keys/<id>/rotate` endpoint generates a fresh key value while preserving the same id and label. The admin panel shows a "Rotate" button with inline confirmation and a one-time copy modal for the new key. Request logs are unaffected — they associate by label, not key value
- **Model type selector in Fetch from Provider modal**: Users can now choose between LLM and Embedding when batch-adding models. LLM shows capability checkboxes (text, vision, tools, reasoning); Embedding auto-sets `['embedding']`
- **Model type selector in Add/Edit Model modal**: Replaces the old embedding checkbox + mutual-exclusion logic with the same Model Type radio pattern

### Changed

- **API key length upgraded**: Default generated keys increased from 36 characters (`rsk-` + 32 hex) to 52 characters (`rsk-` + 48 hex), matching OpenAI's key length (192-bit entropy)

### Fixed

- **SSE streaming proxy compatibility** ([#274](https://github.com/Oaklight/llm-rosetta/issues/274), [#275](https://github.com/Oaklight/llm-rosetta/pull/275)): Vendored `httpserver` v0.1.1 — SSE (`text/event-stream`) streaming responses now use `Transfer-Encoding: chunked` instead of raw byte flushing with `Connection: close`. Fixes Go-based reverse proxies (notably NPS `httputil.ReverseProxy`) misinterpreting SSE data as chunked encoding, producing `invalid byte in chunk length` errors and intermittent connection failures under concurrent load. Upstream fix: [Oaklight/zerodep#101](https://github.com/Oaklight/zerodep/pull/101)
- **Admin panel active tab not loading after login**: `initApp()` now triggers data loading for the currently active tab after successful authentication, fixing the issue where the Request Log tab appeared empty until manually switched away and back
- **Uppercase model type radio labels**: Added `text-transform: none` to `.fetch-type-radios label` to prevent `.form-group label` CSS from uppercasing "Embedding" to "EMBEDDING"

## v0.6.8 — 2026-06-11

### Added

- **Shim-driven reasoning configuration** ([#244](https://github.com/Oaklight/llm-rosetta/issues/244), [#245](https://github.com/Oaklight/llm-rosetta/pull/245)): Reasoning effort mapping is now declarative. Provider shims declare a `ReasoningCapability` in `provider.yaml` — specifying `disabled` strategy (`omit` or `thinking_disabled`), `effort_field`, `effort_map`, and `max_effort` cap — instead of hardcoded converter branches. New shared `reasoning_helpers.py` provides `normalize_reasoning_input()` and `apply_reasoning_config()` used by all four converters
- **Expanded reasoning effort ladder** ([#245](https://github.com/Oaklight/llm-rosetta/pull/245)): IR `ReasoningEffortLevel` expanded to six levels: `minimal`, `low`, `medium`, `high`, `xhigh`, `max`. Input normalization accepts `none` (maps to `mode: disabled`) and provider-native values (`xhigh`, `max`) as first-class efforts. Provider shims declare `effort_map` to convert IR levels to provider-specific strings and `max_effort` to cap the highest level emitted
- **`block_index` on IR stream delta events** ([#246](https://github.com/Oaklight/llm-rosetta/issues/246), [#249](https://github.com/Oaklight/llm-rosetta/pull/249)): `TextDeltaEvent`, `ReasoningDeltaEvent`, and `ToolCallDeltaEvent` now carry an optional `block_index` field, preserving the provider's content block index through IR round-trips
- **`cache_creation_tokens` in `UsageInfo`** ([#252](https://github.com/Oaklight/llm-rosetta/pull/252)): New field on the `UsageInfo` TypedDict for Anthropic cache creation token counts
- **Model-level `thinking_type` in shim reasoning config** ([#256](https://github.com/Oaklight/llm-rosetta/pull/256)): `ReasoningCapability` gains a `thinking_type` field to force the outbound `thinking.type` to `"enabled"` or `"adaptive"`. `ProviderShim` gains `model_reasoning` for per-model overrides keyed by upstream model ID (e.g. Argo `claudeopus47 → thinking_type: adaptive`). The `_normalize_thinking` transform is retired — thinking type normalization is now declarative via shim YAML
- **Anthropic `provider_metadata` on tool calls, tool results, and reasoning blocks** ([#257](https://github.com/Oaklight/llm-rosetta/pull/257)): The Anthropic converter now serializes `provider_metadata` as `_provider_metadata` on `tool_use`, `tool_result`, and `thinking` blocks during IR→provider conversion, and reads it back during provider→IR. Fixes Google `thought_signature` being lost in cross-provider round-trips (Anthropic client → Google upstream), which caused Gemini 2.5+ to reject requests with 400 "missing thought_signature"
- **Response reasoning losslessness across converters** ([#263](https://github.com/Oaklight/llm-rosetta/pull/263)): Reasoning content is now preserved through response-side IR→provider conversion in all converters that previously dropped it:
    - **Google GenAI**: `p_reasoning_to_ir` now captures `thoughtSignature` into `provider_metadata` instead of discarding it; `message_ops` delegates to `content_ops.p_reasoning_to_ir()` instead of constructing a bare `ReasoningPart` inline
    - **Anthropic**: `ir_text_to_p` / `p_text_to_ir` now round-trip `_provider_metadata` on text blocks, matching the treatment already applied to reasoning and tool blocks
    - **OpenAI Chat**: `_build_choice_to_provider` now collects `ReasoningPart` content and emits it as `reasoning_content` on the response message, instead of silently dropping reasoning parts

- **Provider-specific reasoning field normalization** ([#264](https://github.com/Oaklight/llm-rosetta/pull/264)): Shim transforms and config for MiniMax, OpenRouter, and Volcengine reasoning fields:
    - **MiniMax**: `thinking_type: adaptive` (rejects `enabled`); `_inject_reasoning_split` to_transform auto-sets `reasoning_split: true` when thinking is requested; `_parse_think_tags` from_transform extracts `<think>` tags from content as fallback
    - **OpenRouter**: `_rename_reasoning_field` from_transform renames `message.reasoning` → `message.reasoning_content` (OpenRouter uses non-standard field name)
    - **Volcengine**: `thinking_type: enabled` (rejects `adaptive`; overrides base converter's `auto → adaptive` default)

### Changed

- **`_build_ir_usage` return type tightened to `UsageInfo`** ([#253](https://github.com/Oaklight/llm-rosetta/pull/253)): All four converter overrides now return `UsageInfo` instead of `dict[str, Any]`, and `_build_provider_usage` accepts `Mapping[str, Any]` instead of `dict[str, Any]`. Removes all usage-related `ty: ignore` comments
- **Anthropic stream usage handlers deduplicated** ([#253](https://github.com/Oaklight/llm-rosetta/pull/253)): `_handle_message_start_from_p` and `_handle_message_delta_from_p` now call `_build_ir_usage()` instead of duplicating cache field extraction inline (−21 lines)

### Fixed

- **Anthropic stream block index desync after thinking block** ([#246](https://github.com/Oaklight/llm-rosetta/issues/246), [#249](https://github.com/Oaklight/llm-rosetta/pull/249)): During Anthropic→IR→Anthropic streaming round-trip, text deltas after a thinking block used index 0 instead of the correct block index (e.g. 1). The Anthropic `from_p` path now copies `chunk["index"]` onto IR delta events, and the `to_p` path prefers the explicit `block_index` over the context fallback. Fixes Claude CLI "Content block is not a text block" errors
- **Cross-provider stream block boundary synthesis** ([#250](https://github.com/Oaklight/llm-rosetta/issues/250), [#251](https://github.com/Oaklight/llm-rosetta/pull/251)): When converting IR streams from providers without content block events (OpenAI Chat, OpenAI Responses, Google GenAI) to Anthropic format, the serializer now emits synthetic `content_block_stop` / `content_block_start` at content-type transitions (e.g. reasoning → text). Previously text deltas could land inside a synthetic thinking block. Added `current_block_type` tracking to `StreamContext`
- **Stream usage detail propagation** ([#252](https://github.com/Oaklight/llm-rosetta/pull/252)): Cache and detail token fields (`cache_read_tokens`, `cache_creation_tokens`, `prompt_tokens_details`, `completion_tokens_details`, `cachedContentTokenCount`) are now preserved through all four converters' streaming paths. Previously these fields were dropped during stream round-trips
- **OpenAI Chat `thinking.type=auto` passthrough** ([#258](https://github.com/Oaklight/llm-rosetta/pull/258)): IR `mode: "auto"` is not a valid upstream value for OpenAI Chat's `thinking.type`. The OpenAI Chat converter now maps `auto` → `adaptive` before emitting the `thinking` object, and applies the same shim `thinking_type` override + `enabled` → `adaptive` safety fallback that the Anthropic path uses
- **`thinking_type=enabled` fallback when `budget_tokens` missing**: When a shim declares `thinking_type: enabled` but the request has no `budget_tokens` (required by Anthropic for `type: "enabled"`), the converter now automatically falls back to `type: "adaptive"` instead of emitting an invalid payload. Applied to both Anthropic and OpenAI Chat converter paths
- **Unsigned Anthropic reasoning blocks in Argo history** ([#268](https://github.com/Oaklight/llm-rosetta/issues/268), [#269](https://github.com/Oaklight/llm-rosetta/pull/269)): `ReasoningCapability` now supports `unsigned_reasoning_blocks: as_is | preserve`. The `argo--anthropic` shim uses `preserve` so prior assistant `thinking` blocks without a usable signature are not forwarded to Argo, avoiding 400 errors while preserving the reasoning content in `provider_metadata.anthropic.unsigned_reasoning_blocks`

## v0.6.7 — 2026-06-04

### Fixed

- **Embedding endpoint upstream_model alias**: The `/v1/embeddings` passthrough handler now substitutes the `upstream_model` name into the request body before forwarding, matching the behavior of the chat completions proxy handler. Previously model aliases (e.g. `bge-m3` → `BAAI/bge-m3`) were ignored, causing upstream model-not-found errors.
- **Admin test timer leak**: The elapsed-time counter is now tracked globally and cleared when a new test starts, preventing multiple timers from writing alternating values to the same display element.
- **Admin test timeout auto-cancel**: When the browser-side 120s timeout fires, the server-side task is now explicitly cancelled via the API instead of being left running.
- **Server-side test task timeout**: Added `asyncio.wait_for()` with a 120s timeout to `_run_test_task`, so hung upstream calls are terminated server-side instead of lingering until the 300s cleanup window.

## v0.6.6 — 2026-06-03

### Added

- **Admin status bar total requests**: Lifetime request counter shown as the first footer segment with locale-aware thousand separators; per-segment hover tooltips (en/zh) explain each metric
- **Vendor httpclient URL-encoded form data**: `httpclient` v0.4.2 — when `data` is a dict without files, encode as `application/x-www-form-urlencoded` instead of requiring explicit serialization

### Changed

- **Schema sanitization module split**: JSON Schema sanitization extracted from `converters/base/tools.py` into its own `converters/base/schema.py` module for clearer separation of concerns
- **Cyclomatic complexity reduction**: Reduced cognitive complexity across tool ops (cross-converter `extract_part_ids`/`log_orphan_warnings` reuse), gateway auth (`check_admin_auth`), proxy streaming (`process_stream_chunk`), config parsing, logging, and admin routes
- **complexipy threshold**: Raised `max-complexity-allowed` from 15 to 25; added `complexipy-pre-commit` hook definition (commented out) for future enablement

### Fixed

- **Admin footer i18n**: Status bar footer now re-renders on language switch instead of requiring a page refresh
- **Docker non-semver build**: `make build-docker V=dev-test` no longer fails — non-semver `V` values fall back to installing from local wheel instead of `pip install ==<version>`

## v0.6.5 — 2026-06-02

### Added

- **API key label filter** — new dropdown on the Request Log tab to filter entries by API key name
- **Client IP logging** — extracts client IP from `X-Forwarded-For` / `X-Real-IP` / TCP peer address and displays it in a new "Client IP" column on the Request Log tab
- **System clock** — live-updating clock in the admin header for correlating log timestamps with current time
- **Dual-threshold log retention** — success and error request log entries are pruned independently; errors get their own cap (`error_max`) so rare failures are not evicted by a flood of successful traffic
- **DB sizing footer** — admin panel footer shows on-disk database size, entry counts per class, and retention caps

### Fixed

- **Provider filter** — filter now correctly matches entries by provider display name, with three-tier fallback (`target_provider_name` → `target_provider` → API type for legacy NULL rows) to handle backfill gaps and disabled providers
- **`/health` info leak** — endpoint no longer exposes the full provider and model list to unauthenticated callers; now returns only `{"status": "ok"}`
- **i18n completeness** — added missing Chinese translations for footer stats, system time label, filter options, and Client IP column header

### Changed

- **Shim directory layout** — provider shims now support grouped subdirectories (e.g. `argo/anthropic/`, `argo/openai_chat/`)
- **Schema migration** — `_migrate_add_columns()` is now generic, adding any missing nullable columns in a single pass
- **CI** — switched to pre-commit for lint/type checks, pinned ty version

## v0.6.4 — 2026-05-20

### Added

- **Tinyleaf-style settings popup**: Replace the modal-overlay settings dialog with a lightweight centered popup — click outside or press Escape to dismiss, theme and language via `<select>` dropdowns with instant apply, About section with version and project links (GitHub, PyPI, Docker Hub, Docs)
- **Lightweight host IP detection endpoint**: `GET /admin/api/diagnostics/host-ip` reads `/proc/net/route` only (microsecond-level, no network calls); proxy URL placeholders auto-update with the correct Docker host IP on page load
- **Admin login persistence**: Login state stored in `localStorage` with 30-minute inactivity auto-logout, logout button in header, password manager compatibility (proper `<form>`, `autocomplete` attributes)
- **Inline delete confirmation**: Two-step confirm for models, API keys, and request logs replaces native `confirm()` dialogs
- **Test modal improvements**: Cancel button with elapsed timer, chart empty state message, Clone button for providers/models
- **Mobile responsiveness**: Responsive header with wrapping, horizontally scrollable tabs and tables

### Fixed

- **Argo Anthropic response normalization**: Detect and convert OpenAI Chat Completions format responses from Argo's `/v1/messages` endpoint to Anthropic Messages format
- **Model-level `thinking_type` in shim reasoning config** ([#254](https://github.com/Oaklight/llm-rosetta/issues/254), [#256](https://github.com/Oaklight/llm-rosetta/pull/256)): `ReasoningCapability` supports `thinking_type` to force `thinking.type` to `"enabled"` or `"adaptive"`. `ProviderShim` gains `model_reasoning` for per-model overrides keyed by upstream model ID. Argo `claudeopus47 → thinking_type: adaptive` via `model_overrides`. `_normalize_thinking` transform retired — thinking type normalization is now declarative
- **Inline confirm i18n and onclick restore**: Add missing `confirm.sure`/`confirm.yes` translation keys; restore original `onclick` handler after confirmation reverts
- **Reverse proxy caching**: Add `Cache-Control: no-cache, no-store, must-revalidate` on all admin API responses; switch test polling to POST
- **Login overlay loop**: Prevent login overlay from dismissing password manager autofill popups
- **C901 complexity**: Extract `_format_connection_error` helper from `fetch_upstream_models`

### Security

- **Admin login rate limiting**: 5 failed attempts trigger a 5-minute IP lockout

### Changed

- **Settings UI simplified**: Themes reduced to Light/Dark; theme and language selectors moved from header dropdowns into the settings popup

## v0.6.3 — 2026-05-17

### Added

- **Full `custom_tool_call` support for OpenAI Responses API**: Handle the `type: "custom"` tool type end-to-end — request ingestion (coerce to IR `type: "function"` with `_passthrough` for round-trip), response parsing (`custom_tool_call` items with plain-text `input`), and streaming (`response.custom_tool_call_input.delta/done` events). Cross-provider degradation synthesizes a single-string-param JSON Schema so custom tools remain usable on Anthropic/Google
- **`tool_type` field on IR `ToolCallStartEvent`**: Streaming events now carry `tool_type` ("function", "custom", etc.) so converters can emit the correct provider-specific event types
- **Argo shims with `model_id_field` and `upstream_model` alias**: New `argo_openai`, `argo_anthropic`, `argo_google` provider shims that rewrite the model field name for Argo-proxied endpoints. Includes thinking normalization transform for `argo_anthropic`
- **Async server-side test tasks**: Admin panel test requests now run in background tasks, preventing browser connection pool exhaustion on slow models
- **Admin login rate limiting**: Brute-force protection on the admin login endpoint

### Fixed

- **Stored XSS in admin UI**: Escape single quotes in the `esc()` helper to prevent injection via provider/model names
- **`custom_tool_call` streaming type loss in gateway**: `OpenAIResponsesStreamContext.from_base()` now copies `_tool_call_types`, fixing custom tools falling back to `function_call` event types during IR→provider streaming
- **Admin UI regressions**: Fix infinite recursion in fetch models checkbox handler, allow API key editing regardless of `credential_visible` setting, remove prefix real-time preview input lag, fix fetch models prefix losing selections, abort test requests on modal close
- **Reasoning test `max_tokens` too small**: Enforce `budget_tokens >= 1024` for reasoning capability tests
- **httpclient AsyncClient serialization lock**: Update vendored httpclient to v0.4.1, use per-task AsyncClient for test self-calls to avoid deadlock
- **ty type-check errors**: Resolve compatibility issues with ty 0.0.32+

### Changed

- **Admin routes split into subpackage**: Refactored monolithic `routes.py` into `routes/` with dedicated modules for auth, config, keys, observability, and testing
- **CI switched to pre-commit**: Linting now uses `pre-commit run --all-files` (ruff + ty); complexipy suspended pending upstream fix

## v0.6.2 — 2026-05-15

### Added

- **Admin password protection**: `server.admin_password` in config enables a login overlay for the admin panel, using HMAC-based session tokens
- **Credential visibility control**: `server.credential_visible: false` hides API key viewing/copying across the admin UI
- **Provider cascade delete**: Deleting a provider now shows affected models and cascade-deletes them

### Fixed

- **Base URL overwrite**: Switching provider type no longer overwrites user-entered base URLs
- **Request log collapse**: Expanded error detail rows persist across auto-refresh

### Changed

- **Zero-dependency on Python ≥3.11**: Replaced PyYAML with vendored zerodep yaml module

## v0.6.1 — 2026-05-15

### Added

- **`/v1/embeddings` passthrough endpoint**: Proxy embedding requests directly to upstream providers without IR conversion — the OpenAI embeddings format is universal across compatible providers. Includes metrics and request log instrumentation
- **`/v1/models` enriched response**: Model listing now includes `api_standard` (e.g. `"openai_chat"`, `"anthropic"`) and per-model `capabilities` fields
- **"Fetch from Provider" in admin panel**: Query upstream `/v1/models` (or equivalent) endpoint from the Models tab, browse available models with checkboxes, and bulk-add with optional prefix. Already-existing models shown as disabled
- **Model management enhancements**: Provider filter dropdown and model name search in the Models tab
- **Embedding capability and test type**: `embedding` capability in the model editor (mutually exclusive with `vision`/`tools`). Embedding models get a single Test button that POSTs to `/v1/embeddings` and displays dimension count
- **Reasoning capability and test type**: `reasoning` capability with dedicated test that sends `reasoning_effort: "low"`. Mutually exclusive with `embedding`
- **Admin panel tab persistence**: Active tab stored in `localStorage`, survives page refresh

### Fixed

- **Missing event loop in SOCKS5 proxy tests**: Use `asyncio.new_event_loop()` as fallback when prior tests have closed the default event loop
- **Type assertion for httpclient response in fetch_upstream_models**: Resolve ty type-check error for `AsyncClient.get()` return type

## v0.6.0 — 2026-05-15

### Added

- **Provider shim layer with declarative YAML directory**: Shims are now defined as `provider.yaml` + optional `transforms.py` files under `shims/providers/<name>/`, automatically discovered and registered at import time
- **Transform mechanism for provider-specific field adaptation**: Three composable primitives — `strip_fields()`, `rename_field()`, `set_defaults()` — handle field-level differences between a provider's API dialect and its base standard
- **7 new built-in provider shims**: xAI (Grok), Qwen (DashScope), Moonshot (Kimi), MiniMax, Zhipu (GLM), OpenRouter, Volcengine — each with provider-specific transforms where needed
- **Gateway proxy applies shim transforms**: The gateway request/response pipeline now applies `to_transforms` on outbound requests and `from_transforms` on inbound responses and stream chunks
- **Provider logos in admin panel**: Provider shims can declare a `logo` URL (SVG), displayed in the admin panel provider cards
- **SOCKS5 proxy support restored**: Updated vendored `httpclient` from zerodep v0.3.1 to v0.4.0, which includes full SOCKS5 proxy support (RFC 1928/1929, with username/password authentication). Both `--proxy socks5://...` CLI flag and `"proxy": "socks5://..."` config entries now work for all upstream requests

### Changed

- **Shim system refactored to declarative YAML**: Replaced programmatic `builtins.py` with a directory-based system (`shims/providers/*/provider.yaml` + `transforms.py`). Adding a new provider now requires only YAML + optional Python, no changes to core code
- **Vendored `validate` updated to zerodep v0.5.0**: Adds `FieldValidator` and `model_validator` for field-level transform+validate pipelines

### Removed

- **`ModelShim` class removed**: Model-level metadata removed in favor of simpler provider-only shims. The `ProviderShim` dataclass no longer has a `models` field

### Refactored

- **Zero-dependency gateway** ([#178](https://github.com/Oaklight/llm-rosetta/pull/178)): Replaced Starlette + uvicorn + httpx with vendored zerodep `httpserver` and `httpclient` modules. The `[gateway]` extra now has zero external runtime dependencies

### Fixed

- **Deep-merge properties in schema flattening** ([#161](https://github.com/Oaklight/llm-rosetta/issues/161)): Fix `$ref`/`$defs` resolution to deep-merge properties and strip orphaned `required` entries
- **Unconditional usage fallback and StreamContext merge** ([#176](https://github.com/Oaklight/llm-rosetta/pull/176)): Guard against missing usage data and ensure StreamContext state is properly merged

### Known Issues

- **Google tool schema `required` validation** ([#161](https://github.com/Oaklight/llm-rosetta/issues/161)): Some Anthropic tool schemas have `required` entries referencing properties not defined in the schema, causing Google API to reject with `INVALID_ARGUMENT`

## v0.5.3 — 2026-04-25

### Added

- **OpenAI Chat converter: thinking config support** ([#170](https://github.com/Oaklight/llm-rosetta/pull/170)): The OpenAI Chat converter now handles `reasoning_config` in IR requests, mapping to OpenAI's `reasoning_effort` parameter. Enables thinking/extended thinking configuration when routing through the Chat Completions API
- **OpenAI Chat converter: `reasoning_content` field handling**: Non-streaming and streaming responses from reasoning models (e.g., o1, o3) now correctly extract the `reasoning_content` field and convert it to IR `ReasoningPart`, preserving chain-of-thought content during cross-provider conversion
- **Upstream error body in admin request log**: When an upstream provider returns an error, the response body is now included in the admin request log entry, making it easier to diagnose upstream failures without checking server logs
- **Copy entry buttons for providers and models in admin page**: Provider and model entries in the admin panel now have copy/duplicate buttons for quickly creating new entries based on existing configurations

### Fixed

- **`FilePart` excluded from `UserContentPart`** ([#160](https://github.com/Oaklight/llm-rosetta/issues/160), [#162](https://github.com/Oaklight/llm-rosetta/pull/162)): `UserContentPart` union type did not include `FilePart`, causing `validate_ir_request()` to reject any user message containing file content (e.g., PDF attachments sent by Claude Code as Anthropic `document` blocks). The bidirectional conversion logic was already implemented for Anthropic (`document`), Google (`inlineData`), and OpenAI Responses (`input_file`) — only the type definition was missing
- **`google_genai/content_ops.py` unconditional `httpx` import** ([#163](https://github.com/Oaklight/llm-rosetta/issues/163)): Replaced `httpx` with `urllib.request` in the Google GenAI content converter for image URL downloads. `httpx` was only declared as a `[gateway]` optional dependency but was imported unconditionally, causing `ModuleNotFoundError` when installed without `[gateway]` extra
- **Emoji icons replaced with SVG in API key management**: API key action buttons in the admin panel used emoji characters that rendered inconsistently across platforms. Replaced with inline SVG icons and added a key visibility toggle button
- **API key column layout shift**: Fixed CSS layout issue where the API key column width changed when toggling key visibility, causing adjacent buttons to shift position
- **Wheel path glob collision with extras brackets**: Quoted the wheel file path in CI install commands to prevent shell glob expansion when the filename contains `[extras]` bracket syntax

### Refactored

- **SQLite persistence backend**: Replaced the JSONL-based request log and JSON-based metrics persistence with a unified SQLite backend. Provides better write durability, atomic operations, and eliminates log rotation complexity. Vendored `persistdict` from zerodep (v0.4.1) as the key-value storage layer

### CI/Build

- **Install smoke tests**: Added CI smoke tests that verify `pip install` succeeds for both `llm-rosetta` (core) and `llm-rosetta[gateway]` variants, catching missing or circular dependencies early

## v0.5.2 — 2026-04-19

### Fixed

- **Streaming round-trip event inflation** ([#157](https://github.com/Oaklight/llm-rosetta/issues/157)): Fixed multiple scenarios where `Provider A → IR → Provider B` streaming conversion produced more output events than input events:
    - OpenAI Chat, Anthropic, and Google GenAI converters emitted redundant `content_block_end` events when no content block was open, inflating the output stream
    - Google GenAI compound chunks (text + finish in the same SSE frame) triggered duplicate text and finish events. Deferred text/finish payloads via `StreamContext.pending_text` / `pending_finish` so they merge into a single event
    - Tool call events generated spurious `content_block_start` / `content_block_end` wrappers in non-Anthropic targets. Suppressed via `_started` lifecycle guard

### Refactored

- **Unified `stream_response_to_provider` dispatch** ([#157](https://github.com/Oaklight/llm-rosetta/issues/157)): Extracted identical dispatch logic (10-entry `_TO_P_DISPATCH` table + dispatch skeleton) from all 4 provider converters into `BaseConverter`. Each converter now only implements a provider-specific `_post_process_to_provider` hook (OpenAI Chat injects envelope fields; OpenAI Responses injects `sequence_number`). Net reduction: ~27 lines
- **`StreamContext` buffer convenience methods**: Added `buffer_usage()` / `pop_pending_usage()` / `buffer_finish()` / `pop_pending_finish()` to replace manual set-and-clear patterns across all converters

### Changed

- **Pinned dev tooling versions**: `ty>=0.0.31` and `ruff>=0.15.0` now declared in `pyproject.toml` dev dependencies. CI no longer installs them separately — uses versions from `pip install -e ".[all]"`
- **Converter tests added to CI**: `tests/converters/` (1086+ tests) now runs in GitHub Actions alongside `tests/test_types/`
- **Roundtrip inflation regression test**: New pytest-parametrized test suite (`tests/converters/test_roundtrip_inflation.py`, 15 cases) verifies `len(output_events) <= len(input_events)` for all 4 providers across text, reasoning, tool call, and compound scenarios

## v0.5.1 — 2026-04-15

### Added

- **`tool_ops` convenience API** ([#148](https://github.com/Oaklight/llm-rosetta/issues/148)): New top-level `llm_rosetta.tool_ops` module for standalone tool definition conversion without instantiating full converter pipelines. Provides `to_provider()` / `from_provider()` unified dispatch and per-provider shortcuts (`to_openai_chat()`, `to_anthropic()`, etc.). All imports are lazy
- **Multi-key API management**: Admin panel now supports multiple API keys per gateway with per-key labels, create/reveal/delete operations, and usage tracking in request logs
- **Gateway API key authentication**: Configurable API key (`server.api_key`) protects AI request endpoints (`/v1/*`). Supports format-native credential extraction — OpenAI `Authorization: Bearer`, Anthropic `x-api-key`, Google `x-goog-api-key` / `?key=` query param. When no key is configured, all requests pass through (backward compatible)
- **Provider enable/disable**: Each provider now supports an `enabled` field (default `true`). Disabled providers and their models are silently excluded from routing
- **Docker support**: Official `Dockerfile`, `docker-compose.yml`, and Makefile targets (`build-docker`, `push-docker`, `run-docker`) for containerized deployment. Alpine-based image with non-root user, config volume mount, and PUID/PGID support
- **Admin panel enhancements**:
    - Provider toggle switches (enable/disable without deleting)
    - Model search and column sorting
    - Provider rename with automatic model reference updates
    - Network diagnostics button (connectivity check + proxy test)
    - Model testing with collapsible raw request/response details and image preview for vision tests
    - Embedded test image (base64 data URI) to avoid external network downloads
    - `reasoning_effort: 'low'` for reasoning model tests to limit token budget

### Changed

- **Admin panel authentication removed from gateway**: Admin panel endpoints (`/admin/*`) no longer require the gateway API key. Admin access control is delegated to the reverse proxy (e.g. Caddy, Nginx). The gateway API key now only authenticates AI request endpoints (`/v1/*`)
- **C901 cyclomatic complexity enforced at threshold 15**: Progressive reduction from 25 → 20 → 15 across all converters and gateway modules. Extracted cross-provider consistency helpers (`_build_ir_usage`, `_build_provider_usage`, `_convert_tools_from_p`, `_apply_tool_config`) with identical names across all 4 converters
- **`BaseConverter` abstract methods**: Four new abstract methods formalize the cross-provider helper pattern. Preserve-mode hooks documented as convention for providers supporting lossless round-trip
- **Vendored `validate.py` updated to zerodep v0.4.2**: Internal refactor of monolithic `_validate()` into focused helpers; no functional changes

### Fixed

- **User-Agent header for image URL downloads**: Google GenAI content converter now sends `User-Agent: llm-rosetta/1.0 (image fetch)` when downloading image URLs for inline base64 conversion, preventing 403 Forbidden from servers like Wikimedia
- **Image URL download with proxy support**: Image downloads in the Google GenAI converter now respect `HTTPS_PROXY` / `HTTP_PROXY` environment variables
- **Empty content fallback for reasoning models**: Admin panel test results now correctly handle `content: ""` (from reasoning models where all `max_tokens` are consumed by reasoning tokens) instead of showing raw JSON
- **Config file not found error**: Gateway now shows a friendly error message when the config file doesn't exist, instead of a Python traceback
- **ty type checker compatibility**: Added `ty: ignore` annotations for TypedDict vs `dict[str, Any]` mismatches and `FinishReason` Literal type narrowing
- **Google converter crash when thinking consumes all tokens** ([#152](https://github.com/Oaklight/llm-rosetta/issues/152)): Gemini 2.5 Pro with small `max_tokens` could have all tokens consumed by thinking, producing a response with no content parts. The converter now falls back to an empty assistant message instead of failing IR validation

## v0.5.0 — 2026-04-12

### Added

- **Gateway Admin Panel**: Built-in web admin panel at `/admin/` for managing gateway configuration, monitoring traffic, and inspecting request logs without editing config files or restarting the server
    - **Configuration tab**: Visual management of providers (add, edit, rename, delete) and model routing with capabilities (text/vision/tools)
    - **Dashboard tab**: Real-time metrics with summary cards (total requests, error rate, active streams, uptime), rolling 60-second throughput and latency charts, per-provider breakdown
    - **Request Log tab**: Filterable request log with model, provider, and status filters, paginated view with color-coded status codes
    - **8 themes**: Light, Indigo Dark, Dracula, Nord, Solarized, Osaka Jade, One Dark, Rosé Pine — persisted in localStorage
    - **i18n**: English and Chinese language support with localStorage persistence
- **File-based persistence**: Metrics counters (JSON) and request log (JSONL) are automatically saved to disk alongside the config file. Data survives server restarts. Log rotation with gzip compression (2 MB limit, 3 backups)
- **Provider rename**: Renaming a provider automatically updates all model routing references
- **API key security**: Masked keys on provider cards, reveal-on-demand with visibility toggle and copy button in edit modal. Masked values are never written back to config

### Changed

- **Provider names decoupled from API standard types**: Provider names are now user-defined strings (e.g. `"my-openai"`, `"OpenRouter_anthropic"`) instead of being constrained to the 4 standard type identifiers. A separate `type` field specifies the API standard (`openai_chat`, `openai_responses`, `anthropic`, `google`)
- Extracted `write_config()` to `config.py` for shared use by CLI and admin panel

## v0.4.2 — 2026-04-11

### Changed

- **`ReasoningConfig.enabled` replaced with `mode` field**: The boolean `enabled` field has been replaced by `mode: Literal["auto", "enabled", "disabled"]`. This aligns the IR more closely with provider semantics (Anthropic's three-way `thinking.type`, OpenAI Responses' `reasoning.type`). Omitting `mode` retains the previous "provider default" behavior. The `effort` field now lives directly in `ReasoningConfig` rather than being nested

### Fixed

- **Responses API `developer` role mapping**: The OpenAI Responses API uses `role: "developer"` (equivalent to Chat's `"system"`). Previously this role was passed through to IR unchanged, causing validation failures. Now correctly mapped to IR `"system"` during Provider→IR conversion
- **Google GenAI `additionalProperties` rejection**: Google's function_declarations API rejects the `additionalProperties` JSON Schema keyword. Added `extra_strip_keys` parameter to `sanitize_schema()` so providers can strip provider-specific unsupported keywords. Google tool_ops now strips `additionalProperties` recursively from nested schemas
- **Google GenAI `prompt_tokens_details` format mismatch**: Google returns modality token details as `list[ModalityTokenCount]` (e.g. `[{"modality": "TEXT", "token_count": 42}]`) but IR expects `dict[str, int]` (e.g. `{"text_tokens": 42}`). Added bidirectional conversion helpers `_modality_list_to_dict()` and `_dict_to_modality_list()`. Handles both SDK (`token_count`) and REST API (`tokenCount`) field names
- **Cross-format tool call ID prefix mapping**: The Responses API enforces `fc_` prefix on tool call IDs, but Chat uses `call_` and Anthropic uses `toolu_`. Added automatic prefix mapping during Responses conversion to prevent validation failures in cross-format scenarios
- **Adaptive thinking fallback**: When converting IR reasoning config to Anthropic format, `mode: "enabled"` without `budget_tokens` now correctly falls back to `{"type": "adaptive"}` with a warning, instead of producing an invalid `{"type": "enabled"}` without the required `budget_tokens`

## v0.4.1 — 2026-04-10

### Added

- **`force_conversion` parameter for `convert()`**: New `force_conversion: bool = False` keyword-only parameter. When `True`, the full source→IR→target pipeline runs even when source and target providers match, ensuring parameter normalization (e.g. `max_tokens` → `max_completion_tokens` for OpenAI Chat). Default `False` preserves existing passthrough behavior

### Fixed

- **Vendored `validate.py` updated from zerodep v0.4.1**: Applied pyupgrade fixes — `Callable` imported from `collections.abc` instead of `typing` (UP035), `@functools.cache` replaces `@functools.lru_cache(maxsize=None)` (UP033)
- Removed unused `sys` import in benchmark script
- Applied `ruff format` to benchmark scripts

### Changed

- Removed incorrect "Related Projects" section from README — LLM-Rosetta is an independent project, not part of the ToolRegistry ecosystem

## v0.4.0 — 2026-04-09

### Added

- **Metadata preservation for lossless A→IR→A round-trip** (#60, PR #119): New `MetadataMode` (`"strip"` / `"preserve"`) option in `ConversionContext` that captures provider-specific fields during `from_provider` and re-injects them during `to_provider`, enabling lossless round-trip conversion. Helper methods on `ConversionContext`: `store_request_echo()`, `store_response_extras()`, `store_output_items_meta()`, `get_echo_fields()`, `get_output_items_meta()`. Per-provider coverage:
    - **OpenAI Responses**: captures/restores 28+ echo fields (temperature, tools, reasoning, truncation, etc.), per-output-item metadata (id, status, annotations, logprobs), `RESPONSES_REQUIRED_DEFAULTS` dict for spec-required fields with sensible defaults, `sequence_number` on all SSE events
    - **Anthropic**: preserves `stop_sequence`, `container`, citations, and OpenRouter extension usage fields
    - **OpenAI Chat**: now re-emits `refusal` and `annotations` fields in `response_to_provider` (previously dropped)
    - **Google GenAI**: preserves `promptTokensDetails` and `cachedContentTokenCount` in usage metadata
    - **Gateway**: automatically enables preserve mode for both streaming and non-streaming paths; bridges metadata between `from_ctx` and `to_ctx` during streaming

### Fixed

- **Open Responses spec compliance for streaming and non-streaming**: Added required fields to all SSE events (`item_id`, `logprobs`, `annotations`, `status`, `sequence_number`, `output_index`, `content_index`), usage detail breakdowns (`output_tokens_details`, `input_tokens_details`), message item IDs and status for non-streaming output items, `function_call` status field in tool_ops, `service_tier` default to `"default"` (string, not null per spec), `completed_at` in required defaults, `created_at` fallback to current time when not provided, normalized echoed tools with `strict: null`, and metadata bridging from `from_ctx` to `to_ctx` in gateway streaming. All 6 Open Responses compliance tests now pass (schema + semantic)

## v0.3.1 — 2026-04-07

### Fixed

- **`service_tier: None` and `system_fingerprint: None` causing validation errors** (PR #118): OpenAI upstream returns these fields as `null`, but the existence check (`if "key" in dict`) passed and assigned `None` to IR's `NotRequired[str]` field. Changed to value-not-None check in both OpenAI Chat and OpenAI Responses converters. Discovered via [Oaklight/argo-proxy#99](https://github.com/Oaklight/argo-proxy/issues/99)
- **Base `StreamContext` missing provider-specific attributes in Responses streaming** (PR #118): When a gateway passes a base `StreamContext` to `OpenAIResponsesConverter.stream_response_to_provider()`, the method accesses `accumulated_text`, `output_item_emitted`, etc. that only exist on `OpenAIResponsesStreamContext`. Added auto-upgrade via `from_base()` classmethod with metadata caching to preserve state across calls

## v0.3.0 — 2026-04-07

### Added

- **Multimodal tool result support across all 4 converters** (#92, PR #109): Tools can now return multimodal content (text + images + files) as `ToolResultPart.result`. Three providers (Anthropic, OpenAI Responses, Google GenAI) support this natively; content blocks are converted through each provider's `content_ops` layer. See provider support matrix below
- **Lossless multimodal tool result roundtrip for OpenAI Chat** (#92, PR #108): OpenAI Chat Completions only accepts `content: string` for tool messages. Implements a dual encoding strategy — tool message keeps `json.dumps(result)` as data fallback, plus a synthetic user message carries visual content (`image_url` parts) wrapped in `<tool-content call-id="...">` XML tags. Unpacking recovers multimodal structure from the synthetic message (preferred) or falls back to JSON parsing if the synthetic message was trimmed by agent frameworks
- **`extract_all_text()` helper function** (PR #109): Extracts text from both `TextPart` and `ReasoningPart` content — useful for thinking models (e.g. gemini-2.5-flash) that may place answers in reasoning parts rather than text parts
- **`generate_chart` example tool** (PR #109): New multimodal tool in `examples/tools.py` returning `[TextPart, ImagePart]` with inline base64 PNG, plus `multimodal_tools_spec` combining all 3 example tools
- **Multimodal integration tests across all 4 provider SDKs** (PR #109): Two new test scenarios per provider — (A) tool returning multimodal content (text + image), (B) image input combined with tool calls. All 30 tests pass against official APIs: OpenAI Chat 9/9, OpenAI Responses 6/6, Anthropic 8/8, Google GenAI 7/7
- **Runtime IR validation via vendored zero-dependency validator** (#91): `validate_ir_request()`, `validate_ir_response()`, and `validate_ir_messages()` utilities validate IR structures against their TypedDict definitions at runtime. All 4 converters now validate output in `request_from_provider()` and `response_from_provider()`. Replaces manual `BaseMessageOps.validate_messages`. Includes Python <3.11 compatibility for `typing_extensions.TypedDict`
- **Constants validation tests**: 39 new tests across 4 `test_constants.py` files verifying that all reason mapping values are valid IR finish reasons, mapping coverage is complete, event type constants are well-formed, and ID generation produces correct formats
- **Finish reason mapping test coverage**: 38 tests validating reason mapping correctness as a safety net for the constants refactoring
- **`ConversionContext` base class for conversion pipelines** (#106, PR #111): New `ConversionContext` dataclass with `warnings: list[str]`, `options: dict[str, Any]`, and `metadata: dict[str, Any]` — a structured context container for non-streaming conversions. New `BaseConverter.create_conversion_context(**options)` factory method mirrors the existing `create_stream_context()`. All 6 non-streaming `BaseConverter` methods now accept an optional `context: ConversionContext` keyword parameter; converter implementations sync warnings to `context.warnings`. Gateway proxy creates a shared context per request and passes it through the full source→IR→target→response pipeline

### Fixed

- **Contextual error messages for tool conversion failures** (#85, PR #110): When `p_tool_definition_to_ir()` fails on a malformed or unsupported tool definition, the `ValueError` now includes `type=` and `name=` context so users can identify which tool caused the issue. Applied to all 4 converters (OpenAI Chat, OpenAI Responses, Anthropic, Google GenAI) with unit tests
- **OpenAI Responses `tool_choice` format** (PR #109): Was using Chat Completions format (`{"type": "function", "function": {"name": "..."}}`); now uses Responses format (`{"type": "function", "name": "..."}`)
- **OpenAI Responses tool call ID round-trip** (PR #109): Responses API uses `fc_` prefix IDs while IR uses `call_` prefix. The Responses `id` is now preserved in `provider_metadata` separately from `call_id`, enabling lossless round-trip conversion
- **OpenAI Responses reasoning item round-trip** (PR #109): Reasoning models (e.g. gpt-5-nano) emit reasoning items with `id` (rs_ prefix), structured `summary` arrays, and `encrypted_content`. These are now preserved through `provider_metadata` for lossless round-trip — fixes 400 errors when reasoning items were sent back without their original `id`
- **IR validation accepts `None` for optional response fields** (PR #109): `logprobs` and `system_fingerprint` in `IRResponse` now accept `None` values (previously only accepted missing keys)
- **OpenAI Responses `content_filter` finish reason mapped to wrong status** (#90): `content_filter` was incorrectly mapped to `"completed"` status in `response_to_provider` and `stream_response_to_provider`. Now correctly maps to `"incomplete"` status with `incomplete_details.reason = "content_filter"`
- **Anthropic streaming missing `refusal` reason mapping**: The streaming `reason_map` was missing the `refusal` entry present in the non-streaming path, causing Anthropic refusal stop reasons to be silently dropped during streaming. Fixed as a side effect of the constants extraction (#64) — both paths now share the same `ANTHROPIC_REASON_FROM_PROVIDER` dict

### Changed

- **`ReasoningConfig.effort` expanded to 5-level enum** (#100): Effort levels now include `"minimal"`, `"low"`, `"medium"`, `"high"`, `"max"`. Provider-specific mappings: Anthropic maps to `thinking.type="adaptive"` with `thinking.effort`; OpenAI Chat/Responses clamp `"minimal"`→`"low"` and `"max"`→`"high"` (with warnings); Google GenAI maps to `thinking_config.thinking_level`
- **`ReasoningConfig.type` replaced with `ReasoningConfig.enabled`** (#70): The `type: Literal["enabled", "disabled"]` field is replaced with `enabled: bool` to avoid shadowing the Python built-in `type` and provide a more natural API
- **Merged duplicate IR concepts** (#69): Removed `candidate_count` from `GenerationConfig` — use `n` instead (Google GenAI converter maps `n` ↔ `candidate_count` internally). Unified `system_instruction` type from `str | list[dict]` to `str`
- **Normalized `ImagePart`, `FilePart`, `AudioPart` to canonical forms** (#68): Each part now has exactly two canonical forms — URL reference + structured inline data (e.g. `image_data`) — plus a unified `provider_ref: dict[str, Any]` for provider-specific references. Removed redundant top-level `data`/`media_type` fields and replaced `file_id`/`audio_id` with `provider_ref`
- **IR type fields changed from `Iterable` to `list`; function parameters to `Sequence`** (#67): TypedDict fields now use `list` for indexable, serialization-friendly semantics; function parameters use `Sequence` (covariant, read-only). Also fixes a latent generator-consumption bug in `strip_orphaned_tool_config`
- **`StreamContext` now inherits from `ConversionContext`** (#106, PR #111): `StreamContext` is a subclass of `ConversionContext` (IS-A relationship), unifying the context model for streaming and non-streaming paths. File renamed: `base/stream_context.py` → `base/context.py`
- **`StreamContext` converted to dataclass with provider subclass** (#65): `StreamContext` is now a `@dataclass` with typed fields (eliminates defensive `getattr`/`hasattr` patterns). OpenAI Responses-specific state extracted into `OpenAIResponsesStreamContext` subclass. New `BaseConverter.create_stream_context()` factory method

### Refactored

- **Warnings single-source convergence** (#113, PR #115): All 4 converter `request_to_provider` methods now use `ConversionContext` as the single accumulation point for warnings. Eliminates the dual-write pattern where warnings were written to both a local list and `context.warnings`. The returned warnings list IS the same object as `context.warnings` — no duplication possible
- **`ProviderMetadataStore` replaces global metadata cache** (#112, PR #117): The module-level `_provider_metadata_cache` dict in `proxy.py` is replaced with `ProviderMetadataStore` — a class with TTL-based expiration (30 min), max-size eviction (10k entries), and explicit lifecycle management. The store is created per-app in `create_app()` and passed via `app.state`, eliminating implicit global mutation. `close_clients()` renamed to `close_resources()` to also clear the store on shutdown
- **Shrink public API export surface** (#114, PR #116): Reduced `__all__` exports across converter packages to only the primary converter class, removing internal implementation details (`*MessageOps`, `*ContentOps`, `*ConfigOps`, `*ToolOps`, `*Constants`) from the public API. Internal modules remain importable for advanced use but are no longer promoted as public surface
- **Extracted stream event handlers from monolithic methods** (#63): Replaced 8 monolithic `if`/`elif` stream methods (~1,781 lines) across all 4 converters with individual handler methods dispatched via class-level handler tables. Public API unchanged
- **Extracted shared utility functions in OpenAI Responses converter** (#66): `resolve_call_id()` and `build_message_preamble_events()` extracted from `converter.py` into `utils.py` with dedicated unit tests
- **Extracted per-provider constants for reason mappings and magic values** (#64): Inline reason mapping dicts, SSE event type string literals, status-to-reason conditional logic, and ID generation patterns across all 4 converters are now centralized in per-provider `_constants.py` modules. Includes `AnthropicEventType` and `ResponsesEventType` classes, `REASON_FROM_PROVIDER` / `REASON_TO_PROVIDER` dicts, and `generate_tool_call_id()` / `generate_message_id()` helpers

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
