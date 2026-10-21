---
title: Architecture Guide
---

# Architecture Guide

## Converter Structure

LLM-Rosetta uses a hub-and-spoke architecture where each converter handles bidirectional conversion between a specific API format and the shared IR (Intermediate Representation).

Every converter lives under `src/llm_rosetta/converters/<name>/` and extends `BaseConverter`. The base class uses a **composition pattern** — subclasses declare ops classes for each concern:

```
converters/<name>/
├── converter.py      # Main converter class (extends BaseConverter)
├── content_ops.py    # Content part conversion (text, images, refusal, etc.)
├── message_ops.py    # Message-level conversion (roles, multi-turn)
├── tool_ops.py       # Tool definitions, tool calls, tool results
├── config_ops.py     # Request config (temperature, top_p, stream options)
└── _constants.py     # Format-specific constants
```

### Where to Put New Code

New features for an existing format (e.g. a new content type, a new field) should be implemented in the **corresponding ops module** of that converter, not in ad-hoc standalone code.

Reuse shared logic from `converters/base/` wherever possible — the base modules provide common building blocks:

| Base Module | Purpose |
|-------------|---------|
| `content.py` | Content part helpers (text, image, refusal) |
| `messages.py` | Message-level utilities |
| `tools.py` | Tool definition and call helpers |
| `reasoning.py` | Reasoning/thinking field handling |
| `schema.py` | JSON Schema normalization |
| `passthrough.py` | Provider-specific passthrough items |
| `context.py` | `ConversionContext` (carries shim, options, state) |

If you find yourself duplicating logic across converters, consider extracting it into a base module.

## Shim Layer

A `ProviderShim` is a lightweight identity card that declares which base converter a provider uses, plus connection defaults and field-level transforms. Shims live under `src/llm_rosetta/shims/providers/<name>/provider.yaml`.

Converters stay format-generic; shims declare provider-specific differences (e.g. response ID prefix, default headers, field renames).

## Adding a New Converter

To add support for a new API standard:

1. Create a converter directory under `src/llm_rosetta/converters/llm/<name>/`
2. Subclass `BaseConverter` and implement all abstract methods
3. Create ops classes following the pattern above
4. Add one `FormatSpec` to `src/llm_rosetta/formats/catalog.py` and register
   its body detector in `formats/detect.py`, then bind the converter class in
   `converters/registry.py`
5. Add a shim under `src/llm_rosetta/shims/providers/<name>/`
6. Add tests under `tests/converters/`
7. Submit a PR

The catalog record is what makes the format visible: the model vocabulary, the
gateway's provider tables, the SSE / error-envelope / usage-shape maps,
auto-detection, the admin picker and the chat routes all derive from it, so no
other file needs a per-format edit.  See existing converters for reference.

## Provider-Specific Data Preservation

LLM-Rosetta separates portable IR data, durable provider passthrough data, and per-conversion context state:

| Layer | Lifetime | Purpose |
|-------|----------|---------|
| Portable IR (`Message`, content parts, tools, reasoning) | Serializable and reusable | Semantics that can be translated between provider formats |
| `ProviderPassthroughEvent` / `ProviderPassthroughItem` | Serializable and reusable | Opaque provider-native chunks or items that have no portable representation |
| `ConversionContext` / `StreamContext` | One conversion pipeline | Warnings, options, echo fields, original IDs/status/annotations, and state used while reconstructing a response |

`ProviderPassthroughEvent` carries provider-native streaming chunks. `ProviderPassthroughItem` carries non-stream request/history items, while `IRResponse.provider_passthrough_items` stores independent non-stream output items together with their original positions.

Passthrough data is tagged with a converter dialect such as `open_responses` or `messages`. A matching target dialect restores a copied native payload. A different target format drops the item; semantic items produce a conversion warning, while lifecycle/heartbeat stream events are dropped silently to avoid warning floods.

`ConversionContext` remains necessary because it serves a different role. It is an ephemeral side channel for a single request/response pipeline and is not serialized into conversation history. Provider passthrough carriers are durable IR data that can survive caching, persistence, and a later HTTP request.

### Stream Dispatch

The base `stream_response_to_provider()` implementation uses a class-level dispatch table (`_TO_P_DISPATCH`) to route IR stream events to handler methods. Provider converters customize behavior through the `_post_process_to_provider()` hook rather than reimplementing the dispatch logic.

## Design Philosophy: Why Everything Goes Through IR

The gateway forces IR conversion for all routes, including same-format — this is intentional. The project started as (and still primarily is) a translation library, and the gateway's core value is built on top of that: cross-format conversion is what makes it more than just another proxy.

Same-format IR round-trip is the most basic fidelity check. If we can't round-trip a single format without information loss, then cross-format translation doesn't stand a chance either. That's the main reason everything goes through IR, and the conversion overhead has been kept low enough that it hasn't been a practical issue.

Before `25924518` (Aug 2026) the pipeline had no `baseline` mode at all. We added it, together with a fidelity checker, so that translation problems surface **synchronously**: the same request runs through both the IR path and `baseline`, and the two outputs are diffed to catch round-trip regressions. `baseline` is a diagnostic mode for finding translation problems — not a production path.

This is a standing design decision, not an implementation detail: **the gateway never shortcuts same-format routes — every request still converts through IR.** `baseline` exists only to surface translation problems synchronously; there is deliberately no path that skips IR. `prefer_same_format` affects only *which* upstream provider is chosen for a model — it never bypasses IR conversion, and its tier-1 case (same shim and endpoint as the anchor) is a failover ordering, not a shortcut. The preference is tiered and relative to an **anchor** — the provider routing would pick with the flag off; the tiers realign to the client's format within the anchor's endpoint, then its account, then any provider, and the account is derived from the credential source rather than declared. Decision record: [#577, comment](https://github.com/Oaklight/llm-rosetta/issues/577#issuecomment-5461182245).

The same-format notion those decisions rest on is one named property, `ConversionPipeline._same_format`, computed on access from the two provider specs the caller passed (already canonicalized) rather than re-derived from a format string at each site. It gates `baseline` and the intrinsic shortcut. Note it compares *specs*, not converters: a bare `chat_completions` source and a `deepseek--chat_completions` target share a converter class but are deliberately not the same spec, because treating them as one would skip the target shim's IR-level traits (its reasoning config, its transforms) — exactly what a caller targeting a shim expects to get.

Intrinsic (provider-hosted) server tools are gated by the **target shim**, never by the format name. `resolve_intrinsic_tools` reads `ToolsConfig.intrinsic_tools` off the *resolved* shim, and resolves a shim name first, so the gateway's `target_shim=route.shim_name` behaves identically to the library's shim object. Whether a client format can name a server tool natively is likewise a property of the **converter** (`BaseConverter._SUPPORTS_NATIVE_SERVER_TOOLS`), not a hard-coded format string: Chat Completions sets it `False`, which is what enables promotion of a plain function named after a supported kind.

## Intrinsic Tools Are Derived Tool Calls

An intrinsic (provider-hosted) tool is an ordinary tool call with one extra parameter, not a parallel type system. The IR carries it on the same parts a client tool uses: `ToolCallPart` / `ToolResultPart` with `tool_type="intrinsic"`, `ToolDefinition` with `type="intrinsic"`, and the streaming `ToolCallStartEvent` / `ToolCallDeltaEvent` / `ToolResultEvent`. The kind (`web_search`, `code_execution`, `google_search`, ...) is the first-class field `intrinsic_kind`, read through `intrinsic_kind_of()`, which falls back to the legacy `provider_metadata` / `metadata` spelling. Everything else a tool call has (`tool_call_id`, `tool_input`, `result`, `tool_group`, `cache_hint`, `provider_metadata`) is inherited unchanged. The kind is never inferred from a tool name.

**Identity and pairing.** A call and its result pair by `tool_call_id`, never by kind. Two calls of one kind in a turn are two calls.

**One decision per leg.**

| Part | Leg | Policy |
| --- | --- | --- |
| Definition | request | Kept or dropped by the target shim's `ToolsConfig.intrinsic_tools`. Never degraded to a function: the model would call something the client cannot run. |
| Call + result | request history | The target converter says whether it can replay the call (`intrinsic_renderable(kind, fields, leg="history")`, plus `intrinsic_requires_result`). If it can, the pair is written natively. If it cannot, `degrade_intrinsic_history` rewrites the call **and** its result, together, into an ordinary function pair. |
| Call + result | response, stream | Never degraded: a response is a call the client did not run, so handing it over as a client call would tell the client to execute something it never asked for. A part the target cannot render goes to `dropped_intrinsic_call` / `dropped_intrinsic_result` — empty, plus a warning naming the kind. |

The rule lives in the base template, not one converter at a time, so it cannot
hold for some dialects and not others. Measured per dialect (a foreign kind on
each leg):

| dialect | history | response | stream |
| --- | --- | --- | --- |
| messages | degrade | drop | drop |
| google_generate | degrade | drop | drop |
| google_interactions | degrade | drop | drop |
| open_responses | degrade | drop | drop |
| chat_completions | degrade | drop | drop |

`chat_completions` degrades on the history leg even though it has no server-tool
vocabulary at all: a history pair the client already saw is context, and a
function call named after the kind carries it. On the other two legs it drops,
like everyone else; its stream writer records the drop on the stream context so
the arguments delta and the finish flush go with the call rather than leaking
after a start that was never written.

The history pass runs after `resolve_intrinsic_tools` and before `qualify_tool_names`, on same-format routes too. A degraded pair is marked `provider_metadata["_downgraded_from"] = {"type": "intrinsic", "kind": ...}`, takes a name-safe function name derived from the kind (qualified `{kind}_intrinsic` when the client already declared a function of that name), and is reported once in the conversion warnings. After the pass the pair is an ordinary function call, so pairing, orphan repair and name qualification are the ordinary tool flows.

**Converters.** `BaseToolOps` owns the dispatch (`ir_tool_call_to_p`, `ir_tool_result_to_p`); a converter overrides only hooks: `intrinsic_renderable`, `ir_intrinsic_call_to_p`, `ir_intrinsic_result_to_p`, `intrinsic_requires_result`. A hook that returns nothing falls back to the converter's ordinary client path. A converter keeps its own kind tables and native shapes, and nothing else about intrinsics.

**Native fidelity.** `metadata["_native"]` and provider signatures are valid only for their origin provider (`converters/base/helpers/opaque.py`). A foreign one is dropped, or replaced by the target's required placeholder.

**Server-tool ids.** A native server item must carry its target's id prefix (Anthropic `srvtoolu_`, OpenAI `ws_` / `ci_`). `prefixed_tool_call_id` adds it deterministically and leaves an id that already has it alone, so a call and its result stay paired and a same-dialect replay is unchanged.

**What crosses formats.** Every cross-format path was replayed against the live upstreams (Anthropic, OpenAI Responses, Google Interactions); each of the twelve pairs of the matrix answers `200`. A pair crosses natively when the target has the same kind and the IR carries every field the target requires, and as a function pair otherwise (an Anthropic `encrypted_content` or an OpenAI `container_id` cannot be invented). There is no semantic equivalence table between kinds such as `web_search` and `google_search`: kinds are asserted per provider, from what each upstream was measured to accept.

Replay is deliberately lenient on signatures: Google Interactions accepts a `google_search` history with either signature removed, so a result signature is not carried.

## Citations Across Dialects

A citation is an annotation of text the model wrote: which source it drew on, and
often which span. Every dialect has such a thing and no two spell it the same
way, so it gets the same treatment as the tools above — a first-class IR part
with a discriminator, one shared field set, and per-dialect tables.

**The IR part.** `CitationPart` carries `citation_type`, the fields every dialect
shares (`url`, `title`, `cited_text`, `start_index`, `end_index`) flat on the
part, and a dialect's own locator fields under `provider_metadata`, keyed by
provider. Reads go through `converters/base/helpers/citations.py`
(`citation_type_of`, one accessor per shared field), never through the shape of
the fields. The pre-2026 spelling, which nested the shared fields under
`url_citation` / `text_citation` and carried no kind, is still **read** and never
written — the same one-spelling rule `tool_group` and `intrinsic_kind` follow.

**The vocabularies.**

| dialect | native form | kinds |
| --- | --- | --- |
| chat_completions | `message.annotations[]`, payload nested under `url_citation` | `url_citation` |
| open_responses | content-part `annotations[]`, flat | `url_citation`, `file_citation`, `container_file_citation` |
| google_interactions | content-part `annotations[]`, flat, snake_case | `url_citation` |
| google_generate | candidate-level `groundingMetadata` (chunks + supports) | sources as `url_citation` |
| messages | `citations[]` inside the text block they annotate | its own five: `char_location`, `page_location`, `content_block_location`, `search_result_location`, `web_search_result_location` |

Two of those share a shape exactly: the Responses API and Interactions both send
`{start_index, end_index, url, title, type: "url_citation"}`, flat. That pair
converts losslessly; the rest map through the shared fields.

**What converts, measured.** Every dialect's URL-bearing citation converts to the
other four. The one cell that does not is a foreign citation *into* Anthropic,
and it is a property of that API rather than a missing writer: its request
validator accepts exactly the five tags above — `url_citation` and
`text_citation` are rejected outright — and requires each citation to point at a
document or `search_result` block the message carries, or at an `encrypted_index`
the provider itself issued. Nothing else can be constructed honestly, so the
writer returns nothing and the caller reports the drop. In the other direction
Anthropic's reader exports its URL-bearing kinds, so an Anthropic citation can
leave for any other dialect; nothing comes back.

**Structure is per dialect.** Anthropic embeds citations in the text block they
annotate, so the reader splits them into their own parts and the writers hang
them back on the preceding text block. Google has no annotation at all: its
citations are the candidate's `groundingMetadata`, rebuilt from the parts
(sources become chunks, spans become the supports that index into them), while
its search-level metadata — the queries and the rendered entry point — is not a
citation and rides the message or the finish event once.

**Streaming.** A streamed citation is a `citation_delta` event carrying the same
part, emitted where the provider reported it rather than deferred: a candidate
that produced no text still carries its sources. Each dialect feeds it from its
own carrier (`citations_delta`, `response.output_text.annotation.added`,
`text_annotation_delta`, or Google's chunk-level metadata) and the base converter
answers it by default with a warning, so a dialect with no carrier reports the
loss rather than dropping it silently.

**Losses are reported.** A citation a target cannot render — a document citation
aimed at a dialect that only knows URLs, a citation with no text block to hang
off — goes through `report_dropped_citation`, which prefers the conversion's
warning sink and always writes the log line.

## Round-Trip Compatibility

All conversion paths must maintain **round-trip compatibility**. Every change must be tested against these scenarios:

- **A → IR → A** (same-format round-trip): converting to IR and back to the same format must produce a valid, semantically equivalent result. No fields should be silently dropped.
- **A → IR → B** (cross-format): converting from one format to another must produce valid output for the target format, even if the source format has fields with no direct equivalent.
- **A → IR → B → IR → A** (full round-trip): a message that goes through two conversions and back must remain usable. This is the gateway's actual execution path — the request converts inbound, the response converts outbound.

When adding or modifying converter logic, write tests that cover at least the first two scenarios. The gateway's cross-format routing depends on all converters agreeing on IR semantics — a change that breaks one converter's IR output can cascade into failures for every other converter.

## Testing

Tests live under `tests/converters/<name>/` mirroring the converter structure. Key patterns:

- **Unit tests** per ops module — test individual conversion functions in isolation
- **Round-trip tests** — convert A → IR → A and assert equivalence
- **Cross-format tests** — convert A → IR → B and validate the output against format B's spec
- **Streaming tests** — verify stream event ordering and lifecycle

Run `make test` to execute the full suite (excluding integration tests that require API keys).

## Gateway Ops Layer

!!! note "Converter ops vs. gateway ops"
    The converter layer splits conversion logic by concern (`content_ops`,
    `message_ops`, `tool_ops`, `config_ops`).  The **gateway ops layer** is
    a separate concept — every gateway-side mutation is an `OpsBase` subclass
    with automatic audit recording.

### OpsBase Protocol

```
Route handler → constructs Ops subclass → calls execute()
                                            ├── _run()        (business logic)
                                            └── _record()     (audit recording)
                                                 ├── _message()  (human summary)
                                                 └── _details()  (structured metadata)
```

`execute()` runs the operation via `_run()`, then records the result via
`_record()`.  On failure, `_record(result=None, error=exc)` is called
with severity auto-escalated to WARNING, and the original exception is
re-raised.  Recording errors are logged but never mask the operation
result (best-effort audit).

### OpsContext

`OpsContext` is a lightweight service bag created once at app init in
`setup_admin()`.  It holds references to four subsystems — all optional
(`None` when not configured):

| Slot | Type | Purpose |
|------|------|---------|
| `ops_log` | `OpsLog` | Audit event log |
| `request_log` | `RequestLog` | Proxy request traffic log |
| `metrics` | `MetricsCollector` | In-process metrics |
| `persistence` | `PersistenceManager` | SQLite persistence |

### Module Organization

| Module | Contents |
|--------|----------|
| `ops/base.py` | `OpsContext`, `OpsBase` ABC |
| `ops/proxy.py` | `OpsProxyRequest` — hot-path telemetry, overrides `_record` to write `request_log` + `metrics` |
| `ops/data.py` | Data mutation ops (`OpsClearData`, `OpsCleanupData`, `OpsTrimData`, `OpsVacuum`, `OpsRebuildMetrics`, `OpsClearOpsLog`, `OpsPrune`, `OpsPeriodicCleanup`) |
| `ops/security.py` | Auth/session ops (`OpsPasswordChange`, `OpsTokenRotate`, `OpsSessionLogoutAll`) |
| `ops/keys.py` | API key management ops (`OpsKeyCreate`, `OpsKeyUpdate`, `OpsKeyDelete`, `OpsKeyRotate`) |

### Usage Pattern

Route handlers construct an Ops subclass with `OpsContext` and call
`execute()` — recording happens automatically:

```python
from llm_rosetta.gateway.ops import OpsClearData

result = await OpsClearData(request.app.ops_ctx, table="request_log").execute()
```

### OpsProxyRequest (Special Case)

`OpsProxyRequest` overrides `_record()` to write to `request_log` +
`metrics` instead of `ops_log`.  Proxy requests are far too frequent
for the audit log.  Its `_run()` is a no-op — the actual proxy handling
happens outside the ops class.  This class unifies the telemetry
recording interface under the same `execute()` contract.

## Gateway Routing Layer

`gateway/routing/` owns which provider serves a request and, once the request
has been converted, which credential and upstream session identity it is sent
with.  Those are one layer because they read the same inputs — the model, the
client, the conversation — but they cannot be one call.

### Two call points

A pipeline sits between the two decisions.  The provider must be known before
the request body is converted, because the conversion depends on the target
format.  The credential and the session id are keyed on the *converted*
request, because their fingerprints are taken from IR messages.  So the layer
keeps one context per request and runs two chains against it:

```
request ──→ GatewayConfig.resolve(ctx)     provider chosen (before conversion)
            pipeline.ir_request            body converted to IR
        ──→ _apply_request_stages(ctx)     credential and session id (after)
```

`SelectionContext` is created at the edge of the gateway and filled stage by
stage.  The routes and provider tables are installed up front
(`install_route()`), so no stage needs a back-reference to the config object.

### Stages

A stage is one decision.  Each declares the context fields it reads
(`requires`) and writes (`provides`), and `RoutingChain` checks those
declarations when the chain is built: a stage slotted ahead of something it
needs raises `StageOrderError` at construction, not on the first request that
would fail.

| Chain | Stage | Decides |
|-------|-------|---------|
| provider | `AnchorStage` | The provider routing would pick with no preference applied.  The full route's counter advances here. |
| provider | `SameFormatStage` | Realigns the anchor onto a client-format provider, in tiers, when `server.prefer_same_format` is on. |
| request | `ProviderKeyAffinity` | Which key of a key ring serves the conversation: the client plus a two-message IR prefix. |
| request | `SessionHeaderAffinity` | The session id sent in a shim-declared header: the client plus the first message. |

The two request stages always run and each declines when the selected provider
has no use for it — a single key, or no declared session header.  The
same-format stage is only in the chain when the preference is on, because it is
a preference: with the flag off, nothing about selection changes.

### Route identity

`ModelRoute` carries a `route_id` under which its strategy keeps counters.  A
subset of a model's providers — one format, one shim endpoint, one account — is
a route of its own with a distinct id, so tier counters survive across requests
and advance independently of the full route's counter.  Nothing in this layer
reads history: every input is derived from the request in hand, which is what
lets the gateway run without a store.

### Contrast with the Ops layer

The ops layer deliberately has no registry and no chain: its members are a
closed set of mutations whose order never matters.  Routing is the opposite —
stages are ordered, the order is what the checks protect, and adding a stage is
expected — so it is the one layer in the gateway that has a chain.
