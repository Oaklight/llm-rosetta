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

1. Create a converter directory under `src/llm_rosetta/converters/<name>/`
2. Subclass `BaseConverter` and implement all abstract methods
3. Create ops classes following the pattern above
4. Add a shim under `src/llm_rosetta/shims/providers/<name>/`
5. Add tests under `tests/converters/`
6. Submit a PR

See existing converters for reference.

## Provider-Specific Data Preservation

LLM-Rosetta separates portable IR data, durable provider passthrough data, and per-conversion context state:

| Layer | Lifetime | Purpose |
|-------|----------|---------|
| Portable IR (`Message`, content parts, tools, reasoning) | Serializable and reusable | Semantics that can be translated between provider formats |
| `ProviderPassthroughEvent` / `ProviderPassthroughItem` | Serializable and reusable | Opaque provider-native chunks or items that have no portable representation |
| `ConversionContext` / `StreamContext` | One conversion pipeline | Warnings, options, echo fields, original IDs/status/annotations, and state used while reconstructing a response |

`ProviderPassthroughEvent` carries provider-native streaming chunks. `ProviderPassthroughItem` carries non-stream request/history items, while `IRResponse.provider_passthrough_items` stores independent non-stream output items together with their original positions.

Passthrough data is tagged with a converter dialect such as `openai_responses` or `anthropic`. A matching target dialect restores a copied native payload. A different target format drops the item; semantic items produce a conversion warning, while lifecycle/heartbeat stream events are dropped silently to avoid warning floods.

`ConversionContext` remains necessary because it serves a different role. It is an ephemeral side channel for a single request/response pipeline and is not serialized into conversation history. Provider passthrough carriers are durable IR data that can survive caching, persistence, and a later HTTP request.

### Stream Dispatch

The base `stream_response_to_provider()` implementation uses a class-level dispatch table (`_TO_P_DISPATCH`) to route IR stream events to handler methods. Provider converters customize behavior through the `_post_process_to_provider()` hook rather than reimplementing the dispatch logic.

## Design Philosophy: Why Everything Goes Through IR

The gateway forces IR conversion for all routes, including same-format — this is intentional. The project started as (and still primarily is) a translation library, and the gateway's core value is built on top of that: cross-format conversion is what makes it more than just another proxy.

Same-format IR round-trip is the most basic fidelity check. If we can't round-trip a single format without information loss, then cross-format translation doesn't stand a chance either. That's the main reason everything goes through IR, and the conversion overhead has been kept low enough that it hasn't been a practical issue.

The pipeline had no passthrough path at all until `25924518` (Aug 2026). We added it, together with a fidelity checker, to make it easier to do shadow-comparison testing — diff passthrough output against converted output to catch round-trip regressions. It's a testing tool, not a production path we expect people to rely on.

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
