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
