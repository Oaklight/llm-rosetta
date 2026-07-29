---
title: Multi-Turn State Bridge
---

# Multi-Turn State Bridge

When a gateway proxies between two different API formats, each HTTP turn runs an independent conversion pipeline with its own `ConversionContext`. Provider-specific metadata that lives in the context is destroyed after each turn. This page explains the two-layer architecture that bridges state across turns.

## The Problem

Consider a gateway that accepts OpenAI Chat requests and forwards them to a Responses API upstream:

```text
Turn 1 (response):
  Responses upstream → IR → Chat response → client
  ↑ ConversionContext A (destroyed after response completes)

Turn 2 (request):
  Client Chat request → IR → Responses upstream
  ↑ ConversionContext B (brand new, empty)
```

The Responses API uses two identity fields on `function_call` items:

- `id` (e.g., `fc_abc123`) — output item identity
- `call_id` (e.g., `call_xyz789`) — correlation key for matching results

The IR only has `tool_call_id` (mapped to `call_id`). The `id` is stored in `provider_metadata["responses_item_id"]` during Turn 1, but Context A is gone by Turn 2.

This is not a Responses-only problem. Every format has provider-specific data that needs to survive across turns:

| Format | Cross-turn data | Example |
|--------|----------------|---------|
| Responses | `responses_item_id`, `namespace`, `include` | Codex subagent routing |
| Chat | `reasoning_details`, `encrypted_content` | o-series reasoning models |
| Anthropic | `citations`, `cache_control` | Claude citation metadata |
| Google | `thought_signature` | Gemini 2.5+/3 required |

## Architecture: Two Layers

### Layer 1: Converter (library users)

The converter layer provides a standard API on `ConversionContext` for exporting and importing bridge state. Converters don't know about sessions, caches, or HTTP — they just read from and write to the context.

```python
from llm_rosetta.converters.base.context import ConversionContext

# After response conversion — extract state to bridge
ctx1 = ConversionContext()
ir_response = converter.response_from_provider(response, context=ctx1)
bridge_state = ctx1.get_bridge_state()
# bridge_state is a plain dict, safe to serialize (JSON, pickle, etc.)

# Before next request conversion — inject previous state
ctx2 = ConversionContext()
ctx2.set_bridge_state(bridge_state)
provider_request = converter.request_to_provider(ir_request, context=ctx2)
```

Library users manage storage themselves — in-memory dict, Redis, database, or any other mechanism that fits their architecture.

#### What's in bridge state?

Each converter contributes its own keys to the bridge state dict. The structure is converter-scoped to avoid conflicts:

```python
{
    "tool_call_metadata": {
        "call_xyz789": {
            "responses_item_id": "fc_abc123",
            "namespace": "multi_agent_v1.agent_name"
        }
    }
}
```

Only data that (a) has no IR equivalent and (b) is needed on the next turn is included. Ephemeral state like `_sequence_number` or `current_block_index` is excluded.

### Layer 2: Gateway (automatic bridging)

The gateway layer provides `TurnBridge` — a process-level cache that automatically extracts bridge state after each response and injects it before each request. Gateway users don't write any bridging code.

```text
┌─────────────────────────────────────────────────────┐
│ Gateway                                             │
│                                                     │
│  ┌───────────┐    ┌────────────┐    ┌───────────┐   │
│  │ Pipeline 1 │───▶│ TurnBridge │───▶│ Pipeline 2 │  │
│  │ (response) │    │  (cache)   │    │ (request)  │  │
│  └───────────┘    └────────────┘    └───────────┘   │
│                                                     │
└─────────────────────────────────────────────────────┘
```

`TurnBridge` replaces the previous `ProviderMetadataStore` with a broader scope:

| Capability | `ProviderMetadataStore` (old) | `TurnBridge` (new) |
|-----------|------------------------------|-------------------|
| Tool call `provider_metadata` | ✅ | ✅ |
| Response identity (id, model) | ❌ | ✅ |
| Response echo fields | ❌ | ✅ |
| Streaming events | ✅ (tool_call_start only) | ✅ (all events with bridge data) |
| TTL and eviction | ✅ | ✅ |

## Component Overview

The full system has three distinct component groups:

```text
Converter layer (library)
├── ConversionContext        — per-pipeline state carrier
│   ├── get_bridge_state()   — export cross-turn data
│   └── set_bridge_state()   — import previous turn data
├── ProviderPassthroughEvent — IR opaque streaming event (no IR equivalent)
└── ProviderPassthroughItem  — IR opaque non-stream item (no IR equivalent)

Gateway layer
└── TurnBridge               — automatic cross-turn cache with TTL
    ├── cache_from_context()  — extract after response pipeline
    ├── inject_into_context() — inject before request pipeline
    ├── cache_from_stream_event() — extract from streaming events
    └── clear()
```

These components serve different purposes and do not overlap:

- **`ConversionContext`** is the carrier — it threads state through a single pipeline
- **`ProviderPassthroughEvent/Item`** is for data that has no IR representation at all (e.g., `tool_search_call`)
- **`TurnBridge`** is the persistence layer — it stores and retrieves bridge state across HTTP boundaries

## What Cannot Be Bridged

Some data is fundamentally not preservable across format boundaries:

| Data | Reason |
|------|--------|
| `billing`, `completed_at` | Generated by upstream; cannot be synthesized |
| `_sequence_number` | Reset per stream is correct behavior |
| `current_block_index` | Positional counter, not semantic |

Other data can only be bridged within the same process — if the client constructs a fresh request without echoing the full history, the bridge state for those tool calls is not available. This is expected: the bridge supplements client-echoed history, it does not replace it.
