---
title: Provider Reasoning Behavior
---

# Provider Reasoning Behavior

The [Reasoning / Thinking](reasoning.md) guide describes how LLM-Rosetta normalizes reasoning parameters through its IR layer. This page documents **how each provider actually behaves** — which effort values it accepts, how it handles disabled reasoning, what `thinking_type` semantics it enforces, and how reasoning metadata survives cross-provider round-trips.

All data below comes from empirical probing against live provider APIs, not documentation alone. See [Probing methodology](#probing-methodology) for details.

!!! tip "Shim configuration reference"
    Each provider's shim YAML encodes the behaviors documented here. See [Provider Shims](shims.md) for the shim architecture and `provider.yaml` format.

## Effort Value Acceptance

Which `reasoning_effort` / `output_config.effort` values each provider accepts. Values not listed are rejected with a 400 error unless otherwise noted.

### Official Upstream APIs

Probed 2026-06-10 against official endpoints with real API keys ([#185](https://github.com/Oaklight/llm-rosetta/issues/185#issuecomment-4667473346)).

| Provider | Endpoint | `none` | `minimal` | `low` | `medium` | `high` | `xhigh` | `max` |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **OpenAI** | Chat (`reasoning_effort`) | — | — | ✅ | ✅ | ✅ | — | — |
| **OpenAI** | Responses (`reasoning.effort`) | — | — | ✅ | ✅ | ✅ | — | — |
| **Anthropic** | Messages (`output_config.effort`) | — | — | ✅ | ✅ | ✅ | ✅ | ✅ |
| **MiniMax** | Anthropic (`output_config.effort`) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **MiniMax** | Chat (`reasoning_effort`) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **OpenRouter** | Chat (`reasoning_effort`) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ |
| **Volcengine** | Chat (`reasoning_effort`) | ❌ | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ |
| **DeepSeek** | Chat (`reasoning_effort`) | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ |

!!! note
    Google GenAI does not use effort strings. It controls reasoning via `thinking_config.thinking_budget` (integer token count) and `thinking_config.thinking_level` (`"minimal"` / `"low"` / `"medium"` / `"high"`).

### Argo Gateway

Probed 2026-05-23 and 2026-06-09 ([#220](https://github.com/Oaklight/llm-rosetta/issues/220#issuecomment-4526638344)).

Argo proxies requests to upstream providers but applies its own validation. The Anthropic endpoint uses `output_config.effort`:

| Model | `effort` alone | `adaptive` + `effort` | `effort=low` |
|---|:---:|:---:|:---:|
| haiku45 / sonnet45 / opus41 / opus45 | ✅ | ❌ 400 | ✅ |
| sonnet46 / opus46 | ✅ | ✅ | ✅ |
| opus47 | ✅ | ✅ | ✅ |

Only sonnet46, opus46, and opus47 support `thinking` + `output_config` coexistence.

## Disabled Behavior

How each provider disables reasoning. The shim `disabled` field controls which strategy LLM-Rosetta uses.

| Provider | Strategy | Shim `disabled` | Behavior |
|---|---|---|---|
| **OpenAI** (Chat / Responses) | Omit thinking field | `omit` | No `thinking` / `reasoning` object sent |
| **Anthropic** | Explicit disable | `thinking_disabled` | `thinking: { type: "disabled" }` |
| **Google GenAI** | Zero budget | `thinking_budget_zero` | `thinking_config: { thinking_budget: 0 }` |
| **DeepSeek** | Explicit disable | `thinking_disabled` | `thinking: { type: "disabled" }` |
| **Volcengine** (Chat) | Explicit disable | `thinking_disabled` | `thinking: { type: "disabled" }` |
| **Volcengine** (Responses) | Omit thinking field | `omit` | No `thinking` / `reasoning` object sent |
| **MiniMax** (Anthropic) | Explicit disable | `thinking_disabled` | `thinking: { type: "disabled" }` |
| **MiniMax** (Chat) | Explicit disable | `thinking_disabled` | `thinking: { type: "disabled" }` |
| **OpenRouter** | Omit | `omit` | No `thinking` object sent |
| **Argo** (Anthropic) | Explicit disable | `thinking_disabled` | `thinking: { type: "disabled" }` |
| **Argo** (OpenAI Chat) | Omit | `omit` | No `thinking` object sent |

!!! warning "Volcengine and DeepSeek reject `none`"
    Sending `reasoning_effort: "none"` to Volcengine or DeepSeek returns a 400 error. To disable reasoning on these providers, use `thinking.type: "disabled"` — which is what the shim's `disabled: thinking_disabled` strategy does automatically when the IR mode is `disabled`.

## Thinking Type

Whether the provider uses `thinking.type = "enabled"` (requires `budget_tokens`) or `"adaptive"` (model decides). Some providers and models only accept one or the other.

### Provider-Level Defaults

| Provider | `thinking_type` | Notes |
|---|---|---|
| **OpenAI** | — | No `thinking.type` field; reasoning is implicit on reasoning models |
| **Anthropic** | *(both)* | Accepts `enabled` and `adaptive`; `enabled` requires `budget_tokens` |
| **Google GenAI** | — | Uses `thinking_budget` integer, not type strings |
| **DeepSeek** | — | Uses `thinking.type` but converter handles `auto → adaptive` mapping |
| **Volcengine** | `enabled` | Rejects `adaptive`; shim forces `thinking_type: enabled` |
| **MiniMax** (Chat) | `adaptive` | Rejects `enabled`; shim forces `thinking_type: adaptive` |
| **Argo** (Anthropic) | `enabled` | Default; but see model-level overrides below |

### Argo Anthropic Model-Level Overrides

Probed 2026-05-23 ([#220](https://github.com/Oaklight/llm-rosetta/issues/220#issuecomment-4526638344)). Argo's Anthropic endpoint has model-specific thinking behavior:

| Model | `enabled` (no budget) | `enabled` + `budget_tokens` | `adaptive` | Notes |
|---|:---:|:---:|:---:|---|
| haiku45 / sonnet45 | ❌ budget required | ✅ | ❌ | `enabled` + budget only |
| opus41 / opus45 | ❌ budget required | ✅ | ❌ | `enabled` + budget only |
| sonnet46 / opus46 | ❌ budget required | ✅ | ✅ | Both modes accepted |
| **opus47** | ❌ type not accepted | ❌ type not accepted | ✅ | **`adaptive` only** |

!!! info "opus47 shim override"
    The `argo--anthropic` shim declares `thinking_type: enabled` at the provider level, with a `model_overrides` entry for `claudeopus47: thinking_type: adaptive`. See [PR #256](https://github.com/Oaklight/llm-rosetta/pull/256).

### Automatic Fallback

When a shim declares `thinking_type: enabled` but the request has no `budget_tokens` (required by Anthropic for `type: "enabled"`), the converter automatically falls back to `type: "adaptive"`. This prevents invalid payloads without requiring the client to know about provider constraints.

## Provider Metadata Round-Trip

Reasoning blocks carry provider-specific metadata that must survive cross-provider conversions. LLM-Rosetta preserves these in the IR `provider_metadata` field on content parts.

| Metadata Field | Origin Provider | Carried On | Purpose |
|---|---|---|---|
| `signature` | Anthropic | `ReasoningPart` | Cryptographic signature for thinking block replay |
| `thoughtSignature` | Google GenAI | `provider_metadata.google` | Equivalent of Anthropic signature for Gemini 2.5+ |
| `encrypted_content` | Anthropic | `provider_metadata.anthropic` | Encrypted reasoning content (opaque to converters) |
| `reasoning_details` | OpenAI | `provider_metadata.openai` | Extended reasoning metadata |

### Cross-Provider Signature Handling

When routing between providers (e.g. Anthropic client → Google backend), signatures from the source provider must be preserved so that:

1. The source client can replay conversation history with valid signatures
2. The target provider receives its own signature format (if applicable)

LLM-Rosetta achieves this by storing signatures in `provider_metadata` on each content part during IR conversion. The Anthropic converter serializes `provider_metadata` as `_provider_metadata` on `thinking`, `tool_use`, `tool_result`, and `text` blocks ([PR #257](https://github.com/Oaklight/llm-rosetta/pull/257), [PR #263](https://github.com/Oaklight/llm-rosetta/pull/263)).

## Unsigned Reasoning Blocks

Some clients send conversation history containing `thinking` blocks without valid signatures. This happens when:

- The client strips or never received signatures (e.g. Claude CLI with `signature: ""`)
- The conversation crossed providers and the original signature is not applicable

### The Problem

Providers that validate signatures (like Argo's Anthropic endpoint) reject these blocks:

```text
messages.61.content.0.thinking.signature: Field required
messages.9.content.0: Invalid `signature` in `thinking` block
```

### Shim Policy: `unsigned_reasoning_blocks`

`ReasoningCapability` supports an `unsigned_reasoning_blocks` field with two values:

| Value | Behavior |
|---|---|
| `as_is` (default) | Forward unsigned reasoning blocks as-is. The target provider decides whether to accept or reject them. |
| `preserve` | Remove unsigned reasoning blocks from outbound messages. The reasoning content is preserved in `provider_metadata.anthropic.unsigned_reasoning_blocks` on the IR part for downstream consumers. |

Currently only `argo--anthropic` uses `preserve`. Direct Anthropic API calls use `as_is` because the official API handles unsigned blocks differently from Argo's proxy layer.

When `preserve` filters all reasoning parts from an assistant message, the message is skipped entirely (with a warning) rather than sending an empty `content: []` array.

See [#268](https://github.com/Oaklight/llm-rosetta/issues/268) and [PR #269](https://github.com/Oaklight/llm-rosetta/pull/269) for the implementation.

## Shim Configuration Reference

How `ReasoningCapability` fields in `provider.yaml` map to the behaviors documented above.

```yaml
reasoning:
  disabled: thinking_disabled  # "omit" | "thinking_disabled" | "thinking_budget_zero"
  effort_field: output_config.effort  # Where effort value is placed
  thinking_type: enabled       # Force thinking.type: "enabled" | "adaptive"
  max_effort: high             # Cap IR effort at this level
  unsigned_reasoning_blocks: preserve  # "as_is" | "preserve"
  effort_map:                  # IR effort → provider effort string
    minimal: low
    low: low
    medium: medium
    high: high
    xhigh: xhigh
    max: max
  model_overrides:             # Per-model overrides (keyed by upstream model ID)
    claudeopus47:
      thinking_type: adaptive
```

### Field Reference

| Field | Type | Default | Description |
|---|---|---|---|
| `disabled` | string | `"omit"` | Strategy when IR mode is `disabled`. `omit`: don't send thinking field. `thinking_disabled`: send `type: "disabled"`. `thinking_budget_zero`: send `thinking_budget: 0`. |
| `effort_field` | string | `"reasoning_effort"` | Request field path for effort value. `"none"` to suppress effort emission. |
| `thinking_type` | string | *(none)* | Force outbound `thinking.type`. Applied after converter's own mapping. |
| `max_effort` | string | *(none)* | Highest IR effort level to emit. Higher levels are clamped to this value. |
| `unsigned_reasoning_blocks` | string | `"as_is"` | Policy for outbound reasoning blocks without valid signatures. |
| `effort_map` | map | *(identity)* | Maps IR effort levels to provider-specific strings. Unmapped levels are dropped with a warning. |
| `model_overrides` | map | *(none)* | Per-model overrides for any of the above fields. Keyed by the upstream model ID used in the shim's `model_id_field`. |

### Provider Shim Summary

| Provider | `disabled` | `effort_field` | `thinking_type` | `max_effort` | `unsigned_reasoning_blocks` |
|---|---|---|---|---|---|
| **openai** | `omit` | `reasoning_effort` | — | `high` | `as_is` |
| **openai_responses** | `omit` | `reasoning.effort` | — | `high` | `as_is` |
| **anthropic** | `thinking_disabled` | `output_config.effort` | — | — | `as_is` |
| **google** | `thinking_budget_zero` | `none` | — | — | `as_is` |
| **deepseek** | `thinking_disabled` | `reasoning_effort` | — | — | `as_is` |
| **volcengine** (Chat) | `thinking_disabled` | `reasoning_effort` | `enabled` | `high` | `as_is` |
| **volcengine** (Responses) | `omit` | `reasoning.effort` | — | — | `as_is` |
| **minimax** (Anthropic) | `thinking_disabled` | `output_config.effort` | — | — | `as_is` |
| **minimax** (Chat) | `thinking_disabled` | `reasoning_effort` | `adaptive` | — | `as_is` |
| **openrouter** | `omit` | `reasoning_effort` | — | `xhigh` | `as_is` |
| **argo** (Anthropic) | `thinking_disabled` | `output_config.effort` | `enabled` | — | `preserve` |
| **argo** (OpenAI Chat) | `omit` | `reasoning_effort` | — | — | `as_is` |

## Probing Methodology

All behavior data on this page was obtained by sending real API requests to live provider endpoints and observing accept/reject responses. This approach catches undocumented constraints that provider documentation may not cover.

### Sources

- **Argo parameter probing** — [#220](https://github.com/Oaklight/llm-rosetta/issues/220): 24 non-embedding models probed across both OpenAI Chat and Anthropic endpoints. Covered sampling, thinking modes, effort values, tool schemas, and field constraints.
- **Official API effort probing** — [#185 comment](https://github.com/Oaklight/llm-rosetta/issues/185#issuecomment-4667473346): Effort value acceptance tested against OpenAI, Anthropic (via OpenRouter), DeepSeek, Volcengine, and MiniMax.
- **Unsigned reasoning block discovery** — [#268](https://github.com/Oaklight/llm-rosetta/issues/268): Dev-test replay of Claude CLI session revealed Argo Anthropic rejecting `thinking` blocks with empty signatures.

### Discrepancies Found

Provider documentation does not always match actual API behavior. Notable examples from Argo probing:

| What docs say | What actually happens |
|---|---|
| o-series models reject `temperature` / `top_p` | Accepted and silently ignored |
| gpt5 / gpt5mini / gpt5nano accept `temperature` / `top_p` | Only accept `temperature=1.0`; any other value → 400 |
| opus47 supports `thinking.type: "enabled"` | Only accepts `adaptive` |

These discrepancies are why empirical probing is essential — and why shim configuration exists to encode the actual behavior rather than the documented behavior.
