---
title: Reasoning Capability
---

# Reasoning Capability

The `ReasoningCapability` dataclass declares how a provider handles reasoning and thinking configuration. It is the shim-level control surface for the three dimensions of reasoning behavior.

For a conceptual overview, see the [Reasoning / Thinking guide](../guide/reasoning.md).

## Three Dimensions

### Thinking Toggle (`thinking_modes`)

Maps IR reasoning mode to provider-specific thinking type values.

- `None` — provider does not support a thinking block (safe default; standard OpenAI, Google)
- `dict` — IR mode → provider value mapping

```yaml
# Anthropic
thinking_modes:
  auto: adaptive
  enabled: enabled
  disabled: disabled

# DeepSeek (no auto support)
thinking_modes:
  enabled: enabled
  disabled: disabled
```

IR modes not present in the map are silently dropped. This prevents unsupported thinking blocks from reaching the upstream API.

`thinking_default` sets the IR mode when the request has no explicit mode. Must be a key in `thinking_modes`.

### Effort (`effort_field`, `effort_range`)

`effort_field` is the provider-side field path where the effort value is placed:

| Value | Output Structure |
|-------|-----------------|
| `reasoning_effort` | `{"reasoning_effort": v}` |
| `reasoning.effort` | `{"reasoning": {"effort": v}}` |
| `output_config.effort` | `{"output_config": {"effort": v}}` |
| `thinking_level` | `{"thinking_config": {"thinking_level": v}}` |
| `none` | Effort not emitted |

`effort_range` is a `(floor, ceiling)` tuple of IR effort values. Values outside the range are clamped to the nearest boundary. `None` means the full IR ladder (minimal–max) is supported.

```yaml
# OpenAI: minimal–high supported, xhigh/max clamped to high
effort_range: [minimal, high]

# Anthropic: low–max, minimal clamped to low
effort_range: [low, max]

# Argo OpenAI: full range
# effort_range not set = no clamping
```

### Visibility (`visibility_modes`)

Maps IR summary values to provider-specific visibility controls.

- `None` — use converter default behavior
- `dict` — IR value → provider value; values not in the map are omitted

```yaml
# Anthropic (thinking.display)
visibility_modes:
  auto: summarized
  concise: summarized
  detailed: summarized
  none: omitted

# OpenAI (reasoning.summary)
visibility_modes:
  auto: auto
  concise: concise
  detailed: detailed
```

## Other Fields

### Budget (`budget_ratio`)

When set, derives `budget_tokens` as `max(1024, int(max_tokens × ratio))`, clamped to `max_tokens − 1`. Used when the IR requests `mode: enabled` without an explicit `budget_tokens`.

### Response Handling (`unsigned_blocks`)

Controls how unsigned (non-redacted) reasoning blocks are handled in responses. Values: `"as_is"` (default), `"preserve"`.

### Replay (`replay_missing`)

Some thinking-mode upstreams require the model's reasoning to be passed back once a request carries `tools`. If the history does not end on a `user` turn and any assistant message omits the field, the upstream answers `400 The reasoning_content in the thinking mode must be passed back to the API`. A client that replays history across a format boundary (for example Claude Code → `messages` → `chat_completions`) can drop the reasoning for some turns, so a faithful conversion still trips this.

- `"as_is"` (default) — send the replayed history unchanged.
- `"fill_empty"` — declare the requirement. After source → IR conversion, the capability pass adds an empty reasoning part to every assistant message that lacks one, and the target converter renders it in its own dialect (Chat Completions' `reasoning_content: ""`). An empty value satisfies the upstream — verified live on `deepseek-flash` — so the request completes.

The requirement belongs to the **model**, not to a vendor. Declaring it on a shim-wide default protects only that one endpoint; the same model served by another vendor needs its own declaration. For a vendor that fronts several model families, put it in `model_overrides`:

```yaml
# opencode--chat_completions: only the DeepSeek thinking models need replay
reasoning:
  effort_field: reasoning_effort
  effort_range: [minimal, high]
  model_overrides:
    deepseek-v4.1-flash:
      replay_missing: fill_empty
```

`backfill_reasoning_content()` remains available as a body transform for callers that build wire bodies themselves, but shims should declare `replay_missing` instead: a transform mounted on one vendor shim does not cover the same model behind another endpoint.

## Per-Model Overrides

Provider YAMLs support `model_overrides` for model-specific reasoning behavior:

```yaml
reasoning:
  thinking_modes:
    auto: adaptive
    enabled: enabled
    disabled: disabled
  effort_range: [low, max]
  model_overrides:
    claude-haiku-4-5-20251001:
      # Haiku only supports enabled+budget, no auto
      thinking_modes: {enabled: enabled, disabled: disabled}
      budget_ratio: 0.8
      effort_field: none
    claude-opus-4-7:
      # Opus 4.7+ only supports adaptive
      thinking_modes: {auto: adaptive, enabled: adaptive, disabled: disabled}
```

Each model override inherits unset fields from the provider-level config.

## API Reference

::: llm_rosetta.shims.provider_shim.ReasoningCapability
    options:
      show_source: false
      heading_level: 3
