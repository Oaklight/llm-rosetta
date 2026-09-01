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
