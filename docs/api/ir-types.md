---
title: IR Reasoning Types
---

# IR Reasoning Types

The IR reasoning vocabulary defines the fixed value sets used internally by LLM-Rosetta. Provider-side values are free strings defined in each provider's shim config.

These types live in [`llm_rosetta.types.ir.reasoning`][llm_rosetta.types.ir.reasoning].

## IRMode

Controls whether the model performs explicit reasoning.

| Value | Description |
|-------|-------------|
| `"auto"` | Model decides when and how much to think |
| `"enabled"` | Explicit thinking, requires `budget_tokens` on some providers |
| `"disabled"` | No thinking |

Provider mappings are configured per-shim via [`thinking_modes`][llm_rosetta.shims.provider_shim.ReasoningCapability].

## IREffort

The canonical effort ladder, ordered from least to most:

```
minimal < low < medium < high < xhigh < max
```

Each shim declares a supported range via `effort_range`. Values outside the range are clamped to the nearest boundary.

## IRVisibility

Controls how thinking output is returned to the caller.

| Value | Description |
|-------|-------------|
| `"auto"` | Provider decides visibility |
| `"concise"` | Brief summary of reasoning |
| `"detailed"` | Full reasoning output |
| `"none"` | Omit reasoning from response |

Provider mappings are configured per-shim via `visibility_modes`.

## ReasoningConfig

The IR reasoning configuration TypedDict, used in `ir_request["reasoning"]`.

::: llm_rosetta.types.ir.reasoning.ReasoningConfig
    options:
      show_source: false
      heading_level: 3
