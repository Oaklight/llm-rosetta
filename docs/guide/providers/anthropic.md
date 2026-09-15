# Anthropic

Anthropic is supported through a dedicated shim for the Messages API, with model-specific overrides for thinking/reasoning behavior across the Claude model family.

## Endpoints

| Shim Name | Base Type | Default Base URL | API Key Env |
|:---|:---|:---|:---|
| `anthropic` | `anthropic` | `https://api.anthropic.com` | `ANTHROPIC_API_KEY` |

## Reasoning Support

| Field | Value |
|:---|:---|
| Effort field | `output_config.effort` |
| Effort range | `low` → `max` |
| Visibility modes | `auto` → `summarized`, `concise` → `summarized`, `detailed` → `summarized`, `none` → `omitted` |

**Thinking modes:**

| IR Mode | Anthropic Value |
|:---|:---|
| `auto` | `adaptive` |
| `enabled` | `enabled` |
| `disabled` | `disabled` |

### Model Overrides

Different Claude models have different thinking support. The shim applies model-specific overrides automatically:

| Model | Thinking Modes | Budget Ratio | Effort |
|:---|:---|:---:|:---|
| `claude-haiku-4-5-20251001` | `enabled`, `disabled` only | 0.8 | Disabled (Haiku rejects effort param) |
| `claude-opus-4-7` | `adaptive` only (enabled → adaptive) | — | Supported |
| `claude-opus-4-8` | `adaptive` only (enabled → adaptive) | — | Supported |
| All others (e.g. Sonnet 4.6, Opus 4.6) | `adaptive`, `enabled`, `disabled` | — | Supported |

!!! note "Haiku Effort Restriction"
    Claude Haiku 4.5 returns a 400 error when the `effort` parameter is sent. The shim automatically suppresses effort emission for this model.

## Transforms

| Type | Transform | Purpose |
|:---|:---|:---|
| IR | `hoist_late_system_messages()` | Moves late system messages to conversation start |
| IR | `auto_cache_breakpoints()` | Automatically inserts cache breakpoints for prompt caching |

## Gateway Configuration

```jsonc
{
  "providers": {
    "anthropic": {
      "shim": "anthropic",
      "api_key": "${ANTHROPIC_API_KEY}"
    }
  }
}
```
