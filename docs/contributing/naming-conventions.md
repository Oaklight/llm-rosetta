---
title: Naming Conventions
---

# Naming Conventions

This page documents the function naming conventions used throughout
LLM-Rosetta's converter layer. Following these conventions ensures that
conversion direction is always unambiguous, regardless of where a function
is defined.

## Domains

LLM-Rosetta converts between two domains:

| Abbreviation | Meaning |
|---|---|
| `ir` | Intermediate Representation — the hub format |
| `p` | Provider — any provider-specific format (OpenAI, Anthropic, etc.) |

## Rule 1: Internal functions use `{source}_X_to_{target}`

All internal/private functions that convert data between domains encode
**both** the source and target in the name:

```python
# IR → Provider: source prefix is ir_, suffix is _to_p
ir_text_to_p(part: IRTextPart) -> dict
ir_tool_definition_to_p(tool: IRTool) -> dict

# Provider → IR: source prefix is p_, suffix is _to_ir
p_text_to_ir(part: dict) -> IRTextPart
p_tool_definition_to_ir(tool: dict) -> IRTool
```

The preposition `_to_` always points toward the **target**. The prefix
always identifies the **source**. There is no `_from_` in internal
function names.

This applies uniformly to:

- **Ops files** (`content_ops.py`, `config_ops.py`, `tool_ops.py`, `message_ops.py`)
- **Streaming handlers** in `converter.py`
- **Build/convert helpers** anywhere in the converter layer

### Streaming handlers

```python
# Provider event → IR event (P→IR)
_handle_p_choice_to_ir(event)
_handle_p_content_block_delta_to_ir(event)

# IR event → Provider event (IR→P)
_handle_ir_text_delta_to_p(event)
_handle_ir_tool_call_start_to_p(event)
```

### Build helpers

```python
# Build IR usage from provider usage data (P→IR)
_build_p_usage_to_ir(p_usage: dict) -> IRUsage

# Build provider usage from IR usage data (IR→P)
_build_ir_usage_to_p(ir_usage: IRUsage) -> dict
```

## Rule 2: Public API uses natural English

The public converter API (methods on the converter class itself) uses
`_to_provider` / `_from_provider` for readability:

```python
# Public methods — natural English, direction via to/from
request_to_provider(ir_request)      # IR → Provider
request_from_provider(p_request)     # Provider → IR
response_to_provider(ir_response)    # IR → Provider
response_from_provider(p_response)   # Provider → IR

stream_response_to_provider(...)     # IR → Provider (streaming)
stream_response_from_provider(...)   # Provider → IR (streaming)
```

These are the only functions where `_from_` is allowed, and only with the
full word `provider` (never abbreviated `_from_p`).

## Rule 3: Domain abbreviations

- Internal functions always use `ir` and `p` — never `provider`,
  `responses`, `anthropic`, or other long forms.
- Public API methods use the full word `provider`.

## Summary

| Scope | Pattern | Example |
|---|---|---|
| Ops files (public) | `ir_X_to_p` / `p_X_to_ir` | `ir_text_to_p()`, `p_text_to_ir()` |
| Streaming handlers (private) | `_handle_ir_X_to_p` / `_handle_p_X_to_ir` | `_handle_ir_text_delta_to_p()` |
| Build/convert helpers (private) | `_{verb}_{source}_X_to_{target}` | `_build_p_usage_to_ir()` |
| Public converter API | `X_to_provider` / `X_from_provider` | `request_to_provider()` |
