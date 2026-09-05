---
title: Tool Ops
---

# Tool Ops API

!!! info "Renamed in v0.13"
    `to_google_genai()` / `from_google_genai()` → **`to_google_generate()`** / **`from_google_generate()`**.
    Old names remain as deprecated aliases.

The `tool_ops` module provides a **lightweight convenience API** for converting
tool definitions between the IR (Intermediate Representation) and provider-native
formats — without instantiating full converter pipelines.

All imports are lazy, so no provider-specific dependencies are loaded until
the first call for that provider.

## IR ToolDefinition format

Every function in this module consumes or produces the IR `ToolDefinition`
TypedDict:

| Field | Type | Required | Description |
|---|---|---|---|
| `type` | `"function"` \| `"mcp"` | ✓ | Tool kind. Currently only `"function"` is universally supported. |
| `name` | `str` | ✓ | Function name (snake_case recommended). |
| `description` | `str` | ✓ | Plain-language description of what the tool does. |
| `parameters` | `dict` | ✓ | JSON Schema object describing the function arguments. |
| `required_parameters` | `list[str]` | optional | Required parameter names (merged into the JSON Schema on conversion). |
| `metadata` | `dict` | optional | Pass-through extra fields; provider converters may use or ignore these. |

```python
ir_tool: ToolDefinition = {
    "type": "function",
    "name": "get_weather",
    "description": "Get current weather for a city",
    "parameters": {
        "type": "object",
        "properties": {
            "city": {"type": "string", "description": "City name"},
            "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
        },
        "required": ["city"],
    },
}
```

## Provider wire formats

Each provider serialises tools differently. The table below shows the
top-level shape after conversion:

| Provider | Output shape |
|---|---|
| `openai_chat` | `{"type": "function", "function": {"name": …, "description": …, "parameters": {…}}}` |
| `openai_responses` | `{"type": "function", "name": …, "description": …, "parameters": {…}}` |
| `anthropic` | `{"name": …, "description": …, "input_schema": {…}}` |
| `google` | `{"function_declarations": [{"name": …, "description": …, "parameters": {…}}]}` |

## Quick example

```python
from llm_rosetta import tool_ops

ir_tool = {
    "type": "function",
    "name": "get_weather",
    "description": "Get current weather for a city",
    "parameters": {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
    },
}

# ---- IR → provider ----
openai_tool    = tool_ops.to_openai_chat(ir_tool)
responses_tool = tool_ops.to_openai_responses(ir_tool)
anthropic_tool = tool_ops.to_anthropic(ir_tool)
google_tool    = tool_ops.to_google_generate(ir_tool)

# Unified dispatch variant
same_tool = tool_ops.to_provider(ir_tool, provider="anthropic")

# ---- Provider → IR ----
recovered = tool_ops.from_openai_chat(openai_tool)
recovered = tool_ops.from_anthropic(anthropic_tool)
recovered = tool_ops.from_provider(anthropic_tool, provider="anthropic")
```

### Example: Anthropic output

```python
tool_ops.to_anthropic(ir_tool)
# →
{
    "name": "get_weather",
    "description": "Get current weather for a city",
    "input_schema": {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
    },
}
```

### Example: Google GenAI output

```python
tool_ops.to_google_generate(ir_tool)
# →
{
    "function_declarations": [
        {
            "name": "get_weather",
            "description": "Get current weather for a city",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        }
    ]
}
```

!!! note "Google wraps declarations in a list"
    `to_google_generate` always wraps the converted tool inside
    `{"function_declarations": [...]}`.  Likewise, `from_google_generate` can
    return a **list** of IR tools when the input contains multiple declarations.

## Supported providers

| Canonical name | Accepted aliases |
|---|---|
| `openai_chat` | `openai-chat` |
| `openai_responses` | `openai-responses`, `open_responses`, `open-responses` |
| `anthropic` | — |
| `google` | `google-genai` |

## Batch conversion

`tool_ops` converts one tool at a time.  For a list of tools, use a comprehension:

```python
ir_tools = [tool_a, tool_b, tool_c]

# IR → Anthropic
anthropic_tools = [tool_ops.to_anthropic(t) for t in ir_tools]

# Anthropic → IR
ir_recovered = [tool_ops.from_anthropic(t) for t in anthropic_tools]
# filter None in case a tool type is unsupported
ir_recovered = [t for t in ir_recovered if t is not None]
```

For Google, flatten the `function_declarations` list first:

```python
google_tools = [tool_ops.to_google_generate(t) for t in ir_tools]
# google_tools is a list of {"function_declarations": [...]} dicts

# Round-trip: each from_google_generate call may return a list
import itertools
ir_recovered = list(itertools.chain.from_iterable(
    result if isinstance(result, list) else [result]
    for t in google_tools
    if (result := tool_ops.from_google_generate(t)) is not None
))
```

## Error handling

`to_provider` and `from_provider` raise `ValueError` for unrecognised provider
names:

```python
try:
    tool_ops.to_provider(ir_tool, provider="unknown")
except ValueError as exc:
    print(exc)
# Unknown provider: 'unknown'. Supported: openai_chat, openai_responses, ...
```

Per-provider shortcuts (`to_anthropic`, etc.) do not perform name resolution
and never raise `ValueError` for provider names.

---

## Unified dispatch

::: llm_rosetta.tool_ops.to_provider

::: llm_rosetta.tool_ops.from_provider

---

## Per-provider shortcuts

### IR → provider

::: llm_rosetta.tool_ops.to_openai_chat

::: llm_rosetta.tool_ops.to_openai_responses

::: llm_rosetta.tool_ops.to_anthropic

::: llm_rosetta.tool_ops.to_google_generate

### Provider → IR

::: llm_rosetta.tool_ops.from_openai_chat

::: llm_rosetta.tool_ops.from_openai_responses

::: llm_rosetta.tool_ops.from_anthropic

::: llm_rosetta.tool_ops.from_google_generate
