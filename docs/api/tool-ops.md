---
title: Tool Ops
---

# Tool Ops API

!!! info "Renamed in v0.13"
    `to_google_genai()` / `from_google_genai()` → **`to_google_generate()`** / **`from_google_generate()`**.
    The old names were removed in 0.14.0; see [Migrating to 0.14](../guide/migrating-to-0.14.md#removed-shims).

The `tool_ops` module provides a **lightweight convenience API** for converting
tool artifacts between the IR (Intermediate Representation) and provider-native
formats — without instantiating full converter pipelines.

Covers the full tool lifecycle: **definition**, **choice**, **call**, **result**, and **config**.

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
| `tool_group` | `str` | optional | Group the tool was declared in, when the source had containers. Part of the tool's identity: `(name, tool_group)`. |
| `strict` | `bool` | optional | Ask the provider to validate arguments against `parameters`. Chat Completions, Responses and Messages all carry it; the Google formats have no equivalent and drop it. Absent is not `False` — a target that defaults the flag emits its own default. |
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

### Restricting the choice to named tools

`ToolChoice.allowed_tools` narrows "any tool" to a named subset. The wire
spellings disagree, so the IR states the concept once, as tool identities:

```python
ir_choice: ToolChoice = {
    "mode": "any",  # must call one of the listed tools
    "allowed_tools": [{"name": "get_weather"}],
    # {"name": "search", "tool_group": "tasks"} when the tool is grouped
}
```

| Provider | Allow-list wire shape |
|---|---|
| `chat_completions` | `{"type": "allowed_tools", "mode": …, "tools": [{"type": "function", "function": {"name": …}}]}` — read, but **no Chat upstream we measured accepts it** (`api.openai.com` answers 400), so a Chat target warns and widens the choice back to its bare mode |
| `open_responses` | `{"type": "allowed_tools", "mode": …, "tools": [{"type": "function", "name": …}]}`; each entry's `type` must match how the tool was declared, so it is derived from the declaration rather than stored |
| `messages` | no equivalent — the restriction is dropped with a warning |
| `google_generate` | `function_calling_config.allowed_function_names` (a flat list of names) |
| `google_interactions` | `{"allowed_tools": {"mode": "any", "tools": [name, …]}}` |

## Provider wire formats

Each provider serialises tools differently. The table below shows the
top-level shape after conversion:

| Provider | Output shape |
|---|---|
| `chat_completions` | `{"type": "function", "function": {"name": …, "description": …, "parameters": {…}}}` |
| `open_responses` | `{"type": "function", "name": …, "description": …, "parameters": {…}}` |
| `messages` | `{"name": …, "description": …, "input_schema": {…}}` |
| `google_generate` | `{"function_declarations": [{"name": …, "description": …, "parameters": {…}}]}` |
| `google_interactions` | `{"type": "function", "name": …, "description": …, "parameters": {…}}` |

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
openai_tool    = tool_ops.to_chat_completions(ir_tool)
responses_tool = tool_ops.to_open_responses(ir_tool)
anthropic_tool = tool_ops.to_messages(ir_tool)
google_tool    = tool_ops.to_google_generate(ir_tool)
interact_tool  = tool_ops.to_google_interactions(ir_tool)

# Unified dispatch variant
same_tool = tool_ops.to_provider(ir_tool, provider="messages")

# ---- Provider → IR ----
recovered = tool_ops.from_chat_completions(openai_tool)
recovered = tool_ops.from_messages(anthropic_tool)
recovered = tool_ops.from_provider(anthropic_tool, provider="messages")
```

### Example: Anthropic output

```python
tool_ops.to_messages(ir_tool)
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

| Provider name |
|---|
| `chat_completions` |
| `open_responses` |
| `messages` |
| `google_generate` |
| `google_interactions` |

The hyphenated and pre-rename spellings (`openai-chat`, `openai-responses`, `open-responses`, `anthropic`, `google`, `google-genai`, `google-interactions`) are not accepted. They raise `ValueError`, and the message names the replacement; see [Migrating to 0.14](../guide/migrating-to-0.14.md#retired-names).

## Batch conversion

`tool_ops` converts one tool at a time.  For a list of tools, use a comprehension:

```python
ir_tools = [tool_a, tool_b, tool_c]

# IR → Anthropic
anthropic_tools = [tool_ops.to_messages(t) for t in ir_tools]

# Anthropic → IR
ir_recovered = [tool_ops.from_messages(t) for t in anthropic_tools]
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
# Unknown provider: 'unknown'. Supported: chat_completions, open_responses, ...
```

Per-provider shortcuts (`to_messages`, etc.) do not perform name resolution
and never raise `ValueError` for provider names.

---

## Unified dispatch

::: llm_rosetta.tool_ops.to_provider

::: llm_rosetta.tool_ops.from_provider

---


## Full lifecycle dispatch

Beyond tool definitions, `tool_ops` covers the full `BaseToolOps` lifecycle
via unified dispatch functions. These work for all 5 providers.

### Tool choice

```python
ir_choice = {"mode": "auto"}
provider_choice = tool_ops.choice_to_provider(ir_choice, provider="messages")
recovered = tool_ops.choice_from_provider(provider_choice, provider="messages")
```

::: llm_rosetta.tool_ops.choice_to_provider

::: llm_rosetta.tool_ops.choice_from_provider

### Tool call

```python
ir_call = {
    "type": "tool_call",
    "tool_call_id": "call_123",
    "tool_name": "get_weather",
    "tool_input": {"city": "London"},
}
provider_call = tool_ops.call_to_provider(ir_call, provider="chat_completions")
recovered = tool_ops.call_from_provider(provider_call, provider="chat_completions")
```

::: llm_rosetta.tool_ops.call_to_provider

::: llm_rosetta.tool_ops.call_from_provider

### Tool result

```python
ir_result = {
    "type": "tool_result",
    "tool_call_id": "call_123",
    "result": "Sunny, 22°C",
}
provider_result = tool_ops.result_to_provider(ir_result, provider="messages")
recovered = tool_ops.result_from_provider(provider_result, provider="messages")
```

::: llm_rosetta.tool_ops.result_to_provider

::: llm_rosetta.tool_ops.result_from_provider

### Tool config

```python
ir_config = {"tool_choice": "auto"}
provider_config = tool_ops.config_to_provider(ir_config, provider="chat_completions")
recovered = tool_ops.config_from_provider(provider_config, provider="chat_completions")
```

::: llm_rosetta.tool_ops.config_to_provider

::: llm_rosetta.tool_ops.config_from_provider

---

## Per-provider shortcuts

### IR → provider

::: llm_rosetta.tool_ops.to_chat_completions

::: llm_rosetta.tool_ops.to_open_responses

::: llm_rosetta.tool_ops.to_messages

::: llm_rosetta.tool_ops.to_google_generate

::: llm_rosetta.tool_ops.to_google_interactions

### Provider → IR

::: llm_rosetta.tool_ops.from_chat_completions

::: llm_rosetta.tool_ops.from_open_responses

::: llm_rosetta.tool_ops.from_messages

::: llm_rosetta.tool_ops.from_google_generate

::: llm_rosetta.tool_ops.from_google_interactions
