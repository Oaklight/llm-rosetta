---
title: Using Converters
---

# Using Converters

## Creating a Converter

```python
from llm_rosetta import OpenAIChatConverter, AnthropicConverter

converter = OpenAIChatConverter()
```

## Converting Requests

### Provider → IR

```python
ir_request = converter.request_from_provider(provider_request)
```

### IR → Provider

```python
provider_request, warnings = converter.request_to_provider(ir_request)
```

The `warnings` list contains any conversion notes (e.g., unsupported features dropped).

## Converting Responses

```python
# Provider response → IR
ir_response = converter.response_from_provider(provider_response_dict)

# IR → Provider response
provider_response = converter.response_to_provider(ir_response)
```

## Converting Messages Only

For cases where you only need message conversion without the full request/response:

```python
ir_messages = converter.messages_from_provider(provider_messages)
provider_messages, warnings = converter.messages_to_provider(ir_messages)
```

## Cross-Provider Workflow

```python
from llm_rosetta import OpenAIChatConverter, GoogleGenAIConverter

openai_conv = OpenAIChatConverter()
google_conv = GoogleGenAIConverter()

# OpenAI → IR
ir_request = openai_conv.request_from_provider(openai_request)

# IR → Google
google_request, warnings = google_conv.request_to_provider(ir_request)

# Call Google API, get response
google_response = google_client.generate_content(**google_request)

# Google response → IR
ir_response = google_conv.response_from_provider(google_response)
```

## Google: SDK vs REST API Output

By default, the Google converter produces a dict with a nested `config` key designed for the Google GenAI Python SDK (`google.genai`). If you are calling the Google REST API directly (e.g. via `httpx` or `requests`), pass `output_format="rest"` to get a flattened body ready for the HTTP request:

```python
from llm_rosetta import GoogleGenAIConverter

google_conv = GoogleGenAIConverter()

# SDK format (default) — for google.genai SDK
sdk_request, warnings = google_conv.request_to_provider(ir_request)
# sdk_request has: {"contents": [...], "config": {"tools": [...], "temperature": 0.7, ...}}

# REST format — for direct HTTP calls
rest_body, warnings = google_conv.request_to_provider(ir_request, output_format="rest")
# rest_body has: {"contents": [...], "tools": [...], "generationConfig": {"temperature": 0.7, ...}}
```

The `"rest"` format lifts `tools`, `tool_config`, `response_mime_type`, and `response_schema` to the top level, and wraps generation parameters (temperature, top_p, etc.) into a `generationConfig` object — matching the [Google Gemini REST API](https://ai.google.dev/api/generate-content) schema.

## Provider Dialect Differences

Different LLM providers have subtly different requirements for the same conceptual operations. LLM-Rosetta handles these **dialect differences** automatically during conversion, so you don't have to worry about them.

### Tool Schema Sanitization

Upstream endpoints (especially Vertex AI's OpenAI-compatible layer) reject JSON Schema keywords that are valid per the spec but unsupported by their validation:

- Keywords like `propertyNames`, `$schema`, `const`, `deprecated`, `readOnly`, etc.
- `$ref` / `$defs` — resolved by inlining referenced definitions
- `anyOf` / `oneOf` / `allOf` — flattened into simple typed schemas (e.g. nullable unions)

LLM-Rosetta's `sanitize_schema()` (in `converters.base.tools`) strips these recursively from all tool parameter schemas across all 4 converters.

### Orphaned Tool Calls (OpenAI Strict Pairing)

Both the OpenAI Chat Completions API and the Responses API **strictly require** every tool call to have a corresponding tool result. If a tool call is interrupted (e.g. the user cancels mid-execution in an agentic coding tool), the tool call entry remains in the conversation history without a matching result, and OpenAI returns a **400 error**:

- **Chat Completions**: `"An assistant message with 'tool_calls' must be followed by tool messages responding to each 'tool_call_id'."`
- **Responses API**: `"No tool output found for function call <call_id>."`

Other providers (Anthropic, Google) are lenient about this — they simply ignore orphaned tool calls.

LLM-Rosetta handles this automatically during conversion and emits a `WARNING`-level log when orphaned tool calls are detected, so you can trace the fix without it being silent:

- **Cross-format conversion**: Both `OpenAIChatConverter.request_to_provider()` and `OpenAIResponsesConverter.request_to_provider()` fix orphaned tool calls at the IR level before format conversion, injecting synthetic tool result messages for any unmatched `tool_call_id`.
- **Passthrough / direct use**: Import the format-specific function:

```python
# For OpenAI Chat Completions format
from llm_rosetta.converters.openai_chat.tool_ops import fix_orphaned_tool_calls

messages = fix_orphaned_tool_calls(messages)
messages = fix_orphaned_tool_calls(messages, placeholder="[Skipped by user]")

# For OpenAI Responses format
from llm_rosetta.converters.openai_responses.tool_ops import fix_orphaned_tool_calls

items = fix_orphaned_tool_calls(items)
```

### Google camelCase vs snake_case

Google's REST API uses camelCase (`functionDeclarations`, `toolConfig`, `functionCallingConfig`) while the Python SDK uses snake_case (`function_declarations`, `tool_config`). LLM-Rosetta's Google converter accepts both conventions transparently.
