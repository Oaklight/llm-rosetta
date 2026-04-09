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

## Metadata Preservation (Lossless Round-Trip)

By default, LLM-Rosetta performs **semantic-only** conversion — provider-specific fields that have no IR equivalent are stripped. This is fine for most cross-provider workflows, but when you need a **lossless round-trip** (same provider A → IR → A), you can enable metadata preservation mode.

### How It Works

Pass `metadata_mode="preserve"` via `ConversionContext` to capture provider-specific fields during `from_provider` and re-inject them during `to_provider`:

```python
from llm_rosetta import OpenAIResponsesConverter
from llm_rosetta.converters.base import ConversionContext

converter = OpenAIResponsesConverter()
ctx = ConversionContext(options={"metadata_mode": "preserve"})

# Provider → IR (captures echo fields, per-item metadata)
ir_request = converter.request_from_provider(provider_request, context=ctx)

# ... modify IR as needed ...

# IR → same provider (re-injects preserved fields)
provider_request, warnings = converter.request_to_provider(ir_request, context=ctx)
```

For responses:

```python
ir_response = converter.response_from_provider(provider_response, context=ctx)
provider_response = converter.response_to_provider(ir_response, context=ctx)
```

### What Gets Preserved

Each converter preserves different provider-specific fields:

| Provider | Preserved Fields |
|----------|-----------------|
| OpenAI Responses | 28+ request echo fields (temperature, tools, reasoning, truncation, etc.), per-output-item metadata (id, status, annotations, logprobs), `RESPONSES_REQUIRED_DEFAULTS` |
| Anthropic | `stop_sequence`, `container`, citations, OpenRouter extension usage fields |
| OpenAI Chat | `refusal`, `annotations` on choices |
| Google GenAI | `promptTokensDetails`, `cachedContentTokenCount` in usage |

### Gateway: Automatic Preserve Mode

The LLM-Rosetta Gateway automatically uses preserve mode for all conversions — both streaming and non-streaming. This ensures that when a client sends a request in format A and the upstream is also format A (passthrough scenario), all provider-specific fields survive the round-trip without data loss.

### Strip Mode (Default)

When `metadata_mode` is `"strip"` (the default), only IR-mapped fields survive the conversion. This is the recommended mode for cross-provider workflows where provider-specific fields are irrelevant to the target.

## Provider Dialect Differences

Different LLM providers have subtly different requirements for the same conceptual operations. LLM-Rosetta handles these **dialect differences** automatically during conversion, so you don't have to worry about them.

### Tool Schema Sanitization

Upstream endpoints (especially Vertex AI's OpenAI-compatible layer) reject JSON Schema keywords that are valid per the spec but unsupported by their validation:

- Keywords like `propertyNames`, `$schema`, `const`, `deprecated`, `readOnly`, etc.
- `$ref` / `$defs` — resolved by inlining referenced definitions
- `anyOf` / `oneOf` / `allOf` — flattened into simple typed schemas (e.g. nullable unions)

LLM-Rosetta's `sanitize_schema()` (in `converters.base.tools`) strips these recursively from all tool parameter schemas across all 4 converters.

### Tool Call / Result Pairing (Strict Validation)

Most LLM providers **strictly require** bidirectional pairing between tool calls and tool results. Violations in either direction cause a **400 error**:

| Direction | OpenAI | Anthropic | Google |
|---|---|---|---|
| Tool call without result | **400** | **400** | OK |
| Tool result without call | **400** | **400** | OK |

Representative error messages:

- **OpenAI**: `"An assistant message with 'tool_calls' must be followed by tool messages responding to each 'tool_call_id'."` / `"Invalid parameter: messages with role 'tool' must be a response to a preceding message with 'tool_calls'."`
- **Anthropic**: `"tool_use ids were found without tool_result blocks immediately after: <id>"` / `"unexpected tool_use_id found in tool_result blocks: <id>"`
- **OpenAI Responses API**: `"No tool output found for function call <call_id>."`

Only Google Gemini is lenient about both cases. These mismatches commonly occur when:

- A tool call is interrupted (user cancels mid-execution) — leaves a call without a result.
- Context compaction or content filtering removes an assistant message containing `tool_calls` — leaves tool results without a preceding call.

LLM-Rosetta fixes both directions automatically and emits `WARNING`-level logs so you can trace each fix:

- **Orphaned tool_calls** → a synthetic tool result with placeholder content `"[No output available yet]"` is injected.
- **Orphaned tool_results** → the dangling result messages are removed.

**Cross-format conversion**: All converters fix mismatches at the IR level before format conversion.

**Passthrough / direct use**: Import the format-specific function:

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

Google's REST API and CLI tools (e.g. Gemini CLI) use camelCase (`inlineData`, `mimeType`, `functionCall`, `functionResponse`, `functionDeclarations`, `responseMimeType`, `thinkingConfig`, etc.) while the Python SDK uses snake_case. LLM-Rosetta's Google converter accepts both conventions transparently in all layers — content, tools, config, and response fields. All IR→Provider output uses camelCase for REST API compatibility.

For a comprehensive list of all camelCase/snake_case field pairs and other real-world compatibility issues discovered during live testing, see the [Provider & CLI Compatibility Matrix](compatibility.md).
