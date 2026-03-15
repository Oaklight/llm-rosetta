# LLM-Rosetta Cross-Provider Examples

## Overview

These examples demonstrate **LLM-Rosetta**'s ability to conduct multi-turn conversations across **4 different LLM API standards**, seamlessly switching between providers while maintaining full conversation context through the IR (Intermediate Representation) message format.

The 4 supported API standards are:

- **OpenAI Chat Completions** (`oc`)
- **OpenAI Responses** (`or`)
- **Anthropic Messages** (`an`)
- **Google GenAI** (`gg`)

Each provider pair has both **SDK-based** and **REST-based** implementations, with both **non-stream** and **stream** variants, totaling 24 example scripts covering all 6 unique provider combinations × 2 transport methods × 2 modes (non-stream + stream).

## Conversation Scenario

All examples run the same **8-turn conversation** where two providers alternate handling requests. The scenario is a **travel planning** discussion comparing San Francisco and Tokyo:

| Turn | Provider | Type | Description |
|------|----------|------|-------------|
| 1 | A | Text | Ask about San Francisco as a travel destination |
| 2 | B | Image + Text | Identify the Golden Gate Bridge from a photo |
| 3 | A | Tool Call | `get_current_weather` for San Francisco |
| 4 | B | Tool Call | `get_flight_info` from New York to San Francisco |
| 5 | A | Text | Summarize findings so far |
| 6 | B | Image + Text | Identify Tokyo Tower from a photo, compare cities |
| 7 | A | Tool Call | `get_current_weather` for Tokyo |
| 8 | B | Text | Final recommendation based on all gathered info |

### Features Covered

- **Pure text** generation and multi-turn context
- **Image analysis** (vision) with URL-based images
- **Tool calling** (function call + result return loop)
- **Cross-provider context sharing** via IR messages
- **Streaming responses** with real-time text output (stream examples)

### Tools

Two mock tools are available:

- `get_current_weather` — Returns mock weather data for a given location
- `get_flight_info` — Returns mock flight information between two cities

## File Structure

```
examples/
├── README.md              # This file (English)
├── README_zh.md           # Chinese version
├── common.py              # Shared resources: tool definitions, conversation
│                          # turns, helper functions, provider config loaders,
│                          # image URL-to-base64 conversion, stream helpers
├── tools.py               # Legacy tool definitions (used by older examples)
├── sdk_based/             # SDK-based examples (using provider SDKs)
│   ├── cross_oc_an.py             # OpenAI Chat ↔ Anthropic
│   ├── cross_oc_an_stream.py      # OpenAI Chat ↔ Anthropic (stream)
│   ├── cross_oc_gg.py             # OpenAI Chat ↔ Google GenAI
│   ├── cross_oc_gg_stream.py      # OpenAI Chat ↔ Google GenAI (stream)
│   ├── cross_oc_or.py             # OpenAI Chat ↔ OpenAI Responses
│   ├── cross_oc_or_stream.py      # OpenAI Chat ↔ OpenAI Responses (stream)
│   ├── cross_an_gg.py             # Anthropic ↔ Google GenAI
│   ├── cross_an_gg_stream.py      # Anthropic ↔ Google GenAI (stream)
│   ├── cross_an_or.py             # Anthropic ↔ OpenAI Responses
│   ├── cross_an_or_stream.py      # Anthropic ↔ OpenAI Responses (stream)
│   ├── cross_gg_or.py             # Google GenAI ↔ OpenAI Responses
│   └── cross_gg_or_stream.py      # Google GenAI ↔ OpenAI Responses (stream)
└── rest_based/            # REST-based examples (using httpx)
    ├── cross_oc_an.py             # OpenAI Chat ↔ Anthropic
    ├── cross_oc_an_stream.py      # OpenAI Chat ↔ Anthropic (stream)
    ├── cross_oc_gg.py             # OpenAI Chat ↔ Google GenAI
    ├── cross_oc_gg_stream.py      # OpenAI Chat ↔ Google GenAI (stream)
    ├── cross_oc_or.py             # OpenAI Chat ↔ OpenAI Responses
    ├── cross_oc_or_stream.py      # OpenAI Chat ↔ OpenAI Responses (stream)
    ├── cross_an_gg.py             # Anthropic ↔ Google GenAI
    ├── cross_an_gg_stream.py      # Anthropic ↔ Google GenAI (stream)
    ├── cross_an_or.py             # Anthropic ↔ OpenAI Responses
    ├── cross_an_or_stream.py      # Anthropic ↔ OpenAI Responses (stream)
    ├── cross_gg_or.py             # Google GenAI ↔ OpenAI Responses
    └── cross_gg_or_stream.py      # Google GenAI ↔ OpenAI Responses (stream)
```

### Provider Abbreviations

| Abbreviation | Provider |
|--------------|----------|
| `oc` | OpenAI Chat Completions |
| `or` | OpenAI Responses |
| `an` | Anthropic Messages |
| `gg` | Google GenAI |

## Environment Setup

### API Keys

Each provider requires its own API key. Set them as environment variables or use a `.env` file in the project root (loaded automatically via `python-dotenv`).

#### Required Environment Variables

| Variable | Provider | Required |
|----------|----------|----------|
| `OPENAI_API_KEY` | OpenAI (Chat & Responses) | For OpenAI examples |
| `ANTHROPIC_API_KEY` | Anthropic | For Anthropic examples |
| `GOOGLE_API_KEY` | Google GenAI | For Google examples |

#### Optional Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_BASE_URL` | `https://api.openai.com/v1` | OpenAI API base URL |
| `OPENAI_MODEL` | `gpt-4o-mini` | OpenAI model name |
| `OPENAI_RESPONSES_API_KEY` | Falls back to `OPENAI_API_KEY` | Separate key for Responses API |
| `OPENAI_RESPONSES_BASE_URL` | Falls back to `OPENAI_BASE_URL` | Separate base URL for Responses API |
| `OPENAI_RESPONSES_MODEL` | Falls back to `OPENAI_MODEL` | Separate model for Responses API |
| `ANTHROPIC_BASE_URL` | `https://api.anthropic.com` | Anthropic API base URL |
| `ANTHROPIC_MODEL` | `claude-sonnet-4-20250514` | Anthropic model name |
| `GOOGLE_MODEL` | `gemini-2.0-flash` | Google GenAI model name |

#### Example `.env` File

```bash
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini
ANTHROPIC_API_KEY=sk-ant-...
ANTHROPIC_MODEL=claude-sonnet-4-20250514
GOOGLE_API_KEY=AIza...
GOOGLE_MODEL=gemini-2.0-flash
```

### Conda Environment

```bash
conda activate llm-rosetta
```

### Dependencies

Make sure the provider SDKs are installed for SDK-based examples:

```bash
pip install openai anthropic google-genai python-dotenv httpx
```

## Running Examples

### Basic Usage

```bash
# Run from the project root directory
python examples/sdk_based/cross_oc_an.py
python examples/rest_based/cross_oc_an.py

# Stream examples
python examples/sdk_based/cross_oc_an_stream.py
python examples/rest_based/cross_oc_an_stream.py
```

### Network Proxy

OpenAI and Google APIs may require a proxy in restricted network environments. Use `proxychains -q` to route traffic through a proxy:

```bash
# OpenAI + Anthropic (OpenAI needs proxy)
proxychains -q python examples/sdk_based/cross_oc_an.py

# OpenAI + Google (both need proxy)
proxychains -q python examples/sdk_based/cross_oc_gg.py

# Anthropic + Google (Google needs proxy)
proxychains -q python examples/sdk_based/cross_an_gg.py

# Anthropic + OpenAI Responses (OpenAI needs proxy)
proxychains -q python examples/sdk_based/cross_an_or.py
```

Anthropic is typically accessible without a proxy.

## Common Utilities

The `common.py` module provides shared resources used by all example scripts:

- **Tool definitions**: `TOOLS_SPEC` — IR-format tool definitions for `get_current_weather` and `get_flight_info`
- **Mock tool execution**: `execute_tool()` — Returns mock results for tool calls
- **Conversation turns**: `CONVERSATION_TURNS` — The 8-turn conversation scenario definition
- **Message building**: `build_user_message()` — Constructs IR `UserMessage` from turn info
- **Tool call processing**: `process_tool_calls()` — Extracts, executes, and appends tool results
- **Display helpers**: `print_turn_header()`, `print_assistant_response()`, `print_tool_calls()`
- **Provider config loaders**: `get_openai_chat_config()`, `get_openai_responses_config()`, `get_anthropic_config()`, `get_google_config()`
- **Image handling**: `download_image_as_base64()`, `convert_image_urls_to_inline()` — For Google GenAI compatibility
- **Stream helpers** (used by stream examples):
    - `accumulate_stream_to_assistant_message()` — Accumulates IR stream events (`TextDeltaEvent`, `ToolCallStartEvent`, `ToolCallDeltaEvent`) into a complete IR assistant message
    - `print_stream_event()` — Prints stream events in real-time with type-specific formatting (text deltas inline, tool call names, finish reasons, etc.)

## Architecture

### IR Messages as Shared State

The core idea is that all providers share a single list of **IR (Intermediate Representation) messages**. Each provider's converter translates between IR format and the provider's native format:

```
                    ┌─────────────────────┐
                    │   IR Messages List  │
                    │  (shared state)     │
                    └──────────┬──────────┘
                               │
              ┌────────────────┼────────────────┐
              │                │                │
              ▼                ▼                ▼
     ┌────────────────┐ ┌───────────┐ ┌────────────────┐
     │ OpenAI Chat    │ │ Anthropic │ │ Google GenAI   │
     │ Converter      │ │ Converter │ │ Converter      │
     └────────────────┘ └───────────┘ └────────────────┘
```

### Conversion Flow

For each turn, the flow is:

1. Build an IR `UserMessage` and append it to the shared `ir_messages` list
2. Call `converter.request_to_provider(ir_request)` to convert IR → provider-native format
3. Send the request to the provider (via SDK or REST)
4. Call `converter.response_from_provider(response)` to convert provider response → IR format
5. Extract the assistant message and append it to `ir_messages`

### Tool Call Loop

When the assistant responds with tool calls:

1. Extract tool calls from the IR assistant message
2. Execute each tool (mock implementation) and create IR tool result messages
3. Append tool results to `ir_messages`
4. Send the updated conversation back to the **same provider** for a follow-up response

### Stream Mode

The stream examples follow a similar flow but with key differences:

1. Build an IR request and convert it via `converter.request_to_provider(ir_request)` (same as non-stream)
2. Send the request with **streaming enabled** (`stream=True`)
3. For each incoming chunk, call `converter.stream_response_from_provider(chunk, context=ctx)` to convert provider-specific SSE events into **IR stream events**
4. A `StreamContext` object maintains cross-chunk state (e.g., tracking tool call IDs, choice indices)
5. IR stream events include typed events such as `TextDeltaEvent`, `ToolCallStartEvent`, `ToolCallDeltaEvent`, `FinishEvent`, etc.
6. After the stream completes, call `accumulate_stream_to_assistant_message(all_events)` to reconstruct the full IR assistant message from the collected events
7. The reconstructed message is appended to `ir_messages` just like in non-stream mode

This approach enables **real-time text output** while maintaining full compatibility with the IR message format for cross-provider context sharing.

### Image Handling Differences

Different providers handle images differently:

- **OpenAI Chat / Responses**: Support image URLs directly, but may fail to download certain URLs from conversation history. The examples strip image parts from history when sending to OpenAI (`_strip_images()`).
- **Anthropic**: Supports both image URLs and base64 inline data natively.
- **Google GenAI**: Does **not** support image URLs directly. The examples convert image URLs to inline base64 data before sending to Google (`convert_image_urls_to_inline()`).