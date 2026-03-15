---
title: Changelog
---

# Changelog

## Unreleased

### New Features

- Streaming support with `StreamContext` for stateful chunk processing
- `stream_response_from_provider()` and `stream_response_to_provider()` methods
- `accumulate_stream_to_assistant_message()` helper function
- Cross-provider streaming examples (SDK and REST variants)

### Improvements

- Converter architecture refactored to composition-based Ops pattern
- Added camelCase fallback for Google GenAI REST stream/response fields

## [0.0.1] - 2024-12-01

Initial release.

### Added

- Hub-and-spoke converter architecture with central IR format
- `OpenAIChatConverter` for OpenAI Chat Completions API
- `OpenAIResponsesConverter` for OpenAI Responses API
- `AnthropicConverter` for Anthropic Messages API
- `GoogleGenAIConverter` for Google GenAI API
- Bidirectional conversion: `request_to/from_provider`, `response_to/from_provider`, `messages_to/from_provider`
- IR type system: messages, content parts, tools, configs, request/response
- Auto-detection of provider format via `detect_provider()`
- Convenience `convert()` function for one-step format conversion
- Helper functions: `extract_text_content`, `extract_tool_calls`, `create_tool_result_message`
- Full TypedDict annotations for type safety
- 24 cross-provider example scripts (12 SDK-based, 12 REST-based)
