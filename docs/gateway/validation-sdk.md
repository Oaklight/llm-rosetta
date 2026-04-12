---
title: SDK & Integration Tests
---

# SDK & Integration Tests

This page documents the automated test suites used to validate LLM-Rosetta's converter pipelines with real API calls.

!!! info "Last updated: 2026-04-12"
    Tested with llm-rosetta v0.5.0, argo-proxy v3.0.0b7

## Integration Test Suite (`tests/integration/`)

The integration test suite validates all four converter pipelines using both official SDKs and direct REST calls. Each test covers non-streaming, streaming, tool calls, round-trip conversions, and multi-turn conversations.

### Running the tests

```bash
cd /path/to/llm-rosetta

# Set the proxy endpoint and model
export ARGO_PROXY_URL=http://localhost:44511
export MODEL=argo:gpt-4.1-nano  # or any available model

# Run all integration tests
python -m pytest tests/integration/ -v

# Run a specific test suite
python -m pytest tests/integration/test_google_genai_sdk_e2e.py -v
python -m pytest tests/integration/test_openai_chat_sdk_e2e.py -v
python -m pytest tests/integration/test_openai_responses_rest_e2e.py -v
python -m pytest tests/integration/test_anthropic_rest_e2e.py -v
```

### Results Summary

| Test Suite | Tests | Result |
|-----------|:-----:|:------:|
| OpenAI Chat SDK | 9 | **9/9** ✓ |
| OpenAI Responses SDK | 6 | **6/6** ✓ |
| Anthropic SDK | 8 | **8/8** ✓ |
| Google GenAI SDK | 7 | **7/7** ✓ |
| **Total** | **30** | **30/30** ✓ |

### Test Coverage Per Suite

| Test | OpenAI Chat | OpenAI Responses | Anthropic | Google GenAI |
|------|:-----------:|:----------------:|:---------:|:------------:|
| Non-stream basic text | ✓ | ✓ | ✓ | ✓ |
| Non-stream with image | ✓ | — | — | — |
| Non-stream with tool calls | ✓ | ✓ | ✓ | ✓ |
| Streaming text | ✓ | — | ✓ | — |
| Streaming with tool calls | ✓ | — | ✓ | — |
| Multimodal tool result | ✓ | ✓ | ✓ | ✓ |
| Image input + tool calls | ✓ | ✓ | ✓ | ✓ |
| Request round-trip | ✓ | ✓ | ✓ | ✓ |
| Response round-trip | ✓ | ✓ | ✓ | ✓ |
| Multi-turn conversation | — | — | — | ✓ |

---

## Same-Format CLI Validation

Five CLI tools were tested with their native API format routed through the gateway without cross-format conversion:

| CLI Tool | API Format | Source → Target | Chat | Stream | Tool Calls | Multi-Round |
|----------|-----------|-----------------|:----:|:------:|:----------:|:-----------:|
| [Codex CLI](https://github.com/openai/codex) | OpenAI Responses | passthrough | ✓ | ✓ | ✓ | ✓ |
| [Kilo Code](https://kilocode.ai/) | OpenAI Chat | `openai_chat` → `openai_responses` | ✓ | ✓ | ✓ | ✓ |
| [OpenCode](https://opencode.ai/) | OpenAI Chat | `openai_chat` → `openai_responses` | ✓ | ✓ | ✓ | ✓ |
| [Claude Code](https://docs.anthropic.com/en/docs/claude-code) | Anthropic Messages | passthrough | ✓ | ✓ | ✓ | ✓ |
| [Gemini CLI](https://github.com/google-gemini/gemini-cli) | Google GenAI | `google` → `openai_chat` | ✓ | ✓ | ✓ | ✓ |

---

## SDK Test Suite (`llm_api_simple_tests`)

The [llm_api_simple_tests](https://github.com/Oaklight/llm_api_simple_tests) suite runs 5 standardized tests per provider using official SDKs.

### Anthropic SDK via Gateway

**Configuration**: `ANTHROPIC_BASE_URL=http://localhost:8765`, model `anthropic/claude-3-haiku`

| Test | Description | Status |
|------|-------------|:------:|
| `simple_query.py` | Single-turn streaming query | ✓ |
| `multi_round_chat.py` | 3-round conversation | ✓ |
| `multi_round_function_calling.py` | 3-round tool calling | ✓ |
| `multi_round_comprehensive.py` | 3-round with image + tool calls | ✓ |
| `multi_round_image.py` | 3-round vision conversation | ✓ |

### Google GenAI via Gateway (curl)

Multi-round tool calling tested directly via `curl`:

```bash
# Round 1: function call
curl -s http://localhost:44511/v1beta/models/gemini-2.5-flash:generateContent \
  -H "Content-Type: application/json" \
  -H "x-goog-api-key: your_key" \
  -d '{
    "contents": [{"role": "user", "parts": [{"text": "What is 127 * 389?"}]}],
    "tools": [{"functionDeclarations": [{"name": "calculator", "description": "Calculate math", "parameters": {"type": "object", "properties": {"expression": {"type": "string"}}}}]}],
    "toolConfig": {"functionCallingConfig": {"mode": "ANY"}}
  }'
# Expected: functionCall with calculator({expression: "127 * 389"})

# Round 2: send tool result, ask follow-up
curl -s http://localhost:44511/v1beta/models/gemini-2.5-flash:generateContent \
  -H "Content-Type: application/json" \
  -H "x-goog-api-key: your_key" \
  -d '{
    "contents": [
      {"role": "user", "parts": [{"text": "What is 127 * 389?"}]},
      {"role": "model", "parts": [{"functionCall": {"name": "calculator", "args": {"expression": "127 * 389"}}}]},
      {"role": "user", "parts": [{"functionResponse": {"name": "calculator", "response": {"result": 49403}}}]},
      {"role": "user", "parts": [{"text": "Now add 100 to that"}]}
    ],
    "tools": [{"functionDeclarations": [{"name": "calculator", "description": "Calculate math", "parameters": {"type": "object", "properties": {"expression": {"type": "string"}}}}]}],
    "toolConfig": {"functionCallingConfig": {"mode": "ANY"}}
  }'
# Expected: functionCall with calculator({expression: "49403 + 100"})
```

| Round | Request | Model Response | Status |
|:-----:|---------|---------------|:------:|
| 1 | "What is 127 * 389?" + `calculator` tool | `functionCall: calculator({expression: "127 * 389"})` | ✓ |
| 2 | Tool result `49403`, "add 100 to that" | `functionCall: calculator({expression: "49403 + 100"})` | ✓ |
| 3 | Tool result `49503`, mode=AUTO | Text: "The result is 49503." | ✓ |

Tested with both `gemini-2.5-flash-lite` and `gemini-3.1-flash-lite-preview`.
