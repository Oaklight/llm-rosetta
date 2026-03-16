---
title: Examples
---

# Examples

LLM-Rosetta includes comprehensive examples demonstrating cross-provider conversations.

## Available Examples

| Example | Description |
|---------|-------------|
| [Cross-Provider Conversation](cross-provider.md) | Multi-turn chat across different providers |
| [Tool Calling](tool-calling.md) | Function calling with cross-provider conversion |

## Standalone API Test Scripts

The `llm_api_simple_tests/` submodule contains standalone test scripts that use official provider SDKs directly (without LLM-Rosetta). These are useful for verifying provider APIs or gateway compatibility independently.

**Supported providers:** OpenAI Chat, Anthropic, Google GenAI, OpenAI Responses

Each provider has 5 scripts: `simple_query`, `multi_round_chat`, `multi_round_image`, `multi_round_function_calling`, `multi_round_comprehensive`

```bash
cd llm_api_simple_tests
pip install -r requirements.txt

# Run with provider-specific env vars
BASE_URL=https://api.openai.com/v1 API_KEY=sk-... MODEL=gpt-4o \
  python scripts/openai_chat/simple_query.py

# Or set OPENAI_API_KEY, ANTHROPIC_API_KEY, etc. in .env
```

See [`llm_api_simple_tests/README.md`](https://github.com/Oaklight/llm_api_simple_tests) for full documentation.

## Running Cross-Provider Examples

The cross-provider example scripts are in the `examples/` directory of the repository:

```bash
git clone https://github.com/Oaklight/llm-rosetta.git
cd llm-rosetta/examples

# Set up API keys
cp .env.example .env
# Edit .env with your API keys

# Run an example
python sdk_based/cross_openai_chat_anthropic.py
```

Examples are available in two variants:

- **SDK-based** (`sdk_based/`) — use provider SDKs directly
- **REST-based** (`rest_based/`) — use raw HTTP requests
