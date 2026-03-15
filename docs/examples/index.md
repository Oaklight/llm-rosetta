---
title: Examples
---

# Examples

LLMIR includes comprehensive examples demonstrating cross-provider conversations.

## Available Examples

| Example | Description |
|---------|-------------|
| [Cross-Provider Conversation](cross-provider.md) | Multi-turn chat across different providers |
| [Tool Calling](tool-calling.md) | Function calling with cross-provider conversion |

## Running Examples

The example scripts are in the `examples/` directory of the repository:

```bash
git clone https://github.com/Oaklight/llmir.git
cd llmir/examples

# Set up API keys
cp .env.example .env
# Edit .env with your API keys

# Run an example
python sdk_based/cross_openai_chat_anthropic.py
```

Examples are available in two variants:

- **SDK-based** (`sdk_based/`) — use provider SDKs directly
- **REST-based** (`rest_based/`) — use raw HTTP requests
