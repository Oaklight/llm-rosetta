---
title: Cross-Provider Conversation
---

# Cross-Provider Conversation

This example demonstrates a multi-turn conversation that alternates between two different LLM providers, using LLM-Rosetta to convert messages seamlessly.

## Concept

```text
Turn 1: User → Provider A → IR response → append to history
Turn 2: User → IR history → Provider B request → Provider B → IR response
Turn 3: User → IR history → Provider A request → Provider A → IR response
...
```

The conversation history is maintained in IR format. Before each API call, the full history is converted to the target provider's format.

## Example: OpenAI ↔ Anthropic

```python
from llm_rosetta import (
    OpenAIChatConverter, AnthropicConverter,
    extract_text_content, IRRequest, TextPart, UserMessage,
)

openai_conv = OpenAIChatConverter()
anthropic_conv = AnthropicConverter()

# Maintain history in IR format
ir_messages = []

def chat(user_text: str, use_provider: str = "openai"):
    # Add user message to IR history
    ir_messages.append({
        "role": "user",
        "content": [{"type": "text", "text": user_text}],
    })

    # Build IR request
    ir_request: IRRequest = {
        "model": "gpt-4o" if use_provider == "openai" else "claude-sonnet-4-20250514",
        "messages": ir_messages,
        "generation": {"temperature": 0.7, "max_tokens": 1000},
    }

    # Convert to provider format and call API
    if use_provider == "openai":
        req, _ = openai_conv.request_to_provider(ir_request)
        response = openai_client.chat.completions.create(**req)
        ir_resp = openai_conv.response_from_provider(response.model_dump())
    else:
        req, _ = anthropic_conv.request_to_provider(ir_request)
        response = anthropic_client.messages.create(**req)
        ir_resp = anthropic_conv.response_from_provider(response.model_dump())

    # Append assistant response to history
    assistant_msg = ir_resp["choices"][0]["message"]
    ir_messages.append(assistant_msg)

    return extract_text_content(assistant_msg)

# Alternate between providers
print(chat("Hello!", "openai"))
print(chat("Tell me more.", "anthropic"))
print(chat("Thanks!", "openai"))
```
