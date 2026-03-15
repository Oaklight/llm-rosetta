---
title: Using Converters
---

# Using Converters

## Creating a Converter

```python
from llmir import OpenAIChatConverter, AnthropicConverter

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
from llmir import OpenAIChatConverter, GoogleGenAIConverter

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
