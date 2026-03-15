---
title: Auto Detection
---

# Auto Detection

LLM-Rosetta can automatically detect which provider format a request uses.

## Detecting Provider

```python
from llm_rosetta import detect_provider

provider = detect_provider(request_dict)
# Returns: "openai_chat", "openai_responses", "anthropic", "google", or None
```

## Getting a Converter

```python
from llm_rosetta import get_converter_for_provider

converter = get_converter_for_provider("anthropic")
```

## Convenience Conversion

```python
from llm_rosetta import convert

# Auto-detect source, convert to target
result = convert(
    source_body=openai_request,
    target_provider="anthropic",
    source_provider=None,  # auto-detect
)
```
