---
title: 自动检测
---

# 自动检测

LLM-Rosetta 可以自动检测请求使用的提供商格式。

## 检测提供商

```python
from llm_rosetta import detect_provider

provider = detect_provider(request_dict)
# 返回："openai_chat"、"openai_responses"、"anthropic"、"google" 或 None
```

## 获取转换器

```python
from llm_rosetta import get_converter_for_provider

converter = get_converter_for_provider("anthropic")
```

## 便捷转换

```python
from llm_rosetta import convert

# 自动检测来源，转换到目标格式
result = convert(
    source_body=openai_request,
    target_provider="anthropic",
    source_provider=None,  # 自动检测
)
```
