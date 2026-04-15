---
title: Tool Ops
---

# Tool Ops API

`tool_ops` 模块提供了一个轻量级便利 API，用于在 IR（中间表示）和各提供商原生格式之间转换工具定义——无需实例化完整的转换器管道。

## 快速示例

```python
from llm_rosetta import tool_ops

# 定义一个 IR 工具
ir_tool = {
    "type": "function",
    "name": "get_weather",
    "description": "获取城市的当前天气",
    "parameters": {
        "type": "object",
        "properties": {"city": {"type": "string", "description": "城市名称"}},
        "required": ["city"],
    },
}

# 转换为任意提供商格式
openai_tool = tool_ops.to_openai_chat(ir_tool)
anthropic_tool = tool_ops.to_anthropic(ir_tool)
google_tool = tool_ops.to_google_genai(ir_tool)

# 或使用统一调度
tool = tool_ops.to_provider(ir_tool, provider="anthropic")

# 转换回 IR 格式
recovered = tool_ops.from_anthropic(anthropic_tool)
recovered = tool_ops.from_provider(anthropic_tool, provider="anthropic")
```

## 支持的提供商

| 规范名称            | 别名                                 |
|---------------------|--------------------------------------|
| `openai_chat`       | `openai-chat`                        |
| `openai_responses`  | `openai-responses`、`open_responses`、`open-responses` |
| `anthropic`         | —                                    |
| `google`            | `google-genai`                       |

## 统一调度

::: llm_rosetta.tool_ops.to_provider

::: llm_rosetta.tool_ops.from_provider

## 按提供商快捷方法

### IR 转提供商格式

::: llm_rosetta.tool_ops.to_openai_chat

::: llm_rosetta.tool_ops.to_openai_responses

::: llm_rosetta.tool_ops.to_anthropic

::: llm_rosetta.tool_ops.to_google_genai

### 提供商格式转 IR

::: llm_rosetta.tool_ops.from_openai_chat

::: llm_rosetta.tool_ops.from_openai_responses

::: llm_rosetta.tool_ops.from_anthropic

::: llm_rosetta.tool_ops.from_google_genai
