---
title: 工具调用
---

# 工具调用

LLM-Rosetta 提供统一的工具定义格式，适用于所有提供商。

## 以 IR 格式定义工具

```python
from llm_rosetta import ToolDefinition

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "获取指定位置的当前天气",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "城市名称",
                    },
                },
                "required": ["location"],
            },
        },
    }
]
```

## 跨提供商工具调用

```python
from llm_rosetta import OpenAIChatConverter, AnthropicConverter, extract_tool_calls

openai_conv = OpenAIChatConverter()
anthropic_conv = AnthropicConverter()

# 包含工具的 IR 请求
ir_request = {
    "model": "gpt-4o",
    "messages": [
        {"role": "user", "content": [{"type": "text", "text": "巴黎的天气怎么样？"}]}
    ],
    "tools": tools,
    "tool_choice": "auto",
}

# 转换为 OpenAI 格式并调用
openai_req, _ = openai_conv.request_to_provider(ir_request)
response = openai_client.chat.completions.create(**openai_req)
ir_response = openai_conv.response_from_provider(response.model_dump())

# 从 IR 响应中提取工具调用
tool_calls = extract_tool_calls(ir_response["choices"][0]["message"])

# 执行工具并创建结果消息
for tc in tool_calls:
    result = execute_tool(tc["function"]["name"], tc["function"]["arguments"])
    ir_messages.append(create_tool_result_message(tc["id"], result))

# 使用相同的工具结果继续与 Anthropic 对话
ir_request["messages"] = ir_messages
ir_request["model"] = "claude-sonnet-4-20250514"
anthropic_req, _ = anthropic_conv.request_to_provider(ir_request)
```

工具定义和工具调用结果会自动转换为每个提供商的原生格式。
