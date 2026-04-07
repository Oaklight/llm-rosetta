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
from llm_rosetta import OpenAIChatConverter, AnthropicConverter
from llm_rosetta.types.ir import extract_tool_calls, create_tool_result_message

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

## 多模态工具结果

工具可以返回丰富内容（文本 + 图片 + 文件）而非纯字符串。适用于生成图表、图解或其他可视化输出的工具。

```python
from llm_rosetta.types.ir import create_tool_result_message

# 返回多模态内容的工具函数
def generate_chart(chart_type="bar"):
    return [
        {"type": "text", "text": f"生成了 {chart_type} 图表："},
        {"type": "image", "image_data": {"data": "<base64>", "media_type": "image/png"}},
    ]

# 执行工具并创建多模态结果消息
result = generate_chart(**tool_call["function"]["arguments"])
tool_msg = create_tool_result_message(tool_call["id"], result)
```

### 提供商支持

| 提供商 | 多模态工具结果 | 处理方式 |
|-------|:------------:|---------|
| Anthropic | 原生支持 | 内容块（text、image、document） |
| OpenAI Responses | 原生支持 | 内容块（input_text、input_image、input_file） |
| Google Gemini | 原生支持 | inline_data 二进制块 |
| OpenAI Chat | 模拟支持 | 双重编码：`json.dumps()` + 携带可视内容的合成用户消息 |

对于 OpenAI Chat，转换器自动处理双重编码——调用方无需编写特殊代码。
