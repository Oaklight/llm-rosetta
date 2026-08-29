---
title: 工具调用
---

# 工具调用

LLM-Rosetta 提供统一的工具定义格式，适用于所有提供方。

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

## 跨提供方工具调用

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

工具定义和工具调用结果会自动转换为每个提供方的原生格式。

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

### 提供方支持

| 提供方 | 多模态工具结果 | 处理方式 |
|-------|:------------:|---------|
| Anthropic | 原生支持 | 内容块（text、image、document） |
| OpenAI Responses | 原生支持 | 内容块（input_text、input_image、input_file） |
| Google Gemini | 原生支持 | inline_data 二进制块 |
| OpenAI Chat | 模拟支持 | 双重编码：`json.dumps()` + 携带可视内容的合成用户消息 |

对于 OpenAI Chat，转换器自动处理双重编码——调用方无需编写特殊代码。

## 自定义工具调用（OpenAI Responses API）

OpenAI Responses API 支持 `"type": "custom"` 工具变体，供部分扩展和集成使用。llm-rosetta 对其提供端到端支持：接收、流式传输以及跨提供方转发。

### 请求

在工具对象中将 `"type"` 设为 `"custom"` 即可定义自定义工具：

```json
{
    "type": "custom",
    "name": "my_custom_tool",
    "description": "A custom extension tool",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string"}
        },
        "required": ["query"]
    }
}
```

在 Python 中使用 Responses API 转换器：

```python
from llm_rosetta import OpenAIResponsesConverter

conv = OpenAIResponsesConverter()

ir_request = {
    "model": "gpt-4o",
    "input": [
        {"role": "user", "content": [{"type": "input_text", "text": "Run my custom tool."}]}
    ],
    "tools": [
        {
            "type": "custom",
            "name": "my_custom_tool",
            "description": "A custom extension tool",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        }
    ],
}

responses_req, _ = conv.request_to_provider(ir_request)
```

### 响应

模型调用自定义工具时，输出中会包含一个 `custom_tool_call` 条目。与 `function_call` 不同，其 `input` 字段是**纯文本**，而非 JSON：

```json
{
    "type": "custom_tool_call",
    "id": "ctc_abc123",
    "name": "my_custom_tool",
    "input": "Run query: find all active users"
}
```

llm-rosetta 会将其规范化为 IR 中 `type: "function"` 的工具调用，并附加 `_passthrough` 标记，保留足够信息以便在返回路径上重建原始 `custom_tool_call` 格式。

### 流式传输

自定义工具调用的输入流式传输使用两个专用事件，其行为与 `function_call` 对应事件完全一致：

| 事件 | 描述 |
|------|------|
| `response.custom_tool_call_input.delta` | `input` 纯文本字段的增量片段 |
| `response.custom_tool_call_input.done` | 最终组装完成的 `input` 值 |

无需额外处理——流式转换器会累积各个 delta，并通过与普通函数调用相同的 IR 流式接口将其输出。

### 跨提供方行为

Anthropic 和 Google 没有原生的 `"custom"` 工具类型。当 llm-rosetta 将包含自定义工具的请求转发给这两个提供方时，会合成一个带有单个字符串参数（`input`）的标准 function 工具，确保该工具仍可被调用：

```json
{
    "type": "function",
    "function": {
        "name": "my_custom_tool",
        "description": "A custom extension tool",
        "parameters": {
            "type": "object",
            "properties": {
                "input": {"type": "string"}
            },
            "required": ["input"]
        }
    }
}
```

在返回路径上，合成的 function call 结果会在到达原始客户端之前被转换回 `custom_tool_call` 输出条目——整个往返过程对上游提供方和下游消费者均透明。
