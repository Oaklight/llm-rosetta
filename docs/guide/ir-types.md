---
title: IR 类型
---

# IR 类型

中间表示使用基于 TypedDict 的类型，实现零开销的类型安全。

## 消息

```python
from llm_rosetta.types.ir import SystemMessage, UserMessage, AssistantMessage, ToolMessage
```

| 类型 | 角色 | 典型内容 |
|------|------|----------|
| `SystemMessage` | `"system"` | TextPart |
| `UserMessage` | `"user"` | TextPart、ImagePart、FilePart |
| `AssistantMessage` | `"assistant"` | TextPart、ToolCallPart、ReasoningPart |
| `ToolMessage` | `"tool"` | ToolResultPart |

## 内容部分

| 部分 | 描述 |
|------|------|
| `TextPart` | 纯文本内容 |
| `ImagePart` | 图片（URL 或 base64） |
| `FilePart` | 文件附件 |
| `ToolCallPart` | 模型发起的函数调用 |
| `ToolResultPart` | 工具执行结果（字符串或多模态内容） |
| `ReasoningPart` | 模型的思维链 |
| `RefusalPart` | 模型的拒绝回复 |
| `CitationPart` | 来源引用 |
| `AudioPart` | 音频内容 |

## IRRequest

```python
ir_request: IRRequest = {
    "model": "gpt-4o",
    "messages": [...],          # Message 列表
    "tools": [...],             # 可选的 ToolDefinition 列表
    "tool_choice": "auto",      # 可选的 ToolChoice
    "generation": {             # 可选的 GenerationConfig
        "temperature": 0.7,
        "max_tokens": 1000,
    },
}
```

## IRResponse

```python
ir_response: IRResponse = {
    "id": "...",
    "model": "gpt-4o",
    "choices": [
        {
            "message": {...},       # AssistantMessage
            "finish_reason": "stop",
        }
    ],
    "usage": {
        "prompt_tokens": 10,
        "completion_tokens": 20,
        "total_tokens": 30,
    },
}
```

## 工具类型

```python
from llm_rosetta import ToolDefinition, ToolChoice

tool: ToolDefinition = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "获取指定位置的天气",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string"},
            },
            "required": ["location"],
        },
    },
}
```

## 多模态工具结果

`ToolResultPart.result` 可以是字符串（纯文本）或内容部分列表（多模态）：

```python
# 纯文本工具结果
tool_msg = create_tool_result_message(call_id, '{"temperature": "72°F"}')

# 多模态工具结果（文本 + 图片）
tool_msg = create_tool_result_message(call_id, [
    {"type": "text", "text": "生成的图表："},
    {"type": "image", "image_data": {"data": "<base64>", "media_type": "image/png"}},
])
```

三个提供商（Anthropic、OpenAI Responses、Google GenAI）原生支持多模态工具结果。OpenAI Chat 使用双重编码策略——工具消息中 `json.dumps()` 加上携带可视内容的合成用户消息——实现无损往返转换。

## 辅助函数

```python
from llm_rosetta import (
    extract_text_content,
    extract_all_text,
    extract_tool_calls,
    create_tool_result_message,
)

# 仅从 TextPart 内容中提取文本
text = extract_text_content(message)

# 从 TextPart 和 ReasoningPart 中提取文本
# （适用于思考模型如 gemini-2.5-flash）
text = extract_all_text(message)

# 从消息中提取工具调用
tool_calls = extract_tool_calls(message)

# 创建工具结果消息（字符串或多模态列表）
tool_msg = create_tool_result_message(tool_call_id, result)
```
