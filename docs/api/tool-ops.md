---
title: Tool Ops
---

# Tool Ops API

`tool_ops` 模块提供了一个**轻量级便利 API**，用于在 IR（中间表示）和各提供商原生格式之间转换工具定义——无需实例化完整的转换器管道。

所有导入均为懒加载，只有在首次调用对应提供商的函数时才会加载其依赖。

## IR ToolDefinition 格式

本模块所有函数的输入或输出均为 IR `ToolDefinition` TypedDict：

| 字段 | 类型 | 是否必填 | 描述 |
|---|---|---|---|
| `type` | `"function"` \| `"mcp"` | ✓ | 工具类型。目前各提供商普遍支持 `"function"`。 |
| `name` | `str` | ✓ | 函数名（推荐使用 snake_case）。 |
| `description` | `str` | ✓ | 工具功能的自然语言描述。 |
| `parameters` | `dict` | ✓ | 描述函数参数的 JSON Schema 对象。 |
| `required_parameters` | `list[str]` | 可选 | 必填参数名列表（转换时会合并进 JSON Schema）。 |
| `metadata` | `dict` | 可选 | 额外字段直通；各提供商转换器可自行使用或忽略。 |

```python
ir_tool: ToolDefinition = {
    "type": "function",
    "name": "get_weather",
    "description": "获取城市当前天气",
    "parameters": {
        "type": "object",
        "properties": {
            "city": {"type": "string", "description": "城市名称"},
            "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
        },
        "required": ["city"],
    },
}
```

## 各提供商的 wire 格式

每个提供商对工具的序列化方式各不相同，下表列出转换后的顶层结构：

| 提供商 | 输出结构 |
|---|---|
| `openai_chat` | `{"type": "function", "function": {"name": …, "description": …, "parameters": {…}}}` |
| `openai_responses` | `{"type": "function", "name": …, "description": …, "parameters": {…}}` |
| `anthropic` | `{"name": …, "description": …, "input_schema": {…}}` |
| `google` | `{"function_declarations": [{"name": …, "description": …, "parameters": {…}}]}` |

## 快速示例

```python
from llm_rosetta import tool_ops

ir_tool = {
    "type": "function",
    "name": "get_weather",
    "description": "获取城市当前天气",
    "parameters": {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
    },
}

# ---- IR → 提供商格式 ----
openai_tool    = tool_ops.to_openai_chat(ir_tool)
responses_tool = tool_ops.to_openai_responses(ir_tool)
anthropic_tool = tool_ops.to_anthropic(ir_tool)
google_tool    = tool_ops.to_google_genai(ir_tool)

# 统一调度写法
same_tool = tool_ops.to_provider(ir_tool, provider="anthropic")

# ---- 提供商格式 → IR ----
recovered = tool_ops.from_openai_chat(openai_tool)
recovered = tool_ops.from_anthropic(anthropic_tool)
recovered = tool_ops.from_provider(anthropic_tool, provider="anthropic")
```

### 示例：Anthropic 输出

```python
tool_ops.to_anthropic(ir_tool)
# →
{
    "name": "get_weather",
    "description": "获取城市当前天气",
    "input_schema": {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
    },
}
```

### 示例：Google GenAI 输出

```python
tool_ops.to_google_genai(ir_tool)
# →
{
    "function_declarations": [
        {
            "name": "get_weather",
            "description": "获取城市当前天气",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        }
    ]
}
```

!!! note "Google 将声明包裹在列表中"
    `to_google_genai` 始终将转换后的工具包裹在 `{"function_declarations": [...]}` 中。
    同样，当输入包含多个函数声明时，`from_google_genai` 可能返回**列表**。

## 支持的提供商

| 规范名称 | 可接受的别名 |
|---|---|
| `openai_chat` | `openai-chat` |
| `openai_responses` | `openai-responses`、`open_responses`、`open-responses` |
| `anthropic` | — |
| `google` | `google-genai` |

## 批量转换

`tool_ops` 每次转换一个工具，批量转换使用列表推导式：

```python
ir_tools = [tool_a, tool_b, tool_c]

# IR → Anthropic
anthropic_tools = [tool_ops.to_anthropic(t) for t in ir_tools]

# Anthropic → IR
ir_recovered = [tool_ops.from_anthropic(t) for t in anthropic_tools]
# 过滤掉不支持的工具类型返回的 None
ir_recovered = [t for t in ir_recovered if t is not None]
```

对于 Google，需先展开 `function_declarations` 列表：

```python
google_tools = [tool_ops.to_google_genai(t) for t in ir_tools]
# google_tools 是一列 {"function_declarations": [...]} dict

# 回程转换：from_google_genai 每次可能返回列表
import itertools
ir_recovered = list(itertools.chain.from_iterable(
    result if isinstance(result, list) else [result]
    for t in google_tools
    if (result := tool_ops.from_google_genai(t)) is not None
))
```

## 错误处理

`to_provider` 和 `from_provider` 在提供商名称不可识别时抛出 `ValueError`：

```python
try:
    tool_ops.to_provider(ir_tool, provider="unknown")
except ValueError as exc:
    print(exc)
# Unknown provider: 'unknown'. Supported: openai_chat, openai_responses, ...
```

各提供商的快捷函数（`to_anthropic` 等）不进行名称解析，不会因提供商名称抛出 `ValueError`。

---

## 统一调度

::: llm_rosetta.tool_ops.to_provider

::: llm_rosetta.tool_ops.from_provider

---

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
