# 各提供商工具选择类型定义

本文档总结了 OpenAI、Anthropic 和 Google GenAI 三家提供商的工具选择相关类型定义，包括工具选择参数和工具定义。

## 1. OpenAI 工具选择类型

OpenAI 的工具选择类型主要包括以下几个关键类型：

### 1.1 ChatCompletionToolChoiceOptionParam

这是一个 Union 类型，可以是以下几种类型之一：

- `"none"` - 不使用任何工具
- `"auto"` - 自动选择是否使用工具
- `"required"` - 必须使用工具
- `ChatCompletionAllowedToolChoiceParam` - 允许的工具选择参数
- `ChatCompletionNamedToolChoiceParam` - 指定名称的工具选择参数
- `ChatCompletionNamedToolChoiceCustomParam` - 自定义的指定名称工具选择参数

### 1.2 ChatCompletionNamedToolChoiceParam

指定名称的工具选择参数，继承自`dict`类型：

```python
{
    "function": Required[Function],
    "type": Required[Literal["function"]]
}
```

### 1.3 ChatCompletionFunctionToolParam

函数工具参数，继承自`dict`类型：

```python
{
    "function": Required[FunctionDefinition],
    "type": Required[Literal["function"]]
}
```

## 2. Anthropic 工具选择类型

Anthropic 的工具选择类型非常丰富，主要包括以下几个关键类型：

### 2.1 ToolChoiceParam

工具选择参数，可以是以下几种类型之一：

- `ToolChoiceAnyParam` - 允许使用任何工具
- `ToolChoiceAutoParam` - 自动选择是否使用工具
- `ToolChoiceNoneParam` - 不使用任何工具
- `ToolChoiceToolParam` - 指定使用特定工具

### 2.2 ToolChoiceAnyParam

允许使用任何工具的参数：

```python
{
    "type": Required[Literal["any"]],
    "disable_parallel_tool_use": bool
}
```

### 2.3 ToolChoiceAutoParam

自动选择是否使用工具的参数：

```python
{
    "type": Required[Literal["auto"]],
    "disable_parallel_tool_use": bool
}
```

### 2.4 ToolChoiceNoneParam

不使用任何工具的参数：

```python
{
    "type": Required[Literal["none"]]
}
```

### 2.5 ToolChoiceToolParam

指定使用特定工具的参数：

```python
{
    "name": Required[str],
    "type": Required[Literal["tool"]],
    "disable_parallel_tool_use": bool
}
```

### 2.6 ToolParam

工具参数定义：

```python
{
    "input_schema": Required[Union[InputSchemaTyped, Dict[str, object]]],
    "name": Required[str],
    "cache_control": Optional[CacheControlEphemeralParam],
    "description": str,
    "type": Optional[Literal["custom"]]
}
```

## 3. Google GenAI 工具选择类型

Google GenAI 的工具选择类型相对简单，主要包括以下几个关键类型：

### 3.1 ToolConfig

工具配置，用于所有工具：

```python
{
    "function_calling_config": Optional[FunctionCallingConfig],
    "retrieval_config": Optional[RetrievalConfig]
}
```

### 3.2 Tool

工具详情，模型可能用来生成响应的工具：

```python
{
    "function_declarations": Optional[list[FunctionDeclaration]],
    "retrieval": Optional[Retrieval],
    "google_search_retrieval": Optional[GoogleSearchRetrieval],
    "computer_use": Optional[ComputerUse],
    "file_search": Optional[FileSearch],
    "code_execution": Optional[ToolCodeExecution],
    "enterprise_web_search": Optional[EnterpriseWebSearch],
    "google_maps": Optional[GoogleMaps],
    "google_search": Optional[GoogleSearch],
    "url_context": Optional[UrlContext]
}
```

### 3.3 FunctionDeclaration

定义模型可以生成 JSON 输入的函数：

```python
{
    "behavior": Optional[Behavior],
    "description": Optional[str],
    "name": Optional[str],
    "parameters": Optional[Schema],
    "parameters_json_schema": Optional[Any],
    "response": Optional[Schema],
    "response_json_schema": Optional[Any]
}
```

## 4. 三家提供商工具选择类型对比

### 4.1 共同点

1. **工具选择模式**：三家都支持自动选择工具（auto）和不使用工具（none）的模式。
2. **函数/工具定义**：三家都支持定义函数/工具，包括名称、描述和参数。
3. **参数定义**：三家都使用类似 OpenAPI 的方式定义参数。

### 4.2 差异点

1. **类型系统**：

   - OpenAI：使用 TypedDict 和 Union 类型
   - Anthropic：使用 Pydantic 模型和 TypedDict
   - Google：使用 Pydantic 模型

2. **工具类型丰富度**：

   - OpenAI：主要支持函数工具
   - Anthropic：支持多种工具类型，包括自定义工具、Web 搜索、代码执行等
   - Google：支持多种工具类型，包括函数声明、检索、Google 搜索、代码执行等

3. **并行工具使用**：

   - Anthropic 明确支持控制并行工具使用（`disable_parallel_tool_use`参数）
   - OpenAI 和 Google 没有明确的并行工具使用控制

4. **缓存控制**：
   - Anthropic 支持缓存控制（`cache_control`参数）
   - OpenAI 和 Google 没有明确的缓存控制

## 5. 中间表示（IR）设计建议

基于以上分析，我们可以设计一个统一的工具选择中间表示（IR）：

```python
from typing import Dict, List, Literal, Optional, Union, Any
from typing_extensions import TypedDict, Required, NotRequired

class ToolChoiceIR(TypedDict):
    """工具选择中间表示"""
    mode: Required[Literal["none", "auto", "any", "tool"]]
    tool_name: NotRequired[str]  # 当mode为"tool"时必需
    disable_parallel: NotRequired[bool]  # 控制是否禁用并行工具使用

class ToolDefinitionIR(TypedDict):
    """工具定义中间表示"""
    name: Required[str]
    type: Required[Literal["function", "retrieval", "search", "code_execution", "custom"]]
    description: NotRequired[str]
    parameters: NotRequired[Dict[str, Any]]  # OpenAPI格式的参数定义
    required_parameters: NotRequired[List[str]]  # 必需参数列表
    cache_control: NotRequired[Dict[str, Any]]  # 缓存控制参数
    metadata: NotRequired[Dict[str, Any]]  # 提供商特定的元数据
```

### 转换策略

#### OpenAI → IR

```python
def openai_tool_choice_to_ir(tool_choice):
    if tool_choice == "none":
        return {"mode": "none"}
    elif tool_choice == "auto":
        return {"mode": "auto"}
    elif tool_choice == "required":
        return {"mode": "any"}
    elif isinstance(tool_choice, dict) and tool_choice.get("type") == "function":
        return {
            "mode": "tool",
            "tool_name": tool_choice["function"]["name"]
        }
    # 其他情况...
```

#### Anthropic → IR

```python
def anthropic_tool_choice_to_ir(tool_choice):
    if tool_choice["type"] == "none":
        return {"mode": "none"}
    elif tool_choice["type"] == "auto":
        return {
            "mode": "auto",
            "disable_parallel": tool_choice.get("disable_parallel_tool_use", False)
        }
    elif tool_choice["type"] == "any":
        return {
            "mode": "any",
            "disable_parallel": tool_choice.get("disable_parallel_tool_use", False)
        }
    elif tool_choice["type"] == "tool":
        return {
            "mode": "tool",
            "tool_name": tool_choice["name"],
            "disable_parallel": tool_choice.get("disable_parallel_tool_use", False)
        }
    # 其他情况...
```

#### Google → IR

```python
def google_tool_config_to_ir(tool_config):
    # Google没有明确的工具选择参数，默认为auto
    return {"mode": "auto"}
```

#### IR → 各提供商

反向转换过程类似，根据 IR 中的`mode`字段确定转换为哪种提供商特定的工具选择类型。
