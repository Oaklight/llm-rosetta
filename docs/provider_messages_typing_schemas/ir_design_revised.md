# LLM提供商消息中间表示（IR）设计（修订版）

本文档描述了LLM提供商消息中间表示（Intermediate Representation, IR）的设计，用于在不同LLM提供商（OpenAI、Anthropic、Google GenAI）之间转换消息格式。

## 1. 设计目标

1. **统一表示**：创建一个统一的消息格式，能够表示所有提供商的核心功能
2. **类型安全**：使用TypedDict确保类型安全，同时保持Python 3.8兼容性
3. **易于转换**：提供简单的转换机制，在不同提供商之间转换消息
4. **可扩展性**：支持未来可能添加的新功能和提供商
5. **轻量级**：不依赖额外的库，如Pydantic（可选使用）

## 2. 核心类型定义

我们使用TypedDict来定义IR的核心类型，并使用typing_extensions确保Python 3.8兼容性。

```python
from typing import Dict, List, Literal, Optional, Union, Any
from typing_extensions import TypedDict, Required, NotRequired

# 消息角色类型
RoleType = Literal["system", "user", "assistant", "tool", "function"]

# 内容部分类型
ContentPartType = Literal[
    "text", 
    "image", 
    "file", 
    "tool_call", 
    "tool_result",
    "reasoning"
]

# 工具类型
ToolType = Literal["function", "retrieval", "search", "code_execution", "custom", "mcp"]

# 工具选择模式
ToolChoiceMode = Literal["none", "auto", "any", "tool"]

# 基础内容部分
class ContentPartBase(TypedDict):
    """内容部分基类"""
    type: Required[ContentPartType]

# 文本内容
class TextContentPart(ContentPartBase):
    """文本内容部分"""
    type: Required[Literal["text"]]
    text: Required[str]

# 图片内容
class ImageContentPart(ContentPartBase):
    """图片内容部分"""
    type: Required[Literal["image"]]
    image_url: Required[str]
    detail: NotRequired[Literal["low", "high", "auto"]]

# 文件内容（合并了文档和文件）
class FileContentPart(ContentPartBase):
    """文件内容部分"""
    type: Required[Literal["file"]]
    file_url: Required[str]
    file_type: NotRequired[str]
    file_name: NotRequired[str]

# 工具调用参数
class ToolCallParam(TypedDict):
    """工具调用参数"""
    name: Required[str]
    arguments: Required[Dict[str, Any]]
    id: NotRequired[str]  # OpenAI需要，Anthropic和Google不需要

# 工具调用内容
class ToolCallContentPart(ContentPartBase):
    """工具调用内容部分"""
    type: Required[Literal["tool_call"]]
    tool_call: Required[ToolCallParam]
    tool_type: NotRequired[ToolType]  # 工具类型

# 工具结果内容
class ToolResultContentPart(ContentPartBase):
    """工具结果内容部分"""
    type: Required[Literal["tool_result"]]
    tool_call_id: Required[str]  # 对应的工具调用ID
    result: Required[Union[str, Dict[str, Any]]]
    is_error: NotRequired[bool]  # 是否为错误结果

# 思考过程内容（支持reasoning_content/reasoning_details）
class ReasoningContentPart(ContentPartBase):
    """思考过程内容部分"""
    type: Required[Literal["reasoning"]]
    reasoning: Required[str]

# 内容部分联合类型
ContentPart = Union[
    TextContentPart,
    ImageContentPart,
    FileContentPart,
    ToolCallContentPart,
    ToolResultContentPart,
    ReasoningContentPart
]

# 消息元数据
class MessageMetadata(TypedDict):
    """消息元数据"""
    provider_specific: NotRequired[Dict[str, Any]]  # 提供商特定的元数据
    extensions: NotRequired[Dict[str, Any]]  # 扩展字段

# 消息
class Message(TypedDict):
    """消息"""
    role: Required[RoleType]
    content: Required[Union[str, List[ContentPart]]]
    name: NotRequired[str]  # 用于function/tool角色
    metadata: NotRequired[MessageMetadata]

# 工具定义
class ToolDefinition(TypedDict):
    """工具定义"""
    type: Required[ToolType]
    function: NotRequired[Dict[str, Any]]  # 函数定义
    name: NotRequired[str]  # 工具名称
    description: NotRequired[str]  # 工具描述
    parameters: NotRequired[Dict[str, Any]]  # 参数定义
    required_parameters: NotRequired[List[str]]  # 必需参数列表
    metadata: NotRequired[Dict[str, Any]]  # 元数据

# 工具选择
class ToolChoice(TypedDict):
    """工具选择"""
    mode: Required[ToolChoiceMode]
    tool_name: NotRequired[str]  # 当mode为"tool"时必需
    disable_parallel: NotRequired[bool]  # 控制是否禁用并行工具使用
```

## 3. 转换策略

### 3.1 角色映射

| IR角色      | OpenAI角色   | Anthropic角色 | Google角色 |
|------------|-------------|--------------|-----------|
| system     | system      | (API参数)     | system    |
| user       | user        | user         | user      |
| assistant  | assistant   | assistant    | model     |
| tool       | tool        | (内容块)      | (内容块)   |
| function   | function    | (内容块)      | (内容块)   |

### 3.2 内容转换

#### 3.2.1 文本内容

```python
# IR → OpenAI
if isinstance(content_part, dict) and content_part["type"] == "text":
    return content_part["text"]  # OpenAI可以直接使用字符串

# IR → Anthropic
if isinstance(content_part, dict) and content_part["type"] == "text":
    return {"type": "text", "text": content_part["text"]}

# IR → Google
if isinstance(content_part, dict) and content_part["type"] == "text":
    return genai.types.Part.from_text(content_part["text"])
```

#### 3.2.2 图片内容

```python
# IR → OpenAI
if isinstance(content_part, dict) and content_part["type"] == "image":
    return {
        "type": "image_url",
        "image_url": {
            "url": content_part["image_url"],
            "detail": content_part.get("detail", "auto")
        }
    }

# IR → Anthropic
if isinstance(content_part, dict) and content_part["type"] == "image":
    return {
        "type": "image",
        "source": {
            "type": "url",
            "url": content_part["image_url"]
        }
    }

# IR → Google
if isinstance(content_part, dict) and content_part["type"] == "image":
    return genai.types.Part.from_data(
        mime_type="image/jpeg",  # 根据URL推断MIME类型
        data=fetch_image_data(content_part["image_url"])
    )
```

#### 3.2.3 文件内容

```python
# IR → OpenAI
if isinstance(content_part, dict) and content_part["type"] == "file":
    return {
        "type": "file_url",
        "file_url": {
            "url": content_part["file_url"],
            "name": content_part.get("file_name", "")
        }
    }

# IR → Anthropic
if isinstance(content_part, dict) and content_part["type"] == "file":
    return {
        "type": "document",
        "source": {
            "type": "url",
            "url": content_part["file_url"]
        }
    }

# IR → Google
if isinstance(content_part, dict) and content_part["type"] == "file":
    return genai.types.Part.from_data(
        mime_type=guess_mime_type(content_part.get("file_type", "")),
        data=fetch_file_data(content_part["file_url"])
    )
```

#### 3.2.4 工具调用内容

```python
# IR → OpenAI
if isinstance(content_part, dict) and content_part["type"] == "tool_call":
    return {
        "type": "tool_call",
        "tool_call": {
            "id": content_part["tool_call"].get("id", generate_id()),
            "type": "function",
            "function": {
                "name": content_part["tool_call"]["name"],
                "arguments": json.dumps(content_part["tool_call"]["arguments"])
            }
        }
    }

# IR → Anthropic
if isinstance(content_part, dict) and content_part["type"] == "tool_call":
    return {
        "type": "tool_use",
        "id": content_part["tool_call"].get("id", generate_id()),
        "name": content_part["tool_call"]["name"],
        "input": content_part["tool_call"]["arguments"]
    }

# IR → Google
if isinstance(content_part, dict) and content_part["type"] == "tool_call":
    return genai.types.Part(
        function_call=genai.types.FunctionCall(
            name=content_part["tool_call"]["name"],
            args=content_part["tool_call"]["arguments"]
        )
    )
```

#### 3.2.5 工具结果内容

```python
# IR → OpenAI
if isinstance(content_part, dict) and content_part["type"] == "tool_result":
    # OpenAI使用单独的消息
    return {
        "role": "tool",
        "tool_call_id": content_part["tool_call_id"],
        "content": json.dumps(content_part["result"]) if isinstance(content_part["result"], dict) else content_part["result"]
    }

# IR → Anthropic
if isinstance(content_part, dict) and content_part["type"] == "tool_result":
    return {
        "type": "tool_result",
        "tool_use_id": content_part["tool_call_id"],
        "content": content_part["result"],
        "is_error": content_part.get("is_error", False)
    }

# IR → Google
if isinstance(content_part, dict) and content_part["type"] == "tool_result":
    return genai.types.Part(
        function_response=genai.types.FunctionResponse(
            name="", # Google不需要名称
            response=content_part["result"]
        )
    )
```

#### 3.2.6 思考过程内容

```python
# IR → OpenAI
if isinstance(content_part, dict) and content_part["type"] == "reasoning":
    # OpenAI使用扩展字段
    return {
        "content": "",  # 空内容
        "reasoning_content": content_part["reasoning"]
    }

# IR → Anthropic
if isinstance(content_part, dict) and content_part["type"] == "reasoning":
    return {
        "type": "thinking",
        "thinking": content_part["reasoning"]
    }

# IR → Google
if isinstance(content_part, dict) and content_part["type"] == "reasoning":
    return genai.types.Part(
        thinking=content_part["reasoning"]
    )
```

### 3.3 工具定义转换

```python
# IR → OpenAI
def ir_tool_to_openai(tool: ToolDefinition):
    if tool["type"] == "function":
        return {
            "type": "function",
            "function": {
                "name": tool.get("name", ""),
                "description": tool.get("description", ""),
                "parameters": tool.get("parameters", {}),
                "required": tool.get("required_parameters", [])
            }
        }
    # 其他工具类型...

# IR → Anthropic
def ir_tool_to_anthropic(tool: ToolDefinition):
    if tool["type"] == "function":
        return {
            "name": tool.get("name", ""),
            "description": tool.get("description", ""),
            "input_schema": tool.get("parameters", {})
        }
    # 其他工具类型...

# IR → Google
def ir_tool_to_google(tool: ToolDefinition):
    if tool["type"] == "function":
        return genai.types.FunctionDeclaration(
            name=tool.get("name", ""),
            description=tool.get("description", ""),
            parameters=genai.types.Schema(
                type=genai.types.Type.OBJECT,
                properties={
                    k: create_google_schema_property(v)
                    for k, v in tool.get("parameters", {}).get("properties", {}).items()
                },
                required=tool.get("required_parameters", [])
            )
        )
    # 其他工具类型...
```

### 3.4 工具选择转换

```python
# IR → OpenAI
def ir_tool_choice_to_openai(tool_choice: ToolChoice):
    if tool_choice["mode"] == "none":
        return "none"
    elif tool_choice["mode"] == "auto":
        return "auto"
    elif tool_choice["mode"] == "any":
        return "required"
    elif tool_choice["mode"] == "tool":
        return {
            "type": "function",
            "function": {
                "name": tool_choice["tool_name"]
            }
        }

# IR → Anthropic
def ir_tool_choice_to_anthropic(tool_choice: ToolChoice):
    if tool_choice["mode"] == "none":
        return {"type": "none"}
    elif tool_choice["mode"] == "auto":
        return {
            "type": "auto",
            "disable_parallel_tool_use": tool_choice.get("disable_parallel", False)
        }
    elif tool_choice["mode"] == "any":
        return {
            "type": "any",
            "disable_parallel_tool_use": tool_choice.get("disable_parallel", False)
        }
    elif tool_choice["mode"] == "tool":
        return {
            "type": "tool",
            "name": tool_choice["tool_name"],
            "disable_parallel_tool_use": tool_choice.get("disable_parallel", False)
        }

# IR → Google
def ir_tool_choice_to_google(tool_choice: ToolChoice):
    # Google没有明确的工具选择参数，使用ToolConfig
    return genai.types.ToolConfig(
        function_calling_config=genai.types.FunctionCallingConfig(
            mode="auto"  # 默认为auto
        )
    )
```

## 4. 降级策略

当转换到不支持某些功能的提供商时，我们需要降级策略：

### 4.1 系统消息

- **Anthropic**：Anthropic不支持系统消息，将系统消息转换为API参数
  ```python
  if message["role"] == "system":
      api_params["system"] = message["content"]
      continue  # 跳过此消息
  ```

### 4.2 思考过程

- **OpenAI**：使用扩展字段`reasoning_content`
- **不支持思考过程的提供商**：将思考过程转换为普通文本，并添加标记
  ```python
  if content_part["type"] == "reasoning":
      return {
          "type": "text",
          "text": f"[思考过程]\n{content_part['reasoning']}\n[/思考过程]"
      }
  ```

### 4.3 工具调用

- **不支持MCP工具的提供商**：将MCP工具转换为普通函数工具
  ```python
  if tool["type"] == "mcp":
      return {
          "type": "function",
          "function": {
              "name": f"mcp_{tool.get('name', '')}",
              "description": f"MCP工具: {tool.get('description', '')}",
              "parameters": tool.get("parameters", {})
          }
      }
  ```

## 5. 实现建议

### 5.1 使用TypedDict而非Pydantic

为了确保Python 3.8兼容性和减少依赖，我们建议使用TypedDict而非Pydantic。如果需要运行时验证，可以提供可选的验证函数：

```python
def validate_message(data: Dict[str, Any]) -> Message:
    """验证消息数据"""
    if "role" not in data:
        raise ValueError("Missing required field: role")
    if "content" not in data:
        raise ValueError("Missing required field: content")
    
    # 验证角色
    if data["role"] not in ["system", "user", "assistant", "tool", "function"]:
        raise ValueError(f"Invalid role: {data['role']}")
    
    # 验证内容
    if isinstance(data["content"], list):
        for part in data["content"]:
            validate_content_part(part)
    
    return data  # type: ignore

def validate_content_part(data: Dict[str, Any]) -> ContentPart:
    """验证内容部分数据"""
    if "type" not in data:
        raise ValueError("Missing required field: type")
    
    # 根据类型验证
    if data["type"] == "text":
        if "text" not in data:
            raise ValueError("Missing required field: text")
    # 其他类型验证...
    
    return data  # type: ignore
```

### 5.2 转换器接口

```python
class ProviderConverter:
    """提供商转换器接口"""
    
    def to_provider(self, messages: List[Message], tools: List[ToolDefinition] = None, tool_choice: ToolChoice = None) -> Dict[str, Any]:
        """将IR消息转换为提供商特定格式"""
        raise NotImplementedError
    
    def from_provider(self, provider_messages: Any) -> List[Message]:
        """将提供商特定格式转换为IR消息"""
        raise NotImplementedError

class OpenAIConverter(ProviderConverter):
    """OpenAI转换器"""
    
    def to_provider(self, messages: List[Message], tools: List[ToolDefinition] = None, tool_choice: ToolChoice = None) -> Dict[str, Any]:
        """将IR消息转换为OpenAI格式"""
        openai_messages = []
        for message in messages:
            openai_message = self._convert_message(message)
            openai_messages.append(openai_message)
        
        result = {"messages": openai_messages}
        
        if tools:
            result["tools"] = [ir_tool_to_openai(tool) for tool in tools]
        
        if tool_choice:
            result["tool_choice"] = ir_tool_choice_to_openai(tool_choice)
        
        return result
    
    def _convert_message(self, message: Message) -> Dict[str, Any]:
        """转换单个消息"""
        # 实现转换逻辑...
```

### 5.3 使用typing_extensions确保Python 3.8兼容性

```python
# Python 3.8兼容性导入
try:
    from typing import TypedDict, Required, NotRequired
except ImportError:
    from typing_extensions import TypedDict, Required, NotRequired
```

## 6. 使用示例

```python
# 创建IR消息
messages = [
    {
        "role": "system",
        "content": "你是一个有用的助手。"
    },
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "请帮我查询天气。"
            }
        ]
    }
]

# 创建IR工具定义
tools = [
    {
        "type": "function",
        "name": "get_weather",
        "description": "获取指定位置的天气信息",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "位置，如'北京'"
                }
            },
            "required": ["location"]
        },
        "required_parameters": ["location"]
    }
]

# 创建IR工具选择
tool_choice = {
    "mode": "auto"
}

# 转换为OpenAI格式
openai_converter = OpenAIConverter()
openai_request = openai_converter.to_provider(messages, tools, tool_choice)

# 转换为Anthropic格式
anthropic_converter = AnthropicConverter()
anthropic_request = anthropic_converter.to_provider(messages, tools, tool_choice)

# 转换为Google格式
google_converter = GoogleConverter()
google_request = google_converter.to_provider(messages, tools, tool_choice)
```

## 7. 总结

本文档描述了LLM提供商消息中间表示（IR）的设计，用于在不同LLM提供商之间转换消息格式。IR设计基于TypedDict，确保类型安全和Python 3.8兼容性，同时提供了详细的转换策略和降级策略。

主要更新：
1. 简化内容类型：移除了音频类型，合并了文档和文件类型
2. 添加思考过程支持：增加了`reasoning`内容类型
3. 统一工具调用机制：支持包括MCP工具在内的各种工具类型
4. 移除特殊功能和引用系统：专注于核心功能
5. 更新实现策略：使用TypedDict而非Pydantic，确保Python 3.8兼容性
6. 添加工具选择支持：统一三家提供商的工具选择机制