# LLM Provider Converter - IR (Intermediate Representation) 设计文档

**版本**: 1.0  
**日期**: 2024-01-10  
**状态**: Final Design

## 目录

1. [设计目标](#设计目标)
2. [核心架构](#核心架构)
3. [类型定义](#类型定义)
4. [设计决策](#设计决策)
5. [转换规则](#转换规则)
6. [扩展性](#扩展性)
7. [实现指南](#实现指南)

---

## 设计目标

### 主要目标

1. **统一接口**: 提供统一的消息格式，屏蔽不同provider的差异
2. **无损转换**: 尽可能保留所有provider的特性，避免信息丢失
3. **易于使用**: 简单场景简单，复杂场景灵活
4. **面向未来**: 良好的扩展性，支持新的内容类型和工具类型
5. **类型安全**: 完整的TypeScript/Python类型定义

### 非目标

- ❌ 不追求完美的双向转换（某些特性可能单向支持）
- ❌ 不强制所有provider支持所有特性（允许降级）
- ❌ 不创造新的消息语义（只做转换，不做增强）

---

## 核心架构

### 架构选择

基于对OpenAI、Anthropic、Google三家provider的深入分析，我们采用**混合架构**：

```
核心：嵌套结构（Anthropic风格）+ 扩展项（特殊场景）
```

**理由**：
- ✅ **嵌套结构**：90%的场景足够用，简洁清晰
- ✅ **扩展项**：10%的特殊场景（工具链、系统事件等）
- ✅ **渐进式复杂度**：简单场景不需要了解扩展项
- ✅ **向后兼容**：旧代码可以忽略扩展项

### 架构图

```
┌─────────────────────────────────────────────────────────┐
│                    IR Input                              │
│  List[Union[Message, ExtensionItem]]                    │
└─────────────────────────────────────────────────────────┘
                          │
        ┌─────────────────┴─────────────────┐
        │                                   │
        ▼                                   ▼
┌───────────────┐                  ┌──────────────────┐
│    Message    │                  │  ExtensionItem   │
│  (90% cases)  │                  │  (10% cases)     │
└───────────────┘                  └──────────────────┘
        │                                   │
        │                          ┌────────┴────────┐
        │                          │                 │
        ▼                          ▼                 ▼
┌───────────────┐          ┌──────────────┐  ┌──────────────┐
│    Content    │          │ SystemEvent  │  │ToolChainNode │
│     Parts     │          │              │  │              │
└───────────────┘          └──────────────┘  └──────────────┘
        │                          │                 │
┌───────┴────────┐                 ▼                 ▼
│                │          ┌──────────────┐  ┌──────────────┐
▼                ▼          │ BatchMarker  │  │SessionControl│
TextPart    ImagePart       └──────────────┘  └──────────────┘
ToolCallPart ToolResultPart
FilePart    ReasoningPart
```

---

## 类型定义

### 1. 核心消息类型

```python
from typing import TypedDict, Union, List, Literal, Dict, Any, Optional
from typing_extensions import Required, NotRequired

# ============================================================================
# 核心消息
# ============================================================================

class Message(TypedDict):
    """
    核心消息类型，代表对话中的一条消息。
    
    这是IR的主要组成部分，90%的场景只需要使用这个类型。
    """
    role: Required[Literal["system", "user", "assistant"]]
    content: Required[List[ContentPart]]
    metadata: NotRequired[MessageMetadata]

class MessageMetadata(TypedDict, total=False):
    """
    消息的元数据，用于存储额外信息。
    
    Examples:
        - 消息ID、时间戳
        - 流式传输状态
        - 自定义标签
    """
    message_id: str
    timestamp: str
    streaming: StreamingMetadata
    custom: Dict[str, Any]

class StreamingMetadata(TypedDict, total=False):
    """流式传输的元数据"""
    is_streaming: bool
    is_final: bool
    chunk_index: int
```

### 2. 内容部分类型

```python
# ============================================================================
# 内容部分（Content Parts）
# ============================================================================

ContentPart = Union[
    TextPart,
    ImagePart,
    FilePart,
    ToolCallPart,
    ToolResultPart,
    ReasoningPart,
]

# ----------------------------------------------------------------------------
# 文本内容
# ----------------------------------------------------------------------------

class TextPart(TypedDict):
    """纯文本内容"""
    type: Required[Literal["text"]]
    text: Required[str]

# ----------------------------------------------------------------------------
# 图像内容
# ----------------------------------------------------------------------------

class ImagePart(TypedDict):
    """图像内容，支持URL或base64"""
    type: Required[Literal["image"]]
    image_url: NotRequired[str]  # URL形式
    image_data: NotRequired[ImageData]  # base64形式
    detail: NotRequired[Literal["auto", "low", "high"]]  # OpenAI特性

class ImageData(TypedDict):
    """Base64编码的图像数据"""
    data: Required[str]  # base64编码
    media_type: Required[str]  # 如 "image/png"

# ----------------------------------------------------------------------------
# 文件内容
# ----------------------------------------------------------------------------

class FilePart(TypedDict):
    """
    文件内容，支持多种文件类型。
    
    Examples:
        - PDF文档
        - 音频文件
        - 视频文件
    """
    type: Required[Literal["file"]]
    file_url: NotRequired[str]  # URL形式
    file_data: NotRequired[FileData]  # base64形式
    file_name: NotRequired[str]
    file_type: NotRequired[str]  # MIME type

class FileData(TypedDict):
    """Base64编码的文件数据"""
    data: Required[str]  # base64编码
    media_type: Required[str]  # 如 "application/pdf"

# ----------------------------------------------------------------------------
# 工具调用
# ----------------------------------------------------------------------------

class ToolCallPart(TypedDict):
    """
    工具调用内容。
    
    使用两层类型系统：
    - type: 固定为 "tool_call"
    - tool_type: 区分不同的工具类型（function, mcp, web_search等）
    
    这样设计避免了类型爆炸，同时保持扩展性。
    """
    type: Required[Literal["tool_call"]]
    tool_call_id: Required[str]
    tool_name: Required[str]
    tool_input: Required[Dict[str, Any]]
    tool_type: NotRequired[Literal[
        "function",
        "mcp",
        "web_search",
        "code_interpreter",
        "file_search",
    ]]  # 默认为 "function"

# ----------------------------------------------------------------------------
# 工具结果
# ----------------------------------------------------------------------------

class ToolResultPart(TypedDict):
    """
    工具调用的结果。
    
    对应一个ToolCallPart，通过tool_call_id关联。
    """
    type: Required[Literal["tool_result"]]
    tool_call_id: Required[str]
    result: Required[Any]  # 可以是字符串、对象等
    is_error: NotRequired[bool]  # 是否是错误结果

# ----------------------------------------------------------------------------
# 推理内容（Reasoning）
# ----------------------------------------------------------------------------

class ReasoningPart(TypedDict):
    """
    推理过程内容（如OpenAI的reasoning）。
    
    用于存储模型的思考过程，通常不显示给用户。
    """
    type: Required[Literal["reasoning"]]
    reasoning: Required[str]
```

### 3. 扩展项类型

```python
# ============================================================================
# 扩展项（Extension Items）
# ============================================================================

ExtensionItem = Union[
    SystemEvent,
    BatchMarker,
    SessionControl,
    ToolChainNode,
]

# ----------------------------------------------------------------------------
# 系统事件
# ----------------------------------------------------------------------------

class SystemEvent(TypedDict):
    """
    系统级事件，用于记录会话状态变化。
    
    Examples:
        - 会话开始/结束
        - 会话暂停/恢复
        - 超时警告
        - 错误事件
    """
    type: Required[Literal["system_event"]]
    event_type: Required[Literal[
        "session_start",
        "session_pause",
        "session_resume",
        "session_timeout",
        "session_end",
        "error",
        "warning",
    ]]
    timestamp: Required[str]  # ISO 8601格式
    event_data: NotRequired[Dict[str, Any]]
    message: NotRequired[str]

# ----------------------------------------------------------------------------
# 批次标记
# ----------------------------------------------------------------------------

class BatchMarker(TypedDict):
    """
    批次标记，用于标记一组相关的操作。
    
    Examples:
        - 并行工具调用的开始/结束
        - 部分结果的进度跟踪
    """
    type: Required[Literal["batch_marker"]]
    batch_id: Required[str]
    batch_type: Required[Literal["start", "end", "partial"]]
    total_items: NotRequired[int]
    completed_items: NotRequired[int]
    metadata: NotRequired[Dict[str, Any]]

# ----------------------------------------------------------------------------
# 会话控制
# ----------------------------------------------------------------------------

class SessionControl(TypedDict):
    """
    会话控制指令，用于控制工具调用的执行。
    
    Examples:
        - 取消工具调用
        - 修改工具调用参数
        - 暂停/恢复工具执行
    """
    type: Required[Literal["session_control"]]
    control_type: Required[Literal[
        "cancel_tool",
        "modify_tool",
        "pause_tool",
        "resume_tool",
    ]]
    target_id: Required[str]  # 目标tool_call_id
    reason: NotRequired[str]
    new_input: NotRequired[Dict[str, Any]]  # 用于modify_tool

# ----------------------------------------------------------------------------
# 工具链节点
# ----------------------------------------------------------------------------

class ToolChainNode(TypedDict):
    """
    工具链节点，用于表示工具调用的依赖关系。
    
    支持DAG结构，一个工具的输出可以作为另一个工具的输入。
    
    Examples:
        - 搜索 → 总结
        - 数据获取 → 分析 → 可视化
    """
    type: Required[Literal["tool_chain_node"]]
    node_id: Required[str]
    tool_call: Required[ToolCallPart]
    depends_on: NotRequired[List[str]]  # 依赖的节点ID列表
    auto_execute: NotRequired[bool]  # 是否自动执行
```

### 4. 顶层类型

```python
# ============================================================================
# 顶层类型
# ============================================================================

# 完整的IR输入（支持扩展项）
IRInput = List[Union[Message, ExtensionItem]]

# 简化的IR输入（只有消息）
IRInputSimple = List[Message]

# 类型守卫
def is_message(item: Union[Message, ExtensionItem]) -> bool:
    """判断是否是Message"""
    return "role" in item

def is_extension_item(item: Union[Message, ExtensionItem]) -> bool:
    """判断是否是ExtensionItem"""
    return "type" in item and item.get("type") in [
        "system_event",
        "batch_marker",
        "session_control",
        "tool_chain_node",
    ]
```

---

## 设计决策

### 决策1: 嵌套结构 vs 扁平结构

**选择**: 嵌套结构（Anthropic风格）

**理由**:
1. ✅ **语义清晰**: 工具调用和结果都是消息内容的一部分
2. ✅ **原子性**: 一条消息可以包含多个内容部分（文本+工具调用）
3. ✅ **灵活性**: 用户可以在返回工具结果的同时补充信息
4. ✅ **一致性**: 所有内容类型都是ContentPart

**权衡**:
- ❌ 转换到OpenAI Responses API需要提取工具调用到消息级别
- ✅ 但这个转换是机械的，不会丢失信息

### 决策2: 角色系统

**选择**: 3种角色（system, user, assistant）

**理由**:
1. ✅ **通用性**: 三家provider都支持这3种角色
2. ✅ **简洁性**: 不引入额外的角色（如OpenAI的tool角色）
3. ✅ **清晰性**: 工具结果放在user消息中，语义明确

**权衡**:
- ❌ OpenAI的tool角色需要转换为user角色
- ✅ 但这个转换是无损的，只是角色名称不同

### 决策3: 两层类型系统

**选择**: 使用`tool_type`字段区分工具类型

**理由**:
1. ✅ **避免类型爆炸**: 不需要为每种工具类型创建新的ContentPart
2. ✅ **扩展性**: 添加新工具类型只需扩展`tool_type`的字面量
3. ✅ **一致性**: 所有工具调用都是ToolCallPart

**示例**:
```python
# 不好的设计（类型爆炸）
ContentPart = Union[
    TextPart,
    FunctionCallPart,
    MCPCallPart,
    WebSearchCallPart,
    CodeInterpreterCallPart,
    # ... 每种工具一个类型
]

# 好的设计（两层类型）
class ToolCallPart(TypedDict):
    type: Literal["tool_call"]
    tool_type: Literal["function", "mcp", "web_search", ...]
```

### 决策4: 扩展项机制

**选择**: 使用Union类型支持扩展项

**理由**:
1. ✅ **渐进式复杂度**: 简单场景只用Message
2. ✅ **向后兼容**: 旧代码可以过滤掉扩展项
3. ✅ **类型安全**: TypeScript/Python类型检查支持
4. ✅ **灵活性**: 可以表示复杂的交互模式

**使用场景**:
- 工具链（跨消息的依赖关系）
- 系统事件（会话状态变化）
- 批次操作（并行工具调用）
- 会话控制（取消/修改工具调用）

### 决策5: Metadata vs 新类型

**选择**: 使用metadata存储非核心信息

**理由**:
1. ✅ **保持核心简洁**: 核心类型只包含必要字段
2. ✅ **扩展性**: metadata可以存储任意信息
3. ✅ **可选性**: metadata是可选的，不影响基本功能

**示例**:
```python
# 流式传输状态放在metadata中
{
    "role": "user",
    "content": [...],
    "metadata": {
        "streaming": {
            "is_streaming": True,
            "is_final": False,
            "chunk_index": 0
        }
    }
}
```

---

## 转换规则

### 1. IR → Anthropic

```python
def to_anthropic(ir_input: IRInput) -> tuple[List[MessageParam], List[str]]:
    """
    转换IR到Anthropic格式。
    
    Returns:
        - messages: Anthropic消息列表
        - warnings: 转换警告列表
    """
    messages = []
    warnings = []
    
    for item in ir_input:
        if is_message(item):
            # 普通消息：直接转换
            messages.append({
                "role": item["role"],
                "content": convert_content_to_anthropic(item["content"])
            })
        elif item.get("type") == "system_event":
            # 系统事件：Anthropic不支持，记录警告
            warnings.append(f"System event ignored: {item['event_type']}")
        elif item.get("type") == "tool_chain_node":
            # 工具链：展开为普通工具调用
            warnings.append("Tool chain converted to sequential calls")
            # 可以展开为多个消息
        # ... 其他扩展项处理
    
    return messages, warnings

def convert_content_to_anthropic(content: List[ContentPart]) -> List[ContentBlock]:
    """转换内容部分到Anthropic格式"""
    blocks = []
    for part in content:
        if part["type"] == "text":
            blocks.append({"type": "text", "text": part["text"]})
        elif part["type"] == "image":
            blocks.append({
                "type": "image",
                "source": {
                    "type": "base64" if "image_data" in part else "url",
                    "data": part.get("image_data", {}).get("data"),
                    "media_type": part.get("image_data", {}).get("media_type"),
                }
            })
        elif part["type"] == "tool_call":
            blocks.append({
                "type": "tool_use",
                "id": part["tool_call_id"],
                "name": part["tool_name"],
                "input": part["tool_input"]
            })
        elif part["type"] == "tool_result":
            blocks.append({
                "type": "tool_result",
                "tool_use_id": part["tool_call_id"],
                "content": part["result"]
            })
        # ... 其他类型
    return blocks
```

### 2. IR → OpenAI Responses

```python
def to_openai_responses(ir_input: IRInput) -> List[ResponseInputItemParam]:
    """
    转换IR到OpenAI Responses格式。
    
    注意：OpenAI Responses使用扁平结构，需要提取工具调用。
    """
    items = []
    
    for item in ir_input:
        if is_message(item):
            # 检查是否有工具调用
            tool_calls = [p for p in item["content"] if p["type"] == "tool_call"]
            other_content = [p for p in item["content"] if p["type"] != "tool_call"]
            
            if tool_calls and item["role"] == "assistant":
                # Assistant消息有工具调用：拆分
                if other_content:
                    # 先添加文本内容
                    items.append({
                        "type": "message",
                        "role": "assistant",
                        "content": convert_content_to_openai(other_content)
                    })
                # 再添加工具调用
                for tc in tool_calls:
                    items.append({
                        "type": "function_call",
                        "name": tc["tool_name"],
                        "call_id": tc["tool_call_id"],
                        "arguments": json.dumps(tc["tool_input"])
                    })
            else:
                # 普通消息
                items.append({
                    "type": "message",
                    "role": item["role"],
                    "content": convert_content_to_openai(item["content"])
                })
        elif item.get("type") == "system_event":
            # OpenAI Responses支持系统事件
            items.append({
                "type": "system_event",
                "event_type": item["event_type"],
                # ... 其他字段
            })
        # ... 其他扩展项处理
    
    return items
```

### 3. IR → Google GenAI

```python
def to_google(ir_input: IRInput) -> List[Content]:
    """
    转换IR到Google GenAI格式。
    """
    contents = []
    
    for item in ir_input:
        if is_message(item):
            # Google使用 "user" 和 "model" 角色
            role = "model" if item["role"] == "assistant" else "user"
            
            contents.append({
                "role": role,
                "parts": convert_content_to_google(item["content"])
            })
        elif item.get("type") == "system_event":
            # Google不支持系统事件，忽略
            pass
        # ... 其他扩展项处理
    
    return contents

def convert_content_to_google(content: List[ContentPart]) -> List[Part]:
    """转换内容部分到Google格式"""
    parts = []
    for part in content:
        if part["type"] == "text":
            parts.append({"text": part["text"]})
        elif part["type"] == "image":
            parts.append({
                "inline_data": {
                    "mime_type": part.get("image_data", {}).get("media_type"),
                    "data": part.get("image_data", {}).get("data")
                }
            })
        elif part["type"] == "tool_call":
            parts.append({
                "function_call": {
                    "name": part["tool_name"],
                    "args": part["tool_input"]
                }
            })
        elif part["type"] == "tool_result":
            parts.append({
                "function_response": {
                    "name": part["tool_call_id"],  # 需要查找对应的工具名
                    "response": part["result"]
                }
            })
        # ... 其他类型
    return parts
```

### 4. 反向转换（Provider → IR）

```python
# Anthropic → IR
def from_anthropic(messages: List[MessageParam]) -> IRInput:
    """从Anthropic格式转换到IR"""
    ir_input = []
    for msg in messages:
        ir_input.append({
            "role": msg["role"],
            "content": convert_content_from_anthropic(msg["content"])
        })
    return ir_input

# OpenAI → IR
def from_openai_responses(items: List[ResponseInputItemParam]) -> IRInput:
    """从OpenAI Responses格式转换到IR"""
    ir_input = []
    current_message = None
    
    for item in items:
        if item["type"] == "message":
            if current_message:
                ir_input.append(current_message)
            current_message = {
                "role": item["role"],
                "content": convert_content_from_openai(item["content"])
            }
        elif item["type"] == "function_call":
            # 合并到当前assistant消息
            if current_message and current_message["role"] == "assistant":
                current_message["content"].append({
                    "type": "tool_call",
                    "tool_call_id": item["call_id"],
                    "tool_name": item["name"],
                    "tool_input": json.loads(item["arguments"]),
                    "tool_type": "function"
                })
        # ... 其他类型
    
    if current_message:
        ir_input.append(current_message)
    
    return ir_input

# Google → IR
def from_google(contents: List[Content]) -> IRInput:
    """从Google GenAI格式转换到IR"""
    ir_input = []
    for content in contents:
        # Google使用 "model" 角色，转换为 "assistant"
        role = "assistant" if content["role"] == "model" else "user"
        ir_input.append({
            "role": role,
            "content": convert_content_from_google(content["parts"])
        })
    return ir_input
```

---

## 扩展性

### 1. 添加新的内容类型

```python
# 步骤1: 定义新的内容类型
class AudioPart(TypedDict):
    """音频内容"""
    type: Required[Literal["audio"]]
    audio_url: NotRequired[str]
    audio_data: NotRequired[AudioData]
    transcript: NotRequired[str]  # 可选的转录文本

class AudioData(TypedDict):
    """Base64编码的音频数据"""
    data: Required[str]
    media_type: Required[str]  # 如 "audio/mp3"

# 步骤2: 添加到ContentPart联合类型
ContentPart = Union[
    TextPart,
    ImagePart,
    FilePart,
    AudioPart,  # 新增
    ToolCallPart,
    ToolResultPart,
    ReasoningPart,
]

# 步骤3: 更新转换函数
def convert_content_to_anthropic(content: List[ContentPart]) -> List[ContentBlock]:
    blocks = []
    for part in content:
        # ... 现有类型处理
        if part["type"] == "audio":
            # 处理音频（如果provider支持）
            blocks.append({
                "type": "audio",
                "source": {
                    "type": "base64",
                    "data": part.get("audio_data", {}).get("data"),
                    "media_type": part.get("audio_data", {}).get("media_type"),
                }
            })
    return blocks
```

### 2. 添加新的工具类型

```python
# 只需扩展tool_type的字面量
class ToolCallPart(TypedDict):
    type: Required[Literal["tool_call"]]
    tool_call_id: Required[str]
    tool_name: Required[str]
    tool_input: Required[Dict[str, Any]]
    tool_type: NotRequired[Literal[
        "function",
        "mcp",
        "web_search",
        "code_interpreter",
        "file_search",
        "database_query",  # 新增
        "api_call",        # 新增
    ]]
```

### 3. 添加新的扩展项

```python
# 步骤1: 定义新的扩展项类型
class CustomExtension(TypedDict):
    """自定义扩展项"""
    type: Required[Literal["custom_extension"]]
    extension_name: Required[str]
    extension_data: Required[Dict[str, Any]]

# 步骤2: 添加到ExtensionItem联合类型
ExtensionItem = Union[
    SystemEvent,
    BatchMarker,
    SessionControl,
    ToolChainNode,
    CustomExtension,  # 新增
]

# 步骤3: 更新转换函数
def to_anthropic(ir_input: IRInput) -> tuple[List[MessageParam], List[str]]:
    messages = []
    warnings = []
    
    for item in ir_input:
        # ... 现有处理
        if item.get("type") == "custom_extension":
            # 处理自定义扩展项
            warnings.append(f"Custom extension ignored: {item['extension_name']}")
    
    return messages, warnings
```

### 4. 版本演进策略

```python
# 使用版本字段支持向后兼容
class Message(TypedDict):
    role: Required[Literal["system", "user", "assistant"]]
    content: Required[List[ContentPart]]
    metadata: NotRequired[MessageMetadata]
    ir_version: NotRequired[str]  # 如 "1.0", "1.1"

# 转换时检查版本
def convert_message(msg: Message) -> Any:
    version = msg.get("ir_version", "1.0")
    if version == "1.0":
        return convert_v1_0(msg)
    elif version == "1.1":
        return convert_v1_1(msg)
    else:
        raise ValueError(f"Unsupported IR version: {version}")
```

---

## 实现指南

### 1. 项目结构

```
src/llmir/
├── types/
│   ├── __init__.py
│   ├── ir.py              # IR类型定义
│   ├── openai.py          # OpenAI类型定义
│   ├── anthropic.py       # Anthropic类型定义
│   └── google.py          # Google类型定义
├── converters/
│   ├── __init__.py
│   ├── base.py            # 基础转换器
│   ├── to_anthropic.py    # IR → Anthropic
│   ├── to_openai.py       # IR → OpenAI
│   ├── to_google.py       # IR → Google
│   ├── from_anthropic.py  # Anthropic → IR
│   ├── from_openai.py     # OpenAI → IR
│   └── from_google.py     # Google → IR
├── utils/
│   ├── __init__.py
│   ├── validators.py      # 类型验证
│   └── helpers.py         # 辅助函数
└── __init__.py
```

### 2. 实现优先级

**Phase 1: 核心功能**
1. ✅ 定义IR类型（`types/ir.py`）
2. ✅ 实现基础转换器（`converters/base.py`）
3. ✅ 实现Anthropic转换（最简单，直接映射）
4. ✅ 实现OpenAI转换（需要处理扁平化）
5. ✅ 实现Google转换（需要处理角色映射）

**Phase 2: 扩展功能**
1. ⏳ 支持扩展项（系统事件、工具链等）
2. ⏳ 支持流式传输
3. ⏳ 支持批次操作
4. ⏳ 支持会话控制

**Phase 3: 优化和工具**
1. ⏳ 性能优化
2. ⏳ 错误处理和验证
3. ⏳ 文档和示例
4. ⏳ 测试覆盖

### 3. 测试策略

```python
# 测试用例结构
tests/
├── unit/
│   ├── test_ir_types.py           # IR类型测试
│   ├── test_to_anthropic.py       # 转换测试
│   ├── test_to_openai.py
│   └── test_to_google.py
├── integration/
│   ├── test_round_trip.py         # 往返转换测试
│   └── test_real_providers.py     # 真实provider测试
└── fixtures/
    ├── ir_examples.py             # IR示例数据
    ├── anthropic_examples.py      # Anthropic示例数据
    ├── openai_examples.py         # OpenAI示例数据
    └── google_examples.py         # Google示例数据

# 测试示例
def test_simple_text_message():
    """测试简单文本消息的转换"""
    ir_msg = {
        "role": "user",
        "content": [{"type": "text", "text": "Hello"}]
    }
    
    # 转换到Anthropic
    anthropic_msg = to_anthropic([ir_msg])
    assert anthropic_msg[0]["role"] == "user"
    assert anthropic_msg[0]["content"][0]["type"] == "text"
    
    # 往返转换
    ir_msg_back = from_anthropic(anthropic_msg)
    assert ir_msg_back == [ir_msg]

def test_tool_call_conversion():
    """测试工具调用的转换"""
    ir_msg = {
        "role": "assistant",
        "content": [
            {"type": "text", "text": "Let me search for that"},
            {
                "type": "tool_call",
                "tool_call_id": "call_123",
                "tool_name": "web_search",
                "tool_input": {"query": "AI news"},
                "tool_type": "web_search"
            }
        ]
    }
    
    # 转换到各provider
    anthropic_msg = to_anthropic([ir_msg])
    openai_msg = to_openai_responses([ir_msg])
    google_msg = to_google([ir_msg])
    
    # 验证转换结果
    # ...
```

### 4. 使用示例

```python
from llmir import IRInput, to_anthropic, to_openai, to_google

# 创建IR消息
ir_input: IRInput = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "What's the weather in Beijing?"}
        ]
    },
    {
        "role": "assistant",
        "content": [
            {"type": "text", "text": "Let me check that for you."},
            {
                "type": "tool_call",
                "tool_call_id": "call_1",
                "tool_name": "get_weather",
                "tool_input": {"city": "Beijing"},
                "tool_type": "function"
            }
        ]
    },
    {
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_call_id": "call_1",
                "result": "Sunny, 25°C"
            }
        ]
    },
    {
        "role": "assistant",
        "content": [
            {"type": "text", "text": "The weather in Beijing is sunny with a temperature of 25°C."}
        ]
    }
]

# 转换到不同provider
anthropic_messages, warnings = to_anthropic(ir_input)
openai_messages = to_openai_responses(ir_input)
google_contents = to_google(ir_input)

# 使用转换后的消息调用API
# anthropic_client.messages.create(messages=anthropic_messages, ...)
# openai_client.chat.completions.create(messages=openai_messages, ...)
# google_model.generate_content(contents=google_contents, ...)
```

---

## 附录

### A. 设计原则总结

1. **简单优先**: 简单场景应该简单，不要过度设计
2. **渐进式复杂度**: 复杂功能通过扩展项实现，不影响基础使用
3. **类型安全**: 充分利用TypeScript/Python的类型系统
4. **向后兼容**: 新版本应该兼容旧版本
5. **文档优先**: 设计决策应该有清晰的文档说明

### B. 参考资料

- [Anthropic Messages API](https://docs.anthropic.com/claude/reference/messages_post)
- [OpenAI Chat Completions API](https://platform.openai.com/docs/api-reference/chat)
- [OpenAI Responses API](https://platform.openai.com/docs/api-reference/responses)
- [Google Generative AI](https://ai.google.dev/api/python/google/generativeai)

### C. 变更日志

**v1.0 (2024-01-10)**
- 初始设计
- 定义核心类型
- 定义转换规则
- 定义扩展机制

---

**文档结束**