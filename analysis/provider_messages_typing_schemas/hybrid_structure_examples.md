# 混合方案：处理扩展性场景的代码示例

本文档展示混合方案（嵌套为主+扩展项）如何处理各种复杂和未来的扩展场景。

## 核心类型定义

```python
from typing import TypedDict, Union, List, Literal, Dict, Any, Optional
from typing_extensions import Required, NotRequired

# 核心：嵌套结构的消息
class Message(TypedDict):
    role: Required[Literal["system", "user", "assistant"]]
    content: Required[List[ContentPart]]
    metadata: NotRequired[MessageMetadata]

# 内容部分
ContentPart = Union[
    TextPart,
    ImagePart,
    FilePart,
    ToolCallPart,
    ToolResultPart,
    ReasoningPart,
]

# 扩展项（用于特殊场景）
ExtensionItem = Union[
    SystemEvent,
    BatchMarker,
    SessionControl,
    ToolChainNode,
]

# 混合输入：大部分是Message，少数是ExtensionItem
IRInput = List[Union[Message, ExtensionItem]]

# 简化版（只有Message）
IRInputSimple = List[Message]
```

## 场景 1：工具链（跨消息的复杂交互）

### 问题回顾

工具链中，一个工具的输出直接作为另一个工具的输入，不需要 assistant 介入。

### 混合方案解决

```python
# 定义工具链节点（扩展项）
class ToolChainNode(TypedDict):
    type: Required[Literal["tool_chain_node"]]
    node_id: Required[str]
    tool_call: Required[ToolCallPart]
    depends_on: NotRequired[List[str]]  # 依赖的节点ID
    auto_execute: NotRequired[bool]  # 是否自动执行

# 使用示例
conversation = [
    # 普通消息
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "搜索AI新闻并总结"}
        ]
    },

    # Assistant规划工具链
    {
        "role": "assistant",
        "content": [
            {"type": "text", "text": "我会先搜索，然后总结结果"}
        ]
    },

    # 工具链节点1：搜索（扩展项）
    {
        "type": "tool_chain_node",
        "node_id": "node_1",
        "tool_call": {
            "type": "tool_call",
            "tool_call_id": "call_1",
            "tool_name": "web_search",
            "tool_input": {"query": "latest AI news"},
            "tool_type": "web_search"
        },
        "auto_execute": True
    },

    # 工具链节点2：总结（扩展项，依赖节点1）
    {
        "type": "tool_chain_node",
        "node_id": "node_2",
        "tool_call": {
            "type": "tool_call",
            "tool_call_id": "call_2",
            "tool_name": "summarize",
            "tool_input": {
                "text": "{{node_1.result}}"  # 引用节点1的结果
            },
            "tool_type": "function"
        },
        "depends_on": ["node_1"],
        "auto_execute": True
    },

    # 工具链执行完成后，返回普通消息
    {
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_call_id": "call_2",
                "result": "总结：..."
            }
        ]
    },

    # Assistant最终回复
    {
        "role": "assistant",
        "content": [
            {"type": "text", "text": "根据搜索和总结..."}
        ]
    }
]
```

**优势**：

- ✅ 工具链的依赖关系清晰（`depends_on`）
- ✅ 不污染普通消息
- ✅ 可以表示复杂的 DAG 结构
- ✅ 向后兼容：不支持工具链的系统可以忽略这些扩展项

## 场景 2：系统级事件

### 问题回顾

需要支持系统级事件（会话暂停、恢复、超时等），但这些不属于任何角色。

### 混合方案解决

```python
# 定义系统事件（扩展项）
class SystemEvent(TypedDict):
    type: Required[Literal["system_event"]]
    event_type: Required[Literal[
        "session_start",
        "session_pause",
        "session_resume",
        "session_timeout",
        "session_end",
        "error",
        "warning"
    ]]
    timestamp: Required[str]
    event_data: NotRequired[Dict[str, Any]]
    message: NotRequired[str]

# 使用示例
conversation = [
    # 会话开始事件
    {
        "type": "system_event",
        "event_type": "session_start",
        "timestamp": "2024-01-01T10:00:00Z",
        "event_data": {"session_id": "sess_123"}
    },

    # 普通对话
    {
        "role": "user",
        "content": [{"type": "text", "text": "帮我计算..."}]
    },
    {
        "role": "assistant",
        "content": [
            {
                "type": "tool_call",
                "tool_call_id": "call_1",
                "tool_name": "calc",
                "tool_input": {"expression": "2+2"},
                "tool_type": "function"
            }
        ]
    },

    # 会话暂停事件
    {
        "type": "system_event",
        "event_type": "session_pause",
        "timestamp": "2024-01-01T10:05:00Z",
        "message": "User requested pause"
    },

    # ... 一段时间后 ...

    # 会话恢复事件
    {
        "type": "system_event",
        "event_type": "session_resume",
        "timestamp": "2024-01-01T10:15:00Z"
    },

    # 继续对话
    {
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_call_id": "call_1",
                "result": "4"
            }
        ]
    },

    # 超时警告
    {
        "type": "system_event",
        "event_type": "warning",
        "timestamp": "2024-01-01T10:25:00Z",
        "message": "Tool execution took longer than expected",
        "event_data": {"tool_call_id": "call_1", "duration_ms": 600000}
    },

    # 会话结束事件
    {
        "type": "system_event",
        "event_type": "session_end",
        "timestamp": "2024-01-01T10:30:00Z",
        "event_data": {"total_messages": 10, "total_tokens": 1500}
    }
]
```

**优势**：

- ✅ 系统事件不污染角色语义
- ✅ 清晰的时间线
- ✅ 可以插入任何位置
- ✅ 易于过滤（只看消息或只看事件）

## 场景 3：并行工具调用的部分结果

### 问题回顾

多个工具并行调用，结果以不同顺序返回，不想等所有结果都返回。

### 混合方案解决

```python
# 定义批次标记（扩展项）
class BatchMarker(TypedDict):
    type: Required[Literal["batch_marker"]]
    batch_id: Required[str]
    batch_type: Required[Literal["start", "end", "partial"]]
    total_items: NotRequired[int]
    completed_items: NotRequired[int]
    metadata: NotRequired[Dict[str, Any]]

# 使用示例
conversation = [
    {
        "role": "user",
        "content": [{"type": "text", "text": "查询3个城市的天气"}]
    },

    # Assistant发起3个并行工具调用
    {
        "role": "assistant",
        "content": [
            {"type": "text", "text": "我会同时查询这3个城市"},
            {
                "type": "tool_call",
                "tool_call_id": "call_1",
                "tool_name": "get_weather",
                "tool_input": {"city": "Beijing"},
                "tool_type": "function"
            },
            {
                "type": "tool_call",
                "tool_call_id": "call_2",
                "tool_name": "get_weather",
                "tool_input": {"city": "Shanghai"},
                "tool_type": "function"
            },
            {
                "type": "tool_call",
                "tool_call_id": "call_3",
                "tool_name": "get_weather",
                "tool_input": {"city": "Guangzhou"},
                "tool_type": "function"
            }
        ]
    },

    # 批次开始标记
    {
        "type": "batch_marker",
        "batch_id": "batch_weather",
        "batch_type": "start",
        "total_items": 3,
        "metadata": {"description": "Weather queries for 3 cities"}
    },

    # 第一个结果返回（call_3最快）
    {
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_call_id": "call_3",
                "result": "Guangzhou: Sunny, 28°C"
            }
        ]
    },

    # 部分完成标记
    {
        "type": "batch_marker",
        "batch_id": "batch_weather",
        "batch_type": "partial",
        "total_items": 3,
        "completed_items": 1
    },

    # 第二个结果返回（call_1）
    {
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_call_id": "call_1",
                "result": "Beijing: Cloudy, 15°C"
            }
        ]
    },

    # 部分完成标记
    {
        "type": "batch_marker",
        "batch_id": "batch_weather",
        "batch_type": "partial",
        "total_items": 3,
        "completed_items": 2
    },

    # 第三个结果返回（call_2）
    {
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_call_id": "call_2",
                "result": "Shanghai: Rainy, 20°C"
            }
        ]
    },

    # 批次结束标记
    {
        "type": "batch_marker",
        "batch_id": "batch_weather",
        "batch_type": "end",
        "total_items": 3,
        "completed_items": 3
    },

    # Assistant汇总回复
    {
        "role": "assistant",
        "content": [
            {"type": "text", "text": "三个城市的天气如下：..."}
        ]
    }
]
```

**优势**：

- ✅ 可以立即返回部分结果
- ✅ 清晰的批次边界
- ✅ 进度跟踪
- ✅ 不需要等所有结果

## 场景 4：流式工具调用

### 问题回顾

工具调用支持流式返回（如代码执行的实时输出）。

### 混合方案解决

```python
# 使用普通消息+metadata标记流式状态
conversation = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "运行这段Python代码"}
        ]
    },

    {
        "role": "assistant",
        "content": [
            {
                "type": "tool_call",
                "tool_call_id": "call_1",
                "tool_name": "code_interpreter",
                "tool_input": {"code": "for i in range(10): print(i)"},
                "tool_type": "code_interpreter"
            }
        ]
    },

    # 流式结果1
    {
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_call_id": "call_1",
                "result": "0\n1\n2\n"
            }
        ],
        "metadata": {
            "streaming": {
                "is_streaming": True,
                "is_final": False,
                "chunk_index": 0
            }
        }
    },

    # 流式结果2
    {
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_call_id": "call_1",
                "result": "3\n4\n5\n"
            }
        ],
        "metadata": {
            "streaming": {
                "is_streaming": True,
                "is_final": False,
                "chunk_index": 1
            }
        }
    },

    # 流式结果3（最终）
    {
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_call_id": "call_1",
                "result": "6\n7\n8\n9\n"
            }
        ],
        "metadata": {
            "streaming": {
                "is_streaming": True,
                "is_final": True,
                "chunk_index": 2
            }
        }
    },

    {
        "role": "assistant",
        "content": [
            {"type": "text", "text": "代码执行完成"}
        ]
    }
]
```

**优势**：

- ✅ 使用 metadata 而不是新的内容类型
- ✅ 保持消息结构一致
- ✅ 可以通过`is_final`判断是否完成
- ✅ 可以合并所有 chunk 得到完整结果

## 场景 5：工具调用的取消/修改

### 问题回顾

用户想取消或修改一个正在执行的工具调用。

### 混合方案解决

```python
# 定义会话控制（扩展项）
class SessionControl(TypedDict):
    type: Required[Literal["session_control"]]
    control_type: Required[Literal[
        "cancel_tool",
        "modify_tool",
        "pause_tool",
        "resume_tool"
    ]]
    target_id: Required[str]  # 目标tool_call_id
    reason: NotRequired[str]
    new_input: NotRequired[Dict[str, Any]]  # 用于modify_tool

# 使用示例
conversation = [
    {
        "role": "user",
        "content": [{"type": "text", "text": "搜索最新AI新闻"}]
    },

    {
        "role": "assistant",
        "content": [
            {
                "type": "tool_call",
                "tool_call_id": "call_1",
                "tool_name": "web_search",
                "tool_input": {"query": "latest AI news"},
                "tool_type": "web_search"
            }
        ]
    },

    # 用户取消工具调用
    {
        "type": "session_control",
        "control_type": "cancel_tool",
        "target_id": "call_1",
        "reason": "user_cancelled"
    },

    # 用户修改查询
    {
        "role": "user",
        "content": [{"type": "text", "text": "不，搜索AI安全相关的新闻"}]
    },

    {
        "role": "assistant",
        "content": [
            {
                "type": "tool_call",
                "tool_call_id": "call_2",
                "tool_name": "web_search",
                "tool_input": {"query": "AI safety news"},
                "tool_type": "web_search"
            }
        ]
    },

    # 或者直接修改工具调用
    {
        "type": "session_control",
        "control_type": "modify_tool",
        "target_id": "call_2",
        "new_input": {"query": "AI safety and ethics news"}
    },

    # 继续正常流程
    {
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_call_id": "call_2",
                "result": "..."
            }
        ]
    }
]
```

**优势**：

- ✅ 控制操作不污染用户消息
- ✅ 清晰的控制语义
- ✅ 支持多种控制类型
- ✅ 可以追溯操作历史

## 场景 6：复杂的多模态+工具混合

### 问题回顾

用户在一条消息中同时发送文本、图片，并且要求调用工具，同时返回工具结果和补充信息。

### 混合方案解决

```python
conversation = [
    # 用户的复杂输入
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "这是什么动物？"},
            {"type": "image", "image_url": "https://example.com/animal.jpg"},
            {"type": "text", "text": "另外帮我搜索一下它的习性"}
        ]
    },

    # Assistant识别并调用工具
    {
        "role": "assistant",
        "content": [
            {"type": "text", "text": "这是一只熊猫。让我搜索它的习性"},
            {
                "type": "tool_call",
                "tool_call_id": "call_1",
                "tool_name": "web_search",
                "tool_input": {"query": "panda habits"},
                "tool_type": "web_search"
            }
        ]
    },

    # 用户返回工具结果并补充信息（嵌套结构的优势）
    {
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_call_id": "call_1",
                "result": "熊猫主要吃竹子..."
            },
            {"type": "text", "text": "谢谢！另外它们濒危吗？"}
        ]
    },

    # Assistant继续回复
    {
        "role": "assistant",
        "content": [
            {"type": "text", "text": "是的，熊猫是濒危物种..."}
        ]
    }
]
```

**优势**：

- ✅ 工具结果和用户补充在同一条消息中（原子性）
- ✅ 语义清晰：这是用户的一次完整输入
- ✅ 不需要拆分成多个项

## 转换示例：混合方案 → 各 Provider

### 转换到 Anthropic

```python
def to_anthropic(ir_input: IRInput) -> tuple[List[MessageParam], List[str]]:
    """转换IR到Anthropic格式"""
    messages = []
    warnings = []

    for item in ir_input:
        if isinstance(item, dict) and item.get("role"):
            # 普通消息：直接转换
            messages.append({
                "role": item["role"],
                "content": convert_content_parts(item["content"])
            })
        elif isinstance(item, dict) and item.get("type") == "system_event":
            # 系统事件：记录警告，Anthropic不支持
            warnings.append(f"System event ignored: {item['event_type']}")
        elif isinstance(item, dict) and item.get("type") == "tool_chain_node":
            # 工具链：转换为普通工具调用
            warnings.append("Tool chain converted to sequential calls")
            # 可以展开为多个消息
        # ... 其他扩展项处理

    return messages, warnings
```

### 转换到 OpenAI Responses

```python
def to_openai_responses(ir_input: IRInput) -> List[ResponseInputItemParam]:
    """转换IR到OpenAI Responses格式"""
    items = []

    for item in ir_input:
        if isinstance(item, dict) and item.get("role"):
            # 普通消息
            items.append({
                "type": "message",
                "role": item["role"],
                "content": convert_content_parts(item["content"])
            })
        elif isinstance(item, dict) and item.get("type") == "system_event":
            # 系统事件：可以保留（OpenAI Responses支持）
            items.append({
                "type": "system_event",
                "event_type": item["event_type"],
                "timestamp": item["timestamp"]
            })
        elif isinstance(item, dict) and item.get("type") == "tool_chain_node":
            # 工具链：可以展开或保留
            items.append({
                "type": "function_call",
                "name": item["tool_call"]["tool_name"],
                "arguments": item["tool_call"]["tool_input"]
            })
        # ... 其他扩展项处理

    return items
```

## 总结：混合方案的优势

### 1. 渐进式复杂度

```python
# 简单场景：只用Message
simple_conversation: List[Message] = [
    {"role": "user", "content": [{"type": "text", "text": "Hello"}]},
    {"role": "assistant", "content": [{"type": "text", "text": "Hi"}]}
]

# 复杂场景：使用扩展项
complex_conversation: IRInput = [
    {"role": "user", "content": [...]},
    {"type": "system_event", "event_type": "pause", ...},
    {"type": "tool_chain_node", ...},
    {"role": "assistant", "content": [...]}
]
```

### 2. 向后兼容

```python
# 旧代码只处理Message
def process_messages(messages: List[Message]):
    for msg in messages:
        print(msg["role"], msg["content"])

# 新代码可以过滤扩展项
def process_ir_input(ir_input: IRInput):
    messages = [item for item in ir_input if "role" in item]
    process_messages(messages)
```

### 3. 类型安全

```python
def handle_item(item: Union[Message, ExtensionItem]):
    if "role" in item:
        # TypeScript/mypy知道这是Message
        print(item["role"])
    elif item.get("type") == "system_event":
        # TypeScript/mypy知道这是SystemEvent
        print(item["event_type"])
```

### 4. 易于扩展

```python
# 添加新的扩展项类型不影响现有代码
class NewExtensionType(TypedDict):
    type: Required[Literal["new_extension"]]
    # ... 新字段

ExtensionItem = Union[
    SystemEvent,
    BatchMarker,
    SessionControl,
    ToolChainNode,
    NewExtensionType,  # 新增
]
```

## 结论

混合方案提供了：

- ✅ **简单场景的简洁性**：90%的情况只用 Message
- ✅ **复杂场景的灵活性**：10%的情况用扩展项
- ✅ **良好的扩展性**：添加新功能不破坏现有结构
- ✅ **向后兼容性**：旧代码可以忽略扩展项
- ✅ **类型安全**：TypeScript/Python 类型检查支持
- ✅ **易于转换**：到各 provider 的转换逻辑清晰
