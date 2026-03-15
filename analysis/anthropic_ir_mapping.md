# Anthropic Messages API 与 IR Types 映射文档

本文档详细说明了 LLM-Rosetta 的 IR types 与 Anthropic Messages API 的类型映射关系。

## 目录

- [IR Request Types 映射](#ir-request-types-映射)
- [IR Response Types 映射](#ir-response-types-映射)

---

## IR Request Types 映射

### 1. 核心请求参数

#### IRRequest → MessageCreateParams

| IR Type | IR Field | Anthropic Type | Anthropic Field | 说明 |
|---------|----------|----------------|-----------------|------|
| `IRRequest` | `model` | `ModelParam` | `model` | 模型ID，如 "claude-3-5-sonnet-20241022" |
| `IRRequest` | `messages` | `Iterable[MessageParam]` | `messages` | 消息列表，需要转换 |
| `IRRequest` | `system_instruction` | `Union[str, Iterable[TextBlockParam]]` | `system` | 系统提示，独立字段 |

**Anthropic 类型定义：**
```python
# MessageCreateParams 基础类型
class MessageCreateParamsBase(TypedDict, total=False):
    max_tokens: Required[int]  # 必需参数
    messages: Required[Iterable[MessageParam]]  # 必需参数
    model: Required[ModelParam]  # 必需参数
    
    system: Union[str, Iterable[TextBlockParam]]  # 系统提示
    # ... 其他可选参数
```

**映射说明：**
- IR 的 `messages` 字段类型为 `IRInput`（即 `List[Union[Message, ExtensionItem]]`）
- Anthropic 的 `messages` 字段类型为 `Iterable[MessageParam]`
- `system_instruction` 在 Anthropic 中是独立的顶层参数 `system`
- Anthropic 的 `max_tokens` 是**必需参数**，而 IR 中是可选的

---

### 2. 工具相关参数

#### ToolDefinition → ToolParam

| IR Type | IR Field | Anthropic Type | Anthropic Field | 说明 |
|---------|----------|----------------|-----------------|------|
| `ToolDefinition` | `type` | `Optional[Literal["custom"]]` | `type` | IR 支持多种类型，Anthropic 主要是 "custom" |
| `ToolDefinition` | `name` | `str` | `name` | 工具名称 |
| `ToolDefinition` | `description` | `str` | `description` | 工具描述（可选但强烈推荐） |
| `ToolDefinition` | `parameters` | `InputSchema` | `input_schema` | JSON Schema 格式的参数定义 |

**Anthropic 类型定义：**
```python
# 工具参数类型
class ToolParam(TypedDict, total=False):
    input_schema: Required[InputSchema]  # JSON Schema
    name: Required[str]
    
    cache_control: Optional[CacheControlEphemeralParam]
    description: str  # 可选但强烈推荐
    type: Optional[Literal["custom"]]

# InputSchema 类型
InputSchema: TypeAlias = Union[InputSchemaTyped, Dict[str, object]]

class InputSchemaTyped(TypedDict, total=False):
    type: Required[Literal["object"]]
    properties: Optional[Dict[str, object]]
    required: Optional[SequenceNotStr[str]]
```

**映射说明：**
- IR 的 `ToolDefinition` 是扁平结构
- Anthropic 也使用扁平结构，但字段名不同：`parameters` → `input_schema`
- IR 的 `required_parameters` 需要合并到 `input_schema` 的 JSON Schema 中
- Anthropic 支持 `cache_control` 用于提示缓存

#### ToolChoice → ToolChoiceParam

| IR Type | IR Field | Anthropic Value | 说明 |
|---------|----------|-----------------|------|
| `ToolChoice` | `mode: "none"` | `ToolChoiceNoneParam` | 不使用工具 |
| `ToolChoice` | `mode: "auto"` | `ToolChoiceAutoParam` | 自动决定 |
| `ToolChoice` | `mode: "any"` | `ToolChoiceAnyParam` | 必须使用某个工具 |
| `ToolChoice` | `mode: "tool"` | `ToolChoiceToolParam` | 指定特定工具 |

**Anthropic 的 tool_choice 类型：**
```python
ToolChoiceParam: TypeAlias = Union[
    ToolChoiceAutoParam,  # {"type": "auto"}
    ToolChoiceAnyParam,   # {"type": "any"}
    ToolChoiceToolParam,  # {"type": "tool", "name": "..."}
    ToolChoiceNoneParam   # {"type": "none", "disable_parallel_tool_use": bool}
]
```

**映射说明：**
- IR 的 `mode` 值直接对应 Anthropic 的 `type` 字段
- IR 的 `mode: "tool"` 需要转换为 `{"type": "tool", "name": tool_name}`
- Anthropic 的命名更直观：使用 "any" 而不是 "required"

#### ToolCallConfig → disable_parallel_tool_use

| IR Type | IR Field | Anthropic Type | Anthropic Field | 说明 |
|---------|----------|----------------|-----------------|------|
| `ToolCallConfig` | `disable_parallel` | `bool` (in `ToolChoiceNoneParam`) | `disable_parallel_tool_use` | 禁用并行工具调用 |
| `ToolCallConfig` | `max_calls` | - | - | Anthropic 不支持此参数 |

**映射说明：**
- `disable_parallel_tool_use` 在 Anthropic 中是 `ToolChoiceNoneParam` 的一部分
- IR 的 `disable_parallel: true` 直接映射为 Anthropic 的 `disable_parallel_tool_use: true`

---

### 3. 生成控制参数

#### GenerationConfig → 各生成参数

| IR Type | IR Field | Anthropic Type | Anthropic Field | 说明 |
|---------|----------|----------------|-----------------|------|
| `GenerationConfig` | `temperature` | `float` | `temperature` | 0.0-1.0（Anthropic 范围更小） |
| `GenerationConfig` | `top_p` | `float` | `top_p` | 0.0-1.0 |
| `GenerationConfig` | `top_k` | `int` | `top_k` | Anthropic 支持 |
| `GenerationConfig` | `max_tokens` | `int` | `max_tokens` | **必需参数** |
| `GenerationConfig` | `stop_sequences` | `SequenceNotStr[str]` | `stop_sequences` | 停止序列列表 |
| `GenerationConfig` | `frequency_penalty` | - | - | Anthropic 不支持 |
| `GenerationConfig` | `presence_penalty` | - | - | Anthropic 不支持 |
| `GenerationConfig` | `logit_bias` | - | - | Anthropic 不支持 |
| `GenerationConfig` | `seed` | - | - | Anthropic 不支持 |
| `GenerationConfig` | `logprobs` | - | - | Anthropic 不支持 |
| `GenerationConfig` | `n` | - | - | Anthropic 不支持 |

**映射说明：**
- Anthropic 的 `temperature` 范围是 0.0-1.0，而 OpenAI 是 0.0-2.0
- `max_tokens` 在 Anthropic 中是**必需参数**
- Anthropic 支持 `top_k` 采样
- Anthropic 不支持 penalty、logit_bias、seed 等参数

---

### 4. 推理配置

#### ReasoningConfig → thinking

| IR Type | IR Field | Anthropic Type | Anthropic Field | 说明 |
|---------|----------|----------------|-----------------|------|
| `ReasoningConfig` | `type` | `Literal["enabled", "disabled"]` | `thinking.type` | 推理类型 |
| `ReasoningConfig` | `budget_tokens` | `int` | `thinking.budget_tokens` | 推理预算 token 数 |
| `ReasoningConfig` | `effort` | - | - | Anthropic 不支持此字段 |

**Anthropic 类型定义：**
```python
# ThinkingConfig 联合类型
ThinkingConfigParam: TypeAlias = Union[
    ThinkingConfigEnabledParam,   # {"type": "enabled", "budget_tokens": int}
    ThinkingConfigDisabledParam   # {"type": "disabled"}
]
```

**映射说明：**
- Anthropic 使用 `thinking` 配置而不是 `reasoning_effort`
- `budget_tokens` 最小值为 1024
- IR 的 `effort` 字段在 Anthropic 中不支持

---

### 5. 流式输出配置

#### StreamConfig → stream

| IR Type | IR Field | Anthropic Type | Anthropic Field | 说明 |
|---------|----------|----------------|-----------------|------|
| `StreamConfig` | `enabled` | `Union[Literal[True], Literal[False]]` | `stream` | 是否启用流式输出 |
| `StreamConfig` | `include_usage` | - | - | Anthropic 总是在流式输出中包含 usage |

**Anthropic 类型定义：**
```python
# 非流式
class MessageCreateParamsNonStreaming(MessageCreateParamsBase, total=False):
    stream: Literal[False]

# 流式
class MessageCreateParamsStreaming(MessageCreateParamsBase):
    stream: Required[Literal[True]]
```

**映射说明：**
- Anthropic 的流式输出通过 `stream` 参数控制
- Anthropic 在流式输出中总是包含 usage 信息，不需要额外配置

---

### 6. 响应格式配置

#### ResponseFormatConfig → (不直接支持)

Anthropic 不直接支持 `response_format` 参数，但可以通过以下方式实现：
- 在 `system` 提示中指定输出格式
- 使用工具调用来获取结构化输出

---

### 7. 缓存配置

#### CacheConfig → cache_control

| IR Type | IR Field | Anthropic Type | Anthropic Field | 说明 |
|---------|----------|----------------|-----------------|------|
| `CacheConfig` | - | `CacheControlEphemeralParam` | `cache_control` | 缓存控制 |

**Anthropic 类型定义：**
```python
# 缓存控制参数（在内容块级别）
class CacheControlEphemeralParam(TypedDict, total=False):
    type: Required[Literal["ephemeral"]]
```

**映射说明：**
- Anthropic 的缓存控制是在**内容块级别**设置的，而不是请求级别
- 通过在消息、工具定义等位置添加 `cache_control` 字段来启用缓存
- IR 的缓存配置需要转换为内容块级别的 `cache_control`

---

### 8. Provider 扩展参数

以下是 Anthropic 特有的参数，存储在 IR 的 `provider_extensions` 中：

| Anthropic Field | 类型 | 说明 |
|-----------------|------|------|
| `metadata` | `MetadataParam` | 请求元数据 |
| `service_tier` | `Literal["auto", "standard_only"]` | 服务等级 |

---

### 9. 消息参数类型详细映射

IR 的 `Message` 需要根据 `role` 转换为 Anthropic 的 `MessageParam`：

#### Message → MessageParam

| IR Type | IR Field | Anthropic Type | Anthropic Field | 说明 |
|---------|----------|----------------|-----------------|------|
| `Message` | `role` | `Literal["user", "assistant"]` | `role` | Anthropic 不支持 "system" role |
| `Message` | `content` | `Union[str, Iterable[ContentBlockParam]]` | `content` | 消息内容 |

**Anthropic 类型定义：**
```python
class MessageParam(TypedDict, total=False):
    content: Required[Union[str, Iterable[ContentBlockParam]]]
    role: Required[Literal["user", "assistant"]]

# ContentBlockParam 包括：
ContentBlockParam = Union[
    TextBlockParam,
    ImageBlockParam,
    DocumentBlockParam,
    SearchResultBlockParam,
    ThinkingBlockParam,
    RedactedThinkingBlockParam,
    ToolUseBlockParam,
    ToolResultBlockParam,
    ServerToolUseBlockParam,
    WebSearchToolResultBlockParam,
    ContentBlock,  # 响应中的内容块也可以作为输入
]
```

**映射说明：**
- Anthropic 的消息只支持 `"user"` 和 `"assistant"` 角色
- `"system"` 角色需要使用顶层的 `system` 参数
- IR 的 `content: List[ContentPart]` 需要转换为 Anthropic 的内容块格式

---

## IR Response Types 映射

### 1. 顶层响应结构

#### IRResponse → Message

| IR Type | IR Field | Anthropic Type | Anthropic Field | 说明 |
|---------|----------|----------------|-----------------|------|
| `IRResponse` | `id` | `str` | `id` | 响应唯一ID |
| `IRResponse` | `object` | `Literal["message"]` | `type` | IR: "response", Anthropic: "message" |
| `IRResponse` | `created` | - | - | Anthropic 不提供时间戳 |
| `IRResponse` | `model` | `Model` | `model` | 使用的模型 |
| `IRResponse` | `choices` | - | - | Anthropic 不使用 choices 结构 |
| `IRResponse` | `usage` | `Usage` | `usage` | Token 使用统计 |

**Anthropic 类型定义：**
```python
class Message(BaseModel):
    id: str
    content: List[ContentBlock]
    model: Model
    role: Literal["assistant"]
    stop_reason: Optional[StopReason] = None
    stop_sequence: Optional[str] = None
    type: Literal["message"]
    usage: Usage
```

**映射说明：**
- Anthropic 的响应是单个 `Message` 对象，而不是 `choices` 列表
- IR 的 `choices[0].message` 对应 Anthropic 的整个 `Message`
- Anthropic 不提供 `created` 时间戳
- Anthropic 的 `role` 总是 `"assistant"`

---

### 2. 停止原因

#### FinishReason → stop_reason

| IR Value | Anthropic Value | 说明 |
|----------|-----------------|------|
| `"stop"` | `"end_turn"` | 正常停止 |
| `"length"` | `"max_tokens"` | 达到最大长度 |
| `"tool_calls"` | `"tool_use"` | 工具调用 |
| `"content_filter"` | - | Anthropic 没有此值 |
| `"refusal"` | `"refusal"` | 拒绝回答 |
| - | `"stop_sequence"` | 遇到停止序列（Anthropic 特有） |
| - | `"pause_turn"` | 暂停长时间运行（Anthropic 特有） |

**Anthropic 类型定义：**
```python
StopReason: TypeAlias = Literal[
    "end_turn",      # 正常停止
    "max_tokens",    # 达到最大长度
    "stop_sequence", # 遇到停止序列
    "tool_use",      # 工具调用
    "pause_turn",    # 暂停长时间运行
    "refusal"        # 拒绝回答
]
```

---

### 3. 内容部分映射

#### TextPart → TextBlock

| IR Type | IR Field | Anthropic Type | Anthropic Field | 说明 |
|---------|----------|----------------|-----------------|------|
| `TextPart` | `text` | `str` | `text` | 文本内容 |
| - | - | `Optional[List[TextCitation]]` | `citations` | Anthropic 支持引用 |

**Anthropic 类型定义：**
```python
class TextBlock(BaseModel):
    citations: Optional[List[TextCitation]] = None
    text: str
    type: Literal["text"]
```

#### ToolCallPart → ToolUseBlock

| IR Type | IR Field | Anthropic Type | Anthropic Field | 说明 |
|---------|----------|----------------|-----------------|------|
| `ToolCallPart` | `tool_call_id` | `str` | `id` | 工具调用ID |
| `ToolCallPart` | `tool_name` | `str` | `name` | 工具名称 |
| `ToolCallPart` | `tool_input` | `Dict[str, object]` | `input` | IR: Dict, Anthropic: Dict（不是字符串） |

**Anthropic 类型定义：**
```python
class ToolUseBlock(BaseModel):
    id: str
    input: Dict[str, object]  # 直接是字典，不是JSON字符串
    name: str
    type: Literal["tool_use"]
```

**映射说明：**
- Anthropic 的 `input` 是 `Dict[str, object]`，而 OpenAI 是 JSON 字符串
- 不需要序列化/反序列化

#### ReasoningPart → ThinkingBlock

| IR Type | IR Field | Anthropic Type | Anthropic Field | 说明 |
|---------|----------|----------------|-----------------|------|
| `ReasoningPart` | `reasoning` | `str` | `thinking` | 推理内容 |
| - | - | `str` | `signature` | Anthropic 特有的签名字段 |

**Anthropic 类型定义：**
```python
class ThinkingBlock(BaseModel):
    signature: str  # 推理签名
    thinking: str   # 推理内容
    type: Literal["thinking"]
```

**映射说明：**
- Anthropic 使用 `thinking` 而不是 `reasoning`
- Anthropic 提供 `signature` 字段用于验证推理内容

#### CitationPart → TextBlock.citations

| IR Type | IR Field | Anthropic Type | Anthropic Field | 说明 |
|---------|----------|----------------|-----------------|------|
| `CitationPart` | `text_citation` | `TextCitation` | `citations` | 文本引用 |

**Anthropic 类型定义：**
```python
class TextCitation(BaseModel):
    cited_text: str
    type: Literal["text_citation"]
```

**映射说明：**
- Anthropic 的引用是 `TextBlock` 的一部分，而不是独立的内容块
- 支持多种引用类型：`char_location`, `page_location`, `content_block_location`

---

### 4. Token 使用统计

#### UsageInfo → Usage

| IR Type | IR Field | Anthropic Type | Anthropic Field | 说明 |
|---------|----------|----------------|-----------------|------|
| `UsageInfo` | `prompt_tokens` | `int` | `input_tokens` | 输入 token 数 |
| `UsageInfo` | `completion_tokens` | `int` | `output_tokens` | 输出 token 数 |
| `UsageInfo` | `total_tokens` | - | - | Anthropic 不提供总数（需要计算） |
| `UsageInfo` | `cache_read_tokens` | `Optional[int]` | `cache_read_input_tokens` | 缓存读取 token 数 |
| - | `Optional[int]` | `cache_creation_input_tokens` | 缓存创建 token 数（Anthropic 特有） |

**Anthropic 类型定义：**
```python
class Usage(BaseModel):
    cache_creation: Optional[CacheCreation] = None
    cache_creation_input_tokens: Optional[int] = None
    cache_read_input_tokens: Optional[int] = None
    input_tokens: int
    output_tokens: int
    server_tool_use: Optional[ServerToolUsage] = None
    service_tier: Optional[Literal["standard", "priority", "batch"]] = None
```

**映射说明：**
- Anthropic 使用 `input_tokens` 和 `output_tokens` 而不是 `prompt_tokens` 和 `completion_tokens`
- Anthropic 不提供 `total_tokens`，需要计算：`input_tokens + output_tokens`
- Anthropic 提供详细的缓存统计：`cache_creation_input_tokens` 和 `cache_read_input_tokens`

---

## 关键差异总结

### 1. 结构差异

| 方面 | IR | Anthropic |
|------|----|----|
| 系统指令 | 独立字段 `system_instruction` | 顶层参数 `system` |
| 工具定义 | 扁平结构 | 扁平结构，但字段名不同 |
| 工具调用参数 | `Dict[str, Any]` | `Dict[str, object]`（不是字符串） |
| 响应结构 | `choices` 列表 | 单个 `Message` 对象 |
| Token 统计 | `prompt_tokens`, `completion_tokens` | `input_tokens`, `output_tokens` |

### 2. 功能差异

| 功能 | IR | Anthropic |
|------|----|----|
| 必需参数 | `model`, `messages` | `model`, `messages`, `max_tokens` |
| 温度范围 | 0.0-2.0 | 0.0-1.0 |
| top_k 采样 | 支持 | 支持 |
| 推理配置 | `reasoning_effort` | `thinking` (type + budget_tokens) |
| 缓存控制 | 请求级别 | 内容块级别 |
| 引用支持 | 独立内容块 | TextBlock 的一部分 |

### 3. 命名差异

| IR | Anthropic | 说明 |
|----|-----------|------|
| `mode: "any"` | `type: "any"` | 必须使用工具 |
| `stop_sequences` | `stop_sequences` | 相同 |
| `object: "response"` | `type: "message"` | 对象类型 |
| `finish_reason: "stop"` | `stop_reason: "end_turn"` | 停止原因 |
| `finish_reason: "length"` | `stop_reason: "max_tokens"` | 达到最大长度 |
| `finish_reason: "tool_calls"` | `stop_reason: "tool_use"` | 工具调用 |

---

## 转换注意事项

### Request 转换

1. **必需参数处理**：确保 `max_tokens` 有值（Anthropic 必需）
2. **系统指令处理**：将 `system_instruction` 转换为顶层 `system` 参数
3. **温度范围**：确保 `temperature` 在 0.0-1.0 范围内
4. **工具定义**：`parameters` → `input_schema`
5. **缓存控制**：从请求级别转换为内容块级别

### Response 转换

1. **响应结构**：将单个 `Message` 转换为 `choices` 列表
2. **停止原因映射**：`end_turn` → `stop`, `max_tokens` → `length`, `tool_use` → `tool_calls`
3. **Token 统计**：`input_tokens` → `prompt_tokens`, `output_tokens` → `completion_tokens`
4. **计算总数**：`total_tokens = input_tokens + output_tokens`
5. **推理内容**：`thinking` → `reasoning`
6. **引用处理**：从 `TextBlock.citations` 提取为独立的 `CitationPart`

---

## 示例代码片段

### Request 转换示例

```python
# IR → Anthropic
ir_request: IRRequest = {...}

anthropic_params = {
    "model": ir_request["model"],
    "max_tokens": ir_request.get("generation", {}).get("max_tokens", 4096),  # 必需
    "messages": convert_ir_messages_to_anthropic(ir_request["messages"]),
    "system": ir_request.get("system_instruction"),  # 独立参数
    "temperature": min(ir_request.get("generation", {}).get("temperature", 1.0), 1.0),
    "top_p": ir_request.get("generation", {}).get("top_p"),
    "top_k": ir_request.get("generation", {}).get("top_k"),
    "stop_sequences": ir_request.get("generation", {}).get("stop_sequences"),
    "tools": [
        {
            "name": tool["name"],
            "description": tool.get("description"),
            "input_schema": tool.get("parameters")
        }
        for tool in ir_request.get("tools", [])
    ],
    "tool_choice": convert_tool_choice(ir_request.get("tool_choice"))
}
```

### Response 转换示例

```python
# Anthropic → IR
anthropic_response: Message = {...}

ir_response: IRResponse = {
    "id": anthropic_response.id,
    "object": "response",
    "created": int(time.time()),  # Anthropic 不提供，需要自己生成
    "model": anthropic_response.model,
    "choices": [
        {
            "index": 0,
            "message": convert_anthropic_message_to_ir(anthropic_response),
            "finish_reason": {
                "reason": convert_stop_reason(anthropic_response.stop_reason)
            },
        }
    ],
    "usage": {
        "prompt_tokens": anthropic_response.usage.input_tokens,
        "completion_tokens": anthropic_response.usage.output_tokens,
        "total_tokens": anthropic_response.usage.input_tokens + anthropic_response.usage.output_tokens,
        "cache_read_tokens": anthropic_response.usage.cache_read_input_tokens,
    }
}
```

---

## 参考资料

- Anthropic SDK 源码：`/data/pding/miniforge3/envs/llm-rosetta/lib/python3.10/site-packages/anthropic/types/`
- IR Request Types：`src/llm-rosetta/types/ir_request.py`
- IR Response Types：`src/llm-rosetta/types/ir_response.py`