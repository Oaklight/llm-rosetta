# OpenAI Chat Completion API 与 IR Types 映射文档

本文档详细说明了 LLMIR 的 IR types 与 OpenAI Chat Completion API 的类型映射关系。

## 目录

- [IR Request Types 映射](#ir-request-types-映射)
- [IR Response Types 映射](#ir-response-types-映射)

---

## IR Request Types 映射

### 1. 核心请求参数

#### IRRequest → CompletionCreateParams

| IR Type | IR Field | OpenAI Type | OpenAI Field | 说明 |
|---------|----------|-------------|--------------|------|
| `IRRequest` | `model` | `Union[str, ChatModel]` | `model` | 模型ID，ChatModel 是预定义的模型名称字面量 |
| `IRRequest` | `messages` | `Iterable[ChatCompletionMessageParam]` | `messages` | 消息列表，需要转换 |
| `IRRequest` | `system_instruction` | `ChatCompletionSystemMessageParam` | `messages[0]` | 转换为 role="system" 的消息插入数组开头 |

**OpenAI 类型定义：**
```python
# ChatModel 是预定义的模型名称
ChatModel = Literal["gpt-4o", "gpt-4-turbo", "o1", "o3", ...]

# ChatCompletionMessageParam 是消息类型的联合
ChatCompletionMessageParam = Union[
    ChatCompletionDeveloperMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
    ChatCompletionAssistantMessageParam,
    ChatCompletionToolMessageParam,
    ChatCompletionFunctionMessageParam,
]
```

**映射说明：**
- IR 的 `messages` 字段类型为 `IRInput`（即 `List[Union[Message, ExtensionItem]]`）
- OpenAI 的 `messages` 字段类型为 `Iterable[ChatCompletionMessageParam]`
- `system_instruction` 在 OpenAI 中通过在 messages 数组开头插入 `ChatCompletionSystemMessageParam` 实现

---

### 2. 工具相关参数

#### ToolDefinition → ChatCompletionFunctionToolParam

| IR Type | IR Field | OpenAI Type | OpenAI Field | 说明 |
|---------|----------|-------------|--------------|------|
| `ToolDefinition` | `type` | `Literal["function"]` | `type` | IR 支持多种类型，OpenAI 仅支持 "function" |
| `ToolDefinition` | `name` | `str` (in `FunctionDefinition`) | `function.name` | 函数名称 |
| `ToolDefinition` | `description` | `str` (in `FunctionDefinition`) | `function.description` | 函数描述 |
| `ToolDefinition` | `parameters` | `FunctionParameters` (in `FunctionDefinition`) | `function.parameters` | JSON Schema 格式的参数定义 |

**OpenAI 类型定义：**
```python
# 工具参数类型
ChatCompletionFunctionToolParam = TypedDict({
    "type": Required[Literal["function"]],
    "function": Required[FunctionDefinition]
}, total=False)

# 函数定义类型
FunctionDefinition = TypedDict({
    "name": Required[str],
    "description": str,  # optional
    "parameters": FunctionParameters,  # optional, JSON Schema
    "strict": Optional[bool]  # optional, 严格模式
}, total=False)
```

**映射说明：**
- IR 的 `ToolDefinition` 是扁平结构
- OpenAI 使用嵌套结构：`{"type": "function", "function": {...}}`
- IR 的 `required_parameters` 需要合并到 `parameters` 的 JSON Schema 中
- OpenAI 的 `strict` 字段可以存储在 IR 的 `metadata` 中

#### ToolChoice → ChatCompletionToolChoiceOptionParam

| IR Type | IR Field | OpenAI Value | 说明 |
|---------|----------|--------------|------|
| `ToolChoice` | `mode: "none"` | `"none"` | 不使用工具 |
| `ToolChoice` | `mode: "auto"` | `"auto"` | 自动决定 |
| `ToolChoice` | `mode: "any"` | `"required"` | 必须使用工具（OpenAI 用 "required"） |
| `ToolChoice` | `mode: "tool"` | `ChatCompletionNamedToolChoiceParam` | 指定特定工具 |

**OpenAI 的 tool_choice 类型：**
```python
ChatCompletionToolChoiceOptionParam = Union[
    Literal["none", "auto", "required"],
    ChatCompletionNamedToolChoiceParam  # {"type": "function", "function": {"name": "..."}}
]
```

**映射说明：**
- IR 的 `mode: "any"` 映射到 OpenAI 的 `"required"`
- IR 的 `mode: "tool"` 需要转换为 `{"type": "function", "function": {"name": tool_name}}`

#### ToolCallConfig → parallel_tool_calls

| IR Type | IR Field | OpenAI Type | OpenAI Field | 说明 |
|---------|----------|-------------|--------------|------|
| `ToolCallConfig` | `disable_parallel` | `bool` | `parallel_tool_calls` | 需要取反：IR 的 true → OpenAI 的 false |
| `ToolCallConfig` | `max_calls` | - | - | OpenAI Chat 不支持此参数 |

**映射说明：**
- `parallel_tool_calls` 是顶层参数，类型为 `bool`
- IR 的 `disable_parallel: true` 映射为 OpenAI 的 `parallel_tool_calls: false`

---

### 3. 生成控制参数

#### GenerationConfig → 各生成参数

| IR Type | IR Field | OpenAI Type | OpenAI Field | 说明 |
|---------|----------|-------------|--------------|------|
| `GenerationConfig` | `temperature` | `Optional[float]` | `temperature` | 0.0-2.0 |
| `GenerationConfig` | `top_p` | `Optional[float]` | `top_p` | 0.0-1.0 |
| `GenerationConfig` | `top_k` | - | - | OpenAI 不支持 |
| `GenerationConfig` | `max_tokens` | `Optional[int]` | `max_completion_tokens` | 推荐使用 max_completion_tokens |
| `GenerationConfig` | `stop_sequences` | `Union[Optional[str], SequenceNotStr[str], None]` | `stop` | IR 为 List[str]，OpenAI 为 str 或 List[str] |
| `GenerationConfig` | `frequency_penalty` | `Optional[float]` | `frequency_penalty` | -2.0 到 2.0 |
| `GenerationConfig` | `presence_penalty` | `Optional[float]` | `presence_penalty` | -2.0 到 2.0 |
| `GenerationConfig` | `logit_bias` | `Optional[Dict[str, int]]` | `logit_bias` | token ID 到 bias 值的映射 |
| `GenerationConfig` | `seed` | `Optional[int]` | `seed` | 随机种子 |
| `GenerationConfig` | `logprobs` | `Optional[bool]` | `logprobs` | 是否返回 log 概率 |
| `GenerationConfig` | `top_logprobs` | `Optional[int]` | `top_logprobs` | 0-20 |
| `GenerationConfig` | `n` | `Optional[int]` | `n` | 生成选择数量 |

**映射说明：**
- `max_tokens` 在 OpenAI 中有两个字段：`max_tokens`（已废弃）和 `max_completion_tokens`（推荐）
- `stop_sequences` 需要转换：IR 总是 `List[str]`，OpenAI 可以是单个字符串或列表
- 所有参数都是可选的（`Optional`）

---

### 4. 响应格式配置

#### ResponseFormatConfig → CompletionCreateParams.response_format

| IR Type | IR Field | OpenAI Type | OpenAI Value | 说明 |
|---------|----------|-------------|--------------|------|
| `ResponseFormatConfig` | `type: "text"` | `ResponseFormatText` | `{"type": "text"}` | 纯文本 |
| `ResponseFormatConfig` | `type: "json_object"` | `ResponseFormatJSONObject` | `{"type": "json_object"}` | JSON 对象 |
| `ResponseFormatConfig` | `type: "json_schema"` | `ResponseFormatJSONSchema` | `{"type": "json_schema", "json_schema": {...}}` | 结构化输出 |

**OpenAI 结构：**
```python
ResponseFormat = Union[
    ResponseFormatText,           # {"type": "text"}
    ResponseFormatJSONObject,     # {"type": "json_object"}
    ResponseFormatJSONSchema      # {"type": "json_schema", "json_schema": {...}}
]
```

---

### 5. 推理配置

#### ReasoningConfig → reasoning_effort

| IR Type | IR Field | OpenAI Type | OpenAI Field | 说明 |
|---------|----------|-------------|--------------|------|
| `ReasoningConfig` | `effort` | `Optional[ReasoningEffort]` | `reasoning_effort` | 推理努力级别 |
| `ReasoningConfig` | `type` | - | - | OpenAI 不支持此字段 |
| `ReasoningConfig` | `budget_tokens` | - | - | OpenAI 不支持此字段 |

**OpenAI 类型定义：**
```python
ReasoningEffort = Literal["none", "minimal", "low", "medium", "high", "xhigh"]
```

**映射说明：**
- IR 的 `effort: Literal["low", "medium", "high"]` 可以直接映射到 OpenAI 的 `reasoning_effort`
- OpenAI 支持更多级别：`"none"`, `"minimal"`, `"xhigh"`
- IR 的 `type` 和 `budget_tokens` 字段在 OpenAI 中不支持

---

### 6. 流式输出配置

#### StreamConfig → stream + stream_options

| IR Type | IR Field | OpenAI Type | OpenAI Field | 说明 |
|---------|----------|-------------|--------------|------|
| `StreamConfig` | `enabled` | `Union[Literal[True], Literal[False], None]` | `stream` | 是否启用流式输出 |
| `StreamConfig` | `include_usage` | `bool` (in `ChatCompletionStreamOptionsParam`) | `stream_options.include_usage` | 是否包含使用统计 |

**OpenAI 类型定义：**
```python
# stream_options 类型
ChatCompletionStreamOptionsParam = TypedDict({
    "include_usage": bool
}, total=False)
```

**映射说明：**
- `stream` 参数决定是否使用流式输出
- `stream_options` 仅在 `stream=True` 时有效

---

### 7. 缓存配置

#### CacheConfig → prompt_cache_*

| IR Type | IR Field | OpenAI Type | OpenAI Field | 说明 |
|---------|----------|-------------|--------------|------|
| `CacheConfig` | `key` | `str` | `prompt_cache_key` | 缓存键 |
| `CacheConfig` | `retention` | `Optional[Literal["in-memory", "24h"]]` | `prompt_cache_retention` | 缓存保留策略 |

**映射说明：**
- `prompt_cache_key` 用于优化缓存命中率
- `prompt_cache_retention` 设置为 `"24h"` 启用扩展提示缓存

---

### 8. Provider 扩展参数

以下是 OpenAI 特有的参数，存储在 IR 的 `provider_extensions` 中：

| OpenAI Field | 类型 | 说明 |
|--------------|------|------|
| `audio` | `ChatCompletionAudioParam` | 音频输出参数 |
| `metadata` | `Metadata` | 元数据（16个键值对） |
| `modalities` | `List[Literal["text", "audio"]]` | 输出模态 |
| `prediction` | `ChatCompletionPredictionContentParam` | 预测内容 |
| `safety_identifier` | `str` | 安全标识符 |
| `service_tier` | `Literal["auto", "default", "flex", "scale", "priority"]` | 服务等级 |
| `store` | `bool` | 是否存储用于蒸馏/评估 |
| `user` | `str` | 用户标识（已废弃，被 safety_identifier 替代） |
| `verbosity` | `Literal["low", "medium", "high"]` | 响应详细程度 |
| `web_search_options` | `WebSearchOptions` | 网络搜索选项 |

---

### 9. 消息参数类型详细映射

IR 的 `Message` 需要根据 `role` 转换为不同的 OpenAI 消息参数类型：

#### Message (role="system") → ChatCompletionSystemMessageParam

| IR Type | IR Field | OpenAI Type | OpenAI Field | 说明 |
|---------|----------|-------------|--------------|------|
| `Message` | `role: "system"` | `Literal["system"]` | `role` | 系统角色 |
| `Message` | `content` (TextPart) | `Union[str, Iterable[ChatCompletionContentPartTextParam]]` | `content` | 系统消息内容 |

**OpenAI 类型定义：**
```python
ChatCompletionSystemMessageParam = TypedDict({
    "role": Required[Literal["system"]],
    "content": Required[Union[str, Iterable[ChatCompletionContentPartTextParam]]],
    "name": str  # optional
}, total=False)
```

#### Message (role="user") → ChatCompletionUserMessageParam

| IR Type | IR Field | OpenAI Type | OpenAI Field | 说明 |
|---------|----------|-------------|--------------|------|
| `Message` | `role: "user"` | `Literal["user"]` | `role` | 用户角色 |
| `Message` | `content` | `Union[str, Iterable[ChatCompletionContentPartParam]]` | `content` | 用户消息内容，支持多模态 |

**OpenAI 类型定义：**
```python
ChatCompletionUserMessageParam = TypedDict({
    "role": Required[Literal["user"]],
    "content": Required[Union[str, Iterable[ChatCompletionContentPartParam]]],
    "name": str  # optional
}, total=False)

# 内容部分类型
ChatCompletionContentPartParam = Union[
    ChatCompletionContentPartTextParam,
    ChatCompletionContentPartImageParam,
    ChatCompletionContentPartInputAudioParam,
    File
]
```

#### Message (role="assistant") → ChatCompletionAssistantMessageParam

| IR Type | IR Field | OpenAI Type | OpenAI Field | 说明 |
|---------|----------|-------------|--------------|------|
| `Message` | `role: "assistant"` | `Literal["assistant"]` | `role` | 助手角色 |
| `Message` | `content` (TextPart) | `Union[str, Iterable[ContentArrayOfContentPart], None]` | `content` | 助手消息内容 |
| `Message` | `content` (ToolCallPart) | `Iterable[ChatCompletionMessageToolCallUnionParam]` | `tool_calls` | 工具调用 |
| `Message` | `content` (RefusalPart) | `Optional[str]` | `refusal` | 拒绝消息 |
| `Message` | `content` (AudioPart) | `Optional[Audio]` | `audio` | 音频数据 |

**OpenAI 类型定义：**
```python
ChatCompletionAssistantMessageParam = TypedDict({
    "role": Required[Literal["assistant"]],
    "content": Union[str, Iterable[ContentArrayOfContentPart], None],
    "audio": Optional[Audio],
    "function_call": Optional[FunctionCall],  # deprecated
    "name": str,
    "refusal": Optional[str],
    "tool_calls": Iterable[ChatCompletionMessageToolCallUnionParam]
}, total=False)
```

#### ToolResultPart → ChatCompletionToolMessageParam

| IR Type | IR Field | OpenAI Type | OpenAI Field | 说明 |
|---------|----------|-------------|--------------|------|
| `ToolResultPart` | `tool_call_id` | `str` | `tool_call_id` | 工具调用ID |
| `ToolResultPart` | `result` | `Union[str, Iterable[ChatCompletionContentPartTextParam]]` | `content` | 工具执行结果 |

**OpenAI 类型定义：**
```python
ChatCompletionToolMessageParam = TypedDict({
    "role": Required[Literal["tool"]],
    "content": Required[Union[str, Iterable[ChatCompletionContentPartTextParam]]],
    "tool_call_id": Required[str]
}, total=False)
```

**映射说明：**
- IR 的 `Message` 根据 `role` 字段映射到不同的 OpenAI 消息参数类型
- IR 的 `content: List[ContentPart]` 需要根据内容类型分发到不同的 OpenAI 字段
- `ToolResultPart` 需要创建独立的 `ChatCompletionToolMessageParam` 消息

---

## IR Response Types 映射

### 1. 顶层响应结构

#### IRResponse → ChatCompletion

| IR Type | IR Field | OpenAI Type | OpenAI Field | 说明 |
|---------|----------|-------------|--------------|------|
| `IRResponse` | `id` | `ChatCompletion` | `id` | 响应唯一ID |
| `IRResponse` | `object` | `ChatCompletion` | `object` | IR: "response", OpenAI: "chat.completion" |
| `IRResponse` | `created` | `ChatCompletion` | `created` | Unix 时间戳 |
| `IRResponse` | `model` | `ChatCompletion` | `model` | 使用的模型 |
| `IRResponse` | `choices` | `ChatCompletion` | `choices` | 选择结果列表 |
| `IRResponse` | `usage` | `ChatCompletion` | `usage` | Token 使用统计 |
| `IRResponse` | `service_tier` | `ChatCompletion` | `service_tier` | 服务等级 |
| `IRResponse` | `system_fingerprint` | `ChatCompletion` | `system_fingerprint` | 系统指纹 |

---

### 2. 选择结果

#### ChoiceInfo → Choice

| IR Type | IR Field | OpenAI Type | OpenAI Field | 说明 |
|---------|----------|-------------|--------------|------|
| `ChoiceInfo` | `index` | `Choice` | `index` | 选择索引 |
| `ChoiceInfo` | `message` | `Choice` | `message` | 生成的消息 |
| `ChoiceInfo` | `finish_reason` | `Choice` | `finish_reason` | 停止原因 |
| `ChoiceInfo` | `logprobs` | `Choice` | `logprobs` | Log 概率信息 |

---

### 3. 停止原因

#### FinishReason → Choice.finish_reason

| IR Value | OpenAI Value | 说明 |
|----------|--------------|------|
| `"stop"` | `"stop"` | 正常停止 |
| `"length"` | `"length"` | 达到最大长度 |
| `"tool_calls"` | `"tool_calls"` | 工具调用 |
| `"content_filter"` | `"content_filter"` | 内容过滤 |
| `"refusal"` | - | OpenAI 没有此值（但有 refusal 字段） |
| `"error"` | - | OpenAI 没有此值 |
| `"cancelled"` | - | OpenAI 没有此值 |
| - | `"function_call"` | OpenAI 特有（已废弃） |

---

### 4. 消息结构

#### Message → ChatCompletionMessage

| IR Type | IR Field | OpenAI Type | OpenAI Field | 说明 |
|---------|----------|-------------|--------------|------|
| `Message` | `role` | `Literal["assistant"]` | `role` | IR 支持 "system"/"user"/"assistant"，OpenAI 响应只有 "assistant" |
| `Message` | `content` | `Optional[str]` | `content` | IR: List[ContentPart], OpenAI: Optional[str] |
| `Message` | `metadata` | - | - | OpenAI 不支持 |

**OpenAI 类型定义：**
```python
ChatCompletionMessage = BaseModel({
    "content": Optional[str],
    "refusal": Optional[str],
    "role": Literal["assistant"],
    "annotations": Optional[List[Annotation]],
    "audio": Optional[ChatCompletionAudio],
    "function_call": Optional[FunctionCall],  # deprecated
    "tool_calls": Optional[List[ChatCompletionMessageToolCallUnion]]
})
```

**映射说明：**
- IR 的 `content` 是 `List[ContentPart]`，支持多模态
- OpenAI 的 `content` 是 `Optional[str]`，仅支持文本
- OpenAI 的其他内容类型通过独立字段表示（如 `tool_calls`, `refusal`, `audio`, `annotations`）

---

### 5. 内容部分映射

#### TextPart → content

| IR Type | IR Field | OpenAI Type | OpenAI Field | 说明 |
|---------|----------|-------------|--------------|------|
| `TextPart` | `text` | `Optional[str]` | `content` | 文本内容直接映射 |

#### ToolCallPart → tool_calls

| IR Type | IR Field | OpenAI Type | OpenAI Field | 说明 |
|---------|----------|-------------|--------------|------|
| `ToolCallPart` | `tool_call_id` | `str` | `id` | 工具调用ID |
| `ToolCallPart` | `tool_name` | `str` (in `Function`) | `function.name` | 函数名称 |
| `ToolCallPart` | `tool_input` | `str` (in `Function`) | `function.arguments` | IR: Dict, OpenAI: JSON 字符串 |
| `ToolCallPart` | `tool_type` | `Literal["function"]` | `type` | IR 支持多种，OpenAI 仅 "function" |

**OpenAI 类型定义：**
```python
# 工具调用联合类型
ChatCompletionMessageToolCallUnion = Union[
    ChatCompletionMessageFunctionToolCall,
    ChatCompletionMessageCustomToolCall
]

# 函数工具调用
ChatCompletionMessageFunctionToolCall = BaseModel({
    "id": str,
    "type": Literal["function"],
    "function": Function
})

# 函数定义
Function = BaseModel({
    "name": str,
    "arguments": str  # JSON 字符串
})
```

**映射说明：**
- IR 的 `tool_input` 是 `Dict[str, Any]`
- OpenAI 的 `arguments` 是 JSON 字符串，需要序列化/反序列化
- OpenAI 的 `tool_calls` 是一个列表：`Optional[List[ChatCompletionMessageToolCallUnion]]`

#### RefusalPart → refusal

| IR Type | IR Field | OpenAI Type | OpenAI Field | 说明 |
|---------|----------|-------------|--------------|------|
| `RefusalPart` | `refusal` | `Optional[str]` | `refusal` | 拒绝原因文本 |

#### ReasoningPart → (不直接映射)

OpenAI 的推理内容包含在 token 统计中，但不作为独立的内容部分返回。

**映射说明：**
- IR 的 `ReasoningPart` 用于存储推理过程
- OpenAI 通过 `usage.completion_tokens_details.reasoning_tokens` 统计推理 token

#### CitationPart → annotations

| IR Type | IR Field | OpenAI Type | OpenAI Field | 说明 |
|---------|----------|-------------|--------------|------|
| `CitationPart` | `url_citation` | `AnnotationURLCitation` (in `Annotation`) | `url_citation` | URL 引用 |

**OpenAI 类型定义：**
```python
# 注释类型
Annotation = BaseModel({
    "type": Literal["url_citation"],
    "url_citation": AnnotationURLCitation
})

# URL 引用
AnnotationURLCitation = BaseModel({
    "start_index": int,
    "end_index": int,
    "title": str,
    "url": str
})
```

**映射说明：**
- OpenAI 的 `annotations` 是一个列表：`Optional[List[Annotation]]`

#### AudioPart → audio

| IR Type | IR Field | OpenAI Type | OpenAI Field | 说明 |
|---------|----------|-------------|--------------|------|
| `AudioPart` | `audio_id` | `str` | `id` | 音频ID |

**OpenAI 类型定义：**
```python
ChatCompletionAudio = BaseModel({
    "id": str,
    "expires_at": int,
    "data": str,  # base64
    "transcript": str
})
```

**映射说明：**
- OpenAI 的 `audio` 字段类型为 `Optional[ChatCompletionAudio]`

---

### 6. Token 使用统计

#### UsageInfo → CompletionUsage

| IR Type | IR Field | OpenAI Type | OpenAI Field | 说明 |
|---------|----------|-------------|--------------|------|
| `UsageInfo` | `prompt_tokens` | `int` | `prompt_tokens` | 输入 token 数 |
| `UsageInfo` | `completion_tokens` | `int` | `completion_tokens` | 输出 token 数 |
| `UsageInfo` | `total_tokens` | `int` | `total_tokens` | 总 token 数 |
| `UsageInfo` | `reasoning_tokens` | `Optional[int]` (in `CompletionTokensDetails`) | `completion_tokens_details.reasoning_tokens` | 推理 token 数 |
| `UsageInfo` | `prompt_tokens_details` | `Optional[PromptTokensDetails]` | `prompt_tokens_details` | 输入详细统计 |
| `UsageInfo` | `completion_tokens_details` | `Optional[CompletionTokensDetails]` | `completion_tokens_details` | 输出详细统计 |
| `UsageInfo` | `cache_read_tokens` | `Optional[int]` (in `PromptTokensDetails`) | `prompt_tokens_details.cached_tokens` | 缓存读取 token 数 |

**OpenAI 类型定义：**
```python
# 使用统计
CompletionUsage = BaseModel({
    "prompt_tokens": int,
    "completion_tokens": int,
    "total_tokens": int,
    "prompt_tokens_details": Optional[PromptTokensDetails],
    "completion_tokens_details": Optional[CompletionTokensDetails]
})

# 输入详细统计
PromptTokensDetails = BaseModel({
    "cached_tokens": Optional[int],
    "audio_tokens": Optional[int]
})

# 输出详细统计
CompletionTokensDetails = BaseModel({
    "reasoning_tokens": Optional[int],
    "audio_tokens": Optional[int],
    "accepted_prediction_tokens": Optional[int],
    "rejected_prediction_tokens": Optional[int]
})
```

---

## 关键差异总结

### 1. 结构差异

| 方面 | IR | OpenAI |
|------|----|----|
| 工具定义 | 扁平结构 | 嵌套结构 `{"type": "function", "function": {...}}` |
| 消息内容 | `List[ContentPart]` 多模态 | `str` 文本 + 独立字段（tool_calls, refusal 等） |
| 工具调用参数 | `Dict[str, Any]` | JSON 字符串 |
| 系统指令 | 独立字段 `system_instruction` | messages 数组中的 system 消息 |

### 2. 功能差异

| 功能 | IR | OpenAI |
|------|----|----|
| 工具类型 | 支持 function, mcp, web_search, code_interpreter, file_search | 仅支持 function |
| top_k 采样 | 支持 | 不支持 |
| 推理预算 | 支持 `budget_tokens` | 不支持 |
| 扩展项 | 支持 SystemEvent, BatchMarker 等 | 不支持 |

### 3. 命名差异

| IR | OpenAI | 说明 |
|----|--------|------|
| `mode: "any"` | `"required"` | 必须使用工具 |
| `max_tokens` | `max_completion_tokens` | 最大生成 token 数 |
| `stop_sequences` | `stop` | 停止序列 |
| `object: "response"` | `object: "chat.completion"` | 对象类型 |

---

## 转换注意事项

### Request 转换

1. **系统指令处理**：将 `system_instruction` 转换为 messages 数组的第一条消息
2. **工具定义嵌套**：将扁平的 `ToolDefinition` 转换为嵌套的 `{"type": "function", "function": {...}}`
3. **工具选择映射**：`mode: "any"` → `"required"`
4. **并行工具调用**：`disable_parallel` 需要取反为 `parallel_tool_calls`
5. **停止序列**：确保 `stop_sequences` 转换为正确的格式

### Response 转换

1. **内容部分展开**：将 OpenAI 的独立字段（content, tool_calls, refusal 等）转换为 IR 的 `List[ContentPart]`
2. **工具调用参数**：将 JSON 字符串 `arguments` 解析为 `Dict[str, Any]`
3. **对象类型**：`"chat.completion"` → `"response"`
4. **推理 token**：从 `completion_tokens_details.reasoning_tokens` 提取到 `UsageInfo.reasoning_tokens`

---

## 示例代码片段

### Request 转换示例

```python
# IR → OpenAI
ir_request: IRRequest = {...}

openai_params = {
    "model": ir_request["model"],
    "messages": convert_ir_messages_to_openai(ir_request["messages"]),
    "temperature": ir_request.get("generation", {}).get("temperature"),
    "max_completion_tokens": ir_request.get("generation", {}).get("max_tokens"),
    "tools": [
        {
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool.get("description"),
                "parameters": tool.get("parameters")
            }
        }
        for tool in ir_request.get("tools", [])
    ],
    "tool_choice": convert_tool_choice(ir_request.get("tool_choice"))
}
```

### Response 转换示例

```python
# OpenAI → IR
openai_response: ChatCompletion = {...}

ir_response: IRResponse = {
    "id": openai_response.id,
    "object": "response",
    "created": openai_response.created,
    "model": openai_response.model,
    "choices": [
        {
            "index": choice.index,
            "message": convert_openai_message_to_ir(choice.message),
            "finish_reason": {"reason": choice.finish_reason},
            "logprobs": choice.logprobs
        }
        for choice in openai_response.choices
    ],
    "usage": convert_usage(openai_response.usage)
}
```

---

## 参考资料

- OpenAI SDK 源码：`/data/pding/miniforge3/envs/llmir/lib/python3.10/site-packages/openai/types/chat/`
- IR Request Types：`src/llmir/types/ir_request.py`
- IR Response Types：`src/llmir/types/ir_response.py`