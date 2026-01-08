# OpenAI Responses API 与 IR Types 映射文档

本文档详细说明了 LLMIR 的 IR types 与 OpenAI Responses API 原生类型之间的映射关系。

## 目录

1. [IR Request Types 映射](#ir-request-types-映射)
2. [IR Response Types 映射](#ir-response-types-映射)
3. [详细类型对照表](#详细类型对照表)

---

## IR Request Types 映射

### 1. 核心请求参数

| IR Type | OpenAI Responses API Type | 说明 |
|---------|---------------------------|------|
| `IRRequest.model` | `ResponseCreateParams.model: ResponsesModel` | 模型ID |
| `IRRequest.messages` | `ResponseCreateParams.input: Union[str, ResponseInputParam]` | 输入消息 |
| `IRRequest.system_instruction` | `ResponseCreateParams.instructions: Optional[str]` | 系统指令 |

### 2. 工具相关类型

#### 2.1 ToolDefinition → Tool

| IR Field | OpenAI Type | 对应关系 |
|----------|-------------|----------|
| `ToolDefinition.type` | `Tool` (discriminated union) | IR的type映射到OpenAI的具体工具类型 |
| `ToolDefinition.name` | `FunctionTool.function.name` | 函数工具名称 |
| `ToolDefinition.description` | `FunctionTool.function.description` | 函数工具描述 |
| `ToolDefinition.parameters` | `FunctionTool.function.parameters` | JSON Schema参数 |

**OpenAI Tool Types:**
```python
Tool = Union[
    FunctionTool,           # type="function"
    FileSearchTool,         # type="file_search"
    ComputerTool,          # type="computer"
    WebSearchTool,         # type="web_search"
    Mcp,                   # type="mcp"
    CodeInterpreter,       # type="code_interpreter"
    ImageGeneration,       # type="image_generation"
    LocalShell,            # type="local_shell"
    FunctionShellTool,     # type="shell"
    CustomTool,            # type="custom"
    WebSearchPreviewTool,  # type="web_search_preview"
    ApplyPatchTool,        # type="apply_patch"
]
```

#### 2.2 ToolChoice → ToolChoice

| IR Field | OpenAI Type | 说明 |
|----------|-------------|------|
| `ToolChoice.mode: "none"` | `ToolChoiceOptions: "none"` | 不使用工具 |
| `ToolChoice.mode: "auto"` | `ToolChoiceOptions: "auto"` | 自动选择 |
| `ToolChoice.mode: "any"` | `ToolChoiceOptions: "required"` | 必须使用工具 |
| `ToolChoice.mode: "tool"` | `ToolChoiceFunction/ToolChoiceMcp/...` | 指定工具 |

**OpenAI ToolChoice Types:**
```python
ToolChoice = Union[
    ToolChoiceOptions,        # "auto" | "none" | "required"
    ToolChoiceAllowed,        # {"type": "allowed", "allowed_types": [...]}
    ToolChoiceTypes,          # {"type": "types", "types": [...]}
    ToolChoiceFunction,       # {"type": "function", "function": {"name": "..."}}
    ToolChoiceMcp,           # {"type": "mcp", ...}
    ToolChoiceCustom,        # {"type": "custom", ...}
    ToolChoiceApplyPatch,    # {"type": "apply_patch"}
    ToolChoiceShell,         # {"type": "shell"}
]
```

#### 2.3 ToolCallConfig → 请求参数

| IR Field | OpenAI Field | 说明 |
|----------|--------------|------|
| `ToolCallConfig.disable_parallel` | `ResponseCreateParams.parallel_tool_calls: bool` | 禁用并行工具调用（取反） |
| `ToolCallConfig.max_calls` | `ResponseCreateParams.max_tool_calls: Optional[int]` | 最大工具调用数 |

### 3. 生成控制参数

#### 3.1 GenerationConfig → ResponseCreateParams

| IR Field | OpenAI Field | 说明 |
|----------|--------------|------|
| `temperature` | `temperature: Optional[float]` | 温度参数 (0-2) |
| `top_p` | `top_p: Optional[float]` | Nucleus采样 |
| `max_tokens` | `max_output_tokens: Optional[int]` | 最大输出token数 |
| `stop_sequences` | ❌ 不支持 | OpenAI Responses API不支持stop参数 |
| `truncation` | `truncation: Optional[Literal["auto", "disabled"]]` | 截断策略 |
| `logprobs` | `top_logprobs: Optional[int]` | Log概率（通过include参数启用） |
| `top_logprobs` | `top_logprobs: Optional[int]` | Top log概率数量 |

**注意:** OpenAI Responses API 不支持以下参数：
- `top_k` (仅Anthropic/Google支持)
- `frequency_penalty` (仅Chat API支持)
- `presence_penalty` (仅Chat API支持)
- `logit_bias` (仅Chat API支持)
- `seed` (仅Chat API支持)
- `n` (仅Chat API支持)

### 4. 响应格式配置

#### 4.1 ResponseFormatConfig → ResponseTextConfig

| IR Field | OpenAI Field | 说明 |
|----------|--------------|------|
| `type: "text"` | `ResponseTextConfig.type: "text"` | 纯文本 |
| `type: "json_object"` | `ResponseTextConfig.type: "json_object"` | JSON对象 |
| `type: "json_schema"` | `ResponseTextConfig.type: "json_schema"` | JSON Schema |
| `json_schema` | `ResponseTextConfig.json_schema` | Schema定义 |

**OpenAI ResponseTextConfig:**
```python
ResponseTextConfig = Union[
    ResponseFormatTextConfig,           # {"type": "text"}
    ResponseFormatTextJsonSchemaConfig  # {"type": "json_schema", "json_schema": {...}}
]
```

### 5. 推理配置

#### 5.1 ReasoningConfig → Reasoning

| IR Field | OpenAI Field | 说明 |
|----------|--------------|------|
| `effort: "low"` | `Reasoning.effort: "low"` | 低推理努力 |
| `effort: "medium"` | `Reasoning.effort: "medium"` | 中等推理努力 |
| `effort: "high"` | `Reasoning.effort: "high"` | 高推理努力 |
| `type: "enabled"` | `Reasoning.type: "enabled"` | 启用推理 |
| `type: "disabled"` | `Reasoning.type: "disabled"` | 禁用推理 |

**注意:** `budget_tokens` 在OpenAI中不直接支持，通过 `max_output_tokens` 间接控制。

### 6. 流式输出配置

#### 6.1 StreamConfig → stream & StreamOptions

| IR Field | OpenAI Field | 说明 |
|----------|--------------|------|
| `enabled: true` | `stream: true` | 启用流式输出 |
| `enabled: false` | `stream: false` | 禁用流式输出 |
| `include_usage` | `stream_options.include_obfuscation: bool` | 流式选项 |

### 7. 缓存配置

#### 7.1 CacheConfig → 缓存参数

| IR Field | OpenAI Field | 说明 |
|----------|--------------|------|
| `key` | `prompt_cache_key: str` | 缓存键 |
| `retention: "in-memory"` | `prompt_cache_retention: "in-memory"` | 内存缓存 |
| `retention: "24h"` | `prompt_cache_retention: "24h"` | 24小时缓存 |

### 8. Provider特定扩展

以下参数通过 `IRRequest.provider_extensions` 传递：

| Extension Key | OpenAI Field | 说明 |
|---------------|--------------|------|
| `metadata` | `metadata: Optional[Metadata]` | 元数据 (16个key-value对) |
| `user` | `user: str` (已废弃) | 用户标识符 |
| `safety_identifier` | `safety_identifier: str` | 安全标识符 |
| `service_tier` | `service_tier: Literal[...]` | 服务等级 |
| `store` | `store: Optional[bool]` | 是否存储响应 |
| `background` | `background: Optional[bool]` | 后台运行 |
| `conversation` | `conversation: Union[str, ResponseConversationParam]` | 会话ID |
| `previous_response_id` | `previous_response_id: Optional[str]` | 前一个响应ID |
| `prompt` | `prompt: Optional[ResponsePromptParam]` | 提示模板引用 |
| `include` | `include: Optional[List[ResponseIncludable]]` | 额外输出数据 |

---

## IR Response Types 映射

### 1. 顶层响应类型

#### 1.1 IRResponse → Response

| IR Field | OpenAI Field | 说明 |
|----------|--------------|------|
| `id` | `id: str` | 响应唯一ID |
| `object: "response"` | `object: Literal["response"]` | 对象类型 |
| `created` | `created_at: float` | Unix时间戳 |
| `model` | `model: ResponsesModel` | 模型ID |
| `choices` | `output: List[ResponseOutputItem]` | 输出项列表 |
| `usage` | `usage: Optional[ResponseUsage]` | Token使用统计 |
| `service_tier` | `service_tier: Optional[str]` | 服务等级 |
| `system_fingerprint` | ❌ 不存在 | OpenAI Responses API无此字段 |

**注意:** OpenAI Responses API 使用 `output` 而非 `choices`，且结构不同。

### 2. 消息类型

#### 2.1 Message → ResponseInputItem/ResponseOutputItem

OpenAI Responses API 区分输入和输出项：

**输入项 (ResponseInputItem):**
```python
ResponseInputItem = Union[
    EasyInputMessage,                    # 简化消息
    Message,                             # 标准消息
    ResponseOutputMessage,               # 输出消息（可作为输入）
    ResponseFileSearchToolCall,          # 文件搜索工具调用
    ResponseComputerToolCall,            # 计算机工具调用
    ComputerCallOutput,                  # 计算机调用输出
    ResponseFunctionWebSearch,           # Web搜索
    ResponseFunctionToolCall,            # 函数工具调用
    FunctionCallOutput,                  # 函数调用输出
    ResponseReasoningItem,               # 推理项
    ResponseCompactionItemParam,         # 压缩项
    ImageGenerationCall,                 # 图像生成调用
    ResponseCodeInterpreterToolCall,     # 代码解释器调用
    LocalShellCall,                      # 本地Shell调用
    LocalShellCallOutput,                # Shell调用输出
    ShellCall,                           # Shell调用
    ShellCallOutput,                     # Shell输出
    ApplyPatchCall,                      # 补丁应用调用
    ApplyPatchCallOutput,                # 补丁输出
    McpListTools,                        # MCP工具列表
    McpApprovalRequest,                  # MCP批准请求
    McpApprovalResponse,                 # MCP批准响应
    McpCall,                             # MCP调用
    ResponseCustomToolCallOutput,        # 自定义工具输出
    ResponseCustomToolCall,              # 自定义工具调用
    ItemReference,                       # 项引用
]
```

**输出项 (ResponseOutputItem):**
```python
ResponseOutputItem = Union[
    ResponseOutputMessage,               # 输出消息
    ResponseFileSearchToolCall,          # 文件搜索工具调用
    ResponseFunctionToolCall,            # 函数工具调用
    ResponseFunctionWebSearch,           # Web搜索
    ResponseComputerToolCall,            # 计算机工具调用
    ResponseReasoningItem,               # 推理项
    ResponseCompactionItem,              # 压缩项
    ImageGenerationCall,                 # 图像生成调用
    ResponseCodeInterpreterToolCall,     # 代码解释器调用
    LocalShellCall,                      # 本地Shell调用
    ResponseFunctionShellToolCall,       # Shell工具调用
    ResponseFunctionShellToolCallOutput, # Shell工具输出
    ResponseApplyPatchToolCall,          # 补丁工具调用
    ResponseApplyPatchToolCallOutput,    # 补丁工具输出
    McpCall,                             # MCP调用
    McpListTools,                        # MCP工具列表
    McpApprovalRequest,                  # MCP批准请求
    ResponseCustomToolCall,              # 自定义工具调用
]
```

#### 2.2 Message.role 映射

| IR Role | OpenAI Input Role | OpenAI Output Role |
|---------|-------------------|-------------------|
| `"system"` | `"system"` | ❌ 不存在 |
| `"user"` | `"user"` | ❌ 不存在 |
| `"assistant"` | ❌ 不存在 | `"assistant"` (在ResponseOutputMessage中) |

**注意:** OpenAI还支持 `"developer"` 角色（输入）。

### 3. 内容部分类型

#### 3.1 TextPart → ResponseInputText/ResponseOutputText

| IR Field | OpenAI Input | OpenAI Output | 说明 |
|----------|--------------|---------------|------|
| `type: "text"` | `type: "input_text"` | `type: "output_text"` | 文本类型 |
| `text` | `text: str` | `text: str` | 文本内容 |

**OpenAI ResponseOutputText 额外字段:**
```python
class ResponseOutputText:
    text: str
    type: Literal["output_text"]
    annotations: Optional[List[Annotation]] = None  # 注释（引用等）
    logprobs: Optional[Logprobs] = None            # Log概率
```

#### 3.2 ImagePart → ResponseInputImage

| IR Field | OpenAI Field | 说明 |
|----------|--------------|------|
| `type: "image"` | `type: "input_image"` | 图像类型 |
| `image_url` | `image_url: str` | 图像URL |
| `image_data.data` | `image_url: str` (base64) | Base64编码 |
| `detail` | `detail: Literal["auto", "low", "high"]` | 细节级别 |

#### 3.3 FilePart → ResponseInputFile

| IR Field | OpenAI Field | 说明 |
|----------|--------------|------|
| `type: "file"` | `type: "input_file"` | 文件类型 |
| `file_url` | `file_id: str` | 文件ID（OpenAI使用ID而非URL） |
| `file_data` | ❌ 不支持 | OpenAI需要先上传文件获取ID |

#### 3.4 ToolCallPart → ResponseFunctionToolCall

| IR Field | OpenAI Field | 说明 |
|----------|--------------|------|
| `type: "tool_call"` | `type: "function_call"` | 工具调用类型 |
| `tool_call_id` | `call_id: str` | 调用ID |
| `tool_name` | `name: str` | 工具名称 |
| `tool_input` | `arguments: str` (JSON字符串) | 工具参数 |
| `tool_type` | 通过不同的类型区分 | 工具类型 |

**OpenAI 工具调用类型:**
- `ResponseFunctionToolCall` - 函数调用
- `ResponseFileSearchToolCall` - 文件搜索
- `ResponseComputerToolCall` - 计算机使用
- `ResponseCodeInterpreterToolCall` - 代码解释器
- `ResponseFunctionWebSearch` - Web搜索
- `ResponseCustomToolCall` - 自定义工具
- `McpCall` - MCP工具

#### 3.5 ToolResultPart → FunctionCallOutput

| IR Field | OpenAI Field | 说明 |
|----------|--------------|------|
| `type: "tool_result"` | `type: "function_call_output"` | 工具结果类型 |
| `tool_call_id` | `call_id: str` | 调用ID |
| `result` | `output: Union[str, ResponseFunctionCallOutputItemList]` | 结果 |
| `is_error` | ❌ 通过status字段表示 | 错误标识 |

#### 3.6 ReasoningPart → ResponseReasoningItem

| IR Field | OpenAI Field | 说明 |
|----------|--------------|------|
| `type: "reasoning"` | `type: "reasoning"` | 推理类型 |
| `reasoning` | `summary: str` | 推理摘要 |
| `status` | `status: Literal["in_progress", "completed", "incomplete"]` | 状态 |

**OpenAI ResponseReasoningItem 额外字段:**
```python
class ResponseReasoningItem:
    type: Literal["reasoning"]
    status: Literal["in_progress", "completed", "incomplete"]
    id: Optional[str] = None
    summary: Optional[str] = None
    encrypted_content: Optional[str] = None  # 加密的推理内容
```

#### 3.7 RefusalPart → ResponseOutputRefusal

| IR Field | OpenAI Field | 说明 |
|----------|--------------|------|
| `type: "refusal"` | `type: "output_refusal"` | 拒绝类型 |
| `refusal` | `refusal: str` | 拒绝原因 |

#### 3.8 CitationPart → Annotations

OpenAI 通过 `ResponseOutputText.annotations` 处理引用：

```python
# OpenAI 注释类型
Annotation = Union[
    URLCitation,      # URL引用
    FileCitation,     # 文件引用
]

class URLCitation:
    type: Literal["url_citation"]
    text: str
    start_index: int
    end_index: int
    url: str
    title: Optional[str] = None
```

#### 3.9 AudioPart → ResponseInputAudio

| IR Field | OpenAI Field | 说明 |
|----------|--------------|------|
| `type: "audio"` | `type: "input_audio"` | 音频类型 |
| `audio_id` | `audio: str` (base64) | 音频数据 |

### 4. 使用统计

#### 4.1 UsageInfo → ResponseUsage

| IR Field | OpenAI Field | 说明 |
|----------|--------------|------|
| `prompt_tokens` | `input_tokens: int` | 输入token数 |
| `completion_tokens` | `output_tokens: int` | 输出token数 |
| `reasoning_tokens` | `output_tokens_details.reasoning_tokens: int` | 推理token数 |
| `total_tokens` | `total_tokens: int` | 总token数 |
| `cache_read_tokens` | `input_tokens_details.cached_tokens: int` | 缓存读取token数 |

**OpenAI ResponseUsage 结构:**
```python
class ResponseUsage:
    input_tokens: int
    output_tokens: int
    total_tokens: int
    input_tokens_details: Optional[InputTokensDetails] = None
    output_tokens_details: Optional[OutputTokensDetails] = None

class InputTokensDetails:
    cached_tokens: Optional[int] = None
    text_tokens: Optional[int] = None
    audio_tokens: Optional[int] = None
    image_tokens: Optional[int] = None

class OutputTokensDetails:
    text_tokens: Optional[int] = None
    audio_tokens: Optional[int] = None
    reasoning_tokens: Optional[int] = None
```

### 5. 停止原因

#### 5.1 FinishReason → ResponseStatus

| IR Reason | OpenAI Status | 说明 |
|-----------|---------------|------|
| `"stop"` | `"completed"` | 正常完成 |
| `"length"` | `"incomplete"` + `incomplete_details.reason: "max_output_tokens"` | 达到长度限制 |
| `"tool_calls"` | `"completed"` (有工具调用) | 工具调用 |
| `"content_filter"` | `"incomplete"` + `incomplete_details.reason: "content_filter"` | 内容过滤 |
| `"error"` | `"failed"` + `error` 对象 | 错误 |
| `"cancelled"` | `"cancelled"` | 取消 |

**OpenAI ResponseStatus:**
```python
ResponseStatus = Literal[
    "completed",      # 完成
    "failed",         # 失败
    "in_progress",    # 进行中
    "cancelled",      # 取消
    "queued",         # 排队中
    "incomplete",     # 不完整
]
```

---

## 详细类型对照表

### Request 参数完整映射

```python
# IR Request
IRRequest = {
    "model": str,                              # → ResponseCreateParams.model
    "messages": IRInput,                       # → ResponseCreateParams.input
    "system_instruction": str | List[Dict],    # → ResponseCreateParams.instructions
    "tools": List[ToolDefinition],             # → ResponseCreateParams.tools
    "tool_choice": ToolChoice,                 # → ResponseCreateParams.tool_choice
    "tool_config": ToolCallConfig,             # → parallel_tool_calls, max_tool_calls
    "generation": GenerationConfig,            # → temperature, top_p, max_output_tokens, etc.
    "response_format": ResponseFormatConfig,   # → ResponseCreateParams.text
    "stream": StreamConfig,                    # → stream, stream_options
    "reasoning": ReasoningConfig,              # → ResponseCreateParams.reasoning
    "cache": CacheConfig,                      # → prompt_cache_key, prompt_cache_retention
    "provider_extensions": Dict[str, Any],     # → 其他OpenAI特定参数
}

# OpenAI ResponseCreateParams
ResponseCreateParams = {
    "model": ResponsesModel,
    "input": Union[str, ResponseInputParam],
    "instructions": Optional[str],
    "tools": Iterable[ToolParam],
    "tool_choice": ToolChoice,
    "parallel_tool_calls": Optional[bool],
    "max_tool_calls": Optional[int],
    "temperature": Optional[float],
    "top_p": Optional[float],
    "max_output_tokens": Optional[int],
    "top_logprobs": Optional[int],
    "truncation": Optional[Literal["auto", "disabled"]],
    "text": ResponseTextConfigParam,
    "stream": Optional[bool],
    "stream_options": Optional[StreamOptions],
    "reasoning": Optional[Reasoning],
    "prompt_cache_key": str,
    "prompt_cache_retention": Optional[Literal["in-memory", "24h"]],
    # 以下为provider特定参数
    "metadata": Optional[Metadata],
    "safety_identifier": str,
    "service_tier": Optional[Literal[...]],
    "store": Optional[bool],
    "background": Optional[bool],
    "conversation": Optional[Conversation],
    "previous_response_id": Optional[str],
    "prompt": Optional[ResponsePromptParam],
    "include": Optional[List[ResponseIncludable]],
    "user": str,  # 已废弃
}
```

### Response 结构完整映射

```python
# IR Response
IRResponse = {
    "id": str,                                 # → Response.id
    "object": "response",                      # → Response.object
    "created": int,                            # → Response.created_at
    "model": str,                              # → Response.model
    "choices": List[ChoiceInfo],               # → Response.output (结构不同)
    "usage": UsageInfo,                        # → Response.usage
    "service_tier": str,                       # → Response.service_tier
}

# OpenAI Response
Response = {
    "id": str,
    "object": Literal["response"],
    "created_at": float,
    "model": ResponsesModel,
    "output": List[ResponseOutputItem],        # 不是choices!
    "usage": Optional[ResponseUsage],
    "service_tier": Optional[str],
    "status": Optional[ResponseStatus],
    "error": Optional[ResponseError],
    "incomplete_details": Optional[IncompleteDetails],
    # 以下为请求参数的回显
    "instructions": Union[str, List[ResponseInputItem], None],
    "metadata": Optional[Metadata],
    "parallel_tool_calls": bool,
    "temperature": Optional[float],
    "tool_choice": ToolChoice,
    "tools": List[Tool],
    "top_p": Optional[float],
    "background": Optional[bool],
    "conversation": Optional[Conversation],
    "max_output_tokens": Optional[int],
    "max_tool_calls": Optional[int],
    "previous_response_id": Optional[str],
    "prompt": Optional[ResponsePrompt],
    "prompt_cache_key": Optional[str],
    "prompt_cache_retention": Optional[Literal["in-memory", "24h"]],
    "reasoning": Optional[Reasoning],
    "safety_identifier": Optional[str],
    "text": Optional[ResponseTextConfig],
    "top_logprobs": Optional[int],
    "truncation": Optional[Literal["auto", "disabled"]],
    "user": Optional[str],
}
```

---

## 关键差异总结

### 1. 结构差异

- **IR**: 使用 `choices` 数组，每个choice包含一个message
- **OpenAI**: 使用 `output` 数组，直接包含输出项（消息、工具调用等）

### 2. 消息角色差异

- **IR**: `system`, `user`, `assistant` 三种角色
- **OpenAI**: 输入支持 `system`, `user`, `developer`；输出只有 `assistant`

### 3. 内容类型差异

- **IR**: 统一的 `ContentPart` 类型系统
- **OpenAI**: 区分输入和输出类型（`input_text` vs `output_text`）

### 4. 工具调用差异

- **IR**: 统一的 `ToolCallPart` 类型，通过 `tool_type` 区分
- **OpenAI**: 不同的工具调用类型（`function_call`, `file_search_call`, 等）

### 5. 参数支持差异

**IR支持但OpenAI Responses API不支持:**
- `top_k`
- `frequency_penalty`
- `presence_penalty`
- `logit_bias`
- `seed`
- `n` (多个选择)
- `stop_sequences`

**OpenAI Responses API特有:**
- `conversation` (会话管理)
- `previous_response_id` (多轮对话)
- `prompt` (提示模板)
- `include` (额外输出数据)
- `background` (后台运行)
- `store` (存储响应)
- `truncation` (截断策略)

### 6. 文件处理差异

- **IR**: 支持 `file_url` 和 `file_data` (base64)
- **OpenAI**: 只支持 `file_id`，需要先上传文件

---

## 转换注意事项

### Request 转换

1. **消息转换**: IR的 `messages` 需要转换为OpenAI的 `input` 格式
2. **工具定义**: IR的 `ToolDefinition` 需要根据 `type` 转换为对应的OpenAI工具类型
3. **工具选择**: IR的 `ToolChoice.mode` 需要映射到OpenAI的具体类型
4. **参数过滤**: 不支持的参数需要过滤或通过 `provider_extensions` 处理

### Response 转换

1. **输出结构**: OpenAI的 `output` 数组需要转换为IR的 `choices` 结构
2. **内容类型**: 输入/输出类型需要统一为IR的 `ContentPart` 类型
3. **状态映射**: OpenAI的 `status` 和 `incomplete_details` 需要映射为IR的 `finish_reason`
4. **使用统计**: OpenAI的嵌套 `usage` 结构需要扁平化为IR格式

---

## 示例代码

### Request 转换示例

```python
def ir_to_openai_request(ir_request: IRRequest) -> ResponseCreateParams:
    """将IR请求转换为OpenAI Responses API请求"""
    params = {
        "model": ir_request["model"],
        "input": convert_ir_messages_to_input(ir_request["messages"]),
    }
    
    # 系统指令
    if "system_instruction" in ir_request:
        params["instructions"] = ir_request["system_instruction"]
    
    # 工具
    if "tools" in ir_request:
        params["tools"] = [
            convert_tool_definition(tool) 
            for tool in ir_request["tools"]
        ]
    
    # 工具选择
    if "tool_choice" in ir_request:
        params["tool_choice"] = convert_tool_choice(ir_request["tool_choice"])
    
    # 生成参数
    if "generation" in ir_request:
        gen = ir_request["generation"]
        if "temperature" in gen:
            params["temperature"] = gen["temperature"]
        if "top_p" in gen:
            params["top_p"] = gen["top_p"]
        if "max_tokens" in gen:
            params["max_output_tokens"] = gen["max_tokens"]
    
    return params
```

### Response 转换示例

```python
def openai_to_ir_response(openai_response: Response) -> IRResponse:
    """将OpenAI响应转换为IR响应"""
    # 转换output为choices
    choices = []
    for idx, output_item in enumerate(openai_response.output):
        if output_item.type == "message":
            choice = {
                "index": idx,
                "message": convert_output_message(output_item),
                "finish_reason": determine_finish_reason(openai_response),
            }
            choices.append(choice)
    
    return {
        "id": openai_response.id,
        "object": "response",
        "created": int(openai_response.created_at),
        "model": openai_response.model,
        "choices": choices,
        "usage": convert_usage(openai_response.usage) if openai_response.usage else None,
    }
```

---

## 参考资料

- [OpenAI Responses API 文档](https://platform.openai.com/docs/api-reference/responses)
- [OpenAI SDK Types](https://github.com/openai/openai-python/tree/main/src/openai/types/responses)
- [LLMIR IR Types 设计](./ir_design_final.md)