# SDK Body 级别参数结构分析

基于对四个主要 SDK 入口函数的分析，我们发现了完整的"body"参数结构。这些结构代表了每个 SDK 的完整 API 调用参数集合。

## 分类说明

- **输入内容核心参数**: user/system/developer message、对话历史、内容输入等
- **工具相关参数**: tools, tool_choice, tool_use 等
- **生成控制参数**: temperature, top_p, max_tokens 等生成质量控制
- **控制参数**: stream, stop_sequences, metadata 等运行时控制
- **其他**: SDK特有参数、系统参数等

---

## 1. OpenAI Chat Completions

**入口函数**: `openai_client.chat.completions.create()`  
**参数总数**: 37个

### 1.1 输入内容核心参数

| 参数名 | 类型 | 必需 | 说明 |
|--------|------|------|------|
| `messages` | `Iterable[ChatCompletionMessageParam]` | ✅ | 对话消息列表，类型为 `Union[ChatCompletionSystemMessageParam, ChatCompletionUserMessageParam, ChatCompletionAssistantMessageParam, ...]` |
| `model` | `Union[str, ChatModel]` | ✅ | 模型ID，如 `gpt-4o`, `o3` 等 |

### 1.2 工具相关参数

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `tools` | `Iterable[ChatCompletionToolUnionParam]` | `omit` | 可用工具列表，类型为 `Union[Function, CodeInterpreter, FileSearch]` |
| `tool_choice` | `ChatCompletionToolChoiceOptionParam` | `omit` | 工具选择策略，类型为 `Union[Literal["auto", "none", "required"], ChatCompletionToolChoiceFunction]` |
| `parallel_tool_calls` | `bool` | `omit` | 是否允许并行工具调用 |

### 1.3 生成控制参数

| 参数名 | 类型 | 默认值 | 范围/说明 |
|--------|------|--------|-----------|
| `temperature` | `Optional[float]` | `omit` | 范围: `0.0-2.0`，值越低越确定性 |
| `top_p` | `Optional[float]` | `omit` | nucleus采样，范围: `0.0-1.0` |
| `max_completion_tokens` | `Optional[int]` | `omit` | 最大生成的token数 |
| `max_tokens` | `Optional[int]` | `omit` | 最大token数（别名，与max_completion_tokens相同） |
| `n` | `Optional[int]` | `omit` | 生成的选择数量 |
| `frequency_penalty` | `Optional[float]` | `omit` | 频率惩罚，范围: `-2.0 到 2.0` |
| `presence_penalty` | `Optional[float]` | `omit` | 存在惩罚，范围: `-2.0 到 2.0` |
| `logit_bias` | `Optional[Dict[str, int]]` | `omit` | logit偏置，key为token ID，value为偏置值 |
| `seed` | `Optional[int]` | `omit` | 随机种子，用于可重现的输出 |
| `top_logprobs` | `Optional[int]` | `omit` | 返回top k个token的log概率 |

### 1.4 控制参数

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `stop` | `Union[Optional[str], SequenceNotStr[str], None]` | `omit` | 停止序列，字符串或字符串列表 |
| `stream` | `Optional[Literal[False]]` | `omit` | 是否流式输出，默认为是 |
| `stream_options` | `Optional[ChatCompletionStreamOptionsParam]` | `omit` | 流式输出选项，包含`include_usage`等 |
| `response_format` | `completion_create_params.ResponseFormat` | `omit` | 响应格式，如JSON模式 |
| `logprobs` | `Optional[bool]` | `omit` | 是否返回log概率 |
| `user` | `str` | `omit` | 用户标识 |
| `metadata` | `Optional[Metadata]` | `omit` | 元数据，类型为 `Dict[str, str]` |

### 1.5 其他参数

| 参数名 | 类型 | 说明 |
|--------|------|------|
| `audio` | `Optional[ChatCompletionAudioParam]` | 音频输出配置，需配合`modalities: ["audio"]`使用 |
| `modalities` | `Optional[List[Literal["text", "audio"]]]` | 支持的模态类型 |
| `prediction` | `Optional[ChatCompletionPredictionContentParam]` | 预测内容配置 |
| `prompt_cache_key` | `str` | 提示缓存键 |
| `prompt_cache_retention` | `Optional[Literal["in-memory", "24h"]]` | 提示缓存保留策略 |
| `reasoning_effort` | `Optional[ReasoningEffort]` | 推理努力程度，类型为 `Literal["low", "medium", "high"]` |
| `web_search_options` | `completion_create_params.WebSearchOptions` | 网络搜索选项 |
| `service_tier` | `Optional[Literal["auto", "default", "flex", "scale", "priority"]]` | 服务层级 |
| `function_call` | `completion_create_params.FunctionCall` | ⚠️ **已废弃**，使用`tool_choice`替代 |
| `functions` | `Iterable[completion_create_params.Function]` | ⚠️ **已废弃**，使用`tools`替代 |
| `safety_identifier` | `str` | 安全标识符 |
| `store` | `Optional[bool]` | 是否存储响应 |
| `verbosity` | `Optional[Literal["low", "medium", "high"]]` | 详细程度 |

### 1.6 系统参数

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `extra_headers` | `Headers | None` | `None` | 额外HTTP头 |
| `extra_query` | `Query | None` | `None` | 额外查询参数 |
| `extra_body` | `Body | None` | `None` | 额外请求体 |
| `timeout` | `float | httpx.Timeout | None | NotGiven` | `not_given` | 超时设置 |

---

## 2. OpenAI Responses

**入口函数**: `openai_responses_client.responses.create()`  
**参数总数**: 28个

### 2.1 输入内容核心参数

| 参数名 | 类型 | 必需 | 说明 |
|--------|------|------|------|
| `input` | `Union[str, ResponseInputParam]` | ✅ | 输入内容，支持30+种类型，极其复杂 |
| `model` | `ResponsesModel` | ✅ | 模型ID |
| `instructions` | `Optional[str]` | - | 系统指令，类似于system消息 |
| `conversation` | `Optional[response_create_params.Conversation]` | - | 所属对话，用于多轮对话 |

### 2.2 工具相关参数

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `tools` | `Iterable[ToolParam]` | `omit` | 可用工具列表 |
| `tool_choice` | `response_create_params.ToolChoice` | `omit` | 工具选择策略 |
| `parallel_tool_calls` | `Optional[bool]` | `omit` | 是否允许并行工具调用 |
| `max_tool_calls` | `Optional[int]` | `omit` | 最大工具调用数 |

### 2.3 生成控制参数

| 参数名 | 类型 | 默认值 | 范围/说明 |
|--------|------|--------|-----------|
| `temperature` | `Optional[float]` | `omit` | 范围: `0.0-2.0` |
| `top_p` | `Optional[float]` | `omit` | nucleus采样，范围: `0.0-1.0` |
| `max_output_tokens` | `Optional[int]` | `omit` | 最大输出的token数 |
| `top_logprobs` | `Optional[int]` | `omit` | 返回top k个token的log概率 |
| `frequency_penalty` | `Optional[float]` | `omit` | 频率惩罚，范围: `-2.0 到 2.0` |
| `presence_penalty` | `Optional[float]` | `omit` | 存在惩罚，范围: `-2.0 到 2.0` |
| `logit_bias` | `Optional[Dict[str, int]]` | `omit` | logit偏置 |

### 2.4 控制参数

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `stream` | `Optional[Literal[False]]` | `omit` | 是否流式输出 |
| `stream_options` | `Optional[response_create_params.StreamOptions]` | `omit` | 流式输出选项 |
| `response_format` | - | `omit` | 响应格式（未在参数中直接定义） |
| `truncation` | `Optional[Literal["auto", "disabled"]]` | `omit` | 截断策略 |
| `user` | `str` | `omit` | 用户标识 |
| `metadata` | `Optional[Metadata]` | `omit` | 元数据 |

### 2.5 其他参数

| 参数名 | 类型 | 说明 |
|--------|------|------|
| `background` | `Optional[bool]` | 是否在后台运行 |
| `include` | `Optional[List[ResponseIncludable]]` | 包含的额外数据 |
| `previous_response_id` | `Optional[str]` | 前一个响应ID |
| `prompt` | `Optional[ResponsePromptParam]` | 提示模板引用 |
| `prompt_cache_key` | `str` | 提示缓存键 |
| `prompt_cache_retention` | `Optional[Literal["in-memory", "24h"]]` | 提示缓存保留策略 |
| `reasoning` | `Optional[Reasoning]` | 推理配置 |
| `safety_identifier` | `str` | 安全标识符 |
| `service_tier` | `Optional[Literal["auto", "default", "flex", "scale", "priority"]]` | 服务层级 |
| `store` | `Optional[bool]` | 是否存储响应 |
| `text` | `ResponseTextConfigParam` | 文本配置 |
| `audio` | - | 音频配置（未在参数中直接定义） |
| `modalities` | - | 支持的模态类型（未在参数中直接定义） |
| `prediction` | - | 预测配置（未在参数中直接定义） |
| `reasoning_effort` | - | 推理努力程度（未在参数中直接定义） |

### 2.6 系统参数

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `extra_headers` | `Headers | None` | `None` | 额外HTTP头 |
| `extra_query` | `Query | None` | `None` | 额外查询参数 |
| `extra_body` | `Body | None` | `None` | 额外请求体 |
| `timeout` | `float | httpx.Timeout | None | NotGiven` | `not_given` | 超时设置 |

---

## 3. Anthropic Messages

**入口函数**: `anthropic_client.messages.create()`  
**参数总数**: 14个

### 3.1 输入内容核心参数

| 参数名 | 类型 | 必需 | 说明 |
|--------|------|------|------|
| `messages` | `Iterable[MessageParam]` | ✅ | 对话消息列表，类型为 `Union[TextBlockParam, ImageBlockParam]` |
| `model` | `ModelParam` | ✅ | 模型ID，如 `claude-sonnet-4-20250514` |
| `system` | `Union[str, Iterable[TextBlockParam]]` | - | 系统提示，可以是字符串或多个文本块 |
| `max_tokens` | `int` | ✅ | 最大生成的token数（必需） |

### 3.2 工具相关参数

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `tools` | `Iterable[ToolUnionParam]` | `omit` | 可用工具列表，类型为 `Union[ToolParam, ComputerToolParam, BashToolParam, ...]` |
| `tool_choice` | `ToolChoiceParam` | `omit` | 工具选择策略，类型为 `Union[Literal["auto", "none", "any"], ToolChoiceToolParam]` |
| `thinking` | `ThinkingConfigParam` | `omit` | 思考配置，用于推理模型 |

### 3.3 生成控制参数

| 参数名 | 类型 | 默认值 | 范围/说明 |
|--------|------|--------|-----------|
| `temperature` | `float` | `omit` | 范围: `0.0-1.0` |
| `top_p` | `float` | `omit` | nucleus采样，范围: `0.0-1.0` |
| `top_k` | `int` | `omit` | top-k采样，整数值 |

### 3.4 控制参数

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `stop_sequences` | `SequenceNotStr[str]` | `omit` | 停止序列，字符串列表 |
| `stream` | `Literal[False]` | `omit` | 是否流式输出 |
| `metadata` | `MetadataParam` | `omit` | 元数据 |

### 3.5 其他参数

| 参数名 | 类型 | 说明 |
|--------|------|------|
| `service_tier` | `Literal["auto", "standard_only"]` | 服务层级 |

### 3.6 系统参数

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `extra_headers` | `Headers | None` | `None` | 额外HTTP头 |
| `extra_query` | `Query | None` | `None` | 额外查询参数 |
| `extra_body` | `Body | None` | `None` | 额外请求体 |
| `timeout` | `float | httpx.Timeout | None | NotGiven` | `not_given` | 超时设置 |

---

## 4. Google GenerativeAI

**入口函数**: `google_client.models.generate_content()`  
**参数总数**: 3个顶级参数

### 4.1 输入内容核心参数

| 参数名 | 类型 | 必需 | 说明 |
|--------|------|------|------|
| `contents` | `types.ContentListUnionDict` | ✅ | 输入内容列表，类型为 `Union[List[Content], List[Part], ...]` |
| `model` | `str` | ✅ | 模型ID，如 `gemini-2.0-flash` |

### 4.2 工具相关参数 (通过config传递)

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `tools` | `List[Tool]` | `None` | 可用工具列表，在 `config` 中传递 |

### 4.3 GenerationConfig (生成配置对象)

`config` 参数为 `Optional[types.GenerateContentConfigOrDict]`，包含以下字段：

| 参数名 | 类型 | 默认值 | 范围/说明 |
|--------|------|--------|-----------|
| `temperature` | `Optional[float]` | `None` | 控制随机性，值越低越确定性 |
| `top_p` | `Optional[float]` | `None` | nucleus采样 |
| `top_k` | `Optional[float]` | `None` | top-k采样 |
| `max_output_tokens` | `Optional[int]` | `None` | 最大输出的token数 |
| `candidate_count` | `Optional[int]` | `None` | 生成候选数量 |
| `stop_sequences` | `Optional[list[str]]` | `None` | 停止序列列表 |
| `presence_penalty` | `Optional[float]` | `None` | 存在惩罚 |
| `frequency_penalty` | `Optional[float]` | `None` | 频率惩罚 |
| `seed` | `Optional[int]` | `None` | 随机种子 |
| `response_logprobs` | `Optional[bool]` | `None` | 是否返回log概率 |
| `logprobs` | `Optional[int]` | `None` | 返回log概率的token数量 |
| `response_mime_type` | `Optional[str]` | `None` | 响应MIME类型，如 `text/plain`, `application/json` |
| `response_schema` | `Optional[Schema]` | `None` | 响应模式，用于结构化输出 |
| `response_modalities` | `Optional[List[str]]` | `None` | 响应模态 |

### 4.4 控制参数 (通过config传递)

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `system_instruction` | `Optional[ContentUnion]` | `None` | 系统指令 |
| `safety_settings` | `Optional[List[SafetySetting]]` | `None` | 安全设置 |
| `cached_content` | `Optional[str]` | `None` | 缓存内容引用 |
| `thinking_config` | `Optional[ThinkingConfig]` | `None` | 思考配置 |

### 4.5 其他参数 (通过config传递)

| 参数名 | 类型 | 说明 |
|--------|------|------|
| `http_options` | `Optional[HttpOptions]` | HTTP请求选项 |
| `should_return_http_response` | `Optional[bool]` | 是否返回原始HTTP响应 |

---

## 跨SDK参数映射总结

### 输入内容核心参数映射

| 功能 | OpenAI Chat | OpenAI Responses | Anthropic | Google |
|------|-------------|------------------|-----------|--------|
| **消息列表** | `messages` | `input` | `messages` | `contents` |
| **模型** | `model` | `model` | `model` | `model` |
| **系统指令** | `messages[0].role="system"` | `instructions` | `system` | `config.system_instruction` |

### 工具相关参数映射

| 功能 | OpenAI Chat | OpenAI Responses | Anthropic | Google |
|------|-------------|------------------|-----------|--------|
| **工具列表** | `tools` | `tools` | `tools` | `config.tools` |
| **工具选择** | `tool_choice` | `tool_choice` | `tool_choice` | `config.tool_config` |
| **并行调用** | `parallel_tool_calls` | `parallel_tool_calls` | - | - |
| **思考配置** | - | `reasoning` | `thinking` | `config.thinking_config` |

### 生成控制参数映射

| 功能 | OpenAI Chat | OpenAI Responses | Anthropic | Google |
|------|-------------|------------------|-----------|--------|
| **温度** | `temperature` | `temperature` | `temperature` | `config.temperature` |
| **Top-p** | `top_p` | `top_p` | `top_p` | `config.top_p` |
| **Top-k** | - | - | `top_k` | `config.top_k` |
| **最大token** | `max_completion_tokens` | `max_output_tokens` | `max_tokens` | `config.max_output_tokens` |
| **频率惩罚** | `frequency_penalty` | `frequency_penalty` | - | `config.frequency_penalty` |
| **存在惩罚** | `presence_penalty` | `presence_penalty` | - | `config.presence_penalty` |
| **随机种子** | `seed` | - | - | `config.seed` |
| **Log概率** | `logprobs`, `top_logprobs` | `top_logprobs` | - | `config.response_logprobs`, `config.logprobs` |

### 控制参数映射

| 功能 | OpenAI Chat | OpenAI Responses | Anthropic | Google |
|------|-------------|------------------|-----------|--------|
| **停止序列** | `stop` | - | `stop_sequences` | `config.stop_sequences` |
| **流式输出** | `stream` | `stream` | `stream` | 异步迭代响应 |
| **响应格式** | `response_format` | - | - | `config.response_mime_type`, `config.response_schema` |
| **元数据** | `metadata` | `metadata` | `metadata` | - |
| **用户标识** | `user` | `user` | - | - |

这个分析为设计统一的请求Body结构提供了清晰的参考，我们可以看到不同SDK虽然参数名不同，但核心功能是相似的，这正是body级别转换的价值所在。