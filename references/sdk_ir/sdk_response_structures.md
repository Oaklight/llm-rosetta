# SDK 响应结构详细分析

基于对四个主要 SDK 返回类型的深入分析，本文档详细记录了每个 SDK 的完整响应结构，并识别它们的共性和差异。这些结构代表了每个 SDK 的完整 API 响应格式。

## 分析方法

本分析基于对实际安装的 SDK 源码的检查，包括：

- 直接查看 `/data/pding/miniforge3/envs/llm-rosetta/lib/python3.10/site-packages/` 中的类型定义
- 分析每个 SDK 的响应类型结构
- 深入研究内容字段的内部组织方式
- 识别跨 SDK 的共同模式和独特特性

---

## 1. OpenAI Chat Completions 响应结构

**返回类型**: [`ChatCompletion`](https://github.com/openai/openai-python/blob/main/src/openai/types/chat/chat_completion.py)  
**设计理念**: 简洁的聊天完成响应，专注于对话场景

### 1.1 顶层响应字段 (8 个字段)

| 字段名               | 类型                                                                | 必需 | 说明                                |
| -------------------- | ------------------------------------------------------------------- | ---- | ----------------------------------- |
| `id`                 | `str`                                                               | ✅   | 聊天完成的唯一标识符                |
| `object`             | `Literal["chat.completion"]`                                        | ✅   | 对象类型，始终为"chat.completion"   |
| `created`            | `int`                                                               | ✅   | 聊天完成创建时的 Unix 时间戳（秒）  |
| `model`              | `str`                                                               | ✅   | 用于聊天完成的模型                  |
| `choices`            | `List[Choice]`                                                      | ✅   | **核心响应内容** - 聊天完成选择列表 |
| `usage`              | `Optional[CompletionUsage]`                                         | -    | 完成请求的使用统计                  |
| `service_tier`       | `Optional[Literal["auto", "default", "flex", "scale", "priority"]]` | -    | 用于服务请求的处理类型              |
| `system_fingerprint` | `Optional[str]`                                                     | -    | 模型运行的后端配置指纹              |

### 1.2 Choice 结构 (4 个字段)

| 字段名          | 类型                                                                         | 必需 | 说明                                  |
| --------------- | ---------------------------------------------------------------------------- | ---- | ------------------------------------- |
| `index`         | `int`                                                                        | ✅   | 选择在选择列表中的索引                |
| `message`       | `ChatCompletionMessage`                                                      | ✅   | **核心内容** - 模型生成的聊天完成消息 |
| `finish_reason` | `Literal["stop", "length", "tool_calls", "content_filter", "function_call"]` | ✅   | 模型停止生成 token 的原因             |
| `logprobs`      | `Optional[ChoiceLogprobs]`                                                   | -    | 选择的 log 概率信息                   |

### 1.3 ChatCompletionMessage 内容结构 (7 个字段)

这是 OpenAI Chat Completions 的核心内容载体：

| 字段名          | 类型                                                 | 必需 | 说明                                 |
| --------------- | ---------------------------------------------------- | ---- | ------------------------------------ |
| `role`          | `Literal["assistant"]`                               | ✅   | 消息作者的角色，始终为"assistant"    |
| `content`       | `Optional[str]`                                      | -    | **主要文本内容**                     |
| `refusal`       | `Optional[str]`                                      | -    | **拒绝响应** - 模型生成的拒绝消息    |
| `tool_calls`    | `Optional[List[ChatCompletionMessageToolCallUnion]]` | -    | **工具调用** - 模型生成的工具调用    |
| `function_call` | `Optional[FunctionCall]`                             | -    | ⚠️ **已废弃** - 被 tool_calls 替代   |
| `annotations`   | `Optional[List[Annotation]]`                         | -    | **引用注释** - 如网络搜索的 URL 引用 |
| `audio`         | `Optional[ChatCompletionAudio]`                      | -    | **音频响应** - 音频输出模态数据      |

#### 工具调用详细结构

**ChatCompletionMessageToolCallUnion** 包含：

- `ChatCompletionMessageFunctionToolCall` - 函数工具调用
- `ChatCompletionMessageCustomToolCall` - 自定义工具调用

**FunctionCall** (已废弃):

```python
{
    "name": str,        # 要调用的函数名称
    "arguments": str    # JSON格式的函数调用参数
}
```

**Annotation** (URL 引用):

```python
{
    "type": "url_citation",
    "url_citation": {
        "start_index": int,  # 引用开始位置
        "end_index": int,    # 引用结束位置
        "title": str,        # 网络资源标题
        "url": str          # 网络资源URL
    }
}
```

---

## 2. OpenAI Responses API 响应结构

**返回类型**: [`Response`](https://github.com/openai/openai-python/blob/main/src/openai/types/responses/response.py)  
**设计理念**: 高度模块化的响应系统，支持复杂的交互模式

### 2.1 顶层响应字段 (20+个字段)

| 字段名               | 类型                                        | 必需 | 说明                                 |
| -------------------- | ------------------------------------------- | ---- | ------------------------------------ |
| `id`                 | `str`                                       | ✅   | 此响应的唯一标识符                   |
| `object`             | `Literal["response"]`                       | ✅   | 资源的对象类型，始终设置为"response" |
| `created_at`         | `float`                                     | ✅   | 创建此响应时的 Unix 时间戳（秒）     |
| `model`              | `ResponsesModel`                            | ✅   | 用于生成响应的模型 ID                |
| `output`             | `List[ResponseOutputItem]`                  | ✅   | **核心输出内容列表**                 |
| `status`             | `ResponseStatus`                            | ✅   | 响应的状态                           |
| `usage`              | `Optional[ResponseUsage]`                   | -    | token 使用详情                       |
| `instructions`       | `Union[str, List[ResponseInputItem], None]` | -    | 系统消息                             |
| `metadata`           | `Optional[Metadata]`                        | -    | 16 个键值对集合                      |
| `error`              | `Optional[ResponseError]`                   | -    | 错误对象                             |
| `incomplete_details` | `Optional[IncompleteDetails]`               | -    | 不完整原因详情                       |
| `conversation`       | `Optional[Conversation]`                    | -    | 所属对话                             |
| `prompt`             | `Optional[ResponsePrompt]`                  | -    | 使用的提示                           |
| `reasoning`          | `Optional[Reasoning]`                       | -    | 推理配置和输出                       |
| `text`               | `Optional[ResponseTextConfig]`              | -    | 文本配置                             |
| `tools`              | `Optional[List[Tool]]`                      | -    | 可用工具列表                         |
| `tool_choice`        | `Optional[ToolChoice]`                      | -    | 工具选择配置                         |
| `top_logprobs`       | `Optional[int]`                             | -    | 返回最可能 token 的数量              |
| `truncation`         | `Optional[Literal["auto", "disabled"]]`     | -    | 截断策略                             |
| `user`               | `Optional[str]`                             | -    | 用户标识符                           |

### 2.2 ResponseOutputItem 输出内容结构 (18 种类型)

这是 OpenAI Responses 的核心创新 - 高度模块化的输出项系统：

#### 核心输出类型 (3 种)

- `ResponseOutputMessage` - **助手消息输出**
- `ResponseReasoningItem` - **推理过程项**
- `ResponseCompactionItem` - **压缩项**

#### 工具调用类型 (9 种)

- `ResponseFileSearchToolCall` - 文件搜索工具调用
- `ResponseFunctionToolCall` - 函数工具调用
- `ResponseFunctionWebSearch` - 网络搜索函数
- `ResponseComputerToolCall` - 计算机工具调用
- `ResponseCodeInterpreterToolCall` - 代码解释器工具调用
- `ResponseFunctionShellToolCall` - Shell 函数工具调用
- `ResponseApplyPatchToolCall` - 补丁应用工具调用
- `ResponseCustomToolCall` - 自定义工具调用
- `LocalShellCall` - 本地 Shell 调用

#### 工具输出类型 (2 种)

- `ResponseFunctionShellToolCallOutput` - Shell 函数工具调用输出
- `ResponseApplyPatchToolCallOutput` - 补丁应用工具调用输出

#### 特殊功能类型 (4 种)

- `ImageGenerationCall` - 图像生成调用
- `McpCall` - MCP 调用
- `McpListTools` - MCP 工具列表
- `McpApprovalRequest` - MCP 批准请求

### 2.3 ResponseOutputMessage 消息结构 (4 个字段)

OpenAI Responses 中的核心消息类型：

| 字段名    | 类型                                                | 必需 | 说明                        |
| --------- | --------------------------------------------------- | ---- | --------------------------- |
| `id`      | `str`                                               | ✅   | 输出消息的唯一 ID           |
| `role`    | `Literal["assistant"]`                              | ✅   | 消息角色，始终为"assistant" |
| `content` | `List[Content]`                                     | ✅   | **消息内容列表**            |
| `status`  | `Literal["in_progress", "completed", "incomplete"]` | ✅   | 消息状态                    |

#### Content 内容类型 (2 种)

- `ResponseOutputText` - **文本输出**
- `ResponseOutputRefusal` - **拒绝输出**

**ResponseOutputText** 包含文本内容、注释、日志概率等详细信息，支持文件引用、URL 引用等多种注释类型。

### 2.4 关键工具调用类型详细结构

#### ResponseFunctionToolCall (6 个字段)

| 字段名      | 类型                                                          | 必需 | 说明                                       |
| ----------- | ------------------------------------------------------------- | ---- | ------------------------------------------ |
| `type`      | `Literal["function_call"]`                                    | ✅   | 类型标识，始终为"function_call"            |
| `call_id`   | `str`                                                         | ✅   | **工具调用唯一 ID** - 模型生成的唯一标识符 |
| `name`      | `str`                                                         | ✅   | **函数名称** - 要调用的函数名              |
| `arguments` | `str`                                                         | ✅   | **函数参数** - JSON 字符串格式的参数       |
| `id`        | `Optional[str]`                                               | -    | 工具调用的唯一 ID（平台级别）              |
| `status`    | `Optional[Literal["in_progress", "completed", "incomplete"]]` | -    | **执行状态**                               |

#### ResponseCustomToolCall (4 个字段)

| 字段名    | 类型                          | 必需 | 说明                                       |
| --------- | ----------------------------- | ---- | ------------------------------------------ |
| `type`    | `Literal["custom_tool_call"]` | ✅   | 类型标识，始终为"custom_tool_call"         |
| `call_id` | `str`                         | ✅   | **自定义工具调用 ID** - 用于映射到工具输出 |
| `name`    | `str`                         | ✅   | **自定义工具名称**                         |
| `input`   | `str`                         | ✅   | **工具输入** - 模型生成的输入数据          |
| `id`      | `Optional[str]`               | -    | OpenAI 平台中的唯一 ID                     |

#### McpCall (9 个字段)

| 字段名                | 类型                                                                               | 必需 | 说明                                |
| --------------------- | ---------------------------------------------------------------------------------- | ---- | ----------------------------------- |
| `type`                | `Literal["mcp_call"]`                                                              | ✅   | 类型标识，始终为"mcp_call"          |
| `id`                  | `str`                                                                              | ✅   | **工具调用唯一 ID**                 |
| `name`                | `str`                                                                              | ✅   | **运行的工具名称**                  |
| `server_label`        | `str`                                                                              | ✅   | **MCP 服务器标签**                  |
| `arguments`           | `str`                                                                              | ✅   | **工具参数** - JSON 字符串格式      |
| `approval_request_id` | `Optional[str]`                                                                    | -    | **批准请求 ID** - 用于后续批准/拒绝 |
| `output`              | `Optional[str]`                                                                    | -    | **工具输出**                        |
| `error`               | `Optional[str]`                                                                    | -    | **错误信息**                        |
| `status`              | `Optional[Literal["in_progress", "completed", "incomplete", "calling", "failed"]]` | -    | **执行状态**                        |

#### ResponseReasoningItem (6 个字段)

| 字段名              | 类型                                                          | 必需 | 说明                        |
| ------------------- | ------------------------------------------------------------- | ---- | --------------------------- |
| `type`              | `Literal["reasoning"]`                                        | ✅   | 类型标识，始终为"reasoning" |
| `id`                | `str`                                                         | ✅   | **推理内容唯一标识符**      |
| `summary`           | `List[Summary]`                                               | ✅   | **推理摘要内容**            |
| `content`           | `Optional[List[Content]]`                                     | -    | **推理文本内容**            |
| `encrypted_content` | `Optional[str]`                                               | -    | **加密的推理内容**          |
| `status`            | `Optional[Literal["in_progress", "completed", "incomplete"]]` | -    | **推理状态**                |

**Summary 子结构**:

```python
{
    "type": "summary_text",
    "text": str  # 推理输出摘要
}
```

**Content 子结构**:

```python
{
    "type": "reasoning_text",
    "text": str  # 推理文本
}
```

### 2.5 服务器端工具类型

#### ResponseFunctionWebSearch (4 个字段)

| 字段名   | 类型                                                         | 必需 | 说明                                |
| -------- | ------------------------------------------------------------ | ---- | ----------------------------------- |
| `type`   | `Literal["web_search_call"]`                                 | ✅   | 类型标识，始终为"web_search_call"   |
| `id`     | `str`                                                        | ✅   | **网络搜索工具调用唯一 ID**         |
| `action` | `Action`                                                     | ✅   | **搜索动作详情** - 包含具体搜索行为 |
| `status` | `Literal["in_progress", "searching", "completed", "failed"]` | ✅   | **搜索状态**                        |

**Action 联合类型** (3 种):

- `ActionSearch` - 执行网络搜索查询
- `ActionOpenPage` - 打开搜索结果中的特定 URL
- `ActionFind` - 在加载的页面中搜索模式

#### ResponseCodeInterpreterToolCall (6 个字段)

| 字段名         | 类型                                                                          | 必需 | 说明                                    |
| -------------- | ----------------------------------------------------------------------------- | ---- | --------------------------------------- |
| `type`         | `Literal["code_interpreter_call"]`                                            | ✅   | 类型标识，始终为"code_interpreter_call" |
| `id`           | `str`                                                                         | ✅   | **代码解释器工具调用唯一 ID**           |
| `container_id` | `str`                                                                         | ✅   | **运行代码的容器 ID**                   |
| `code`         | `Optional[str]`                                                               | -    | **要运行的代码**                        |
| `outputs`      | `Optional[List[Output]]`                                                      | -    | **代码执行输出** - 日志或图像           |
| `status`       | `Literal["in_progress", "completed", "incomplete", "interpreting", "failed"]` | ✅   | **执行状态**                            |

**Output 联合类型** (2 种):

- `OutputLogs` - 代码执行日志
- `OutputImage` - 代码生成的图像

---

## 3. Anthropic Messages 响应结构

**返回类型**: [`Message`](https://github.com/anthropics/anthropic-sdk-python/blob/main/src/anthropic/types/message.py)  
**设计理念**: 简洁优雅的内容块架构，平衡功能性和易用性

### 3.1 顶层响应字段 (8 个字段)

| 字段名          | 类型                   | 必需 | 说明                                  |
| --------------- | ---------------------- | ---- | ------------------------------------- |
| `id`            | `str`                  | ✅   | 唯一对象标识符                        |
| `type`          | `Literal["message"]`   | ✅   | 对象类型，始终为"message"             |
| `role`          | `Literal["assistant"]` | ✅   | 生成消息的对话角色，始终为"assistant" |
| `content`       | `List[ContentBlock]`   | ✅   | **核心内容** - 模型生成的内容块数组   |
| `model`         | `Model`                | ✅   | 完成提示的模型                        |
| `usage`         | `Usage`                | ✅   | 计费和速率限制使用情况                |
| `stop_reason`   | `Optional[StopReason]` | -    | 停止原因                              |
| `stop_sequence` | `Optional[str]`        | -    | 生成的自定义停止序列                  |

### 3.2 ContentBlock 内容结构 (6 种类型)

Anthropic 的核心创新 - 统一的内容块架构：

#### 基础内容块

- `TextBlock` - **文本内容块**
- `ThinkingBlock` - **思考过程块**
- `RedactedThinkingBlock` - **编辑后的思考块**

#### 工具相关块

- `ToolUseBlock` - **工具使用块**
- `ServerToolUseBlock` - **服务器工具使用块**
- `WebSearchToolResultBlock` - **网络搜索工具结果块**

### 3.3 详细内容块结构

#### TextBlock (3 个字段)

| 字段名      | 类型                           | 必需 | 说明                                        |
| ----------- | ------------------------------ | ---- | ------------------------------------------- |
| `type`      | `Literal["text"]`              | ✅   | 内容块类型，始终为"text"                    |
| `text`      | `str`                          | ✅   | **文本内容**                                |
| `citations` | `Optional[List[TextCitation]]` | -    | **引用支持** - 根据文档类型返回不同引用格式 |

#### ToolUseBlock (4 个字段)

| 字段名  | 类型                  | 必需 | 说明                         |
| ------- | --------------------- | ---- | ---------------------------- |
| `type`  | `Literal["tool_use"]` | ✅   | 内容块类型，始终为"tool_use" |
| `id`    | `str`                 | ✅   | **工具使用唯一标识符**       |
| `name`  | `str`                 | ✅   | **工具名称**                 |
| `input` | `Dict[str, object]`   | ✅   | **工具输入参数**             |

#### ThinkingBlock

用于推理模型的思考过程，包含模型的内部推理步骤，通常不显示给最终用户。

#### WebSearchToolResultBlock (3 个字段)

| 字段名        | 类型                                | 必需 | 说明                                       |
| ------------- | ----------------------------------- | ---- | ------------------------------------------ |
| `type`        | `Literal["web_search_tool_result"]` | ✅   | 内容块类型，始终为"web_search_tool_result" |
| `tool_use_id` | `str`                               | ✅   | **关联的工具使用 ID**                      |
| `content`     | `WebSearchToolResultBlockContent`   | ✅   | **搜索结果内容**                           |

包含网络搜索工具的执行结果，是 Anthropic 特有的服务器端工具功能。

#### ServerToolUseBlock

Anthropic 的服务器端工具使用块，用于内部工具如 web_search 的调用。

---

## 4. Google GenerativeAI 响应结构

**返回类型**: [`GenerateContentResponse`](https://github.com/google/generative-ai-python/blob/main/google/generativeai/types/generation_types.py)  
**设计理念**: 多模态优先的 Parts 组合架构，原生支持多种内容类型

### 4.1 顶层响应字段 (8 个字段)

| 字段名                               | 类型                                              | 必需 | 说明                              |
| ------------------------------------ | ------------------------------------------------- | ---- | --------------------------------- |
| `candidates`                         | `Optional[List[Candidate]]`                       | -    | **核心内容** - 模型返回的响应变体 |
| `prompt_feedback`                    | `Optional[GenerateContentResponsePromptFeedback]` | -    | 提示的内容过滤结果                |
| `usage_metadata`                     | `Optional[GenerateContentResponseUsageMetadata]`  | -    | 响应的使用元数据                  |
| `create_time`                        | `Optional[datetime.datetime]`                     | -    | 请求时间戳                        |
| `model_version`                      | `Optional[str]`                                   | -    | 生成响应的模型版本                |
| `response_id`                        | `Optional[str]`                                   | -    | 响应标识符                        |
| `sdk_http_response`                  | `Optional[HttpResponse]`                          | -    | 完整 HTTP 响应                    |
| `automatic_function_calling_history` | `Optional[List[Content]]`                         | -    | 自动函数调用历史                  |
| `parsed`                             | `Optional[Union[pydantic.BaseModel, dict, Enum]]` | -    | 解析后的结构化响应                |

### 4.2 Candidate 候选结构 (7 个字段)

Google 的核心响应容器：

| 字段名                   | 类型                                   | 必需 | 说明                |
| ------------------------ | -------------------------------------- | ---- | ------------------- |
| `content`                | `Optional[Content]`                    | -    | **候选内容部分**    |
| `finish_reason`          | `Optional[FinishReason]`               | -    | 模型停止生成的原因  |
| `safety_ratings`         | `Optional[List[SafetyRating]]`         | -    | 候选的安全评级列表  |
| `citation_metadata`      | `Optional[CitationMetadata]`           | -    | **引用元数据**      |
| `token_count`            | `Optional[int]`                        | -    | 候选中的 token 数量 |
| `grounding_attributions` | `Optional[List[GroundingAttribution]]` | -    | **基础归因信息**    |
| `index`                  | `Optional[int]`                        | -    | 候选在列表中的索引  |

### 4.3 Content 内容结构

Google 的多模态内容组织方式：

#### Content 字段

- `parts` - **内容部分列表**，可包含文本、图像、函数调用等
- `role` - 内容的角色（"user"或"model"）

#### Parts 类型详细结构

Google 的 Parts 系统支持丰富的内容类型，基于 Part 基类：

#### Part 基础结构

Part 是一个包含多种可选字段的基类，每个 Part 实例只应设置一个字段：

| 字段名                  | 类型                            | 说明                                |
| ----------------------- | ------------------------------- | ----------------------------------- |
| `text`                  | `Optional[str]`                 | **文本内容**                        |
| `inline_data`           | `Optional[Blob]`                | **内联数据** - 如图像的 base64 编码 |
| `file_data`             | `Optional[FileData]`            | **文件数据** - URI 引用的文件       |
| `function_call`         | `Optional[FunctionCall]`        | **函数调用**                        |
| `function_response`     | `Optional[FunctionResponse]`    | **函数响应**                        |
| `executable_code`       | `Optional[ExecutableCode]`      | **可执行代码**                      |
| `code_execution_result` | `Optional[CodeExecutionResult]` | **代码执行结果**                    |
| `media_resolution`      | `Optional[PartMediaResolution]` | **媒体分辨率设置**                  |

#### 关键 Parts 类型

**TextPart** (通过 text 字段):

```python
{
    "text": str  # 纯文本内容
}
```

**FunctionCallPart** (通过 function_call 字段):

```python
{
    "function_call": {
        "name": str,           # 函数名称
        "args": Dict[str, Any] # 函数参数
    }
}
```

**FunctionResponsePart** (通过 function_response 字段):

```python
{
    "function_response": {
        "name": str,                    # 函数名称
        "response": Dict[str, Any]      # 函数响应数据
    }
}
```

**ExecutableCodePart** (通过 executable_code 字段):

```python
{
    "executable_code": {
        "language": str,  # 编程语言
        "code": str       # 代码内容
    }
}
```

**CodeExecutionResultPart** (通过 code_execution_result 字段):

```python
{
    "code_execution_result": {
        "outcome": str,     # 执行结果状态
        "output": str       # 执行输出
    }
}
```

### 4.4 使用统计详细结构 (11 个字段)

Google 提供了最详细的使用统计信息：

| 字段名                           | 类型                                 | 说明                            |
| -------------------------------- | ------------------------------------ | ------------------------------- |
| `prompt_token_count`             | `Optional[int]`                      | 提示中的总 token 数             |
| `candidates_token_count`         | `Optional[int]`                      | 生成候选中的总 token 数         |
| `total_token_count`              | `Optional[int]`                      | 整个请求的总 token 数           |
| `cached_content_token_count`     | `Optional[int]`                      | 缓存内容中的 token 数           |
| `thoughts_token_count`           | `Optional[int]`                      | 模型"思考"输出的 token 数       |
| `tool_use_prompt_token_count`    | `Optional[int]`                      | 工具执行结果中的 token 数       |
| `prompt_tokens_details`          | `Optional[List[ModalityTokenCount]]` | 提示中每种模态的 token 计数     |
| `candidates_tokens_details`      | `Optional[List[ModalityTokenCount]]` | 候选中每种模态的 token 计数     |
| `cache_tokens_details`           | `Optional[List[ModalityTokenCount]]` | 缓存内容中每种模态的 token 计数 |
| `tool_use_prompt_tokens_details` | `Optional[List[ModalityTokenCount]]` | 工具结果中每种模态的 token 计数 |
| `traffic_type`                   | `Optional[TrafficType]`              | 此请求的流量类型                |

---

## 跨 SDK 共性分析

### 核心响应模式的强一致性

所有四个 SDK 都遵循相同的核心响应模式：

#### 1. 文本内容输出 ✅

- **OpenAI Chat**: `ChatCompletionMessage.content` (string)
- **OpenAI Responses**: `ResponseOutputMessage` → `ResponseOutputText.text`
- **Anthropic**: `ContentBlock` → `TextBlock.text`
- **Google**: `Candidate.content.parts` → `TextPart.text`

#### 2. 工具调用支持 ✅

- **OpenAI Chat**: `ChatCompletionMessage.tool_calls[]`
- **OpenAI Responses**: 9 种专门的工具调用类型
- **Anthropic**: `ContentBlock` → `ToolUseBlock`
- **Google**: `Candidate.content.parts` → `FunctionCallPart`

#### 3. 拒绝/安全过滤 ✅

- **OpenAI Chat**: `ChatCompletionMessage.refusal`
- **OpenAI Responses**: `ResponseOutputRefusal`
- **Anthropic**: `stop_reason: "refusal"`
- **Google**: `safety_ratings` + `finish_reason`

#### 4. 引用/注释系统 ✅

- **OpenAI Chat**: `ChatCompletionMessage.annotations`
- **OpenAI Responses**: `ResponseOutputText` 支持多种注释
- **Anthropic**: `TextBlock.citations`
- **Google**: `CitationMetadata` + `GroundingAttribution`

#### 5. 使用统计信息 ✅

- **OpenAI Chat**: `usage` (CompletionUsage)
- **OpenAI Responses**: `usage` (ResponseUsage)
- **Anthropic**: `usage` (Usage)
- **Google**: `usage_metadata` (最详细的统计)

#### 6. 停止原因说明 ✅

- **OpenAI Chat**: `choices[].finish_reason`
- **OpenAI Responses**: `status`
- **Anthropic**: `stop_reason`
- **Google**: `candidates[].finish_reason`

### 统一的结构模式

#### 容器-内容二级结构

所有 SDK 都采用容器-内容的二级结构：

- **OpenAI Chat**: `choices[]` → `message`
- **OpenAI Responses**: `output[]` → 各种 OutputItem
- **Anthropic**: `content[]` → 各种 ContentBlock
- **Google**: `candidates[]` → `content.parts[]`

#### 类型化内容设计

所有 SDK 都使用联合类型或多态设计：

- **OpenAI Chat**: 通过字段区分（content, tool_calls, refusal 等）
- **OpenAI Responses**: 18 种不同的 OutputItem 类型
- **Anthropic**: 6 种 ContentBlock 类型
- **Google**: 7+种 Parts 类型

### 设计哲学对比

| SDK                  | 设计复杂度 | 功能丰富度 | 多模态支持 | 工具生态   | 设计理念   |
| -------------------- | ---------- | ---------- | ---------- | ---------- | ---------- |
| **OpenAI Chat**      | ⭐⭐       | ⭐⭐⭐     | ⭐⭐       | ⭐⭐⭐     | 简洁实用   |
| **OpenAI Responses** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐     | ⭐⭐⭐⭐⭐ | 极度模块化 |
| **Anthropic**        | ⭐⭐⭐     | ⭐⭐⭐⭐   | ⭐⭐       | ⭐⭐⭐     | 优雅简洁   |
| **Google**           | ⭐⭐⭐⭐   | ⭐⭐⭐⭐   | ⭐⭐⭐⭐⭐ | ⭐⭐⭐     | 多模态优先 |

## 关键发现与 Body 级别转换可行性

### 1. 高度的功能一致性

四个 SDK 在核心功能上表现出惊人的一致性：

- ✅ 所有 SDK 都支持文本、工具调用、拒绝、引用、使用统计、停止原因
- ✅ 所有 SDK 都采用容器-内容的二级结构
- ✅ 所有 SDK 都使用类型化的内容设计

### 2. Body 级别转换完全可行

由于强一致性，响应的 body 级别转换是完全可行的：

#### 核心映射策略

1. **文本内容** → 直接映射各 SDK 的文本字段
2. **工具调用** → 标准化工具调用格式(id, name, input)
3. **状态信息** → 统一停止原因和完成状态
4. **元数据** → 标准化使用统计和时间戳
5. **引用系统** → 统一引用和注释格式

#### 转换复杂度评估

- **Anthropic ↔ 其他**: ⭐ 简单（直接映射）
- **OpenAI Chat ↔ 其他**: ⭐⭐ 中等（需要结构调整）
- **OpenAI Responses ↔ 其他**: ⭐⭐⭐ 复杂（需要类型合并/拆分）
- **Google ↔ 其他**: ⭐⭐ 中等（需要 Parts 转换）

### 3. 统一 IRResponse 设计指导原则

基于这个分析，统一的响应 IR 应该：

1. **包含所有四个 SDK 的共同功能** - 文本、工具、拒绝、引用、统计
2. **采用容器-内容二级结构** - 便于扩展和类型安全
3. **支持渐进式功能映射** - 从基础功能到高级功能
4. **保持向后兼容性** - 支持各 SDK 的特有功能

### 4. 实现优先级建议

**Phase 1: 核心功能**

- 文本内容转换
- 基础工具调用转换
- 使用统计映射
- 停止原因标准化

**Phase 2: 高级功能**

- 引用和注释系统
- 多模态内容支持
- 复杂工具类型转换
- 流式响应支持

## 关键类型兼容性分析

### 1. 工具调用类型的统一映射

基于详细的结构分析，我们发现工具调用在四个 SDK 中有清晰的映射关系：

#### 核心工具调用字段映射

| 功能         | OpenAI Chat                       | OpenAI Responses          | Anthropic                   | Google                      |
| ------------ | --------------------------------- | ------------------------- | --------------------------- | --------------------------- |
| **工具 ID**  | `tool_calls[].id`                 | `call_id`                 | `ToolUseBlock.id`           | 通过 name 关联              |
| **工具名称** | `tool_calls[].function.name`      | `name`                    | `ToolUseBlock.name`         | `function_call.name`        |
| **工具参数** | `tool_calls[].function.arguments` | `arguments` (JSON string) | `ToolUseBlock.input` (Dict) | `function_call.args` (Dict) |
| **执行状态** | -                                 | `status`                  | -                           | -                           |

#### 高级工具类型兼容性

- **MCP 工具**: OpenAI Responses 原生支持，其他 SDK 可通过自定义工具兼容
- **服务器端工具**: Anthropic 的 WebSearch，OpenAI 的 WebSearch/CodeInterpreter，Google 的 CodeExecution
- **自定义工具**: OpenAI Responses 的 CustomToolCall，可映射到其他 SDK 的通用工具调用

### 2. 推理内容(Reasoning)的跨 SDK 支持

#### 推理内容映射策略

| SDK                  | 推理支持  | 实现方式                        | 兼容性     |
| -------------------- | --------- | ------------------------------- | ---------- |
| **OpenAI Responses** | ✅ 原生   | `ResponseReasoningItem`         | 完整支持   |
| **Anthropic**        | ✅ 原生   | `ThinkingBlock`                 | 完整支持   |
| **Google**           | ✅ 原生   | `ExecutableCodePart` (思考过程) | 部分支持   |
| **OpenAI Chat**      | ⚠️ 第三方 | `reasoning_content` 字段扩展    | 可扩展支持 |

### 3. 服务器端工具的统一抽象

#### CustomServerSideToolResponse 设计建议

基于各 SDK 的服务器端工具分析，我们可以设计统一的服务器端工具响应：

```python
class CustomServerSideToolResponse:
    type: Literal["server_side_tool"]
    tool_type: Literal["web_search", "code_execution", "file_search", "image_generation"]
    tool_id: str
    tool_name: str
    status: Literal["in_progress", "completed", "failed", "searching", "interpreting"]

    # 通用输出字段
    output: Optional[str]
    error: Optional[str]

    # 特定工具类型的详细信息
    web_search_details: Optional[WebSearchDetails]
    code_execution_details: Optional[CodeExecutionDetails]
    file_search_details: Optional[FileSearchDetails]
    image_generation_details: Optional[ImageGenerationDetails]
```

#### 跨 SDK 服务器端工具映射

| 工具类型     | OpenAI Responses                  | Anthropic                  | Google                | 统一映射                   |
| ------------ | --------------------------------- | -------------------------- | --------------------- | -------------------------- |
| **网络搜索** | `ResponseFunctionWebSearch`       | `WebSearchToolResultBlock` | -                     | `web_search_details`       |
| **代码执行** | `ResponseCodeInterpreterToolCall` | -                          | `CodeExecutionResult` | `code_execution_details`   |
| **文件搜索** | `ResponseFileSearchToolCall`      | -                          | `FileDataPart`        | `file_search_details`      |
| **图像生成** | `ImageGenerationCall`             | -                          | -                     | `image_generation_details` |

### 4. 实现优先级与兼容性策略

#### Phase 1: 核心兼容 (必须实现)

1. **基础工具调用转换** - 支持 function_call 在四个 SDK 间的转换
2. **文本内容转换** - 统一文本响应格式
3. **状态信息映射** - finish_reason, stop_reason 等的标准化

#### Phase 2: 高级功能兼容 (应该实现)

1. **推理内容支持** - ResponseReasoningItem ↔ ThinkingBlock 转换
2. **MCP 工具兼容** - OpenAI MCP ↔ 其他 SDK 自定义工具
3. **引用系统统一** - annotations, citations, grounding_attributions

#### Phase 3: 服务器端工具兼容 (可以实现)

1. **CustomServerSideToolResponse** - 统一的服务器端工具抽象
2. **跨 SDK 服务器工具映射** - web_search, code_execution 等
3. **工具状态同步** - 统一的工具执行状态管理

**结论**: 通过详细的结构分析，我们确认了四个 SDK 在核心功能上的高度一致性，并识别了关键的兼容点。ResponseFunctionToolCall、ResponseCustomToolCall、McpCall、ResponseReasoningItem 等关键类型为我们提供了清晰的转换路径。基于这些分析，我们可以设计一个功能完整的 IRResponse，支持从基础工具调用到高级推理内容的全面转换，为 body 级别的响应转换提供了坚实的技术基础。
