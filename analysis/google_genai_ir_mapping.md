# IR Types to Google GenAI SDK Mapping

本文档详细描述了 LLMIR IR (Intermediate Representation) 类型与 `google-genai` SDK 原生类型之间的映射关系。

## 1. Request Mapping (请求映射)

IR 请求主要映射到 `google.genai.types.GenerateContentConfig` 和 `google.genai.types.Content`。

### 1.1 IRRequest 顶层映射

| IRRequest 字段 | Google GenAI SDK 类型/字段 | 说明 |
| :--- | :--- | :--- |
| `model` | `model` (str) | 模型 ID |
| `messages` | `contents` (List[Content]) | 对话内容列表 |
| `system_instruction` | `config.system_instruction` (ContentUnion) | 系统指令 |
| `tools` | `config.tools` (List[Tool]) | 工具定义列表 |
| `tool_choice` | `config.tool_config.function_calling_config.mode` | 工具选择模式映射 |
| `generation` | `config` (GenerateContentConfig) | 生成配置参数 |
| `response_format` | `config.response_mime_type` / `response_schema` | 响应格式配置 |

### 1.2 GenerationConfig 映射

| IR GenerationConfig 字段 | Google GenerateContentConfig 字段 | 说明 |
| :--- | :--- | :--- |
| `temperature` | `temperature` | 温度控制 |
| `top_p` | `top_p` | Nucleus 采样 |
| `top_k` | `top_k` | Top-k 采样 |
| `max_tokens` | `max_output_tokens` | 最大生成 token 数 |
| `stop_sequences` | `stop_sequences` | 停止序列 |
| `frequency_penalty` | `frequency_penalty` | 频率惩罚 |
| `presence_penalty` | `presence_penalty` | 存在惩罚 |
| `seed` | `seed` | 随机种子 |
| `candidate_count` | `candidate_count` | 候选数量 |

### 1.3 Message & ContentPart 映射

IR 的 `Message` 映射到 Google 的 `Content` 类型。

| IR Message/Part | Google SDK 类型 | 映射细节 |
| :--- | :--- | :--- |
| `Message.role` | `Content.role` | `user` -> `user`, `assistant` -> `model` |
| `TextPart` | `Part.text` | 纯文本内容 |
| `ImagePart` | `Part.inline_data` (Blob) | 图像数据 (base64) |
| `FilePart` | `Part.file_data` (FileData) | 文件数据 (URI) |
| `ToolCallPart` | `Part.function_call` (FunctionCall) | 工具调用请求 |
| `ToolResultPart` | `Part.function_response` (FunctionResponse) | 工具执行结果 |

---

## 2. Response Mapping (响应映射)

IR 响应主要从 `google.genai.types.GenerateContentResponse` 映射而来。

### 2.1 IRResponse 顶层映射

| IRResponse 字段 | Google GenerateContentResponse 字段 | 说明 |
| :--- | :--- | :--- |
| `id` | `response_id` | 响应唯一 ID |
| `model` | `model_version` | 实际使用的模型版本 |
| `choices` | `candidates` (List[Candidate]) | 生成的候选结果列表 |
| `usage` | `usage_metadata` | Token 使用统计 |

### 2.2 ChoiceInfo & FinishReason 映射

| IR ChoiceInfo 字段 | Google Candidate 字段 | 说明 |
| :--- | :--- | :--- |
| `index` | `index` | 候选索引 |
| `message` | `content` (Content) | 生成的消息内容 |
| `finish_reason` | `finish_reason` | 停止原因映射 |

**FinishReason 映射细节:**
- `STOP` -> `stop`
- `MAX_TOKENS` -> `length`
- `SAFETY` -> `content_filter`
- `MALFORMED_FUNCTION_CALL` -> `error`

### 2.3 UsageInfo 映射

| IR UsageInfo 字段 | Google UsageMetadata 字段 | 说明 |
| :--- | :--- | :--- |
| `prompt_tokens` | `prompt_token_count` | 输入 Token 数 |
| `completion_tokens` | `candidates_token_count` | 输出 Token 数 |
| `total_tokens` | `total_token_count` | 总 Token 数 |
| `reasoning_tokens` | `thoughts_token_count` | 推理/思考消耗的 Token |

---

## 3. 特殊类型映射 (Special Types)

### 3.1 Reasoning (思考过程)
- **IR**: `ReasoningPart` (type="reasoning")
- **Google SDK**: `Part.thought` (bool) 和 `Part.text`。当 `thought=True` 时，该 Part 的 `text` 内容即为思考过程。

### 3.2 Tool Call Metadata
- **IR**: `ToolCallPart.provider_metadata`
- **Google SDK**: `Part.thought_signature`。在 Gemini 3 中，工具调用通常伴随一个 `thought_signature`，IR 将其存储在 `provider_metadata` 中以便后续轮次带回。

### 3.3 Response Format
- **IR**: `ResponseFormatConfig`
- **Google SDK**: 映射到 `GenerateContentConfig` 的 `response_mime_type` (如 `application/json`) 和 `response_schema` (Schema 对象)。