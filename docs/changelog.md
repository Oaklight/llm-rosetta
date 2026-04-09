---
title: 更新日志
---

# 更新日志

LLM-Rosetta 的所有重要变更均记录于此。本项目遵循 [Keep a Changelog](https://keepachangelog.com/) 规范。

## v0.4.0 — 2026-04-09

### 新增

- **元数据保留：实现无损 A→IR→A 往返转换** (#60, PR #119)：`ConversionContext` 新增 `MetadataMode`（`"strip"` / `"preserve"`）选项，在 `from_provider` 阶段捕获提供商特有字段，在 `to_provider` 阶段重新注入，实现无损往返转换。`ConversionContext` 新增辅助方法：`store_request_echo()`、`store_response_extras()`、`store_output_items_meta()`、`get_echo_fields()`、`get_output_items_meta()`。各提供商覆盖范围：
    - **OpenAI Responses**：捕获/恢复 28+ 回显字段（temperature、tools、reasoning、truncation 等）、逐输出项元数据（id、status、annotations、logprobs），`RESPONSES_REQUIRED_DEFAULTS` 字典提供规范要求字段的合理默认值，所有 SSE 事件包含 `sequence_number`
    - **Anthropic**：保留 `stop_sequence`、`container`、citations 及 OpenRouter 扩展 usage 字段
    - **OpenAI Chat**：`response_to_provider` 现在重新输出 `refusal` 和 `annotations` 字段（此前被丢弃）
    - **Google GenAI**：保留 usage 元数据中的 `promptTokensDetails` 和 `cachedContentTokenCount`
    - **网关**：流式和非流式路径自动启用 preserve 模式；流式传输中在 `from_ctx` 和 `to_ctx` 之间桥接元数据

### 修复

- **Open Responses 规范合规性（流式与非流式）**：所有 SSE 事件添加必需字段（`item_id`、`logprobs`、`annotations`、`status`、`sequence_number`、`output_index`、`content_index`），添加 usage 详细分解（`output_tokens_details`、`input_tokens_details`），非流式输出项生成消息 item ID 和 status，tool_ops 添加 `function_call` status 字段，`service_tier` 默认值改为 `"default"`（字符串而非 null，符合规范），required defaults 中添加 `completed_at`，未提供时 `created_at` 回退到当前时间，规范化回显工具的 `strict: null`，网关流式传输中从 `from_ctx` 到 `to_ctx` 桥接元数据。全部 6 个 Open Responses 合规性测试通过（schema + 语义）

## v0.3.1 — 2026-04-07

### 修复

- **`service_tier: None` 和 `system_fingerprint: None` 导致验证错误** (PR #118)：OpenAI 上游返回这些字段为 `null`，但存在性检查（`if "key" in dict`）通过后将 `None` 赋值给 IR 的 `NotRequired[str]` 字段。在 OpenAI Chat 和 OpenAI Responses 两个转换器中改为值非空检查。发现于 [Oaklight/argo-proxy#99](https://github.com/Oaklight/argo-proxy/issues/99) 测试过程中
- **基类 `StreamContext` 在 Responses 流式传输中缺少提供商特定属性** (PR #118)：当网关传入基类 `StreamContext` 给 `OpenAIResponsesConverter.stream_response_to_provider()` 时，方法访问的 `accumulated_text`、`output_item_emitted` 等字段仅存在于 `OpenAIResponsesStreamContext` 子类。新增 `from_base()` 类方法自动升级，并通过 metadata 缓存保持跨调用状态

## v0.3.0 — 2026-04-07

### 新增

- **全部 4 个转换器支持多模态工具结果** (#92, PR #109)：工具现在可以返回多模态内容（文本 + 图片 + 文件）作为 `ToolResultPart.result`。三个提供商（Anthropic、OpenAI Responses、Google GenAI）原生支持；内容块通过每个提供商的 `content_ops` 层转换。详见下方提供商支持矩阵
- **OpenAI Chat 多模态工具结果无损往返** (#92, PR #108)：OpenAI Chat Completions 的工具消息仅接受 `content: string`。实现双重编码策略——工具消息保留 `json.dumps(result)` 作为数据兜底，同时合成用户消息携带可视内容（`image_url` 部分）包裹在 `<tool-content call-id="...">` XML 标签中。解包时优先从合成消息恢复多模态结构，若合成消息被智能体框架裁剪则回退到 JSON 解析
- **`extract_all_text()` 辅助函数** (PR #109)：从 `TextPart` 和 `ReasoningPart` 内容中提取文本——适用于思考模型（如 gemini-2.5-flash），这类模型可能将答案放在 reasoning 部分而非 text 部分
- **`generate_chart` 示例工具** (PR #109)：`examples/tools.py` 中新增多模态工具，返回 `[TextPart, ImagePart]`（含内联 base64 PNG），以及 `multimodal_tools_spec` 组合全部 3 个示例工具
- **全部 4 个提供商 SDK 的多模态集成测试** (PR #109)：每个提供商新增两个测试场景——(A) 工具返回多模态内容（文本 + 图片），(B) 图片输入结合工具调用。全部 30 个测试通过官方 API 验证：OpenAI Chat 9/9、OpenAI Responses 6/6、Anthropic 8/8、Google GenAI 7/7
- **运行时 IR 验证（零依赖内嵌验证器）** (#91)：`validate_ir_request()`、`validate_ir_response()` 和 `validate_ir_messages()` 工具函数，在运行时对 IR 结构进行 TypedDict 定义验证。4 个转换器的 `request_from_provider()` 和 `response_from_provider()` 现自动验证输出。替代手动 `BaseMessageOps.validate_messages` 实现。包含 Python <3.11 的 `typing_extensions.TypedDict` 兼容修复
- **常量验证测试**：4 个 `test_constants.py` 文件中新增 39 个测试，验证所有 reason 映射值均为合法 IR finish reason、映射覆盖完整、事件类型常量格式正确、ID 生成产出正确格式
- **Finish reason 映射测试覆盖**：38 个测试验证 reason 映射正确性，为常量重构提供安全网
- **转换管线 `ConversionContext` 基类** (#106, PR #111)：新增 `ConversionContext` 数据类，包含 `warnings: list[str]`、`options: dict[str, Any]` 和 `metadata: dict[str, Any]`——为非流式转换提供结构化上下文容器。新增 `BaseConverter.create_conversion_context(**options)` 工厂方法，与已有的 `create_stream_context()` 对称。全部 6 个 `BaseConverter` 非流式方法现在接受可选的 `context: ConversionContext` 关键字参数；各转换器实现将警告同步到 `context.warnings`。网关代理为每个请求创建共享 context 并沿完整的 source→IR→target→response 管线传递

### 修复

- **工具转换失败时提供上下文错误信息** (#85, PR #110)：当 `p_tool_definition_to_ir()` 在处理格式错误或不支持的工具定义时失败，`ValueError` 现在包含 `type=` 和 `name=` 上下文信息，帮助用户识别是哪个工具导致了问题。已应用于全部 4 个转换器（OpenAI Chat、OpenAI Responses、Anthropic、Google GenAI），附带单元测试
- **OpenAI Responses `tool_choice` 格式** (PR #109)：此前使用 Chat Completions 格式（`{"type": "function", "function": {"name": "..."}}`），现在使用 Responses 格式（`{"type": "function", "name": "..."}`）
- **OpenAI Responses 工具调用 ID 往返** (PR #109)：Responses API 使用 `fc_` 前缀 ID，IR 使用 `call_` 前缀。Responses 的 `id` 现在单独保存在 `provider_metadata` 中（与 `call_id` 分开），实现无损往返转换
- **OpenAI Responses 推理项往返** (PR #109)：推理模型（如 gpt-5-nano）发出带有 `id`（rs_ 前缀）、结构化 `summary` 数组和 `encrypted_content` 的推理项。这些信息现通过 `provider_metadata` 保留以实现无损往返——修复了推理项缺少原始 `id` 回传时导致的 400 错误
- **IR 验证接受可选响应字段的 `None` 值** (PR #109)：`IRResponse` 中的 `logprobs` 和 `system_fingerprint` 现在接受 `None` 值（此前仅接受缺失键）
- **OpenAI Responses `content_filter` finish reason 映射到错误状态** (#90)：`content_filter` 此前被错误映射到 `"completed"` 状态（`response_to_provider` 和 `stream_response_to_provider`）。现正确映射到 `"incomplete"` 状态，附带 `incomplete_details.reason = "content_filter"`
- **Anthropic 流式传输缺少 `refusal` reason 映射**：流式传输的 `reason_map` 缺少非流式路径中存在的 `refusal` 条目，导致 Anthropic refusal 停止原因在流式传输期间被静默丢弃。作为常量提取的副作用修复（#64）——两条路径现在共享同一个 `ANTHROPIC_REASON_FROM_PROVIDER` 字典

### 变更

- **`ReasoningConfig.effort` 扩展为 5 级枚举** (#100)：Effort 级别新增 `"minimal"`、`"low"`、`"medium"`、`"high"`、`"max"`。提供商映射：Anthropic 映射到 `thinking.type="adaptive"` 配合 `thinking.effort`；OpenAI Chat/Responses 将 `"minimal"` 钳位为 `"low"`、`"max"` 钳位为 `"high"`（附带警告）；Google GenAI 映射到 `thinking_config.thinking_level`
- **`ReasoningConfig.type` 替换为 `ReasoningConfig.enabled`** (#70)：`type: Literal["enabled", "disabled"]` 字段替换为 `enabled: bool`，避免遮蔽 Python 内建 `type`，提供更自然的 API
- **合并重复的 IR 概念** (#69)：移除 `GenerationConfig` 中的 `candidate_count`——改用 `n`（Google GenAI 转换器内部映射 `n` ↔ `candidate_count`）。`system_instruction` 类型从 `str | list[dict]` 统一为 `str`
- **规范化 `ImagePart`、`FilePart`、`AudioPart` 为标准形式** (#68)：每种 Part 现在恰好有两种标准形式——URL 引用 + 结构化内联数据（如 `image_data`）——加上统一的 `provider_ref: dict[str, Any]` 用于提供商特定引用。移除冗余的顶层 `data`/`media_type` 字段，`file_id`/`audio_id` 替换为 `provider_ref`
- **IR 类型字段从 `Iterable` 改为 `list`；函数参数改为 `Sequence`** (#67)：TypedDict 字段使用 `list` 以支持索引和序列化；函数参数使用 `Sequence`（协变、只读）。同时修复 `strip_orphaned_tool_config` 中 `any()` 消耗单次迭代器的潜在 bug
- **`StreamContext` 继承自 `ConversionContext`** (#106, PR #111)：`StreamContext` 现为 `ConversionContext` 的子类（IS-A 关系），统一流式与非流式路径的上下文模型。文件重命名：`base/stream_context.py` → `base/context.py`
- **`StreamContext` 转为 dataclass 并引入提供商子类** (#65)：`StreamContext` 现为 `@dataclass`（消除防御性 `getattr`/`hasattr` 模式）。OpenAI Responses 特有状态提取至 `OpenAIResponsesStreamContext` 子类。新增 `BaseConverter.create_stream_context()` 工厂方法

### 重构

- **Warnings 单源收敛** (#113, PR #115)：4 个转换器的 `request_to_provider` 方法现在统一使用 `ConversionContext` 作为警告的唯一积累点。消除了之前警告同时写入本地列表和 `context.warnings` 的双写模式。返回的 warnings 列表与 `context.warnings` 是同一个对象——不可能产生重复
- **`ProviderMetadataStore` 替代全局 metadata 缓存** (#112, PR #117)：`proxy.py` 中的模块级 `_provider_metadata_cache` 字典替换为 `ProviderMetadataStore` 类——提供 TTL 过期（30 分钟）、最大容量淘汰（10k 条目）和显式生命周期管理。Store 在 `create_app()` 中按应用创建并通过 `app.state` 传递，消除隐式全局状态变更。`close_clients()` 重命名为 `close_resources()` 以在关闭时同步清理 store
- **缩减公共 API 导出面** (#114, PR #116)：各转换器包的 `__all__` 导出精简为仅包含主转换器类，移除内部实现细节（`*MessageOps`、`*ContentOps`、`*ConfigOps`、`*ToolOps`、`*Constants`）。内部模块仍可通过显式导入使用，但不再作为公共 API 面推广
- **将单体流式方法拆分为事件处理器** (#63)：4 个转换器中 8 个单体 `if`/`elif` 流式方法（约 1,781 行）替换为通过类级处理器表分发的独立处理器方法。公共 API 不变
- **提取 OpenAI Responses 转换器共享工具函数** (#66)：`resolve_call_id()` 和 `build_message_preamble_events()` 从 `converter.py` 提取至 `utils.py`，附带专用单元测试
- **提取各提供商常量用于 reason 映射及魔法值** (#64)：4 个转换器中散布的内联 reason 映射字典、SSE 事件类型字符串字面量、status-to-reason 条件逻辑和 ID 生成模式，现已集中到各提供商的 `_constants.py` 模块中。包含 `AnthropicEventType` 和 `ResponsesEventType` 常量类、`REASON_FROM_PROVIDER` / `REASON_TO_PROVIDER` 字典，以及 `generate_tool_call_id()` / `generate_message_id()` 辅助函数

## v0.2.6 — 2026-03-29

### 修复

- **Responses API 转换后 Chat Completions tool 消息顺序错乱** *([@caidao22](https://github.com/caidao22))*：Codex CLI 在 Responses API 格式中会在 `function_call_output` 与其他项目（如用户警告消息）之间交错排列——在 Responses API 中通过 `call_id` 匹配是合法的。但经 IR → Chat Completions 转换后，交错的消息打破了 OpenAI Chat API 约束（`role: "tool"` 消息必须紧跟其 `assistant` `tool_calls`），导致上游返回 400 错误。在 `OpenAIChatMessageOps.ir_messages_to_p()` 中新增 `_reorder_tool_messages()` 后处理步骤，将 tool 响应重新归组到对应的 assistant 消息之后
- **无工具定义时剥离孤立的 `tool_choice`/`tool_config`** *([@caidao22](https://github.com/caidao22))*：Codex 上下文压缩可能移除所有工具定义但保留 `tool_choice`（如 `"auto"`），导致上游 API 拒绝请求（*"tool_choice is set but no tools are provided"*）。在四个转换器中新增 `strip_orphaned_tool_config()`——属于 Codex 压缩修复家族：`fix_orphaned_tool_calls_ir`（孤立 tool_call/result 配对）、`_reorder_tool_messages`（tool 消息排序）。同时将 `fix_orphaned_tool_calls_ir` 扩展到 Google GenAI 转换器以保持完整性（#87）
- **流式事件顺序修正**：四个提供商转换器（OpenAI Chat、OpenAI Responses、Anthropic、Google GenAI）中 `UsageEvent` 现在在 `FinishEvent` 之前发出。此前 `FinishEvent` 先处理，导致 `response.completed` 携带 `output_tokens=0`——下游消费者（如 Codex token 追踪）看到的是过时的用量数据。对于跨 chunk 场景（OpenAI Chat 在不同 chunk 中发送 `finish_reason` 和 `usage`），`FinishEvent` 现在将 `response.completed` 延迟到 `StreamEndEvent` 中发出，后者会合并待处理的 usage 数据
- **Anthropic/Google → Chat 流式传输中并行工具调用被合并**：Anthropic 和 Google GenAI 的 `stream_response_from_provider` 发出的 `ToolCallStartEvent` 和 `ToolCallDeltaEvent` 缺少 `tool_call_index`。路由到 Chat Completions 时，所有并行工具调用默认索引为 0，导致客户端 SDK 将它们合并为一个调用。Anthropic 现在从 `context._tool_call_order` 位置派生 `tool_call_index`；Google 从 context 中的注册顺序计算（#88, #89）
- **Responses `function_call` 输出缺少 `id` 字段**：非流式 `response_to_provider` 在 `function_call` 输出项上缺少 `id` 字段。流式传输使用合成的 `fc_` 前缀，可能通过 `p_tool_call_to_ir` 回退路径泄漏到 IR。统一两条路径，直接使用 `call_id` 作为 `id`（无前缀）
- **Responses 流式传输 `item_id` 及空 `tool_call_id` 解析** *([@caidao22](https://github.com/caidao22))*：`StreamContext` 新增 `item_id` 追踪（`tool_call_item_id_map`，双向映射）。Responses `stream_response_to_provider` 现在在 `output_item.added` 上发出 `item.id`，在 `function_call_arguments.delta/done` 事件上发出 `item_id`（非 `call_id`）。纵深防御：通过 context 的 `tool_call_index` 解析空 `tool_call_id`（#86）
- **非 function 类型工具名被添加类型前缀** *([@caidao22](https://github.com/caidao22))*：非 function 的 IR 工具定义（如 `type="custom"`、`name="apply_patch"`）转换时被添加了类型前缀（`custom_apply_patch`），导致工具调用匹配失败（客户端期望原始名称）。OpenAI Chat 和 Responses 转换器现在直接使用 `ir_tool["name"]`（#84）

## v0.2.5 — 2026-03-23

### 修复

- **Anthropic `input_schema` 无参数工具缺少 `type` 字段**：无参数的 MCP 工具生成 `input_schema: {}`，但 Anthropic 要求必须包含 `"type"` 字段。现在当 schema 字典缺少 `type` 字段时默认为 `{"type": "object"}`——修复了 Google GenAI 或 OpenAI Responses 工具调用路由到 Anthropic 上游时出现的 `tools.0.custom.input_schema.type: Field required` 错误
- **Google GenAI 全栈 camelCase 字段处理**：Gemini CLI 和 Google REST API 使用 camelCase（`inlineData`、`fileData`、`mimeType`、`fileUri`、`functionCall`、`functionResponse`、`finishReason`、`usageMetadata`、`responseMimeType`、`responseSchema`、`thinkingConfig`、`maxOutputTokens`、`stopSequences` 等），但转换器此前仅接受 snake_case。content_ops、config_ops、tool_ops、message_ops 和 converter 中所有 P→IR 方法现在同时接受两种命名；所有 IR→P 方法统一输出 camelCase 以兼容 REST API
- **Google→IR 转换丢失图片/音频/文件数据**：`p_part_to_ir` 只检查 `inline_data`（snake_case），但 Gemini CLI 发送 `inlineData`（camelCase）——二进制内容被静默丢弃并输出 `不支持的Part类型` 警告。修复方式：在分发入口处规范化 camelCase 键名
- **跨格式图片转换失败（Google → OpenAI/Anthropic）**：Google 的 `p_image_to_ir` 生成的 `ImagePart` 使用顶层 `data` + `media_type` 字段，但 OpenAI Chat、Anthropic 和 OpenAI Responses 的 `ir_image_to_p` 仅检查 `image_url` 和嵌套的 `image_data`——导致 `ValueError`。三个目标转换器现在都增加了对顶层字段的兜底处理（#68）
- **Google GenAI tool_call_id 对账**：Google `functionCall` 没有 ID 字段，P→IR 时生成 UUID。但 Gemini CLI 为 `functionResponse` 分配自有 ID（格式：`name_timestamp_index`），造成不匹配。新增 `_reconcile_tool_call_ids` 方法，按函数名称匹配工具结果与工具调用，修复孤立 tool_call 错误
- **tool_call_id 超出 OpenAI 40 字符限制**：生成的 ID 使用 `call_{name}_{8hex}` 格式——MCP 工具名如 `mcp_toolregistry-hub-server_datetime-now` 产生 54 字符 ID。缩短为 `call_{24hex}`（固定 29 字符）
- **Google→IR 工具结果的 role 映射**：`functionResponse` 部分生成 `role: "user"` 的 IR 消息，导致 `fix_orphaned_tool_calls_ir`（检查 `role: "tool"`）无法检测。现在将 `functionResponse` 分离为 `role: "tool"` 消息，并在 `_IR_TO_GOOGLE_ROLE` 中添加显式 `"tool": "user"` 映射
- **混合内容消息排序**：当 Google 消息同时包含 `functionResponse` 和 `inlineData` 时，内容部分排在工具结果之前，打断了 OpenAI 要求的 `assistant(tool_calls) → tool(response)` 顺序。修复后工具结果排在内容部分之前
- **Google 内建工具（googleSearch、codeExecution）**：`p_tool_definition_to_ir` 对没有 `name` 字段的工具条目返回 `None`；converter 跳过这些条目，不再产生空 `function.name` 错误
- **网关：Starlette `on_shutdown` 弃用兼容**：将已弃用的 `on_shutdown` 参数替换为 `lifespan` 异步上下文管理器——修复与 Starlette 0.38+（移除了 `on_shutdown`/`on_startup`）的兼容性

### 新增

- **StreamContext**：`get_tool_call_args()` 和 `get_pending_tool_calls()` 方法，用于在流式处理期间查询已积累的工具调用状态

### 变更

- **`BaseToolOps.p_tool_definition_to_ir` 返回类型**：改为 `ToolDefinition | list[ToolDefinition] | None`，支持不可转换的工具条目

### 新增（文档）

- **提供商与 CLI 兼容性矩阵**：新增指南页面，记录通过格式转换代理实际集成测试 Gemini CLI、Claude Code 和 OpenCode 时发现的真实问题

## v0.2.4 — 2026-03-22

### 新增

- **`fix_orphaned_tool_calls()` 工具函数**：`converters/openai_chat/tool_ops.py`、`converters/openai_responses/tool_ops.py` 和 `converters/anthropic/tool_ops.py` 中的公开函数，**双向**检测和修复工具调用/结果的配对问题 — 为孤立的工具调用注入合成占位结果，**同时**移除没有匹配调用的孤立工具结果。OpenAI（Chat 和 Responses）及 Anthropic 严格要求此配对关系（否则返回 400 错误），仅 Google Gemini 对此宽松。在所有严格配对转换器的 `request_to_provider()` 中通过 IR 层级自动修复；检测到孤立工具调用或结果时输出 `WARNING` 级别日志（#82, #84）

### 修复

- **Anthropic→IR `tool_result` 消息的 role 规范化**：Anthropic 将 `tool_result` 块放在 `role: "user"` 消息中，但 IR 使用 `role: "tool"`（与 OpenAI 一致）。Anthropic 转换器现在将纯 `tool_result` 的 user 消息规范化为 `role: "tool"`，并将混合 `tool_result` + text 的消息拆分为独立的 `role: "tool"` 和 `role: "user"` IR 消息。修复了跨格式转换（如 Anthropic → OpenAI Chat）中 `fix_orphaned_tool_calls_ir()` 无法检测已回答工具调用的问题（#84）
- **OpenAI Responses→IR `function_call_output` 的 role 规范化**：`function_call_output` 和 `mcp_call_output` 项此前被归入 `role: "user"` 的 IR 消息，但 IR 对工具结果使用 `role: "tool"`。Responses 转换器现在将这些项归入 `role: "tool"` 消息，修复了跨格式转换（如 Responses → OpenAI Chat）中 `fix_orphaned_tool_calls_ir()` 无法检测已回答工具调用的问题（#84）

### 新增（文档）

- **提供商方言差异指南**：在转换器指南中新增章节（中英文），记录工具 schema 清理、孤立工具调用处理、Google camelCase/snake_case 差异

## v0.2.3 — 2026-03-22

### 修复

- **所有转换器均执行工具 schema 清洗**：此前 `_sanitize_schema()` 仅在 OpenAI Chat 转换器中调用。Google GenAI、OpenAI Responses 和 Anthropic 转换器现在也在发送到上游前清洗工具参数 schema，防止 Vertex AI 等严格端点拒绝请求（#80）
- **移除非标准 `ref` 和 `$schema` 关键字**：OpenCode 内置工具使用不带 `$` 前缀的裸 `ref` 字段和顶层 `$schema`，均被 Vertex AI 拒绝。已添加到不支持关键字黑名单（#80）
- **通过内联解析 `$ref`/`$defs` 引用**：JSON Schema `$ref` 引用现在通过从 `$defs`/`definitions` 内联被引用的定义来解析，两个关键字均从输出中移除。支持嵌套和链式引用（#80）
- **流式传输中工具调用参数未累积**：OpenAI Chat、Anthropic 和 Google GenAI 转换器在 `StreamContext` 中注册了工具调用，但在流式传输期间从未调用 `append_tool_call_args()` 累积参数增量。这导致工具调用参数到达上游时为空（如 MCP 工具返回 `'query' is a required property`）。此前仅 OpenAI Responses 转换器正确处理（#81）
- **OpenAI Chat 流式工具调用 ID 解析**：仅携带 `index` 而无 `id` 的增量 chunk 产生了空字符串 `tool_call_id`。现在通过 chunk 索引从 `StreamContext._tool_call_order` 解析有效 ID（#81）

### 变更

- **`sanitize_schema` 提取至 `converters/base/tools.py`**：Schema 清洗工具函数（此前为 `openai_chat/tool_ops.py` 中的私有函数 `_sanitize_schema`）现已提升为 `converters/base/tools.py` 中的公开共享函数，通过 `converters.base` 导出。所有 4 个转换器的 `tool_ops.py` 均从共享位置导入，消除了跨转换器的交叉导入依赖（#66）

## v0.2.2 — 2026-03-22

### 修复

- **Anthropic SSE 输出缺少 `content_block_stop`**：将 OpenAI Chat 流式响应转换为 Anthropic SSE 格式时，`content_block_stop` 事件未在 `message_delta` 之前发送，导致 Claude Code 静默丢弃响应内容。Anthropic 转换器现在在处理 `FinishEvent` 时为任何打开的内容块发送 `content_block_stop`（#77）
- **上游预检 chunk 被误判为流结束**：Argo API 在实际内容之前发送一个 `choices: []` 且 `id`/`model` 为空的预检 chunk。OpenAI Chat 转换器现在仅在流已实际启动后才将空 choices chunk 视为流结束（`context.is_started` 守卫）（#77）

## v0.2.1 — 2026-03-20

### 新增

- **网关请求/响应体日志**：可配置的调试日志，支持彩色输出、请求体脱敏和截断 — 通过配置（`"debug": {"verbose": true, "log_bodies": true}`）、环境变量（`LLM_ROSETTA_VERBOSE`、`LLM_ROSETTA_LOG_BODIES`）或 `--verbose` CLI 参数启用
- **Google `request_to_provider()` 支持 `output_format="rest"`**：传入 `output_format="rest"` 可直接获得 REST API 格式的请求体，`tools`/`tool_config` 提升至顶层，生成参数包装在 `generationConfig` 中 — 无需再手动进行 SDK→REST 格式转换

### 变更

- **网关模块化重构**：将 `app.py`（1057 行）拆分为 `proxy.py`（代理引擎、SSE 处理、上游请求）、`cli.py`（CLI 入口、argparse、子命令）和精简后的 `app.py`（路由处理、应用工厂，约 210 行）
- **Google REST 请求体转换迁移至核心包**：`_fixup_google_body()` 逻辑从 `gateway/proxy.py` 迁移至 `GoogleGenAIConverter._to_rest_body()`，消除了网关和全部 6 个 REST 示例中的重复 SDK→REST 转换代码

### 修复

- OpenAI Responses 流式传输：为 `response.completed` 添加缺失的 `id`/`object`/`model` 字段，为文本增量事件添加 `output_index`/`content_index`，并补充完整的生命周期事件（`output_item.added`、`content_part.added`、`content_part.done`、`output_item.done`）（#56）
- OpenAI Chat 流式传输：`tool_calls` 条目现在始终包含必需的 `index` 字段，当上游 IR 事件未明确提供时默认为 `0`（#57）
- OpenAI Chat 流式传输：usage-only 数据块现在包含 `"choices": []`，以满足要求每个 `chat.completion.chunk` 必须包含 `choices` 数组的客户端验证（#55）
- `stream_options`（Chat Completions 专用字段）不再泄漏到 OpenAI Responses API 请求中 — Responses 转换器的 `ir_stream_config_to_p()` 之前错误地输出了 `stream_options`，导致 Chat 格式客户端（Kilo、OpenCode）通过网关代理到 Responses API 时被上游拒绝（#58）
- Google GenAI 转换器现在可以处理 REST 格式请求中顶层的 tools 和 tool_config（除了 SDK 格式的 `config.tools`）— 之前只识别 SDK 格式，导致网关代理请求中的工具定义被静默丢弃（#59）
- Google camelCase `functionDeclarations` 未解析：`p_tool_definition_to_ir()` 现在同时处理 `functionDeclarations`（camelCase/REST）和 `function_declarations`（snake_case/SDK），并提取所有声明而非仅第一个。同时为 `functionCallingConfig`/`allowedFunctionNames` 和 `toolConfig` 添加 camelCase 支持 — 修复 Gemini CLI 通过网关的工具调用（#61）
- Google 流式工具调用被拆分为两个 chunk：`stream_response_to_provider()` 现在延迟 `tool_call_start`，在 `tool_call_delta` 时发送完整的 `function_call`（name + args），匹配 Google API 的原生格式（#62）

## v0.2.0 — 2026-03-18

### 新增

- **独立 API 测试脚本** (`llm_api_simple_tests/`)：20 个测试脚本（每个提供商 5 个），直接使用官方 SDK，覆盖简单查询、多轮对话、图片、函数调用和综合场景 — 作为 git 子模块从 [Oaklight/llm_api_simple_tests](https://github.com/Oaklight/llm_api_simple_tests) 引入
- **LLM-Rosetta Gateway**：跨提供商 HTTP 代理的 REST 网关应用
- CLI 入口 (`llm-rosetta-gateway`) 及网关包结构
- 网关配置文件自动发现：依次搜索 `./config.jsonc`、`~/.config/llm-rosetta-gateway/config.jsonc`、`~/.llm-rosetta-gateway/config.jsonc`
- `--edit` / `-e` 标志：在 `$EDITOR` 中打开配置文件（回退到 nano/vi/vim）
- `--version` / `-V` 标志：显示当前版本
- ASCII 艺术启动横幅，支持 `--no-banner` 选项抑制显示
- `add provider <name>` 子命令：添加提供商条目到配置（支持 `--api-key`、`--base-url` 参数或交互式提示；已知提供商自动填充默认值）
- `add model <name>` 子命令：添加模型路由条目（支持 `--provider` 参数或交互式提示）
- **网关提供商模块** (`providers.py`)：集中管理提供商定义，包括认证头构建器、URL 模板、默认基础 URL 和 API 密钥环境变量名
- **API 密钥轮转**：每个提供商支持逗号分隔的多 API 密钥，通过 `KeyRing` 轮询使用
- **代理支持**：全局 `server.proxy` 和逐提供商 `proxy` 配置，支持 HTTP/SOCKS 代理；CLI `--proxy` 参数覆盖配置
- Makefile 新增 `test-integration` 目标，使用 `proxychains`（如已安装）运行集成测试
- `init` 子命令：在 XDG 默认位置 (`~/.config/llm-rosetta-gateway/`) 创建模板 `config.jsonc` 文件
- **模型列表端点**：`GET /v1/models`（兼容 OpenAI 和 Anthropic SDK）和 `GET /v1beta/models`（Google GenAI SDK 格式）— 使三种 SDK 的 `client.models.list()` 均可正常使用（#54）

### 变更

- 最低 Python 版本提升至 3.10+；迁移至标准库 `typing`（移除 `typing_extensions`）
- 使用 `ruff` 格式化整个代码库
- 更新 Makefile，增加 `lint`、`test` 和 `build` 目标
- 新增 `ty`（类型检查器）配置
- 在 `pyproject.toml` 中配置 `ruff` lint 规则（`E`、`F`、`UP`）；忽略 `UP007`（Union 语法）和 `E501`（行长度）
- 现代化 `src/`、`tests/`、`examples/` 和 `scripts/` 中的 typing 导入 — 将 `typing.Dict`、`List`、`Tuple`、`Optional`、`Type` 替换为标准库内建类型

### 修复

- 修复 Anthropic 提供商流式传输中 usage tokens 为 `null` 时的崩溃 — 所有转换器中 `TypeError: NoneType + int`（将 `.get("*_tokens", 0)` 替换为 `.get("*_tokens") or 0`）
- 网关提供商 `base_url` 验证 — 配置错误（如 `https:example.com` 缺少 `//`）时提前报错并给出清晰提示
- 网关依赖新增 `socksio` 以支持 SOCKS 代理（`httpx[socks]`）
- 补充 `types` 包缺失的 `__init__.py`
- 更新文档中 `git clone` URL，从 `llm-rosetta` 改为 `llm-rosetta`
- 解决 `src/` 中所有 `ty` 类型检查器诊断（31 → 0）：
    - 修复 `is_part_type()` TypeGuard 类型窄化 — 替换为特定类型守卫函数（`is_text_part` 等）
    - 补充缺失的 TypedDict 字段：`TextPart`/`ReasoningPart` 上的 `provider_metadata`，`ImagePart`/`FilePart` 上的 `file_id`
    - 修复 `IRRequest.messages` 类型，从 `Required[Message]` 改为 `Required[Iterable[Message]]`
    - 使用 `cast()` 桥接 `dict[str, Any]` 中间值到 TypedDict 返回类型
    - 修复转换器响应构建器中的 dict 字面量类型推断冲突
- 解决 `tests/` 中所有 `ty` 类型检查器诊断（1506 → 0）：
    - 为传递给期望 TypedDict 参数的函数的 dict 字面量添加 `cast()` 包装（`GenerationConfig`、`IRRequest`、`IRResponse`、`ToolDefinition`、`ToolChoice` 等）
    - 使用 `cast(list[Any], ...)` 或 `cast(Message, ...)` 窄化 `Message | ExtensionItem` 联合类型结果
    - 将 `Iterable` 内容字段转换为 `list` 以支持下标和 `len()` 访问
    - 在对可选返回类型进行下标访问前添加 `assert ... is not None` 守卫
    - 修复 `FinishReason`，从裸字符串改为 TypedDict 形式 `{"reason": "stop"}`
    - 修复 `IRResponse.object` 字面量，从 `"chat.completion"` 改为 `"response"`
- 解决 `src/` 和 `tests/` 中所有 `ruff` lint 违规（UP035 弃用导入、F401 未使用导入）
- Google `thought_signature` 在网关往返中的保留 — 新版 Google 模型要求在函数调用部分中回传 `thoughtSignature`；网关现在按 `tool_call_id` 缓存 `provider_metadata`（含 `thought_signature`），并在后续请求中重新注入，支持流式和非流式模式（#51）
- OpenAI Responses 转换器现在支持全部 3 种 `input` 格式：裸字符串（`"input": "hello"`）、简写列表（`[{"role": "user", "content": "hi"}]`）和结构化列表 — 此前仅支持结构化格式，导致 OpenAI Python SDK 发送的简写项被静默丢弃，跨提供商转换到 Anthropic 或 Google 时生成空 IR 消息

---

## 2026-03-15 — 品牌重塑为 LLM-Rosetta

### 变更

- **项目从 LLM-Rosetta 重命名为 LLM-Rosetta**，涵盖所有代码、文档及配置
- 包名从 `llm-rosetta` 改为 `llm_rosetta`；`pyproject.toml` 相应更新
- 使用 Zensical 全面重写英文 (`docs_en`) 和中文 (`docs_zh`) 文档
- README（中/英）更新品牌标识、徽章及 `pyproject.toml` 元数据

---

## 2026-03-06 — 流式传输与 StreamContext

### 新增

- **`StreamContext`**：为所有 4 个提供商提供有状态的流式数据块处理
- 所有转换器新增 `stream_response_from_provider()` 和 `stream_response_to_provider()` 方法
- `accumulate_stream_to_assistant_message()` 辅助函数
- `BaseConverter` 新增流式抽象方法（`stream_response_to_provider`、`stream_response_from_provider`）
- 4 种新 IR 流式事件类型：`StreamStart`、`StreamEnd`、`ContentBlockStart`、`ContentBlockEnd`
- `ReasoningDeltaEvent` 及 IR 流式类型新增 `tool_call_index` 字段
- 所有提供商组合的跨提供商流式示例（SDK 和 REST 版本）
- 示例中图片下载新增本地文件缓存和重试逻辑

### 变更

- 流式方法签名更新，增加可选 `context` 参数
- 移除已弃用的 `from_provider` 方法；`auto_detect` 更新为新 API
- 移除过时的单提供商示例脚本（已被跨提供商示例替代）
- `_normalize()` 提取至 `BaseConverter` 作为共享工具方法

### 修复

- Google GenAI REST 流式/响应字段的 camelCase 回退处理
- Anthropic 流式转换器：`thinking_delta`、`signature_delta`、`tool_call_id` 处理
- OpenAI Chat 流式转换器：`reasoning_content`、空字符串、`tool_call_index` 处理
- 补充测试包发现所需的 `__init__.py`
- `google_genai_rest_e2e` 集成测试中的 `from_provider` 调用

---

## 2026-02-14 — 跨提供商示例与流式转换器

### 新增

- 所有 4 个提供商的**流式转换器**：OpenAI Chat、Anthropic、Google GenAI、OpenAI Responses
- 所有提供商的流式转换器单元测试
- **6 个跨提供商对话示例**（基于 SDK）：OpenAI Chat ↔ Anthropic、OpenAI Chat ↔ Google GenAI、OpenAI Chat ↔ OpenAI Responses、Anthropic ↔ Google GenAI、Anthropic ↔ OpenAI Responses、Google GenAI ↔ OpenAI Responses
- 跨提供商对话示例的公共资源模块
- Google GenAI 兼容性的图片 URL 转内联 base64 辅助工具
- OpenAI Responses E2E 集成测试（REST + SDK）
- OpenAI Responses Ops 类及转换器的单元测试
- 示例 README（中英文）

### 变更

- **OpenAI Responses 转换器**重构为 Bottom-Up Ops 模式
- 重构后清理：移除弃用工具和空目录

### 修复

- 为 Google GenAI 提供商兼容性将图片 URL 转换为内联 base64

---

## 2026-02-13 — Bottom-Up Ops 架构

### 新增

- **Google GenAI 转换器**使用 Bottom-Up Ops 模式重建
- **OpenAI Responses API** 类型的 TypedDict 副本
- **Google GenAI SDK** 类型的 TypedDict 副本
- Google GenAI REST 和 SDK E2E 集成测试
- `google_genai` 转换器 Ops 类的单元测试
- Anthropic SDK 和 REST E2E 集成测试
- OpenAI Chat E2E 测试拆分为 SDK 和 REST 版本
- **GitHub Actions** CI/CD 工作流及 Dependabot 配置

### 变更

- **Anthropic 转换器**重新设计为 Bottom-Up Ops 架构
- 导入更新为使用新的 `google_genai` 转换器模块
- 移除旧的 `google/` 转换器及遗留测试

---

## 2026-02-12 — 转换器重新设计

### 新增

- **Anthropic SDK** 类型的 TypedDict 副本
- **OpenAI Chat** 类型的 TypedDict 副本，包含向后兼容性和测试
- 保留遗留 body 转换器设计作为历史参考

### 变更

- **OpenAI Chat 转换器**使用 Bottom-Up Ops 架构重新设计
- 修复整个代码库的 ruff lint 错误

---

## 2026-01-06 — 分层架构与文档

### 新增

- 初始化英文和中文文档结构（`docs_en`、`docs_zh`）
- 完整的错误处理文档
- OpenAI Chat Converter 集成测试
- `BaseConverter` 测试类的完整 mock 实现
- 基础转换器的文件处理功能
- 提供商到 IR 的映射文档

### 变更

- 转换器基类完善为分层抽象模板
- 所有 4 个转换器重构为分层架构（Anthropic、OpenAI Chat、OpenAI Responses、Google GenAI）
- IR 内容/部件转换方法的类型注解更新
- IR 类型系统重组和增强
- 代码注释和文档字符串添加英文翻译

### 修复

- 修正推理内容字段断言
- 修复 OpenAI Chat Completions 转换器的文件内容处理

---

## 2026-01-05 — 自动检测与包成熟化

### 新增

- **`detect_provider()`**：自动检测提供商格式
- **`convert()`**：一步格式转换便捷函数
- 消息验证中支持 `developer` 角色
- `BaseConverter`、Anthropic、Google GenAI 和 OpenAI 转换器的综合验证测试
- 工具调用和工具定义转换测试
- pytest 配置及 `pytest-cov` 依赖
- 竞品分析文档

### 变更

- **包重命名**：从 `llm-provider-converter` 改为 `llm-rosetta`
- 所有提供商标准化 IR 格式用法
- 示例中使用 `Message` 类标准化消息创建
- 测试套件从 unittest 迁移至 pytest
- 提取公共逻辑至共享工具模块

### 修复

- OpenAI Responses 转换器中无当前消息上下文时的独立工具调用处理
- Google GenAI Pydantic 模型处理重新排序以兼容元组
- OpenAI 单文本部件的内容处理逻辑简化

---

## 2026-01-04 — 示例与打包

### 新增

- `pyproject.toml` 包配置
- 带工具集成的多轮对话示例
- 多轮对话示例中的 Anthropic 切换
- 多轮对话示例中的 Google GenAI 函数调用

### 变更

- 工具函数从转换器移至 IR 类型模块
- OpenAI Chat 转换器代码格式化改进
- 移除弃用的多提供商查询和天气工具模块

---

## 2025-12-24 — 初始实现

### 新增

- **IR 类型系统**：消息、内容部分、工具、配置、请求/响应的中间表示类型
- **`BaseConverter`** 抽象类：LLM 提供商转换基类
- **`AnthropicConverter`**：Anthropic Messages API 双向转换
- **`OpenAIChatConverter`**：OpenAI Chat Completions API 双向转换
- **`OpenAIResponsesConverter`**：OpenAI Responses API 双向转换
- **`GoogleGenAIConverter`**：Google GenAI SDK 格式双向转换
- 所有 4 个转换器的综合测试套件
- 包初始化和导出
- 带模拟数据的天气工具示例

---

## 2025-12-09 — 研究与设计

### 新增

- 初始项目结构
- LLM 提供商消息类型 schema 文档及比较
- 提供商消息 IR 设计文档
- 跨提供商 MCP 支持对比（OpenAI、Anthropic、Google）
- Google GenAI Interactions API 类型分析
- 多提供商查询示例函数
- 查询示例中增加 OpenAI Responses API 支持
