---
title: Changelog
---

# Changelog

All notable changes to LLM-Rosetta are documented here. This project follows [Keep a Changelog](https://keepachangelog.com/) conventions.

## [Unreleased]

## v0.7.0a1 — 2026-06-27

### 新增

- **可观测性包** ([#341](https://github.com/Oaklight/llm-rosetta/issues/341))：将 `MetricsCollector`、`RequestLog`、`RequestLogEntry`、`PersistenceManager` 和 `ProfilerState` 从 `gateway/admin/` 提取到新的顶层 `llm_rosetta.observability` 包。这些模块与框架无关，任何 LLM 代理消费者（如 argo-proxy）均可直接使用，无需依赖网关的配置系统或 HTTP 服务器。`gateway/admin/` 模块通过重新导出保持完全向后兼容
- **混合性能分析系统** ([#339](https://github.com/Oaklight/llm-rosetta/pull/339))：`ConversionPipeline.profile` 中内置 always-on 的 `perf_counter` 阶段计时（source_to_ir_ms、ir_transforms_ms、ir_to_target_ms 等），加上按需的 per-request pyinstrument 深度分析（通过 admin API 控制）。核心库新增 `DeepProfiler` 上下文管理器（`llm_rosetta.profiling`）。新增 `[profiling]` 可选依赖组。Gateway admin 端点：`POST /admin/api/profiling/enable`、`GET /admin/api/profiling/results`、`GET /admin/api/profiling/results/<index>`、`POST /admin/api/profiling/disable`、`DELETE /admin/api/profiling/results`
- **性能分析管理 UI** ([#339](https://github.com/Oaklight/llm-rosetta/pull/339))：管理面板新增 "Profiling" 区域，包含启用/停用控制、结果列表、火焰图下载（单个和批量）以及重启提示
- **错误转储功能** ([#341](https://github.com/Oaklight/llm-rosetta/issues/341))：Fire-and-forget 错误转储系统，在上游/转换失败时捕获完整请求上下文。哈希前进行图片卸载以实现基于内容的去重，zlib 压缩，10K 条目上限并级联修剪。proxy.py/app.py 中 4 个触发点覆盖上游错误、流头部错误、流块错误和转换错误。新导出函数：`dump_error()`、`offload_images()`、`compute_body_hash()`、`compress_body()`、`decompress_body()`（从 `llm_rosetta.observability`）
- **指标重建** ([#340](https://github.com/Oaklight/llm-rosetta/pull/340))：`POST /admin/api/metrics/rebuild` 端点及管理面板 "Rebuild Counters" 按钮。使用 `fetchmany(5000)` 批量迭代和原子交换从请求日志历史中重建所有指标计数器，避免暴露半重建状态

### 修复

- **指标按提供商名称分组** ([#340](https://github.com/Oaklight/llm-rosetta/pull/340))：面板 breakdown 区域现在按提供商显示名称分组，而非按 API 类型（此前将所有 Anthropic 格式的提供商合并为一行）
- **配置文件写入安全**：`write_config()` 现使用文件锁确保跨进程安全
- **Vendored httpserver 更新至 0.2.1**：对格式错误的请求返回正确的 HTTP 错误响应，而非静默断开连接
- **Vendored SSE 更新至 0.3.2**：解析器初始化使用构造函数参数，而非初始化后修改

### 变更

- **开发工具版本锁定**：在 `[project.optional-dependencies]` 中锁定 `ruff==0.15.20` 和 `ty==0.0.54`，防止上游工具发版导致 CI 漂移

## v0.7.0a0 — 2026-06-25

### 新增

- **ConversionPipeline 类** ([#322](https://github.com/Oaklight/llm-rosetta/pull/332))：高层编排类，封装完整的 Phase 1→2→4 转换生命周期。提供 `convert_request()`、`convert_response()`、`create_stream_processor()` 及 `on_ir_ready` 回调用于元数据存储集成。一次性保护防止意外复用
- **路由层** ([#323](https://github.com/Oaklight/llm-rosetta/pull/331))：核心库中的 `ResolvedRoute` 冻结数据类和 `Router` 协议。`GatewayConfig.resolve()` 将模型查找、provider 类型、shim 绑定、capabilities 和 reasoning 覆盖整合为单一类型化结果
- **能力模块** ([#335](https://github.com/Oaklight/llm-rosetta/pull/336))：`capabilities.py` 包含 `enforce_reasoning()`（IR 前）和 `enforce_vision()`（IR 后）——平台级能力约束，与 provider 特定的 shim 变换分离
- **IRTransform 系统** ([#330](https://github.com/Oaklight/llm-rosetta/pull/334))：`TransformContext` 数据类、`IRTransform` 可调用类型、`apply_ir_transforms()` 执行器和 `_NamedIRTransform` 包装器。IR 层变换现在在 `ProviderShim.ir_transforms` 上声明式配置，与 body 层 `Transform` 分离
- **IR 变换工厂函数**：`strip_non_vision_images()`、`truncate_images(max, pattern)`、`unwind_parallel_tool_calls(pattern)` —— 产出 `IRTransform` 可调用对象的工厂函数
- **消息级变换原语** ([#328](https://github.com/Oaklight/llm-rosetta/pull/333))：`replace_message_field()`、`default_message_field()`、`strip_fields_for_model()`，用于 `messages[]` 嵌套字段操作
- **Transport 层** ([#321](https://github.com/Oaklight/llm-rosetta/pull/329))：`UpstreamTransport` 协议、`HttpTransport` 实现、`UpstreamResponse`/`UpstreamStream` 类型、`HttpClientPool`、`send_passthrough()` 用于非转换端点
- **`resolve_shim()` 公共函数**：从私有 `_resolve_shim()` 提升为 `provider_shim.py` 上的公共 API

### 破坏性变更

- **`ProviderShim` 字段移除**：删除 `max_images`、`max_images_pattern`、`unwind_parallel_tool_calls`、`unwind_parallel_tool_calls_pattern` —— 这些能力现通过 `ir_transforms` 元组使用工厂函数声明（`truncate_images()`、`unwind_parallel_tool_calls()`）
- **`apply_shim_to_ir()` 行为变更**：不再硬编码图片/unwind 操作，改为声明式读取 `shim.ir_transforms`。重命名为 `apply_ir_transforms()`（旧名称为弃用别名）
- **Gateway handler 签名变更**：`handle_non_streaming` 和 `handle_streaming` 接收 `route: ResolvedRoute` 而非 6 个松散参数

### 重构

- **Pipeline 重命名** ([#330](https://github.com/Oaklight/llm-rosetta/pull/334))：`apply_shim_to_ir()` → `apply_ir_transforms()`、`setup_shim_context()` → `configure_context()`。旧名称发出 `DeprecationWarning`
- **Gateway proxy.py**：handler 内部使用 `ConversionPipeline`。删除 `_resolve_target_transforms`、`process_stream_chunk`
- **Embeddings handler**：使用 `transport.send_passthrough()` 替代直接访问 `HttpTransport._pool`。从旧的 `resolve_model()` 迁移至统一的 `resolve()` API，用共享的 `_record_telemetry()` 替换内联遥测代码
- **认证函数重命名**：`_openai_auth` → `openai_auth` 等（去掉下划线，公共 API）
- **移除 `GatewayConfig.resolve_model()`**：旧的 5-tuple API 已被返回 `ResolvedRoute` + `ProviderInfo` 的 `resolve()` 取代。移除重复的 `DEFAULT_CAPABILITIES` 类变量

### 修复

- **恢复旧 `converters/base/` 导入路径** ([#317](https://github.com/Oaklight/llm-rosetta/pull/317))：在旧路径提供向后兼容 shim 模块
- **`sanitize_schema` 剥离 `exclusiveMinimum`/`exclusiveMaximum`** ([#337](https://github.com/Oaklight/llm-rosetta/pull/337))：Google GenAI API 拒绝工具定义中的 JSON Schema draft 6+ 数值约束
- **停止为 OpenAI Responses API 生成 `reasoning.type`** ([#337](https://github.com/Oaklight/llm-rosetta/pull/337))：OpenAI 和 Volcengine Responses API 拒绝 `reasoning.type` —— reasoning 仅通过 `reasoning.effort` 控制。v0.6.8 的历史 bug

## v0.6.12 — 2026-06-23

### 修复

- **恢复 `converters/base/` 旧导入路径** ([#310](https://github.com/Oaklight/llm-rosetta/issues/310))：v0.6.11 的 helpers/ 重组意外破坏了外部调用方依赖的导入路径。现在 `sanitize_schema`、`extract_part_ids`、`log_orphan_warnings`、`fix_orphaned_tool_calls_ir`、`strip_orphaned_tool_config` 重新从 `converters.base.tools` 导出，并在 `converters.base.schema`、`converters.base.tool_content`、`converters.base.cache` 提供兼容性 shim 模块，重定向到它们在 `helpers/` 下的新位置。规范导入路径仍为 `llm_rosetta.converters.base.helpers`；按旧路径导入的现有代码（如 `from llm_rosetta.converters.base.tools import sanitize_schema`）无需改动即可继续工作。缓存单例在新旧路径间共享

## v0.6.11 — 2026-06-21

### 新增

- **Admin 面板服务方 UX 增强** ([#292](https://github.com/Oaklight/llm-rosetta/pull/292))：服务方标签页三项改进：
    - **多密钥条目列表**：API 密钥字段自动检测逗号分隔的密钥（轮转），切换为多个 `<input type="password">` 输入框。始终显示 `+ 添加密钥` 按钮。眼睛和复制按钮统一在底部
    - **服务方搜索栏**：服务方数量超过 6 个时自动显示，支持按名称、类型、Base URL 过滤
    - **网格/列表视图切换**：两个图标按钮切换卡片网格和紧凑单列列表视图，偏好保存在 localStorage
- **Request ID 传播** ([#296](https://github.com/Oaklight/llm-rosetta/pull/296), [#122](https://github.com/Oaklight/llm-rosetta/issues/122))：每个代理请求生成或继承 `X-Request-ID` 头。向上游传播，包含在所有响应头中（包括错误响应），并以 `[request_id]` 前缀记录日志，实现端到端可追踪
- **增强健康检查端点** ([#297](https://github.com/Oaklight/llm-rosetta/pull/297), [#127](https://github.com/Oaklight/llm-rosetta/issues/127))：
    - `/health` — 返回运行时间、请求总数、最近一小时错误数、每个服务方的健康状态快照（成功率、平均延迟、最后错误）。始终 HTTP 200；`status` 字段显示 `"ok"` 或 `"degraded"`
    - `/health/live` — 始终 200（Kubernetes 存活探针）
    - `/health/ready` — 所有服务方健康时 200，任一服务方严重不健康时 503（Kubernetes 就绪探针）
- **Admin API 的 CORS 限制** ([#294](https://github.com/Oaklight/llm-rosetta/pull/294), [#233](https://github.com/Oaklight/llm-rosetta/issues/233))：`/admin/api/*` 端点不再发送 `Access-Control-Allow-Origin: *`。新增配置选项 `server.admin_cors_origins`（列表，默认 `[]`）允许显式指定允许的来源。`/v1/*` 代理端点不受影响
- **Shim 层图片数量限制** ([#301](https://github.com/Oaklight/llm-rosetta/pull/301), [#299](https://github.com/Oaklight/llm-rosetta/issues/299))：`ProviderShim` 新增 `max_images` 和 `max_images_pattern` 字段。超限时最早的图片被替换为 `[image omitted due to limit]`，保留最近的 N 张。Argo OpenAI shim 声明 `max_images: 50`，pattern 为 `^(gpt|o\d)` — 仅 GPT/o 模型被截断；经过同一服务方的 Gemini 和 Claude 不受影响
- **视觉能力运行时检查** ([#314](https://github.com/Oaklight/llm-rosetta/pull/314), [#313](https://github.com/Oaklight/llm-rosetta/issues/313))：没有 `vision` 能力的模型会自动将所有图片替换为 `[image not available]`，而非直接转发给上游导致不明错误（如 DeepSeek 的 "unknown variant `image_url`"）。Gateway 日志会记录 warning 包含图片数量和模型名
- **Unix 域套接字支持** ([#315](https://github.com/Oaklight/llm-rosetta/pull/315))：Gateway 可通过 `--socket/-S` CLI 参数或 `server.socket` 配置字段监听 Unix 套接字而非 TCP。适用于共享多用户主机上的安全部署（`127.0.0.1` 仍会暴露给所有本地用户）。套接字文件权限限制为仅所有者可访问（`0600`），关闭时自动清理
- **并行工具调用展开** ([#303](https://github.com/Oaklight/llm-rosetta/pull/303), [#300](https://github.com/Oaklight/llm-rosetta/issues/300))：`ProviderShim` 新增 `unwind_parallel_tool_calls` 和 `unwind_parallel_tool_calls_pattern` 字段。启用后，并行工具调用（一条 assistant 消息包含多个 `tool_call`）会在转发前展开为顺序调用-结果对。Argo OpenAI shim 以 `^gemini` pattern 启用 — Gemini 模型获得顺序对；GPT/o 模型不受影响

### 变更

- **`converters/base/` 重组为 helpers/ 子包** ([#311](https://github.com/Oaklight/llm-rosetta/pull/311), [#312](https://github.com/Oaklight/llm-rosetta/pull/312), [#310](https://github.com/Oaklight/llm-rosetta/issues/310))：工具函数从 `converters/base/` 平铺目录提取到 `converters/base/helpers/`。抽象基类（Ops 模式契约）保留在顶层；实现工具（`cache`、`schema`、`tool_orphan_fix`、`tool_content`、`tool_call_unwind`、`image_limit`、`reasoning`）移至 `helpers/`。`tools.py` 从 428 行精简到 185 行（纯 ABC）。`reasoning_helpers.py` 从 `converters/` 根目录移入。`orphan_fix.py` 重命名为 `tool_orphan_fix.py` 保持 `tool_*` 前缀一致。`helpers/__init__.py` 重新导出公共函数

- **移除 Argo `_normalize_thinking` 废弃代码** ([#304](https://github.com/Oaklight/llm-rosetta/pull/304), [#192](https://github.com/Oaklight/llm-rosetta/issues/192))：从 Argo Anthropic shim 中移除了已废弃的 `_normalize_thinking` 函数、`_BUDGET_RATIO` 和 `_ADAPTIVE_THINKING_MODELS`——这些已被 `provider.yaml` 中声明式的 `reasoning.model_overrides` 取代，但代码和 19 个测试仍然保留着
- **实验性扩展类型标记** ([#302](https://github.com/Oaklight/llm-rosetta/pull/302), [#71](https://github.com/Oaklight/llm-rosetta/issues/71))：`SystemEvent`、`BatchMarker`、`SessionControl`、`ToolChainNode` 从 `types.ir.extensions` 移至 `types.ir.extensions_experimental`。旧导入路径仍可用但会触发 `DeprecationWarning`。这些类型从默认 `types.ir` 命名空间移除，可通过 `from llm_rosetta.types.ir import experimental` 访问
- **Admin 面板 i18n**：中文翻译从"服务商"更新为"服务方"（对混合商业和自建服务方更中性）
- **请求日志时间戳** ([#298](https://github.com/Oaklight/llm-rosetta/pull/298))：现在显示日期和时间（如 "06/19, 20:25:29"），而非仅显示时间

### 修复

- **Admin 面板认证内容闪烁** ([#291](https://github.com/Oaklight/llm-rosetta/pull/291))：消除了配置 `admin_password` 时登录遮罩出现前短暂显示管理内容的问题。通过 CSS（`body.auth-pending`）在异步认证检查完成前隐藏主界面
- **Admin 密码未解析环境变量** ([#293](https://github.com/Oaklight/llm-rosetta/pull/293))：如果 `admin_password` 包含未解析的 `${...}` 占位符，网关现在拒绝启动，防止可预测的字面量字符串被用作密码
- **`is_image_part` 类型守卫支持 OpenAI 格式** ([#306](https://github.com/Oaklight/llm-rosetta/pull/306))：`is_image_part()` 现在同时匹配 `type: "image"`（IR 规范格式）和 `type: "image_url"`（OpenAI 格式保留在 IR 中），修复了 OpenAI 格式请求的图片截断被静默跳过的问题
- **工具结果中的图片纳入截断计数** ([#308](https://github.com/Oaklight/llm-rosetta/pull/308), [#299](https://github.com/Oaklight/llm-rosetta/issues/299))：`truncate_images()` 现在扫描 `tool_result.result` 列表内的图片，而不仅是直接消息内容。修复了在 IR 层面 ≤50 张图片但 OpenAI Chat 转换器解包工具结果图片后超过 50 张的请求。同时优化了 `deepcopy`，仅复制受影响的消息而非整个对话
- **Argo Gemini 并行工具调用失败** ([#303](https://github.com/Oaklight/llm-rosetta/pull/303), [#300](https://github.com/Oaklight/llm-rosetta/issues/300))：Claude Code 发起并行工具调用时，所有通过 Argo 的 Gemini 模型都报 "function response parts ≠ function call parts" 错误。根因：Argo 内部的 OpenAI→Gemini 转换不会将独立的 tool result 消息合并为单个 `functionResponse` Content 块。通过在转发前将并行工具调用展开为顺序对修复

## v0.6.10 — 2026-06-18

### Added

- **Process-level conversion cache** ([#276](https://github.com/Oaklight/llm-rosetta/issues/276), [#279](https://github.com/Oaklight/llm-rosetta/pull/279), [#281](https://github.com/Oaklight/llm-rosetta/pull/281), [#283](https://github.com/Oaklight/llm-rosetta/pull/283)): Per-entry LRU cache with access-refreshed TTL (default 30 min) for tool conversion, schema sanitization, and IR validation. Eliminates repeated work for unchanged tool definitions and messages across conversation turns
    - **Hub-and-spoke architecture**: conversion caches (spokes) are converter-specific; IR validation cache (hub) is converter-agnostic and shared across all converters
    - **Per-entry caching**: individual tools and messages cached by content hash — partial tool changes only re-convert the changed entries, and cross-agent tool overlap shares cache entries
    - **Incremental message validation**: only newly appended messages are validated; previously-seen messages are skipped via the IR validation hub
    - **Mutation detection**: `check_integrity()` on test teardown catches accidental in-place mutation of cached objects; optional `verify=True` mode for runtime self-healing
    - **Benchmark**: 4.4× warm-path speedup (3250 µs → 527 µs local); 33% TTFB reduction in production (11.4 ms → 7.6 ms)
- **`validate_tools()`** ([#283](https://github.com/Oaklight/llm-rosetta/pull/283)): New standalone IR validation function for tool definition lists, symmetric with `validate_messages()`
- **OpenRouter Anthropic shim** ([#284](https://github.com/Oaklight/llm-rosetta/pull/284)): OpenRouter's Anthropic-compatible Messages endpoint is now a first-class provider type. The single `openrouter` shim is split into `openrouter--openai_chat` (Chat Completions) and `openrouter--anthropic` (Messages API), letting OpenRouter route Claude models through the native Anthropic format
- **Admin panel per-model reasoning override** ([#288](https://github.com/Oaklight/llm-rosetta/pull/288)): The model edit modal now displays the effective reasoning config (`thinking_type`, `budget_tokens_ratio`, `disabled_strategy`) with a source badge (provider / model_override / config) and inline editing. Overrides are persisted to `config.jsonc` and resolved at runtime with priority: config override > shim model_override > shim provider default
- **`budget_tokens_default_ratio` reasoning capability** ([#287](https://github.com/Oaklight/llm-rosetta/pull/287)): `ReasoningCapability` gains a `budget_tokens_default_ratio` field. When a provider requires `thinking.type=enabled` but the caller omits `budget_tokens`, a default is derived as `min(max(1024, max_tokens × ratio), max_tokens - 1)` instead of falling back to the unsupported `adaptive` type

### Changed

- **`_convert_tools_from_p` no longer abstract** ([#281](https://github.com/Oaklight/llm-rosetta/pull/281)): Default implementation in `BaseConverter` handles all providers (including Google's list/None return). Per-converter overrides removed — 90 lines of duplicated code eliminated
- **Complete Claude thinking model_overrides** ([#287](https://github.com/Oaklight/llm-rosetta/pull/287)): Added per-model thinking overrides for the Anthropic and Argo shims based on tested support matrices — Haiku 4.5 (`enabled`+budget), Opus 4.7/4.8 (`adaptive`-only), Sonnet 4 on Argo (`enabled`+budget)
- **Model "Clone" replaces "Copy"** ([#290](https://github.com/Oaklight/llm-rosetta/pull/290)): The model row's clone action now opens a prefilled model modal (provider, capabilities, upstream model, and effective reasoning config) with a blank name, matching the provider row's "Clone" behavior — instead of copying a YAML snippet to the clipboard. The model name in the table remains click-to-copy

### Fixed

- **Haiku 4.5 `adaptive` thinking 400 errors** ([#287](https://github.com/Oaklight/llm-rosetta/pull/287)): Haiku 4.5 supports extended thinking but only accepts `thinking.type=enabled` + `budget_tokens`, not `adaptive`. The previous fallback to `adaptive` when no budget was provided caused 400 errors on Anthropic Official, Argo, and OpenRouter. The new `budget_tokens_default_ratio` derives a budget instead
- **Haiku 4.5 `effort` parameter 400 errors** ([#289](https://github.com/Oaklight/llm-rosetta/pull/289)): The `effort` parameter (`output_config.effort`) is only supported on Opus 4.5/4.6/4.7/4.8 and Sonnet 4.6 — not Haiku. Anthropic Official rejected `reasoning_effort` on Haiku 4.5 with a 400. The Haiku model_override now sets `effort_field: none` to drop the unsupported field while keeping the working `thinking.type=enabled` + budget path
- **OpenRouter Anthropic reasoning effort field** ([#284](https://github.com/Oaklight/llm-rosetta/pull/284)): The `openrouter--anthropic` shim uses `output_config.effort` (Anthropic format) instead of the OpenAI Chat `reasoning_effort` field
- **`.env` secret leakage in Docker builds**: Docker build context no longer includes `.env` files, preventing API keys from being baked into image layers

## v0.6.9 — 2026-06-13

### Added

- **API key rotate**: New `POST /admin/api/keys/<id>/rotate` endpoint generates a fresh key value while preserving the same id and label. The admin panel shows a "Rotate" button with inline confirmation and a one-time copy modal for the new key. Request logs are unaffected — they associate by label, not key value
- **Model type selector in Fetch from Provider modal**: Users can now choose between LLM and Embedding when batch-adding models. LLM shows capability checkboxes (text, vision, tools, reasoning); Embedding auto-sets `['embedding']`
- **Model type selector in Add/Edit Model modal**: Replaces the old embedding checkbox + mutual-exclusion logic with the same Model Type radio pattern

### Changed

- **API key length upgraded**: Default generated keys increased from 36 characters (`rsk-` + 32 hex) to 52 characters (`rsk-` + 48 hex), matching OpenAI's key length (192-bit entropy)

### Fixed

- **SSE streaming proxy compatibility** ([#274](https://github.com/Oaklight/llm-rosetta/issues/274), [#275](https://github.com/Oaklight/llm-rosetta/pull/275)): Vendored `httpserver` v0.1.1 — SSE (`text/event-stream`) streaming responses now use `Transfer-Encoding: chunked` instead of raw byte flushing with `Connection: close`. Fixes Go-based reverse proxies (notably NPS `httputil.ReverseProxy`) misinterpreting SSE data as chunked encoding, producing `invalid byte in chunk length` errors and intermittent connection failures under concurrent load. Upstream fix: [Oaklight/zerodep#101](https://github.com/Oaklight/zerodep/pull/101)
- **Admin panel active tab not loading after login**: `initApp()` now triggers data loading for the currently active tab after successful authentication, fixing the issue where the Request Log tab appeared empty until manually switched away and back
- **Uppercase model type radio labels**: Added `text-transform: none` to `.fetch-type-radios label` to prevent `.form-group label` CSS from uppercasing "Embedding" to "EMBEDDING"

## v0.6.8 — 2026-06-11

### Added

- **Shim-driven reasoning configuration** ([#244](https://github.com/Oaklight/llm-rosetta/issues/244), [#245](https://github.com/Oaklight/llm-rosetta/pull/245)): Reasoning effort mapping is now declarative. Provider shims declare a `ReasoningCapability` in `provider.yaml` — specifying `disabled` strategy (`omit` or `thinking_disabled`), `effort_field`, `effort_map`, and `max_effort` cap — instead of hardcoded converter branches. New shared `reasoning_helpers.py` provides `normalize_reasoning_input()` and `apply_reasoning_config()` used by all four converters
- **Expanded reasoning effort ladder** ([#245](https://github.com/Oaklight/llm-rosetta/pull/245)): IR `ReasoningEffortLevel` expanded to six levels: `minimal`, `low`, `medium`, `high`, `xhigh`, `max`. Input normalization accepts `none` (maps to `mode: disabled`) and provider-native values (`xhigh`, `max`) as first-class efforts. Provider shims declare `effort_map` to convert IR levels to provider-specific strings and `max_effort` to cap the highest level emitted
- **`block_index` on IR stream delta events** ([#246](https://github.com/Oaklight/llm-rosetta/issues/246), [#249](https://github.com/Oaklight/llm-rosetta/pull/249)): `TextDeltaEvent`, `ReasoningDeltaEvent`, and `ToolCallDeltaEvent` now carry an optional `block_index` field, preserving the provider's content block index through IR round-trips
- **`cache_creation_tokens` in `UsageInfo`** ([#252](https://github.com/Oaklight/llm-rosetta/pull/252)): New field on the `UsageInfo` TypedDict for Anthropic cache creation token counts
- **Model-level `thinking_type` in shim reasoning config** ([#256](https://github.com/Oaklight/llm-rosetta/pull/256)): `ReasoningCapability` gains a `thinking_type` field to force the outbound `thinking.type` to `"enabled"` or `"adaptive"`. `ProviderShim` gains `model_reasoning` for per-model overrides keyed by upstream model ID (e.g. Argo `claudeopus47 → thinking_type: adaptive`). The `_normalize_thinking` transform is retired — thinking type normalization is now declarative via shim YAML
- **Anthropic `provider_metadata` on tool calls, tool results, and reasoning blocks** ([#257](https://github.com/Oaklight/llm-rosetta/pull/257)): The Anthropic converter now serializes `provider_metadata` as `_provider_metadata` on `tool_use`, `tool_result`, and `thinking` blocks during IR→provider conversion, and reads it back during provider→IR. Fixes Google `thought_signature` being lost in cross-provider round-trips (Anthropic client → Google upstream), which caused Gemini 2.5+ to reject requests with 400 "missing thought_signature"
- **Response reasoning losslessness across converters** ([#263](https://github.com/Oaklight/llm-rosetta/pull/263)): Reasoning content is now preserved through response-side IR→provider conversion in all converters that previously dropped it:
    - **Google GenAI**: `p_reasoning_to_ir` now captures `thoughtSignature` into `provider_metadata` instead of discarding it; `message_ops` delegates to `content_ops.p_reasoning_to_ir()` instead of constructing a bare `ReasoningPart` inline
    - **Anthropic**: `ir_text_to_p` / `p_text_to_ir` now round-trip `_provider_metadata` on text blocks, matching the treatment already applied to reasoning and tool blocks
    - **OpenAI Chat**: `_build_choice_to_provider` now collects `ReasoningPart` content and emits it as `reasoning_content` on the response message, instead of silently dropping reasoning parts

- **Provider-specific reasoning field normalization** ([#264](https://github.com/Oaklight/llm-rosetta/pull/264)): Shim transforms and config for MiniMax, OpenRouter, and Volcengine reasoning fields:
    - **MiniMax**: `thinking_type: adaptive` (rejects `enabled`); `_inject_reasoning_split` to_transform auto-sets `reasoning_split: true` when thinking is requested; `_parse_think_tags` from_transform extracts `<think>` tags from content as fallback
    - **OpenRouter**: `_rename_reasoning_field` from_transform renames `message.reasoning` → `message.reasoning_content` (OpenRouter uses non-standard field name)
    - **Volcengine**: `thinking_type: enabled` (rejects `adaptive`; overrides base converter's `auto → adaptive` default)

### Changed

- **`_build_ir_usage` return type tightened to `UsageInfo`** ([#253](https://github.com/Oaklight/llm-rosetta/pull/253)): All four converter overrides now return `UsageInfo` instead of `dict[str, Any]`, and `_build_provider_usage` accepts `Mapping[str, Any]` instead of `dict[str, Any]`. Removes all usage-related `ty: ignore` comments
- **Anthropic stream usage handlers deduplicated** ([#253](https://github.com/Oaklight/llm-rosetta/pull/253)): `_handle_message_start_from_p` and `_handle_message_delta_from_p` now call `_build_ir_usage()` instead of duplicating cache field extraction inline (−21 lines)

### Fixed

- **Anthropic stream block index desync after thinking block** ([#246](https://github.com/Oaklight/llm-rosetta/issues/246), [#249](https://github.com/Oaklight/llm-rosetta/pull/249)): During Anthropic→IR→Anthropic streaming round-trip, text deltas after a thinking block used index 0 instead of the correct block index (e.g. 1). The Anthropic `from_p` path now copies `chunk["index"]` onto IR delta events, and the `to_p` path prefers the explicit `block_index` over the context fallback. Fixes Claude CLI "Content block is not a text block" errors
- **Cross-provider stream block boundary synthesis** ([#250](https://github.com/Oaklight/llm-rosetta/issues/250), [#251](https://github.com/Oaklight/llm-rosetta/pull/251)): When converting IR streams from providers without content block events (OpenAI Chat, OpenAI Responses, Google GenAI) to Anthropic format, the serializer now emits synthetic `content_block_stop` / `content_block_start` at content-type transitions (e.g. reasoning → text). Previously text deltas could land inside a synthetic thinking block. Added `current_block_type` tracking to `StreamContext`
- **Stream usage detail propagation** ([#252](https://github.com/Oaklight/llm-rosetta/pull/252)): Cache and detail token fields (`cache_read_tokens`, `cache_creation_tokens`, `prompt_tokens_details`, `completion_tokens_details`, `cachedContentTokenCount`) are now preserved through all four converters' streaming paths. Previously these fields were dropped during stream round-trips
- **OpenAI Chat `thinking.type=auto` passthrough** ([#258](https://github.com/Oaklight/llm-rosetta/pull/258)): IR `mode: "auto"` is not a valid upstream value for OpenAI Chat's `thinking.type`. The OpenAI Chat converter now maps `auto` → `adaptive` before emitting the `thinking` object, and applies the same shim `thinking_type` override + `enabled` → `adaptive` safety fallback that the Anthropic path uses
- **`thinking_type=enabled` fallback when `budget_tokens` missing**: When a shim declares `thinking_type: enabled` but the request has no `budget_tokens` (required by Anthropic for `type: "enabled"`), the converter now automatically falls back to `type: "adaptive"` instead of emitting an invalid payload. Applied to both Anthropic and OpenAI Chat converter paths
- **Unsigned Anthropic reasoning blocks in Argo history** ([#268](https://github.com/Oaklight/llm-rosetta/issues/268), [#269](https://github.com/Oaklight/llm-rosetta/pull/269)): `ReasoningCapability` now supports `unsigned_reasoning_blocks: as_is | preserve`. The `argo--anthropic` shim uses `preserve` so prior assistant `thinking` blocks without a usable signature are not forwarded to Argo, avoiding 400 errors while preserving the reasoning content in `provider_metadata.anthropic.unsigned_reasoning_blocks`

## v0.6.7 — 2026-06-04

### Fixed

- **Embedding endpoint upstream_model alias**: The `/v1/embeddings` passthrough handler now substitutes the `upstream_model` name into the request body before forwarding, matching the behavior of the chat completions proxy handler. Previously model aliases (e.g. `bge-m3` → `BAAI/bge-m3`) were ignored, causing upstream model-not-found errors.
- **Admin test timer leak**: The elapsed-time counter is now tracked globally and cleared when a new test starts, preventing multiple timers from writing alternating values to the same display element.
- **Admin test timeout auto-cancel**: When the browser-side 120s timeout fires, the server-side task is now explicitly cancelled via the API instead of being left running.
- **Server-side test task timeout**: Added `asyncio.wait_for()` with a 120s timeout to `_run_test_task`, so hung upstream calls are terminated server-side instead of lingering until the 300s cleanup window.

## v0.6.6 — 2026-06-03

### Added

- **Admin status bar total requests**: Lifetime request counter shown as the first footer segment with locale-aware thousand separators; per-segment hover tooltips (en/zh) explain each metric
- **Vendor httpclient URL-encoded form data**: `httpclient` v0.4.2 — when `data` is a dict without files, encode as `application/x-www-form-urlencoded` instead of requiring explicit serialization

### Changed

- **Schema sanitization module split**: JSON Schema sanitization extracted from `converters/base/tools.py` into its own `converters/base/schema.py` module for clearer separation of concerns
- **Cyclomatic complexity reduction**: Reduced cognitive complexity across tool ops (cross-converter `extract_part_ids`/`log_orphan_warnings` reuse), gateway auth (`check_admin_auth`), proxy streaming (`process_stream_chunk`), config parsing, logging, and admin routes
- **complexipy threshold**: Raised `max-complexity-allowed` from 15 to 25; added `complexipy-pre-commit` hook definition (commented out) for future enablement

### Fixed

- **Admin footer i18n**: Status bar footer now re-renders on language switch instead of requiring a page refresh
- **Docker non-semver build**: `make build-docker V=dev-test` no longer fails — non-semver `V` values fall back to installing from local wheel instead of `pip install ==<version>`

## v0.6.5 — 2026-06-02

### Added

- **API key label filter** — new dropdown on the Request Log tab to filter entries by API key name
- **Client IP logging** — extracts client IP from `X-Forwarded-For` / `X-Real-IP` / TCP peer address and displays it in a new "Client IP" column on the Request Log tab
- **System clock** — live-updating clock in the admin header for correlating log timestamps with current time
- **Dual-threshold log retention** — success and error request log entries are pruned independently; errors get their own cap (`error_max`) so rare failures are not evicted by a flood of successful traffic
- **DB sizing footer** — admin panel footer shows on-disk database size, entry counts per class, and retention caps

### Fixed

- **Provider filter** — filter now correctly matches entries by provider display name, with three-tier fallback (`target_provider_name` → `target_provider` → API type for legacy NULL rows) to handle backfill gaps and disabled providers
- **`/health` info leak** — endpoint no longer exposes the full provider and model list to unauthenticated callers; now returns only `{"status": "ok"}`
- **i18n completeness** — added missing Chinese translations for footer stats, system time label, filter options, and Client IP column header

### Changed

- **Shim directory layout** — provider shims now support grouped subdirectories (e.g. `argo/anthropic/`, `argo/openai_chat/`)
- **Schema migration** — `_migrate_add_columns()` is now generic, adding any missing nullable columns in a single pass
- **CI** — switched to pre-commit for lint/type checks, pinned ty version

## v0.6.4 — 2026-05-20

### Added

- **Tinyleaf-style settings popup**: Replace the modal-overlay settings dialog with a lightweight centered popup — click outside or press Escape to dismiss, theme and language via `<select>` dropdowns with instant apply, About section with version and project links (GitHub, PyPI, Docker Hub, Docs)
- **Lightweight host IP detection endpoint**: `GET /admin/api/diagnostics/host-ip` reads `/proc/net/route` only (microsecond-level, no network calls); proxy URL placeholders auto-update with the correct Docker host IP on page load
- **Admin login persistence**: Login state stored in `localStorage` with 30-minute inactivity auto-logout, logout button in header, password manager compatibility (proper `<form>`, `autocomplete` attributes)
- **Inline delete confirmation**: Two-step confirm for models, API keys, and request logs replaces native `confirm()` dialogs
- **Test modal improvements**: Cancel button with elapsed timer, chart empty state message, Clone button for providers/models
- **Mobile responsiveness**: Responsive header with wrapping, horizontally scrollable tabs and tables

### Fixed

- **Argo Anthropic response normalization**: Detect and convert OpenAI Chat Completions format responses from Argo's `/v1/messages` endpoint to Anthropic Messages format
- **Model-level `thinking_type` in shim reasoning config** ([#254](https://github.com/Oaklight/llm-rosetta/issues/254), [#256](https://github.com/Oaklight/llm-rosetta/pull/256)): `ReasoningCapability` supports `thinking_type` to force `thinking.type` to `"enabled"` or `"adaptive"`. `ProviderShim` gains `model_reasoning` for per-model overrides keyed by upstream model ID. Argo `claudeopus47 → thinking_type: adaptive` via `model_overrides`. `_normalize_thinking` transform retired — thinking type normalization is now declarative
- **Inline confirm i18n and onclick restore**: Add missing `confirm.sure`/`confirm.yes` translation keys; restore original `onclick` handler after confirmation reverts
- **Reverse proxy caching**: Add `Cache-Control: no-cache, no-store, must-revalidate` on all admin API responses; switch test polling to POST
- **Login overlay loop**: Prevent login overlay from dismissing password manager autofill popups
- **C901 complexity**: Extract `_format_connection_error` helper from `fetch_upstream_models`

### Security

- **Admin login rate limiting**: 5 failed attempts trigger a 5-minute IP lockout

### Changed

- **Settings UI simplified**: Themes reduced to Light/Dark; theme and language selectors moved from header dropdowns into the settings popup

## v0.6.3 — 2026-05-17

### Added

- **Full `custom_tool_call` support for OpenAI Responses API**: Handle the `type: "custom"` tool type end-to-end — request ingestion (coerce to IR `type: "function"` with `_passthrough` for round-trip), response parsing (`custom_tool_call` items with plain-text `input`), and streaming (`response.custom_tool_call_input.delta/done` events). Cross-provider degradation synthesizes a single-string-param JSON Schema so custom tools remain usable on Anthropic/Google
- **`tool_type` field on IR `ToolCallStartEvent`**: Streaming events now carry `tool_type` ("function", "custom", etc.) so converters can emit the correct provider-specific event types
- **Argo shims with `model_id_field` and `upstream_model` alias**: New `argo_openai`, `argo_anthropic`, `argo_google` provider shims that rewrite the model field name for Argo-proxied endpoints. Includes thinking normalization transform for `argo_anthropic`
- **Async server-side test tasks**: Admin panel test requests now run in background tasks, preventing browser connection pool exhaustion on slow models
- **Admin login rate limiting**: Brute-force protection on the admin login endpoint

### Fixed

- **Stored XSS in admin UI**: Escape single quotes in the `esc()` helper to prevent injection via provider/model names
- **`custom_tool_call` streaming type loss in gateway**: `OpenAIResponsesStreamContext.from_base()` now copies `_tool_call_types`, fixing custom tools falling back to `function_call` event types during IR→provider streaming
- **Admin UI regressions**: Fix infinite recursion in fetch models checkbox handler, allow API key editing regardless of `credential_visible` setting, remove prefix real-time preview input lag, fix fetch models prefix losing selections, abort test requests on modal close
- **Reasoning test `max_tokens` too small**: Enforce `budget_tokens >= 1024` for reasoning capability tests
- **httpclient AsyncClient serialization lock**: Update vendored httpclient to v0.4.1, use per-task AsyncClient for test self-calls to avoid deadlock
- **ty type-check errors**: Resolve compatibility issues with ty 0.0.32+

### Changed

- **Admin routes split into subpackage**: Refactored monolithic `routes.py` into `routes/` with dedicated modules for auth, config, keys, observability, and testing
- **CI switched to pre-commit**: Linting now uses `pre-commit run --all-files` (ruff + ty); complexipy suspended pending upstream fix

## v0.6.2 — 2026-05-15

### Added

- **Admin password protection**: `server.admin_password` in config enables a login overlay for the admin panel, using HMAC-based session tokens
- **Credential visibility control**: `server.credential_visible: false` hides API key viewing/copying across the admin UI
- **Provider cascade delete**: Deleting a provider now shows affected models and cascade-deletes them

### Fixed

- **Base URL overwrite**: Switching provider type no longer overwrites user-entered base URLs
- **Request log collapse**: Expanded error detail rows persist across auto-refresh

### Changed

- **Zero-dependency on Python ≥3.11**: Replaced PyYAML with vendored zerodep yaml module

## v0.6.1 — 2026-05-15

### Added

- **`/v1/embeddings` passthrough endpoint**: Proxy embedding requests directly to upstream providers without IR conversion — the OpenAI embeddings format is universal across compatible providers. Includes metrics and request log instrumentation
- **`/v1/models` enriched response**: Model listing now includes `api_standard` (e.g. `"openai_chat"`, `"anthropic"`) and per-model `capabilities` fields
- **"Fetch from Provider" in admin panel**: Query upstream `/v1/models` (or equivalent) endpoint from the Models tab, browse available models with checkboxes, and bulk-add with optional prefix. Already-existing models shown as disabled
- **Model management enhancements**: Provider filter dropdown and model name search in the Models tab
- **Embedding capability and test type**: `embedding` capability in the model editor (mutually exclusive with `vision`/`tools`). Embedding models get a single Test button that POSTs to `/v1/embeddings` and displays dimension count
- **Reasoning capability and test type**: `reasoning` capability with dedicated test that sends `reasoning_effort: "low"`. Mutually exclusive with `embedding`
- **Admin panel tab persistence**: Active tab stored in `localStorage`, survives page refresh

### Fixed

- **Missing event loop in SOCKS5 proxy tests**: Use `asyncio.new_event_loop()` as fallback when prior tests have closed the default event loop
- **Type assertion for httpclient response in fetch_upstream_models**: Resolve ty type-check error for `AsyncClient.get()` return type

## v0.6.0 — 2026-05-15

### Added

- **Provider shim layer with declarative YAML directory**: Shims are now defined as `provider.yaml` + optional `transforms.py` files under `shims/providers/<name>/`, automatically discovered and registered at import time
- **Transform mechanism for provider-specific field adaptation**: Three composable primitives — `strip_fields()`, `rename_field()`, `set_defaults()` — handle field-level differences between a provider's API dialect and its base standard
- **7 new built-in provider shims**: xAI (Grok), Qwen (DashScope), Moonshot (Kimi), MiniMax, Zhipu (GLM), OpenRouter, Volcengine — each with provider-specific transforms where needed
- **Gateway proxy applies shim transforms**: The gateway request/response pipeline now applies `to_transforms` on outbound requests and `from_transforms` on inbound responses and stream chunks
- **Provider logos in admin panel**: Provider shims can declare a `logo` URL (SVG), displayed in the admin panel provider cards
- **SOCKS5 proxy support restored**: Updated vendored `httpclient` from zerodep v0.3.1 to v0.4.0, which includes full SOCKS5 proxy support (RFC 1928/1929, with username/password authentication). Both `--proxy socks5://...` CLI flag and `"proxy": "socks5://..."` config entries now work for all upstream requests

### Changed

- **Shim system refactored to declarative YAML**: Replaced programmatic `builtins.py` with a directory-based system (`shims/providers/*/provider.yaml` + `transforms.py`). Adding a new provider now requires only YAML + optional Python, no changes to core code
- **Vendored `validate` updated to zerodep v0.5.0**: Adds `FieldValidator` and `model_validator` for field-level transform+validate pipelines

### Removed

- **`ModelShim` class removed**: Model-level metadata removed in favor of simpler provider-only shims. The `ProviderShim` dataclass no longer has a `models` field

### Refactored

- **Zero-dependency gateway** ([#178](https://github.com/Oaklight/llm-rosetta/pull/178)): Replaced Starlette + uvicorn + httpx with vendored zerodep `httpserver` and `httpclient` modules. The `[gateway]` extra now has zero external runtime dependencies

### Fixed

- **Deep-merge properties in schema flattening** ([#161](https://github.com/Oaklight/llm-rosetta/issues/161)): Fix `$ref`/`$defs` resolution to deep-merge properties and strip orphaned `required` entries
- **Unconditional usage fallback and StreamContext merge** ([#176](https://github.com/Oaklight/llm-rosetta/pull/176)): Guard against missing usage data and ensure StreamContext state is properly merged

### Known Issues

- **Google tool schema `required` validation** ([#161](https://github.com/Oaklight/llm-rosetta/issues/161)): Some Anthropic tool schemas have `required` entries referencing properties not defined in the schema, causing Google API to reject with `INVALID_ARGUMENT`

## v0.5.3 — 2026-04-25

### Added

- **OpenAI Chat converter: thinking config support** ([#170](https://github.com/Oaklight/llm-rosetta/pull/170)): The OpenAI Chat converter now handles `reasoning_config` in IR requests, mapping to OpenAI's `reasoning_effort` parameter. Enables thinking/extended thinking configuration when routing through the Chat Completions API
- **OpenAI Chat converter: `reasoning_content` field handling**: Non-streaming and streaming responses from reasoning models (e.g., o1, o3) now correctly extract the `reasoning_content` field and convert it to IR `ReasoningPart`, preserving chain-of-thought content during cross-provider conversion
- **Upstream error body in admin request log**: When an upstream provider returns an error, the response body is now included in the admin request log entry, making it easier to diagnose upstream failures without checking server logs
- **Copy entry buttons for providers and models in admin page**: Provider and model entries in the admin panel now have copy/duplicate buttons for quickly creating new entries based on existing configurations

### Fixed

- **`FilePart` excluded from `UserContentPart`** ([#160](https://github.com/Oaklight/llm-rosetta/issues/160), [#162](https://github.com/Oaklight/llm-rosetta/pull/162)): `UserContentPart` union type did not include `FilePart`, causing `validate_ir_request()` to reject any user message containing file content (e.g., PDF attachments sent by Claude Code as Anthropic `document` blocks). The bidirectional conversion logic was already implemented for Anthropic (`document`), Google (`inlineData`), and OpenAI Responses (`input_file`) — only the type definition was missing
- **`google_genai/content_ops.py` unconditional `httpx` import** ([#163](https://github.com/Oaklight/llm-rosetta/issues/163)): Replaced `httpx` with `urllib.request` in the Google GenAI content converter for image URL downloads. `httpx` was only declared as a `[gateway]` optional dependency but was imported unconditionally, causing `ModuleNotFoundError` when installed without `[gateway]` extra
- **Emoji icons replaced with SVG in API key management**: API key action buttons in the admin panel used emoji characters that rendered inconsistently across platforms. Replaced with inline SVG icons and added a key visibility toggle button
- **API key column layout shift**: Fixed CSS layout issue where the API key column width changed when toggling key visibility, causing adjacent buttons to shift position
- **Wheel path glob collision with extras brackets**: Quoted the wheel file path in CI install commands to prevent shell glob expansion when the filename contains `[extras]` bracket syntax

### Refactored

- **SQLite persistence backend**: Replaced the JSONL-based request log and JSON-based metrics persistence with a unified SQLite backend. Provides better write durability, atomic operations, and eliminates log rotation complexity. Vendored `persistdict` from zerodep (v0.4.1) as the key-value storage layer

### CI/Build

- **Install smoke tests**: Added CI smoke tests that verify `pip install` succeeds for both `llm-rosetta` (core) and `llm-rosetta[gateway]` variants, catching missing or circular dependencies early

## v0.5.2 — 2026-04-19

### Fixed

- **Streaming round-trip event inflation** ([#157](https://github.com/Oaklight/llm-rosetta/issues/157)): Fixed multiple scenarios where `Provider A → IR → Provider B` streaming conversion produced more output events than input events:
    - OpenAI Chat, Anthropic, and Google GenAI converters emitted redundant `content_block_end` events when no content block was open, inflating the output stream
    - Google GenAI compound chunks (text + finish in the same SSE frame) triggered duplicate text and finish events. Deferred text/finish payloads via `StreamContext.pending_text` / `pending_finish` so they merge into a single event
    - Tool call events generated spurious `content_block_start` / `content_block_end` wrappers in non-Anthropic targets. Suppressed via `_started` lifecycle guard

### Refactored

- **Unified `stream_response_to_provider` dispatch** ([#157](https://github.com/Oaklight/llm-rosetta/issues/157)): Extracted identical dispatch logic (10-entry `_TO_P_DISPATCH` table + dispatch skeleton) from all 4 provider converters into `BaseConverter`. Each converter now only implements a provider-specific `_post_process_to_provider` hook (OpenAI Chat injects envelope fields; OpenAI Responses injects `sequence_number`). Net reduction: ~27 lines
- **`StreamContext` buffer convenience methods**: Added `buffer_usage()` / `pop_pending_usage()` / `buffer_finish()` / `pop_pending_finish()` to replace manual set-and-clear patterns across all converters

### Changed

- **Pinned dev tooling versions**: `ty>=0.0.31` and `ruff>=0.15.0` now declared in `pyproject.toml` dev dependencies. CI no longer installs them separately — uses versions from `pip install -e ".[all]"`
- **Converter tests added to CI**: `tests/converters/` (1086+ tests) now runs in GitHub Actions alongside `tests/test_types/`
- **Roundtrip inflation regression test**: New pytest-parametrized test suite (`tests/converters/test_roundtrip_inflation.py`, 15 cases) verifies `len(output_events) <= len(input_events)` for all 4 providers across text, reasoning, tool call, and compound scenarios

## v0.5.1 — 2026-04-15

### Added

- **`tool_ops` convenience API** ([#148](https://github.com/Oaklight/llm-rosetta/issues/148)): New top-level `llm_rosetta.tool_ops` module for standalone tool definition conversion without instantiating full converter pipelines. Provides `to_provider()` / `from_provider()` unified dispatch and per-provider shortcuts (`to_openai_chat()`, `to_anthropic()`, etc.). All imports are lazy
- **Multi-key API management**: Admin panel now supports multiple API keys per gateway with per-key labels, create/reveal/delete operations, and usage tracking in request logs
- **Gateway API key authentication**: Configurable API key (`server.api_key`) protects AI request endpoints (`/v1/*`). Supports format-native credential extraction — OpenAI `Authorization: Bearer`, Anthropic `x-api-key`, Google `x-goog-api-key` / `?key=` query param. When no key is configured, all requests pass through (backward compatible)
- **Provider enable/disable**: Each provider now supports an `enabled` field (default `true`). Disabled providers and their models are silently excluded from routing
- **Docker support**: Official `Dockerfile`, `docker-compose.yml`, and Makefile targets (`build-docker`, `push-docker`, `run-docker`) for containerized deployment. Alpine-based image with non-root user, config volume mount, and PUID/PGID support
- **Admin panel enhancements**:
    - Provider toggle switches (enable/disable without deleting)
    - Model search and column sorting
    - Provider rename with automatic model reference updates
    - Network diagnostics button (connectivity check + proxy test)
    - Model testing with collapsible raw request/response details and image preview for vision tests
    - Embedded test image (base64 data URI) to avoid external network downloads
    - `reasoning_effort: 'low'` for reasoning model tests to limit token budget

### Changed

- **Admin panel authentication removed from gateway**: Admin panel endpoints (`/admin/*`) no longer require the gateway API key. Admin access control is delegated to the reverse proxy (e.g. Caddy, Nginx). The gateway API key now only authenticates AI request endpoints (`/v1/*`)
- **C901 cyclomatic complexity enforced at threshold 15**: Progressive reduction from 25 → 20 → 15 across all converters and gateway modules. Extracted cross-provider consistency helpers (`_build_ir_usage`, `_build_provider_usage`, `_convert_tools_from_p`, `_apply_tool_config`) with identical names across all 4 converters
- **`BaseConverter` abstract methods**: Four new abstract methods formalize the cross-provider helper pattern. Preserve-mode hooks documented as convention for providers supporting lossless round-trip
- **Vendored `validate.py` updated to zerodep v0.4.2**: Internal refactor of monolithic `_validate()` into focused helpers; no functional changes

### Fixed

- **User-Agent header for image URL downloads**: Google GenAI content converter now sends `User-Agent: llm-rosetta/1.0 (image fetch)` when downloading image URLs for inline base64 conversion, preventing 403 Forbidden from servers like Wikimedia
- **Image URL download with proxy support**: Image downloads in the Google GenAI converter now respect `HTTPS_PROXY` / `HTTP_PROXY` environment variables
- **Empty content fallback for reasoning models**: Admin panel test results now correctly handle `content: ""` (from reasoning models where all `max_tokens` are consumed by reasoning tokens) instead of showing raw JSON
- **Config file not found error**: Gateway now shows a friendly error message when the config file doesn't exist, instead of a Python traceback
- **ty type checker compatibility**: Added `ty: ignore` annotations for TypedDict vs `dict[str, Any]` mismatches and `FinishReason` Literal type narrowing
- **Google converter crash when thinking consumes all tokens** ([#152](https://github.com/Oaklight/llm-rosetta/issues/152)): Gemini 2.5 Pro with small `max_tokens` could have all tokens consumed by thinking, producing a response with no content parts. The converter now falls back to an empty assistant message instead of failing IR validation

## v0.5.0 — 2026-04-12

### Added

- **Gateway Admin Panel**: Built-in web admin panel at `/admin/` for managing gateway configuration, monitoring traffic, and inspecting request logs without editing config files or restarting the server
    - **Configuration tab**: Visual management of providers (add, edit, rename, delete) and model routing with capabilities (text/vision/tools)
    - **Dashboard tab**: Real-time metrics with summary cards (total requests, error rate, active streams, uptime), rolling 60-second throughput and latency charts, per-provider breakdown
    - **Request Log tab**: Filterable request log with model, provider, and status filters, paginated view with color-coded status codes
    - **8 themes**: Light, Indigo Dark, Dracula, Nord, Solarized, Osaka Jade, One Dark, Rosé Pine — persisted in localStorage
    - **i18n**: English and Chinese language support with localStorage persistence
- **File-based persistence**: Metrics counters (JSON) and request log (JSONL) are automatically saved to disk alongside the config file. Data survives server restarts. Log rotation with gzip compression (2 MB limit, 3 backups)
- **Provider rename**: Renaming a provider automatically updates all model routing references
- **API key security**: Masked keys on provider cards, reveal-on-demand with visibility toggle and copy button in edit modal. Masked values are never written back to config

### Changed

- **Provider names decoupled from API standard types**: Provider names are now user-defined strings (e.g. `"my-openai"`, `"OpenRouter_anthropic"`) instead of being constrained to the 4 standard type identifiers. A separate `type` field specifies the API standard (`openai_chat`, `openai_responses`, `anthropic`, `google`)
- Extracted `write_config()` to `config.py` for shared use by CLI and admin panel

## v0.4.2 — 2026-04-11

### Changed

- **`ReasoningConfig.enabled` replaced with `mode` field**: The boolean `enabled` field has been replaced by `mode: Literal["auto", "enabled", "disabled"]`. This aligns the IR more closely with provider semantics (Anthropic's three-way `thinking.type`, OpenAI Responses' `reasoning.type`). Omitting `mode` retains the previous "provider default" behavior. The `effort` field now lives directly in `ReasoningConfig` rather than being nested

### Fixed

- **Responses API `developer` role mapping**: The OpenAI Responses API uses `role: "developer"` (equivalent to Chat's `"system"`). Previously this role was passed through to IR unchanged, causing validation failures. Now correctly mapped to IR `"system"` during Provider→IR conversion
- **Google GenAI `additionalProperties` rejection**: Google's function_declarations API rejects the `additionalProperties` JSON Schema keyword. Added `extra_strip_keys` parameter to `sanitize_schema()` so providers can strip provider-specific unsupported keywords. Google tool_ops now strips `additionalProperties` recursively from nested schemas
- **Google GenAI `prompt_tokens_details` format mismatch**: Google returns modality token details as `list[ModalityTokenCount]` (e.g. `[{"modality": "TEXT", "token_count": 42}]`) but IR expects `dict[str, int]` (e.g. `{"text_tokens": 42}`). Added bidirectional conversion helpers `_modality_list_to_dict()` and `_dict_to_modality_list()`. Handles both SDK (`token_count`) and REST API (`tokenCount`) field names
- **Cross-format tool call ID prefix mapping**: The Responses API enforces `fc_` prefix on tool call IDs, but Chat uses `call_` and Anthropic uses `toolu_`. Added automatic prefix mapping during Responses conversion to prevent validation failures in cross-format scenarios
- **Adaptive thinking fallback**: When converting IR reasoning config to Anthropic format, `mode: "enabled"` without `budget_tokens` now correctly falls back to `{"type": "adaptive"}` with a warning, instead of producing an invalid `{"type": "enabled"}` without the required `budget_tokens`

## v0.4.1 — 2026-04-10

### Added

- **`force_conversion` parameter for `convert()`**: New `force_conversion: bool = False` keyword-only parameter. When `True`, the full source→IR→target pipeline runs even when source and target providers match, ensuring parameter normalization (e.g. `max_tokens` → `max_completion_tokens` for OpenAI Chat). Default `False` preserves existing passthrough behavior

### Fixed

- **Vendored `validate.py` updated from zerodep v0.4.1**: Applied pyupgrade fixes — `Callable` imported from `collections.abc` instead of `typing` (UP035), `@functools.cache` replaces `@functools.lru_cache(maxsize=None)` (UP033)
- Removed unused `sys` import in benchmark script
- Applied `ruff format` to benchmark scripts

### Changed

- Removed incorrect "Related Projects" section from README — LLM-Rosetta is an independent project, not part of the ToolRegistry ecosystem

## v0.4.0 — 2026-04-09

### Added

- **Metadata preservation for lossless A→IR→A round-trip** (#60, PR #119): New `MetadataMode` (`"strip"` / `"preserve"`) option in `ConversionContext` that captures provider-specific fields during `from_provider` and re-injects them during `to_provider`, enabling lossless round-trip conversion. Helper methods on `ConversionContext`: `store_request_echo()`, `store_response_extras()`, `store_output_items_meta()`, `get_echo_fields()`, `get_output_items_meta()`. Per-provider coverage:
    - **OpenAI Responses**: captures/restores 28+ echo fields (temperature, tools, reasoning, truncation, etc.), per-output-item metadata (id, status, annotations, logprobs), `RESPONSES_REQUIRED_DEFAULTS` dict for spec-required fields with sensible defaults, `sequence_number` on all SSE events
    - **Anthropic**: preserves `stop_sequence`, `container`, citations, and OpenRouter extension usage fields
    - **OpenAI Chat**: now re-emits `refusal` and `annotations` fields in `response_to_provider` (previously dropped)
    - **Google GenAI**: preserves `promptTokensDetails` and `cachedContentTokenCount` in usage metadata
    - **Gateway**: automatically enables preserve mode for both streaming and non-streaming paths; bridges metadata between `from_ctx` and `to_ctx` during streaming

### Fixed

- **Open Responses spec compliance for streaming and non-streaming**: Added required fields to all SSE events (`item_id`, `logprobs`, `annotations`, `status`, `sequence_number`, `output_index`, `content_index`), usage detail breakdowns (`output_tokens_details`, `input_tokens_details`), message item IDs and status for non-streaming output items, `function_call` status field in tool_ops, `service_tier` default to `"default"` (string, not null per spec), `completed_at` in required defaults, `created_at` fallback to current time when not provided, normalized echoed tools with `strict: null`, and metadata bridging from `from_ctx` to `to_ctx` in gateway streaming. All 6 Open Responses compliance tests now pass (schema + semantic)

## v0.3.1 — 2026-04-07

### Fixed

- **`service_tier: None` and `system_fingerprint: None` causing validation errors** (PR #118): OpenAI upstream returns these fields as `null`, but the existence check (`if "key" in dict`) passed and assigned `None` to IR's `NotRequired[str]` field. Changed to value-not-None check in both OpenAI Chat and OpenAI Responses converters. Discovered via [Oaklight/argo-proxy#99](https://github.com/Oaklight/argo-proxy/issues/99)
- **Base `StreamContext` missing provider-specific attributes in Responses streaming** (PR #118): When a gateway passes a base `StreamContext` to `OpenAIResponsesConverter.stream_response_to_provider()`, the method accesses `accumulated_text`, `output_item_emitted`, etc. that only exist on `OpenAIResponsesStreamContext`. Added auto-upgrade via `from_base()` classmethod with metadata caching to preserve state across calls

## v0.3.0 — 2026-04-07

### Added

- **Multimodal tool result support across all 4 converters** (#92, PR #109): Tools can now return multimodal content (text + images + files) as `ToolResultPart.result`. Three providers (Anthropic, OpenAI Responses, Google GenAI) support this natively; content blocks are converted through each provider's `content_ops` layer. See provider support matrix below
- **Lossless multimodal tool result roundtrip for OpenAI Chat** (#92, PR #108): OpenAI Chat Completions only accepts `content: string` for tool messages. Implements a dual encoding strategy — tool message keeps `json.dumps(result)` as data fallback, plus a synthetic user message carries visual content (`image_url` parts) wrapped in `<tool-content call-id="...">` XML tags. Unpacking recovers multimodal structure from the synthetic message (preferred) or falls back to JSON parsing if the synthetic message was trimmed by agent frameworks
- **`extract_all_text()` helper function** (PR #109): Extracts text from both `TextPart` and `ReasoningPart` content — useful for thinking models (e.g. gemini-2.5-flash) that may place answers in reasoning parts rather than text parts
- **`generate_chart` example tool** (PR #109): New multimodal tool in `examples/tools.py` returning `[TextPart, ImagePart]` with inline base64 PNG, plus `multimodal_tools_spec` combining all 3 example tools
- **Multimodal integration tests across all 4 provider SDKs** (PR #109): Two new test scenarios per provider — (A) tool returning multimodal content (text + image), (B) image input combined with tool calls. All 30 tests pass against official APIs: OpenAI Chat 9/9, OpenAI Responses 6/6, Anthropic 8/8, Google GenAI 7/7
- **Runtime IR validation via vendored zero-dependency validator** (#91): `validate_ir_request()`, `validate_ir_response()`, and `validate_ir_messages()` utilities validate IR structures against their TypedDict definitions at runtime. All 4 converters now validate output in `request_from_provider()` and `response_from_provider()`. Replaces manual `BaseMessageOps.validate_messages`. Includes Python <3.11 compatibility for `typing_extensions.TypedDict`
- **Constants validation tests**: 39 new tests across 4 `test_constants.py` files verifying that all reason mapping values are valid IR finish reasons, mapping coverage is complete, event type constants are well-formed, and ID generation produces correct formats
- **Finish reason mapping test coverage**: 38 tests validating reason mapping correctness as a safety net for the constants refactoring
- **`ConversionContext` base class for conversion pipelines** (#106, PR #111): New `ConversionContext` dataclass with `warnings: list[str]`, `options: dict[str, Any]`, and `metadata: dict[str, Any]` — a structured context container for non-streaming conversions. New `BaseConverter.create_conversion_context(**options)` factory method mirrors the existing `create_stream_context()`. All 6 non-streaming `BaseConverter` methods now accept an optional `context: ConversionContext` keyword parameter; converter implementations sync warnings to `context.warnings`. Gateway proxy creates a shared context per request and passes it through the full source→IR→target→response pipeline

### Fixed

- **Contextual error messages for tool conversion failures** (#85, PR #110): When `p_tool_definition_to_ir()` fails on a malformed or unsupported tool definition, the `ValueError` now includes `type=` and `name=` context so users can identify which tool caused the issue. Applied to all 4 converters (OpenAI Chat, OpenAI Responses, Anthropic, Google GenAI) with unit tests
- **OpenAI Responses `tool_choice` format** (PR #109): Was using Chat Completions format (`{"type": "function", "function": {"name": "..."}}`); now uses Responses format (`{"type": "function", "name": "..."}`)
- **OpenAI Responses tool call ID round-trip** (PR #109): Responses API uses `fc_` prefix IDs while IR uses `call_` prefix. The Responses `id` is now preserved in `provider_metadata` separately from `call_id`, enabling lossless round-trip conversion
- **OpenAI Responses reasoning item round-trip** (PR #109): Reasoning models (e.g. gpt-5-nano) emit reasoning items with `id` (rs_ prefix), structured `summary` arrays, and `encrypted_content`. These are now preserved through `provider_metadata` for lossless round-trip — fixes 400 errors when reasoning items were sent back without their original `id`
- **IR validation accepts `None` for optional response fields** (PR #109): `logprobs` and `system_fingerprint` in `IRResponse` now accept `None` values (previously only accepted missing keys)
- **OpenAI Responses `content_filter` finish reason mapped to wrong status** (#90): `content_filter` was incorrectly mapped to `"completed"` status in `response_to_provider` and `stream_response_to_provider`. Now correctly maps to `"incomplete"` status with `incomplete_details.reason = "content_filter"`
- **Anthropic streaming missing `refusal` reason mapping**: The streaming `reason_map` was missing the `refusal` entry present in the non-streaming path, causing Anthropic refusal stop reasons to be silently dropped during streaming. Fixed as a side effect of the constants extraction (#64) — both paths now share the same `ANTHROPIC_REASON_FROM_PROVIDER` dict

### Changed

- **`ReasoningConfig.effort` expanded to 5-level enum** (#100): Effort levels now include `"minimal"`, `"low"`, `"medium"`, `"high"`, `"max"`. Provider-specific mappings: Anthropic maps to `thinking.type="adaptive"` with `thinking.effort`; OpenAI Chat/Responses clamp `"minimal"`→`"low"` and `"max"`→`"high"` (with warnings); Google GenAI maps to `thinking_config.thinking_level`
- **`ReasoningConfig.type` replaced with `ReasoningConfig.enabled`** (#70): The `type: Literal["enabled", "disabled"]` field is replaced with `enabled: bool` to avoid shadowing the Python built-in `type` and provide a more natural API
- **Merged duplicate IR concepts** (#69): Removed `candidate_count` from `GenerationConfig` — use `n` instead (Google GenAI converter maps `n` ↔ `candidate_count` internally). Unified `system_instruction` type from `str | list[dict]` to `str`
- **Normalized `ImagePart`, `FilePart`, `AudioPart` to canonical forms** (#68): Each part now has exactly two canonical forms — URL reference + structured inline data (e.g. `image_data`) — plus a unified `provider_ref: dict[str, Any]` for provider-specific references. Removed redundant top-level `data`/`media_type` fields and replaced `file_id`/`audio_id` with `provider_ref`
- **IR type fields changed from `Iterable` to `list`; function parameters to `Sequence`** (#67): TypedDict fields now use `list` for indexable, serialization-friendly semantics; function parameters use `Sequence` (covariant, read-only). Also fixes a latent generator-consumption bug in `strip_orphaned_tool_config`
- **`StreamContext` now inherits from `ConversionContext`** (#106, PR #111): `StreamContext` is a subclass of `ConversionContext` (IS-A relationship), unifying the context model for streaming and non-streaming paths. File renamed: `base/stream_context.py` → `base/context.py`
- **`StreamContext` converted to dataclass with provider subclass** (#65): `StreamContext` is now a `@dataclass` with typed fields (eliminates defensive `getattr`/`hasattr` patterns). OpenAI Responses-specific state extracted into `OpenAIResponsesStreamContext` subclass. New `BaseConverter.create_stream_context()` factory method

### Refactored

- **Warnings single-source convergence** (#113, PR #115): All 4 converter `request_to_provider` methods now use `ConversionContext` as the single accumulation point for warnings. Eliminates the dual-write pattern where warnings were written to both a local list and `context.warnings`. The returned warnings list IS the same object as `context.warnings` — no duplication possible
- **`ProviderMetadataStore` replaces global metadata cache** (#112, PR #117): The module-level `_provider_metadata_cache` dict in `proxy.py` is replaced with `ProviderMetadataStore` — a class with TTL-based expiration (30 min), max-size eviction (10k entries), and explicit lifecycle management. The store is created per-app in `create_app()` and passed via `app.state`, eliminating implicit global mutation. `close_clients()` renamed to `close_resources()` to also clear the store on shutdown
- **Shrink public API export surface** (#114, PR #116): Reduced `__all__` exports across converter packages to only the primary converter class, removing internal implementation details (`*MessageOps`, `*ContentOps`, `*ConfigOps`, `*ToolOps`, `*Constants`) from the public API. Internal modules remain importable for advanced use but are no longer promoted as public surface
- **Extracted stream event handlers from monolithic methods** (#63): Replaced 8 monolithic `if`/`elif` stream methods (~1,781 lines) across all 4 converters with individual handler methods dispatched via class-level handler tables. Public API unchanged
- **Extracted shared utility functions in OpenAI Responses converter** (#66): `resolve_call_id()` and `build_message_preamble_events()` extracted from `converter.py` into `utils.py` with dedicated unit tests
- **Extracted per-provider constants for reason mappings and magic values** (#64): Inline reason mapping dicts, SSE event type string literals, status-to-reason conditional logic, and ID generation patterns across all 4 converters are now centralized in per-provider `_constants.py` modules. Includes `AnthropicEventType` and `ResponsesEventType` classes, `REASON_FROM_PROVIDER` / `REASON_TO_PROVIDER` dicts, and `generate_tool_call_id()` / `generate_message_id()` helpers

## v0.2.6 — 2026-03-29

### Fixed

- **Chat Completions tool message ordering after Responses API conversion** *([@caidao22](https://github.com/caidao22))*: Codex CLI interleaves `function_call_output` with other items (e.g. user warnings) in Responses API format — valid there since items match by `call_id`. But after IR → Chat Completions conversion, the interleaved messages break the OpenAI Chat API constraint that `role: "tool"` messages must immediately follow their `assistant` `tool_calls`, causing upstream 400 errors. Added `_reorder_tool_messages()` post-processing in `OpenAIChatMessageOps.ir_messages_to_p()` that groups tool responses back to their corresponding assistant messages
- **Orphaned `tool_choice`/`tool_config` stripped when no tools defined** *([@caidao22](https://github.com/caidao22))*: Codex context compaction can drop all tool definitions while keeping `tool_choice` (e.g. `"auto"`), causing upstream APIs to reject with *"tool_choice is set but no tools are provided"*. Added `strip_orphaned_tool_config()` in all four converters — part of the same Codex compaction fix family as `fix_orphaned_tool_calls_ir` (orphaned tool_call/result pairing) and `_reorder_tool_messages` (tool message ordering). Also extended `fix_orphaned_tool_calls_ir` to Google GenAI converter for completeness (#87)
- **Stream event ordering**: `UsageEvent` is now emitted before `FinishEvent` in all four provider converters (OpenAI Chat, OpenAI Responses, Anthropic, Google GenAI). Previously `FinishEvent` was processed first, causing `response.completed` to carry `output_tokens=0` — downstream consumers (e.g. Codex token tracking) saw stale usage data. For cross-chunk scenarios (OpenAI Chat sends `finish_reason` and `usage` in separate chunks), `FinishEvent` now defers `response.completed` to `StreamEndEvent` which merges any pending usage
- **Parallel tool calls merged into one in Anthropic/Google → Chat streaming**: Anthropic and Google GenAI `stream_response_from_provider` emitted `ToolCallStartEvent` and `ToolCallDeltaEvent` without `tool_call_index`. When routing to Chat Completions, all parallel tool calls defaulted to index 0, causing the client SDK to merge them into a single call. Anthropic now derives `tool_call_index` from `context._tool_call_order` position; Google computes it from registration order in context (#88, #89)
- **Missing `id` field on Responses `function_call` output**: Non-streaming `response_to_provider` was missing the `id` field on `function_call` output items. Streaming used a synthetic `fc_` prefix that could leak into IR via `p_tool_call_to_ir` fallback path. Unified both paths to use `call_id` directly as `id` (no prefix)
- **Responses streaming `item_id` and empty `tool_call_id` resolution** *([@caidao22](https://github.com/caidao22))*: Added `item_id` tracking to `StreamContext` (`tool_call_item_id_map`, bidirectional mapping). Responses `stream_response_to_provider` now emits `item.id` on `output_item.added` and `item_id` (not `call_id`) on `function_call_arguments.delta/done` events. Defense-in-depth: resolves empty `tool_call_id` by `tool_call_index` via context (#86)
- **Non-function tool names mangled with type prefix** *([@caidao22](https://github.com/caidao22))*: Non-function IR tool definitions (e.g. `type="custom"`, `name="apply_patch"`) were converted with a type prefix (`custom_apply_patch`), breaking tool_call matching since the client expects the original name. Both OpenAI Chat and Responses converters now use `ir_tool["name"]` directly (#84)

## v0.2.5 — 2026-03-23

### Fixed

- **Anthropic `input_schema` missing `type` for parameterless tools**: MCP tools with no parameters produce `input_schema: {}`, but Anthropic requires `"type"` to be present. Now defaults to `{"type": "object"}` when the schema dict lacks a `type` field — fixes `tools.0.custom.input_schema.type: Field required` errors when routing Google GenAI or OpenAI Responses tool calls to Anthropic upstream
- **Google GenAI camelCase field handling across the full converter stack**: Gemini CLI and the Google REST API use camelCase (`inlineData`, `fileData`, `mimeType`, `fileUri`, `functionCall`, `functionResponse`, `finishReason`, `usageMetadata`, `responseMimeType`, `responseSchema`, `thinkingConfig`, `maxOutputTokens`, `stopSequences`, etc.), but the converter only accepted snake_case. All P→IR methods in content_ops, config_ops, tool_ops, message_ops, and converter now accept both conventions; all IR→P methods now output camelCase for REST API compatibility
- **Image/audio/file data lost during Google→IR conversion**: `p_part_to_ir` checked for `inline_data` (snake_case) but Gemini CLI sends `inlineData` (camelCase) — binary content was silently dropped with a `不支持的Part类型` warning. Fixed by normalizing camelCase keys at the dispatch entry point
- **Cross-format image conversion failure (Google → OpenAI/Anthropic)**: Google's `p_image_to_ir` produces `ImagePart` with top-level `data` + `media_type` fields, but OpenAI Chat, Anthropic, and OpenAI Responses `ir_image_to_p` only checked `image_url` and nested `image_data` — threw `ValueError`. All three target converters now handle top-level fields as a fallback path (#68)
- **Google GenAI tool_call_id reconciliation**: Google `functionCall` has no ID field, so UUIDs are generated during P→IR. But Gemini CLI assigns its own IDs to `functionResponse` (format: `name_timestamp_index`), creating a mismatch. New `_reconcile_tool_call_ids` method matches tool results to tool calls by function name, fixing orphaned tool_call errors
- **tool_call_id exceeds OpenAI 40-character limit**: Generated IDs used `call_{name}_{8hex}` format — MCP tool names like `mcp_toolregistry-hub-server_datetime-now` produced 54-char IDs. Shortened to `call_{24hex}` (fixed 29 chars)
- **Google→IR role mapping for tool results**: `functionResponse` parts produced `role: "user"` IR messages, so `fix_orphaned_tool_calls_ir` (which checks `role: "tool"`) couldn't detect them. Now separates `functionResponse` into `role: "tool"` messages with explicit `"tool": "user"` in `_IR_TO_GOOGLE_ROLE`
- **Mixed content message ordering**: When a Google message contains both `functionResponse` and `inlineData`, the content parts were emitted before tool results, breaking OpenAI's required `assistant(tool_calls) → tool(response)` ordering. Tool results now precede content parts in the split
- **Google built-in tools (googleSearch, codeExecution)**: `p_tool_definition_to_ir` now returns `None` for tool entries without a `name` field; converter skips them instead of producing empty `function.name` errors
- **Gateway: Starlette `on_shutdown` deprecation**: Replaced deprecated `on_shutdown` parameter with `lifespan` async context manager — fixes compatibility with Starlette 0.38+ which removed `on_shutdown`/`on_startup`

### Added

- **StreamContext**: `get_tool_call_args()` and `get_pending_tool_calls()` methods for querying accumulated tool call state during streaming

### Changed

- **`BaseToolOps.p_tool_definition_to_ir` return type**: Now `ToolDefinition | list[ToolDefinition] | None` to support unconvertible tool entries

### Added (Documentation)

- **Provider & CLI Compatibility Matrix**: New guide page documenting real-world issues found during live integration testing with Gemini CLI, Claude Code, and OpenCode through format-converting proxies

## v0.2.4 — 2026-03-22

### Added

- **`fix_orphaned_tool_calls()` utilities**: Public functions in `converters/openai_chat/tool_ops.py`, `converters/openai_responses/tool_ops.py`, and `converters/anthropic/tool_ops.py` that detect mismatched tool calls/results and fix them bidirectionally — injecting synthetic placeholder results for orphaned calls **and** removing orphaned results without matching calls. OpenAI (Chat & Responses) and Anthropic strictly require this pairing (return 400 otherwise); only Google Gemini is lenient. Automatically applied at the IR level during `request_to_provider()` for all strict-pairing converters; emits `WARNING`-level log when orphaned tool calls or results are detected (#82, #84)

### Fixed

- **Anthropic→IR role normalization for `tool_result` messages**: Anthropic places `tool_result` blocks in `role: "user"` messages, but IR uses `role: "tool"` (like OpenAI). The Anthropic converter now normalizes pure `tool_result` user messages to `role: "tool"`, and splits mixed `tool_result` + text messages into separate `role: "tool"` and `role: "user"` IR messages. This fixes `fix_orphaned_tool_calls_ir()` failing to detect answered tool calls in cross-format conversions (e.g. Anthropic → OpenAI Chat) (#84)
- **OpenAI Responses→IR role normalization for `function_call_output` items**: `function_call_output` and `mcp_call_output` items were grouped into `role: "user"` IR messages, but IR uses `role: "tool"` for tool results. The Responses converter now groups these items into `role: "tool"` messages, fixing `fix_orphaned_tool_calls_ir()` failing to detect answered tool calls when converting Responses → other formats (e.g. Responses → OpenAI Chat) (#84)

### Added (Documentation)

- **Provider Dialect Differences guide**: New section in the Converters guide (EN + ZH) documenting tool schema sanitization, orphaned tool call handling, and Google camelCase/snake_case differences

## v0.2.3 — 2026-03-22

### Fixed

- **Tool schema sanitization applied to all converters**: `_sanitize_schema()` was previously only called in the OpenAI Chat converter. Google GenAI, OpenAI Responses, and Anthropic converters now also sanitize tool parameter schemas before sending to upstream, preventing rejections from strict endpoints like Vertex AI (#80)
- **Non-standard `ref` and `$schema` keywords stripped**: OpenCode's built-in tools use a bare `ref` field (without `$` prefix) and `$schema` at the top level, both rejected by Vertex AI. Added to the unsupported keywords blocklist (#80)
- **`$ref`/`$defs` resolved by inlining**: JSON Schema `$ref` references are now resolved by inlining the referenced definition from `$defs`/`definitions`, and both keys are removed from the output. Supports nested and chained references (#80)
- **Streaming tool call arguments not accumulated**: OpenAI Chat, Anthropic, and Google GenAI converters registered tool calls in `StreamContext` but never called `append_tool_call_args()` to accumulate argument deltas during streaming. This caused tool call arguments to arrive empty at upstream (e.g., MCP tools returning `'query' is a required property`). Only the OpenAI Responses converter was correct (#81)
- **OpenAI Chat streaming tool call ID resolution**: Delta-only chunks (carrying `index` but no `id`) produced an empty-string `tool_call_id`. Now resolves the effective ID from `StreamContext._tool_call_order` using the chunk index (#81)

### Changed

- **`sanitize_schema` extracted to `converters/base/tools.py`**: The schema sanitization utility (previously `_sanitize_schema` private to `openai_chat/tool_ops.py`) is now a public shared function in `converters/base/tools.py`, exported via `converters.base`. All 4 converter `tool_ops.py` files import from the shared location instead of cross-importing from `openai_chat` (#66)

## v0.2.2 — 2026-03-22

### Fixed

- **Missing `content_block_stop` in Anthropic SSE output**: When converting OpenAI Chat streaming responses to Anthropic SSE format, `content_block_stop` events were not emitted before `message_delta`, causing Claude Code to silently discard response content. The Anthropic converter now emits `content_block_stop` for any open content block when processing a `FinishEvent` (#77)
- **Upstream preflight chunk misinterpreted as stream end**: Argo API sends a preflight chunk with `choices: []` and empty `id`/`model` before actual content. The OpenAI Chat converter now only treats empty-choices chunks as stream-end after the stream has actually started (`context.is_started` guard) (#77)

## v0.2.1 — 2026-03-20

### Added

- **Gateway request/response body logging**: configurable debug logging with colorized output, body sanitization and truncation — enable via config (`"debug": {"verbose": true, "log_bodies": true}`), env vars (`LLM_ROSETTA_VERBOSE`, `LLM_ROSETTA_LOG_BODIES`), or `--verbose` CLI flag
- **Google `output_format="rest"` for `request_to_provider()`**: pass `output_format="rest"` to get a REST API–ready request body with `tools`/`tool_config` at top level and generation params wrapped in `generationConfig` — eliminates the need for manual SDK→REST fixups

### Changed

- **Gateway modularization**: split `app.py` (1057 lines) into `proxy.py` (proxy engine, SSE handling, upstream requests), `cli.py` (CLI entry point, argparse, subcommands), and a slimmed `app.py` (route handlers, app factory, ~210 lines)
- **Moved Google REST body fixup to core**: `_fixup_google_body()` logic moved from `gateway/proxy.py` into `GoogleGenAIConverter._to_rest_body()`, removing duplicated SDK→REST transforms from the gateway and all 6 REST examples

### Fixed

- OpenAI Responses streaming: added missing `id`/`object`/`model` fields to `response.completed`, `output_index`/`content_index` to text delta events, and proper lifecycle events (`output_item.added`, `content_part.added`, `content_part.done`, `output_item.done`) (#56)
- OpenAI Chat streaming: `tool_calls` entries now always include the required `index` field, defaulting to `0` when not explicitly provided by the upstream IR event (#57)
- OpenAI Chat streaming: usage-only chunk now includes `"choices": []` to satisfy clients that validate every `chat.completion.chunk` must contain a `choices` array (#55)
- `stream_options` (Chat Completions-only field) no longer leaks into OpenAI Responses API requests — the Responses converter's `ir_stream_config_to_p()` was incorrectly emitting `stream_options`, causing upstream rejection when Chat-format clients (Kilo, OpenCode) were proxied to the Responses API (#58)
- Google GenAI converter now handles tools and tool_config in REST-format requests (top-level fields) in addition to SDK format (`config.tools`) — previously only SDK format was recognized, silently stripping tool definitions from gateway-proxied requests (#59)
- Google camelCase `functionDeclarations` not parsed: `p_tool_definition_to_ir()` now handles both `functionDeclarations` (camelCase/REST) and `function_declarations` (snake_case/SDK), and extracts all declarations instead of only the first. Also added camelCase support for `functionCallingConfig`/`allowedFunctionNames` and `toolConfig` in request parsing — fixes Gemini CLI tool calling through the gateway (#61)
- Google streaming tool calls split into two chunks: `stream_response_to_provider()` now defers `tool_call_start` and emits the complete `function_call` (name + args) in a single chunk on `tool_call_delta`, matching the Google API's native format (#62)

## v0.2.0 — 2026-03-18

### Added

- **Standalone API test scripts** (`llm_api_simple_tests/`): 20 test scripts (5 per provider) using official SDKs directly, covering simple query, multi-round chat, image, function calling, and comprehensive scenarios — added as a git submodule from [Oaklight/llm_api_simple_tests](https://github.com/Oaklight/llm_api_simple_tests)
- **LLM-Rosetta Gateway**: REST gateway application for cross-provider HTTP proxying
- CLI entry point (`llm-rosetta-gateway`) and package structure for the gateway
- Gateway config auto-discovery at `./config.jsonc`, `~/.config/llm-rosetta-gateway/config.jsonc`, `~/.llm-rosetta-gateway/config.jsonc`
- `--edit` / `-e` flag to open config file in `$EDITOR` (falls back to nano/vi/vim)
- `--version` / `-V` flag showing current version
- ASCII art startup banner with `--no-banner` to suppress
- `add provider <name>` subcommand for adding provider entries to config (with `--api-key`, `--base-url` flags or interactive prompts; known providers auto-fill defaults)
- `add model <name>` subcommand for adding model routing entries (with `--provider` flag or interactive prompt)
- **Gateway providers module** (`providers.py`): centralized provider definitions with auth-header builders, URL templates, default base URLs, and API key env-var names
- **API key rotation**: round-robin `KeyRing` for comma-separated API keys per provider
- **Proxy support**: global `server.proxy` and per-provider `proxy` config for HTTP/SOCKS proxies; CLI `--proxy` flag overrides config
- Makefile `test-integration` target using `proxychains` (if available) for integration tests
- `init` subcommand to create a template `config.jsonc` at the XDG default location (`~/.config/llm-rosetta-gateway/`)
- **Model listing endpoints**: `GET /v1/models` (compatible with both OpenAI and Anthropic SDKs) and `GET /v1beta/models` (Google GenAI SDK format) — enables `client.models.list()` across all three SDKs (#54)

### Changed

- Bumped minimum Python to 3.10+; migrated to stdlib `typing` (removed `typing_extensions`)
- Applied `ruff` formatter across the entire codebase
- Updated Makefile with `lint`, `test`, and `build` targets
- Added `ty` (type checker) configuration
- Configured `ruff` lint rules (`E`, `F`, `UP`) in `pyproject.toml`; ignore `UP007` (Union syntax) and `E501` (line length)
- Modernized typing imports across `src/`, `tests/`, `examples/`, and `scripts/` — replaced `typing.Dict`, `List`, `Tuple`, `Optional`, `Type` with stdlib builtins

### Fixed

- Streaming crash with Anthropic provider when usage tokens are `null` — `TypeError: NoneType + int` in all converters (replaced `.get("*_tokens", 0)` with `.get("*_tokens") or 0`)
- Gateway provider `base_url` validation — fail early with clear error on config typos like `https:example.com` (missing `//`)
- Added `socksio` to gateway dependencies for SOCKS proxy support (`httpx[socks]`)
- Added missing `__init__.py` for `types` package
- Updated `git clone` URL from `llm-rosetta` to `llm-rosetta` in documentation
- Resolved all `ty` type checker diagnostics in `src/` (31 → 0):
    - Fixed `is_part_type()` TypeGuard narrowing — replaced with specific type guard functions (`is_text_part`, etc.)
    - Added missing TypedDict fields: `provider_metadata` on `TextPart`/`ReasoningPart`, `file_id` on `ImagePart`/`FilePart`
    - Fixed `IRRequest.messages` type from `Required[Message]` to `Required[Iterable[Message]]`
    - Used `cast()` to bridge `dict[str, Any]` intermediates to TypedDict return types
    - Fixed dict literal type inference conflicts in converter response builders
- Resolved all `ty` type checker diagnostics in `tests/` (1506 → 0):
    - Added `cast()` wrappers on dict literals passed to functions expecting TypedDict parameters (`GenerationConfig`, `IRRequest`, `IRResponse`, `ToolDefinition`, `ToolChoice`, etc.)
    - Narrowed `Message | ExtensionItem` union results with `cast(list[Any], ...)` or `cast(Message, ...)`
    - Converted `Iterable` content fields to `list` for subscript and `len()` access
    - Added `assert ... is not None` guards before subscripting optional return types
    - Fixed `FinishReason` from bare string to TypedDict form `{"reason": "stop"}`
    - Fixed `IRResponse.object` literal from `"chat.completion"` to `"response"`
- Resolved all `ruff` lint violations in `src/` and `tests/` (UP035 deprecated imports, F401 unused imports)
- Google `thought_signature` preservation through gateway round-trips — newer Google models require `thoughtSignature` echoed back in function call parts; the gateway now caches `provider_metadata` (including `thought_signature`) keyed by `tool_call_id` and re-injects it on subsequent requests for both streaming and non-streaming modes (#51)
- OpenAI Responses converter now handles all 3 `input` formats: bare string (`"input": "hello"`), shorthand list (`[{"role": "user", "content": "hi"}]`), and structured list — previously only the structured format was supported, causing the OpenAI Python SDK's shorthand items to be silently dropped and producing empty IR messages when cross-converting to Anthropic or Google providers

---

## 2026-03-15 — Rebrand to LLM-Rosetta

### Changed

- **Project renamed from LLM-Rosetta to LLM-Rosetta** across all code, docs, and configuration
- Package renamed from `llm-rosetta` to `llm_rosetta`; `pyproject.toml` updated accordingly
- Documentation fully rewritten with Zensical for both English (`docs_en`) and Chinese (`docs_zh`)
- README (EN/ZH) updated with new branding, badges, and `pyproject.toml` metadata

---

## 2026-03-06 — Streaming & StreamContext

### Added

- **`StreamContext`** for stateful stream chunk processing across all 4 providers
- `stream_response_from_provider()` and `stream_response_to_provider()` methods on all converters
- `accumulate_stream_to_assistant_message()` helper function
- Stream abstract methods (`stream_response_to_provider`, `stream_response_from_provider`) added to `BaseConverter`
- 4 new IR stream event types: `StreamStart`, `StreamEnd`, `ContentBlockStart`, `ContentBlockEnd`
- `ReasoningDeltaEvent` and `tool_call_index` field on IR stream types
- Cross-provider streaming examples for all provider pairs (SDK and REST variants)
- Local file cache and retry logic for image downloads in examples

### Changed

- Stream method signatures updated with optional `context` parameter
- Deprecated `from_provider` methods removed; `auto_detect` updated to new API
- Obsolete single-provider example scripts removed (replaced by cross-provider examples)
- `_normalize()` extracted to `BaseConverter` as a shared utility

### Fixed

- camelCase fallback for Google GenAI REST stream/response fields
- Anthropic stream converter: `thinking_delta`, `signature_delta`, `tool_call_id` handling
- OpenAI Chat stream converter: `reasoning_content`, empty string, `tool_call_index` handling
- Missing `__init__.py` for test package discovery
- `from_provider` calls in `google_genai_rest_e2e` integration test

---

## 2026-02-14 — Cross-Provider Examples & Stream Converters

### Added

- **Stream converters** for all 4 providers: OpenAI Chat, Anthropic, Google GenAI, OpenAI Responses
- Stream converter unit tests for all providers
- **6 cross-provider conversation examples** (SDK-based): OpenAI Chat ↔ Anthropic, OpenAI Chat ↔ Google GenAI, OpenAI Chat ↔ OpenAI Responses, Anthropic ↔ Google GenAI, Anthropic ↔ OpenAI Responses, Google GenAI ↔ OpenAI Responses
- Common resources module for cross-provider conversation examples
- Image URL to inline base64 conversion helpers for Google GenAI compatibility
- OpenAI Responses E2E integration tests (REST + SDK)
- Unit tests for OpenAI Responses Ops classes and converter
- Examples README in English and Chinese

### Changed

- **OpenAI Responses converter** restructured to Bottom-Up Ops Pattern
- Post-refactor cleanup: removed deprecated utils and empty directories

### Fixed

- Image URLs converted to inline base64 for Google GenAI provider compatibility

---

## 2026-02-13 — Bottom-Up Ops Architecture

### Added

- **Google GenAI converter** rebuilt with Bottom-Up Ops Pattern
- TypedDict replicas of **OpenAI Responses API** types
- TypedDict replicas of **Google GenAI SDK** types
- Google GenAI REST and SDK E2E integration tests
- Unit tests for `google_genai` converter Ops classes
- Anthropic SDK and REST E2E integration tests
- OpenAI Chat E2E tests split into SDK and REST versions
- **GitHub Actions** CI/CD workflows and Dependabot configuration

### Changed

- **Anthropic converter** redesigned with bottom-up Ops architecture
- Imports updated to use new `google_genai` converter module
- Old `google/` converter and legacy tests removed

---

## 2026-02-12 — Converter Redesign

### Added

- TypedDict replicas of **Anthropic SDK** types
- TypedDict replicas of **OpenAI Chat** types with backward compatibility and tests
- Legacy body converter design preserved as historical reference

### Changed

- **OpenAI Chat converter** redesigned with bottom-up Ops architecture
- Ruff lint errors fixed across entire codebase

---

## 2026-01-06 — Layered Architecture & Documentation

### Added

- English and Chinese documentation structures initialized (`docs_en`, `docs_zh`)
- Comprehensive error handling documentation
- OpenAI Chat Converter integration tests
- Comprehensive mock implementations for `BaseConverter` test class
- File handling functionality in base converter
- Provider-to-IR mapping documentation

### Changed

- Converter base refined with layered abstract template
- All 4 converters restructured with layered architecture (Anthropic, OpenAI Chat, OpenAI Responses, Google GenAI)
- Type annotations updated for IR content/part conversion methods
- IR type system reorganized and enhanced
- English translations added to code comments and docstrings

### Fixed

- Reasoning content field assertion corrected
- File content handling in OpenAI Chat Completions converter

---

## 2026-01-05 — Auto-Detection & Package Maturity

### Added

- **`detect_provider()`** for automatic provider format auto-detection
- **`convert()`** convenience function for one-step format conversion
- `developer` role support in message validation
- Comprehensive validation tests for `BaseConverter`, Anthropic, Google GenAI, and OpenAI converters
- Tool call and tool definition conversion tests
- pytest configuration and `pytest-cov` dependency
- Competitive analysis document

### Changed

- **Package renamed** from `llm-provider-converter` to `llm-rosetta`
- IR format usage standardized across all providers
- Message creation standardized using `Message` class in examples
- Test suite migrated from unittest to pytest
- Common logic extracted into shared utility modules

### Fixed

- Standalone tool calls without current message context in OpenAI Responses converter
- Google GenAI Pydantic model handling reordered for tuple compatibility
- OpenAI content handling logic simplified for single text parts

---

## 2026-01-04 — Examples & Packaging

### Added

- `pyproject.toml` for package configuration
- Multi-turn chat example with tool integration
- Anthropic handover in multi-turn chat example
- Google GenAI function calling in multi-turn chat example

### Changed

- Utility functions moved from converters to IR types module
- OpenAI Chat converter code formatting improved
- Deprecated multi-provider query and weather tool modules removed

---

## 2025-12-24 — Initial Implementation

### Added

- **IR type system**: intermediate representation types for messages, content parts, tools, configs, request/response
- **`BaseConverter`** abstract class for LLM provider conversion
- **`AnthropicConverter`**: bidirectional Anthropic Messages API conversion
- **`OpenAIChatConverter`**: bidirectional OpenAI Chat Completions API conversion
- **`OpenAIResponsesConverter`**: bidirectional OpenAI Responses API conversion
- **`GoogleGenAIConverter`**: bidirectional Google GenAI SDK format conversion
- Comprehensive test suites for all 4 converters
- Package initialization and exports
- Weather tool example with mock data

---

## 2025-12-09 — Research & Design

### Added

- Initial project structure
- LLM provider message typing schemas documentation and comparison
- Provider messages IR design documentation
- MCP support comparison across providers (OpenAI, Anthropic, Google)
- Google GenAI Interactions API type analysis
- Multi-provider query example function
- OpenAI Responses API support in query examples
