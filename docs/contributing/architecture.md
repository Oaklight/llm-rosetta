---
title: 架构指南
---

# 架构指南

## 转换器结构

LLM-Rosetta 使用轴辐式（hub-and-spoke）架构，每个转换器负责特定 API 格式与共享 IR（Intermediate Representation）之间的双向转换。

每个 chat 转换器位于 `packages/llm-rosetta/src/llm_rosetta/converters/llm/<name>/` 下；embedding、rerank 与 decision 转换器分别位于 `converters/embedding/`、`converters/rerank/` 与 `converters/decision/`。所有转换器都继承 `BaseConverter`。基类使用**组合模式**——子类通过类属性声明各关注点的 ops 类：

```
converters/llm/<name>/
├── converter.py      # 主转换器类（继承 BaseConverter）
├── content_ops.py    # 内容块转换（文本、图片、refusal 等）
├── message_ops.py    # 消息级转换（角色、多轮对话）
├── tool_ops.py       # 工具定义、工具调用、工具结果
├── config_ops.py     # 请求配置（temperature、top_p、流式选项）
└── _constants.py     # 格式特定的常量
```

### 新代码放哪里

为已有格式添加新功能（如新的内容类型、新的字段）时，应在该转换器**对应的 ops 模块**中实现，而非写在独立的临时代码中。

尽可能复用 `converters/base/` 中的共享逻辑——基础模块提供了通用构建块：

| 基础模块 | 用途 |
|---------|------|
| `content.py` | 内容块辅助函数（文本、图片、refusal） |
| `messages.py` | 消息级工具函数 |
| `tools.py` | 工具定义和调用辅助函数 |
| `reasoning.py` | Reasoning/thinking 字段处理 |
| `schema.py` | JSON Schema 规范化 |
| `passthrough.py` | Provider 特定透传项 |
| `context.py` | `ConversionContext`（承载 shim、选项、状态） |

如果发现跨转换器重复逻辑，应考虑提取到基础模块中。

## Shim 层

`ProviderShim` 是一个轻量级身份卡，声明 provider 使用的基础转换器以及连接默认值和字段级变换。Shim 位于 `packages/llm-rosetta/src/llm_rosetta/shims/providers/<vendor>/<model-type>/`，chat 在模型类型下再保留一层格式目录。

转换器保持格式通用；shim 声明 provider 特定的差异（如 response ID 前缀、默认请求头、字段重命名）。

## 添加新的转换器

要添加对新 API 标准的支持：

1. 在 `packages/llm-rosetta/src/llm_rosetta/converters/llm/<name>/` 下创建转换器目录
2. 继承 `BaseConverter` 并实现所有抽象方法
3. 按照上述模式创建 ops 类
4. 在 `packages/llm-rosetta/src/llm_rosetta/formats/catalog.py` 中添加一条 `FormatSpec`，在
   `formats/detect.py` 中注册其请求体探测器，并在 `converters/registry.py`
   中绑定转换器类
5. 在 `packages/llm-rosetta/src/llm_rosetta/shims/providers/<vendor>/llm/<format>/` 下添加 shim
6. 在 `tests/converters/` 下添加测试
7. 提交 PR

这条 catalog 记录决定了该格式能否被看见：模型词表、网关的 provider 表、
SSE／错误信封／用量形状映射、自动检测、admin 选择器以及聊天路由都由它派生，
因此不需要在任何其它文件中为该格式做修改。可参考现有转换器。

## Provider 特定数据保留

LLM-Rosetta 将 portable IR 数据、可持久化的 provider passthrough 数据和单次转换上下文状态分为三层：

| 层 | 生命周期 | 用途 |
|----|----------|------|
| Portable IR（`Message`、内容部分、工具、推理） | 可序列化、可复用 | 可在不同 provider 格式之间转换的语义 |
| `ProviderPassthroughEvent` / `ProviderPassthroughItem` | 可序列化、可复用 | 没有 portable 表达的 provider 原生 chunk 或 item |
| `ConversionContext` / `StreamContext` | 单次转换管线 | warning、option、echo 字段、原始 ID/status/annotation，以及响应重建状态 |

`ProviderPassthroughEvent` 携带 provider 原生流式 chunk。`ProviderPassthroughItem` 携带非流式请求/历史 item；`IRResponse.provider_passthrough_items` 保存独立的非流式输出 item 及其原始位置。

Passthrough 数据带有 converter dialect 标签，例如 `open_responses` 或 `messages`。目标 dialect 相同时恢复原生 payload 的副本；目标格式不同时丢弃。语义 item 会产生 conversion warning；生命周期/心跳 stream event 为避免 warning 泛滥而静默丢弃。

`ConversionContext` 仍然必要，因为它承担不同职责：它是单次 request/response 管线内的临时 side channel，不会序列化进 conversation history。Provider passthrough carrier 是可持久化的 IR 数据，可以跨缓存、持久化和后续 HTTP 请求继续存在。

### 流式分派

基类 `stream_response_to_provider()` 实现使用类级分派表（`_TO_P_DISPATCH`）将 IR 流式事件路由到处理器方法。各 provider converter 通过 `_post_process_to_provider()` 钩子定制行为，无需重新实现分派逻辑。

## 设计理念：为什么所有请求都走 IR

网关对所有路由——包括同格式路由——都强制走 IR 转换，这是有意为之。项目的出发点是（现在仍然主要是）一个格式转换库，网关的核心价值建立在这之上：跨格式转换才是它区别于普通代理的地方。

同格式的 IR 往返是最基本的保真度检验。如果连单一格式的往返都会丢信息，跨格式转换就更不用提了。这是所有请求都走 IR 的根本原因，而且转换开销控制得够低，实际使用中不构成问题。

在 `25924518`（2026 年 8 月）之前，pipeline 根本没有 `baseline` 模式。后来加上它和配套的保真度检查器，是为了让翻译问题**同步暴露**：同一个请求同时走 IR 路径和 `baseline`，把两者输出做 diff，抓往返回归问题。`baseline` 是定位翻译问题的诊断模式，不是生产路径。

这是一项长期的设计决定，而非实现细节：**网关不会对同格式路由走捷径——所有请求依然经过 IR 转换。** `baseline` 只为同步暴露翻译问题而存在，不存在绕过 IR 的路径。`prefer_same_format` 只影响为某个模型*挑选哪个*上游 provider——它绝不会绕过 IR 转换，其第 1 层（与锚点同 shim、同 endpoint）也只是故障转移排序，而非捷径。该偏好是分层的，且相对一个**锚点**——即关闭该开关时路由会挑中的 provider；各层依次在锚点的 endpoint、锚点的账号、再到任意 provider 范围内把格式对齐到客户端，账号由凭据来源推导而非声明。决策记录：[#577 评论](https://github.com/Oaklight/llm-rosetta/issues/577#issuecomment-5461182245)。

上述判断所依赖的"同格式"概念，现在是 pipeline 上的一个命名属性 `ConversionPipeline._same_format`：访问时由调用方传入的两个 provider spec（已规范化）直接比较得出，而不是在每个使用点各自从格式字符串重新推导。它同时决定 `baseline` 与 intrinsic 捷径。注意它比较的是 **spec 而非 converter**：裸 `chat_completions` 源与 `deepseek--chat_completions` 目标共享同一个 converter 类，但在这里刻意为不同 spec——若当作同一个，就会跳过目标 shim 的 IR 级特性（其 reasoning 配置、其 transforms），而这正是指定 shim 的调用方期望得到的东西。

Intrinsic（provider 托管的）服务端工具由**目标 shim**决定，与格式名无关。`resolve_intrinsic_tools` 从*已解析*的 shim 上读 `ToolsConfig.intrinsic_tools`，并会先把 shim 名解析成对象，因此网关的 `target_shim=route.shim_name` 与库层传 shim 对象的行为完全一致。某个客户端格式能否原生声明服务端工具，同样是 **converter** 的属性（`BaseConverter._SUPPORTS_NATIVE_SERVER_TOOLS`）而非硬编码的格式字符串：Chat Completions 置为 `False`，这正是允许把以受支持 kind 命名的普通 function 提升为 intrinsic 的开关。

## Intrinsic 工具是派生的工具调用

Intrinsic（provider 托管的）工具就是多了一个参数的普通工具调用，而不是一套平行的类型系统。IR 用客户端工具所用的同一组结构承载它：`ToolCallPart` / `ToolResultPart` 带 `tool_type="intrinsic"`，`ToolDefinition` 带 `type="intrinsic"`，流式则是 `ToolCallStartEvent` / `ToolCallDeltaEvent` / `ToolResultEvent`。kind（`web_search`、`code_execution`、`google_search` 等）是一等字段 `intrinsic_kind`，通过 `intrinsic_kind_of()` 读取，它会回退到旧的 `provider_metadata` / `metadata` 写法。工具调用的其余字段（`tool_call_id`、`tool_input`、`result`、`tool_group`、`cache_hint`、`provider_metadata`）原样继承。kind 绝不从工具名推断。

**身份与配对。** 调用与结果按 `tool_call_id` 配对，绝不按 kind。同一轮里两个同 kind 的调用就是两个调用。

**每条路径只做一次决定。**

| 对象 | 路径 | 策略 |
| --- | --- | --- |
| 工具定义 | 请求 | 由目标 shim 的 `ToolsConfig.intrinsic_tools` 决定保留或丢弃。绝不降级成 function：模型会去调用一个客户端运行不了的函数。 |
| 调用 + 结果 | 请求历史 | 由目标 converter 判断能否重放该调用（`intrinsic_renderable(kind, fields, leg="history")`，外加 `intrinsic_requires_result`）。能则原生写出；不能则由 `degrade_intrinsic_history` 把调用**和**它的结果一起改写成普通 function 对。 |
| 调用 + 结果 | 响应、流式 | 永不降级：响应是客户端没有执行过的调用，把它当成客户端调用交出去，等于让客户端去执行它从未请求的东西。目标渲染不了的内容交给 `dropped_intrinsic_call` / `dropped_intrinsic_result` —— 返回空，并给出带 kind 的 warning。 |

这条规则放在 base 的模板里，而不是逐个 converter 各写一遍，因此不会出现"某些格式遵守、某些不遵守"的情况。按格式实测（每条路径都取一个外来 kind）：

| 格式 | 历史 | 响应 | 流式 |
| --- | --- | --- | --- |
| messages | 降级 | 丢弃 | 丢弃 |
| google_generate | 降级 | 丢弃 | 丢弃 |
| google_interactions | 降级 | 丢弃 | 丢弃 |
| open_responses | 降级 | 丢弃 | 丢弃 |
| chat_completions | 降级 | 丢弃 | 丢弃 |

`chat_completions` 在历史路径上降级，尽管它完全没有服务端工具词汇：客户端已经看到过的历史属于上下文，用 kind 命名的普通 function 调用可以承载它。另外两条路径它和其他格式一样丢弃；它的流式写出会把丢弃记在 stream context 上，这样参数 delta 和 finish flush 会跟着那个从未写出的 start 一起消失，而不是漏出去。

历史降级这一步在 `resolve_intrinsic_tools` 之后、`qualify_tool_names` 之前运行，同格式路由也会跑。降级后的函数对带有 `provider_metadata["_downgraded_from"] = {"type": "intrinsic", "kind": ...}` 标记，名字由 kind 派生且对名字字符安全（客户端已声明同名函数时改为 `{kind}_intrinsic`），并在转换 warnings 里上报一次。这一步之后它就是普通 function 调用，配对、孤儿修复、工具名限定走的都是普通工具流程。

**Converter。** `BaseToolOps` 负责分派（`ir_tool_call_to_p`、`ir_tool_result_to_p`）；converter 只覆盖钩子：`intrinsic_renderable`、`ir_intrinsic_call_to_p`、`ir_intrinsic_result_to_p`、`intrinsic_requires_result`。钩子不返回内容时，回落到 converter 原有的客户端路径。converter 自己只保留 kind 表和原生形状，关于 intrinsic 的其他事情都不归它管。

**原生保真。** `metadata["_native"]` 和 provider 签名只对其来源 provider 有效（`converters/base/helpers/opaque.py`）。外来的会被丢弃，或换成目标要求的占位值。

**服务端工具 id。** 原生服务端条目必须带目标要求的 id 前缀（Anthropic 是 `srvtoolu_`，OpenAI 是 `ws_` / `ci_`）。`prefixed_tool_call_id` 以确定性的方式加上前缀，已经带前缀的 id 保持不变，所以调用与结果仍然配对，同 provider 回放也不受影响。

**哪些内容能跨格式。** 每条跨格式路径都对真实上游（Anthropic、OpenAI Responses、Google Interactions）做过回放；矩阵里的十二对都返回 `200`。目标有同名 kind 且 IR 带齐目标所需字段时，原生写出；否则写成 function 对（Anthropic 的 `encrypted_content`、OpenAI 的 `container_id` 无法凭空造出）。`web_search` 与 `google_search` 这类 kind 之间没有语义等价表：kind 按 provider 声明，依据是每个上游被实测接受的内容。

回放对签名刻意从宽：Google Interactions 接受去掉任一签名的 `google_search` 历史，所以 result 签名不做携带。

## 引用（citation）的跨格式转换

引用是模型对自己所写文本的标注：用了哪个来源，通常还包括哪一段。每种 dialect
都有这个东西，但没有两种拼法相同，所以它和上面的工具一样处理：IR 里一个带判别式
的一等部件、一套共有的公共字段、以及逐 dialect 的映射表。

**IR 部件。** `CitationPart` 带 `citation_type` 判别式，各家共有的字段
（`url`、`title`、`cited_text`、`start_index`、`end_index`）平铺在部件上，
dialect 私有的定位字段放在 `provider_metadata` 下（按 provider 分键）。读取一律走
`converters/base/helpers/citations.py`（`citation_type_of` 加逐字段的访问器），
**不从字段形状推断**。2026 年以前那种把公共字段嵌在 `url_citation` /
`text_citation` 里、且不带 kind 的旧拼法仍然**可读**、但不再写出——与 `tool_group`、
`intrinsic_kind` 遵循同一条"一个概念一种写法"的规则。

**各家的形态。**

| dialect | 原生位置 | 种类 |
| --- | --- | --- |
| chat_completions | `message.annotations[]`，载荷嵌在 `url_citation` 键下 | `url_citation` |
| open_responses | content part 的 `annotations[]`，平铺 | `url_citation`、`file_citation`、`container_file_citation` |
| google_interactions | content part 的 `annotations[]`，平铺，snake_case | `url_citation` |
| google_generate | candidate 级的 `groundingMetadata`（chunks + supports） | 来源表达为 `url_citation` |
| messages | 它所标注的那个 text block 里的 `citations[]` | 它自己的五种：`char_location`、`page_location`、`content_block_location`、`search_result_location`、`web_search_result_location` |

其中两家形态完全一致：Responses API 与 Interactions 都发平铺的
`{start_index, end_index, url, title, type: "url_citation"}`。这一对是无损互转的，
其余各家通过公共字段映射。

**实测的转换范围。** 每个 dialect 带 URL 的引用都能转到另外四家。唯一不成立的格子
是**外来引用写进 Anthropic**，而这是那个 API 的性质、不是缺一个 writer：它的请求
校验器只接受上面那五个 tag（`url_citation`、`text_citation` 直接拒绝），而且每条
citation 必须指向消息里真实存在的 document 或 `search_result` block，或者它自己
签发的 `encrypted_index`。除此之外无法诚实地构造出来，所以 writer 返回空、由调用方
报告这次丢弃。反方向是通的：Anthropic 的读侧会把带 URL 的那几种导出，因此
Anthropic 的引用可以流向其他任何 dialect，但没有东西能流回来。

**结构因 dialect 而异。** Anthropic 把 citation 嵌在它所标注的 text block 里，所以
读侧把它们拆成独立部件、写侧再挂回前面那个 text block。Google 完全没有 annotation：
它的引用就是 candidate 的 `groundingMetadata`，由部件重建（来源变成 chunks、span
变成指向它们的 supports），而它的搜索级 metadata——查询词与渲染出的入口——不是
引用，只在消息或 finish 事件上带一份。

**流式。** 流式引用是 `citation_delta` 事件，带同一个部件，在 provider 报告它的
位置发出、不推迟：一个没有产生文本的 candidate 也照样带着它的来源。每个 dialect
从自己的载体喂它（`citations_delta`、`response.output_text.annotation.added`、
`text_annotation_delta`，或 Google 的 chunk 级 metadata），而 base converter 对
它默认以告警作答，因此没有原生载体的 dialect 会报告这次丢失，而不是静默丢弃。

**丢失会被报告。** 目标无法呈现的引用——发往只认 URL 的 dialect 的文档引用、没有
text block 可挂的引用——都走 `report_dropped_citation`：优先写进本次转换的 warning
sink，同时总是写日志。

## Round-Trip 兼容性

所有转换路径必须保持 **round-trip 兼容性**。每个变更都必须针对以下场景进行测试：

- **A → IR → A**（同格式 round-trip）：转换为 IR 再转回相同格式，必须产生有效且语义等价的结果。不允许静默丢弃字段。
- **A → IR → B**（跨格式）：从一种格式转换为另一种格式，必须为目标格式产生有效输出，即使源格式存在没有直接对应的字段。
- **A → IR → B → IR → A**（完整 round-trip）：经过两次转换再转回的消息必须保持可用。这是网关的实际执行路径——请求在入站时转换，响应在出站时转换。

添加或修改转换器逻辑时，至少要编写覆盖前两个场景的测试。网关的跨格式路由依赖于所有转换器在 IR 语义上的一致性——一个转换器的 IR 输出出错，会导致其他所有转换器的级联失败。

## 测试

测试位于 `tests/converters/<name>/` 下，与转换器结构对应。关键模式：

- **单元测试** — 按 ops 模块测试，隔离验证单个转换函数
- **Round-trip 测试** — A → IR → A 转换并断言等价性
- **跨格式测试** — A → IR → B 转换并验证输出符合格式 B 的 spec
- **流式测试** — 验证流事件顺序和生命周期

运行 `make test` 执行完整测试套件（不包括需要 API key 的集成测试）。

## Gateway Ops 层

!!! note "转换器 ops 与 gateway ops 的区别"
    转换器层按关注点拆分转换逻辑（`content_ops`、`message_ops`、
    `tool_ops`、`config_ops`）。**Gateway ops 层**是一个独立的概念——
    每个网关侧的变更操作都是一个 `OpsBase` 子类，自动进行审计记录。

### OpsBase 协议

```
路由处理器 → 构造 Ops 子类 → 调用 execute()
                                ├── _run()        （业务逻辑）
                                └── _record()     （审计记录）
                                     ├── _message()  （人类可读摘要）
                                     └── _details()  （结构化元数据）
```

`execute()` 通过 `_run()` 执行操作，然后通过 `_record()` 记录结果。
失败时，`_record(result=None, error=exc)` 被调用，严重性自动提升为
WARNING，原始异常被重新抛出。记录过程的错误会被日志记录但不会掩盖
操作结果（尽力审计）。

### OpsContext

`OpsContext` 是一个轻量级的服务包，在应用初始化时由 `setup_admin()` 创建一次。
它持有四个子系统的引用——均为可选（未配置时为 `None`）：

| 槽位 | 类型 | 用途 |
|------|------|------|
| `ops_log` | `OpsLog` | 审计事件日志 |
| `request_log` | `RequestLog` | 代理请求流量日志 |
| `metrics` | `MetricsCollector` | 进程内指标 |
| `persistence` | `PersistenceManager` | SQLite 持久化 |

### 模块组织

| 模块 | 内容 |
|------|------|
| `ops/base.py` | `OpsContext`、`OpsBase` ABC |
| `ops/proxy.py` | `OpsProxyRequest` — 热路径遥测，覆写 `_record` 以写入 `request_log` + `metrics` |
| `ops/data.py` | 数据变更 ops（`OpsClearData`、`OpsCleanupData`、`OpsTrimData`、`OpsVacuum`、`OpsRebuildMetrics`、`OpsClearOpsLog`、`OpsPrune`、`OpsPeriodicCleanup`） |
| `ops/security.py` | 认证/会话 ops（`OpsPasswordChange`、`OpsTokenRotate`、`OpsSessionLogoutAll`） |
| `ops/keys.py` | API 密钥管理 ops（`OpsKeyCreate`、`OpsKeyUpdate`、`OpsKeyDelete`、`OpsKeyRotate`） |

### 使用模式

路由处理器用 `OpsContext` 构造 Ops 子类并调用 `execute()`——
记录自动完成：

```python
from llm_rosetta_gateway.ops import OpsClearData

result = await OpsClearData(request.app.ops_ctx, table="request_log").execute()
```

### OpsProxyRequest（特殊情况）

`OpsProxyRequest` 覆写 `_record()` 以写入 `request_log` + `metrics`
而非 `ops_log`。代理请求的频率过高，不适合写入审计日志。它的
`_run()` 是空操作——实际的代理处理发生在 ops 类之外。这个类将遥测
记录接口统一到同一个 `execute()` 契约下。

## Gateway 路由层

`gateway/routing/` 决定两件事：哪个 provider 服务这个请求，以及请求转换
之后用哪份凭据、带哪个上游 session 身份发出。它们是同一层，因为读的是
同一组输入——model、客户端、会话——但它们不能合成一次调用。

### 两个调用点

两个决定之间隔着一个 pipeline。provider 必须在请求体转换**之前**确定，
因为转换依赖目标格式；凭据与 session id 的指纹取自 IR 消息，只能在
**转换之后**决定。所以这一层每个请求保有一个 context，分两条链跑：

```
request ──→ GatewayConfig.resolve(ctx)     provider 选定（转换之前）
            pipeline.ir_request            请求体转换为 IR
        ──→ _apply_request_stages(ctx)     凭据与 session id（转换之后）
```

`SelectionContext` 在网关入口处创建，由各阶段逐步填充。路由表与 provider
表在开始时就装进去了（`install_route()`），因此任何阶段都不需要反向引用
config 对象。

### 阶段

一个阶段就是一个决定。每个阶段声明自己读（`requires`）写（`provides`）的
context 字段，`RoutingChain` 在**构建**时校验这些声明：如果一个阶段排在
它所依赖的东西之前，构造时就会抛 `StageOrderError`，而不是等到第一个会
失败的请求。

| 链 | 阶段 | 决定什么 |
|----|------|----------|
| provider | `AnchorStage` | 关掉偏好时路由本来会选的 provider。整条 route 的计数器在这里推进。 |
| provider | `SameFormatStage` | `server.prefer_same_format` 打开时，按 tier 把锚点调整到说客户端格式的 provider。 |
| request | `ProviderKeyAffinity` | key ring 里的哪把 key 服务这个会话：客户端身份加 IR 的前两条消息。 |
| request | `SessionHeaderAffinity` | 写进 shim 声明头里的 session id：客户端身份加首条消息。 |

两个请求阶段总是运行，各自在被选中的 provider 用不到它时弃权——只有一把
key，或没有声明 session header。同格式阶段只在偏好打开时入链，因为它是
偏好：开关关掉，选择行为完全不变。

### route 身份

`ModelRoute` 带一个 `route_id`，策略按它在其中记账。一个 model 的 provider
子集——同一种格式、同一个 shim 端点、同一个账号——是一条独立的 route，
id 各不相同，因此 tier 的计数器跨请求存活，并且与整条 route 的计数器各自
推进。这一层不读任何历史：每个输入都来自手上这个请求，这也是网关不需要
存储就能运行的原因。

### 与 Ops 层的对比

Ops 层刻意没有 registry，也没有链：它的成员是一组封闭的变更，顺序从无
意义。路由正好相反——阶段有顺序，顺序正是校验要保护的东西，而且预期还会
增加——所以它是网关里唯一带链的一层。
