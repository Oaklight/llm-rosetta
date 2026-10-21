---
title: Changelog
---

# Changelog

All notable changes to LLM-Rosetta are documented here. This project follows [Keep a Changelog](https://keepachangelog.com/) conventions.

## [未发布]

### 新增

- **所有分页视图都能直接跳转到指定页**（PR [#1149](https://github.com/Oaklight/llm-rosetta/pull/1149)）：只有 Prev/Next 时远端页面几乎不可达——Request Log 有数十万行（每页 30 条即 8943 页），Error Dumps 也有几十页。Request Log、Server Ops Log 与 Error Dumps 三个分页控件现在都在 Prev/Next 旁增加了页码输入框与 Go 按钮，在输入框内按 Enter 等效。超出范围的数字会收敛到第一页/最后一页，输入为空或非数字时保持当前页不变，只有一页时输入框禁用。页码运算放在无 DOM 依赖的 `core/pagination.js` 中，由 `tests/js/pagination.test.mjs` 固定；输入框里的半成品数字能扛过日志每 3 秒的自动刷新——只有当它的值与上次同步的页码一致时才会跟随页面更新。

- **Dashboard 页的调试面板可折叠**（PR [#1081](https://github.com/Oaklight/llm-rosetta/pull/1081)）：Profiling 与 Content Capture 是按需开启的调试工具，但两者都把完整表格直接铺在页面上——即使 profiling 未开启，Error Dumps 及其后的内容也被推到很下面。现在两个区块都是 `<details>`，摘要行保留 ON/OFF 徽标，折叠时也能看到当前状态；面板按需展开，展开/折叠状态跨刷新保留。键盘操作与箭头样式都由 `<details>` 提供，与 Server Settings 区块一致。

- **Providers 页的 Server Settings 折叠为一行摘要**（PR [#1080](https://github.com/Oaklight/llm-rosetta/pull/1080)）：全局代理卡片是部署期设置，大多数运维人员不会用到，却占据了 provider 列表上方约 190px 的手机首屏，桌面端也把所有 provider 卡片向下推。现在该区块是一个 `<details>`，摘要行显示当前生效值（“未设置全局代理”或具体 URL），默认折叠；展开后仍是原来的卡片，展开状态会跨刷新保留。`<details>` 自带键盘操作与展开状态，摘要文本由 provider 渲染流程重新同步，因此会随面板一起切换语言。折叠行会去掉 URL 中的凭据（`http://user:pass@host` 显示为 `http://host`），输入框与已保存的值保持不变。

- **API 密钥表可以按任意列排序**（PR [#1078](https://github.com/Oaklight/llm-rosetta/pull/1078)）：密钥表此前只有服务端给出的顺序，想回答“某一时间窗内哪个密钥消耗最多 token”或“最近创建的是哪个”，只能通读整张表。现在 Label、Created、Last Used、Requests、Input、Output、Total 都是可排序列，交互与箭头和 Models 表一致：点击当前列切换升降序，点击其他列从升序开始；用量列跟随所选的 5h/24h/7d/30d 时间窗，缺失日期视为最旧，internal 与 Unattributed 行始终排在末尾而不会与有标签的密钥交错。未点击任何列之前保持服务端顺序。

- **提供方额度按凭据读取，耗尽的凭据会退出轮换**（epic [#1003](https://github.com/Oaklight/llm-rosetta/issues/1003)，issue [#1004](https://github.com/Oaklight/llm-rosetta/issues/1004)、[#884](https://github.com/Oaklight/llm-rosetta/issues/884)）：订阅型通道按窗口计量，此前网关只有在请求被上游拒绝后才知道某个窗口已经耗尽。新增的 `gateway/quota` 层把三类事实分开：**shim** 声明 probe 以及这条通道的额度从哪里读（`quota: {probe, url, interval, demote_at}`，与其它连接事实并列）；**provider** 持有凭据，并可以覆盖其中任意字段；**状态**按 `(probe, sha256(key)[:16])` 保存一份读数和至多一条降级记录，因此两个 provider 共用一个 key 时共享额度，而同一个 key 用在两条通道上则是两份额度。枢纽只保留契约与解析，vendor 的 probe 随 shim 的 `quota.py` 发布（跨多个格式的 vendor 放组目录），端点、认证、解码和报错措辞都在 `provider.yaml` 旁边。内置两个：`opencode_go` 读取 `/zen/go/v1` 上的 `{base_url}/usage`（percent 是**已用**比例；`rate-limited` 会在低于 100% 时关闭窗口；Zen 通道会直接报告没有按 key 的端点，而不是去探测一个 404），`openai_codex` 读取 `https://chatgpt.com/backend-api/wham/usage`，使用计划的 access token 以及从其 JWT claim 取出的 `chatgpt-account-id`——为 Sign in with ChatGPT 工作（[#1055](https://github.com/Oaklight/llm-rosetta/issues/1055)）预先接通。`KeyRing` 的 round-robin 与 affinity 不变，只多一个谓词：已降级的 key 会被跳过；若所有 key 都被降级则退回普通轮换，因为降级只是建议。耗尽的窗口降级到其重置时间；若 vendor 在 429 响应体里点名窗口（`metadata.limitName`），则立即带内降级并把请求换 key 重试，而**探测失败**永不降级。声明了 `quota.interval` 的 provider 会通过 deferred-startup 任务在后台轮询；provider 卡片新增 **Quota** 按钮，调用 `GET /admin/api/config/providers/<name>/quota`（按凭据和窗口逐行显示，`?refresh=1` 绕过一分钟缓存），无 key 通道的 429 则给整个 provider 一个短暂冷却而不是降级 key。同一次变更还落地了 **OpenCode Zen 公共免费通道** `opencode_zen_free--chat_completions`：上游拒绝第三方客户端，因此 shim 声明指纹请求头、`x-opencode-session`/`x-opencode-request` 的 OpenCode 时间戳标识格式、网关要求的唯一 `read` 工具（仅在客户端未声明时注入，注入工具产生的调用在返回路径上移除）以及 `stream_required`——强制向上游流式请求，并把源格式事件重新拼装成非流式的 Chat Completions 回复。shim 声明的静态请求头现在优先于客户端透传的同名头，避免把调用方的 `User-Agent` 交给一个校验客户端形状的通道；客户端自带的 session id 仍然原样透传。见[提供方额度](guide/quota.md)与 [OpenCode 页面](guide/providers/opencode.md#public-free-lane)。

- **Models 的 provider 过滤器显示 provider logo**（PR [#1077](https://github.com/Oaklight/llm-rosetta/pull/1077)）：该过滤器原本是原生 `<select>`，只能列出名称，而旁边的 provider 卡片都带有厂商标识。现在换成一个轻量自定义列表：触发器显示所选 provider 的 logo 与名称，每一行显示与 provider 卡片相同的标识（优先配置里的 `logo`，其次是 shim 的）；隐藏 input 继续承载 `modelProviderFilter.value`，因此重置按钮和 Providers → Models 跳转都不受影响。键盘操作：Enter/Space/ArrowDown 打开，方向键与 Home/End 移动，PageUp/PageDown 每次跳五行，输入字母可快速定位，Enter 提交，Escape 关闭并把焦点还给触发器。手机上该控件独占一行。

- **面板导航加入图标，并在手机上变为底栏**（PR [#1075](https://github.com/Oaklight/llm-rosetta/pull/1075)）：每个一级标签的文字前现在有一个 16px 图标；≤768px 时标签条改为固定底栏——五个等宽目标、图标在上文字在下、激活项带主题色指示条——而不再是一条会隐藏末尾目的地的横向滚动条。769–1100px 之间收紧标签内边距并在需要滚动前隐藏配置路径；手机上头部保持单行：先隐藏 “System Time” 文字，480px 以下隐藏 “gateway admin” 副标题，360px 以下连时钟一起隐藏。诊断状态栏移到导航栏上方，保持一行可横向滚动。图标与文字是兄弟节点——若在标签上使用 `data-i18n`，切换语言时 `textContent` 会被整体重写并抹掉 SVG。顺带修复了点击测试中发现的跳动：当 Request Log 过滤到一页就能放下时滚动条消失，布局视口随之变宽，固定导航与状态栏会移动；现在 `scrollbar-gutter: stable` 在每个页面预留滚动条槽位，两条栏宽度恒定。

- **`shims.presets` 可按名称预载独立 shim 包**（epic [#927](https://github.com/Oaklight/llm-rosetta/issues/927)）：配置现在可以写 `"shims": {"presets": ["argo"]}`，无需再复制仓库 URL 与归档路径。每个预设键（`alcf`、`argo`、`asksage`）都会展开为 catalog 中对应的远程源，先于显式 `remotes` 应用；同一 URL 的显式 remote 会合并到对应预设上（显式键优先），因此固定 `ref` 时无需重复 `path`。仓库都是公开的，预设不带 `token_env`。Settings → Shim Sources 新增 **Presets** 字段用于填写同样的键，原有的预设选择器仍会把展开后的 remote 追加到列表供审阅；`GET /admin/api/config` 以 `shim_source_presets` 暴露 catalog；若配置使用了某个未声明源的 shim，启动时会报错并指明包名、预设与展开后的 remote。

- **窄屏下 Models 表变为卡片列表**（PR [#1074](https://github.com/Oaklight/llm-rosetta/pull/1074)）：手机宽度下，六列表格把模型名压缩到只剩几个字符，能力徽章则堆叠在 40px 宽的列里，一行已无法说明它描述的是哪个模型。现在 768px 以下每行渲染为一张卡片——首行是完整且可换行的模型名及其上游标记，下一行并排显示类型与能力徽章，再下一行单独显示 provider，最后一行以分隔线隔开，放置启用开关与 Test / 编辑 / 更多操作。展开的 provider 明细（`N providers`）会把权重与上游字段铺满卡片宽度而不是溢出，无匹配结果的状态也横跨整张卡片。因为隐藏的列头是唯一的排序入口，工具栏在该宽度下提供同样的三种排序键（模型 / provider / 状态）的选择器与方向切换按钮，并与表头按钮共用同一状态；卡片行带有显式的 `row` / `cell` 角色，每个单元格都标明所属列，无障碍树仍能报告带表头的表格。桌面端渲染不变。空状态单元格的 `colspan` 也从 4 修正为表格实际的 6 列。

- **请求日志的自动刷新可以暂停**（PR [#1073](https://github.com/Oaklight/llm-rosetta/pull/1073)）：两个日志视图此前每 3 秒重建一次表格，正在选中的文本会被清除，行也可能在阅读与点击之间发生位移。现在 Requests / Server Ops Log 切换旁新增"暂停/恢复刷新"按钮：暂停会清除两个刷新定时器，并在切换视图和标签页后保持暂停；恢复时先拉取一次再重启定时器，而手动改过滤条件、翻页或切换视图时即使处于暂停也会立即拉取一次。该状态仅限当前会话——重新加载页面会回到未暂停状态。

- **API 密钥可以搜索，并在弹窗中重命名**（PR [#1072](https://github.com/Oaklight/llm-rosetta/pull/1072)）：密钥表此前无法按标签收窄 20 多条记录，重命名用的又是浏览器阻塞式 `prompt()`，既不遵循面板主题，也无法用常规弹窗方式关闭。现在搜索框按标签过滤行，并在有查询时显示 `x / y 个密钥`（没有标签的未归属用量行在搜索时隐藏），另有独立的"无匹配"空状态。重命名改为页内弹窗：回车保存，Esc、取消或点击遮罩关闭，当前标签会被预选中；行的标签改从 DOM 读取，不再嵌进 `onclick` 字符串，因此包含引号的标签不会再破坏编辑操作。这也堵住了一个注入向量：旧的 `onclick` 把标签放进 HTML 属性中的 JS 字符串，而 `esc()` 生成的 `&#39;` 会在脚本解析前被 HTML 解析器还原成 `'`，构造过的标签因此可能在点击时执行；现在标签改从 DOM 读取，只嵌入密钥 id。

- **请求日志可以按时间窗收窄，并且每页条数可选**（PR [#1071](https://github.com/Oaklight/llm-rosetta/pull/1071)）：日志此前只有模型/provider/状态/密钥过滤，想回答"最近一小时的失败"只能在混杂流量中翻页，而错误记录区早已提供时间范围。`GET /admin/api/requests` 现在接受 `since` / `until` 两个 ISO 8601 边界——`until` 不含端点，相邻时间窗不会重复计入边界时刻；`Z` 与显式偏移都会归一到 UTC，以匹配存储的时间戳；无法解析的值返回 `400` 并指出该值。面板提供与错误记录视图相同的预设（最近 1 小时/24 小时/7 天/30 天，以及本地日期的自定义范围），并增加每页条数（30/50/100/200），切换过滤条件时保持所选值；"重置"会连同其余过滤器一起清除时间过滤。分页被钳制在 1–500 条，非整数的 `limit`/`offset` 返回 `400`——在 SQLite 中负的 `LIMIT` 表示*不设上限*，绝不能让这样的值到达查询——不带 UTC 偏移的时间戳会被拒绝，而不是静默按 UTC 处理。错误转储与服务运维日志列表也采用同样的钳制与 `400` 处理。

- **ALCF 与 AskSage 离开核心，Settings 选择器预载全部独立 shim 包**（issue [#666](https://github.com/Oaklight/llm-rosetta/issues/666)，epic [#927](https://github.com/Oaklight/llm-rosetta/issues/927)）：ALCF 与 AskSage 的 shim 是在 0.14.0 开发周期引入的，从未进入稳定版 —— ALCF 只出现在 `0.14.0.dev3`/`dev4` 测试版中，AskSage 一次也没有 —— 因此它们现在就迁往各自的独立仓库，而不必等到 0.16.0 的窗口。自 `0.6.3` 起随稳定版发布的 Argo shim 则保留内置直到该版本。三个独立包现在统一记录在 `llm_rosetta.shims.catalog` 中，**Settings → Shim Sources** 无论是否安排了移除，都会把每个独立包（包括 Argo，便于提前采用其源）作为一个一键预设提供。当时三个仓库都是私有的，预设写明了 `token_env: GITHUB_TOKEN`。**同一周期内已被取代：** 现在它们已公开，预设不再携带 `token_env`。`alcf--*` 与 `asksage--*` 旧别名随内置 shim 一并移除；独立包只注册规范名称。仍在使用这些名称的配置现在会直接指出提供它的独立包以及应声明的 `shims.remotes` 条目，而不只是报告缺少 `base_url`。参见[独立 Shim 包](guide/shims.md#standalone-shim-packages)。**同一周期内已被取代：** Argo 也在 0.14.0 发布前离开了核心；见破坏性变更中的 Argo 条目。

- **网关启动后做的数据修复现在会报告自己做了什么，其中一步还能单独重跑**（epic [#1027](https://github.com/Oaklight/llm-rosetta/issues/1027)，PR [#1044](https://github.com/Oaklight/llm-rosetta/pull/1044)）：socket 打开后跑的那五步数据修复——写入时尚无该列的行的 provider 名、某个 key 的 `last_used`、把 cache token 排除在外的 token 总数、在 bucket 列存在之前写入的行的用量归属、以及错误转储对应的 request log id——此前只把计数放进一个本地 dict，打印一行日志就丢掉。现在它们是同一个 registry 里的 `Step`，结果会被保留，并由两个端点对外呈现：`GET /admin/api/maintenance`（每步：是否跑过、改了多少、花了多久、失败原因）与 `POST /admin/api/maintenance/<name>` 按名字重跑一步；后者在该步失败时返回它记录下的失败，而不是 500。界面出现之前就存在的两个单一用途路由——`POST /admin/api/keys/backfill-last-used` 与 `POST /admin/api/error-dumps/backfill-log-ids`——对调用方保持不变（同样的方法、同样的 `{"updated": n}` 响应体），但已是同一批 step 的处理函数，而不是第二套接线。`GET /health/ready` 增加 `maintenance` 对象（`state` 为 `not_started` / `running` / `done` / `partial`、失败的步骤名，以及每步的 `changed` 与 `seconds`），使探针能区分"修复都完成了"与"启动任务结束了"；失败的修复**故意**不会让探针返回 503：这些步骤都可以重试，为一次失败的修复挡住流量，正是 deferred 策略要避免的那种故障。详见[管理 API 参考](api/admin.md#maintenance)。

- **引用（citation）现在在每种 dialect、每个 leg、包含流式在内都能转换**（issue [#459](https://github.com/Oaklight/llm-rosetta/issues/459)，PR [#1025](https://github.com/Oaklight/llm-rosetta/pull/1025)）：引用此前不携带**种类**——读侧靠"两个嵌套字典里哪个恰好存在"来分支，写侧各自把本 dialect 的拼法直接写进 IR，于是半数 dialect 读不懂另一半写的（`open_responses` 与 `messages` 的读函数**根本没有调用点**）。现在 `CitationPart` 带 `citation_type` 判别式，各家共有的字段（`url`、`title`、`cited_text`、`start_index`、`end_index`）平铺在部件上，provider 私有字段放在 `provider_metadata` 下——与 `tool_group`（#999）、`intrinsic_kind`（#1010）同一套做法，2026 年以前那种嵌套拼法仍然**可读**、不再写出；读法集中在 `converters/base/helpers/citations.py`，任何写侧都不再按形状分支。五个 dialect 的读写与两个 leg 全部接通，流式也终于有了载体：`citation_delta` 事件与非流式的部件一一对应，分别由 `citations_delta`（Anthropic）、`response.output_text.annotation.added`（Responses）、`text_annotation_delta`（Interactions）与 `choices[].delta.annotations`（chat completions）喂入，base converter 对它默认以告警作答，因此没有原生形态的 dialect 会报告这次丢失而不是静默丢弃。GenerateContent 完全没有 annotation——它的引用就是 candidate 的 `groundingMetadata`，写出时由部件重建（来源变成 chunks、span 变成指向它们的 supports，`maps` 包装也能往返），而它的搜索级 metadata（查询词与渲染出的入口）从"每条 citation 一份"改成整个 turn 一份。转换网格是对真实上游实测后钉进 `tests/converters/citation_matrix/` 的：拼 `url_citation` 的四家——chat completions、Responses、Interactions、GenerateContent——彼此自由互转，而 Anthropic 那一列是**实测得出的丢弃、而非没实现**：它的校验器只接受自己那五个 tag（`url_citation` 与 `text_citation` 直接 400），且要求每条 citation 指向消息自身携带的 document 或 `search_result` block，或它自己签发的 `encrypted_index`。**能转换的是"带 URL 的引用"，无论它原本是哪一种**：Anthropic 的 `search_result_location` 或 `web_search_result_location` 可以单向流往那四家，反向则不行。Anthropic 的形态是结构性的，所以读侧把 citation 从它所标注的 text block 里拆出来，写侧再挂回前面那个 text block。目标无法呈现的引用一律走 `report_dropped_citation`：优先写进本次转换的 warning sink，同时总是写日志。

- **Argo shim 被列入移除时间表，网关会如实告知**（epic [#927](https://github.com/Oaklight/llm-rosetta/issues/927)，PR [#1036](https://github.com/Oaklight/llm-rosetta/pull/1036)）：三个 Argo shim 将在 **0.16.0** 迁往独立的仓库——它们是一条面向具体部署的通道，上游按自己的节奏变化，因此改为以独立 shim 包的形式维护：网关通过顶层 `shims` 段声明它们（[#929](https://github.com/Oaklight/llm-rosetta/pull/929)），无需发布新的核心版本即可使用。（ALCF 与 AskSage 已在 0.14.0 离开核心；见上方独立包条目。）这里没有删除任何东西；新的是「移除」这件事本身被*声明*出来——既告诉你，也告诉代码，且出自同一份清单。`llm_rosetta.shims.removals` 为每个即将下线的 shim 保存一条记录——移除它的版本、替代它的 `shims.remotes` 条目，以及随之不再解析的旧拼写——而 `detect_shim_removals()` 是**纯函数**（没有 I/O、不碰注册表、不写日志），因此启动告警、管理面板与 CI 从同一处读到同一个答案，日后它也可以作为维护模块中的一个 `LOAD_TIME` 步骤（[#1027](https://github.com/Oaklight/llm-rosetta/issues/1027)）而不必拖着状态走。网关会**为每个受影响的提供方记录一条告警**，指明该 shim、移除它的版本，以及应当改为声明的源；若某个 shim 已由声明的源提供，则保持静默，因为该源会在重建时覆盖内置版本——于是日志归于安静，正是迁移已落地的确认。一个*失败*的源不会让任何东西静默，因为它什么也没覆盖；**被禁用**的提供方仍会被报告，因为它一旦重新启用，配置就会立刻出问题。**Settings → Shim Sources** 会列出同样的告警，并提供一键「添加替代源」，该操作只向 remotes 文本框追加内容——从不保存——因此原有的 Save 仍是唯一的写入者，运维可以逐字核对即将写入的内容。**`make check-shim-removals`** 把这份时间表约束到代码上：当版本低于某个发布时它通过，因为此时仓库*本就应当*仍列出这些 shim；一旦版本到达该发布，则只要该 shim 仍被注册，*或*只要还有任何东西仍引用它，检查就会失败——无论是一个示例、一个 Docker 文件、一个 UI 原型、一个辅助脚本、文档，还是源码；其中遗留的 `LEGACY_SHIM_ALIASES` 条目与 shim 自身的 `provider.yaml` 同等算数。changelog 豁免（其历史不会被改写），清单自身豁免（它按设计列出每一个计划移除的 shim），而带有 `shim-removal-ok` 的行也豁免——0.16.0 的迁移表正是借此才被允许写出旧名。**同一周期内已被取代：** 移除实际发生在 0.14.0 而非 0.16.0；这里描述的机制仍然保留。迁移本身不受这一切影响：替代 shim 保留原有名称，因此声明源就是全部工作。参见[独立 Shim 包](guide/shims.md#standalone-shim-packages)；删除工作由 [#919](https://github.com/Oaklight/llm-rosetta/issues/919) / [#666](https://github.com/Oaklight/llm-rosetta/issues/666) 跟踪，每个仓库各有一个后续 issue（`-argo#2`、`-asksage#2`、`-alcf#2`），涵盖重新同步、CI，以及第一个 tag —— 这也是管理台预置项至今不带 `ref` 的原因。

- **工具的 `strict` 与 tool_choice 白名单现在能穿过 IR**（epic [#462](https://github.com/Oaklight/llm-rosetta/issues/462)）：Chat Completions 与 Responses 共享的两个字段在穿过 IR 时都丢了，而且是静默地丢。`strict`——要求提供方按工具 schema 校验调用参数的那个标志——此前只有 Responses 转换器会读、且只存在 `metadata["strict"]` 里，于是带 `strict: true` 的 Chat 请求到达 Responses 上游时变成 `strict: false`，而反向的 Responses→Chat 则完全丢掉（Chat→Chat 同样丢）。它现在是 IR 的一等字段 `ToolDefinition.strict`，五个格式里有**三个**都有这个概念（`chat_completions` 与 `messages` 现在都会读写它；`open_responses` 读自己的字段、同时仍接受旧的 metadata 写法；两个 Google 格式没有对应概念，丢弃该标志）。线上实测：Chat 与 Responses 都接受 `strict: true`；Anthropic 也接受，但除 schema 设了 `additionalProperties: false` 之外一律返回 `400`——客户端的 schema 不该由我们代改，所以不代改。第二个字段是 `tool_choice` 白名单：四个格式、四种写法，而 IR 此前把它当作不透明的 Responses 形状 `provider_extensions` 存着——于是其他目标全都丢掉这个限制，Google 目标甚至连第二个名字都保不住（`allowedFunctionNames` 与 Interactions 的 `allowed_tools` 都被截断到第一个，因为 `ToolChoice` 没地方放其余的）。现在 `ToolChoice.allowed_tools` 装的是工具身份——`(name, tool_group)`，与 IR 其余部分用的是同一对——Responses、Google generateContent 与 Google Interactions 都能读写，且列出每一个名字。来源给了分组时，命名处理会用它解析限制，因此裸名被多个工具共用的工具也能解析出来，而分组在解析后被用掉，不会留下来与名字互相矛盾；指名了未声明工具的限制仍会告警。只有两家只能放宽：Chat Completions 与 Anthropic Messages 没有白名单变体，于是选择降级回裸 mode（"必须调用其中之一"的选择保留 `required`）并告警说明模型此后可以调用任何已声明的工具——限制消失与可选字段被丢弃不是一回事。Chat Completions 是更尖锐的一例：变体写在 OpenAI 的 schema 里，线上 API 却拒绝它（`400 Missing required parameter: 'tool_choice.function.'`），所以这个形状会被读取——客户端可能把它交给库、再路由到确实支持它的格式——但永远不会被写出。线上 API 还教了我们一件事：白名单条目的 `type` 必须与工具的声明方式一致，因此它取自声明而非存进 IR，否则被降级的 custom 工具会让它变味。顺带修掉 IR 类型指南里那个示范：它写的是嵌套的 wire 形状，而不是扁平的 IR 形状。

- **用量报告可以直接从 shell 读取**（issue [#1023](https://github.com/Oaklight/llm-rosetta/issues/1023)，epic [#1016](https://github.com/Oaklight/llm-rosetta/issues/1016) 的第 3 阶段）：`llm-rosetta-usage` 打印管理面板展示的那些内容——同样的滚动窗口、同样的分类——给不打算开浏览器的运维人员。它是同一份报告的第三个读取方，而不是第二份实现：存储通过 `PersistenceManager` 打开，报告由 `UsageReport.build` 构建，而 `--json` 输出的就是这份报告本身而非它的某种渲染——subject 这一路切分原样携带管理 API 返回的键，因此读命令的脚本与读面板的脚本看到同一个形状，并有测试对二者做比对。`--by subject,type,model` 把同一次读取切三种，而每一路的行都加总为存储的总量，因为余量是被报告出来而不是被丢掉；`--top` 作用于每一张表——包括随网关见过的凭据数量增长的主体表——而每张表的合计仍统计该切分下全部的行。两处刻意的拒绝：不存在的存储被报为缺失而不是被创建（打错的路径不能留下一个空数据库，读起来像是一个没有流量的存储）；存储以**只读方式打开**——SQLite 自身的只读模式，不备份、不做完整性检查、不建表、不迁移——因此 schema 比当前安装版本旧的存储会被拒绝，并说明两条出路（先启动一次网关，或传 `--migrate`，后者完全按网关的方式打开：先备份、再迁移）；而运维无法写入的存储（挂载卷、快照副本）只要 schema 是当前的就能正常读取。归属以只读方式读取网关的 keys 数据库，即一张表的两列，因为 `observability` 不得导入网关；没有它时所有 token 都被报告为未归属，且命令会在 stderr 上说明——并区分"数据库不存在"与"数据库读不出来"——而不是让一张全零的按主体表格读成"没人花过这些"。每张表都按它所展示的窗口排序、并列时由最宽窗口决定，因此一行的位置总能由旁边的数字解释。内部流量的主体与标签从网关的 keystore 迁到了 `usage.report`（`INTERNAL_SUBJECT`、`INTERNAL_LABEL`），使写入这些行的代码与对它们分组的代码无法互相漂移；`PersistenceManager.create(backup=False, migrate=False)` 是新增的一对开关——后者完全不写入地打开存储，前提是调用方自己比对过 `expected_schema_version()` 与 `schema_version()`——让以读为主的调用方读取一个不属于它的存储。参见 [可观测性——`llm-rosetta-usage` 命令](api/observability.md#the-llm-rosetta-usage-command)。

- **用量台账开始统计非 chat 的模型类型**（issue [#1019](https://github.com/Oaklight/llm-rosetta/issues/1019)，epic [#1016](https://github.com/Oaklight/llm-rosetta/issues/1016) 的第 4 阶段）：`/v1/embeddings`、`/v1/rerank` 与 `/v1/decision` 由各自的 pipeline 承载、消耗的是同一批上游额度，却**根本不写 `request_log`**——于是"用量"实际只是 chat 的用量，按 key 的面板与仪表盘的滚动总量都把这些 token 漏掉了。schema 迁移 v6 新增 `request_log.model_type`（`llm`、`embedding`、`rerank`、`decision`，`DEFAULT 'llm'`，因此历史行无需重写即保持"chat"含义）；这个轴按 registry 的词汇命名，而不是"model family"——后者在 LLM 语境里指一个模型系列。`usage/normalize.py` **按模型类型**声明"台账列 → IR 字段"的映射，因为同一列在各家族里的字段名并不相同：decision IR 本来就报 `input_tokens`/`output_tokens`，embedding 与 rerank 的 IR 类型没有输出侧，rerank 的缓存计数器叫 `cached_tokens`；未知类型会被拒绝而不是当作 chat 读取，而不提供总量的家族仍走累加规则。`UsageReport.by_model_type` 在按主体切分之外，把同一批行再按类型切一次；仪表盘日表（`/admin/api/token-usage`）按天返回同样的拆分并接受 `model_type=` 过滤——日表自身的字段含义不变，因此面板无需改动。三个处理器都通过 `OpsProxyRequest`（chat 路由已在用的写入路径）落账，而不是再长出第二条路径：两条路径以不同方式填充同一行，正是 [#1014](https://github.com/Oaklight/llm-rosetta/pull/1014) 的问题所在。注意部署后仪表盘总量与按 key 面板会**同时**开始统计这些请求——没有任何视图统计的 token，就是没人能交代的 token。

- **用量计量成为独立的子系统，位于网关之外**（epic [#1016](https://github.com/Oaklight/llm-rosetta/issues/1016)；阶段 [#1017](https://github.com/Oaklight/llm-rosetta/issues/1017) / [#1020](https://github.com/Oaklight/llm-rosetta/issues/1020)，接在 [#1014](https://github.com/Oaklight/llm-rosetta/pull/1014) 的对账修复之后）：`llm_rosetta.observability.usage` 现在负责界定"我们搬了多少 token、算在谁头上"，并且不导入任何网关模块——没有 HTTP、没有配置、没有凭据。`semantics.py` 一次性声明每种 API 标准把缓存计数器放在哪里、以及它是**可加**于输入（Messages）还是输入的**子集**（OpenAI 形、Google 形）；流式提取器与回填修复都由这份声明推导，任何转换器与之不符都会让测试失败，因此 [#1014](https://github.com/Oaklight/llm-rosetta/pull/1014) 抓到的那种"同一语义两处编码"不会再漂移。`normalize.py` 把任意家族的 usage 转成台账各列，并给出 `token_total`（厂商给了总量就采用，否则累加各计数器：思考 token 从不累加，缓存只在它未被计入输入时才累加）。`report.py` 承载让各视图能对上的分类——**已归属**、**内部**、以及按标签拆分的**未归属**余量——于是 `Σ 面板各行 == 仪表盘总量 == Σ 台账` 由构造保证而非碰巧成立；没有任何凭据的库调用方得到的是同样的数字，只不过全部落在余量里。新增的公开 API：`UsageLedger`——内存台账，记录 IR usage 字典，并对同样的行给出与 SQLite 路径完全相同的 `UsageReport`，无需存储、无需 schema，因此脚本、批处理任务或测试不用起网关就能计量用量。API Keys 面板现在把内部令牌的流量与余量各自单列一行，并在某个主体与另一个存活 key 共享标签时给出标记，而不是报出一个日志无法支撑的数字；仪表盘的滚动总量与面板现在累加同一个 `total_tokens` 列。见[可观测性——用量](api/observability.md#usage)。

- **三处"没有真正丢弃"的丢弃**（[#1010](https://github.com/Oaklight/llm-rosetta/pull/1010) 第二轮复审）：`open_responses` 的响应写出器会无条件 append 模板的返回值，而它渲染不了的服务端调用返回的是 `{}`，于是消息条目之前多出一个没有 `type` 的条目（其他格式都正常）。该分支现在与旁边的 `_append_intrinsic_result_item` 一样有守卫，并把 warnings sink 传下去。orphan fixer 只从 `assistant` 和 `tool` 消息里收集已应答的工具 id，因此历史里的配对一旦降级成普通客户端调用，落在 `role:"user"` 消息里的结果就不再计为应答，fixer 会在真实结果旁边、用同一个 id 再插入一个 `[No output available yet]`；现在任意角色的消息里的结果 id 都会被收集。另外 `_report_dropped_intrinsic` 在调用方没有传 `warnings` sink 时直接返回且不打日志，而它替换掉的各 converter 代码是无条件记日志的——pipeline 会传 sink，所以只有库调用方受静默影响；现在日志总是写出，有 sink 时优先写 sink。

- **响应与流式路径不再降级服务端调用**（intrinsic 重构的后续，来自 [#1010](https://github.com/Oaklight/llm-rosetta/pull/1010) 的复审）：这条不变量此前只在五个格式中的三个成立。`open_responses` 的**响应**路径、`chat_completions` 的**响应和流式**路径仍会把客户端从未执行过的服务端调用当成 `function_call` 交给客户端；而且 `open_responses` 流式会丢弃它自己非流式路径所降级的内容，导致客户端恰好在不流式时看到一个 function 调用。规则现在放在 base 的 `BaseToolOps` 模板里（`dropped_intrinsic_call` / `dropped_intrinsic_result`，返回空并给出 warning），不再逐个 converter 各写一遍，因此所有格式在这两条路径上都丢弃，只有**历史**路径会降级。丢弃的返回值是 falsy，所以那些无条件 append 写出结果的调用点现在会跳过它——这也正是可能把空字典塞进 Chat Completions `tool_calls` 列表的原因。同一次复审还发现两点：两个没有 id、kind 也不同的 intrinsic 调用共用同一个键，被一起改写成了最后看到的那个 kind（现在无 id 调用按位置取键，无 id 的结果与它之前的调用配对）；降级警告对"目标确实有这个 kind、只是 IR 部件缺少该形态所需字段"的情况也写成"目标无法重放"，现在两种情况会分别说明，并且对目标要求必须有结果却没有结果的调用单独报告。

- **Intrinsic 工具是派生的工具调用，跨格式服务端工具历史现在可以回放**（issue [#839](https://github.com/Oaklight/llm-rosetta/issues/839)，后续 [#996](https://github.com/Oaklight/llm-rosetta/issues/996)、[#994](https://github.com/Oaklight/llm-rosetta/issues/994)）：intrinsic 的 kind 现在是 `ToolCallPart`、`ToolResultPart`、`ToolDefinition`、`ToolCallStartEvent`、`ToolCallDeltaEvent`、`ToolResultEvent` 上的一等字段 `intrinsic_kind`，通过 `intrinsic_kind_of()` 读取，并回退到旧的 `provider_metadata` / `metadata` 写法；`set_intrinsic_kind` 已删除，`get_intrinsic_kind` / `get_definition_kind` 作为废弃别名保留。kind 不再从工具名推断。`BaseToolOps` 负责 intrinsic 的分派，converter 只覆盖钩子（`intrinsic_renderable`、`ir_intrinsic_call_to_p`、`ir_intrinsic_result_to_p`、`intrinsic_requires_result`），钩子无结果时回落到其原有的客户端路径。新的 pipeline 步骤 `degrade_intrinsic_history` 取代了历史里外来 kind 的五种 converter 本地策略：目标无法重放的调用，会连同其结果一起按 `tool_call_id` 改写成普通 function 对（带 `_downgraded_from` 标记，名字取自 kind，客户端已声明同名函数时加限定，在 warnings 里上报一次）；响应和流式路径永不把服务端调用降级成客户端调用，目标渲染不了的内容直接丢弃。`strip_intrinsic_tools` 已删除，#996 的调用/结果配对判断移到共享的 `history_native_calls`。对真实上游回放全部跨格式路径后，发现并修复了：服务端工具 id 缺少目标要求的前缀（`srvtoolu_`、`ws_`、`ci_`）；Open Responses 写出器忽略 assistant 回合里内联的 `tool_result`；降级后不带 `type` 的搜索结果块被拒（现在作为 JSON 文本携带）；Anthropic 的 `server_tool_use` 没有配对结果时按对降级；外来的代码执行载荷（`command` 与 `{code, language}`、纯输出字符串与结果对象）；以及结果被放在 `role: "tool"` 消息里而不是 assistant 回合中调用的旁边。Anthropic、OpenAI Responses、Google Interactions 之间的十二对跨格式路径现在都返回 `200`。行为变化：Anthropic 没有对应工具的服务端调用 kind（例如 Google 的 `google_search` 或 OpenAI 的 `code_interpreter`）不再以伪造的 `server_tool_use` 到达 Messages 客户端，而是从响应和流中丢弃并给出 warning。Google Generate 的流式 delta 不再重复 start 事件上的原生 part（Google 整块发送该调用，所以签名只有一个载体），reasoning 签名的 origin 与 converter tag 绑定，并有测试固定外来 origin 被丢弃的行为。覆盖请求、响应、流式三条路径的统一 warning 系统在 [#1009](https://github.com/Oaklight/llm-rosetta/issues/1009) 跟踪。参见架构指南中新增的"Intrinsic 工具是派生的工具调用"一节。

- **工具调用的分组在 IR 中升为一等字段**（issue [#999](https://github.com/Oaklight/llm-rosetta/issues/999)，父 issue [#953](https://github.com/Oaklight/llm-rosetta/issues/953)）：工具**调用**现在通过 `ToolCallPart.tool_group`（流式形态为 `ToolCallStartEvent.tool_group`）记录它所属的容器 —— 与工具**定义**一直在用的字段同名同义 —— 而不再藏在私有的 `provider_metadata["namespace"]` 里。Open Responses 转换器从 wire 读到 `namespace` 时写入该字段，序列化回 wire 时再读出来，非流式与流式两条路径都如此；流式路径下，上游 `function_call` 的 namespace 现在也会带到 `tool_call_start` 事件上，此前只有非流式路径会携带。读取保持向后兼容：只带旧 metadata 键的调用仍能解析（经新增的 `tool_call_group` 辅助函数），而写入一等字段时会清除旧的 metadata 键，两种拼写因此不会各说各话。
- **DeepSeek 与火山引擎 Responses shim 开启工具分组**（issue [#998](https://github.com/Oaklight/llm-rosetta/issues/998)，父 issue [#953](https://github.com/Oaklight/llm-rosetta/issues/953)）：`deepseek--open_responses` 与 `volcengine--open_responses` 现声明 `tools.tool_groups: true`。发往这两个 provider 的请求会保留客户端声明的 `namespace` 工具容器，不再被扁平化为带前缀的函数名，从而与两个上游原生支持的容器行为对齐。shim 加载器（`_load_single_provider`）现已支持从 `provider.yaml` 中解析 `tools.tool_groups`。通过嵌套 `tool_choice` 形式指定容器子工具仍然无法承载：IR `ToolChoice` 没有分组槽位，读取侧会回退到 `auto`，现在这次回退会写入转换 warning，而不再静默丢弃（父 issue [#953](https://github.com/Oaklight/llm-rosetta/issues/953)）。
- **面向能承载工具分组的目标**（issue [#953](https://github.com/Oaklight/llm-rosetta/issues/953)）：shim 现在可以声明 `tools.tool_groups: true`，Responses 写入侧就会把客户端的分组以 `{"type": "namespace", …}` 容器的形式重新发出，而不再把组名折进工具名里。这个声明是**按 provider** 的、不能由 base format 推导：OpenRouter 的 Chat Completions 通道接受容器而 OpenAI 的不接受，Anthropic 与 Google 只对自家 MCP 服务器带来的工具分组。容器内的子工具按客户端自己的名字发出。组随名字一起上线，命名阶段因此只在本容器的成员之间去重，顶层的 `wait` 不再逼着 `tasks` 里的同名子工具以 `tasks_wait` 上线。DeepSeek 和 Volcengine 上实测通过，两者都接受与顶层工具同名的子工具，并在调用上回显 namespace。响应腿把这个回显当作身份的一半，即 `(name, namespace)`，因此同名会解析到容器子工具而不是顶层工具；库为无名容器生成的组会被丢弃，不会交给客户端。客户端没给描述的容器以 `""` 发出，因为 OpenAI 和 OpenRouter 拒绝缺省描述并返回 `400`，却接受空字符串。同一改动移除了 contested 名字机制：生成组 id 让它不可达，八种碰撞形状走 pipeline 没有一种能触发它。

- **管理面板 API 密钥新增按 key 的令牌用量**（issue [#967](https://github.com/Oaklight/llm-rosetta/issues/967)）：管理台 API Keys 标签页现在按滚动窗口 5h / 24h / 7d / 30d 展示每个 key 的请求数与令牌总量，分段控件由接口返回的窗口列表生成。归属按**凭证**而非标签：`KeyStore.create_key` 在创建时推导一次稳定 bucket `uuid5(namespace, sha256(secret))`（取完整 hex —— bucket 只作分组键，从不展示），轮换时重算，因此轮换后的 key 从零开始；bucket 挂在 `KeyContext` 上，由代理写入新增的 `request_log.api_key_bucket` 列（schema 迁移 v4）。报表由一条走索引的 `GROUP BY` 现算各窗口的条件求和 —— 不建 rollup 表，日志读取也改为显式列名、不再依赖物理列序。因此重命名 key 不再影响其用量统计；没有对应存活 key 的 bucket 会被丢弃，未归属的行（内部令牌、匿名）写 NULL 而非空字符串。已有 key 的 bucket 依据存储的 key hash 回填；历史日志行不做归属推断，不按标签猜测。用量单元格上的 tooltip 会说明轮换会启用新 bucket。（此条描述的归属规则已由上方的用量子系统条目扩展：凭据已不存在的行现在作为**未归属**上报而非被丢弃，因此面板与仪表盘仍然对得上，内部令牌也有了自己的 bucket。）
- **`scripts/asksage-token.py`**：AskSage 的推理端点不接受账号 API key，只接受由 `POST {host}/user/get-token-with-api-key` 换来的短期 access token，因此 `asksage--*` provider 此前根本无法通过 `token_command` 接入。该脚本执行这次交换并把 token 打到 stdout（诊断信息走 stderr），可以像 `alcf-token.py` 一样直接填进 provider 的 `token_command`。`--status` 从 JWT 的 `exp` 声明读出有效期，`--probe` 发一个最小的 `/responses` 请求确认 token 真的被接受。凭据只从环境变量读取（`ASKSAGE_TOKEN_EMAIL`、`ASKSAGE_TOKEN_API_KEY`，并兼容 ANL 示例客户端的 `ASK_SAGE_API`），不从命令行参数读取——进程表会把参数暴露给机器上的其他进程。这两个变量名刻意避开了 shim 的 `api_key_env`：每个 `asksage--*` shim 都声明 `connection.api_key_env: ASKSAGE_API_KEY`，而网关会先把该变量的值填成 provider 的 `api_key`、再去读 `token_command`，随后以互斥为由报错——也就是说只要 `ASKSAGE_API_KEY` 存在于网关进程的环境里，`token_command` provider 根本建不起来。两个 AskSage host 的凭据也不通用：Argonne 的 API key 会被 `api.asksage.ai` 的交换拒绝，返回 `{"response": "Key doesnt exist", "status": 400}`；而 `GET {host}/server/openai/v1/models` 不校验凭据（伪造的、空的、甚至不带 auth header 都返回同样的列表），因此能列出模型并不能说明 key 可用。
- **外部 Shim 源**（epic [#927](https://github.com/Oaklight/llm-rosetta/issues/927)，issue [#666](https://github.com/Oaklight/llm-rosetta/issues/666)）：新增顶层 `shims` 配置段，让 gateway 可以从额外目录（`shims.paths`）和远程归档（`shims.remotes`）加载 provider shim，归档会被拉取并缓存到本地。这样一套 shim 可以维护在独立仓库里，**无需往 gateway 镜像里安装任何东西**即可消费 —— 官方镜像既没有 `pip` 也没有 `git`。远程拉取不依赖 git：GitHub 仓库 URL 会被改写到 codeload 的 tarball 端点，其它站点给出显式归档 URL（可含 `{ref}`）即可，因此 GitLab 和自建 forge 都不需要按站点写代码。省略 `ref` 时会解析成具体 commit SHA 并记录进缓存 stamp（配置文件本身不会被改写）；固定 `ref`、给出 `sha256`、或 `refresh: "never"` 只拉取一次，`refresh: "always"` 每次都拉，默认 `auto` 在超过 `remote_ttl` 后重新拉取。拉取失败时会告警并回退到旧缓存。解压会拒绝绝对路径、目录穿越与符号链接/硬链接成员，下载体积与解压体积都有上限。某个源失败时默认只告警并跳过（在管理面板中可见），除非 `shims.strict: true` 使其直接拒绝启动。新增公开 API：`llm_rosetta.shims.rebuild_shims()` / `resolve_remote()` 与 `llm_rosetta.gateway.apply_shim_sources()`；`create_app` 会自动应用该段。管理面板 Settings 标签页新增 **Shim Sources** 区块，可编辑 paths/remotes、查看每个源的加载状态，并重新加载或强制拉取。注意：声明的源可以携带在加载时被执行的 `transforms.py`（与安装插件包同一信任级别），因此应固定版本并只使用信任的源。
- **OpenCode 付费 shim**（issue [#924](https://github.com/Oaklight/llm-rosetta/issues/924)）：新增 `opencode` shim 组 —— `opencode--chat_completions`、`opencode--messages`、`opencode--open_responses`，指向 OpenCode 的 Go 订阅通道（`https://opencode.ai/zen/go/v1`）；Zen 按量付费通道即同一组 shim 配合 `base_url: https://opencode.ai/zen/v1`。由于 Go 通道要求每个推理请求都带稳定的 `x-opencode-session`，因此新增通用 shim 字段 `connection.session_header`：声明后，网关会注入该头，其值按会话派生（复用 affinity identity —— 同一会话始终得到同一个 id），客户端自带该头时则原样透传。keyless 免费通道另由 issue [#884](https://github.com/Oaklight/llm-rosetta/issues/884) 跟踪。
- **重复工具定义处理** (issue [#903](https://github.com/Oaklight/llm-rosetta/issues/903))：新增 IR 层 pass `dedupe_tool_definitions`（位于 `capabilities.py`，由 `ConversionPipeline.convert_request` 调用），将逐字节相同的重复工具定义折叠为首次出现的那一个，并对「同名但定义不同」的冲突发出 warning、但不丢弃任何一方。来自具名 namespace 容器的工具不受影响——namespace 是其身份的一部分；但自身没有名字的容器不属于此列：它把子工具摊平后 `namespace` 为 `""`，而空字符串限定不了任何东西，因此其中逐字节相同的子工具与其他重复项一视同仁，现在同样会被折叠，而不再以重复工具名的形式抵达严格的上游（issue [#913](https://github.com/Oaklight/llm-rosetta/issues/913)）。Anthropic、DeepSeek、xAI 和 Google 都会以 `400` 拒绝重复工具名，而 OpenAI 等则接受，因此同一个客户端请求此前会因目标 provider 不同而时成时败；现在完全相同的重复项在所有目标上都能通过，真正的名称冲突也可归因，而不再以含糊的上游报错呈现。
- **免费模型源 —— Kilo** (PR [#888](https://github.com/Oaklight/llm-rosetta/pull/888), issue [#883](https://github.com/Oaklight/llm-rosetta/issues/883), epic [#885](https://github.com/Oaklight/llm-rosetta/issues/885))：新增 `kilo--openai_chat` shim，指向 Kilo 公共网关。新增 shim 字段 `display_name`、`free_source` 和 `connection.keyless`；keyless provider 在未配置 `api_key` 时不发送任何鉴权头，配置了则回退到 Bearer，因此同一 shim 既服务免费池也服务付费账号。作为 keyless 条目时，管理台将其命名为 **Kilo (Free)**，归入新的 **Free Resource** 分区（未配置免费源时不显示），在卡片上提供 **Refresh** 按钮（重新拉取清单并提供一键应用新增模型），并显示中性的第三方披露。该 shim 的 `model_list_transform` 只暴露上游 `isFree` 切片。
- **管理台 provider type 选择器新增「基础格式」** (PR [#915](https://github.com/Oaklight/llm-rosetta/pull/915), epic [#885](https://github.com/Oaklight/llm-rosetta/issues/885))：Provider Type 下拉现在分 **Base Formats**（五个 chat 标准 + `open_responses`，来自 `_PROVIDER_REGISTRY` 旁的 `BASE_FORMATS` 常量）与 **Providers** 两组，并在格式名与厂商 shim 名不同时标注推荐 shim。现在可以直接建一个「裸格式」provider 指向自建或兼容端点，不必手改配置。
- **外部 shim 的 transforms** (PR [#892](https://github.com/Oaklight/llm-rosetta/pull/892), issues [#272](https://github.com/Oaklight/llm-rosetta/issues/272)/[#273](https://github.com/Oaklight/llm-rosetta/issues/273))：从包外加载的 shim 现在可以在 `transforms.py` 里使用相对导入。插件加载器把每个 providers 根目录注册为合成包命名空间（按 resolve 后的路径取 key，因此两个同名 group/leaf 的根目录互不干扰），失败加载会回滚本次注册的模块，重载时清理命名空间。

- **统一 Ops 审计层** (PR [#856](https://github.com/Oaklight/llm-rosetta/pull/856), issue [#846](https://github.com/Oaklight/llm-rosetta/issues/846), 子 PR [#851](https://github.com/Oaklight/llm-rosetta/pull/851)/[#852](https://github.com/Oaklight/llm-rosetta/pull/852)/[#853](https://github.com/Oaklight/llm-rosetta/pull/853)/[#854](https://github.com/Oaklight/llm-rosetta/pull/854)/[#855](https://github.com/Oaklight/llm-rosetta/pull/855))：新增 `gateway/ops/` 包，每个网关端的数据变更操作（数据清理、密码修改、token 轮转、API key 增删改、周期性清理）都是一个带自动审计记录的类型化 `OpsBase` 子类，包括失败操作也会被记录。弥补了破坏性管理操作不留痕迹的审计缺口。原本分散的 `_record_telemetry()`/`_log_key_event()`/内联 `ops_log.add()` 调用点替换为 `OpsProxyRequest`、`OpsClearData`/`OpsCleanupData`/`OpsTrimData`/`OpsVacuum`/`OpsRebuildMetrics`/`OpsClearOpsLog`/`OpsPrune`/`OpsPeriodicCleanup`、`OpsPasswordChange`/`OpsTokenRotate`/`OpsSessionLogoutAll`，以及 `OpsKeyCreate`/`OpsKeyUpdate`/`OpsKeyDelete`/`OpsKeyRotate`。
- **持久化/DB 模块重设计** (PR [#869](https://github.com/Oaklight/llm-rosetta/pull/869), issue [#848](https://github.com/Oaklight/llm-rosetta/issues/848), 子 PR [#859](https://github.com/Oaklight/llm-rosetta/pull/859)/[#860](https://github.com/Oaklight/llm-rosetta/pull/860)/[#861](https://github.com/Oaklight/llm-rosetta/pull/861)/[#863](https://github.com/Oaklight/llm-rosetta/pull/863)/[#864](https://github.com/Oaklight/llm-rosetta/pull/864)/[#866](https://github.com/Oaklight/llm-rosetta/pull/866)/[#867](https://github.com/Oaklight/llm-rosetta/pull/867))：异步 `config_lock()`（不再使用阻塞事件循环的 `fcntl.flock`）；周期性 WAL checkpoint（5 分钟一次，64MB 上限）并在启动时检查；schema 版本表配合前向迁移注册表和外键约束；`RetentionPolicy`/`RetentionTracker` 取代原本四套独立的清理逻辑；不受清理影响的单调递增计数器 `lifetime_total_requests`/`lifetime_total_errors`；`KeyStore` 由同步 `sqlite3` 迁移到异步 `aiosqlite`；启动时通过 `sqlite3.backup()` 备份并执行完整性检查。
- **转换流水线性能优化** (PR [#847](https://github.com/Oaklight/llm-rosetta/pull/847), issue [#828](https://github.com/Oaklight/llm-rosetta/issues/828), 子 PR [#842](https://github.com/Oaklight/llm-rosetta/pull/842)/[#843](https://github.com/Oaklight/llm-rosetta/pull/843)/[#844](https://github.com/Oaklight/llm-rosetta/pull/844))：三层 IR 校验缓存优化（`pickle` 缓存键、`GenerationalSet` 淘汰策略、epoch guard 跳过未变化列表的逐项哈希），将 `anthropic → anthropic` 热路径转换从约 18ms 降至约 2.7ms。Google converter 工具结果名称解析优化为 O(1)（原为 O(n²)）。新增 `scripts/bench_pipeline.py` / `make bench` 基准测试套件，覆盖全部 10 条 converter 路径的合成负载与分阶段计时。
- **`ToolMessage.batch_id`** (PR [#841](https://github.com/Oaklight/llm-rosetta/pull/841), issue [#837](https://github.com/Oaklight/llm-rosetta/issues/837))：新增 `batch_id: NotRequired[str]` 字段，用于标记同一 assistant 工具调用轮次产生的工具结果。`assign_tool_batch_ids()` 和 `merge_tool_messages()` 修复了一个结构性缺陷：Anthropic 和 Google generateContent 的出站 converter（要求所有工具结果位于单条消息中）此前仅依赖消息相邻关系来分组并行工具调用，并不可靠。
- **流式处理便捷方法** (PR [#873](https://github.com/Oaklight/llm-rosetta/pull/873), issue [#872](https://github.com/Oaklight/llm-rosetta/issues/872))：`StreamProcessor.iter_events(chunks)` 和 `BaseConverter.iter_stream_from_provider(chunks, context)` 遍历上游 chunk 并在迭代结束时自动刷出延迟的终止事件，调用方不再需要手动调用 `process_chunk(None)`。
- **Decision 模型范式** (PRs [#705](https://github.com/Oaklight/llm-rosetta/pull/705), [#709](https://github.com/Oaklight/llm-rosetta/pull/709), [#711](https://github.com/Oaklight/llm-rosetta/pull/711), [#718](https://github.com/Oaklight/llm-rosetta/pull/718))：用于概率化结构决策的新模型类别。三种 IR 问题类型（`assertion`、`choice`、`score`），支持 LLM structured output、reranker 相关性评分和 embedding 余弦相似度三种 converter 后端。网关路由 `/v1/decision` 和 `/v1/systemone`。
- **Decision IR：共享 entry 抽象** (PR [#887](https://github.com/Oaklight/llm-rosetta/pull/887), issue [#889](https://github.com/Oaklight/llm-rosetta/issues/889))：三种 IR 问题类型统一为 `criteria: list[DecisionEntry]` —— `assertion` 为 0 或 2 个 label 为 `False`/`True` 的 entry；`choice`/`score` 为 N 个按 `label` 索引的 entry。概率分布按 `str(label)` 作 key（score 去掉 `legend`）；新增 `RefusalAnswer`、多模态 `state`（content parts）、可选 `confidence`/`unknown_probability`，推导逻辑位于 `converters/decision/derived.py`。
- **OpenAI Decisions converter** (PR [#887](https://github.com/Oaklight/llm-rosetta/pull/887), issue [#889](https://github.com/Oaklight/llm-rosetta/issues/889))：`OpenAIDecisionsConverter` 把 IR 映射到 OpenAI 的 `POST /v1/decisions` wire（`input` + `questions` **数组**，类型为 `predicate`/`choice`/`score` 且带 `name`；答案按 `name` 作 key，含 `refusal`）。注册为 `openai_decisions` decision 格式；网关路由 `/v1/decisions`；provider shim `openai_decisions`。
- **System One 多模态（`images[]`）** (issue [#889](https://github.com/Oaklight/llm-rosetta/issues/889))：`SystemOneDecisionConverter` 现在读写 System One 家族的顶层 `images` 扩展（Cloudflare Clef、classifier.dev）——IR 的 `state` content parts 与 `state`（文本）+ `images`（内联 base64 data URL）双向映射，多模态决策证据可与 Jev 家族 provider 往返。
- **Decision 格式更名** (PR [#887](https://github.com/Oaklight/llm-rosetta/pull/887), issue [#889](https://github.com/Oaklight/llm-rosetta/issues/889))：System One decision 格式更名为 `systemone`（原 `typesafe`），converter 类更名为 `SystemOneDecisionConverter`（原 `TypeSafeDecisionConverter`）——以 System One 这一由 TypeSafe/Jev、Clef、Laya、classifier.dev 共用的中立标准命名，而非单一 provider。
- **Decision 源头格式探测** (issue [#889](https://github.com/Oaklight/llm-rosetta/issues/889))：网关 decision handler 现在从路由和请求体探测源头格式（`/v1/decisions` → `openai_decisions`，`/v1/systemone` → `systemone`，否则按 body 形状），不再只依赖 `default_decision_format`。
- **多 Provider 路由** (PRs [#664](https://github.com/Oaklight/llm-rosetta/pull/664), [#663](https://github.com/Oaklight/llm-rosetta/pull/663), [#662](https://github.com/Oaklight/llm-rosetta/pull/662))：每个模型可配置多个上游 Provider，支持加权轮询、API key 亲和性优化 prompt cache 命中率、按请求 token 用量追踪。
- **ALCF Provider 支持** (PRs [#673](https://github.com/Oaklight/llm-rosetta/pull/673), [#675](https://github.com/Oaklight/llm-rosetta/pull/675), [#679](https://github.com/Oaklight/llm-rosetta/pull/679), [#689](https://github.com/Oaklight/llm-rosetta/pull/689), [#680](https://github.com/Oaklight/llm-rosetta/pull/680), [#681](https://github.com/Oaklight/llm-rosetta/pull/681)):ALCF 推理集群的 shim —— 由独立的 [llm-rosetta-shim-alcf](https://github.com/Oaklight/llm-rosetta-shim-alcf) 包提供 —— 包含 Globus OAuth2 token 管理、401 自动刷新和 30 天 session 策略检测。`token_command` 配置字段支持任何有过期 token 的 Provider 动态刷新 API key。
- **AskSage provider 支持**（PR [#814](https://github.com/Oaklight/llm-rosetta/pull/814)）：AskSage OpenAI、Anthropic 和 Gemini 端点的 shim，由独立的 [llm-rosetta-shim-asksage](https://github.com/Oaklight/llm-rosetta-shim-asksage) 包提供。新增 `connection.auth_header` 字段支持 per-shim auth header 覆盖。
- **模型类型注册表** (PRs [#730](https://github.com/Oaklight/llm-rosetta/pull/730), [#735](https://github.com/Oaklight/llm-rosetta/pull/735), [#736](https://github.com/Oaklight/llm-rosetta/pull/736), [#737](https://github.com/Oaklight/llm-rosetta/pull/737))：声明式模型类型注册表（LLM、embedding、rerank、decision）。添加新类型只需一个描述符——admin UI 徽章、provider 弹窗、测试菜单和路由全部从注册表元数据驱动。
- **Provider 熔断器** (PR [#727](https://github.com/Oaklight/llm-rosetta/pull/727))：三态熔断器，对持续高错误率的 Provider 短路请求。可按 Provider 配置，默认禁用。
- **Admin 面板改进**：hash 路由标签页导航，支持浏览器前进/后退；独立 Settings 标签页；服务运维日志，支持事件过滤和保留策略；httponly cookie session 认证；日志/错误转储批量清理；token 用量明细弹窗；多窗口速率限制；多 Provider 模型编辑与内联权重控制。
- **路由智能** (PRs [#784](https://github.com/Oaklight/llm-rosetta/pull/784), [#785](https://github.com/Oaklight/llm-rosetta/pull/785), [#770](https://github.com/Oaklight/llm-rosetta/pull/770), [#676](https://github.com/Oaklight/llm-rosetta/pull/676))：hop-count header 路由环检测、同格式 provider 亲和性避免不必要转换、cache 感知亲和性路由、rendezvous hashing key 亲和性。
- **保真度验证** (PRs [#772](https://github.com/Oaklight/llm-rosetta/pull/772), [#796](https://github.com/Oaklight/llm-rosetta/pull/796))：持久化往返转换基线与回归检测。同格式路由 always-on shadow diff。
- **双向工具命名空间映射** (PR [#753](https://github.com/Oaklight/llm-rosetta/pull/753))：Provider 间工具名无冲突翻译，支持往返保持。
- **可观测性** (PRs [#702](https://github.com/Oaklight/llm-rosetta/pull/702), [#755](https://github.com/Oaklight/llm-rosetta/pull/755), [#756](https://github.com/Oaklight/llm-rosetta/pull/756))：cache 和 reasoning token 追踪、TTFB 指标、客户端断连检测、vendored tracing profiler 替代 pyinstrument。
- **工具名冲突显式化** (issue [#907](https://github.com/Oaklight/llm-rosetta/issues/907))：`dedupe_tool_definitions` 的冲突分支现在同时输出到客户端和 error dump。当请求以同一个工具名声明了多个不同定义、并被上游以 `400` 或 `422` 拒绝请求体时，网关会把一段 `[llm-rosetta]` 提示追加到客户端收到的 `error.message`，点名冲突的工具名，使一句裸的 "Tool names must be unique" 可以归因到客户端自己的工具列表，而不是看起来像网关故障。结构化映射（`{名称: 不同定义数}`，最多列出十个名字）会以新增的 `tool_name_collisions` 列写入 error dump（schema migration v3），并在 admin 的 dump 查看器中展示。解析后的错误文档只有 `error.message` 发生变化；响应体会被重新序列化，因此空白与转义可能被规整，而非 ASCII 文本保持可读。非 JSON 响应体、信封中没有字符串 message 的响应体、或已带有该标记的响应体一律保持原样；`401`/`403`/`404`/`429` 与 `5xx` 同样不做标注。dump 这一半已接好线并有单元测试，在 dump 行能正常写入的环境下也会出现在 admin 查看器中；但在 2026-10-08 及之后新建的库上尚不可端到端观测，因为 error dump 的插入会被 `request_log` 外键挡住——request log 行是在错误路径之后才写入的——于是既没有 dump，也就没有冲突映射（issue [#964](https://github.com/Oaklight/llm-rosetta/issues/964)）。早于该外键存在的库不受影响。

- **Embedding 与 rerank 端点补全，并加入 Google GenAI**（epic [#1050](https://github.com/Oaklight/llm-rosetta/issues/1050)，issue [#926](https://github.com/Oaklight/llm-rosetta/issues/926)）：`google_generate` 加入 embedding 转换器注册表——单输入用 `embedContent`，多输入用 `batchEmbedContents`，模型放在 URL 中，批量响应不带用量元数据——因此 `/v1/embeddings` 现在可在 OpenAI、Cohere、Jina、Voyage 与 Google 之间转换。八个提供方 shim（`openai--embedding`、`cohere--embedding`、`cohere--rerank`、`jina--embedding`、`jina--rerank`、`voyage--embedding`、`voyage--rerank`、`google--embedding`）声明各自端点的格式、`connection.default_path` 与连接默认值，统一提供方只需一行 `embedding_shim` / `rerank_shim` 即可承载 embedding 或 rerank 流量。源格式检测改由 model type descriptor 负责（`gateway/source_formats.py`），并识别 Google 形状的请求体（`content.parts`、`requests`）。`default_embedding_format` 新增 `google_generate`。参见 [embedding 格式](guide/embedding-formats.md) 与 [非 LLM 端点](gateway/configuration.md#non-llm-endpoints)。

- **Decision 端点与 embedding、rerank 采用同一套解析方式**（issue [#926](https://github.com/Oaklight/llm-rosetta/issues/926)）：`decision` descriptor 现在负责源格式探测与各 format 的上游路径——`openai_decisions` 使用 `/v1/decisions`，`systemone` 使用 `/v1/systemone`——统一条目只需声明 `decision_format`。旧的 `decision_providers` 段现在会被解析（此前被忽略），其中文档写法的 `"type": "typesafe"` 会经由 shim 解析为对应 format；`decision_models` 也接受 failover 的 `providers` 列表。两个内置 shim 改用规范名 `openai--openai_decisions` 与 `typesafe--systemone`，裸名保留为别名。参见 [decision 格式](guide/decision-formats.md)。


### 变更（**破坏性变更**）

- **（破坏性变更）Argo shim 离开核心包**（epic [#927](https://github.com/Oaklight/llm-rosetta/issues/927)）：`argo--chat_completions`、`argo--messages` 与 `argo--open_responses` 现在和 ALCF、AskSage 一样，随独立的 [llm-rosetta-shim-argo](https://github.com/Oaklight/llm-rosetta-shim-argo) 包发布。**移除原计划于 0.16.0，本周期改为 0.14.0 —— 在任何稳定版发布该时间表之前**，因此目前不再有任何计划移除的内置 shim。shim 名称不变，但使用过内置版本的配置必须先声明源 —— `"shims": {"presets": ["argo"]}` —— 否则启动会报错并指明包与需要声明的条目。旧别名 `argo--openai_chat`、`argo--anthropic` 与 `argo--openai_responses` 不再解析到 shim；它们仍作为拼写映射保留，使旧配置先规范化并进入那条指明包名的错误 —— 请改用规范名称。`scripts/check_shim_removals.py` 仍会约束未来的移除；不过清单为空时启动告警与检查都是惰性的：绿色的 `shim-removal-check` 目前并不构成覆盖证据。

  **迁移：** 添加预设后重载 gateway：

  ```jsonc
  { "shims": { "presets": ["argo"] } }
  ```

  独立包注册相同的规范名称，因此提供方条目无需改动。参见[独立 Shim 包](guide/shims.md#standalone-shim-packages)。

- **（破坏性）Embedding 流量不再回退到 chat 提供方**（issue [#926](https://github.com/Oaklight/llm-rosetta/issues/926)）：当模型没有 embedding 路由时，`/v1/embeddings` 会把请求不经过转换直接转发给该模型的 chat 提供方，前提是上游恰好实现了 OpenAI 形状的 embedding。这个前提并不属于 chat 提供方的契约，而且该回退会让写错的模型名落到一个无关端点上。现在 embedding 请求只通过 embedding 端点路由：没有 `embedding_models` 条目（统一提供方上没有 `"type": "embedding"`）的模型返回 `404`。

  **迁移：** 声明端点。统一提供方添加 `embedding_format`（如 `"openai"`、`"cohere"`、`"google_generate"`）或 `embedding_shim`（如 `"openai--embedding"`、`"google--embedding"`）；使用旧配置段时，把模型加入 `embedding_models`、提供方加入 `embedding_providers`。


- **（破坏性）ALCF shim 采用规范的 `alcf_<cluster>--<format>` 名称，选择器的分组改为声明而非解析**（epic [#927](https://github.com/Oaklight/llm-rosetta/issues/927)，PR [#1035](https://github.com/Oaklight/llm-rosetta/pull/1035)）：shim 名的形式是 `vendor--base`，管理台选择器也按此读取，但 `alcf--metis` 把*部署*放在了本应属于 wire 格式的位置——这正是 Provider Type 下拉框显示出 `metis · chat_completions` 的原因（有两处调用点在标签不是某个 base 时会追加 base），也是该语法无处安放每个 cluster 的第二种格式的原因。于是 cluster 移入 provider 段（`alcf--metis` → `alcf_metis--chat_completions`，`minerva` 与 `sophia` 同理），而三种旧拼写通过 `LEGACY_SHIM_ALIASES` 继续可用：`get_shim()` 仍能解析它们，`correct_legacy_names` 会在加载时改写配置里的 `providers.*.shim` / `.type`，并照例留下一次性的 `.pre-migration.bak`。它们会与其余旧名一起在 0.16.0 移除。（**同一周期内被取代：** ALCF shim 已在 0.14.0 离开核心，因此重命名后的内置 shim 及其别名不会发布；独立的 `llm-rosetta-shim-alcf` 包注册规范名称。下文的 `group` / `group_label` / `display_name` 语义仍然有效。）分组同时不再是推断：`provider.yaml` 新增 `group`（顶层）与 `group_label`（其标题），二者都回退到名字切分，因此两者都不声明的 shim——包括所有第三方与外部 shim——渲染结果与以往完全一致。`display_name` 回归其文档含义，即该 shim 自己的标签，并且**仅当 shim 同时声明了分组字段时**才提供行标签；对没有声明分组字段的 shim，则保留旧读法，即由 `display_name` 充当组标题——否则厂商名会被打印两次，一次作为组标题、一次出现在它唯一的行上，而即将迁移的 shim 恰恰就是按这套正被取代的 schema 编写的第三方 shim。Kilo 的 `display_name: Kilo` 出于同样的原因改为 `group_label: Kilo`。现在每个解析出的值都携带产生它的分支（`declared` / `derived` / 旧的 `display_name`），因此「未声明的 shim 按设计不受影响」与「它恰好今天不受影响」始终可以区分开来。树构建器从 `tabs/providers.js` 移到纯函数式的 `core/shim-tree.js`，并由 `tests/js/shim-tree.test.mjs`（`make test-js`，作为 CI 步骤运行）将它对照 `group` 之前那个构建器的一份参照实现（oracle）进行比对。同时还恢复了 [#917](https://github.com/Oaklight/llm-rosetta/pull/917) 遗留的各厂商 `--anthropic` 别名：自它把 `anthropic` base 重命名为 `messages` 以来，`argo--anthropic`、`asksage--anthropic`、`minimax--anthropic` 与 `openrouter--anthropic` 一直无法解析；现在只要某个别名指向未注册的 shim，`test_every_shim_alias_target_is_registered` 就会失败。

- **（破坏性）Google SDK 形态把 `system_instruction` 移入 `config`** (issue [#936](https://github.com/Oaklight/llm-rosetta/issues/936))：`GoogleGenerateConverter` 在 `output_format="sdk"`（默认值，也是 `convert()` / `convert_response()` 返回的形态）下此前会输出 `{"model", "contents", "config", "system_instruction"}`。这个顶层键让该字典无法按文档所述的方式使用：`client.models.generate_content(**body)` 在从 1.0.0 到 2.29.0 的所有 `google-genai` 版本上都会抛 `TypeError: unexpected keyword argument 'system_instruction'`，因为 `Models.generate_content()` 只接受 `model`、`contents` 和 `config`；而绕过它、只传这三个参数会让 system instruction 静默地从请求里消失。现在该形态**正好就是** `generate_content` 的关键字参数，指令位于 `config` 内部——那才是 `GenerateContentConfig` 声明它的位置，也是 `llm_rosetta.types.google.GenerateContentRequest` 早已记录的形态。`request_from_provider` 两种位置都能读取。REST 形态（`output_format="rest"`，即 gateway 路径）逐字节不变。

  **迁移：** 读取 `convert()` / `convert_response()` / `request_to_provider(output_format="sdk")` 的输出时，把 `body["system_instruction"]` 改为 `body["config"]["system_instruction"]`。

  | 变更前 | 变更后 |
  |--------|--------|
  | `{"model", "contents", "config", "system_instruction"}` | `{"model", "contents", "config": {"system_instruction": …}}` |

  此前会抛 `TypeError` 的 `generate_content(**body)` 调用现在可以直接工作，且 system instruction 会真正送达模型。

- **（破坏性）`tool_ops` provider 名规范化** (issue [#921](https://github.com/Oaklight/llm-rosetta/issues/921), epic [#905](https://github.com/Oaklight/llm-rosetta/issues/905))：`tool_ops` 便利 API 现以 **base converter 类型**作为规范 provider 名——`anthropic`、`google_generate`、`google_interactions`、`chat_completions`、`open_responses`。此前所有写法（`openai_chat`、`openai_responses`、`google`、`openai-chat`、`openai-responses`、`open-responses`、`google-genai`、`google-generate`、`chat-completions` 等）仍可用但会发出 `DeprecationWarning`，并在 **0.16.0** 移除（[#919](https://github.com/Oaklight/llm-rosetta/issues/919)）。迁移：`tool_ops.to_provider(tool, provider="openai-chat")` → `provider="chat_completions"`。

- **Gateway 模块重组** (PRs [#739](https://github.com/Oaklight/llm-rosetta/pull/739), [#738](https://github.com/Oaklight/llm-rosetta/pull/738), [#744](https://github.com/Oaklight/llm-rosetta/pull/744))：后端文件移入 `gateway/middleware/` 和 `gateway/pipelines/` 子包；admin JS 移入 `js/core/`、`js/tabs/`、`js/components/`。**旧的扁平 import 路径（`gateway.auth`、`gateway.embeddings` 等）不再有效**——请使用 `gateway.middleware.*` / `gateway.pipelines.*`。参见 `argo-proxy` [#179](https://github.com/Oaklight/argo-proxy/pull/179) 的迁移。

- **（破坏性）Anthropic base 重命名为 `messages`，shim 名称规范化** (PR [#923](https://github.com/Oaklight/llm-rosetta/pull/923), issue [#917](https://github.com/Oaklight/llm-rosetta/issues/917))：Anthropic Messages API 的 base 类型现为中性名 `messages`——该格式已被众多 provider 复刻而成为事实标准，因此不再绑定 `anthropic` 厂商。converter 模块移至 `llm_rosetta.converters.messages`，类名改为 `MessagesConverter`（`AnthropicConverter` / `converters.anthropic` 保留为已弃用别名）。Shim 名称现统一为规范形式 `<provider>--<base>`：`openai--chat_completions`、`openai--open_responses`、`anthropic--messages`、`google--google_generate`、`google--google_interactions`，其余为 `<provider>--chat_completions` / `<provider>--open_responses`。shim 注册表只认规范名——base *格式* 永远不会解析到厂商 shim（`get_shim("google_generate")` 返回 `None`；`google_generate` 是格式名，`google--google_generate` 才是 shim 名）。Base converter 现为厂商中立：裸 `type: <base>` 不再附带隐式的厂商 `base_url` 或 API-key 环境变量——它们仅在管理台作为占位提示显示，因此必须显式设置 `base_url` 或使用 `shim:`。为便于过渡，网关会在启动时改写 `providers.*.shim` / `providers.*.type` 中的旧名，只改写一次且幂等，并保存 `.pre-migration.bak`；无法映射的名称会以指向文档的 `ValueError` 拒绝。旧名在库边界仍会警告并解析，直至 **0.16.0**（[#919](https://github.com/Oaklight/llm-rosetta/issues/919)）移除。

  | 旧名 | 规范名 |
  |--------|-----------|
  | base `anthropic` | `messages` |
  | shim `anthropic` | `anthropic--messages` |
  | shim `openai` | `openai--chat_completions` |
  | shim `openai_responses` | `openai--open_responses` |
  | shim `google` / `google_generate` | `google--google_generate` |
  | shim `google_interactions` | `google--google_interactions` |
  | shim `<provider>--openai_chat` | `<provider>--chat_completions` |
  | shim `<provider>--openai_responses` | `<provider>--open_responses` |
  | type `openai_chat` | `chat_completions` |
  | type `google` / `google-genai` | `google_generate` |
  | base `google_generate` / `google_interactions` | 不变 |

### 变更

- **升级网关变成一个看得见的版本对**（epic [#1127](https://github.com/Oaklight/llm-rosetta/issues/1127)、PR [#1164](https://github.com/Oaklight/llm-rosetta/pull/1164)）：拆分之后网关只声明了裸的 `llm-rosetta` 依赖，pip 可以把它与任意核心配在一起；而 `--version` 打印的是 `llm_rosetta.__version__`——核心的版本号，当时相等，一旦两者漂移就是错的。现在依赖写成 `llm-rosetta>=X.Y.0.dev0,<X.(Y+1)`（同 minor 兼容：核心 patch 不需要新网关，核心升 minor 则需要配套网关），`llm-rosetta-gateway --version` 输出 `llm-rosetta-gateway X.Y.Z (llm-rosetta X.Y.Z)`，`GET /health` 带上 `versions: {gateway, core}`，管理配置 payload 增加 `core_version`，面板页脚只在两者不同时追加 `· core v…`。新增 [升级](gateway/upgrading.md) 页面，说明版本策略、pip / Docker 两条升级路径，以及回滚时应当成对安装的版本。

- **网关成为独立发行包**（issue [#1157](https://github.com/Oaklight/llm-rosetta/issues/1157)、PR [#1158](https://github.com/Oaklight/llm-rosetta/pull/1158)、epic [#1127](https://github.com/Oaklight/llm-rosetta/issues/1127) 的 Phase B）：`llm-rosetta` 现在只发布转换核心，网关以 `llm-rosetta-gateway` 独立发布。仓库里并存两个包——`packages/llm-rosetta/`（导入根 `llm_rosetta`）与 `packages/llm-rosetta-gateway/`（导入根 `llm_rosetta_gateway`）——网关声明对核心发行包的依赖。`llm_rosetta.gateway` 在 0.16.0 之前（[#919](https://github.com/Oaklight/llm-rosetta/issues/919)）仍可导入，作为弃用 shim：导入时会发出 `DeprecationWarning`，且每个子模块都解析为与 `llm_rosetta_gateway` 对应模块相同的对象，因此进程内不会出现两份配置或两份注册表。对使用者没有变化：`pip install "llm-rosetta[gateway]"` 依旧会装上网关（该 extra 是一个发行周期的元包别名），命令行仍是 `llm-rosetta-gateway`，路由、CLI 参数与管理面板均不变。开发时两个包一起装（`pip install -e packages/llm-rosetta -e packages/llm-rosetta-gateway`）；CI 对两者分别做 lint、测试、构建与发布，发布流程会校验两个版本号，Docker 镜像同时安装两者。vendored 模块在重新 vendor 拆分之前（epic 的 Phase A 第 3 项）仍随核心包发布。

- **转换过程中的最后一处网络调用变成注入点**（issue [#1155](https://github.com/Oaklight/llm-rosetta/issues/1155)、epic [#1127](https://github.com/Oaklight/llm-rosetta/issues/1127) 的 Phase A）：Google Generate 转换器此前用 `urllib.request` 自行下载 URL 图片，使翻译核心自己发起网络 I/O，网关也无法把自身的代理或客户端应用到这次下载上。现在 `ConversionContext` 携带可选的 `image_fetcher(url) -> (bytes, content_type) | None`，`ConversionPipeline(image_fetcher=...)` 会把它交给创建的每个上下文（请求、流式以及 fidelity 影子检查），转换器向 fetcher 要字节而不是直接使用 `urllib`。未提供 fetcher 时库的默认行为不变——仍是 `urllib`，包括 `HTTPS_PROXY`/`HTTP_PROXY` 处理——现有调用方无需改动。网关传入 `proxy.image_fetch.fetch_image`，它基于 vendored HTTP 客户端、沿用同样的 30 秒超时与 User-Agent，成为其转换路径上唯一触网的地方。测试在把 `urllib` 打桩为抛错的前提下用 fetcher 驱动整条 pipeline，并单独固定默认路径。

- **单个端点出错只降级该端点，而不是整个 provider**（issue [#1106](https://github.com/Oaklight/llm-rosetta/issues/1106)、PR [#1150](https://github.com/Oaklight/llm-rosetta/pull/1150)）：`GatewayConfig` 在一个循环里构建 embedding、rerank 与 decision 三个池，只要有一个 provider 的 `embedding_shim` 有问题——名字不存在，或该 shim 服务于另一种模型类型——异常就会从循环中抛出，导致整个配置构建失败，一个端点的问题让所有 provider 一起下线。现在每个端点单独解析：失败只跳过该端点，provider 的其他端点与其余 provider 继续服务，失败原因以 `_auto_disabled_endpoints` 记录在 provider 条目上，按模型类型索引。Admin payload 以与 `_auto_disabled` 相同的脱敏形状暴露它（仅 reason 与 category），因此诊断映射永远不会带出凭据。这是 #1105 为 chat 路径所做的隔离在 typed 路径上的另一半。

- **`gateway.app` 与 `gateway.proxy` 拆分为包**（epic [#1118](https://github.com/Oaklight/llm-rosetta/issues/1118)，issue [#1112](https://github.com/Oaklight/llm-rosetta/issues/1112)）：两个模块都承载了过多职责，难以定位。`gateway/app/` 现在按资源拆分路由面——`routes_llm`（共享处理器与五个聊天处理器）、`routes_models`、`routes_health`、`routes_typed`、`tasks`，以及 `factory`（create_app、扩展点、auth/CORS/生命周期安装、运行器）——运行中应用的 config 放在 `app/_state.py`，因为 `create_app` 写入它、处理器读取它。`gateway/proxy/` 把代理引擎拆成 `errors`、`metadata`、`fidelity`、`preflight`、`engine`（非流式）、`streaming` 与 `stream_merge`。两者都保持原有导入路径：模块导出的每个名字都由包门面继续导出。同格式 fidelity 检查现在只有一份实现（`gateway/fidelity_check.py`），两种上报方式——聊天路径写入 profile 并在 critical 差异时 dump，typed 处理器只记日志——不再有两份近似副本。

- **Admin 的 config 路由按资源拆分**（epic [#1118](https://github.com/Oaklight/llm-rosetta/issues/1118)，issue [#1113](https://github.com/Oaklight/llm-rosetta/issues/1113)）：`gateway/admin/routes/config.py` 原有 1794 行、46 个处理器，覆盖六类资源，因此每个 provider 或 model 功能都要改同一个文件。现在它是一个包——`payload`、`providers`、`models`、`model_providers`、`settings`、`shims`、`upstream`、`bulk`——由 `register_config_routes(app)` 统一注册 18 个 config 端点，`routes/__init__.py` 只需调用一次。原本只属于 `routes/__init__.py` 的 `_guard` 移到了 `_shared.py`，因为两类资源都要用它。所有端点路径与响应结构保持不变，处理器同时从包门面继续导出（`_build_model_entry`、`_append_provider_to_model`、`_shim_status_payload` 等），既有导入不受影响。

- **`gateway/config.py` 拆分為包**（epic [#1118](https://github.com/Oaklight/llm-rosetta/issues/1118)，issue [#1111](https://github.com/Oaklight/llm-rosetta/issues/1111)）：该模块原有 1886 行，每新增一个 provider 字段或模型类型都要改它。`from llm_rosetta.gateway.config import GatewayConfig` 保持不变；其背后 `config/io.py` 负责 JSONC 读取与文件锁，`config/loader.py` 负责发现与读写门面，`config/schema.py` 存放值类型，`config/parse.py` 负责 provider／model／typed 池的解析，`config/settings.py` 负责 server／auth／限流／调试段落与校验，`config/resolve.py` 负责选择、provider 选取与环路检测，`config/core.py` 保留 `GatewayConfig` 本身，其实例属性现在在类级别统一声明。行为不变：同一套测试全部通过，旧模块导出的每个名字都由包门面继续导出。

- **非 LLM（typed）模型类型统一为一份解析器、一份解析方法和一份处理器**（epic [#1118](https://github.com/Oaklight/llm-rosetta/issues/1118)，issue [#1109](https://github.com/Oaklight/llm-rosetta/issues/1109)）：embedding、rerank 与 decision 各自带有一份路由类型、provider 解析器、解析方法、探测器包装和约 280 行的处理器，而这些副本已经互相偏离。旧的 `embedding_providers` / `rerank_providers` 段落会忽略 `embedding_shim` / `rerank_shim`、描述符的按格式路径，以及统一 `providers:` 条目本就支持的 provider 级 `proxy` 键；各段落的默认路径来源也各不相同。现在只有一份 `TypedRoute`、一份由描述符的 `config_format_key` / `config_path_key` / `config_shim_key` / `default_format` / `default_path` 驱动的 `_parse_typed_config`、一份 `resolve_typed`（`resolve_embedding` / `resolve_rerank` / `resolve_decision` 名称不变）、一份探测器访问函数，以及一份处理器骨架（`gateway/pipelines/typed.py`）——三种类型只提供各自的 hook，处理器本身退化为一份 spec 加三行入口。`GatewayConfig.__init__` 改为遍历模型类型注册表来构建 `<type>_providers` / `<type>_provider_infos` / `<type>_models` 池与 `default_<type>_format`，因此新注册的类型无需改动 `config.py`。有三处对齐是有意为之：旧段落现在会遵循以 `<type>_shim` 声明的 shim、遵循描述符的按格式路径（旧的 cohere rerank provider 现在走 `/v2/rerank`，与统一条目一致），并遵循 provider 级 `proxy`。`route.<type>_path` 改为 `route.path`。新增的表征测试记录了三种类型各自的解析行为，另有一个“第四种类型”测试在不改动 `config.py`、`app.py` 与各 pipeline 模块的前提下，端到端注册并跑通一个假类型。

- **格式目录（format catalog）成为所有“按格式”事实的唯一归属**（epic [#1118](https://github.com/Oaklight/llm-rosetta/issues/1118)，issue [#1108](https://github.com/Oaklight/llm-rosetta/issues/1108)）：过去新增一种聊天格式需要改动 `src/` 下约十八个非格式本地文件。现在 `formats/catalog.py` 为每个 `(model_type, format)` 保存一条 `FormatSpec` 记录——wire family、认证构造器键、URL 模板、SSE 格式化器键、用量形状、聊天路由路径、默认上游路径、模型列表路径、选择器顺序、检测顺序——其余一切由它派生：模型词表、网关的 provider registry／基础类型提示／custom tools 默认值／选择器顺序、SSE、错误信封与用量形状映射、连通性探测的模型列表路径、`auto_detect.detect_provider`（改为按优先级遍历探测器列表，不再是一条 `if`/`elif` 链）、各 model type 描述符的格式列表与聊天路由表，以及转换器注册表的展示顺序。`tool_ops` 改为通过转换器注册表解析格式的 `ToolOps` 类，不再为每种格式写一个 `if`。新增测试只通过 catalog 注册一个假格式，并断言它出现在自动检测、配置校验、admin payload、路由注册、SSE 格式化与用量归一化中。派生还带来一处行为变化：`/v1beta/interactions` 上的错误现在使用 Google 信封（此前落到 OpenAI 信封），因为 path→信封 映射由 catalog 的路由路径生成。

- **LLM 转换器按 model type 归组**（epic [#1050](https://github.com/Oaklight/llm-rosetta/issues/1050)）：五个 chat format 包现在位于 `converters/llm/` 之下——`chat_completions`、`messages`、`open_responses`、`google_generate`、`google_interactions`——与 `embedding/`、`rerank/`、`decision/` 并列，目录结构与 `(model_type, format)` 词汇一致。公共 API 不变：`llm_rosetta.converters` 仍导出全部转换器类。`converters.google_generate` 与 `converters.google_interactions` 在 0.13.0 中已可导入，旧路径继续解析到同一模块对象；`chat_completions`、`messages`、`open_responses` 是 0.13.0 之后才出现的名字，不保留旧路径。已弃用的 `anthropic/`、`openai_chat/`、`openai_responses/`、`google_genai/` 包继续保留，并从 `converters/llm/` 重新导出。

- **Embedding、rerank 与 decision 请求即使格式相同也经过 IR**（issue [#926](https://github.com/Oaklight/llm-rosetta/issues/926)）：三个非 LLM 转换管线此前在客户端格式与上游格式相同时原样返回请求体，同格式请求因而绕过 IR。现在它们遵循 chat 管线一直采用的规则（[#577](https://github.com/Oaklight/llm-rosetta/issues/577)）：无论两端格式是否相同，每个请求和响应都按 源格式 → IR → 目标格式 转换。因此同格式的 OpenAI embedding 请求会被规范化（`input` 变成列表、未知字段被丢弃），上游响应也会从 IR 重建，而不是逐字节透传；转换器无法解析的请求体现在会以 `400`（`conversion_error`）失败，而不是被直接放行。这样生产路径与测试路径只有一条转换链，同格式流量也能获得提供方相关的规范化。同格式流量还会执行 chat 路由一直使用的 fidelity 影子对比：把原始请求体与其 IR 往返结果比较，关键字段的差异以 `[FIDELITY]` 警告记录。

- **shim 现在声明自己的能力类别与 wire format**（epic [#1050](https://github.com/Oaklight/llm-rosetta/issues/1050)）：`ProviderShim` 新增 `model_type`（`llm`、`embedding`、`rerank`、`decision`）与 `format`，loader 按 `(model_type, format)` 对校验，唯一的转换器注册表在 `converters/registry.py`；model type descriptor 的格式列表从它派生，因此新增一个 converter 后，配置、路由与 admin payload 不需要再改第二处。`base` 仍作为 `format` 的弃用别名被接受——内置 YAML 已迁移为 `model_type` + `format`——`resolve_format()` / `resolve_model_type()` 取代 `resolve_base()`，后者现在会发出 `DeprecationWarning`。两个 decision shim 改为声明真实方言（`typesafe` → `systemone`，`openai_decisions` → `openai_decisions`），而不再是类别名，因此 `resolve_base("typesafe")` 返回 `"systemone"`。gateway 的 `provider_types` 现在也记录该格式（此前记录的是 `decision` 类别），而 `resolve_model_type()` 仍返回类别。外部 shim 若只写 `base: decision` 而不写 `format`，会被拒绝，并在错误信息里列出可用格式；声明 `format` 的 shim 必须同时声明 `model_type`（只有声明 `base` 且不写 `format` 的 shim 可以省略），以免 embedding shim 默认被当作 chat shim 加载。

- **网关为每个提供方保存 `(model_type, format)` 身份，非 LLM 端点通过各自的 shim 解析**（issue [#1052](https://github.com/Oaklight/llm-rosetta/issues/1052)，epic [#1050](https://github.com/Oaklight/llm-rosetta/issues/1050) 的第 2 片）：`GatewayConfig.provider_identities` 现在为每个提供方保存一个冻结的 `ProviderIdentity(model_type, format)`，而 `provider_types` 仍返回格式，供路由、observability 与管理面板使用。没有 chat 身份的提供方可以只由端点 shim 定义——`{"embedding_shim": "acme--embedding"}`——并从该 shim 取得身份；同时声明了端点 shim 的 chat 提供方保持 chat 身份。作为 `type` 使用的裸非 chat 格式（`cohere`、`jina`、`voyage`）会解析到已知的非 LLM 类别（优先选择该提供方声明了端点的类别），而不再默认成 `llm`。`embedding_shim` / `rerank_shim` / `decision_shim` 与已有的 `*_format` / `*_path` 并列：shim 提供格式、`connection.default_path`、`base_url` 与 key env，显式的 `*_format` 或 `*_path` 仍然优先，`model_type` 与端点不匹配的 shim 会在启动时被拒绝。发往 `google_generate` 上游的 embedding 请求改用 Google 鉴权头，旧的 `embedding_providers` / `rerank_providers` 段也按同样方式规范化 `format` / `type`。`_validate` 拒绝把 LLM 模型路由到身份为 `embedding`、`rerank` 或 `decision` 的提供方。**行为变更：** 这类配置此前可以加载，现在会拒绝启动；错误信息会点名该模型与提供方。管理面板中，LLM 选择器只列出 `model_type: llm` 的 shim，并把裸基础格式与服务方 shim 显示为两个带标签的分组；每个非 LLM 端点段增加按类别过滤的 shim 选择器（同时设置格式与路径、填充空的连接字段），`saveProvider` 会保存端点 shim 键，`PUT /admin/api/config/providers/<name>` 也接受用端点 shim 提供连接默认值。

- **网关的选择逻辑成了一层，由有序的阶段组成**（issue [#906](https://github.com/Oaklight/llm-rosetta/issues/906)，PR [#1048](https://github.com/Oaklight/llm-rosetta/pull/1048)，epic [#905](https://github.com/Oaklight/llm-rosetta/issues/905)）：三套 affinity——路由策略的客户端亲和、按会话固定 API key、以及分级同格式偏好——本来是同一个概念的三份实现，既没有共享的 identity 类型，也没有一处能说清"谁在什么时候跑"。现在它们在 `gateway/routing/` 里。选择仍然发生在两个调用点，因为中间隔着 pipeline：provider 在请求体转换**之前**选定，凭据与上游 session id 在**之后**决定，因为它们的指纹取自 IR。每个请求一个 `SelectionContext` 把状态横跨两次调用串起来，四个阶段往里填：`AnchorStage`（关掉偏好时路由本来会选的 provider）、`SameFormatStage`（分级逻辑原样搬来），以及 IR 产生之后的 `ProviderKeyAffinity` 与 `SessionHeaderAffinity`。`Identity` 把客户端（key 的 hash）与会话（首条消息的指纹）拆成两半，两个请求阶段按不同精度读取它们。每个阶段声明自己读写的 context 字段，`RoutingChain` 在**构建**时就拒绝顺序错误的链，而不是等到第一个会失败的请求。行为不变：同样的指纹、同样的 tier 顺序、同样的相对锚点选择、同质 route 短路、特意不推进计数器的就绪回退，以及 401 重试之间共享的 key ring。路由计数器改为按 route id 记账，不再依赖 provider 列表对象的身份；一个 model 的 provider 子集也是独立的一条 route。**API 变更**：`GatewayConfig.resolve()` 改为接收 context（`resolve(ctx)`），由 `selection_context(model, source_format)` 构造；树外调用方若按位置传 `(source_provider, model)` 会直接报错。

- **`keys.db` 与配置文件现在也有了账本，所有修复集中在同一个界面**（epic [#1027](https://github.com/Oaklight/llm-rosetta/issues/1027)，PR [#1045](https://github.com/Oaklight/llm-rosetta/pull/1045)）：最后三处"把既有状态带到代码期望的样子"的机制此前各走各的路，而且都不留记录。key store 里两个 `PRAGMA table_info` 助手——它们添加 `last_used` 与 `usage_bucket`、不打任何日志、并靠"列在不在"判断是否已应用——现在是一个 `MANDATORY` registry，账本是 `keys.db` 自己的 `schema_version` 表；bucket 回填成为独立的一步，而不再挂在 DDL 助手的尾巴上；建表语句不再带 `usage_bucket`，于是新库与任何旧版本的库经由同一条路到达同一个 schema。config 的结构迁移与 provider/shim 名称纠正成为 `LOAD_TIME` steps：有顺序（先结构、后名称）、各自报告是否改动，账本仍是 `config_version`——因为记录写在文档自己身上。`GET /admin/api/maintenance` 现在列出进程里所有 registry（`request_log`、`api_keys`、`config`、`deferred`），每一步带 store、是否跑过、改了多少；`POST /admin/api/maintenance/<name>` 对 load-time 步骤返回 `409`：它的上下文是**上一次**加载读到的文档，从请求里重跑等于去改一份过期副本。本版本之前写的 `keys.db` 由一个具名步骤按列推断其版本——这个推断无法删除（早于账本的库恰恰就是没有记录的那种），但它现在只对完全没有账本的库运行，而今后新建的库都不再需要它。

- **工具组在目标已知处命名**//github.com/Oaklight/llm-rosetta/issues/953)）：撞名的工具现在在**目标侧**做限定，而不是在拍平容器的源转换器里，因此这一步判定的是真正会发往上游的名字。自身没有名字的容器会得到一个由内容派生的短标识（`g` + 七位 base32，32 位，同一请求稳定，prompt cache 因此不受影响），而不是拍平成空组；畸形容器按**每次请求**报告一次，而不是每个进程一次。**生成组**内逐字节相同的子工具会折叠成一个（#913）；而**客户端命名**的组仍把它们分开——这既是 #955 的决定，也是 DeepSeek 与 Volcengine 对容器内重名的硬性要求。其余行为未变：顶层与具名组的撞名、字节相同的折叠、以及 `tool_name_collisions` 都与之前一致。

- **Intrinsic 与 baseline 改为按 provider 判定，而非格式名**（issue [#912](https://github.com/Oaklight/llm-rosetta/issues/912)，epic [#905](https://github.com/Oaklight/llm-rosetta/issues/905)）：pipeline 里最后两处按字面格式名判定的逻辑已移除。`ConversionPipeline._same_format` 现在是单一命名属性，比较调用方传入的两个 provider spec；它同时决定 `baseline` 与 intrinsic 捷径。它刻意比较 **spec 而非 converter**——裸 `chat_completions` 源与 `deepseek--chat_completions` 目标共享同一个 converter 类，但必须保持为不同 spec，否则同格式直通会跳过目标 shim 的 IR 级特性（其 reasoning 配置、其 transforms）。"客户端格式能否原生声明服务端工具"从 `self._source_provider == "chat_completions"` 移到 converter 属性 `BaseConverter._SUPPORTS_NATIVE_SERVER_TOOLS`（Chat Completions 为 `False`）：判定方向不变，但该分类现在归属于它所属的格式本身——并且比它所取代的字面名更宽：以 shim 拼写的 chat-completions 源（`deepseek--chat_completions`）现在也会把以受支持 kind 命名的普通 function 提升，而 `== "chat_completions"` 的比较此前不会。同一格式、同一 converter，同样无法原生声明服务端工具；仅当目标 shim 声明了该 kind 时才会发生。

- **工具分组成为 IR 字段**（issue [#953](https://github.com/Oaklight/llm-rosetta/issues/953)）：`ToolDefinition` 新增 `tool_group` 与 `tool_group_description`，工具所属的组不再寄存在 `metadata.namespace` 里。`namespace` 一词从此专指 Responses API 家族在同一条腿上的 wire 字段——而 vendor-neutral 的 Open Responses spec 根本没有分组概念——因此它只出现在读写 provider 请求体的地方。行为不变：容器仍在源侧拍平，扁平目标下撞名的子工具仍会改名；把这一决定推迟到目标感知阶段、以及让工具**调用**也带上自身的组，都在同一个 issue 上跟进。

- **同格式偏好改为分层**（issue [#911](https://github.com/Oaklight/llm-rosetta/issues/911)，epic [#905](https://github.com/Oaklight/llm-rosetta/issues/905)）：`server.prefer_same_format` 不再只是对 base 类型做字符串匹配。它先确定一个**锚点**——即关闭该开关时路由会挑中的 provider，因此全量 route 的计数推进与关闭时完全一致——再分层次把格式对齐到客户端：与锚点同 shim、同 endpoint（仅决定故障转移顺序，因为 shim 本身已固定格式）→ 同一上游**账号**的客户端格式 → 该格式下的任意 provider → 都不满足则回到锚点走跨格式。这让该偏好真正起作用：锚点落在某个多格式厂商上的客户端会留在该厂商的账号内，而不必换到只是恰好讲这种格式的另一个账号。账号身份由凭据来源推导、无需声明——同一个 `token_command`，或同一个 key 值（字面 key 与由环境变量解析出的同值 key 视为同一个 key）；比较的只是凭据哈希，keyless provider 永不归组，且 `base_url` 被刻意排除，因此用同一个 key、从不同路径提供多种格式的厂商仍算一个账号。不新增配置字段。旧的 `(model, source_provider)` 子路由缓存已移除——选择结果现在通过锚点依赖客户端身份——并且每次请求都会推进全量 route 的计数，因此开启该开关不再改变锚点分布——tier 3 仍可能让请求由另一个账号服务，而这正是该功能的目的。

- **管理台类型选择器：顶级 / 基础类型拆分**（承接 PR [#917](https://github.com/Oaklight/llm-rosetta/pull/917)）：provider 弹窗里原本平铺罗列 `vendor--base` shim 名的 **Provider Type** `<select>` 已替换为同一行的 **顶级名称** 下拉与 **基础类型** 下拉。顶级名称下拉展示各厂商 logo，并在面板内提供 `顶级 | 树形` 切换：顶级视图只列第一级名称，树形视图像 `tree` 命令那样把每个厂商的基础类型缩进展开（仅在根行显示 logo）。厂商中立的 base format（`chat_completions`、`open_responses`、`messages`、`google_generate`、`google_interactions`）作为各自无子项的根节点出现；shim 名中不含 `--` 分隔符的厂商（如 `moonshot`）把其基础类型作为唯一子项展示。切换基础类型会改写 provider 的 `type`，视图选择持久化在 `localStorage`。provider 卡片现在会在 logo 旁以徽章形式显示解析出的基础类型。暗色模式下 logo 改为落在浅色底片上而非做反色，因为反色会让彩色厂商标识（如 alcf）偏色。

- **Provider 名称规范化** (issue [#908](https://github.com/Oaklight/llm-rosetta/issues/908), PR [#914](https://github.com/Oaklight/llm-rosetta/pull/914))：Google generateContent 格式的规范名现为 `google_generate`；`google` 与 `google-genai` 成为已弃用别名，会在所有边界（网关 JSONC 的 `type:`/`shim:`、请求探测、库层 `source_provider`/`target_provider`、shim 解析）自动迁移，经由新的共享机制（`llm_rosetta.provider_names`）。旧写法仍可用并发出 `DeprecationWarning`；迁移是加载/入口处的归一化，不是磁盘上的改写。内置 Google shim 现以规范名 `google_generate` 注册。这些已弃用写法计划在 **0.16.0** 移除（见 [#919](https://github.com/Oaklight/llm-rosetta/issues/919)）。

- **`openai_chat` 重命名为 `chat_completions`** (issue [#910](https://github.com/Oaklight/llm-rosetta/issues/910), PR [#916](https://github.com/Oaklight/llm-rosetta/pull/916), epic [#905](https://github.com/Oaklight/llm-rosetta/issues/905))：OpenAI Chat Completions 的 base 类型现为中性名 `chat_completions`；converter 模块移至 `converters/chat_completions/`，类名改为 `ChatCompletionsConverter`。旧写法 `openai_chat` / `openai-chat` 会在所有边界自动迁移（发弃用警告），`OpenAIChatConverter` / `converters.openai_chat` 的 import 仍可用。旧名计划在 **0.16.0** 移除（见 [#919](https://github.com/Oaklight/llm-rosetta/issues/919)）。

- **Responses 收敛到单一 `open_responses` base** (issue [#909](https://github.com/Oaklight/llm-rosetta/issues/909), PR [#920](https://github.com/Oaklight/llm-rosetta/pull/920), epic [#905](https://github.com/Oaklight/llm-rosetta/issues/905))：`open_responses` 现为唯一的 Responses base；OpenAI profile 成为名为 `openai--open_responses` 的 shim（`base: open_responses`），通过 shim 设置（`response_id_prefix: resp_`、`preserve_fields`、`store_default: true`）启用 OpenAI profile。converter 模块移至 `converters/open_responses/`（旧 import 路径经弃用兼容包保留）；`OpenAIResponsesConverter` 保留为行为不变的 preset。旧名 `openai_responses` 会自动迁移并发弃用警告；计划在 **0.16.0** 移除（见 [#919](https://github.com/Oaklight/llm-rosetta/issues/919)）。**行为变化：** 两名现在是同一 base，`prefer_same_format` 可能在两者间路由，且彼此不再丢弃 provider-bound passthrough 项（推翻了 #894 的一部分）。

- **`open_responses` 与 `openai_responses` 分化** (issue [#493](https://github.com/Oaklight/llm-rosetta/issues/493))：Open Responses 规范现在作为基础 converter，OpenAI Responses API 作为其派生 profile。`OpenResponsesConverter` 实现厂商中立标准（无响应 ID 前缀、无状态的 `store` 默认值、规范级回显字段）；`OpenAIResponsesConverter` 派生自它，恢复 OpenAI 的 `resp_` 前缀、`store: true` 默认值以及 OpenAI 专有的生命周期字段。`OpenAIResponses*` ops 类重命名为 `OpenResponses*`，并保留向后兼容别名。`open_responses` 拥有独立的厂商中立 shim（无默认主机、无前缀）；网关不再将其默认为 OpenAI 的 URL，未配置 `base_url` 时抛出明确错误。`allowed_tools` 的 `tool_choice` 变体现在可无损往返，而不再退化为 `auto`；`compaction` item 在同格式往返中保留（转换到不同方言时丢弃）。由于规范没有 canonical host，网关现将 `type: open_responses` 默认指向 OpenRouter（`https://openrouter.ai/api/v1`，key env 为 `OPENROUTER_API_KEY`）；依赖旧隐式 OpenAI 默认值或面向其他主机的部署需显式设置 `base_url`。
- **`converters/base/tools/` 包** (PR [#871](https://github.com/Oaklight/llm-rosetta/pull/871), issue [#858](https://github.com/Oaklight/llm-rosetta/issues/858))：`converters/base/tools.py` 重构为一个包，整合了此前散落在 `helpers/` 下的全部 8 个工具相关 helper 模块——`call_id.py`、`content.py`、`orphan_fix.py`、`call_unwind.py`、`intrinsic.py`、`batch.py`、`schema.py`、`multimodal_patch.py`——以及用于 `BaseToolOps` 的 `ops.py`。外部 import 路径（如 `from llm_rosetta.converters.base.tools import sanitize_schema`）和旧的 `helpers/` 路径均通过重新导出 shim 保留兼容。无行为变化。
- **Google Generate intrinsic 工具往返支持** (PR [#874](https://github.com/Oaklight/llm-rosetta/pull/874))：`code_execution` 调用/结果以及 `google_search`/`code_execution` 工具定义现在通过 `tool_ops.py` 往返（此前硬编码在 `message_ops.py` 中），使该 converter 在 intrinsic 工具 A→IR→A 保真度上与 Anthropic、OpenAI Responses、Google Interactions 持平。同时支持 snake_case 和 camelCase 两种 provider key 风格。
- **跨 Provider 统一不透明 reasoning 数据映射** (PR [#865](https://github.com/Oaklight/llm-rosetta/pull/865), issue [#857](https://github.com/Oaklight/llm-rosetta/issues/857))：`ReasoningPart.signature` 现专用于与可见文本共存的验证签名（Anthropic `thinking.signature`、Google `thoughtSignature`、Google Interactions `thought.signature`）；`ReasoningPart.redacted_data` 专用于不透明/加密的内容块（Anthropic `redacted_thinking.data`、OpenAI Responses/Chat 的 `encrypted_content`）。此前 OpenAI Responses 和 Chat 将 `encrypted_content` 存放在 `signature`/`provider_metadata` 中；Google Generate 此前将 `thoughtSignature` 存放在 `provider_metadata` 中。流式 `ReasoningDeltaEvent.encrypted_content` 不受影响。
- **Intrinsic IR 变换** (PR [#875](https://github.com/Oaklight/llm-rosetta/pull/875), issue [#849](https://github.com/Oaklight/llm-rosetta/issues/849))：`BaseConverter` 新增 `_INTRINSIC_IR_TRANSFORMS` 类属性，用于声明格式级 IR 变换——即使未配置 shim 也会执行。Anthropic、Google Generate、Google Interactions converter 将 `hoist_late_system_messages()` 声明为 intrinsic（Anthropic 额外包含 `auto_cache_breakpoints()`），因为这些格式不支持 inline system messages。OpenAI Chat/Responses 保持空默认，其格式原生支持 inline system messages。
- **Transforms 包** (PR [#877](https://github.com/Oaklight/llm-rosetta/pull/877), issue [#876](https://github.com/Oaklight/llm-rosetta/issues/876))：所有变换原语从 `shims/transforms.py` 提取到新的顶层 `llm_rosetta/transforms/` 包（`ir.py` 为 IR 级，`body.py` 为 body 级）。消除了 converter→shim 的跨层依赖。旧的 `shims/transforms.py` 保留为 deprecated 重新导出——请改用 `from llm_rosetta.transforms import`。
- **Intrinsic 工具类型** (PR [#840](https://github.com/Oaklight/llm-rosetta/pull/840), issue [#181](https://github.com/Oaklight/llm-rosetta/issues/181))：将独立的服务端工具类型字面量（`web_search`、`code_interpreter`、`file_search`）替换为统一的 `intrinsic` IR tool_type。具体种类存储在 `provider_metadata["intrinsic_kind"]` 中。Intrinsic 工具在同格式转换中透传，跨格式转换中剥离。五个 converter 全部更新：Anthropic `server_tool_use` + `*_tool_result` block、OpenAI Responses `web_search_call`/`code_interpreter_call`/`file_search_call`/`shell_call`/`computer_call`、Google Generate `executableCode`/`codeExecutionResult`、Google Interactions 全部 6 种 server step type。共享 helper 位于 `converters/base/tools/intrinsic.py`（由 PR #871 从 `helpers/tool_intrinsic.py` 迁移而来）。
- **Shim 变换** (PRs [#684](https://github.com/Oaklight/llm-rosetta/pull/684), [#685](https://github.com/Oaklight/llm-rosetta/pull/685), [#716](https://github.com/Oaklight/llm-rosetta/pull/716), [#799](https://github.com/Oaklight/llm-rosetta/pull/799), [#812](https://github.com/Oaklight/llm-rosetta/pull/812))：新增 `default_tool_description` 和 `response_body_transforms` 原语；Argo 和 DeepSeek Responses API shim；`ProviderShim` 字段重组为 `ConnectionConfig`/`ToolsConfig`。
- **构建系统** (PR [#824](https://github.com/Oaklight/llm-rosetta/pull/824))：从 `deploy-dev` 提取独立的 `build-wheel` Makefile 目标。Makefile 和 CI 的版本解析统一为 `sed -n`，兼容 POSIX/macOS。
- **补齐 0.13.0 名称的弃用警告**（PR [#1101](https://github.com/Oaklight/llm-rosetta/pull/1101)，issue [#1100](https://github.com/Oaklight/llm-rosetta/issues/1100)）：gateway 的 `auth` / `headers` / `embeddings` 再导出、`converters.google_generate` / `converters.google_interactions` 包路径以及 `OpenAIChatConverter` 等类别名，现在会在调用者所在行发出 `DeprecationWarning`，并保留到 **0.16.0**（[#919](https://github.com/Oaklight/llm-rosetta/issues/919)）。0.10.0 时代的 `gateway.admin.metrics` / `persistence` / `request_log` 再导出改为导入时告警，而不是在尚无警告的情况下直接删除；`minimax--openai_chat`、`openrouter--openai_chat` 与 `volcengine--openai_responses` 三种 shim 拼写保留到 0.16.0。从未进入稳定版的兼容层——`gateway.admin.ops_log`、decision/embedding/rerank 的 pipeline stub、decision shim 别名以及 `deepseek--*` / `kilo--*` 拼写——已移除。

### 修复

- **服务方卡片会显示降级的端点**（issue [#1106](https://github.com/Oaklight/llm-rosetta/issues/1106)、PR [#1162](https://github.com/Oaklight/llm-rosetta/pull/1162)）：当一个服务方同时提供多种模型类型时，其中一个端点解析失败不会影响其它端点（#1150），但面板从未显示这一点——标记写在运行时解析结果上，而 `GET /admin/api/config` 读取的是磁盘文档，因此 `_auto_disabled_endpoints` 从未进入 payload，卡片只能画整体禁用标记。现在 payload 会叠加运行时标记（只保留 reason 与 category，与 `_auto_disabled` 相同的脱敏形状），卡片为每个降级端点渲染一个标签，标明模型类型与失败类别，完整校验原因放在 tooltip 中；卡片边框只做轻微警示，因为该服务方的其它端点仍在服务。

- **失效的 shim 会给出规范拼写与应声明的 preset**（issue [#1106](https://github.com/Oaklight/llm-rosetta/issues/1106)、PR [#1162](https://github.com/Oaklight/llm-rosetta/pull/1162)）：ALCF、Argo 与 AskSage 移入独立包之后，仍然写着它们某个 shim 的配置只会得到一句 “unknown provider type”——不会提示 `alcf--sophia` 在 #1035 中已改为 `alcf_sophia--chat_completions`，也不会提示只要声明 `shims.presets: ['alcf']` 就能修好。更糟的是这些旧拼写根本不在 `LEGACY_SHIM_ALIASES` 中，查找之前连规范化都不会发生。现在 shim catalogue 记录每个包的 `legacy_names`，别名表覆盖这些名字，并由同一个 helper 从这份记录生成提示——改名的 shim 给出规范名与 preset，规范名但未声明包的只给出 preset。这条提示会出现在失效 shim 可能暴露的全部三处：缺少 `base_url` 的报错、provider 已配置 `base_url` 却退化为 Bearer 认证与通用 URL 模板时的那条 warning、以及 provider 卡片背后的 `unknown_shim` 校验类别。有测试从两个方向保证 catalogue 与别名表同步；`tests/integration/test_shim_presets_live.py` 则通过真实网络声明 preset——固定 catalogue 的仓库 URL 与 `path` 仍然可加载、缓存命中时不会重复抓取、旧的 ALCF 拼写经已声明的 preset 能解析成功、以及撤掉 preset 后 provider 会被降级且提示中会点名该 preset。

- **加载期迁移有实际改动时，保存配置不再死锁**（issue [#1106](https://github.com/Oaklight/llm-rosetta/issues/1106)、PR [#1150](https://github.com/Oaklight/llm-rosetta/pull/1150)）：admin 变更先取 `async_config_lock`，再加载配置文档；若加载期迁移修改了内容，它会写回文档，而写回会在同一线程、同一 sidecar 文件上再取一次 `config_lock`。`flock` 是按打开文件描述计数的，于是第二次获取永久阻塞——而且阻塞在事件循环线程上，在响应构造之前就把网关冻住。任何对未版本化或旧版本配置的保存都会命中：日志打出 `config: applying migration v0 -> v1`，然后再无响应。现在锁会记录持有者线程，同一线程的嵌套获取直接放行、不碰文件；其他线程仍会在 `flock` 上等待，跨线程与跨进程互斥不变。此问题是在为 provider 开关响应补运行时覆盖时发现的。

- **emerald 主题的卡片覆盖层不再吞掉点击**（PR [#1149](https://github.com/Oaklight/llm-rosetta/pull/1149)）：该主题用绝对定位的 `::before` 在每个 KPI 卡片上绘制网格，而 `pointer-events: auto` 让这层装饰在命中测试时位于卡片内容之上——于是所有点击都落在卡片本身，Token 卡片的 “详情” 链接在 Firefox 中永远打不开 token 用量弹窗（切到该主题后 Chromium 同样如此）。现在两层装饰性覆盖层都是 `pointer-events: none`。

- **推理回放是模型能力，而不是 vendor transform**（epic [#1143](https://github.com/Oaklight/llm-rosetta/issues/1143)，issue [#948](https://github.com/Oaklight/llm-rosetta/issues/948) 的后续）：只要历史不以 `user` 结尾，DeepSeek 的思考模式就会拒绝带工具、却缺少 `reasoning_content` 的 assistant 轮次。#948 把 `backfill_reasoning_content()` 挂在 `deepseek--chat_completions` 上修复了它，但只保护了一个端点。2026-10-21 生产网关把 `deepseek-v4.1-flash` 的流量切到 OpenCode Go 线路，同样的 Claude Code 历史开始返回 `400 The reasoning_content in the thinking mode must be passed back to the API`——三份错误 dump 都带 69 个工具，历史以工具结果或末尾 system 消息结束，其中 53–83 条 assistant 轮次没有推理，因为客户端在回放时丢掉了它。转换本身是忠实的：把录下来的转换后请求体回放到 `api.deepseek.com` 可原样复现 400，给缺失的轮次补上 `reasoning_content: ""` 后返回 200，修复后的 pipeline 输出同样返回 200。现在该要求由目标模型声明：`ReasoningCapability.replay_missing`（`as_is` | `fill_empty`）由与其他能力约束并列的新 IR pass `backfill_reasoning_replay()` 执行，字段由目标所用的转换器按自己的方言产出。DeepSeek shim 在 provider 级声明；OpenCode shim 用一条 `model_overrides` 规则为 DeepSeek 家族声明，同一 shim 代理的 Claude、GPT、Gemini 模型不受影响，并用一个 provider 级段落写明 Chat 基础能力，保证它们的 effort 阶梯不变。`model_overrides` 的键现在先按上游模型 ID 精确匹配，再按 shell 通配模式（`fnmatch`，其 `*` 也匹配 `/`）依次匹配，因此新的 DeepSeek 快照、或带 vendor 前缀的 ID（如 `deepseek/deepseek-v3`）无需修改 shim 即可覆盖。挂在 vendor shim 上的 transform 已移除，管理端的推理覆盖也不会再把该策略重置。`backfill_reasoning_content()` 仍对自行构造 wire body 的调用方公开，但不再有任何内置 shim 挂载它；pass 先运行，保留该 transform 的自定义 shim 会看到字段已经存在（transform 是幂等的）。

- **上游要求字符串时，Chat 的工具内容就是字符串**（issue [#1121](https://github.com/Oaklight/llm-rosetta/issues/1121)、PR [#1125](https://github.com/Oaklight/llm-rosetta/pull/1125)）：内容为文本块列表的 Messages `tool_result`，以及所有 `is_error` 结果，都会被写成 `role: "tool"` 的**列表**内容。Chat 转换器声明了 `_SUPPORTS_MULTIMODAL_TOOL_RESULT = False`，但只有图片/文件 packing 路径拿到了这个标志，纯文本与错误结果因此绕过了它——OpenAI 兼容的 gRPC 线路以 `tool message content must be a string` 拒绝请求体。写入端现在把转换后的内容压成一个字符串：文本块按换行拼接，仍残留的非文本块以 JSON 序列化保留（沿用旧的 residual 意图），`is_error` 则通过 `mark_tool_error_text` 追加为 `[Tool error]` 标记行。声明支持 block 的目标仍会收到 block 列表与标记 block，packing 路径与按目标的能力覆盖保持原样。

- **分段控件可用键盘操作**（PR [#1099](https://github.com/Oaklight/llm-rosetta/pull/1099)）：筛选组（提供方类型、模型类型、请求/Ops 日志视图）声明了 `role="radio"` 选项却没有激活键——回车无效，空格会滚动页面；用量时间窗和模型类型组包裹的是原生 radio 输入，方向键会移动 radio，但 `active` 样式类留在原地。现在回车/空格可激活聚焦选项，Home/End 与方向键移动选择并在两端循环，聚焦选项显示焦点环（选中项改用强调色前景色描边，避免与填充同色），基于 radio 的组在 `change` 事件中同步样式类。模型类型筛选在选项集合不变时原地更新，方向键不再把焦点丢回页面。

- **模型弹窗里的能力选项可用键盘操作**（issue [#1097](https://github.com/Oaklight/llm-rosetta/issues/1097)，PR [#1098](https://github.com/Oaklight/llm-rosetta/pull/1098)）：能力 chip 是 `<div>`，由处理器手动翻转隐藏的复选框；于是聚焦复选框按空格时浏览器先切换一次，点击冒泡到 chip 后处理器又翻回去——状态永远不变，也没有任何焦点提示。现在 chip 改为 label，点击或空格只切换一次，处理器只根据 `cb.checked` 同步 `active` 类；`:focus-within` 描边显示键盘焦点所在，隐藏输入框的样式加了类型限定，弹窗的 `width:100%` 规则不再把它撑开。

- **手机上的 Error Dumps 可读**（issue [#1095](https://github.com/Oaklight/llm-rosetta/issues/1095)，PR [#1096](https://github.com/Oaklight/llm-rosetta/pull/1096)）：八列表格保留 780px 最小宽度，手机上看得到勾选框、折成三行的时间戳、模型与路由的开头——状态、阶段、错误预览和行内操作都在屏幕之外。现在 768px 以下的每条 dump 都是一张卡片：首行时间与状态，然后是模型、路由、错误文本（改为换行而非省略），最后是阶段与操作，单元格保留显式角色与列名。预览的单行截断移入 class，卡片得以覆盖它；桌面端保持不变。

- **Settings 页在手机上不再横向滚动**（issue [#1093](https://github.com/Oaklight/llm-rosetta/issues/1093)，PR [#1094](https://github.com/Oaklight/llm-rosetta/pull/1094)）：在 390px 视口下页面宽度达到 619px，Settings 会横向滚动，固定的底栏也随之漂移。日志保留（Log Retention）网格带有内联的三列模板，压过了手机端的堆叠规则；其八个清理按钮也不会换行。现在三列定义移入 class 并在 768px 以下堆叠，按钮行允许换行。桌面端仍保持三列。

- **手机端 Dashboard 的 KPI 网格会填满最后一行，Total Requests 也会分组显示**（issue [#1091](https://github.com/Oaklight/llm-rosetta/issues/1091)，PR [#1092](https://github.com/Oaklight/llm-rosetta/pull/1092)）：手机端两列网格里的五张卡片会让第五张（tokens）独自占一行，网格的间隙色透出来像一张空的半卡片，看起来像是缺少了指标。现在当卡片数为奇数时，最后一张会横跨两列；偶数时仍保持每行两张。Total Requests 也改为与页脚和面板其他数字一致的千分位分组。

- **手机上的 API Keys 页可以看到 token 用量**（issue [#1089](https://github.com/Oaklight/llm-rosetta/issues/1089)，PR [#1090](https://github.com/Oaklight/llm-rosetta/pull/1090)）：七列表格保留 760px 最小宽度，手机上看得到 Label、Created 与 Last Used，而这个页面真正要展示的用量数字都在屏幕之外。现在 768px 以下的每个 key 都是卡片：首行是标签与操作，其下是四个用量数字，最后是日期；因为表头被隐藏，每个数值在卡片上自带列名。隐藏表头同时也去掉了排序入口，因此工具栏新增与 Models 相同的排序下拉框和方向按钮；选择 **Default** 会恢复服务端顺序并禁用方向按钮。桌面端保持不变。

- **手机上的 Server Ops Log 可读，详情行也可用键盘操作**（issue [#1087](https://github.com/Oaklight/llm-rosetta/issues/1087)，PR [#1088](https://github.com/Oaklight/llm-rosetta/pull/1088)）：五列表格保留 760px 最小宽度，手机上看得到 Time、Event 与 Severity，而说明发生了什么事的 Message 被推到屏幕外；详情行也只能点击展开。现在 768px 以下的每一行都是卡片（时间/事件/严重度，然后是消息，最后是来源），单元格保留显式角色与列名；详情行是 tab 停靠点，带 `aria-expanded` 与 `aria-controls`，Enter 或空格执行相同的切换，行键从 data 属性读取而不再拼进内联 JavaScript。桌面端保持不变。

- **Request Log 的错误行可以用键盘展开**（issue [#1085](https://github.com/Oaklight/llm-rosetta/issues/1085)，PR [#1086](https://github.com/Oaklight/llm-rosetta/pull/1086)）：行内错误详情此前只能通过点击展开——没有 tab 停靠点、没有按键处理，也没有 `aria-expanded`，因此在任何视口下都只能鼠标操作，辅助技术也无法得知某行是否展开。现在错误行可以聚焦；Enter 或空格执行与点击相同的切换，`aria-expanded` 与 `aria-controls` 随状态更新，并且按键处理会忽略来自行内 dump 按钮的事件，避免激活按钮时同时折叠该行。展开状态的键从“时间戳+模型名”改为日志 id：它跨刷新稳定，也不再让模型名进入内联 JavaScript。

- **手机上的 Request Log 不再需要横向滚动才能读到 Status**（issue [#1083](https://github.com/Oaklight/llm-rosetta/issues/1083)，PR [#1084](https://github.com/Oaklight/llm-rosetta/pull/1084)）：九列表格保留 760px 最小宽度并在容器内横向滚动，390px 屏幕上只显示 Time、Model 与 Source → Target——Status（读日志时最关心的列）在屏幕之外，而且没有任何提示还有更多列。现在 768px 以下的每一行都渲染为卡片：第一行是时间与状态，其下是模型与路由，再下面是 mode/key/ip 以及 tokens/duration。被隐藏的表头行仍留在 DOM 中，作为每个单元格的可访问名称（显式 `role="cell"` 与 `aria-labelledby`）；展开的错误详情仍是卡片下方的整宽区块。桌面端与 Server Ops Log 保持不变。

- **可排序表头现在可以用键盘操作**（issue [#1079](https://github.com/Oaklight/llm-rosetta/issues/1079)，PR [#1082](https://github.com/Oaklight/llm-rosetta/pull/1082)）：Models 与 API Keys 的列表头是 `<th onclick=…>`，键盘完全无法排序这两张表，屏幕阅读器也无从知道哪一列在排序、朝哪个方向——唯一的提示是一个伪元素画出的 ▲/▼。现在每个 `th.sortable` 都是 tab 停靠点，Enter 或空格执行同一个排序处理器（空格会 `preventDefault`，避免页面滚动）；共享的 `syncAriaSort()` 把 `aria-sort` 写到当前列，其余列不写该属性（即隐式的 `none`），新增的可排序表格因此直接继承这套行为，而不必各自重复推导。聚焦的表头会绘制 2px 强调色描边并向内偏移 2px，避免被表格的滚动容器裁掉。

- **管理面板现在能在手机尺寸的视口中使用**（PR [#1070](https://github.com/Oaklight/llm-rosetta/pull/1070)）：在 768px 以下，固定的状态页脚会折成两行并遮住页面内容，标签栏会裁掉 *Request Log*，provider/model 类型筛选器在右边缘被截断，而所有表格都把全部列挤进约 340px，导致模型表和请求日志表无法阅读。现在页脚保持单行、可横向滚动并在边缘淡出，并优先显示错误数与 DB/WAL 体积，而不是容量比例；五个标签在 375px 视口中都能放下；分段筛选器改为滚动而不是裁切；工具栏搜索独占一行；表格保留可读的最小宽度并在容器内横向滚动；Add Model 的名称/provider 行在手机上改为纵向堆叠；16px 的表单字号可避免 iOS 在聚焦时缩放。桌面布局保持不变。

- **网关现在服务的是代码所期望的那份文档，而不是磁盘上那份**（issue [#1046](https://github.com/Oaklight/llm-rosetta/issues/1046)，PR [#1047](https://github.com/Oaklight/llm-rosetta/pull/1047)）：带版本的配置迁移此前只有一个调用方——`JsoncConfigIO`，而它只由 admin 重载路径构造——因此 `load_config`（CLI 自己的启动路径，也是文档里读配置的方式）原样返回磁盘上那份文档。对一份 `config_version: 0` 的文件来说，旧的 `rerank_providers` / `embedding_providers` 从未进入统一的 `providers` 表——也就是鉴权、限流和 admin 的 provider 列表所读的那张表（rerank 与 embedding 的*路由*一直可用，因为 `GatewayConfig` 自己会重新解析这两个旧段落），而文件本身也停在版本 0，直到一次 admin 读取把它迁移。现在两个模块级读取器都经由读取器，所以步骤在任何读取配置的地方都会跑；CLI 的启动路径把它跑出的 registry 交给它构造的 config，这正是被服务的配置也会出现在维护面载入期集合里的原因；`load_gateway_config(path)` 是给程序化使用的新入口，因为 `GatewayConfig(load_config(path))` 会跑完步骤却丢掉它们的 registry。**行为变更**：启动现在可以写配置——原文件先被复制为 `.pre-migration.bak`；由于写入用的是 `json.dump`，一次会迁移的启动会把带注释的 JSONC 文档重写成纯 JSON，**注释不会保留**。写回是尽力而为：只读的配置目录（比如挂载的 ConfigMap）会拿到内存中迁移好的文档并记一条警告，文件保持原样，而不是变成一次拒绝启动。无法迁移到当前版本的文档会拒绝启动，错误会指名是哪一步、哪个路径；`config_version` 本身是记账而非输入，因此被引号包住的数字或 JSON 浮点会被认作它们写的版本，读不出来的值按"没有记录"处理并照常迁移，而声明了*比本版本更新*的版本的文档会原样服务并记一条警告。

- **一步启动修复失败，不再让后面四步被跳过**（epic [#1027](https://github.com/Oaklight/llm-rosetta/issues/1027)，PR [#1044](https://github.com/Oaklight/llm-rosetta/pull/1044)）：socket 打开后跑的那五步数据修复共用同一个 `try`，于是第一个失败——数据库被锁、旧版本从未创建过某列——会静默跳过它之后的一切。而运维打开管理面板最关心的两步（token 总数与用量归属）恰恰排在最后，也就是说最能掩盖问题的那个失败，掩盖得最多。现在每一步各自失败：失败会带步骤名记入日志、并记在该步骤名下，它后面的步骤照跑，成功的步骤也照旧报告改了多少。这次改动在缺陷原本所在的地方验证：那个只断言"就绪标志被清掉"的测试，现在同时断言后续步骤确实被调用了。

- **引用可能丢得毫无痕迹，而 Anthropic 的 citation 词汇表被建模成了一个线上 API 明确拒绝的 tag**（issue [#459](https://github.com/Oaklight/llm-rosetta/issues/459)，PR [#1025](https://github.com/Oaklight/llm-rosetta/pull/1025)）：chat completions 的响应写入在 docstring 里承诺"由调用方报告这次丢失"，而调用方没有 `else`——一条发往 chat completions 的 Anthropic `char_location` 既不进 sink 也不落日志，而同一条引用走 Messages 写入时会正常报告；它的历史侧在本该写 annotation 的地方只有一个 `pass`（实测：该端点**接受**请求中 assistant 消息上的 `annotations`）；它的流式读侧没有 annotation 分支，写侧则回落到 base 的丢弃，于是流式引用入向丢失、出向发不出去（实测：搜索模型会在 `choices[].delta.annotations` 里流式回传，同样嵌在 `url_citation` 下）。三处都已接通，并且丢弃断言改为驱动响应写入而非直接调用上报 helper——会静默回退的不是 helper，是调用点。同一轮 review 还定下两条约定：源 dialect 没有消息级 span 的引用写**不带** span，而不是 `0/0`（后者读起来是"引用前几个字符"而非"位置未知"）；矩阵的写侧单元格改为按原生拼法的**精确值**断言而非真值断言，因此改一个种类名、或让外来 tag 漏过去，都会红。Anthropic 的 wire 类型一并修正：它此前把整套 citation 词汇表建模成单一的 `{"type": "text_citation"}`，而该 tag 会被 API 以 `400 Input tag 'text_citation' found using 'type' does not match any of the expected tags` 拒绝，现在改为它真正接受的五种。

- **API 参考文档曾让读者去点一个空链接**（PR [#1040](https://github.com/Oaklight/llm-rosetta/pull/1040)）：`IREmbeddingResponse` 的 docstring 把一个 JSON 路径写成 `` `embeddings.{type}[][]` ``，而 Markdown 会把结尾的方括号读成一个标签为空的*折叠式引用链接*——没有任何定义能匹配它——于是 API 参考构建报告 `unresolved autoref`，构建出的页面上也留下一个没有目标的锚点。现在这两处（类与模块的 docstring）都是代码 span，它们本来也一直是：方括号是路径的一部分，不是标记。

- **`open_responses` 的 intrinsic（服务端工具）通路在两个方向、流式与非流式下全部损坏**（issue [#985](https://github.com/Oaklight/llm-rosetta/issues/985)，PR [#987](https://github.com/Oaklight/llm-rosetta/pull/987)）：intrinsic 调用或结果要么凭空消失，要么被改写成客户端调用，`web_search` / `code_interpreter` 轮次根本无法往返。流式读取上，`web_search_call` / `code_interpreter_call` 输出项完全不产生 IR 事件；现在 `_handle_p_output_item_added_to_ir` 把它们派发为 intrinsic `ToolCallStartEvent`——kind 由 item 类型经共享的 `_ITEM_TO_INTRINSIC_KIND` 表推导，绝不从工具名推断——并把 `done` 载荷（`action` / `code` / `outputs`）作为 `ToolCallDeltaEvent` 搬运，而 `*_call_output` item 变为一次性的 `ToolResultEvent`，由非流式路径同一个 reader 构造。流式写出上，intrinsic `ToolCallStartEvent` 以前会变成客户端 `function_call`、`ToolResultEvent` 会被丢弃；现在 writer 重新发出 provider 自己的服务端 item 类型，跨 delta 保留载荷，在 `output_item.done` / `response.completed` 重建 item，对结果则回放存下的原生 item 或重建 `*_call_output`——搭在调用 item 上的 web-search 结果、以及任何无法渲染的 kind 都静默丢弃，服务端调用永不降级为客户端调用。非流式写出上，`_build_server_item` 现在发出 API 要求的字段——每个服务端 item 都带 `id`（web search 还从读侧恢复 `status`），interpreter 带 `code` / `container_id` / `outputs`——且不再把客户端 `call_id` / `arguments` 对放到服务端 item 上。实现一次于 `OpenResponsesConverter`（厂商中立的 base），`OpenAIResponsesConverter` 直接继承，并用 `gpt-5-nano` 对 OpenAI Responses API 实测：每种 intrinsic kind 在两种模式下都能正确配对。

- **intrinsic 结果只按 kind 与其调用配对，导致两个 writer 发生漂移**（issue [#996](https://github.com/Oaklight/llm-rosetta/issues/996)，在 PR [#987](https://github.com/Oaklight/llm-rosetta/pull/987) 中一并修复）：结果 writer 仅凭 *kind* 决定是否丢弃 intrinsic 结果，而调用 writer 依据 `(kind, fields, leg)`，两者因此漂移：被降级为客户端 `function_call` 的调用可能完全没有输出，而被 IR 原生回放的调用又可能被塞进一个 `call_id` 没有任何客户端调用对应的 `function_call_output`。现在两个 writer 共用同一个谓词 `_intrinsic_call_degrades_on_history`，且 `ir_messages_to_p` 会预扫描消息列表得到 `degraded_intrinsic_calls`（`call_id` 已 sanitize → bool）并传给结果 writer，因为调用与结果位于不同消息：历史腿上被降级的调用配上对应的 `function_call_output`，原生回放的调用则不配，响应腿保留原生 `*_call_output`，而 `web_search` 一律丢弃。被字面量遮蔽、重复定义的 `_INTRINSIC_KIND_TO_RESULT_ITEM` 合并为派生形式，使调用与结果的 item 映射不再可能漂移。新增 8 个测试把每种 intrinsic kind 送过 message writer 并断言配对契约，两处守卫都做了变异验证（去掉任一都会让三个测试变红）。

- **`google_generate` 的 intrinsic（服务端工具）通路在两个方向、流式与非流式下全部损坏**（issue [#986](https://github.com/Oaklight/llm-rosetta/issues/986)，PR [#988](https://github.com/Oaklight/llm-rosetta/pull/988)）：非流式读取上，code execution 轮次抛出 `ValidationError`；非流式写出把服务端调用写成客户端 `functionCall` 并丢弃结果；流式读取丢掉服务端 part；流式写出则降级调用并丢失结果。现在 `executableCode` / `codeExecutionResult` 会读入一个通过校验的 IR 响应、part 按序内联——流式 reader 发出 intrinsic `ToolCallStartEvent`（kind 记在 provider metadata 中，原生 part 一并携带以供同方言还原）外加一个输入 delta，使其它格式也能收到输入，`codeExecutionResult` 发出按 kind 配对的一次性 `ToolResultEvent`——writer 重新发出 Google 自己的 part 类型，要么直接从原生 part 发出，要么在收尾时由累积的输入重建，并由 `tool_result` 处理器写出 `codeExecutionResult`。服务端调用绝不会被写成客户端 `functionCall`，Google 无法渲染的 intrinsic kind 仍静默丢弃而非降级；被截断或输入非对象的服务端调用 flush 会被丢弃而不是发出，客户端结果也不会被写进模型轮次。用 `gemini-2.5-flash` 经转换器本身对两种模式做了实测。

- **`google_interactions` 的 intrinsic（服务端工具）通路在全部四个方向都损坏，其中一处还在回放时引发真实的 `400`**（issue [#992](https://github.com/Oaklight/llm-rosetta/issues/992)，PR [#993](https://github.com/Oaklight/llm-rosetta/pull/993)）：非流式读取丢掉服务端结果、丢失调用的 `signature`；非流式写出在服务端调用上带 `name`，被 API 以 `400 Unknown parameter 'name'` 拒绝；流式读取不产生任何 intrinsic 事件；流式写出把调用降级为 `function_call` 并丢弃结果。现在 `_do_response_from_provider` 会把 `role: "tool"` 的服务端结果消息并入 assistant 轮次，使结果在响应模型中得以保留；`p_server_call_to_ir` 把 `step["signature"]` 读入 `provider_metadata.google.thought_signature` 并以规范 kind 作为调用标识，`ir_function_call_to_p` 去掉 `name` 并还原签名，`p_server_result_to_ir` 保留载荷的线上形态（`google_search_result` 为数组、`code_execution_result` 为字符串——回放会把被拍平的字符串以 `400 Expected an array, got string` 拒绝）；流式 reader 与 writer 发出 intrinsic 事件并重新发出 provider 的 step 类型，`tool_result` 处理器注册在本转换器自己的 dispatch 上。参数片段会累积为一个对象，无法渲染的 intrinsic kind 会被丢弃而非编造 step 类型。kind 始终由 step 类型折叠得出，绝不从 step 或工具名推断，服务端调用也绝不降级为客户端调用。用 `gemini-3.7-flash` 在 `/v1beta/interactions` 上实测：`google_search` 与 `code_execution` 轮次都能经 IR 往返，且库写出的轮次回放返回 `200`。

- **assistant 轮次里携带服务端工具结果会让 IR 校验失败，侥幸没失败时结果也会被丢掉**（issue [#965](https://github.com/Oaklight/llm-rosetta/issues/965)）：Anthropic 会把服务端执行的工具结果内联在 assistant 轮次里返回，而 messages reader 把所有内联 `tool_result` 一律规整成独立的 `role: "tool"` 消息——那条规则是为**请求历史**写的，客户端的结果确实必须变成 tool 消息。在响应上，这个拆分把结果变成了一个**列表**，而 `ResponseChoice.message` 只放得下一条消息，于是 `[server_tool_use, web_search_tool_result, text]` 直接抛 `Value at 'choices[0].message' does not match any variant of ...`。现在拆分的依据是**结果本身是什么**而不是角色：客户端结果照旧规整成 tool 消息，服务端结果留在产生它的那一轮里——`AssistantContentPart` 现在也接纳这种部件。响应 writer 会重建这个内联结果而不是丢弃它——修复前实测，过一次 IR 只剩 `[server_tool_use, text]`，结果块直接消失。review 中还牵出两个相关缺陷并一并修掉：共享的孤儿修复只在 `role: "tool"` 消息里数「答案」，于是内联的服务端结果让对应调用显得无人回答，系统会注入一个 `[No output available yet]` 占位——而且**每一种目标格式**都会（四个 writer 全都调用它），现在 assistant 消息里的 `tool_result` 部件同样算作答案；另一个是内联在 assistant 轮次里的**客户端**结果会因同一条路径丢掉 payload 而被占位顶替，按结果类型（而非角色）拆分后恢复原样。线上验证：抓到的 web-search 轮次逐块往返，`caller` 与结果 payload 都在。

- **流式返回的服务端工具调用被改写成客户端工具调用**（issue [#974](https://github.com/Oaklight/llm-rosetta/issues/974)）：Anthropic 把服务端执行的工具调用作为 `server_tool_use` 块返回，而流式 reader 对它和客户端的 `tool_use` 发出的是同一种 `ToolCallStartEvent`、不说明是哪一种，writer 又写死了 `"type": "tool_use"`——于是同格式的流会把服务端调用降级成客户端调用。等两侧都能用之后这就不再是外观问题：客户端回放时会变成「客户端发起的调用」却带着服务端的 `*_tool_result`，Anthropic 会把这一轮理解成由调用方自己执行了工具。writer 现在镜像 `MessagesToolOps.ir_tool_call_to_p`——也就是非流式一直有的那条分叉——`tool_type == "intrinsic"` 时以规范 kind 重发 `server_tool_use`（kind 由 `get_intrinsic_kind` 解析，绝不按工具名推断），否则重发 `tool_use`；reader 则给事件打上 `tool_type="intrinsic"`，并把 kind 放在 `provider_metadata["intrinsic_kind"]`，与非流式 `ToolCallPart` 的携带方式完全一致。补上这个标记顺带堵住了块里有、而流里丢掉的两个字段：`ToolCallStartEvent` 新增 `cache_hint`，非 `direct` 的 `caller` 也经 `provider_metadata` 带出，并且只在客户端 `tool_use` 上还原——与 `ir_tool_call_to_p` 的限制一致。以 Anthropic web search 实测：抓到的 `['server_tool_use', 'web_search_tool_result']` 原样、按序回吐为同样两个块，调用的 name 与 `input` 完好，包括携带 `{"query": ...}` 的那串 `input_json_delta`。

- **配置 reload 后 `token_command` provider 不再被 seeding，导致每次请求都以 401 被拒，直到重启**（issue [#976](https://github.com/Oaklight/llm-rosetta/issues/976)）：`_reload_gateway_config` —— 管理台每次保存、reload 端点、以及 shim 源变更都走这条路径 —— 会重建 `GatewayConfig` 并换进 app state，但 token seeding 与 refresh loop 只在 `run_gateway` 里启动过一次。于是重建后的 `token_command` provider 拿着 pending sentinel、且没有任何 refresher：路由会跳过它（`ready` 为 false），而管理台的模型列表与连通性测试处理器不经过这道闸门，直接把 `Authorization: Bearer __token_pending__` 发给上游，并把得到的 401 报成「该 provider 可能不支持列出模型」。在一台生产网关上复现：三个 ALCF provider 都用 `token_command`，新 mint 的 token 直接请求 `/resource_server/metis/models` 返回 200，走网关却是 401；reload 时日志打印了 6 行 "uses token_command"，之后却没有一行 "Seeded API key"。现在 `DeferredStartup.apply_config_reload` 会把管理器重新挂到替换后的 config 上，先取消绑定在旧 `ProviderInfo` 上的 refresh task，再在请求路径之外重新 seeding 新 provider；管理台处理器在 provider 尚未 seeding 时返回「仍在启动」，而不再代发 sentinel。
- **流式返回的服务端工具结果从来进不了 IR，于是在客户端流里凭空消失**（issue [#972](https://github.com/Oaklight/llm-rosetta/issues/972)）：Anthropic 会把服务端执行的工具结果作为 `*_tool_result` 块放在同一轮 assistant 里返回——`web_search_tool_result`、`code_execution_tool_result` 以及 `bash_` 那几个变体——而且**整块塞在单个 `content_block_start` 里**，后面没有 delta（对抗 `claude-haiku-4-5-20251001` 配 `web_search_20250305` 的真实抓包）。流式 reader 会把相邻的 `server_tool_use` **调用**块折叠掉，对结果块却什么都不发，客户端流里也就没有承载它的东西；非流式那条路早就映射了这些块，但在 intrinsic 工具声明让能力变得可达之前（#181 / `6c1b0f02cd`），根本没有任何东西会产生这样的块去走那条路。IR 的流式事件词表里也没有「工具结果」这种事件，所以这不是漏了一个分支。现在有了一个一次性的 `ToolResultEvent`——`tool_call_id`、`result`、`tool_type`、`is_error`、`cache_hint`、`block_index`、`provider_metadata`——逐字段镜像 `ToolResultPart`；Anthropic reader 复用非流式路径同一个 helper 构造它，Anthropic writer 在原始 index 上回放该块，命中 `_native` 时逐字重放，从而保住 IR 没有建模的字段（`caller`、`content` 的原始形状）。渲染不了它的 converter 不注册 handler 即丢弃，与 intrinsic 结果在别处已有的尽力而为降级一致。端到端验证：抓到的流可以逐字节往返，且结果块上的缓存断点会以 `cache_hint` 保留而不是丢失。没有 `_native` 时的重建块不再向 `*_tool_result` 写 `is_error`，因为 Anthropic 直接拒收该字段——`400 ... .is_error: Extra inputs are not permitted`；`_native` 上的方言守卫也有测试钉住，防止外来方言的负载被原样拷到 Anthropic 的 body 上。

- **三条 lane 发出了上游不接受的取值或字段名，其中一条还把端点打崩**（issue [#961](https://github.com/Oaklight/llm-rosetta/issues/961)）：`alcf--sophia` 背后是 ALCF Inference Service 上的 vLLM，它的 `reasoning_effort` 只认 `low`/`medium`/`high`，其余一律返回 **500**（不是 400），连 vLLM 文档里作为「关闭思考」写着的 `none` 也一样。而该 shim 只声明了 `effort_field`、没有 `effort_range`，于是 IR 阶梯里的 `minimal`/`xhigh`/`max` 会原样发出去，客户端要 `max` 就直接把请求打挂。这条 lane 还把推理字段命名为 `reasoning`（vLLM 已从 `reasoning_content` 改名），而 Chat Completions reader 只找后者，因此它的推理在两条腿上都被丢掉。现在该 shim 声明 `effort_range: [low, high]`（`minimal` 夹到 `low`，`xhigh`/`max` 夹到 `high`），并在 `message` 与 `delta` 上改名；线上端到端验证：要 `max` 的请求现在发出 `high` 并返回 200（此前 500），同一份响应在 Anthropic 侧会产出 thinking 块加正文。`minimax--messages` 是相反的情况：它根本没有 effort 参数——其兼容层文档把 thinking 描述为开/关/省略，而不是深度旋钮——这一点 #757 早就实测过，因此它重新声明 `effort_field: none`，取代 PR #956 加上的 `output_config.effort`；用 `none` 而不是删掉该键，是因为省略后会被 loader 默认值填回，扁平的 `reasoning_effort` 又会被发出去。`minimax--chat_completions` 则确实会校验取值（`reasoning_effort="banana"` 返回 `400 invalid params`），文档列的是 `low`…`max`，所以补上 `effort_range: [low, max]`。`openrouter--chat_completions` 存在与 ALCF 相同的改名漏洞——只处理 `message`——其流式推理同样被丢弃，现在也覆盖 `delta`。每条 lane 都有测试钉住，其中包括 pipeline 层验证：清空某个 shim 的 `pre_ir_transforms` 会让测试失败，而不是悄无声息地通过。

- **在启用 `error_dumps` 外键的数据库上，error dump 恢复写入**（issue [#964](https://github.com/Oaklight/llm-rosetta/issues/964)）：dump 会用 `error_dumps.request_log_id` 指向本次请求的日志行，但那一行是在请求结束时才写入的——晚于捕获该 dump 的错误处理。在有这个外键的库上（即 2026-10-08 及之后新建的任何库），插入会违反约束，而 `dump_error` 的兜底 except 把它吞掉，于是响应返回之前捕获的 dump 全部丢失；只有流中途的 dump 幸存，因为那时日志行早已写完。网关现在给 handler 传一个会把这些写入排队的 store，等日志行落地后再重放。早于该外键存在的库从来不做检查、一直正常，这正是丢失没有在所有部署上暴露的原因。仍然失败的 dump 写入现在以 `WARNING` 连同 traceback 记录，而不是 `INFO`，不会再无声无息。

- **把 OpenAI Responses 的服务端 item 读回 IR 时会丢载荷**（[#963](https://github.com/Oaklight/llm-rosetta/issues/963) 的后续）：`p_tool_call_to_ir` 用 `arguments` 构造 `tool_input`，但原生服务端 item 的载荷放在按类型区分的字段里——`web_search_call` 嵌在 `action` 下，`code_interpreter_call` 把 `code`/`outputs`/`container_id` 放在同级。两者都读成 `tool_input: {}`，query 与代码因此丢失；同时 `tool_name` 被写成原始 wire 类型（`web_search_call`），而不是定义侧与 `intrinsic_kind` 早已使用的规范 kind。review 又找出同一处的另外三个问题：只提取了 `search` 一种 action 形状，`open_page` 与 `find` 仍读成空、回放时退化成不带 action 的裸 `web_search_call`；写侧把 `queries` 硬写成 `[query]`，多 query 搜索的其余部分被丢；以及写侧读的是 `tool_input["query"]`，而 `file_search_call` 从不设这个键。现在载荷整体提取——action 自己的字段，判别符单独留存并在写回时重建——`tool_name` 用规范 kind，所有形状都能往返。用线上 `gpt-5-nano` 的真实响应验证，而非手写 fixture。

- **较新的 Anthropic code execution 版本会产出没有任何 shim 声明的 intrinsic kind**（[#963](https://github.com/Oaklight/llm-rosetta/issues/963) 的后续）：`code_execution_20250825` 及更新版本底层跑 bash，并以另一个运行时名上报——调用是名为 `bash_code_execution` 的 `server_tool_use`，结果块类型是 `bash_code_execution_tool_result`。读侧直接拿 wire 拼写当 kind，于是落到 `bash_code_execution`，而没有任何 shim 声明它，跨格式历史会退化而不是映射到 intrinsic kind——在 Responses 上变成以原始 kind 为名的普通 function call，在 Messages 上仍是名为原始 kind 的 `server_tool_use`。（只有*定义*侧才是无名的：目标表达不了的 intrinsic kind 会 emit `{}`。）现在这些别名在调用与结果两侧都归一到规范 `code_execution`，流式读取路径也一并归一——只改非流式会让同一响应在两条路径上结果不一致。用线上 `claude-haiku-4-5` 的真实响应验证，并确认 API 接受归一后的名字用于回放。写入侧继续 emit `code_execution_20250522` 是正确的——API 仍然接受它，而客户端声明的新版本会经现有 `_native` 载荷原样往返。

- **以 shim 拼写作为 `target_provider` 时，三个能力判定都看不到它**（issue [#912](https://github.com/Oaklight/llm-rosetta/issues/912)，epic [#905](https://github.com/Oaklight/llm-rosetta/issues/905)）：`ConversionPipeline("openai_chat", "deepseek--chat_completions")`——把 shim 名写进 target *provider*、而不传 `target_shim=`——对 `{"reasoning_effort": "none"}` 完全不产出 reasoning 字段，因为 `enforce_reasoning`、`enforce_custom_tools`、`apply_ir_transforms` 各自读的是原始的 `target_shim` 参数（这种写法下为 `None`），而不是 pipeline 已经从 target provider 解析出的 shim。同样构造改传 `target_shim=<shim 对象>` 会产出 `{"thinking": {"type": "disabled"}}`，即同一目标的两种写法结果不一致。现在三者都改为使用已解析的 shim，也就是 response-id 前缀与 shim 声明的 Responses traits 一直在用的那个。把每个 chat shim 都作为 target provider 扫一遍可见，影响比那一对构造更广：七个 shim 现在会应用目标的 reasoning 配置，而此前什么都不产出（`deepseek--chat_completions`、`deepseek--open_responses`、`minimax--chat_completions`、`moonshot`、`volcengine--chat_completions`、`volcengine--open_responses`、`zhipu`）；另有六个不再把 custom tool 降级成无名 function（`alcf--minerva`、`alcf--sophia`、`argo--chat_completions`、`argo--open_responses`、`openai--chat_completions`、`openai--open_responses`）——同一机制，经由 custom-tools 判定。用户真正会注意到的是请求体而不是能力字段：每个 chat-completions shim 都带 `hoist_late_system_messages()`，因此位置靠后的 `{"role": "system"}` 消息在"隐含 target"下会原样内联，而显式写法会把它提升为用户轮次上的 `<system>…</system>`——这改变了上游看到的内容，或者干脆让上游拒绝该请求；同理，指向 `alcf--sophia`（`custom_tools: true`）的 `custom` 工具此前会被降级成带合成 `input` schema 的 function，而不是被保留。目标 shim 自身的 body transforms 现在对"隐含 target"同样生效——例如 `deepseek--chat_completions` 会剥掉 `n`、`logit_bias`、`seed` 以及各 penalty 字段，而在把 shim 名当作 target provider 的写法下，这些字段此前会原样送到上游。影响范围限于把 *shim 名* 当作 target provider 的调用方：裸格式的 target 推断不出 shim（`resolve_shim("chat_completions")` 为 `None`），而网关总是传 shim 名，因此其请求不受影响。现在"把 shim 名当作 target provider"与"显式传 `target_shim=`"在全部 32 个 chat shim 上产出完全相同的 body。

- **网关会丢弃所有 intrinsic 工具定义**（issue [#912](https://github.com/Oaklight/llm-rosetta/issues/912)，epic [#905](https://github.com/Oaklight/llm-rosetta/issues/905)）：`gateway/proxy.py` 构造 pipeline 时传的是 `target_shim=route.shim_name`——一个 shim *名*——而 `resolve_intrinsic_tools` 直接从这个参数上读 `ToolsConfig.intrinsic_tools`，于是拿到字符串时什么也读不到，不论 provider 的 shim 声明了什么，所有 `type="intrinsic"` 工具定义都被丢弃。库层路径传的是 shim 对象，一直不受影响，因此同一个请求经 `convert()` 保留了 intrinsic 工具，经网关却被丢掉。现在该判定会像它的同类判定（`enforce_custom_tools`、`enforce_reasoning`）一样先把 shim 名解析成对象，pipeline 也改为传入它本就已经为其它特性解析好的目标 shim。

- **Chat Completions 中同时携带 `content` 与 `reasoning_content` 的 chunk 会把正文 delta 排在前面**（issue [#950](https://github.com/Oaklight/llm-rosetta/issues/950)，PR [#951](https://github.com/Oaklight/llm-rosetta/pull/951)）：部分上游会把推理的尾巴和第一个答案 token 塞进同一个 `delta`，而 `ChatCompletionsConverter._handle_p_choice_to_ir()` 先读 `content`，于是该 chunk 先生成 `TextDeltaEvent` 再生成 `ReasoningDeltaEvent`。序列化为 Anthropic 时，text 块先打开，剩下的推理只能在其**之后**另开一个 `thinking` 块，客户端于是把答案渲染成碎片（`text "基本" / thinking "。" / text "对，但我想把这一点坐实。"`）。线上 xAI `grok-4-fast-reasoning` 会间歇性发出 `{"content": "No", "reasoning_content": " is 7 times 13."}`：改动前块序为 `thinking[0] / text[1] / thinking[2] / text[3]`，现在为 `thinking[0] → text[1]`，推理尾巴并入已打开的 thinking 块。reader 现在先读 `reasoning_content` 再读 `content`，这也与非流式路径一致——在改动前的 commit 上，同时带两个字段的 message 就已经转换为 `[thinking, text]`，流式才是异类——并且与 sdcb/chats、LiteLLM、LobeChat、new-api 的做法一致，它们都把双字段 chunk 拆成推理在先。`is not None` 判断（PR [#393](https://github.com/Oaklight/llm-rosetta/pull/393)，issue [#357](https://github.com/Oaklight/llm-rosetta/issues/357)）与 Anthropic writer 丢弃空推理的行为均未改动，`tool_calls` 仍排在最后。由于改动位于 reader，所有下游 writer 都会受影响，不只是 Anthropic：`chat_completions` 交换两个 delta 的顺序，`open_responses` 交换 `output_text.delta` 与 `reasoning_summary_text.delta`，`google_generate` 把 `thought` part 提到 text part 之前，`google_interactions` 不变。修复是按 chunk 的——若流的 delta 持续在两种字段间交替，仍会产生每次推理一个块，因为**跨** chunk 重排不在范围内（没有网关这么做，且会破坏 thinking 签名）；这一遗留行为已在代码注释与 issue 的 known-limitation 一节中写明。official DeepSeek（`deepseek-reasoner`、开启 thinking 的 `deepseek-chat`）与 opencode-go 的 `deepseek-v4.1-flash`、`deepseek-v4-pro` 共探测约 14 条流（含工具循环），均不会发出双字段 chunk，因此该改动对它们无影响。

- **DeepSeek 思考模式会拒绝重新回放时不带推理内容的跨格式历史**（issue [#948](https://github.com/Oaklight/llm-rosetta/issues/948)）：DeepSeek 的思考模式要求：一旦请求涉及工具，就必须回放模型的推理内容——按其文档所说，「对于携带 `tools` 参数的请求，后续所有请求都必须完整回传 `reasoning_content`，即使某一轮模型并未发起工具调用」。若消息列表不以 `user` 结尾、且任一 assistant 消息缺该字段，就返回 `400 The reasoning_content in the thinking mode must be passed back to the API`。跨格式回放历史的 agent 客户端（Claude Code → 网关 `messages` → `chat_completions`）会丢掉部分轮次的推理内容，于是转换后的历史触发了该错误。转换本身是正确的——上游缺这个字段是因为源请求里就没有——因此 `deepseek--chat_completions` shim 现在应用新的 body transform `backfill_reasoning_content()`：当历史涉及工具、且最后一条消息不是 `user` 时，为每条缺少该字段的 assistant 消息补上 `reasoning_content: ""`，无论它是否发起过工具调用。`deepseek-flash` 实测矩阵：以工具结果结尾（带或不带 `tools` 参数）的历史、以及带 `tools` 且以 assistant 回复结尾的历史，原样发送都失败，而每条 assistant 消息都带上该字段后都通过；只填工具调用那条、或只填结尾那条，仍然失败；以 `user` 结尾则什么都不需要；且 `reasoning_content: ""` 在所有情形下都被接受，包括设置了 `reasoning_effort`、显式关闭 thinking、以及在 `deepseek-v4-pro` 上。把捕获到的失败请求体端到端回放，改动前 400，改动后 200。

- **`tool_result.is_error` 被两个 OpenAI 目标静默丢弃**（issue [#879](https://github.com/Oaklight/llm-rosetta/issues/879)）：工具结果上的 `is_error: true` 在 `messages`（原生 `is_error`）、`google_interactions`（原生 `is_error`）与 `google_generate`（把该位编码为 `response.error` 而非 `response.output`）上都能保留，但 `chat_completions` 和 `open_responses` 根本没有承载它的字段——两者原样输出 payload、`ctx.warnings` 为空，因此这个丢失端到端都不可见。该标志并非摆设：当 payload 本身没有点明失败时（工具以非零码退出却不打印任何内容），模型会把结果读成成功并就此收工，而仅凭该标志它就会重试——在 `gpt-5-nano` 上实测复现：同样一句中性的 `"(no output)"`，无标志时返回 `content='DONE'`，有标志时重新发起 `deploy` 工具调用。现在两个目标都会在结果末尾追加一个 `[Tool error]` 文本块（Chat Completions 用 `{"type": "text"}`，Responses 用 `{"type": "input_text"}`，`custom_tool_call_output` 与 `function_call_output` 均接受），并记录一条带 tool_call_id 的 warning。裸字符串 payload 会被归一化为「一个内容块 + 末尾标志块」，因此输出形态不取决于 IR 里原本是什么；只有 `true` 会触发，`false` 与缺失一律不动，空 payload 则只留下标志块。该标志刻意采用纯文本而非机器可解析的信封——回程不做还原，因为一旦文本点明了失败，标志本身已不再改变模型行为。两个转换后的 body 均已对真实 API 实测通过（`/chat/completions` 与 `/responses` 均返回 HTTP 200），内容块列表这一形态也在八个 OpenAI 兼容端点上跑过（OpenAI、DeepSeek、xAI、MiniMax、智谱、火山、OpenRouter、SiliconFlow）——全部接受，没有任何一家要求必须是裸字符串。两个 OpenAI **来源**在遇到外部 body 携带非标准 `is_error` 时同样会读取它，所以反方向也不再静默；唯一无处安放的情况——intrinsic 结果在 Responses 上会变成原生 server item——现在会给出 warning，而不是让标志无声消失。
- **`open_responses` shim 未经核实就声明了 OpenAI profile**（issue [#899](https://github.com/Oaklight/llm-rosetta/issues/899)）：#909 收敛到单一 `open_responses` base 时，把同一段 opt-in 的 OpenAI profile 块——`store_default: true` 加 `preserve_fields`（`access_programs`、`billing`、`content_filters`、`moderation`、`tool_usage`）——复制到了 `argo`、`asksage`、`deepseek`、`volcengine` 和 `opencode` 上，于是五个 provider 都声称自己具备中立 spec 并不存在的服务端存储。逐个打真实上游即可判定：OpenRouter 对 `store: true` 直接返回 `400 store: true is not supported on this proxy. Each response request is independent.`；DeepSeek 忽略该字段、恒返回 `store: false`，且 id 是不带前缀的 UUID；OpenCode 根本不回显 `store`，其 GPT 模型返回带 `resp_` 前缀的 id，而 DeepSeek 模型返回 UUID。像 OpenCode 这样的多厂商代理，用一套 shim 级 trait 无法描述。ARGO 和 AskSage 恰好是另一面镜子：两者都返回带 `resp_` 前缀的 id、并实际填充回显字段，因此都补上了此前缺失的 `response_id_prefix`；但两者都忽略 `store`（ARGO 恒报 `store: false`，AskSage 在部分模型上报 `false`、在另一些模型上干脆不含该字段），因此都去掉了 `store_default`。Volcengine 恰好相反，这也是「把所有 `store_default` 一律删掉」会得出错误结论的原因：ARK 返回带 `resp_` 前缀的 id，且是唯一一家**真做服务端存储**的非 OpenAI provider —— 请求里不传 `store` 会得到 `store: true`，传 `store: false` 也会被遵守 —— 因此它保留 `store_default: true`，只补前缀、删掉那五个从不返回的生命周期字段。现在 `deepseek--open_responses` 与 `opencode--open_responses` 不再声明任何 profile trait，`openrouter--open_responses` 写明了不声明的原因，`openai--open_responses` 保留完整 profile（已实测验证）。新增开发工具 `scripts/dev/probe/probe_responses_shims.py`，把每个 shim 声明的 trait 与真实上游逐一比对，出现矛盾时以非零退出码结束；六家全部由它判定完成。
- **内部 `_llm_rosetta_provider_metadata` 载体从库中泄漏**（issue [#941](https://github.com/Oaklight/llm-rosetta/issues/941)）：converter 会在 provider 格式对象上附带一个内部键 `_llm_rosetta_provider_metadata`，以便在进程内的 IR→provider→IR 往返中还原 provider 专有值（Google 的 `thoughtSignature`、不透明值的出处）。此前只有网关在发送前剥离它，因此直接使用库的调用方——`get_converter_for_provider(...).request_to_provider(ir)` 或 `ConversionPipeline.convert_request()`——会拿到被上游拒绝的 body：Anthropic 返回 `400 ... Extra inputs are not permitted`，OpenAI Responses 返回 `400 Unknown parameter: 'input[1]._llm_rosetta_provider_metadata'`，Google 返回 `400 ... Unknown name "_llm_rosetta_provider_metadata"`。Chat Completions 会静默忽略该键，这也是这个泄漏一直没被发现的原因。现在 `request_to_provider`、`response_to_provider` 和 `messages_to_provider` 都会剥离该载体，输出可直接发送；ops 层的 `ir_*_to_p` / `p_*_to_ir` 函数仍保留它，作为进程内往返机制。随之有一处刻意的行为变更：Google 的 refusal 在 ops 层仍能往返，但把一个**干净的线上 body** 读回来现在会得到 text part——因为 Google 从不在线上发出该载体。网关自身的剥离逻辑作为防御性 guard 保留，并改为委托给共享的 `converters.base.helpers.strip_internal_metadata` helper。载体键带命名空间（`_llm_rosetta_provider_metadata`），因此不会与调用方自带的 `tool_input` 或工具 schema 键冲突。
- **跨 provider 的不透明推理/签名值处理** (issue [#881](https://github.com/Oaklight/llm-rosetta/issues/881)，issues [#880](https://github.com/Oaklight/llm-rosetta/issues/880)/[#930](https://github.com/Oaklight/llm-rosetta/issues/930))：不透明推理值——Anthropic 的 `thinking.signature` / `redacted_thinking.data`、OpenAI 的 `encrypted_content`、Google 的 `thoughtSignature` / `function_call.signature`——只对签发的 provider 有效，但 IR 一直用与 provider 无关的字段名存储它们、且不记录出处，于是转换器会把它发给任意目标。当目标校验该字段时请求被拒：`openai_responses → anthropic` 会把异源的 `encrypted_content` 当成 `redacted_thinking.data` 发出（`400 Invalid data in redacted_thinking block`），而 `anthropic → google`（Gemini 3）重放 `functionCall` 历史时没有可用的签名（`400 ... missing a thought_signature`）。现在转换器在读取时把出处记录到保留键 `provider_metadata["_opaque_origin"]` 下：Anthropic 与 OpenAI 会**丢弃**异源的不透明值；Google 目标在签名确为 Google 来源时照常输出，否则在每轮**首个** `functionCall` 上注入文档规定的 dummy `skip_thought_signature_validator`。Google 的推理读取端还会把签名镜像到 `provider_metadata.google.thought_signature`，使 Google 的值在经**ops 层**穿过一次异源转换器（Google → Anthropic → Google）后仍能还原——顶层 `*_to_provider` API 会剥离该载体，见 [#941](https://github.com/Oaklight/llm-rosetta/issues/941)。覆盖非流式请求/响应路径**以及**流式写出路径。已用真实 API 验证：注入的 dummy 在 Gemini 3.7 与 Gemini 2.5 上均返回 HTTP 200，任意签名在 Gemini 3 上仍被拒绝，异源 `encrypted_content` 转发到 Anthropic 时不再产生 `redacted_thinking`；同 provider 往返仍保留真实值。此处有一处刻意的行为变更：经过 Anthropic 一跳后，**只有**不透明值、没有可见文本的推理块现在也会被丢弃，而此前会被原样发出 —— 那种块结构上无损但线上非法（Anthropic 会拒绝异源的不透明字段），因此这条边界上的往返有意变为有损。出处按**基础格式**粒度（`messages`、`open_responses`、`google_generate` 等）记录，而非按 shim。
- **`google_generate` 流式出站丢失 `thoughtSignature`；`open_responses` 缺少工具调用元数据载体** (issue [#930](https://github.com/Oaklight/llm-rosetta/issues/930)，并入 [#881](https://github.com/Oaklight/llm-rosetta/issues/881))：GG 的流式写出注册了工具调用却丢弃了它的 `provider_metadata`，导致流式 `functionCall` 丢失签名——把 Google Interactions 流转成 GG 目标时会撞上 Gemini 3 的缺签名 `400`。现在签名经 `StreamContext` 传递并在刷出的 `functionCall` 上写出，缺失时注入 dummy。`OpenResponsesToolOps.ir_tool_call_to_p` / `p_tool_call_to_ir` 现在也像 Messages 与 Google Generate 的 ops 一样承载 `provider_metadata`，使带 Google 签名的工具调用能穿过 `open_responses` 一跳。
- **`google_generate` 流式读/写路径丢失 `thoughtSignature`**（issue [#939](https://github.com/Oaklight/llm-rosetta/issues/939)）：Google 的思考回合会把签名放在末尾一个独立的 part 上，该 part 的 `text` 是空占位符 —— 流式 reader 正是在这里丢掉了签名（取而代之发出一个多余的空 `text_delta`），而流式 writer 则从不写 `thoughtSignature`，于是 `google_generate → google_generate` 的流式往返丢掉了非流式路径能保住的签名。现在 reader 会为这种 part 发出仅含签名的 reasoning delta，并跳过占位文本；writer 会在 thought part 上写出 `thoughtSignature`，按非流式路径同样的规则做来源门控（优先取 Google 命名空间载体，其次取本来源的扁平签名；异源签名被丢弃，但可见的推理文本仍照常发出）。这同时确定了 `ReasoningDeltaEvent` 的契约：`reasoning` 与 `signature` 是相互独立的字段、同一事件可同时携带，因此 Anthropic 与 Interactions 的 writer 现在会为合并事件发出**两个** delta，而不再丢掉可见文本。已用真实 API 验证：Gemini 3.7 的流式思考回合被完整捕获，带签名回放后返回 HTTP 200。


- **`google_output_format` 未做校验** (issue [#901](https://github.com/Oaklight/llm-rosetta/issues/901)，PR [#900](https://github.com/Oaklight/llm-rosetta/pull/900) 的后续)：`ConversionPipeline` 仅在值恰好为 `"rest"` 时选择 REST body，其余任何值都会静默落到 SDK 形态——也就是 REST 端点以 HTTP 400 `Unknown name "config"` 拒绝的那种——因此 `"REST"`、`"restful"` 或任何拼写错误都会生成一个只在上游才失败、且没有任何提示的 body。该参数现改为 `Literal["rest", "sdk"]` 类型，目标为 Google Generate 时传入其他值会在构造阶段抛出 `ValueError`——目标根本不会读取该值时仍按忽略处理，因此非 Google 路由不受影响；`GoogleGenerateConverter.request_to_provider()` 的 `output_format` 参数也做同样校验，绕过 pipeline 的直接调用同样会被拦下。
- **`google_generate` 目标输出 SDK body** (PR [#900](https://github.com/Oaklight/llm-rosetta/pull/900)，issue [#882](https://github.com/Oaklight/llm-rosetta/issues/882))：`ConversionPipeline` 只在目标 provider 字符串恰好为 `"google"` 时选择 Generative Language REST 输出格式。规范转换器名 `google_generate`（`get_converter_for_provider` 可识别）会落到 SDK body 形态（`config: {...}`），而 REST 端点会以 HTTP 400 `Unknown name "config"` 拒绝。现在两个名称都会选择 REST 输出格式（解析到 `google_generate` 的 shim 名，如 `asksage--google_generate`，会先经归一化处理）；`google_output_format="sdk"` 仍可显式退出。
- **网关丢弃 shim 连接默认值** (PR [#886](https://github.com/Oaklight/llm-rosetta/pull/886))：`GatewayConfig` 向 `build_provider_info` 传的是基础转换器类型而非 shim 名，导致 shim 名丢失、每个 shim 的 `connection.*` 默认值（`base_url`、`api_key_env`、`auth_header`、`models_path`）被静默忽略。通过 `type: <shim>--<base>` 配置且未显式写 `base_url` 的 provider 会落到基础类型的 host，shim 的 auth header 覆盖（如 AskSage 的 `x-access-tokens`）也会失效。
- **管理台创建 keyless（免费源）provider** (PR [#915](https://github.com/Oaklight/llm-rosetta/pull/915))：provider 路由要求 `api_key`，导致免费源条目被 `400` 拒绝、根本没有落盘。现在 keyless shim 允许不带 key；`keyless` 在所有 api_key 来源之后计算，因此环境里无关的 base-type 变量（例如恰好存在的 `OPENAI_API_KEY`）不会把 provider 变成带鉴权；空的 `api_key`/`base_url` 也不再遮蔽 shim 默认值。
- **Anthropic 往返保真度** (PR [#850](https://github.com/Oaklight/llm-rosetta/pull/850), issue [#845](https://github.com/Oaklight/llm-rosetta/issues/845))：对话中间的 `system` 消息此前在存在顶层 `system` 时被静默丢弃，不存在时又被提升到错误位置——现通过共享的 `hoist_system_in_converter()` 步骤修复，该步骤现在无论是否配置 shim，都会在 Anthropic、Google Generate、Google Interactions converter 中无条件执行。`redacted_thinking` block 此前因为 dispatch 表不识别该内容类型而被完全丢弃——新增 `ReasoningPart.redacted_data` 字段及双向转换。`caller: {type: direct}` 和 `is_error: false` 此前在每次往返时都会被注入到工具结果中，即使源数据中并不存在——现在仅在源数据显式包含时才会输出。
- **OpenAI Responses 多轮 reasoning item** (PR [#832](https://github.com/Oaklight/llm-rosetta/pull/832))：一个轮次中包含多个 reasoning item（思考 → 调用工具 → 再次思考）时，出站侧会将它们合并为一个——id 固定为第一个 item 的 id，而后续 item 的 `encrypted_content` 会覆盖前一个，导致 id 与内容不匹配，Argo 原生 `/v1/responses` 会返回 `400 invalid_encrypted_content`。现在每个 reasoning item 通过 `ReasoningItemState` 列表分别追踪自己的 id 和 encrypted content。
- **OpenAI Responses 自定义工具输出被还原为 `function_call_output`** (PR [#833](https://github.com/Oaklight/llm-rosetta/pull/833))：重放请求历史时 `custom_tool_call_output` 类型丢失，原因是工具类型查找依赖一个仅在 response 侧才会填充的 context map，解析请求时从未填充。现在 `ToolResultPart` 直接携带 `tool_type`，使同格式 `openai_responses → openai_responses` 的历史往返不再累积保真度警告。
- **Anthropic 并行工具调用流式 index** (PR [#835](https://github.com/Oaklight/llm-rosetta/pull/835))：将包含多个交错工具调用的 Chat Completions SSE 流转换为 Anthropic 格式时，所有工具复用了同一个内容块 index `0`，参数片段也被路由到错误的 block。现在每个工具调用拥有独立的 Anthropic block index，参数增量正确路由，流结束时关闭所有打开的 block。
- **`gateway` 同步/异步兼容性问题影响下游消费方** (PR [#836](https://github.com/Oaklight/llm-rosetta/pull/836))：持久化迁移到 `aiosqlite` 后，`create_app`、`setup_admin`、`dump_error`、`_flush_now` 变成了协程，悄悄破坏了 `argo-proxy>=3.3.3`（同步调用方，未设版本上限）——`create_app` 启动时崩溃，其余几个创建的协程从未被 await。新增 `@sync_compat` 装饰器，在同步上下文中通过 `asyncio.run()` 运行，在异步上下文中调度为 `Task`，同时保留 `async def` 签名供类型检查器识别。另新增 `gateway.headers`/`gateway.auth`/`gateway.embeddings` 向后兼容 shim 模块。
- **流式处理终止事件缺失** (PR [#870](https://github.com/Oaklight/llm-rosetta/pull/870), issue [#838](https://github.com/Oaklight/llm-rosetta/issues/838))：当 OpenAI 兼容上游未发送 `usage` 对象（未设置 `stream_options.include_usage`）时，`StreamEndEvent` 从未被发出，导致下游 converter 无法刷出 `message_delta`（含 `stop_reason`）和 `message_stop`。`@anthropic-ai/sdk` 流式 helper 报错"stream ended without producing a Message"。修复方式：当 `finish_reason` 到达但无 `usage` 时延迟 `StreamEndEvent`，在下一个 chunk 或 `None` EOF 哨兵到达时刷出。
- **Provider shim 审计** (PR [#758](https://github.com/Oaklight/llm-rosetta/pull/758))：对照最新 API 文档进行系统审计，通过真实 API 调用验证。主要修复：Anthropic Claude 5 模型 adaptive thinking、OpenAI `max` effort 支持、Volcengine `ARK_API_KEY` 环境变量、MiniMax/Moonshot/Zhipu/OpenRouter reasoning 配置修正。
- **Provider affinity 路由** (PR [#823](https://github.com/Oaklight/llm-rosetta/pull/823))：`AffinityRoundRobinStrategy` 使用了 Python 按进程随机化的 `hash()`，导致重启和多 worker 部署时 session affinity 失效。替换为 `hashlib.sha256`。
- **OpenAI Responses 流式处理** (PRs [#765](https://github.com/Oaklight/llm-rosetta/pull/765), [#766](https://github.com/Oaklight/llm-rosetta/pull/766), [#788](https://github.com/Oaklight/llm-rosetta/pull/788), [#792](https://github.com/Oaklight/llm-rosetta/pull/792), [#800](https://github.com/Oaklight/llm-rosetta/pull/800), [#810](https://github.com/Oaklight/llm-rosetta/pull/810))：字符串工具输出不再被 JSON 强制转换；reasoning 在同 turn 含文本/工具时保留；DeepSeek `reasoning_content` 提取；response 级字段和工具定义通过 IR 往返不丢失；`.done` 帧的 function_call 参数正确捕获。
- **Anthropic `budget_tokens` 与 adaptive thinking** (PR [#805](https://github.com/Oaklight/llm-rosetta/pull/805))：当 `thinking.type` 为 `adaptive` 时剥离 `budget_tokens`（Anthropic API 拒绝此组合）。
- **工具 schema 清理** (PRs [#747](https://github.com/Oaklight/llm-rosetta/pull/747), [#825](https://github.com/Oaklight/llm-rosetta/pull/825))：保留多分支 `anyOf`/`oneOf` union；缺少 `properties` 时剥离 `required`。
- **性能** (PRs [#778](https://github.com/Oaklight/llm-rosetta/pull/778), [#782](https://github.com/Oaklight/llm-rosetta/pull/782), [#783](https://github.com/Oaklight/llm-rosetta/pull/783), [#774](https://github.com/Oaklight/llm-rosetta/pull/774), [#777](https://github.com/Oaklight/llm-rosetta/pull/777))：异步 SQLite 持久化、O(1) 清理查询和元数据淘汰、fd 耗尽防护、CLOSE_WAIT socket 泄漏修复。
- **Admin UI** (PRs [#816](https://github.com/Oaklight/llm-rosetta/pull/816), [#820](https://github.com/Oaklight/llm-rosetta/pull/820), [#821](https://github.com/Oaklight/llm-rosetta/pull/821))：路由环 PR 后 Models 页面空白、favicon SVG 颜色、provider 路由列布局、深色主题文字可见性、toast z-index、settings 弹窗与 modal 冲突。
- **Shim 加载** (PRs [#667](https://github.com/Oaklight/llm-rosetta/pull/667), [#674](https://github.com/Oaklight/llm-rosetta/pull/674))：`max_tool_description_length` 和 `hoist_system_messages` 未从 provider YAML 加载；CI 守卫测试防止未来字段遗漏。

- **不完整的工具输出与未解析的流式工具 delta**（issue [#1006](https://github.com/Oaklight/llm-rosetta/issues/1006)、[#1026](https://github.com/Oaklight/llm-rosetta/issues/1026)，PR [#1067](https://github.com/Oaklight/llm-rosetta/pull/1067)）：OpenAI Responses 的 `function_call_output` 带 `status="incomplete"` 时现在映射为 IR `is_error=True`，Anthropic 侧的 `tool_result` 会报告失败，而不是把被截断的结果当作成功；状态同时写入 `provider_metadata`，因此同格式 Responses 往返会回放 `status="incomplete"`，而不是追加 `[Tool error]` 标记并把状态改写为 `completed`。未文档化的 `is_error` 字段不再覆盖原生 `status`（Responses API 会以 `400 Unknown parameter` 拒绝 `is_error`）。流式侧，Chat reader 记录按 choice 隔离的 provider index → call ID 关联，仅重复 index 的 fragment 也能保持身份（包括稀疏 index 和迟到 ID），后续冲突 ID 不会另开一个调用；Messages writer 按注册的 ID 路由，并且只写入打开中的 tool-use 块，因此 `input_json_delta` 不会再落入 text 块、已关闭的块或被复用为 text 的 index。无法解析身份的 fragment 会带警告丢弃，未发出过有身份工具调用的 choice 不再把 `tool_calls` 作为 finish reason。

### 基础设施

- **管理面板的配色方案只声明取值**（epic [#1154](https://github.com/Oaklight/llm-rosetta/issues/1154)、PR [#1159](https://github.com/Oaklight/llm-rosetta/pull/1159)）：`components.css` 里有六条规则其实是"主题在描述组件"——emerald 的页面纹理与卡片覆盖层、让 minimal 的 KPI 网格无缝相接的两条规则、以及 emerald 的主按钮字重——而其中那个覆盖层正是 #1149 里吞掉 Token 卡片点击的元凶。现在这些取值下沉为 `base.css` 中的 token（`--stats-gap`、`--stats-frame-*`、`--card-*`、`--page-texture*`、`--btn-primary-weight`），组件无条件消费它们，因此 `base.css` 之外不再出现任何 `[data-scheme]` 选择器；装饰层归组件所有，并自带 `pointer-events: none`。`tests/js/theme-contract.test.mjs` 在 CI 中守住这条边界：`base.css` 之外的 scheme 选择器、scheme 块里声明非自定义属性、以及容器上的定位装饰层接收点击，都会让测试失败。页面纹理层由 token 控制是否生成，因为一个铺满视口的 fixed 图层会把 Chromium 的文字渲染从次像素抗锯齿切到灰度抗锯齿——这是实测结果，且没有任何 CSS 取值发生变化。

- **CI 按路径过滤运行 core 与 gateway 分组**（epic [#1127](https://github.com/Oaklight/llm-rosetta/issues/1127)、PR [#1160](https://github.com/Oaklight/llm-rosetta/pull/1160)）：此前任何改动都会跑同一套内容——converter 套件、网关测试、管理面板 UI 测试、完整的 dev-extras 套件以及两个 wheel 的 smoke 测试——无论改的是核心发行包还是只有网关。现在 `changes` job 用 `dorny/paths-filter` 算出两个标记，由依赖方向决定重跑什么：改动 **core** 时两组都跑（网关依赖核心）；只改 **gateway** 时跳过核心套件（网关组仍跑 `tests/gateway` 与管理面板 UI 测试）；推送到 **master** 时始终全跑，合并提交不会被少测。smoke 测试拆成 `smoke-test (core)` 与 `smoke-test (gateway)`，按同样规则门控；修改 `ci.yml` 会刻意同时置位两个标记，CI 自身的改动永远不会只测一半。

- **公共 API 破坏性检查现在能看到"可达"的名字，而不只是"定义"的名字**（issue [#1123](https://github.com/Oaklight/llm-rosetta/issues/1123)、PR [#1148](https://github.com/Oaklight/llm-rosetta/pull/1148)）：检查器此前用"`__module__` 等于该模块"来推断没有 `__all__` 的模块的公开面，于是两种调用方实际可达的名字被漏掉。其一是导入别名：`routing.ProviderType` 因为是 `Literal` 别名而报告 `typing`，所以当包拆分补上 `__all__` 时，这个名字在 before 集合里也一并消失，检查结果仍是 0 处破坏。其二是常量：数据对象根本没有 `__module__`，`observability.ops_log` 里 32 个 ALL_CAPS 名字全在公开面之外，删掉 `ALL_EVENT_TYPES` 也不会有任何提示。现在推断的公开面改为从模块源码读取：模块内定义的类与函数、UPPER_CASE 常量，以及从另一个 `llm_rosetta` 模块导入的名字（即有意为之的 re-export）；标准库导入与模块对象仍被排除。以 `v0.13.0` 为基线，修正后的回退逻辑报出了旧检查称为 0 的 20 处可达名字移除——`pipeline` 与 `tool_ops` 曾再导出的、后来迁往 `converters.base`、`shims`、`transforms`、`capabilities` 与 `types.ir` 的契约，以及被 error-dump 上限取代的 `persistence.DEFAULT_ERROR_MAX`——它们都在 `public_api_allowlist.json` 中连同规范位置一并登记，使迁移被记录下来，而不是被静默放过。

- **导入边界守卫进入快速 CI job**（issue [#1137](https://github.com/Oaklight/llm-rosetta/issues/1137)、PR [#1138](https://github.com/Oaklight/llm-rosetta/pull/1138)、epic [#1118](https://github.com/Oaklight/llm-rosetta/issues/1118)）：用于阻止 `llm_rosetta.gateway` 进入翻译核心的 AST 与运行时检查此前只在执行全量套件的 `dev-extras-test` 中运行，而 `lint-and-test` 只跑 converters 与类型测试——因此回归必须等慢 job 才会变红。现在快速 job 会在 converter 与类型套件之前单独运行 `tests/test_core_import_boundary.py` 与 `tests/observability/test_import_boundary.py`。两个文件依赖很轻，复用 job 现有环境，该步骤约耗时一秒。此改动源于 rebase [#1107](https://github.com/Oaklight/llm-rosetta/pull/1107) 时发现的 shim 模块导入 `gateway.quota` 的问题——它此前只被全量套件抓到。

- **探测成为一个组件：一套 harness、一份带版本的报告**（issue [#1135](https://github.com/Oaklight/llm-rosetta/issues/1135)、PR [#1136](https://github.com/Oaklight/llm-rosetta/pull/1136)、epic [#1128](https://github.com/Oaklight/llm-rosetta/issues/1128)）：此前的探测脚本各自打印各自的格式，也没有任何东西消费这些输出。现在 `tools/conformance/` 负责运行一个 suite 并写出一份带 schema 版本的 JSON 报告（`schema_version: 1`），每个用例带有状态——`pass`、`fail`（跑完了但结果错误）、`error`（无法运行）或 `skip`（不可比较）——以及耗时、详情与数据。单个用例不会中断其余用例；每个用例的超时与可选的有限重试把挂起或瞬时失败限制住；进入报告的每个字符串都会经过三层脱敏——suite 实际解析出的凭据精确值（`--secret-env`）、敏感映射键下的值（`x-api-key`、`authorization`、`token` 等），以及与网关共用的凭据模式，外加 `user:pass@` 与 `key=value`。第一个 suite 包装 SDK 扫描：每个 provider 一个用例，只有当 `tools/conformance/baselines/sdk-types.json` 中不存在的字段出现时才失败，SDK 版本变化则跳过而不是报告噪声。`python -m tools.conformance run --suite drift --out report.json` 在失败时以非零码退出。harness 位于 `src/llm_rosetta` 之外，因此不会随 wheel 发布。

- **SDK 类型扫描不能再靠“什么都没扫到”通过**（issue [#1131](https://github.com/Oaklight/llm-rosetta/issues/1131)、PR [#1134](https://github.com/Oaklight/llm-rosetta/pull/1134)、epic [#1128](https://github.com/Oaklight/llm-rosetta/issues/1128)）：扫描器此前指向 `llm_rosetta.types.anthropic.*`，而 #923 已将其改名为 `types.messages`，并且吞掉所有 `ImportError`，于是该 provider 从报告中消失、进程仍以 0 退出。现在它读取规范的类型树；声明过的模块无法导入即视为错误，报告模块名并以非零码退出；同时把人工复核过的同名冲突与真实漂移分开——Chat 的 `Function` 是“调用”对 SDK 的“定义”，Responses 的 `FunctionToolParam` 是我们的包装对 SDK 的内层定义，Google 的 `FunctionResponsePart` 是我们嵌套在 function response 内的 part 对 SDK 的 part 列表；每一项都会在报告中列出，而不是被静默忽略。`--json` 输出机器可读报告——四个 provider、漂移、新类型与同名冲突——定时扫描工作流会将其作为 artifact 上传，供 conformance kit 使用。修正模块映射后暴露出真实漂移：排除复核过的同名冲突后，Messages 9、OpenAI Chat 8、OpenAI Responses 14、Google 12。

- **外部 shim 源离开核心包**（issue [#1130](https://github.com/Oaklight/llm-rosetta/issues/1130)、PR [#1133](https://github.com/Oaklight/llm-rosetta/pull/1133)、epic [#1118](https://github.com/Oaklight/llm-rosetta/issues/1118)）：`shims/sources.py`、`shims/removals.py` 与 `shims/catalog.py` 在运行时只服务于 gateway——加载器通过 vendored HTTP client 拉取远程归档，catalog 与移除清单供 Settings 面板和启动告警使用——但 `shims/__init__.py` 会急切导入这三者，因此每次 `import llm_rosetta.shims` 都会把该 client 带进翻译核心。它们现在位于 `llm_rosetta.gateway.shims`，并重新导出同样的名称（`rebuild_shims`、`resolve_remote`、`SHIM_SOURCE_PACKAGES`、`SHIM_REMOVALS` 等）。这些名称从 `llm_rosetta.shims` 与顶层 `llm_rosetta` 命名空间中移除：三个模块都是在 v0.13.0 之后才加入的，因此 0.16.0 的别名政策（[#919](https://github.com/Oaklight/llm-rosetta/issues/919)）不覆盖它们，而 core 侧的重新导出会违反 [#1132](https://github.com/Oaklight/llm-rosetta/pull/1132) 加入的边界守卫。`make check-shim-removals` 会跟随清单到新位置。Gateway 行为不变。

- **observability 包改为惰性加载**（issue [#1129](https://github.com/Oaklight/llm-rosetta/issues/1129)、PR [#1132](https://github.com/Oaklight/llm-rosetta/pull/1132)、epic [#1118](https://github.com/Oaklight/llm-rosetta/issues/1118)）：包根此前会急切导入 metrics 收集器、SQLite 存储、request log、ops log、capture、retention、maintenance 与错误转储，因此 `pipeline.chat` 对 `observability.usage.semantics` 的唯一一次使用会初始化整个 gateway 侧遥测栈——共十六个模块，包括 `persistence`、`request_log`、`metrics` 与 `error_dump`。`observability/__init__` 现在是 PEP 562 facade：每个公共名称在首次访问时才从定义它的模块解析，`from llm_rosetta.observability import X` 仍然可用，`__all__` 不变。`observability.usage` 仍留在 [#1016](https://github.com/Oaklight/llm-rosetta/issues/1016) 放置的位置，并且仍是 core 唯一导入的部分。新增的 `tests/test_core_import_boundary.py` 用两道检查守住边界：扫描所有非 gateway 模块的 AST（任意深度的绝对、相对与动态导入），以及子进程检查——导入 `converters`、`ir`、`pipeline` 与 `shims` 后，`sys.modules` 中不得出现 gateway 或 gateway 侧遥测模块。行为不变。

- **IR 逻辑有了唯一的家**（issue [#1115](https://github.com/Oaklight/llm-rosetta/issues/1115)、PR [#1126](https://github.com/Oaklight/llm-rosetta/pull/1126)、epic [#1118](https://github.com/Oaklight/llm-rosetta/issues/1118)）：body/IR 变换及其无状态辅助函数、无状态的工具辅助函数，以及能力 pass，从 `transforms/`、`converters/base/helpers/`、`converters/base/tools/` 与 `capabilities/` 移入 `ir/primitives/`、`ir/tools/` 与 `ir/passes/`。依赖方向现在被明确下来——`types.ir <- ir.primitives` / `ir.tools <- ir.passes <- converters`——并由 AST 测试（`tests/test_ir_import_boundary.py`）守住：primitives 或 tools 只要在任意深度伸手进 `converters` 就失败，绝对导入、相对导入与动态导入一视同仁。所有旧路径仍然可达：`llm_rosetta.transforms`、`converters/base/helpers/*` 与 `converters/base/tools/*` 从 `ir/` 重新导出，`capabilities/` 是一个 facade。0.13.0 中已可导入的三个辅助模块——`image_limit`、`cache_breakpoints` 与 `system_message_hoist`——会发出 `DeprecationWarning` 直到 0.16.0（[#919](https://github.com/Oaklight/llm-rosetta/issues/919)）；0.13.0 之后才出现的路径则静默重新导出。`tests/test_ir_compat_shims.py` 固定了这两种行为。行为不变。

- **Gateway 的维护步骤集中到一个包**（issue [#1114](https://github.com/Oaklight/llm-rosetta/issues/1114)、PR [#1119](https://github.com/Oaklight/llm-rosetta/pull/1119)、epic [#1118](https://github.com/Oaklight/llm-rosetta/issues/1118)）：config、keystore、name、migration、backfill、dump 与 deferred-startup 这些步骤从 `gateway/` 顶层移入 `gateway/maintenance/`。gateway 包与 maintenance 包的 facade 都惰性解析公共名称，因此 `import gateway.keystore` 不会再连带加载 HTTP app（以及它引入的 `startup` 与 `dumps`）；`config`、`names` 与 `migrations` 也不会被加载。行为不变。`llm_rosetta.gateway.migrations` 在 0.13.0 中可导入，旧路径会继续以 `DeprecationWarning` 工作到 0.16.0（[#919](https://github.com/Oaklight/llm-rosetta/issues/919)）；其余六个路径从未进入正式版本，因此不保留别名。

- **最后四个顶层模块变成了包**（issue [#1116](https://github.com/Oaklight/llm-rosetta/issues/1116)、PR [#1122](https://github.com/Oaklight/llm-rosetta/pull/1122)、epic [#1118](https://github.com/Oaklight/llm-rosetta/issues/1118)）：`pipeline`、`profiling`、`routing` 与 `fidelity` 现在都是包，导入路径保持不变。`pipeline/` 把聊天引擎（`chat.py`）与带类型的流水线（`typed.py`）分开，后者重新导出 embedding、rerank 与 decision 流水线，使四种范式共用同一个导入点；`fidelity/` 把检查器（`checker.py`）与持久化基线（`baseline.py`）分开。`llm_rosetta.fidelity_baseline` 移到 `fidelity/baseline.py` 且不保留别名，因为它在 0.13.0 中无法导入。旧模块暴露过的名字都仍然可达：`fidelity.DEFAULT_SEVERITY_BY_KIND` 现在声明在 `__all__` 中；旧扁平模块在模块作用域绑定的名字——`routing.ProviderType` / `SoftErrorPattern`、`fidelity.canonical_provider_name`、`profiling.Profiler` / `TracingProfiler`——也仍可从包根导入；`tests/test_top_level_packages.py` 守住这一表面。`fidelity` 与 `routing` 加入了公共 API 兼容性检查。行为不变。

- **凭据清洗移入 `observability`**（issue [#1117](https://github.com/Oaklight/llm-rosetta/issues/1117)、PR [#1120](https://github.com/Oaklight/llm-rosetta/pull/1120)、epic [#1118](https://github.com/Oaklight/llm-rosetta/issues/1118)）：`sanitize_headers`、`scrub_credential_patterns` 与 `sanitize_upstream_error` 从 `gateway/middleware/sanitize.py` 移到 `llm_rosetta.observability.sanitize`，因此 `observability.error_dump` 不再（哪怕惰性地）导入 gateway 来清洗转储。旧路径会静默地重新导出同一批函数：它在 0.13.0 中不可导入，因此不在已公布的移除计划（[#919](https://github.com/Oaklight/llm-rosetta/issues/919)）内。新增的基于 AST 的边界测试会扫描 `observability` 的每个模块——绝对导入、相对导入与动态导入一视同仁——只要有模块伸手进 `gateway` 就失败；只覆盖公共入口的运行时测试会漏掉函数内的导入，这正是该扫描存在的理由。行为不变。

- **聊天路由与带类型的端点共用同一个请求生命周期**（issue [#1110](https://github.com/Oaklight/llm-rosetta/issues/1110)、PR [#1124](https://github.com/Oaklight/llm-rosetta/pull/1124)、epic [#1118](https://github.com/Oaklight/llm-rosetta/issues/1118)）：embedding、rerank 与 decision 此前跑在一个复制了聊天处理器遥测、性能分析与错误转储代码的包装器里，而这份拷贝已经漂移：没有熔断器，错误转储不会等到对应的 request-log 行出现，日志里记成 `source == target ==` 模型类型，而且每个带类型请求会因为处理器自己又记了一次用量而写出**两行** ledger。现在 `gateway/lifecycle.py` 为所有路由统一持有请求 id、计时、request-log/metrics 记录、深度性能分析、延迟错误转储与熔断器。带类型处理器在解析出 provider 后通过 `begin_typed()` 询问熔断器，并通过 `finish_typed()` 交出用量；两者在处理器被直接调用时都是空操作，因此处理器在网关之外仍然可用。聊天路由表现在来自 `llm` descriptor 的 `RouteSpec` 列表：`create_app` 不再重复声明那五条路径，路由指向未知 source format 时会在启动阶段失败。**行为变化：** 带类型路由获得熔断器与能关联到日志行的错误转储；一个带类型请求只写一行 ledger，而不是两行；校验失败（坏 JSON、缺少或未知模型）因为由生命周期承接，现在也会写 request-log 行；记录的耗时包含解析与 provider 解析。

- **测试文件之间不再把 shim 注册表留空**（PR [#1040](https://github.com/Oaklight/llm-rosetta/pull/1040)）：`tests/test_shim_loader.py` 会在每个测试前后清空注册表——它的测试会注册合成 shim 并对整张注册表做断言——却没有任何东西把内置 shim 放回去，于是进程里一个都不剩。其他每个文件都通过 `get_shim` 解析 shim，而排在它之后收集的文件对每个名字都拿到 `None`。这个问题一直不可见，因为它取决于收集顺序：读取注册表的那些文件按字母序都在前面，排在它之后的文件又会在各自的 fixture 里重新加载注册表。`pytest tests/test_shim_loader.py tests/test_provider_name_alias.py` 147 个中失败了 21 个；现在一个模块级 teardown 每个文件重建一次，该文件离开进程时与它进入时保持原样。

- **Test Release 工作流** (PR [#686](https://github.com/Oaklight/llm-rosetta/pull/686))：发布 dev 版本到 Test PyPI，自动构建 Docker dev 镜像。
- **Vendored zerodep 更新** (PRs [#658](https://github.com/Oaklight/llm-rosetta/pull/658), [#710](https://github.com/Oaklight/llm-rosetta/pull/710), [#798](https://github.com/Oaklight/llm-rosetta/pull/798), [#780](https://github.com/Oaklight/llm-rosetta/pull/780))。
- **公共 API 兼容性检查**（PR [#1102](https://github.com/Oaklight/llm-rosetta/pull/1102)）：CI 会把最新稳定版 `vX.Y.Z` 标签的 `__all__` 名称与可调用签名，和每个 PR、每次推送到 master 的工作树做对比，因此被删除的名称或参数会在评审阶段而不是发布之后被发现。破坏性发现会使任务失败，除非旧拼写保留为带弃用警告的别名；有意的破坏需要连同原因记录在 `scripts/dev/public_api_allowlist.json`。新增名称与可选参数只作信息报告。本地可用 `make check-public-api` 运行同一检查。

## v0.13.0 — 2026-09-08

### 新增

- **Google Interactions API 转换器** (Issue [#581](https://github.com/Oaklight/llm-rosetta/issues/581))：新增 `converters/google_interactions/` 模块，完整支持 Google Interactions API (`/v1beta/interactions`) 的双向格式转换。支持类型化步骤（user_input、model_output、thought、function_call、function_result）、`thinking_level` ↔ IR effort 映射、Interactions SSE 流式生命周期事件（`interaction.created/completed`、`step.start/delta/stop`）以及 MCP server 工具定义。连续的 assistant 角色步骤自动合并为单个 `AssistantMessage`。无需修改 IR 类型系统。
- **重命名 `google_genai` → `google_generate`** (PR [#641](https://github.com/Oaklight/llm-rosetta/pull/641))：generateContent 转换器的模块、类和 shim base 重命名，避免与同时覆盖两套 API 的 `google-genai` SDK 名称混淆。`GoogleGenAIConverter` 和 `from llm_rosetta.converters.google_genai` 保留为弃用别名。
- **Google Interactions gateway 路由** (PR [#647](https://github.com/Oaklight/llm-rosetta/pull/647))：注册 `POST /v1beta/interactions` 端点。`GoogleInteractionsConverter` 从顶层包导出。
- **"添加新转换器"检查清单** 写入 CLAUDE.md (PR [#647](https://github.com/Oaklight/llm-rosetta/pull/647))：11 项门控清单，覆盖模块、测试、导出、自动检测、shim、gateway 路由、文档、CI smoke test 和测试注册表更新。
- **重定位超长工具描述** (PR [#625](https://github.com/Oaklight/llm-rosetta/pull/625))：当自定义工具描述超过可配置阈值时，完整文本会被移至 late system message 以防止上游 400 错误。阈值可在 model、provider 和 shim 三个层级配置。
- **可配置 Nuitka 构建参数与优化** (PR [#639](https://github.com/Oaklight/llm-rosetta/pull/639), [#640](https://github.com/Oaklight/llm-rosetta/pull/640))：新增 `NUITKA_EXTRA_FLAGS` 变量用于二进制体积实验；将最优参数组合（LTO、去除 docstrings、nofollow 排除）设为默认值；从二进制构建中移除 pyinstrument。
- **跨格式往返测试** (PR [#649](https://github.com/Oaklight/llm-rosetta/pull/649))：20 个测试验证 Interactions ↔ OpenAI Chat / Anthropic / google_generate 的请求和响应保真度。
- **扩展 `tool_ops` 便利 API** (PR [#653](https://github.com/Oaklight/llm-rosetta/pull/653))：添加 `google_interactions` 提供方支持（此前是唯一缺失的转换器），并暴露完整的 `BaseToolOps` 生命周期——`choice_to_provider`/`choice_from_provider`、`call_to_provider`/`call_from_provider`、`result_to_provider`/`result_from_provider`、`config_to_provider`/`config_from_provider`。70 个测试覆盖全部 5 个提供方。
- **监控面板多选批量下载** (PR [#655](https://github.com/Oaklight/llm-rosetta/pull/655))：性能分析、内容捕获和错误记录表格新增复选框选择和批量操作栏。错误记录支持批量下载和批量删除，选择状态跨分页保持。
- **请求日志 ↔ 错误记录互相跳转** (PR [#655](https://github.com/Oaklight/llm-rosetta/pull/655))：4xx/5xx 请求日志条目显示跳转到错误记录的按钮；错误记录行显示跳转到请求日志的按钮。跳转时清除筛选条件、导航到正确页码并高亮目标行。"匹配日志"按钮和启动时自动回填，通过时间戳近似匹配（±0.1s）和模型别名解析链接未关联的记录。
- **API 密钥最后使用时间** (PR [#655](https://github.com/Oaklight/llm-rosetta/pull/655))：API 密钥表格新增 `last_used` 列，启动时自动迁移。认证 hook 中更新，5 分钟节流。启动时从请求日志回填，支持手动"刷新最后使用"按钮。
- **能力 badge SVG 图标** (PR [#655](https://github.com/Oaklight/llm-rosetta/pull/655))：模型表格能力 badge 改用 `cap-badge` 样式并配有内联 SVG 图标（text、vision、tools、reasoning）。模型表格列宽重新平衡（模型 19%、类型 15% 居中、能力 23%）。

### 修复

- **Google Interactions provider URL 注册表** (PR [#648](https://github.com/Oaklight/llm-rosetta/pull/648))：`google_generate` 和 `google_interactions` 的 URL 模板缺失，导致上游请求 404。
- **Google Interactions 工具调用状态** (PR [#649](https://github.com/Oaklight/llm-rosetta/pull/649))：Google generateContent 对工具调用返回 `STOP`——转换器现在检测 `function_call` 步骤并将状态覆盖为 `requires_action`。
- **Google Interactions 流式传输** (PR [#649](https://github.com/Oaklight/llm-rosetta/pull/649))：在 `SSE_FORMATTERS` 注册表中添加 `google_generate` 和 `google_interactions`；添加 IR→provider 流式处理器；当上游省略 `ContentBlockStartEvent` 时合成 `step.start`/`step.stop` 事件；将 `interaction.completed` 延迟到 `stream_end` 以包含 usage 数据。
- **Google Interactions 思考透传** (PR [#649](https://github.com/Oaklight/llm-rosetta/pull/649))：启用 `thinking_level` 时在 IR 中设置 `include_thoughts=True`，确保上游 Google API 返回思考内容。


- **安全：已删除/轮换的 API key 仍可通过鉴权** (PR [#652](https://github.com/Oaklight/llm-rosetta/pull/652))：auth hook 中有一个 `config_fallback` 字典（启动时从 `server.api_keys` 构建），在 admin 面板删除/轮换 key 时从未被清除。已撤销的 key 可以绕过 keystore 验证，通过此过期的 fallback 继续鉴权。现已完全移除 `config_fallback` 路径——SQLite keystore 是唯一的 API key 鉴权源。
- **安全：credential_visible 默认值和持久化** (PR [#654](https://github.com/Oaklight/llm-rosetta/pull/654))：默认值从 `true` 翻转为 `false`，未设置 admin 密码时强制关闭。PUT 端点和 GET 凭据查看端点增加 403 守卫。Admin UI 在无密码时禁用切换按钮。配置 API 响应中敏感字段（admin_password、api_keys）已做脱敏处理。
- **服务器时间标签** (PR [#655](https://github.com/Oaklight/llm-rosetta/pull/655))："系统时间"改为"服务器时间"以更准确反映含义。时区显示从缩写格式（CDT）改为 GMT±N 格式。

### 变更

- **API key 管理现仅支持 SQLite** (PR [#652](https://github.com/Oaklight/llm-rosetta/pull/652))：配置文件中的 `server.api_keys` 和 `server.api_key` 字段不再被解析或用于鉴权。所有 API key 管理现在完全通过 admin 面板 + SQLite keystore 进行。配置文件中已有的 key 会被静默忽略。`keystore.import_from_config()` 迁移路径已被移除。

### 内部

- **Vendor zerodep `jsonx` 模块** (PR [#652](https://github.com/Oaklight/llm-rosetta/pull/652))：用 vendored 的 `jsonx` 解析器替换 `gateway/config.py` 中手写的 `_strip_jsonc_comments` 正则。新增支持 `#` 注释、尾逗号，以及更好的错误行号定位。

## v0.12.0 — 2026-09-03

### 新增

- **Namespace 工具展开** (PR [#626](https://github.com/Oaklight/llm-rosetta/pull/626))：将 `type: "namespace"` 工具容器（Codex 使用）展开为独立的 IR 工具，附带命名空间元数据和跨命名空间去重。限定名称保持在 64 字符限制内。
- **`additional_tools` 输入项支持** (PR [#623](https://github.com/Oaklight/llm-rosetta/pull/623))：从 Codex Responses API 的 `additional_tools` 输入项中提取工具定义。Namespace 工具通过 #626 管线自动展开。
- **Custom tool 三态控制** (PR [#624](https://github.com/Oaklight/llm-rosetta/pull/624))：`supports_custom_tools` 现为 `None | bool` — `None` 遵从 shim 默认值，显式 `False` 即使 shim 声明支持也强制降级。Gateway 配置 `supports_custom_tools: false` 现在正确生效。
- **可配置 `data_dir`** (PR [#619](https://github.com/Oaklight/llm-rosetta/pull/619))：gateway 持久化存储位置可通过 `--data-dir` CLI 参数或配置文件中的 `server.data_dir` 指定。Keys DB 和请求日志 DB 默认存放于此目录。
- **Admin Logo 选择器** (PR [#630](https://github.com/Oaklight/llm-rosetta/pull/630))：紧凑图标按钮配合可搜索下拉菜单。优先显示常用 provider 图标，完整列表按需从 jsdelivr (`@lobehub/icons-static-svg`) 获取。支持自定义 URL 回退和键盘导航。
- **Admin 布局编辑器** (PR [#629](https://github.com/Oaklight/llm-rosetta/pull/629))：`design/ui/layout-editor.html` 下的拖拽开发工具，用于原型设计 provider 弹窗字段布局，支持预览模式和 HTML 导出。
- **Provider `models_path` 和 `logo` 配置字段** (PR [#617](https://github.com/Oaklight/llm-rosetta/pull/617))：在 gateway 配置和 admin UI 中支持 per-provider 模型列表端点路径覆盖和 logo URL。
- **推理能力强制检查**：当上游模型未声明推理能力时，阻止 `reasoning` tool choice。
- **Admin 图表重设计** (PR [#600](https://github.com/Oaklight/llm-rosetta/pull/600))：用阶梯图（吞吐量）+ 散点图（延迟）替换折线图，提供更准确的可视化。
- **Admin 无障碍改进** (PRs [#603](https://github.com/Oaklight/llm-rosetta/pull/603)、[#606](https://github.com/Oaklight/llm-rosetta/pull/606))：模型开关的键盘导航、焦点陷阱、ARIA 属性。

### 修复

- **Anthropic 流式工具调用绑定** (PR [#628](https://github.com/Oaklight/llm-rosetta/pull/628))：使用 `chunk["index"]` 解析 `input_json_delta` 的 tool call ID，而非最后注册的 key。修复交错并行工具调用产生错误 tool ID 的问题。
- **Admin hint 弹窗裁切** (PR [#631](https://github.com/Oaklight/llm-rosetta/pull/631))：从 CSS absolute 定位切换到 JS 驱动的 fixed 定位，智能检测上/下方向。弹窗不再被 modal 的 `overflow-y: auto` 裁切。
- **`tool_call_id` 清理**：在 provider 输出边界清理包含下游 provider 不接受字符的 ID。
- **`model_list_transform` 防御处理**：当 model 配置中缺少 `internal_id` 时的防御性处理。
- **Admin 错误转储按钮**点击无响应的问题。
- **Admin provider 列表视图**重新设计为紧凑单行布局。
- **日志导入**移至模块级别以避免重复导入。

### 变更

- **Admin provider 弹窗布局**重排：Logo + Provider Name 一行，API Key 全宽，Base URL + Models Listing Path 一行（69/31），Proxy URL + Timeout 一行（69/31）。移除 Models Path 和 Timeout 字段的提示文本。
- **Shim `ReasoningCapability` 重设计**：新增 `thinking_modes`、`effort_range` 和 `visibility_modes` 字段 (PR [#614](https://github.com/Oaklight/llm-rosetta/pull/614))。
- **Argo shim** 在模型显示名中使用 `argo:` 前缀，并在获取对话框中显示上游 provider。
- **Admin 响应式布局**统一 provider 卡片标记，新增响应式列表阶段 (PR [#612](https://github.com/Oaklight/llm-rosetta/pull/612))。

## v0.11.2 — 2026-08-30

### 新增

- **原生 `tool_search` 透传** (PR [#593](https://github.com/Oaklight/llm-rosetta/pull/593))：vendor `sparse_search`，在 shim schema 中新增 `tool_search_mode`，支持 Responses API 流式和非流式路径下的原生 `tool_search` 透传。
- **Provider 连接测试** (PR [#588](https://github.com/Oaklight/llm-rosetta/pull/588))：新增 `POST /admin/api/config/providers/<name>/test-connectivity` 端点，探测 provider 的 base URL 和各配置端点（models、embedding、rerank）的可达性。显示原始和规范化后的 URL 以帮助诊断双重前缀问题。Admin UI 的 provider 卡片新增"测试"按钮。
- **`.well-known/change-password` 重定向** (PR [#579](https://github.com/Oaklight/llm-rosetta/pull/579))：`GET /.well-known/change-password` 返回 302 重定向到 `/admin#change-password`，自动打开设置面板。启用浏览器和密码管理器集成（遵循 [web 标准](https://web.dev/articles/change-password-url)）。

### 修复

- **重复 chat finish 事件** (PR [#592](https://github.com/Oaklight/llm-rosetta/pull/592))：当上游在同一 choice index 上重复发送 `finish_reason` 时（如 OpenAI 开启 `stream_options.include_usage`）进行去重。在 `StreamContext` 中新增 per-choice finish 追踪。修复 [#589](https://github.com/Oaklight/llm-rosetta/issues/589)。
- **Embedding 路由 `upstream_model` 映射** (PR [#586](https://github.com/Oaklight/llm-rosetta/pull/586))：embedding 专用路由现在从 `model_upstream_names` 应用 `upstream_model` 名称映射，与 chat 回退路由行为一致。此前模型别名（如 `argo:text-embedding-3-small` → `v3small`）被忽略，导致上游 404。
- **Embedding/Rerank URL 双重版本前缀** (PR [#588](https://github.com/Oaklight/llm-rosetta/pull/588))：`ProviderInfo` 现在自动检测 `base_url` 尾部的版本段（如 `/v1`）是否会与 `url_template` 路径开头重复，并自动去除。修复了 `base_url: "https://api.openai.com/v1"` + `embedding_path: "/v1/embeddings"` 产生 `/v1/v1/embeddings` 的问题。
- **Logo 图标居中** (commit [f4c89e3](https://github.com/Oaklight/llm-rosetta/commit/f4c89e3))：修正 icon SVG 中石碑轮廓的居中，改用透明背景并通过 `prefers-color-scheme` media query 自动适配亮暗主题。
- **流式响应 profiler 延迟停止** (PR [#633](https://github.com/Oaklight/llm-rosetta/pull/633))：pyinstrument profiler 现在在整个流式生命周期内运行，而不是在 handler 返回 `StreamingResponse` 时提前停止。
- **启动时自动重建指标计数器** (PR [#643](https://github.com/Oaklight/llm-rosetta/pull/643))：检测非正常关机后的计数器偏差，从请求日志自动重建。
- **OpenAI Responses 输入项 `status` 字段** (PR [#650](https://github.com/Oaklight/llm-rosetta/pull/650))：为所有输入项类型添加 `"status": "completed"`。修复火山引擎（豆包模型）400 `MissingParameter` 错误。

### 变更

- **统一端点 URL 构造** (PR [#588](https://github.com/Oaklight/llm-rosetta/pull/588))：embedding 和 rerank handler 现在使用 `ProviderInfo.upstream_url()`（基于模板）而非临时 f-string 拼接，与 chat 路径架构一致。
- **Vendor 更新** (PR [#594](https://github.com/Oaklight/llm-rosetta/pull/594))：httpclient 0.4.5→0.4.6，sse 0.3.2→0.3.3。

## v0.11.1 — 2026-08-29

### 修复

- **OpenAI Responses reasoning 加密状态** (PR [#576](https://github.com/Oaklight/llm-rosetta/pull/576))：强制同格式 Responses→Responses 流式转换现在保留 `response.output_item.done` 中的 `encrypted_content` 和源 reasoning item ID。此前 IR round-trip 会丢失两者，导致需要回放完成态 reasoning 的客户端（如 `store: false` / ZDR 流程）无法正常工作。修复 [#575](https://github.com/Oaklight/llm-rosetta/issues/575)。
- **Google GenAI reasoning 配置 round-trip** (PRs [#582](https://github.com/Oaklight/llm-rosetta/pull/582), [#583](https://github.com/Oaklight/llm-rosetta/pull/583), [#584](https://github.com/Oaklight/llm-rosetta/pull/584))：从 REST `generationConfig` 中解析入站 `thinkingConfig`，将 reasoning effort 映射到 `thinkingLevel`，在所有转换器间转发 `summary`/`include_thoughts`。
- **Responses API reasoning summary 转发** (PR [#582](https://github.com/Oaklight/llm-rosetta/pull/582))：在出站 Responses API 请求中转发 `reasoning.summary`。

### 变更

- **Gateway `create_app` 可组合化** (PR [#578](https://github.com/Oaklight/llm-rosetta/pull/578))：通过 `GatewayExtensions` 重构 `create_app` 以支持扩展。
- **IR `ReasoningDeltaEvent`** 新增 `encrypted_content` 和 `provider_metadata` 字段声明，与 `ToolCallStartEvent` 保持一致。

## v0.11.0 — 2026-08-28


### 变更

- **Admin UI 配色系统重设计** (PRs [#564](https://github.com/Oaklight/llm-rosetta/pull/564), [#566](https://github.com/Oaklight/llm-rosetta/pull/566), [#571](https://github.com/Oaklight/llm-rosetta/pull/571))：将单一的明/暗主题替换为双配色方案系统 — **Minimal**（Vercel 风格纯黑白）和 **Emerald**（Neon 风格绿色主题），各有明暗两种模式（共 4 种组合）。架构从 JS 驱动的 `THEMES` 对象迁移到 CSS 复合选择器（`[data-scheme][data-mode]`）。设置面板现有独立的配色方案和模式选择器。
- **Admin UI 字体与一致性** (PR [#570](https://github.com/Oaklight/llm-rosetta/pull/570))：将字号从 10 级收敛到 7 级。提取 `.btn-disabled` 类替代内联禁用样式。统一海拔模型——移除设置区域的阴影，统一为 border-hover 模式。标准化表单输入字体（代码输入用等宽字体，下拉选择用无衬线字体）。
- **Admin UI 响应式布局** (PR [#565](https://github.com/Oaklight/llm-rosetta/pull/565))：移动端标签栏溢出滚动、表格滚动包裹、内容区最大宽度 1400px、Provider 卡片网格 280px 最小宽度支持 4 列、移动端设置面板内边距缩小。
- **Admin UI CSS 变量架构** (PR [#571](https://github.com/Oaklight/llm-rosetta/pull/571))：将配色方案的结构差异（表头排版、徽章圆角、图表柱形颜色）从 CSS 选择器覆盖迁移到自定义属性。新增配色方案现在只需在 `base.css` 中添加一个变量块。

### 修复

- **Admin UI CSS bug** (PR [#564](https://github.com/Oaklight/llm-rosetta/pull/564))：修复未定义的 `var(--hover)`、合并重复的 `.btn-danger` 定义、为暗色主题添加 `--purple`、将所有硬编码颜色替换为 CSS 变量和 `color-mix()`。
- **错误转储覆盖率** (PRs [#572](https://github.com/Oaklight/llm-rosetta/pull/572), [#573](https://github.com/Oaklight/llm-rosetta/pull/573))：为 4 个之前未覆盖的失败路径添加 `dump_error()` — 请求阶段转换错误（400）、非流式连接错误（502）、响应阶段转换错误（502）和流式中间错误块。提取 `DumpContext` 数据类简化参数传递。
- **Emoji 空状态图标** 替换为 SVG 线条图标（柱状图、相机、文件夹），匹配极简设计语言。
- **Rosetta Stone SVG favicon** — 将 emoji favicon（🔀）替换为项目的罗塞塔石碑轮廓，同时作为 `<link rel="icon">` 和服务端 `/favicon.ico` 提供。
- **Admin UI 配置路径简化** (PR [#565](https://github.com/Oaklight/llm-rosetta/pull/565)) — 仅显示文件名，完整路径在 tooltip 中。系统时钟现在显示时区缩写。
- **Admin UI Toast 居中** (PR [#565](https://github.com/Oaklight/llm-rosetta/pull/565)) — Toast 通知现在水平居中，不再固定在右下角。
- **OpenAI Responses reasoning 输入生命周期** (PR [#569](https://github.com/Oaklight/llm-rosetta/pull/569))：修复 output-only 字段（`status: "completed"`、合成 `rs_` ID）泄漏到 Responses 请求输入项的问题。来自 Chat/Anthropic/Google 的无来源证明的 reasoning 现在被省略而非分配虚假身份。有真实 Responses 来源的 reasoning 保留原始 ID 和 summary，但移除 output-only 的 status。修复 [#568](https://github.com/Oaklight/llm-rosetta/issues/568)。
- **Admin UI "Allowed Shims" 隐藏** (PR [#574](https://github.com/Oaklight/llm-rosetta/pull/574))：从 API 密钥页面移除了令人困惑的 "Allowed Shims" 列和弹窗输入框——"shim" 是内部概念。后端存储不变（默认 `["*"]`）。

### 新增

- **Admin UI 无障碍性** (PR [#565](https://github.com/Oaklight/llm-rosetta/pull/565))：ARIA 角色（`tablist`/`tab`/`tabpanel`、`radiogroup`/`radio`、`dialog`）、标签和分段控件的方向键导航、弹窗焦点陷阱。
- **Rosetta Stone 头部标志** — 管理面板头部的 SVG 石碑轮廓图标。
- **GitHub 仓库图标** — `design/logo/out/rosetta-icon-github.svg`，用于 GitHub 仓库设置。
- **Admin UI 国际化** — 新增配色方案/模式选择器的中英文标签。
- **设计演示** — `design/ui` 分支上的主题对比演示（Minimal、Emerald、Vercel、Neon、Metallic 风格）。

## v0.10.0 — 2026-08-26

### 新增

- **Nuitka 独立二进制文件** (PR [#555](https://github.com/Oaklight/llm-rosetta/pull/555)): 6 个平台的预编译单文件可执行程序 — linux-x86_64 (glibc + musl)、linux-arm64 (glibc + musl)、macOS arm64、Windows x86_64。无需 Python 运行时。包含 pyinstrument 性能分析支持。
- **基于二进制的 Docker 镜像** (PR [#555](https://github.com/Oaklight/llm-rosetta/pull/555)): 三种镜像变体 — `alpine`（musl 二进制，~21 MB，默认）、`glibc`（busybox:glibc，~25 MB）、`python`（pip 安装，~80 MB）。Alpine 变体同时标记为 `:latest` 和 `:<version>`。
- **Makefile 构建目标** (PR [#555](https://github.com/Oaklight/llm-rosetta/pull/555)): `build-binary`、`build-binary-musl`、`build-docker-alpine`、`build-docker-glibc`、`build-docker-python`，用于本地和 CI 构建。

### 变更

- **Docker 权限模型** (PR [#555](https://github.com/Oaklight/llm-rosetta/pull/555)): 移除 su-exec/PUID/PGID，改用 Docker 原生的 `USER appuser` + `--user` 参数。使用 `docker run --user $(id -u):$(id -g)` 进行自定义 UID 映射。
- **Docker 默认镜像** — `:latest` 现在指向 Alpine 二进制镜像（~21 MB），而非 Python 镜像（~80 MB）。Python 镜像仍可通过 `:<version>-python` 获取。

## v0.9.0 — 2026-08-21

### 新增

- **Rerank IR 类型与转换器族** (PR [#506](https://github.com/Oaklight/llm-rosetta/pull/506))：5 个 TypedDict IR 类型和 3 个转换器（Jina、Cohere、Voyage），覆盖所有主流 rerank API 格式族。
- **Embedding IR 类型与转换器族** (PR [#510](https://github.com/Oaklight/llm-rosetta/pull/510))：6 个 IR 类型和 4 个转换器（OpenAI、Jina、Voyage、Cohere），支持跨 provider 的 embedding 格式转换。
- **网关 Rerank 代理** (PRs [#511](https://github.com/Oaklight/llm-rosetta/pull/511), [#512](https://github.com/Oaklight/llm-rosetta/pull/512))：`/v1/rerank` 和 `/v2/rerank` 端点，通过 IR 进行跨格式转换。配置驱动路由：`rerank_providers`、`rerank_models`、`default_rerank_format`。`/v2/rerank` 自动检测 Cohere 格式。
- **网关 Embedding IR 转换模式** (PR [#517](https://github.com/Oaklight/llm-rosetta/pull/517))：将 `/v1/embeddings` 从直接透传升级为 IR 转换模式（OpenAI↔Cohere↔Jina↔Voyage）。向后兼容——没有配置 `embedding_providers` 的配置仍使用透传模式。
- **`UpstreamTimeoutError`** (PR [#513](https://github.com/Oaklight/llm-rosetta/pull/513))：区分上游超时（504）和连接错误（502）。Rerank 和 embedding 处理器在超时时返回正确的 504。
- **`x-rosetta-conversion: passthrough` 头部** — 当响应转换失败时返回，让客户端可以检测到回退了原始上游格式。
- **Rerank API 格式文档** — 双语（中/英）文档，覆盖 Jina、Cohere、Siliconflow、Voyage rerank API 格式，含 provider 族谱和 IR 映射表。
- **Embedding API 格式文档** — 双语（中/英）文档，覆盖 OpenAI、Cohere、Jina、Voyage embedding API 格式。
- **ConversionPipeline 透传模式** (PR [#520](https://github.com/Oaklight/llm-rosetta/pull/520))：`force_conversion` 参数（默认 `True`）。当 `False` 且 source == target 时，跳过 IR round-trip——修复同格式代理时的信息丢失（如 Claude Code → 网关 → Anthropic 上游）。
- **转换器实例缓存** (PR [#520](https://github.com/Oaklight/llm-rosetta/pull/520))：`get_converter_for_provider()` 在模块级 dict 中缓存实例，消除每请求的转换器分配。
- **保真度检查器** (`fidelity.py`, PR [#520](https://github.com/Oaklight/llm-rosetta/pull/520))：对比原始 body 和 round-trip 后的 body，检测 IR 转换信息损失。两种模式：`"critical"`（按格式检查关键字段，~0.01ms）和 `"full"`（递归叶级 diff）。通过 `fidelity_mode` 参数接入 pipeline 透传路径，实现后台监控。
- **`StreamProcessorProtocol`** (PR [#520](https://github.com/Oaklight/llm-rosetta/pull/520))：`StreamProcessor` 和 `PassthroughStreamProcessor` 的共享 `Protocol`，含终端事件检测。
- **Rerank 源格式自动检测** (PR [#522](https://github.com/Oaklight/llm-rosetta/pull/522)) — 从请求体中的 `top_k` 检测 Voyage 格式；`/v2/rerank` 路径推断 Cohere 格式。
- **Embedding 源格式自动检测** (PR [#521](https://github.com/Oaklight/llm-rosetta/pull/521)) — 从请求体字段自动检测 embedding 源格式（`input_type` → Cohere/Jina/Voyage；`encoding_format` 候选值消歧）。
- **Admin `disabled_tabs` 参数** (PR [#505](https://github.com/Oaklight/llm-rosetta/pull/505))：`setup_admin(disabled_tabs=["metrics"])` 在初始化时隐藏 admin UI 标签页。
- **Shim `multimodal_tool_result` 能力声明** (PRs [#523](https://github.com/Oaklight/llm-rosetta/pull/523), [#524](https://github.com/Oaklight/llm-rosetta/pull/524))：`ProviderShim` 现在可以在 YAML 中声明 `multimodal_tool_result: true/false` 以覆盖转换器的类级别默认值。该标志通过 `ConversionContext.options` 在 `convert()` 和 `ConversionPipeline` 中传递。Chat 转换器将其透传至 `_convert_tool_result_with_packing`，使多模态内容在 provider 原生支持时得以保留。
- **流式拒绝事件** (PR [#528](https://github.com/Oaklight/llm-rosetta/pull/528))：OpenAI Responses 转换器支持 `response.refusal.delta` / `response.refusal.done` SSE 事件。新增 `RefusalDeltaEvent` IR 流事件类型，支持完整的 p→ir→p 往返转换。完成 issue [#431](https://github.com/Oaklight/llm-rosetta/issues/431)。

### 修复

- **Google GenAI 多模态工具结果处理** (PR [#525](https://github.com/Oaklight/llm-rosetta/pull/525))：工具结果中的结构化内容块（`list[ContentPart]`）现在原样保留，不再通过 `json.dumps`/`str()` 扁平化。dict 内容使用 `json.dumps`（而非 `str()` 产生无效的 Python repr）。`_is_content_block_list` 守卫区分类型化内容块和普通数据列表。
- **Chat 转换器多模态内容丢失** (PR [#524](https://github.com/Oaklight/llm-rosetta/pull/524))：`_do_request_to_provider` 未将 `supports_multimodal_tool_result` 传递给 `ir_messages_to_p`，导致 shim 覆盖在实际请求路径中无效。此外，`_convert_tool_result_with_packing` 在标志为 True 时仍然从工具消息中剥离图片——图片被打包但未重新注入，导致内容静默丢失。
- **测试顺序不稳定** (PR [#523](https://github.com/Oaklight/llm-rosetta/pull/523))：`test_shims.py` fixture 现在保存/恢复全局 shim 注册表而非清空，防止跨模块测试失败。

### 变更

- **统一 `convert()` 和 `ConversionPipeline`** (PR [#520](https://github.com/Oaklight/llm-rosetta/pull/520))：两者现在都支持双 shim（source + target）变换和响应转换。`ConversionPipeline` 内部委托给 `convert()`，消除代码分歧。
- **统一出站传输层** (PR [#516](https://github.com/Oaklight/llm-rosetta/pull/516))：将 `send_request` 和 `send_passthrough` 合并为单一 `send(provider_info, url, body)` 方法。URL 构建和流式标志注入从传输层移至代理层。净减 68 行。
- **文档结构重组** — 拆分为三个顶级标签页：IR 类型系统、库、网关。API 参考合并到各标签页内。移除独立的 API 参考标签页。

## v0.8.2 — 2026-08-09

### 新增

- **Per-provider/model 超时覆盖** (PR [#502](https://github.com/Oaklight/llm-rosetta/pull/502))：支持按服务方和模型粒度配置上游超时，取代全局统一值。优先级：`model.timeout > provider.timeout > server.upstream_timeout`。Admin UI 在服务方和模型弹窗中增加超时输入框。
- **Prompt cache 保持** (PR [#499](https://github.com/Oaklight/llm-rosetta/pull/499))：将 `hoist_late_system_messages` IR 变换接入全部 15 个 provider shim。对话中间的 system/developer 消息被改写为 user 角色 `[System: ...]` 信封，保持 prompt cache 前缀稳定。
- **Per-provider hoist 开关** (PR [#499](https://github.com/Oaklight/llm-rosetta/pull/499))：gateway 配置中的 `hoist_system_messages` 布尔值，可通过 admin UI 复选框按 provider 覆盖，带 (i) 提示弹窗。
- **SQLite API 密钥存储** (PR [#496](https://github.com/Oaklight/llm-rosetta/pull/496))：将 API 密钥存储从明文配置迁移至 SQLite，使用哈希验证。

### 变更

- **降低圈复杂度** — 重构 base converter helpers、Google GenAI converter、OpenAI Responses converter、gateway config 和 SOCKS5 test handler，降低圈复杂度。
- **complexipy v6 pre-commit hook** — 启用 complexipy v6 pre-commit hook；升级至 v6.2.0，阈值提升至 30。
- **配置覆盖解析统一** (PR [#501](https://github.com/Oaklight/llm-rosetta/pull/501))：将 per-provider 开关解析统一为 Pattern C —— `config.resolve()` 使用 shim 默认值作为 fallback，下游类型从 `bool | None` 简化为 `bool`。

### 修复

- **提示弹窗悬停** (PR [#499](https://github.com/Oaklight/llm-rosetta/pull/499))：admin UI 的提示弹窗现在可以悬停选中文字（CSS `::before` 桥接图标与弹窗的间隙）。
- **Abort-path 测试替身** (PR [#497](https://github.com/Oaklight/llm-rosetta/pull/497))：用真实 `StreamContext` 替换 `_FakeContext`/`_FakeProcessor`，消除接口漂移风险。

### 移除

- **向后兼容别名** (PR [#492](https://github.com/Oaklight/llm-rosetta/pull/492))：移除 openai_responses 和 google_genai converter 中的 `to_provider()` 兼容别名。
- **废弃兼容 shims** — 移除 base converter 和各 converter 中的废弃向后兼容 shims 和别名。
- **过期文档** — 移除过期的 base converter README 文件。
- **多余文件** — 移除项目根目录中的多余 `validate.py` 文件。

## v0.8.1 — 2026-08-07

### Bug 修复

- **Custom tool grammar 格式** (PR [#489](https://github.com/Oaklight/llm-rosetta/pull/489))：修复 grammar 约束的 custom tools（如 Codex `apply_patch`）在 Chat Completions 上游失败的问题。Responses API 使用扁平 `format` 格式（`{type, syntax, definition}`），Chat Completions 要求嵌套在 `format.grammar` 下。在 Chat 边界添加了双向幂等的格式转换。
- **流式 null union 成员** (PR [#489](https://github.com/Oaklight/llm-rosetta/pull/489))：修复流式 custom tool call delta 导致 `AttributeError` 崩溃的问题。Provider 在每个 delta 中序列化 union 的所有成员，非活跃成员设为 `null`；`dict.get("type", "function")` 在 key 存在但值为 null 时返回 `None`。现在从有数据的 payload 推断类型，回退到 context 中注册的类型。

### 改进

- **Tool call order 公开 API** (PR [#490](https://github.com/Oaklight/llm-rosetta/pull/490))：4 个 converter 中对 `_tool_call_order` 的私有访问全部替换为 `StreamContext` 上的公开方法：`tool_call_ids`、`tool_call_count`、`resolve_tool_call_id_by_index()`、`get_tool_call_index()`。
- **O(1) tool call index 查找** — `get_tool_call_index()` 使用反向 dict 替代线性 `list.index()` 扫描。

## v0.8.0 — 2026-08-06

### Spec 合规性

对所有四个转换器进行系统性检查，确保输出符合官方 API 规范。由 [llm-comply](https://github.com/Oaklight/llm-comply) 合规性测试驱动。

- **Anthropic usage 字段** — 发送所有 spec 要求的 usage 字段（`input_tokens`、`output_tokens`、`cache_creation_input_tokens`、`cache_read_input_tokens`）。响应中始终发送 `stop_sequence` 和 `stop_details`。
- **Anthropic caller、citations、container** — 在 Anthropic 响应输出中添加 `caller`、`citations`、`container` 字段。
- **Anthropic 流式 `message_start.input_tokens`** ([#424](https://github.com/Oaklight/llm-rosetta/issues/424)，PR [#425](https://github.com/Oaklight/llm-rosetta/pull/425))：修复 `message_start` 始终报告 `input_tokens=0`，通过交换 `UsageEvent`/`StreamStartEvent` 的发送顺序。
- **OpenAI Chat `logprobs`** — 响应 choices 中始终发送 `logprobs`（spec 要求的 nullable 字段）。
- **OpenAI Chat `finish_reason`** — 所有流式 chunk 中包含 `finish_reason: null`；将 IR refusal reason 显式映射为 `stop`。
- **OpenAI Chat `annotations`** — 将 annotations 嵌套在 `url_citation` 下并始终发送该字段。
- **OpenAI Responses `response.in_progress`** — 按 spec 发送 `response.in_progress` 流式事件。
- **Google `responseId` / `modelVersion`** — 在流式 chunk 中包含 `responseId` 和 `modelVersion`。
- **Google `ModalityTokenCount`** — 过滤非标准 modality 值；在 Google 和 IR 格式之间规范化 schema type 大小写。
- **Google 流式 usage** — 在跨格式流式中于 `stream_end` 时刷新待处理的 usage；提取 `_build_stream_usage_metadata` 辅助函数。

### 跨格式 Refusal 处理

- **完整 refusal 支持** ([#429](https://github.com/Oaklight/llm-rosetta/issues/429))：所有 4 个转换器双向处理 refusal：
    - OpenAI Chat ([#430](https://github.com/Oaklight/llm-rosetta/issues/430))：`refusal` 字段始终存在（nullable），流式 `delta.refusal` 累积。
    - Anthropic ([#432](https://github.com/Oaklight/llm-rosetta/issues/432))：结构化 `stop_reason: "refusal"` + `stop_details`（category/explanation），包括流式。
    - Open Responses ([#431](https://github.com/Oaklight/llm-rosetta/issues/431))：`RefusalContent` 类型解析和生成。
    - Google ([#433](https://github.com/Oaklight/llm-rosetta/issues/433))：`promptFeedback.blockReason` 处理，补全缺失的 `finishReason` 值（`BLOCKLIST`、`PROHIBITED_CONTENT`、`SPII`、`IMAGE_SAFETY`），通过 `_provider_metadata` 标记实现 refusal round-trip。
- **IR RefusalPart** ([#427](https://github.com/Oaklight/llm-rosetta/pull/427))：`RefusalPart` 加入 `AssistantContentPart` union——refusal 响应不再因 IR 验证失败而返回 502。

### 响应标识与元数据

- **Shim 驱动的 response ID 前缀** ([#410](https://github.com/Oaklight/llm-rosetta/issues/410)，PR [#420](https://github.com/Oaklight/llm-rosetta/pull/420))：`ProviderShim` 在 `provider.yaml` 中声明 `response_id_prefix`。Converter 在输入时 strip 源前缀，输出时添加目标前缀。OpenAI (`chatcmpl-`)、Anthropic (`msg_`)、OpenAI Responses (`resp_`) 前缀已声明。
- **`completed_at` 时间戳** ([#410](https://github.com/Oaklight/llm-rosetta/issues/410))：Responses converter 在 completed 状态时设置 `completed_at` 为 Unix 时间戳（之前始终为 `null`）。
- **函数调用项 ID 保留** — 在所有转换路径中保留 `function_call` 项 ID。
- **HTTP 边界 strip `_provider_metadata`** ([#422](https://github.com/Oaklight/llm-rosetta/issues/422)，PR [#423](https://github.com/Oaklight/llm-rosetta/pull/423))：内部 `_provider_metadata` 字段不再泄漏到出站 HTTP 请求或下游响应中。

### Responses API 流式

- **统一 `output_index`** ([#418](https://github.com/Oaklight/llm-rosetta/issues/418)，PR [#419](https://github.com/Oaklight/llm-rosetta/pull/419))：用 `OpenAIResponsesStreamContext` 上的单一递增计数器替代分散计算。所有输出项类型统一通过 `next_output_index()` 分配索引。
- **Reasoning 输出项** ([#407](https://github.com/Oaklight/llm-rosetta/issues/407)、[#408](https://github.com/Oaklight/llm-rosetta/issues/408)，PR [#419](https://github.com/Oaklight/llm-rosetta/pull/419))：非流式 reasoning 项始终包含 `id` 和 `status` 字段。流式 reasoning 具备完整生命周期事件。使用 SHA-256 哈希确定性生成 reasoning ID。
- **Reasoning 项顺序** ([#437](https://github.com/Oaklight/llm-rosetta/issues/437)，PR [#438](https://github.com/Oaklight/llm-rosetta/pull/438))：修复 `response_to_provider` 输出顺序——reasoning 项现在正确排在 message 项之前。
- **消息阶段（phase）保留** ([#440](https://github.com/Oaklight/llm-rosetta/issues/440)，PR [#441](https://github.com/Oaklight/llm-rosetta/pull/441))：Responses API 的 `phase` 字段（`commentary`/`final_answer`）在所有转换路径和流式中保留。Pipeline 在流式 context 之间桥接 phase。
- **SSE `[DONE]` 终止符** ([#409](https://github.com/Oaklight/llm-rosetta/issues/409))：Gateway 在 `response.completed` 后发送 `[DONE]` 作为最终 SSE 事件。

### IR 与架构

- **Provider 透传事件** — 新增非流式和流式 provider 透传 IR 类型，允许转换器转发 provider 特定数据而不丢失。
- **空 reasoning 内容** — OpenAI Chat 转换器保留空 reasoning 内容而非丢弃。
- **多模态 tool result 载荷重复** ([#480](https://github.com/Oaklight/llm-rosetta/issues/480)、PR [#482](https://github.com/Oaklight/llm-rosetta/pull/482))：目标格式为 OpenAI Chat 时，tool result 中的图片会被发送两次 —— 一次是合成 user 消息中的真实图片块，一次是 `role: "tool"` 消息体中经 `json.dumps()` 序列化的无效 base64 文本 —— 使载荷恰好翻倍，触发上游请求大小限制。现已打包的块会从 tool 消息中剥离；打包失败的块予以保留。

### 网关

- **CORS 预检 auth 绕过** (PR [#404](https://github.com/Oaklight/llm-rosetta/pull/404)、[#405](https://github.com/Oaklight/llm-rosetta/pull/405))：对 CORS 预检请求跳过认证；加强为要求同时具备 `Origin` + `Access-Control-Request-Method` 头部。
- **auth 错误响应 CORS 头部** — auth 错误响应中添加 CORS 头部。
- **Bearer token 降级** — 对所有认证策略接受 `Bearer` token 作为降级方式。
- **Embedding 请求 ID** — 对齐 embedding 请求 ID 与上游格式。
- **上游流式中断时发送终止事件** ([#481](https://github.com/Oaklight/llm-rosetta/issues/481)、PR [#483](https://github.com/Oaklight/llm-rosetta/pull/483))：上游连接在流式传输中途断开时，网关会直接关闭 SSE 连接而不发送任何终止事件 —— 等待终止事件的客户端只能得到一句 `stream closed before response.completed`，无从得知原因。现在网关会发送符合目标格式的终止事件并携带上游原因（Responses 为 `response.failed` + `[DONE]`，Chat 为 error chunk + `[DONE]`，Anthropic 为 `event: error`，Google 为 error 对象）。流已正常结束或客户端已断开时跳过。新增 `StreamProcessor.source_context`，以及 `StreamContext.next_sequence_number` 和 `StreamContext.outbound_response_id`。
- **结构化 JSON 日志** (PR [#468](https://github.com/Oaklight/llm-rosetta/pull/468))：可配置 `debug.log_format`（`json`/`text`/`auto`）。JSON 模式每行输出一个 JSON 对象，UTC ISO 8601 时间戳，结构化 extras 通过 allowlist 提升为顶层键。`auto` 非 TTY 时为 `json`，交互式为 `text`。支持管理 API 热加载。
- **流内上游错误 surface** (PR [#454](https://github.com/Oaklight/llm-rosetta/pull/454))：上游在 200 SSE 流内报告请求错误时（如 Argo 对超限 tools 发送 `event: error`），之前错误块会被静默吞掉，客户端收到成功但空的响应。现在检测裸 error envelope，发送格式匹配的错误事件，并正常终止流。
- **管理面板 custom tools 开关** (PR [#467](https://github.com/Oaklight/llm-rosetta/pull/467))：在管理面板 provider 设置中暴露 `supports_custom_tools` 复选框，含 (i) 提示 tooltip。
- **可配置超时** (PR [#463](https://github.com/Oaklight/llm-rosetta/pull/463))：`server.upstream_timeout` 和 `server.read_timeout` 配置项（均默认 300 秒）。
- **根路径重定向** (PR [#461](https://github.com/Oaklight/llm-rosetta/pull/461))：`server.root_redirect` 配置项，将 `GET /` 重定向到管理面板。
- **匿名访问选项** — `server.open_on_no_keys` 在未配置 API key 时允许匿名访问。
- **管理面板 modal 优化** — CSS 修复、展平提示 tooltip、i18n 对齐。


### Shim 与转换

- **自动注入 Anthropic 缓存断点** ([#464](https://github.com/Oaklight/llm-rosetta/issues/464)、[#465](https://github.com/Oaklight/llm-rosetta/issues/465)，PR [#469](https://github.com/Oaklight/llm-rosetta/pull/469))：跨格式请求（OpenAI/Gemini → Anthropic）缺少缓存语义时，自动通过 `auto_cache_breakpoints` IR 转换注入最多 4 个 `cache_hint` 断点。断点位置：最后一个 tool 定义、system 指令尾部、最后两条 user 消息。挂载在 `argo--anthropic` 和 `openrouter--anthropic` shim 上。两种模式：`none_only`（默认，已有 hint 时跳过）和 `fill_gaps`（按段独立填充）。
- **非支持 Chat 上游的 custom tool 降级** ([#460](https://github.com/Oaklight/llm-rosetta/issues/460)，PR [#486](https://github.com/Oaklight/llm-rosetta/pull/486))：`ProviderShim` 新增 `supports_custom_tools` 标志（默认 `False`）。目标 Chat 上游不支持 `{type: "custom"}` tool 定义时，请求时降级为 `{type: "function"}`，响应时恢复原始类型。仅 OpenAI shim 设为 `true`。

### 转换器

- **OpenAI Chat 原生 custom tool 支持** — `openai_chat` 转换器原生处理 `custom` tool 类型，`ConversionContext` 新增 `set_tool_call_type()` 公开 API。
- **`custom_tool_call_output` 项类型** — OpenAI Responses 转换器支持 `custom_tool_call_output` 输入项。
- **BaseConverter 模板方法重构** — `BaseConverter` 转换为模板方法模式，通过 `__init_subclass__` 强制 `_PASSTHROUGH_RESTORE_KEY`。

### CI 与文档

- **按需合规性测试** — 添加 [llm-comply](https://github.com/Oaklight/llm-comply) GitHub Actions 工作流，对网关进行 schema/spec 级合规性测试。
- **README 添加合规性测试章节** — 链接到 llm-comply、托管服务和 CI 工作流。

## v0.7.3 — 2026-07-25

### 新增

- **模型启用/禁用开关** ([#382](https://github.com/Oaklight/llm-rosetta/pull/382))：管理面板中为每个模型添加了 ON/OFF 药丸开关。禁用的模型不参与路由（`_parse_models` 跳过 `enabled: false`）。后端路由：`toggle_model`、`bulk_update_models`（批量启用/禁用/删除）。
- **Embedding 测试菜单** ([#382](https://github.com/Oaklight/llm-rosetta/pull/382))：Embedding 模型现在显示专属测试选项——Embedding、批量（文本数组）、套娃 Matryoshka（用户指定维度）、多模态（图片）。Matryoshka 使用自定义 modal 替代原生 `prompt()` 弹窗。
- **URL 模板管理面板 UI**：可在 provider 和 model 卡片中直接配置自定义上游 URL 模板。

### 变更

- **模型 modal 三 tab 布局** ([#389](https://github.com/Oaklight/llm-rosetta/pull/389))：重新设计模型编辑/添加弹窗为三 tab 布局——基本（名称+Provider 并排、分段 LLM/Embedding 控件、药丸样式能力标签）、路由（URL 模板 + 流式展开链接）、转换（展平系统消息 + 推理配置）。替换了原来的长滚动单面板表单。
- **模型表格 UI 重构** ([#382](https://github.com/Oaklight/llm-rosetta/pull/382))：新增 checkbox 列支持多选，顶部显示批量操作栏（启用/禁用/删除）。Clone 和 Delete 收入 ⋯ 下拉菜单。Test 按钮在 LLM 和 embedding 模型间统一宽度。
- **原子化配置写入** ([#387](https://github.com/Oaklight/llm-rosetta/pull/387))：`write_config` 改用 `tempfile.mkstemp` + `fsync` + `os.replace` 实现崩溃安全的跨平台原子写入。移除所有平台特定锁代码（`fcntl`/`msvcrt`）。读者永远不会看到写了一半的文件。
- **跨进程配置串行化** ([#387](https://github.com/Oaklight/llm-rosetta/pull/387))：新增 `config_lock(path)` 上下文管理器，使用 `.lock` sidecar 文件配合 `fcntl.flock`（Unix）/ `msvcrt.locking`（Windows）实现跨进程互斥。保护多个 gateway 实例共享同一配置文件的场景。14 个 admin route handler 全部包裹以串行化 read-modify-write 周期。

### 修复

- **Windows 兼容性** ([#381](https://github.com/Oaklight/llm-rosetta/pull/381))：Gateway 不再在顶层导入仅 Unix 可用的 `fcntl` 模块。
- **保留上游 User-Agent 头** ([#385](https://github.com/Oaklight/llm-rosetta/pull/385))：Gateway 现在将客户端的 `User-Agent` 头传递给上游 provider，而非丢弃。
- 修复 usage token details 中 null 值在 IR 校验前未过滤的问题。
- 修复 `test_auth` 与 `open_on_no_keys` 行为不一致的问题 ([#388](https://github.com/Oaklight/llm-rosetta/pull/388))。

### 安全

- **无 API key 时拒绝请求** ([#383](https://github.com/Oaklight/llm-rosetta/pull/383))：当 `api_keys` 为空且未设置 `open_on_no_keys` 时，API 请求现在返回 403 而非静默放行。

## v0.7.2 — 2026-07-20

### 新增

- **管理面板 `custom_head` 注入** ([#378](https://github.com/Oaklight/llm-rosetta/pull/378))：`setup_admin()` 接受可选的 `custom_head` HTML 片段，注入到 `</head>` 之前。下游项目可注入 `<style>`/`<script>` 标签来定制管理面板 UI，无需修改参考 `admin.html`。按值缓存，无每次请求开销。
- **管理面板 `branding` 品牌配置** ([#378](https://github.com/Oaklight/llm-rosetta/pull/378))：`setup_admin(..., branding={title, subtitle, version, links, attribution})` 可定制页头、登录页面和设置页脚。通过 `custom_head` 序列化为 `window.__branding`；`admin.html` 中的消费脚本负责修改 DOM。新增元素 ID：`brandTitle`、`brandLoginTitle`、`brandFooterName`、`brandFooterLinks`。未提供 branding 时，默认 llm-rosetta 标识不变。

### 变更

- 升级 vendored `httpclient` 0.4.4 → 0.4.5——修复 fd 泄漏问题：`close()` 未关闭 `_async_writer`，导致 `__del__` 无法清理泄漏的异步流式响应。
- **提取 `ConfigIO` 协议用于管理面板配置读写** ([#376](https://github.com/Oaklight/llm-rosetta/pull/376))：管理面板路由现在通过 `ConfigIO` 协议而非直接导入 `load_config`/`load_config_raw`/`write_config`。默认 `JsoncConfigIO` 实现保持现有行为不变；下游项目（如 argo-proxy）可通过 `setup_admin(..., config_io=...)` 提供替代实现。内部辅助函数 `_get_config_path` 和 `_get_config_io` 在值缺失时抛出描述性 `RuntimeError`，移除了路由处理器中 16 处冗余的空值检查。
- 内容捕获表格中将 Unicode emoji（🔍）替换为内联 SVG，确保跨平台渲染一致。

### 修复

- 修复 branding JSON 序列化中 `</` 未转义的问题，防止 branding 值包含 `</script>` 时导致 `<script>` 标签断裂。

## v0.7.1 — 2026-07-16

### 修复

- **Anthropic 和 Google 的工具 Schema 清理** ([#372](https://github.com/Oaklight/llm-rosetta/issues/372))：Anthropic 拒绝工具参数 schema 中的 OpenAPI `nullable` 扩展（例如 Pydantic 生成的 JSON Schema）。新增 `convert_nullable_to_type_array()` helper，递归地将 `"nullable": true` 转换为标准 JSON Schema `"type": [T, "null"]`。Anthropic converter 现在会剥离 `title` 字段并转换 `nullable` 为 type 数组；Google GenAI converter 剥离 `title`（保留 `nullable`——Google 支持该字段）。同时处理了 `nullable: true` 与 `anyOf`/`oneOf` 共存但无 `type` 字段的边界情况。
- **`flatten_system` 复选框布局和国际化** 修复（网关管理面板）。

### 变更

- 升级 vendored `validate` 0.6.0 → 0.6.1（支持 dataclass 实例）。
- 限制 Dependabot 仅监控 LLM SDK 依赖。

### 新增

- SDK 类型覆盖扫描器和手动 CI 工作流，用于追踪 provider SDK 类型对齐情况。

## v0.7.0 — 2026-07-10

### 新增

- **Anthropic `cache_control` 保留** ([#362](https://github.com/Oaklight/llm-rosetta/pull/362))：IR 部件（`TextPart`、`ImagePart`、`FilePart`、`ReasoningPart`、`ToolCallPart`、`ToolResultPart`、`ToolDefinition`）新增 `cache_hint` 字段，支持 Anthropic block 级 `cache_control` 在 IR 管线中的往返传递。Anthropic converter 在输入时读取 `cache_control` → `cache_hint`，在输出时写回；非 Anthropic converter 静默忽略 `cache_hint`，确保跨格式安全。
- **`flatten_system_content()` 变换** ([#370](https://github.com/Oaklight/llm-rosetta/issues/370))：新增 body 级变换工厂，将系统消息内容数组展平为纯文本字符串。OpenAI Chat converter 现在为系统消息输出结构化内容数组（保留 `cache_hint` 的 block 边界）；`flatten_system_content()` 在需要时降级为纯字符串以兼容上游。支持 per-model `flatten_system` 网关配置，Gemini 模型自动检测。管理面板包含开关控件。

### 修复

- **OpenAI SDK 2.45+ 兼容性**：在 `InputTokensDetails`（Responses API）和 `PromptTokensDetails`（Chat Completions API）TypedDict 副本中添加 `cache_write_tokens` 字段，以匹配上游 SDK 变更。

### Changed

- **Transform 字段重命名** —— `from_transforms` → `pre_ir_transforms`，`to_transforms` → `post_ir_transforms`（`ProviderShim` 上的字段）。旧名称在构造函数参数和 `transforms.py` 导出中均作为向后兼容别名继续有效。
- **`system_instruction` 统一为 `list[TextPart]`** ([#364](https://github.com/Oaklight/llm-rosetta/issues/364))：IR 中 `system_instruction` 的规范形式从 `str` 改为 `list[TextPart]`。单个字符串 `"You are helpful."` 表示为 `[TextPart(type="text", text="You are helpful.")]`。确保所有 converter 间结构一致，并支持 block 级元数据（如 Anthropic prompt caching 的 `cache_hint`）在 IR 管线中流转。四个 converter 均已更新。**Breaking**：直接将 `ir_request["system_instruction"]` 当 `str` 读取的代码需要改为处理 `list[TextPart]`。

## v0.7.0a1 — 2026-06-27

### 新增

- **可观测性包** ([#341](https://github.com/Oaklight/llm-rosetta/issues/341))：将 `MetricsCollector`、`RequestLog`、`RequestLogEntry`、`PersistenceManager` 和 `ProfilerState` 从 `gateway/admin/` 提取到新的顶层 `llm_rosetta.observability` 包。这些模块与框架无关，任何 LLM 代理消费者（如 argo-proxy）均可直接使用，无需依赖网关的配置系统或 HTTP 服务器。`gateway/admin/` 模块通过重新导出保持完全向后兼容
- **混合性能分析系统** ([#339](https://github.com/Oaklight/llm-rosetta/pull/339))：`ConversionPipeline.profile` 中内置 always-on 的 `perf_counter` 阶段计时（source_to_ir_ms、ir_transforms_ms、ir_to_target_ms 等），加上按需的 per-request pyinstrument 深度分析（通过 admin API 控制）。核心库新增 `DeepProfiler` 上下文管理器（`llm_rosetta.profiling`）。新增 `[profiling]` 可选依赖组。Gateway admin 端点：`POST /admin/api/profiling/enable`、`GET /admin/api/profiling/results`、`GET /admin/api/profiling/results/<index>`、`POST /admin/api/profiling/disable`、`DELETE /admin/api/profiling/results`
- **性能分析管理 UI** ([#339](https://github.com/Oaklight/llm-rosetta/pull/339))：管理面板新增 "Profiling" 区域，包含启用/停用控制、结果列表、火焰图下载（单个和批量）以及重启提示
- **错误转储功能** ([#341](https://github.com/Oaklight/llm-rosetta/issues/341))：Fire-and-forget 错误转储系统，在上游/转换失败时捕获完整请求上下文。哈希前进行图片卸载以实现基于内容的去重，zlib 压缩，10K 条目上限并级联修剪。proxy.py/app.py 中 4 个触发点覆盖上游错误、流头部错误、流块错误和转换错误。新导出函数：`dump_error()`、`offload_images()`、`compute_body_hash()`、`compress_body()`、`decompress_body()`（从 `llm_rosetta.observability`）
- **指标重建** ([#340](https://github.com/Oaklight/llm-rosetta/pull/340))：`POST /admin/api/metrics/rebuild` 端点及管理面板 "Rebuild Counters" 按钮。使用 `fetchmany(5000)` 批量迭代和原子交换从请求日志历史中重建所有指标计数器，避免暴露半重建状态

### 修复

- **指标按提供方名称分组** ([#340](https://github.com/Oaklight/llm-rosetta/pull/340))：面板 breakdown 区域现在按提供方显示名称分组，而非按 API 类型（此前将所有 Anthropic 格式的提供方合并为一行）
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

- **Metadata preservation for lossless A→IR→A round-trip** (Issue [#60](https://github.com/Oaklight/llm-rosetta/issues/60), PR [#119](https://github.com/Oaklight/llm-rosetta/pull/119)): New `MetadataMode` (`"strip"` / `"preserve"`) option in `ConversionContext` that captures provider-specific fields during `from_provider` and re-injects them during `to_provider`, enabling lossless round-trip conversion. Helper methods on `ConversionContext`: `store_request_echo()`, `store_response_extras()`, `store_output_items_meta()`, `get_echo_fields()`, `get_output_items_meta()`. Per-provider coverage:
    - **OpenAI Responses**: captures/restores 28+ echo fields (temperature, tools, reasoning, truncation, etc.), per-output-item metadata (id, status, annotations, logprobs), `RESPONSES_REQUIRED_DEFAULTS` dict for spec-required fields with sensible defaults, `sequence_number` on all SSE events
    - **Anthropic**: preserves `stop_sequence`, `container`, citations, and OpenRouter extension usage fields
    - **OpenAI Chat**: now re-emits `refusal` and `annotations` fields in `response_to_provider` (previously dropped)
    - **Google GenAI**: preserves `promptTokensDetails` and `cachedContentTokenCount` in usage metadata
    - **Gateway**: automatically enables preserve mode for both streaming and non-streaming paths; bridges metadata between `from_ctx` and `to_ctx` during streaming

### Fixed

- **Open Responses spec compliance for streaming and non-streaming**: Added required fields to all SSE events (`item_id`, `logprobs`, `annotations`, `status`, `sequence_number`, `output_index`, `content_index`), usage detail breakdowns (`output_tokens_details`, `input_tokens_details`), message item IDs and status for non-streaming output items, `function_call` status field in tool_ops, `service_tier` default to `"default"` (string, not null per spec), `completed_at` in required defaults, `created_at` fallback to current time when not provided, normalized echoed tools with `strict: null`, and metadata bridging from `from_ctx` to `to_ctx` in gateway streaming. All 6 Open Responses compliance tests now pass (schema + semantic)

## v0.3.1 — 2026-04-07

### Fixed

- **`service_tier: None` and `system_fingerprint: None` causing validation errors** (PR [#118](https://github.com/Oaklight/llm-rosetta/pull/118)): OpenAI upstream returns these fields as `null`, but the existence check (`if "key" in dict`) passed and assigned `None` to IR's `NotRequired[str]` field. Changed to value-not-None check in both OpenAI Chat and OpenAI Responses converters. Discovered via [Oaklight/argo-proxy#99](https://github.com/Oaklight/argo-proxy/issues/99)
- **Base `StreamContext` missing provider-specific attributes in Responses streaming** (PR [#118](https://github.com/Oaklight/llm-rosetta/pull/118)): When a gateway passes a base `StreamContext` to `OpenAIResponsesConverter.stream_response_to_provider()`, the method accesses `accumulated_text`, `output_item_emitted`, etc. that only exist on `OpenAIResponsesStreamContext`. Added auto-upgrade via `from_base()` classmethod with metadata caching to preserve state across calls

## v0.3.0 — 2026-04-07

### Added

- **Multimodal tool result support across all 4 converters** (Issue [#92](https://github.com/Oaklight/llm-rosetta/issues/92), PR [#109](https://github.com/Oaklight/llm-rosetta/pull/109)): Tools can now return multimodal content (text + images + files) as `ToolResultPart.result`. Three providers (Anthropic, OpenAI Responses, Google GenAI) support this natively; content blocks are converted through each provider's `content_ops` layer. See provider support matrix below
- **Lossless multimodal tool result roundtrip for OpenAI Chat** (Issue [#92](https://github.com/Oaklight/llm-rosetta/issues/92), PR [#108](https://github.com/Oaklight/llm-rosetta/pull/108)): OpenAI Chat Completions only accepts `content: string` for tool messages. Implements a dual encoding strategy — tool message keeps `json.dumps(result)` as data fallback, plus a synthetic user message carries visual content (`image_url` parts) wrapped in `<tool-content call-id="...">` XML tags. Unpacking recovers multimodal structure from the synthetic message (preferred) or falls back to JSON parsing if the synthetic message was trimmed by agent frameworks
- **`extract_all_text()` helper function** (PR [#109](https://github.com/Oaklight/llm-rosetta/pull/109)): Extracts text from both `TextPart` and `ReasoningPart` content — useful for thinking models (e.g. gemini-2.5-flash) that may place answers in reasoning parts rather than text parts
- **`generate_chart` example tool** (PR [#109](https://github.com/Oaklight/llm-rosetta/pull/109)): New multimodal tool in `examples/tools.py` returning `[TextPart, ImagePart]` with inline base64 PNG, plus `multimodal_tools_spec` combining all 3 example tools
- **Multimodal integration tests across all 4 provider SDKs** (PR [#109](https://github.com/Oaklight/llm-rosetta/pull/109)): Two new test scenarios per provider — (A) tool returning multimodal content (text + image), (B) image input combined with tool calls. All 30 tests pass against official APIs: OpenAI Chat 9/9, OpenAI Responses 6/6, Anthropic 8/8, Google GenAI 7/7
- **Runtime IR validation via vendored zero-dependency validator** (Issue [#91](https://github.com/Oaklight/llm-rosetta/issues/91)): `validate_ir_request()`, `validate_ir_response()`, and `validate_ir_messages()` utilities validate IR structures against their TypedDict definitions at runtime. All 4 converters now validate output in `request_from_provider()` and `response_from_provider()`. Replaces manual `BaseMessageOps.validate_messages`. Includes Python <3.11 compatibility for `typing_extensions.TypedDict`
- **Constants validation tests**: 39 new tests across 4 `test_constants.py` files verifying that all reason mapping values are valid IR finish reasons, mapping coverage is complete, event type constants are well-formed, and ID generation produces correct formats
- **Finish reason mapping test coverage**: 38 tests validating reason mapping correctness as a safety net for the constants refactoring
- **`ConversionContext` base class for conversion pipelines** (Issue [#106](https://github.com/Oaklight/llm-rosetta/issues/106), PR [#111](https://github.com/Oaklight/llm-rosetta/pull/111)): New `ConversionContext` dataclass with `warnings: list[str]`, `options: dict[str, Any]`, and `metadata: dict[str, Any]` — a structured context container for non-streaming conversions. New `BaseConverter.create_conversion_context(**options)` factory method mirrors the existing `create_stream_context()`. All 6 non-streaming `BaseConverter` methods now accept an optional `context: ConversionContext` keyword parameter; converter implementations sync warnings to `context.warnings`. Gateway proxy creates a shared context per request and passes it through the full source→IR→target→response pipeline

### Fixed

- **Contextual error messages for tool conversion failures** (Issue [#85](https://github.com/Oaklight/llm-rosetta/issues/85), PR [#110](https://github.com/Oaklight/llm-rosetta/pull/110)): When `p_tool_definition_to_ir()` fails on a malformed or unsupported tool definition, the `ValueError` now includes `type=` and `name=` context so users can identify which tool caused the issue. Applied to all 4 converters (OpenAI Chat, OpenAI Responses, Anthropic, Google GenAI) with unit tests
- **OpenAI Responses `tool_choice` format** (PR [#109](https://github.com/Oaklight/llm-rosetta/pull/109)): Was using Chat Completions format (`{"type": "function", "function": {"name": "..."}}`); now uses Responses format (`{"type": "function", "name": "..."}`)
- **OpenAI Responses tool call ID round-trip** (PR [#109](https://github.com/Oaklight/llm-rosetta/pull/109)): Responses API uses `fc_` prefix IDs while IR uses `call_` prefix. The Responses `id` is now preserved in `provider_metadata` separately from `call_id`, enabling lossless round-trip conversion
- **OpenAI Responses reasoning item round-trip** (PR [#109](https://github.com/Oaklight/llm-rosetta/pull/109)): Reasoning models (e.g. gpt-5-nano) emit reasoning items with `id` (rs_ prefix), structured `summary` arrays, and `encrypted_content`. These are now preserved through `provider_metadata` for lossless round-trip — fixes 400 errors when reasoning items were sent back without their original `id`
- **IR validation accepts `None` for optional response fields** (PR [#109](https://github.com/Oaklight/llm-rosetta/pull/109)): `logprobs` and `system_fingerprint` in `IRResponse` now accept `None` values (previously only accepted missing keys)
- **OpenAI Responses `content_filter` finish reason mapped to wrong status** (Issue [#90](https://github.com/Oaklight/llm-rosetta/issues/90)): `content_filter` was incorrectly mapped to `"completed"` status in `response_to_provider` and `stream_response_to_provider`. Now correctly maps to `"incomplete"` status with `incomplete_details.reason = "content_filter"`
- **Anthropic streaming missing `refusal` reason mapping**: The streaming `reason_map` was missing the `refusal` entry present in the non-streaming path, causing Anthropic refusal stop reasons to be silently dropped during streaming. Fixed as a side effect of the constants extraction (Issue [#64](https://github.com/Oaklight/llm-rosetta/issues/64)) — both paths now share the same `ANTHROPIC_REASON_FROM_PROVIDER` dict

### Changed

- **`ReasoningConfig.effort` expanded to 5-level enum** (Issue [#100](https://github.com/Oaklight/llm-rosetta/issues/100)): Effort levels now include `"minimal"`, `"low"`, `"medium"`, `"high"`, `"max"`. Provider-specific mappings: Anthropic maps to `thinking.type="adaptive"` with `thinking.effort`; OpenAI Chat/Responses clamp `"minimal"`→`"low"` and `"max"`→`"high"` (with warnings); Google GenAI maps to `thinking_config.thinking_level`
- **`ReasoningConfig.type` replaced with `ReasoningConfig.enabled`** (Issue [#70](https://github.com/Oaklight/llm-rosetta/issues/70)): The `type: Literal["enabled", "disabled"]` field is replaced with `enabled: bool` to avoid shadowing the Python built-in `type` and provide a more natural API
- **Merged duplicate IR concepts** (Issue [#69](https://github.com/Oaklight/llm-rosetta/issues/69)): Removed `candidate_count` from `GenerationConfig` — use `n` instead (Google GenAI converter maps `n` ↔ `candidate_count` internally). Unified `system_instruction` type from `str | list[dict]` to `str`
- **Normalized `ImagePart`, `FilePart`, `AudioPart` to canonical forms** (Issue [#68](https://github.com/Oaklight/llm-rosetta/issues/68)): Each part now has exactly two canonical forms — URL reference + structured inline data (e.g. `image_data`) — plus a unified `provider_ref: dict[str, Any]` for provider-specific references. Removed redundant top-level `data`/`media_type` fields and replaced `file_id`/`audio_id` with `provider_ref`
- **IR type fields changed from `Iterable` to `list`; function parameters to `Sequence`** (Issue [#67](https://github.com/Oaklight/llm-rosetta/issues/67)): TypedDict fields now use `list` for indexable, serialization-friendly semantics; function parameters use `Sequence` (covariant, read-only). Also fixes a latent generator-consumption bug in `strip_orphaned_tool_config`
- **`StreamContext` now inherits from `ConversionContext`** (Issue [#106](https://github.com/Oaklight/llm-rosetta/issues/106), PR [#111](https://github.com/Oaklight/llm-rosetta/pull/111)): `StreamContext` is a subclass of `ConversionContext` (IS-A relationship), unifying the context model for streaming and non-streaming paths. File renamed: `base/stream_context.py` → `base/context.py`
- **`StreamContext` converted to dataclass with provider subclass** (Issue [#65](https://github.com/Oaklight/llm-rosetta/issues/65)): `StreamContext` is now a `@dataclass` with typed fields (eliminates defensive `getattr`/`hasattr` patterns). OpenAI Responses-specific state extracted into `OpenAIResponsesStreamContext` subclass. New `BaseConverter.create_stream_context()` factory method

### Refactored

- **Warnings single-source convergence** (Issue [#113](https://github.com/Oaklight/llm-rosetta/issues/113), PR [#115](https://github.com/Oaklight/llm-rosetta/pull/115)): All 4 converter `request_to_provider` methods now use `ConversionContext` as the single accumulation point for warnings. Eliminates the dual-write pattern where warnings were written to both a local list and `context.warnings`. The returned warnings list IS the same object as `context.warnings` — no duplication possible
- **`ProviderMetadataStore` replaces global metadata cache** (Issue [#112](https://github.com/Oaklight/llm-rosetta/issues/112), PR [#117](https://github.com/Oaklight/llm-rosetta/pull/117)): The module-level `_provider_metadata_cache` dict in `proxy.py` is replaced with `ProviderMetadataStore` — a class with TTL-based expiration (30 min), max-size eviction (10k entries), and explicit lifecycle management. The store is created per-app in `create_app()` and passed via `app.state`, eliminating implicit global mutation. `close_clients()` renamed to `close_resources()` to also clear the store on shutdown
- **Shrink public API export surface** (Issue [#114](https://github.com/Oaklight/llm-rosetta/issues/114), PR [#116](https://github.com/Oaklight/llm-rosetta/pull/116)): Reduced `__all__` exports across converter packages to only the primary converter class, removing internal implementation details (`*MessageOps`, `*ContentOps`, `*ConfigOps`, `*ToolOps`, `*Constants`) from the public API. Internal modules remain importable for advanced use but are no longer promoted as public surface
- **Extracted stream event handlers from monolithic methods** (Issue [#63](https://github.com/Oaklight/llm-rosetta/issues/63)): Replaced 8 monolithic `if`/`elif` stream methods (~1,781 lines) across all 4 converters with individual handler methods dispatched via class-level handler tables. Public API unchanged
- **Extracted shared utility functions in OpenAI Responses converter** (Issue [#66](https://github.com/Oaklight/llm-rosetta/issues/66)): `resolve_call_id()` and `build_message_preamble_events()` extracted from `converter.py` into `utils.py` with dedicated unit tests
- **Extracted per-provider constants for reason mappings and magic values** (Issue [#64](https://github.com/Oaklight/llm-rosetta/issues/64)): Inline reason mapping dicts, SSE event type string literals, status-to-reason conditional logic, and ID generation patterns across all 4 converters are now centralized in per-provider `_constants.py` modules. Includes `AnthropicEventType` and `ResponsesEventType` classes, `REASON_FROM_PROVIDER` / `REASON_TO_PROVIDER` dicts, and `generate_tool_call_id()` / `generate_message_id()` helpers

## v0.2.6 — 2026-03-29

### Fixed

- **Chat Completions tool message ordering after Responses API conversion** *([@caidao22](https://github.com/caidao22))*: Codex CLI interleaves `function_call_output` with other items (e.g. user warnings) in Responses API format — valid there since items match by `call_id`. But after IR → Chat Completions conversion, the interleaved messages break the OpenAI Chat API constraint that `role: "tool"` messages must immediately follow their `assistant` `tool_calls`, causing upstream 400 errors. Added `_reorder_tool_messages()` post-processing in `OpenAIChatMessageOps.ir_messages_to_p()` that groups tool responses back to their corresponding assistant messages
- **Orphaned `tool_choice`/`tool_config` stripped when no tools defined** *([@caidao22](https://github.com/caidao22))*: Codex context compaction can drop all tool definitions while keeping `tool_choice` (e.g. `"auto"`), causing upstream APIs to reject with *"tool_choice is set but no tools are provided"*. Added `strip_orphaned_tool_config()` in all four converters — part of the same Codex compaction fix family as `fix_orphaned_tool_calls_ir` (orphaned tool_call/result pairing) and `_reorder_tool_messages` (tool message ordering). Also extended `fix_orphaned_tool_calls_ir` to Google GenAI converter for completeness (Issue [#87](https://github.com/Oaklight/llm-rosetta/issues/87))
- **Stream event ordering**: `UsageEvent` is now emitted before `FinishEvent` in all four provider converters (OpenAI Chat, OpenAI Responses, Anthropic, Google GenAI). Previously `FinishEvent` was processed first, causing `response.completed` to carry `output_tokens=0` — downstream consumers (e.g. Codex token tracking) saw stale usage data. For cross-chunk scenarios (OpenAI Chat sends `finish_reason` and `usage` in separate chunks), `FinishEvent` now defers `response.completed` to `StreamEndEvent` which merges any pending usage
- **Parallel tool calls merged into one in Anthropic/Google → Chat streaming**: Anthropic and Google GenAI `stream_response_from_provider` emitted `ToolCallStartEvent` and `ToolCallDeltaEvent` without `tool_call_index`. When routing to Chat Completions, all parallel tool calls defaulted to index 0, causing the client SDK to merge them into a single call. Anthropic now derives `tool_call_index` from `context._tool_call_order` position; Google computes it from registration order in context (#88, #89)
- **Missing `id` field on Responses `function_call` output**: Non-streaming `response_to_provider` was missing the `id` field on `function_call` output items. Streaming used a synthetic `fc_` prefix that could leak into IR via `p_tool_call_to_ir` fallback path. Unified both paths to use `call_id` directly as `id` (no prefix)
- **Responses streaming `item_id` and empty `tool_call_id` resolution** *([@caidao22](https://github.com/caidao22))*: Added `item_id` tracking to `StreamContext` (`tool_call_item_id_map`, bidirectional mapping). Responses `stream_response_to_provider` now emits `item.id` on `output_item.added` and `item_id` (not `call_id`) on `function_call_arguments.delta/done` events. Defense-in-depth: resolves empty `tool_call_id` by `tool_call_index` via context (Issue [#86](https://github.com/Oaklight/llm-rosetta/issues/86))
- **Non-function tool names mangled with type prefix** *([@caidao22](https://github.com/caidao22))*: Non-function IR tool definitions (e.g. `type="custom"`, `name="apply_patch"`) were converted with a type prefix (`custom_apply_patch`), breaking tool_call matching since the client expects the original name. Both OpenAI Chat and Responses converters now use `ir_tool["name"]` directly (Issue [#84](https://github.com/Oaklight/llm-rosetta/issues/84))

## v0.2.5 — 2026-03-23

### Fixed

- **Anthropic `input_schema` missing `type` for parameterless tools**: MCP tools with no parameters produce `input_schema: {}`, but Anthropic requires `"type"` to be present. Now defaults to `{"type": "object"}` when the schema dict lacks a `type` field — fixes `tools.0.custom.input_schema.type: Field required` errors when routing Google GenAI or OpenAI Responses tool calls to Anthropic upstream
- **Google GenAI camelCase field handling across the full converter stack**: Gemini CLI and the Google REST API use camelCase (`inlineData`, `fileData`, `mimeType`, `fileUri`, `functionCall`, `functionResponse`, `finishReason`, `usageMetadata`, `responseMimeType`, `responseSchema`, `thinkingConfig`, `maxOutputTokens`, `stopSequences`, etc.), but the converter only accepted snake_case. All P→IR methods in content_ops, config_ops, tool_ops, message_ops, and converter now accept both conventions; all IR→P methods now output camelCase for REST API compatibility
- **Image/audio/file data lost during Google→IR conversion**: `p_part_to_ir` checked for `inline_data` (snake_case) but Gemini CLI sends `inlineData` (camelCase) — binary content was silently dropped with a `不支持的Part类型` warning. Fixed by normalizing camelCase keys at the dispatch entry point
- **Cross-format image conversion failure (Google → OpenAI/Anthropic)**: Google's `p_image_to_ir` produces `ImagePart` with top-level `data` + `media_type` fields, but OpenAI Chat, Anthropic, and OpenAI Responses `ir_image_to_p` only checked `image_url` and nested `image_data` — threw `ValueError`. All three target converters now handle top-level fields as a fallback path (Issue [#68](https://github.com/Oaklight/llm-rosetta/issues/68))
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

- **Anthropic→IR role normalization for `tool_result` messages**: Anthropic places `tool_result` blocks in `role: "user"` messages, but IR uses `role: "tool"` (like OpenAI). The Anthropic converter now normalizes pure `tool_result` user messages to `role: "tool"`, and splits mixed `tool_result` + text messages into separate `role: "tool"` and `role: "user"` IR messages. This fixes `fix_orphaned_tool_calls_ir()` failing to detect answered tool calls in cross-format conversions (e.g. Anthropic → OpenAI Chat) (Issue [#84](https://github.com/Oaklight/llm-rosetta/issues/84))
- **OpenAI Responses→IR role normalization for `function_call_output` items**: `function_call_output` and `mcp_call_output` items were grouped into `role: "user"` IR messages, but IR uses `role: "tool"` for tool results. The Responses converter now groups these items into `role: "tool"` messages, fixing `fix_orphaned_tool_calls_ir()` failing to detect answered tool calls when converting Responses → other formats (e.g. Responses → OpenAI Chat) (Issue [#84](https://github.com/Oaklight/llm-rosetta/issues/84))

### Added (Documentation)

- **Provider Dialect Differences guide**: New section in the Converters guide (EN + ZH) documenting tool schema sanitization, orphaned tool call handling, and Google camelCase/snake_case differences

## v0.2.3 — 2026-03-22

### Fixed

- **Tool schema sanitization applied to all converters**: `_sanitize_schema()` was previously only called in the OpenAI Chat converter. Google GenAI, OpenAI Responses, and Anthropic converters now also sanitize tool parameter schemas before sending to upstream, preventing rejections from strict endpoints like Vertex AI (Issue [#80](https://github.com/Oaklight/llm-rosetta/issues/80))
- **Non-standard `ref` and `$schema` keywords stripped**: OpenCode's built-in tools use a bare `ref` field (without `$` prefix) and `$schema` at the top level, both rejected by Vertex AI. Added to the unsupported keywords blocklist (Issue [#80](https://github.com/Oaklight/llm-rosetta/issues/80))
- **`$ref`/`$defs` resolved by inlining**: JSON Schema `$ref` references are now resolved by inlining the referenced definition from `$defs`/`definitions`, and both keys are removed from the output. Supports nested and chained references (Issue [#80](https://github.com/Oaklight/llm-rosetta/issues/80))
- **Streaming tool call arguments not accumulated**: OpenAI Chat, Anthropic, and Google GenAI converters registered tool calls in `StreamContext` but never called `append_tool_call_args()` to accumulate argument deltas during streaming. This caused tool call arguments to arrive empty at upstream (e.g., MCP tools returning `'query' is a required property`). Only the OpenAI Responses converter was correct (Issue [#81](https://github.com/Oaklight/llm-rosetta/issues/81))
- **OpenAI Chat streaming tool call ID resolution**: Delta-only chunks (carrying `index` but no `id`) produced an empty-string `tool_call_id`. Now resolves the effective ID from `StreamContext._tool_call_order` using the chunk index (Issue [#81](https://github.com/Oaklight/llm-rosetta/issues/81))

### Changed

- **`sanitize_schema` extracted to `converters/base/tools.py`**: The schema sanitization utility (previously `_sanitize_schema` private to `openai_chat/tool_ops.py`) is now a public shared function in `converters/base/tools.py`, exported via `converters.base`. All 4 converter `tool_ops.py` files import from the shared location instead of cross-importing from `openai_chat` (Issue [#66](https://github.com/Oaklight/llm-rosetta/issues/66))

## v0.2.2 — 2026-03-22

### Fixed

- **Missing `content_block_stop` in Anthropic SSE output**: When converting OpenAI Chat streaming responses to Anthropic SSE format, `content_block_stop` events were not emitted before `message_delta`, causing Claude Code to silently discard response content. The Anthropic converter now emits `content_block_stop` for any open content block when processing a `FinishEvent` (Issue [#77](https://github.com/Oaklight/llm-rosetta/issues/77))
- **Upstream preflight chunk misinterpreted as stream end**: Argo API sends a preflight chunk with `choices: []` and empty `id`/`model` before actual content. The OpenAI Chat converter now only treats empty-choices chunks as stream-end after the stream has actually started (`context.is_started` guard) (Issue [#77](https://github.com/Oaklight/llm-rosetta/issues/77))

## v0.2.1 — 2026-03-20

### Added

- **Gateway request/response body logging**: configurable debug logging with colorized output, body sanitization and truncation — enable via config (`"debug": {"verbose": true, "log_bodies": true}`), env vars (`LLM_ROSETTA_VERBOSE`, `LLM_ROSETTA_LOG_BODIES`), or `--verbose` CLI flag
- **Google `output_format="rest"` for `request_to_provider()`**: pass `output_format="rest"` to get a REST API–ready request body with `tools`/`tool_config` at top level and generation params wrapped in `generationConfig` — eliminates the need for manual SDK→REST fixups

### Changed

- **Gateway modularization**: split `app.py` (1057 lines) into `proxy.py` (proxy engine, SSE handling, upstream requests), `cli.py` (CLI entry point, argparse, subcommands), and a slimmed `app.py` (route handlers, app factory, ~210 lines)
- **Moved Google REST body fixup to core**: `_fixup_google_body()` logic moved from `gateway/proxy.py` into `GoogleGenAIConverter._to_rest_body()`, removing duplicated SDK→REST transforms from the gateway and all 6 REST examples

### Fixed

- OpenAI Responses streaming: added missing `id`/`object`/`model` fields to `response.completed`, `output_index`/`content_index` to text delta events, and proper lifecycle events (`output_item.added`, `content_part.added`, `content_part.done`, `output_item.done`) (Issue [#56](https://github.com/Oaklight/llm-rosetta/issues/56))
- OpenAI Chat streaming: `tool_calls` entries now always include the required `index` field, defaulting to `0` when not explicitly provided by the upstream IR event (Issue [#57](https://github.com/Oaklight/llm-rosetta/issues/57))
- OpenAI Chat streaming: usage-only chunk now includes `"choices": []` to satisfy clients that validate every `chat.completion.chunk` must contain a `choices` array (Issue [#55](https://github.com/Oaklight/llm-rosetta/issues/55))
- `stream_options` (Chat Completions-only field) no longer leaks into OpenAI Responses API requests — the Responses converter's `ir_stream_config_to_p()` was incorrectly emitting `stream_options`, causing upstream rejection when Chat-format clients (Kilo, OpenCode) were proxied to the Responses API (Issue [#58](https://github.com/Oaklight/llm-rosetta/issues/58))
- Google GenAI converter now handles tools and tool_config in REST-format requests (top-level fields) in addition to SDK format (`config.tools`) — previously only SDK format was recognized, silently stripping tool definitions from gateway-proxied requests (Issue [#59](https://github.com/Oaklight/llm-rosetta/issues/59))
- Google camelCase `functionDeclarations` not parsed: `p_tool_definition_to_ir()` now handles both `functionDeclarations` (camelCase/REST) and `function_declarations` (snake_case/SDK), and extracts all declarations instead of only the first. Also added camelCase support for `functionCallingConfig`/`allowedFunctionNames` and `toolConfig` in request parsing — fixes Gemini CLI tool calling through the gateway (Issue [#61](https://github.com/Oaklight/llm-rosetta/issues/61))
- Google streaming tool calls split into two chunks: `stream_response_to_provider()` now defers `tool_call_start` and emits the complete `function_call` (name + args) in a single chunk on `tool_call_delta`, matching the Google API's native format (Issue [#62](https://github.com/Oaklight/llm-rosetta/issues/62))

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
- **Model listing endpoints**: `GET /v1/models` (compatible with both OpenAI and Anthropic SDKs) and `GET /v1beta/models` (Google GenAI SDK format) — enables `client.models.list()` across all three SDKs (Issue [#54](https://github.com/Oaklight/llm-rosetta/issues/54))

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
- Updated `git clone` URL from `llmir` to `llm-rosetta` in documentation
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
- Google `thought_signature` preservation through gateway round-trips — newer Google models require `thoughtSignature` echoed back in function call parts; the gateway now caches `provider_metadata` (including `thought_signature`) keyed by `tool_call_id` and re-injects it on subsequent requests for both streaming and non-streaming modes (Issue [#51](https://github.com/Oaklight/llm-rosetta/issues/51))
- OpenAI Responses converter now handles all 3 `input` formats: bare string (`"input": "hello"`), shorthand list (`[{"role": "user", "content": "hi"}]`), and structured list — previously only the structured format was supported, causing the OpenAI Python SDK's shorthand items to be silently dropped and producing empty IR messages when cross-converting to Anthropic or Google providers

---

## 2026-03-15 — Rebrand to LLM-Rosetta

### Changed

- **Project renamed from LLMIR to LLM-Rosetta** across all code, docs, and configuration
- Python import package name set to `llm_rosetta` (underscore); `pyproject.toml` updated accordingly
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
