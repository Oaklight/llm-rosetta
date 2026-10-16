---
title: 可观测性
---

# 可观测性 API 参考

`llm_rosetta.observability` 包提供了可复用的、与框架无关的构建模块，用于指标
收集、请求日志记录、SQLite 持久化和按需性能分析。任何基于 `llm-rosetta` 构建
的 HTTP 代理都可以直接导入此包——无需依赖网关的配置系统或 HTTP 服务器。

```python
from llm_rosetta.observability import (
    MetricsCollector,
    PersistenceManager,
    ProfilerState,
    RequestLog,
    RequestLogEntry,
    dump_error,
    offload_images,
    compute_body_hash,
    compress_body,
    decompress_body,
    OpsLog,
    OpsLogEntry,
)
```

---

## MetricsCollector

轻量级进程内指标收集器。所有数据结构均为纯 Python 对象——无框架依赖。
专为单线程 asyncio 事件循环设计（无需锁）。

```python
metrics = MetricsCollector()
metrics.record_request(
    model="gpt-4o",
    source="chat_completions",
    target="messages",
    status_code=200,
    duration_ms=150.0,
    is_stream=False,
    provider_name="My Anthropic",
)
snapshot = metrics.snapshot(series_seconds=60)
```

### 核心方法

| 方法 | 描述 |
|------|------|
| `record_request(...)` | 记录已完成的代理请求 |
| `snapshot(series_seconds)` | 返回可 JSON 序列化的指标快照 |
| `export_counters()` | 返回用于持久化的计数器（不含时间序列） |
| `load_counters(data)` | 从导出的字典恢复计数器 |
| `rebuild_counters(rows)` | 从请求日志行重建所有计数器 |
| `provider_health_snapshot()` | 按提供方的健康状态 |
| `any_critical_provider()` | 任一提供方处于严重不健康状态时返回 True |

---

## RequestLogEntry

表示单条已记录代理请求的冻结数据类。

```python
entry = RequestLogEntry.create(
    model="gpt-4o",
    source_provider="chat_completions",
    target_provider="messages",
    is_stream=False,
    status_code=200,
    duration_ms=123.4,
    target_provider_name="My Anthropic",
)
```

### 字段

| 字段 | 类型 | 描述 |
|------|------|------|
| `id` | `str` | 自动生成的 UUID hex |
| `timestamp` | `str` | ISO 8601 时间戳 |
| `model` | `str` | 模型名称 |
| `source_provider` | `str` | 源 API 格式 |
| `target_provider` | `str` | 目标 API 格式 |
| `is_stream` | `bool` | 是否使用流式传输 |
| `status_code` | `int` | HTTP 状态码 |
| `duration_ms` | `float` | 请求耗时（毫秒） |
| `error_detail` | `str \| None` | 错误信息（如有） |
| `api_key_label` | `str \| None` | API 密钥标签 |
| `target_provider_name` | `str \| None` | 提供方显示名称 |
| `client_ip` | `str \| None` | 客户端 IP 地址 |
| `profile` | `dict \| None` | 性能分析数据 |

---

## RequestLog

支持可选 SQLite 持久化的代理请求日志。当提供 `PersistenceManager` 时委托给
SQLite，否则回退到内存环形缓冲区。

```python
# 仅内存
log = RequestLog(max_entries=500)

# 带持久化
persistence = PersistenceManager("/var/data/myproxy")
log = RequestLog(persistence=persistence)

log.add(entry)
entries, total = log.get_entries(limit=50, status="error")
```

### 核心方法

| 方法 | 描述 |
|------|------|
| `add(entry)` | 记录请求日志条目 |
| `get_entries(...)` | 分页、过滤查询（最新优先） |
| `get_entry(entry_id)` | 按 ID 查询单条 |
| `get_api_key_labels()` | 去重的 API 密钥标签 |
| `update_profile(entry_id, data)` | 合并性能分析数据到现有条目 |
| `clear()` | 删除所有条目 |

---

## PersistenceManager

基于 SQLite 的请求日志和指标计数器持久化。使用 WAL 日志模式，支持双阈值
保留策略（成功/错误独立修剪）。

```python
pm = PersistenceManager(
    data_dir="/var/data/myproxy",
    success_max=50000,
    error_max=10000,
)
```

### 核心方法

| 方法 | 描述 |
|------|------|
| `insert_log_entries(entries)` | 批量插入请求日志条目 |
| `query_log_entries(...)` | 带分页的过滤查询 |
| `save_metrics(data)` | 持久化指标计数器 |
| `load_metrics()` | 加载已持久化的指标 |
| `count_log_entries()` | 总条目数 |
| `count_success_entries()` | 成功条目数（status < 400） |
| `count_error_entries()` | 错误条目数（status ≥ 400） |
| `db_file_sizes()` | 磁盘文件大小 |
| `insert_dump_body(body_hash, data, orig_bytes)` | 插入压缩的 body 数据，按哈希去重 |
| `insert_error_dump(dump_id, ...)` | 插入错误转储记录并在超出容量时修剪 |
| `query_error_dumps(limit, offset, ...)` | 过滤、分页查询错误转储（最新优先） |
| `get_error_dump(dump_id)` | 按 ID 返回单条错误转储 |
| `get_dump_body(body_hash)` | 按哈希返回压缩的 body 数据 |
| `count_error_dumps()` | 错误转储总条目数 |
| `clear_error_dumps()` | 删除所有错误转储及孤立的 body 数据 |
| `close()` | 提交并关闭数据库 |

### 保留默认值

| 常量 | 默认值 | 描述 |
|------|--------|------|
| `DEFAULT_SUCCESS_MAX` | 50,000 | 最大成功条目数 |
| `DEFAULT_ERROR_MAX` | 10,000 | 最大错误条目数 |

---

## ProfilerState

管理按需的请求级 pyinstrument 性能分析会话。与框架无关的数据层——将
`ProfilerState` 接入 Web 框架的路由处理器由消费者实现。

```python
state = ProfilerState(max_results=20)
state.enable(requests=5)

if state.should_profile():
    profiler = state.create_profiler()
    profiler.start()
    # ... 执行工作 ...
    profiler.stop()
    state.store_result(profiler, model="gpt-4o", duration_ms=150.0)
```

### 核心方法

| 方法 | 描述 |
|------|------|
| `enable(requests)` | 为接下来的 N 个请求启用分析 |
| `disable()` | 手动禁用 |
| `should_profile()` | 检查并消费一个分析配额 |
| `create_profiler()` | 创建 `DeepProfiler` 实例 |
| `store_result(profiler, ...)` | 存储分析结果 |
| `status()` | 当前分析状态字典 |
| `clear_results()` | 移除所有已存储的结果 |

---

## 操作日志 (OpsLog)

操作日志记录服务器级别的操作事件——数据变更、安全事件、密钥管理、
启动/关闭——并附带结构化元数据。有 SQLite 持久化时委托给持久化层，
否则回退到内存环形缓冲区。

```python
from llm_rosetta.observability import OpsLog, OpsLogEntry
```

### OpsLogEntry 字段

| 字段 | 类型 | 描述 |
|------|------|------|
| `id` | `str` | 自动生成的 UUID hex |
| `timestamp` | `str` | ISO 8601 时间戳 |
| `event_type` | `str` | 事件类别（见下表） |
| `severity` | `str` | `"info"`、`"warning"` 或 `"error"` |
| `message` | `str` | 人类可读摘要 |
| `details` | `dict \| None` | 结构化元数据（因事件类型而异） |
| `source` | `str \| None` | 来源子系统 |

### 事件类型

| 常量 | 字符串 | 描述 |
|------|--------|------|
| `EVENT_STARTUP` | `startup` | 服务器启动 |
| `EVENT_SHUTDOWN` | `shutdown` | 优雅停止 |
| `EVENT_CONFIG_RELOAD` | `config_reload` | 配置重载 |
| `EVENT_ADMIN_SETUP` | `admin_setup` | 管理面板初始化 |
| `EVENT_KEY_CREATE` | `key_create` | API 密钥创建 |
| `EVENT_KEY_UPDATE` | `key_update` | API 密钥更新 |
| `EVENT_KEY_DELETE` | `key_delete` | API 密钥删除 |
| `EVENT_KEY_ROTATE` | `key_rotate` | API 密钥轮换 |
| `EVENT_HEALTH_CHANGE` | `health_status_change` | 提供方健康状态变化 |
| `EVENT_DATA_CLEARED` | `data_cleared` | 表数据清除 |
| `EVENT_DATA_CLEANUP` | `data_cleanup` | 基于时间的清理 |
| `EVENT_DATA_TRIMMED` | `data_trimmed` | 裁剪至保留上限 |
| `EVENT_DATA_VACUUMED` | `data_vacuumed` | 数据库 vacuum |
| `EVENT_DATA_REBUILT` | `data_rebuilt` | 从日志重建指标 |
| `EVENT_OPS_LOG_CLEARED` | `ops_log_cleared` | 操作日志已清除 |
| `EVENT_PASSWORD_CHANGED` | `password_changed` | 管理密码已更改 |
| `EVENT_TOKEN_ROTATED` | `token_rotated` | 内部代理令牌已轮换 |
| `EVENT_SESSION_LOGOUT_ALL` | `session_logout_all` | 所有管理会话已失效 |

### 来源

| 常量 | 字符串 | 使用者 |
|------|--------|--------|
| `SOURCE_GATEWAY` | `gateway` | 启动、关闭 |
| `SOURCE_ADMIN` | `admin` | 管理面板操作、会话注销 |
| `SOURCE_KEYS` | `keys` | API 密钥管理 |
| `SOURCE_CONFIG` | `config` | 配置重载 |
| `SOURCE_PERSISTENCE` | `persistence` | 后台修剪、定期清理 |
| `SOURCE_AUTH` | `auth` | 密码更改、令牌轮换 |

### 严重性级别

| 常量 | 字符串 | 用途 |
|------|--------|------|
| `SEVERITY_INFO` | `info` | 非破坏性操作（默认） |
| `SEVERITY_WARNING` | `warning` | 破坏性数据变更、安全事件 |
| `SEVERITY_ERROR` | `error` | 失败（`_run()` 抛出异常时自动设置） |

### OpsLog 方法

| 方法 | 描述 |
|------|------|
| `add(entry)` | 记录一个操作事件 |
| `get_entries(...)` | 分页、过滤查询（最新在前） |
| `clear()` | 删除所有条目，返回数量 |

### ops_log 与 request_log 对比

| 方面 | ops_log | request_log |
|------|---------|-------------|
| 记录内容 | 变更、安全事件、生命周期 | 代理请求流量 |
| 体量 | 低（管理操作、后台任务） | 高（每个代理请求） |
| 写入者 | `OpsBase._record()`（默认） | `OpsProxyRequest._record()`（覆写） |
| 典型事件 | `data_cleared`、`key_create`、`password_changed` | 每个 `/v1/*` 代理请求 |

### OpsContext

`OpsContext` 是连接 ops 与可观测性子系统的轻量级服务包。
在应用初始化时由 `setup_admin()` 创建一次：

```python
from llm_rosetta.gateway.ops import OpsContext

ctx = OpsContext(
    ops_log=ops_log,
    request_log=request_log,
    metrics=metrics,
    persistence=persistence,  # 均为可选——None 表示禁用该子系统
)
app.ops_ctx = ctx
```

---

## 错误转储 (Error Dump)

当上游或转换错误发生时，网关可以捕获完整的请求上下文，供后续重放或调试使用。
`error_dump` 模块提供了一组辅助函数，用于卸载图片、计算哈希、压缩并通过
`PersistenceManager` 存储转储数据。

所有公开函数均为即发即忘安全（fire-and-forget safe）——内部捕获并记录异常，
调用方无需 `try/except` 包装。

### `dump_error()`

记录错误上下文的主入口。成功时返回转储 ID，持久化禁用或发生异常时返回 `None`。

```python
from llm_rosetta.observability import dump_error

dump_id = dump_error(
    persistence,
    request_body={"model": "gpt-4o", "messages": [...]},
    response_text="Internal Server Error",
    converted_body=converted,
    model="gpt-4o",
    source_provider="chat_completions",
    target_provider="messages",
    provider_name="My Anthropic",
    status_code=500,
    error_phase="upstream",
    upstream_url="https://api.anthropic.com/v1/messages",
    request_log_id=entry.id,
)
```

#### 参数

| 参数 | 类型 | 描述 |
|------|------|------|
| `persistence` | `PersistenceManager \| None` | 持久化管理器（`None` → 空操作） |
| `request_body` | `dict \| None` | 原始请求体字典 |
| `response_text` | `str \| None` | 上游错误响应文本（截断至 64 KB） |
| `converted_body` | `dict \| None` | 转换后的目标格式请求体（如有） |
| `model` | `str \| None` | 请求中的模型名称 |
| `source_provider` | `str \| None` | 源 API 格式（如 `"chat_completions"`） |
| `target_provider` | `str \| None` | 目标 API 格式（如 `"messages"`） |
| `provider_name` | `str \| None` | 人类可读的提供方名称 |
| `status_code` | `int \| None` | 上游 HTTP 状态码 |
| `error_phase` | `str \| None` | `"upstream"`、`"stream_header"`、`"stream_chunk"` 或 `"conversion"` 之一 |
| `upstream_url` | `str \| None` | 被调用的上游 URL |
| `request_log_id` | `str \| None` | 关联到请求日志条目的外键 |

超过 10 MB（`MAX_BODY_BYTES`）的请求体将被跳过——仅存储元数据。

### `offload_images(body)`

将内联的 base64 图片数据 URI 替换为 SHA256 摘要占位符。返回深拷贝——原始
body 不会被修改。

```python
from llm_rosetta.observability import offload_images

cleaned = offload_images(request_body)
# base64 数据 → "[image image/png sha256:abc123… 450KB]"
```

| 参数 | 类型 | 描述 |
|------|------|------|
| `body` | `dict[str, Any]` | 请求/响应体字典 |

**返回值：** 深拷贝，其中所有 `data:image/…;base64,…` 字符串均被替换为人类可读的占位符。

### `compute_body_hash(body)`

对规范化 JSON（键排序、无空白）计算 SHA256 哈希。

```python
from llm_rosetta.observability import compute_body_hash

h = compute_body_hash({"model": "gpt-4o", "messages": []})
# "e3b0c44298fc1c149afb..."
```

| 参数 | 类型 | 描述 |
|------|------|------|
| `body` | `dict[str, Any]` | 要计算哈希的 body 字典 |

**返回值：** 十六进制编码的 SHA256 摘要字符串。

### `compress_body(body)`

将 body 字典进行 JSON 序列化并 zlib 压缩。

```python
from llm_rosetta.observability import compress_body

compressed, original_size = compress_body(request_body)
```

| 参数 | 类型 | 描述 |
|------|------|------|
| `body` | `dict[str, Any]` | 要压缩的 body 字典 |

**返回值：** `(compressed_bytes, original_size)` 元组。

### `decompress_body(data)`

将 zlib 压缩的 JSON body 解压还原为字典。是 `compress_body` 的逆操作。

```python
from llm_rosetta.observability import decompress_body

body = decompress_body(compressed)
```

| 参数 | 类型 | 描述 |
|------|------|------|
| `data` | `bytes` | zlib 压缩的 JSON 字节数据 |

**返回值：** 反序列化后的字典。

---

## 用量（Usage） {: #usage }

`llm_rosetta.observability.usage` 负责界定"我们搬了多少 token、算在谁头上"。
它不涉及 HTTP、配置和凭据：库的使用者不需要这些东西就能记录用量并读取报告；
网关只是它的一个消费者，额外提供凭据主体（subject）和 SQLite 存储。

### 台账的列

| 列 | 含义 |
|--------|------|
| `input_tokens` | 输入（prompt）token，以厂商的计数为准 |
| `output_tokens` | 输出（completion）token |
| `total_tokens` | 规范总量——见下文的 `token_total` |
| `cache_read_tokens` | 缓存命中 |
| `cache_creation_tokens` | 缓存写入 |
| `reasoning_tokens` | 思考 token；**包含在** `output_tokens` 之内，从不额外累加 |

`token_total(usage)` 负责填充 `total_tokens`。厂商给了总量就直接采用；否则把
各计数器相加，其中两条规则并不显然：

- **思考 token 从不累加**——它已经包含在输出 token 里。
- **缓存只在它未被计入输入时才累加。** Messages API 把缓存 token 记在输入
  之*外*（可加），OpenAI 与 Google 记在输入之*内*（是其子集）。谁属于哪一种
  只在 `llm_rosetta.observability.usage.semantics` 里声明一次；一旦某个转换器
  与声明不一致，测试就会失败。

`usage_columns(usage)` 把 IR 的 usage 字典转成上述各列。usage 里没有出现的
计数器会被留空而不是写成 0，这样"没有缓存"和"未报告缓存"仍然可以区分；值不是
整数的字段同样不算计数，会按"未报告"处理——计数以字符串形式到达的 payload
按"没有用量"处理，而不是在算术里抛出异常。

### 模型类型（Model types）

每一行还带一个**模型类型**——`llm`、`embedding`、`rerank` 或 `decision`，
即 `gateway/model_types.py` 注册的那套名字。同一列在不同家族里对应的 IR 字段
并不相同：chat 向厂商要的是 `prompt_tokens` 与 `completion_tokens`，decision
IR 本来就报 `input_tokens`/`output_tokens`，embedding 与 rerank 的 IR 类型
没有输出侧，而 rerank 的缓存计数器叫 `cached_tokens`。因此"列 → 字段"的映射
按模型类型声明，`usage_columns(usage, model_type)` 与
`token_total(usage, model_type)` 读取对应的一份；不属于这四种的名字会被拒绝
而不是被猜。规范总量的规则在各家族之间不变：厂商给了总量就采用，否则累加各
部分；自身不提供总量的家族永远走累加路径。

`UsageReport.by_model_type` 把同一批行再按类型切一次——
`report.for_model_type("embedding", "24h")`——与上面的三类正交；没有给出类型的
分组按 chat 计，因为在**该列出现之前**写入的行就是这个含义。网关的日表
（`GET /admin/api/token-usage`）按天返回同样的拆分，并接受 `model_type=` 过滤。

### UsageLedger

记录用量，然后查询它累计成什么——不需要网关，也不需要数据库：

```python
from llm_rosetta.observability import UsageLedger

ledger = UsageLedger()
ledger.record(resp["usage"], model="gpt-5-nano", subject="batch-7")
ledger.record(resp["usage"], model="gpt-5-nano", subject="batch-7")

ledger.total("24h")
# {'request_count': 2, 'input_tokens': 40, 'output_tokens': 10, ...}
```

| 方法 | 说明 |
|--------|------|
| `record(usage, *, model=None, subject=None, label=None, model_type="llm", at=None, warnings=None)` | 记录一次调用；当 usage 未给出任何 token 计数时返回 `None` |
| `groups()` | 按主体与窗口汇总后的记录——与已存储台账的读取结果形状一致 |
| `report(subjects=...)` | 构建 `UsageReport`；默认把台账见过的每个主体都计入归属 |
| `total(window)` | 某个窗口内的全部 token，无论归属如何 |
| `records()` / `len(ledger)` / `clear()` | 取回记录、计数或丢弃 |

窗口是滚动的（`5h`、`24h`、`7d`、`30d`），并且在生成报告时以当前时钟为准，
因此记录会自行滑出较窄的窗口。`at=` 可以把记录标注在过去某个时刻，这正是
回放日志所需要的。

有两处行为在使用前值得了解：

- **没有给出任何 token 计数的 usage 不会留下记录**，并返回 `None`。响应本身
  就没带 usage 时这是对的，键名拼错时则是错的——而从外部看两者无法区分，因此
  传入 `warnings=[]` 就能知道它看到了哪些键。显式的 0 是计数：报告"没有消耗"
  的调用仍是一次真实调用，会以零 token 记下请求数。值是字符串的计数同样不算
  计数，会按"没有用量"处理，而不是在算术里抛出异常。
- **`report()` 会把台账记录过的每个主体都计入归属**，因为这里没有 keystore
  来判断谁仍然存活。若一本台账由彼此不互相负责的几方共用，应显式传入
  `subjects=`——名字（`subjects=["batch-7"]`）或 `(subject, label)` 二元组。
  网关读同一批行的方式不同，且是有意为之：凭据已被删除的主体在网关那边落入
  余量，因为网关**确实**有 keystore 可以问。

### UsageReport

```python
from llm_rosetta.observability import UsageReport

report = UsageReport.build(
    groups=await persistence.query_usage_groups(),
    subjects=[(bucket, label) for ...],       # 可以归属的主体
    internal_subject=INTERNAL_SUBJECT,        # 网关自身的流量
    internal_label=INTERNAL_LABEL,
)
report.total("24h")
```

每一行都恰好落入三类之一，而 `total()` 会把三类全部相加：

| 类别 | 含义 |
|-------|------|
| 已归属（attributed） | 主体是调用方指定过的主体 |
| 内部（internal） | 调用方自身的流量——在网关中即管理面板所用的令牌 |
| 未归属（unattributed） | 其余一切：凭据已被删除、在主体概念出现之前写入的行、以及未携带任何主体的请求。予以保留，并按标签拆分 |

保留第三类正是报告能够对得上的原因。仪表盘统计每一行，因此一份悄悄丢掉
无法归属行的报告，必然与仪表盘的总量不符；反之两者的数字在构造上就是同一个，
而 API Keys 面板会把剩余部分单独列成一行。

`to_payload(names)` 把报告渲染为管理 API 的 JSON，其中 `names` 把每个主体映射
到它应归档的名称——网关借此把凭据 bucket 映射为 key 的 id，并为其余量改名。

`INTERNAL_SUBJECT` 与 `INTERNAL_LABEL`（均为 `"internal"`）就声明在这套分类旁边，
而不是声明在网关里：写入管理令牌那些行的代码，与任何对它们分组的读取方，从此
只有一个名字。

### `llm-rosetta-usage` 命令 { #the-llm-rosetta-usage-command }

网关在管理面板里渲染这份报告，库通过 `UsageReport.build` 拿到它；第三个读取方
是 shell：

```console
$ llm-rosetta-usage --db /var/lib/llm-rosetta --window 24h
store: /var/lib/llm-rosetta
window: 24h (rolling)

subject             requests  input  output  cache read  cache write  reasoning  total
------------------  --------  -----  ------  ----------  -----------  ---------  -----
team-a                     2   1,200     800         400            0          0  2,000
team-b                     2     600       0           0            0          0    600
internal                   1      10       5           0            0          0     15
(unattributed)             4     140      40           0            0          0    180
TOTAL                     10   1,952     850         400            0          0  2,802
```

| 选项 | 含义 |
|------|------|
| `--db PATH` | 必填。包含 `gateway.db` 的数据目录，或该文件本身 |
| `--window NAME` | 表格展示哪个滚动窗口：`5h`、`24h`（默认）、`7d`、`30d` |
| `--by CUT[,CUT]` | 按什么切分：`subject`（默认）、`type`（模型类型）、`model` |
| `--keys-db PATH` | 用于归属的 keys 数据库——默认取存储旁边的 `keys.db` |
| `--top N` | 每张表展示多少行；`0` 表示全部。表的合计仍为该切分下全部行的合计，且 `--json` 永不截断 |
| `--migrate` | 按网关的方式打开存储：先备份、再迁移。默认关闭 |
| `--json` | 输出 JSON 而不是表格 |

每张表都按**它所展示的那个窗口**排序，并列时由最宽的窗口决定——因此一行的位置
总能由它旁边的数字解释，而流量全落在 `--window` 之外的模型会排在有流量的模型
之后。每张表的最后一行是该切分下**全部**行的合计，而不只是显示出来的那些，所以
`--top` 不会让合计与存储不符。`--top` 作用于三张表，包括主体表——那正是随网关
见过的凭据数量增长的那张。

`--json` 输出的就是这份报告本身，而不是它的第二种渲染：subject 这一路切分会
原样带上管理 API 返回的那些键（`windows`、`keys`、`internal`、`unattributed`），
因此读命令的脚本与读面板的脚本看到的是同一个形状。其余切分会追加
`model_types` 与 `models`，且 `--top` 从不截断它们。

它通过 `PersistenceManager` 打开存储——也就是网关自己的读取方——但**不写入**：
不备份、不建表、不迁移。一次查询不该留下数据库副本，也不该对另一个进程正在
服务的存储加写锁。因此，schema 比当前安装版本旧的存储会被**拒绝**而不是被迁移，
并说明两条出路：先启动一次网关，或传 `--migrate`——后者完全按网关的方式打开存储
（先备份，再迁移）。不存在的存储同样被拒绝而不是被创建：打错路径不能留下一个空
数据库，读起来像是一个没有流量的存储。

你无法写入的存储——挂载卷、快照副本——只要 SQLite 能挂接上它需要的东西就能读取
（有进程正在服务该存储时，边车文件就在那里）。一个只读存储究竟能不能读，取决于
SQLite 关于 `-wal`/`-shm` 的规则，而不取决于存储的新旧；因此命令报告它实际观察到
的事实而不是猜测：写前日志挂接不上的存储**读起来像是空的**，就按这个如实拒绝，
绝不报成"schema 版本 0"——那会让读者去迁移一个本来就已是最新版本的存储。

归属需要仍然存在的凭据，它们位于网关的 keys 数据库，命令以只读方式读取它——
一张表的两列。没有它时，所有 token 都会被报告为未归属，命令会在 stderr 上说明
——并且区分"数据库不存在"与"数据库读不出来"——这是对一个小一点的问题给出的
真实答案：总量仍是存储的总量，缺的只是名字。

`total_tokens` 为 NULL 的行——存在于网关启动时那次修复之前写入的行——会按存储中
的原样报告，因此可能读成 `input 4,000 / total 0`。这与管理面板显示的同一个数字；
修复在网关启动时执行，而本命令刻意不写入。

---

## 向后兼容性

所有类仍可从原始 `gateway.admin` 路径导入：

```python
# 以下导入仍然有效（从 observability 重新导出）
from llm_rosetta.gateway.admin.metrics import MetricsCollector
from llm_rosetta.gateway.admin.request_log import RequestLog, RequestLogEntry
from llm_rosetta.gateway.admin.persistence import PersistenceManager
from llm_rosetta.gateway.admin.routes.profiling import ProfilerState
```

新代码应直接从 `llm_rosetta.observability` 导入。
