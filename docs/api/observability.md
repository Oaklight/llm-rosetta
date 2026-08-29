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
    source="openai_chat",
    target="anthropic",
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
    source_provider="openai_chat",
    target_provider="anthropic",
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
    source_provider="openai_chat",
    target_provider="anthropic",
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
| `source_provider` | `str \| None` | 源 API 格式（如 `"openai_chat"`） |
| `target_provider` | `str \| None` | 目标 API 格式（如 `"anthropic"`） |
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
