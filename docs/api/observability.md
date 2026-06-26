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
| `provider_health_snapshot()` | 按提供商的健康状态 |
| `any_critical_provider()` | 任一提供商处于严重不健康状态时返回 True |

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
| `target_provider_name` | `str \| None` | 提供商显示名称 |
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
