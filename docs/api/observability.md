---
title: Observability
---

# Observability API Reference

The `llm_rosetta.observability` package provides reusable, framework-agnostic
building blocks for metrics collection, request logging, SQLite persistence,
and on-demand profiling.  Any HTTP proxy built on top of `llm-rosetta` can
import from this package directly — no dependency on the gateway's config
system or HTTP server.

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

Lightweight in-process metrics.  All data structures are plain Python objects —
no framework dependencies.  Designed for single-threaded asyncio event loops
(no locks required).

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

### Key methods

| Method | Description |
|--------|-------------|
| `record_request(...)` | Record a completed proxy request |
| `snapshot(series_seconds)` | Return a JSON-serializable metrics snapshot |
| `export_counters()` | Return counters for persistence (no time-series) |
| `load_counters(data)` | Restore counters from exported dict |
| `rebuild_counters(rows)` | Rebuild all counters from request log rows |
| `provider_health_snapshot()` | Per-provider health status |
| `any_critical_provider()` | True if any provider is critically unhealthy |

---

## RequestLogEntry

Frozen dataclass representing a single logged proxy request.

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

### Fields

| Field | Type | Description |
|-------|------|-------------|
| `id` | `str` | Auto-generated UUID hex |
| `timestamp` | `str` | ISO 8601 timestamp |
| `model` | `str` | Model name |
| `source_provider` | `str` | Source API format |
| `target_provider` | `str` | Target API format |
| `is_stream` | `bool` | Whether streaming was used |
| `status_code` | `int` | HTTP status code |
| `duration_ms` | `float` | Request duration in milliseconds |
| `error_detail` | `str \| None` | Error message (if any) |
| `api_key_label` | `str \| None` | API key label |
| `target_provider_name` | `str \| None` | Provider display name |
| `client_ip` | `str \| None` | Client IP address |
| `profile` | `dict \| None` | Profiling data |

---

## RequestLog

Proxy request log with optional SQLite persistence.  Delegates to
`PersistenceManager` when available, otherwise falls back to an in-memory
ring buffer.

```python
# In-memory only
log = RequestLog(max_entries=500)

# With persistence
persistence = PersistenceManager("/var/data/myproxy")
log = RequestLog(persistence=persistence)

log.add(entry)
entries, total = log.get_entries(limit=50, status="error")
```

### Key methods

| Method | Description |
|--------|-------------|
| `add(entry)` | Record a request log entry |
| `get_entries(...)` | Paginated, filtered query (newest-first) |
| `get_entry(entry_id)` | Single entry by ID |
| `get_api_key_labels()` | Distinct API key labels |
| `update_profile(entry_id, data)` | Merge profile data into existing entry |
| `clear()` | Delete all entries |

---

## PersistenceManager

SQLite-backed persistence for request logs and metrics counters.  Uses WAL
journal mode and dual-threshold retention (success/error independently pruned).

```python
pm = PersistenceManager(
    data_dir="/var/data/myproxy",
    success_max=50000,
    error_max=10000,
)
```

### Key methods

| Method | Description |
|--------|-------------|
| `insert_log_entries(entries)` | Bulk insert request log entries |
| `query_log_entries(...)` | Filtered query with pagination |
| `save_metrics(data)` | Persist metrics counters |
| `load_metrics()` | Load persisted metrics |
| `count_log_entries()` | Total entry count |
| `count_success_entries()` | Successful entries (status < 400) |
| `count_error_entries()` | Error entries (status ≥ 400) |
| `db_file_sizes()` | On-disk byte sizes |
| `close()` | Commit and close the database |

### Retention defaults

| Constant | Default | Description |
|----------|---------|-------------|
| `DEFAULT_SUCCESS_MAX` | 50,000 | Max successful entries |
| `DEFAULT_ERROR_MAX` | 10,000 | Max error entries |

---

## ProfilerState

Manages on-demand per-request pyinstrument profiling sessions.  Framework-agnostic
data layer — route handlers that wire `ProfilerState` into a web framework live
in the consumer.

```python
state = ProfilerState(max_results=20)
state.enable(requests=5)

if state.should_profile():
    profiler = state.create_profiler()
    profiler.start()
    # ... do work ...
    profiler.stop()
    state.store_result(profiler, model="gpt-4o", duration_ms=150.0)
```

### Key methods

| Method | Description |
|--------|-------------|
| `enable(requests)` | Enable profiling for next N requests |
| `disable()` | Manually disable |
| `should_profile()` | Check and consume one profiling slot |
| `create_profiler()` | Create a `DeepProfiler` instance |
| `store_result(profiler, ...)` | Store profiling result |
| `status()` | Current profiling status dict |
| `clear_results()` | Remove all stored results |

---

## Backward compatibility

All classes remain importable from their original `gateway.admin` locations:

```python
# These still work (re-exports from observability)
from llm_rosetta.gateway.admin.metrics import MetricsCollector
from llm_rosetta.gateway.admin.request_log import RequestLog, RequestLogEntry
from llm_rosetta.gateway.admin.persistence import PersistenceManager
from llm_rosetta.gateway.admin.routes.profiling import ProfilerState
```

New code should import from `llm_rosetta.observability` directly.
