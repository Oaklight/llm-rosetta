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

Lightweight in-process metrics.  All data structures are plain Python objects —
no framework dependencies.  Designed for single-threaded asyncio event loops
(no locks required).

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
    source_provider="chat_completions",
    target_provider="messages",
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
| `insert_dump_body(body_hash, data, orig_bytes)` | Insert a compressed body blob, deduplicating by hash |
| `insert_error_dump(dump_id, ...)` | Insert an error dump record and prune if over capacity |
| `query_error_dumps(limit, offset, ...)` | Filtered, paginated query of error dumps (newest-first) |
| `get_error_dump(dump_id)` | Return a single error dump by ID |
| `get_dump_body(body_hash)` | Return the compressed body blob for a hash |
| `count_error_dumps()` | Total number of error dump entries |
| `clear_error_dumps()` | Delete all error dumps and orphaned bodies |
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

## Operations Log (OpsLog)

The operations log captures server-level operational events — data mutations,
security events, key management, startup/shutdown — with structured metadata.
It delegates to SQLite persistence when available and falls back to an
in-memory ring buffer otherwise.

```python
from llm_rosetta.observability import OpsLog, OpsLogEntry
```

### OpsLogEntry Fields

| Field | Type | Description |
|-------|------|-------------|
| `id` | `str` | Auto-generated UUID hex |
| `timestamp` | `str` | ISO 8601 timestamp |
| `event_type` | `str` | Event category (see table below) |
| `severity` | `str` | `"info"`, `"warning"`, or `"error"` |
| `message` | `str` | Human-readable summary |
| `details` | `dict \| None` | Structured metadata (event-type specific) |
| `source` | `str \| None` | Originating subsystem |

### Event Types

| Constant | String | Description |
|----------|--------|-------------|
| `EVENT_STARTUP` | `startup` | Server started |
| `EVENT_SHUTDOWN` | `shutdown` | Graceful stop |
| `EVENT_CONFIG_RELOAD` | `config_reload` | Configuration reloaded |
| `EVENT_ADMIN_SETUP` | `admin_setup` | Admin panel initialized |
| `EVENT_KEY_CREATE` | `key_create` | API key created |
| `EVENT_KEY_UPDATE` | `key_update` | API key updated |
| `EVENT_KEY_DELETE` | `key_delete` | API key deleted |
| `EVENT_KEY_ROTATE` | `key_rotate` | API key rotated |
| `EVENT_HEALTH_CHANGE` | `health_status_change` | Provider health changed |
| `EVENT_DATA_CLEARED` | `data_cleared` | Table data cleared |
| `EVENT_DATA_CLEANUP` | `data_cleanup` | Age/date-based cleanup |
| `EVENT_DATA_TRIMMED` | `data_trimmed` | Trimmed to retention cap |
| `EVENT_DATA_VACUUMED` | `data_vacuumed` | Database vacuumed |
| `EVENT_DATA_REBUILT` | `data_rebuilt` | Metrics rebuilt from log |
| `EVENT_OPS_LOG_CLEARED` | `ops_log_cleared` | Ops log cleared |
| `EVENT_PASSWORD_CHANGED` | `password_changed` | Admin password changed |
| `EVENT_TOKEN_ROTATED` | `token_rotated` | Internal proxy token rotated |
| `EVENT_SESSION_LOGOUT_ALL` | `session_logout_all` | All admin sessions invalidated |

### Sources

| Constant | String | Used By |
|----------|--------|---------|
| `SOURCE_GATEWAY` | `gateway` | Startup, shutdown |
| `SOURCE_ADMIN` | `admin` | Admin panel actions, session logout |
| `SOURCE_KEYS` | `keys` | API key management |
| `SOURCE_CONFIG` | `config` | Config reload |
| `SOURCE_PERSISTENCE` | `persistence` | Background prune, periodic cleanup |
| `SOURCE_AUTH` | `auth` | Password change, token rotation |

### Severity Levels

| Constant | String | Usage |
|----------|--------|-------|
| `SEVERITY_INFO` | `info` | Non-destructive operations (default) |
| `SEVERITY_WARNING` | `warning` | Destructive data mutations, security events |
| `SEVERITY_ERROR` | `error` | Failures (auto-set when `_run()` raises) |

### OpsLog Methods

| Method | Description |
|--------|-------------|
| `add(entry)` | Record an operational event |
| `get_entries(...)` | Paginated, filtered query (newest-first) |
| `clear()` | Remove all entries, return count |

### ops_log vs request_log

| Aspect | ops_log | request_log |
|--------|---------|-------------|
| Records | Mutations, security events, lifecycle | Proxy request traffic |
| Volume | Low (admin actions, background tasks) | High (every proxy request) |
| Written by | `OpsBase._record()` (default) | `OpsProxyRequest._record()` (override) |
| Typical events | `data_cleared`, `key_create`, `password_changed` | Every `/v1/*` proxy request |

### OpsContext

`OpsContext` is a lightweight service bag connecting ops to observability
subsystems.  Created once at app init in `setup_admin()`:

```python
from llm_rosetta.gateway.ops import OpsContext

ctx = OpsContext(
    ops_log=ops_log,
    request_log=request_log,
    metrics=metrics,
    persistence=persistence,  # all optional — None disables that subsystem
)
app.ops_ctx = ctx
```

---

## Error Dump

When an upstream or conversion error occurs, the gateway can capture the full
request context for later replay or debugging.  The `error_dump` module provides
helpers that offload images, hash, compress, and store the dump payload via
`PersistenceManager`.

All public functions are fire-and-forget safe — they catch and log exceptions
internally so callers never need `try/except` wrappers.

### `dump_error()`

The primary entry point for recording error context.  Returns the dump ID on
success, or `None` when persistence is disabled or an exception occurs.

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

#### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `persistence` | `PersistenceManager \| None` | Persistence manager (`None` → no-op) |
| `request_body` | `dict \| None` | Original request body dict |
| `response_text` | `str \| None` | Upstream error response text (truncated to 64 KB) |
| `converted_body` | `dict \| None` | Converted target-format body, if available |
| `model` | `str \| None` | Model name from the request |
| `source_provider` | `str \| None` | Source API format (e.g. `"chat_completions"`) |
| `target_provider` | `str \| None` | Target API format (e.g. `"messages"`) |
| `provider_name` | `str \| None` | Human-readable provider name |
| `status_code` | `int \| None` | Upstream HTTP status code |
| `error_phase` | `str \| None` | One of `"upstream"`, `"stream_header"`, `"stream_chunk"`, `"conversion"` |
| `upstream_url` | `str \| None` | The upstream URL that was called |
| `request_log_id` | `str \| None` | Foreign key to the request log entry |

Bodies larger than 10 MB (`MAX_BODY_BYTES`) are skipped — only metadata is
stored.

### `offload_images(body)`

Replace inline base64 image data URIs with SHA256 digest placeholders.
Returns a deep copy — the original body is not mutated.

```python
from llm_rosetta.observability import offload_images

cleaned = offload_images(request_body)
# base64 data → "[image image/png sha256:abc123… 450KB]"
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `body` | `dict[str, Any]` | Request/response body dict |

**Returns:** A deep copy with every `data:image/…;base64,…` string replaced by
a human-readable placeholder.

### `compute_body_hash(body)`

SHA256 of canonicalized JSON (sorted keys, no whitespace).

```python
from llm_rosetta.observability import compute_body_hash

h = compute_body_hash({"model": "gpt-4o", "messages": []})
# "e3b0c44298fc1c149afb..."
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `body` | `dict[str, Any]` | Body dict to hash |

**Returns:** Hex-encoded SHA256 digest string.

### `compress_body(body)`

JSON-serialize and zlib-compress a body dict.

```python
from llm_rosetta.observability import compress_body

compressed, original_size = compress_body(request_body)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `body` | `dict[str, Any]` | Body dict to compress |

**Returns:** `(compressed_bytes, original_size)` tuple.

### `decompress_body(data)`

Decompress a zlib-compressed JSON body back to a dict.  Inverse of
`compress_body`.

```python
from llm_rosetta.observability import decompress_body

body = decompress_body(compressed)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `data` | `bytes` | zlib-compressed JSON bytes |

**Returns:** Deserialized dict.

---

## Usage

`llm_rosetta.observability.usage` owns what "how many tokens did we move, and
for whom" means.  It has no HTTP, no configuration and no credentials: a
library user records usage and reads a report without any of that, and the
gateway is one consumer of it, supplying credential subjects and SQLite
storage.

### The ledger's columns

| Column | Meaning |
|--------|---------|
| `input_tokens` | Prompt tokens, as the provider counted them |
| `output_tokens` | Completion tokens |
| `total_tokens` | The canonical total — see `token_total` below |
| `cache_read_tokens` | Cache hits |
| `cache_creation_tokens` | Cache writes |
| `reasoning_tokens` | Thinking tokens; **inside** `output_tokens`, never added |

`token_total(usage)` fills `total_tokens`.  A total the provider sent is taken
as given; otherwise the counters are added up, under two rules that are not
obvious:

- **Reasoning is never added** — it rides inside the output tokens.
- **Cache is added only where it is not already inside the prompt.**  The
  Messages API reports its cache tokens *beside* the input (additive); OpenAI
  and Google report theirs *inside* it (a subset).  Which is which is declared
  once, in `llm_rosetta.observability.usage.semantics`, and a test fails if a
  converter ever disagrees with the declaration.

`usage_columns(usage)` turns an IR usage dict into those columns.  A counter
the usage never named is left out rather than written as zero, so "no cache"
and "cache not reported" stay distinguishable; a non-integer value is not a
count either, and is treated as unreported — a payload whose counts arrived as
strings measures as no usage rather than raising out of the arithmetic.

### Model types

Every row also carries a **model type** — `llm`, `embedding`, `rerank` or
`decision`, the names `gateway/model_types.py` registers.  The column does not
carry the same IR field name in every family: chat asks the provider for
`prompt_tokens` and `completion_tokens`, the decision IR already reports
`input_tokens`/`output_tokens`, the embedding and rerank IR types have no
output side, and rerank's cache counter is `cached_tokens`.  So the
column-to-field map is declared per model type, `usage_columns(usage,
model_type)` and `token_total(usage, model_type)` read the right one, and a
type that is not one of the four is refused rather than guessed at.  The
canonical total rule is unchanged across families: a provider total wins,
otherwise the parts are added, and a family with no total of its own always
takes the additive path.

`UsageReport.by_model_type` reports the same rows cut by type —
`report.for_model_type("embedding", "24h")` — orthogonally to the classes
above, and a group that names no type counts as chat, which is what a row
written before the column existed means.  The gateway's daily table
(`GET /admin/api/token-usage`) returns the same split per day and accepts a
`model_type=` filter.

### UsageLedger

Record usage, then ask what it came to — no gateway, no database:

```python
from llm_rosetta.observability import UsageLedger

ledger = UsageLedger()
ledger.record(resp["usage"], model="gpt-5-nano", subject="batch-7")
ledger.record(resp["usage"], model="gpt-5-nano", subject="batch-7")

ledger.total("24h")
# {'request_count': 2, 'input_tokens': 40, 'output_tokens': 10, ...}
```

| Method | Description |
|--------|-------------|
| `record(usage, *, model=None, subject=None, label=None, model_type="llm", at=None, warnings=None)` | Record one call; returns the record, or `None` when the usage named no token count |
| `groups()` | The records summed per subject and window — the shape a stored ledger's reader returns |
| `report(subjects=...)` | Build a `UsageReport`; defaults to attributing every subject the ledger has seen |
| `total(window)` | Every recorded token in one window, however it is attributed |
| `records()` / `len(ledger)` / `clear()` | Read the records back, count them, or drop them |

Windows are rolling (`5h`, `24h`, `7d`, `30d`) and are measured against the
clock as the report is built, so a record ages out of the narrow ones on its
own.  `at=` dates a record at a past instant, which is what replaying a log
needs.

Two behaviours are worth knowing before relying on them:

- **A usage dict that names no token count records nothing** and returns
  `None`.  That is right for a response that carried no usage and wrong for a
  misspelled key, and from the outside the two look identical — so pass
  `warnings=[]` and the call says which keys it saw.  Explicit zeros are a
  count: a call that reported none is a call that happened, and it is recorded
  as a request with zero tokens.  A count that arrived as a string is not a
  count either, and measures as no usage rather than raising out of the
  arithmetic.
- **`report()` attributes every subject the ledger has recorded**, because
  there is no keystore here to say which of them are still live.  A ledger
  shared between parties that do not answer for each other should pass
  `subjects=` explicitly — names (`subjects=["batch-7"]`) or
  `(subject, label)` pairs.  The gateway reads the same rows differently, and
  deliberately: a subject whose credential is gone lands in the report's
  remainder there, because the gateway *does* have a keystore to ask.

### UsageReport

```python
from llm_rosetta.observability import UsageReport

report = UsageReport.build(
    groups=await persistence.query_usage_groups(),
    subjects=[(bucket, label) for ...],       # what may be attributed
    internal_subject=INTERNAL_SUBJECT,        # the gateway's own traffic
    internal_label=INTERNAL_LABEL,
)
report.total("24h")
```

Every row lands in exactly one of three classes, and `total()` sums all three:

| Class | What it is |
|-------|------------|
| attributed | The subject is one the caller named |
| internal | The caller's own traffic — in the gateway, the admin panel's token |
| unattributed | Everything else: a credential that is gone, a row written before subjects existed, a request that carried none.  Kept, with a per-label split |

Keeping the third class is what makes the report add up.  The dashboard counts
every row, so a report that quietly dropped the rows it could not attribute
would disagree with the dashboard's total; instead the two are the same number
by construction, and the API Keys panel shows the remainder as its own row.

`to_payload(names)` renders it as the admin API's JSON, with `names` mapping
each subject to the name to file it under — the gateway maps a credential
bucket to the key's id, and renames the remainder.

`INTERNAL_SUBJECT` and `INTERNAL_LABEL` (`"internal"`) are declared beside that
classification rather than in the gateway, so the code that writes the admin
token's rows and any reader that groups them name one thing once.

### The `llm-rosetta-usage` command

The gateway renders this report in its admin panel and a library gets it from
`UsageReport.build`; the third reader is a shell:

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

| Option | Meaning |
|--------|---------|
| `--db PATH` | Required. The data directory holding `gateway.db`, or that file |
| `--window NAME` | Which rolling window the tables show: `5h`, `24h` (default), `7d`, `30d` |
| `--by CUT[,CUT]` | What to cut by: `subject` (default), `type` (model type), `model` |
| `--keys-db PATH` | The keys database, for attribution — default `keys.db` beside the store |
| `--top N` | How many rows each table shows; `0` for all.  A table's total still sums every row of its cut, and `--json` is never truncated |
| `--migrate` | Open the store as the gateway would: backup, then migrations.  Off by default |
| `--json` | Print JSON instead of tables |

Every table is ordered by the window it shows, the widest window breaking
ties — so a row's position is explained by the numbers printed beside it, and a
model whose traffic is all outside `--window` sorts below one that has some.  A
table's last row is the total of *every* row of that cut, not only the ones
shown, so `--top` cannot make the total disagree with the store.  `--top`
bounds all three tables, including the subject table, which is the one that
grows with every credential the gateway has ever seen.

`--json` is the report itself rather than a second rendering of it: the subject
cut adds exactly the keys the admin API returns (`windows`, `keys`, `internal`,
`unattributed`), so a script reading the command and one reading the panel see
one shape.  The other cuts add `model_types` and `models`, and `--top` never
truncates them.

It opens the store through `PersistenceManager` — the gateway's own reader —
but **without writing to it**: no backup, no table creation, no migrations.  A
lookup should not leave a copy of the database behind, and it should not take a
write lock on a store another process is serving.  A store whose schema is
older than the installed build is therefore *refused* rather than migrated, with
the two ways out named: start the gateway once, or pass `--migrate`, which opens
the store exactly as the gateway does (backup first, then the migrations).  A
store that does not exist is refused too, rather than created — a mistyped path
must not leave an empty database that reads as a store with no traffic.

A store you cannot write — a mounted volume, a snapshot copy — reads fine as
long as SQLite can attach what it needs (another process serving the store
leaves the sidecars it attaches to).  Whether a given read-only store can be
read at all is SQLite's rule about `-wal`/`-shm` and not about the store's age,
so the command reports what it observed rather than guessing: a store whose
write-ahead log cannot be attached *reads as empty*, and it is refused as
exactly that, never as "schema version 0" — which would send the reader to
migrate a store that is already current.

Attribution needs the credentials that still exist, and they live in the
gateway's keys database, which the command reads read-only — two columns of one
table.  Without it every token is reported as unattributed and the command says
so on stderr — distinguishing a missing database from an unreadable one — which
is a true answer to a smaller question, since the totals are still the store's
and only the names are missing.

A row whose `total_tokens` is NULL — a store whose rows predate the gateway's
startup repair — is reported as it is stored, so it can read as `input 4,000 /
total 0`.  That is the same figure the admin panel shows; the repair runs when
the gateway starts, and this command deliberately does not write.

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
