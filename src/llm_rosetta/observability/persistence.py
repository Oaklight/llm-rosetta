"""SQLite-based persistence for observability data.

Stores request log entries and metrics counters in a single SQLite
database (``gateway.db``) using WAL journal mode.  Automatically
migrates legacy JSONL/JSON files on first startup.

All methods that touch the database are ``async`` — SQLite I/O runs on
a dedicated worker thread via the vendored ``aiosqlite`` module so
callers never block the asyncio event loop.

This module is framework-agnostic and can be used by any consumer
(the llm-rosetta gateway, argo-proxy, or standalone scripts).
"""

from __future__ import annotations

import asyncio
import gzip
import json
import logging
import warnings
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from pathlib import Path
from typing import Any

from llm_rosetta._vendor import aiosqlite

from .retention import RetentionPolicy, RetentionTracker

logger = logging.getLogger("llm-rosetta.observability")

_DB_FILENAME = "gateway.db"

# Legacy filenames for migration
_LEGACY_LOG = "request_log.jsonl"
_LEGACY_METRICS = "metrics.json"

# Backward-compatible aliases — values from RetentionPolicy defaults
_RP_DEFAULTS = RetentionPolicy()
DEFAULT_SUCCESS_MAX = _RP_DEFAULTS.success_max
DEFAULT_MAX_AGE_DAYS = _RP_DEFAULTS.max_age_days
DEFAULT_OPS_INFO_MAX = _RP_DEFAULTS.ops_info_max
DEFAULT_OPS_WARN_MAX = _RP_DEFAULTS.ops_warn_max

_PRUNE_BATCH_SIZE = 5000
_VACUUM_THRESHOLD = 1000

# WAL checkpoint defaults
DEFAULT_WAL_MAX_BYTES = 64 * 1024 * 1024  # 64 MB
_WAL_CHECKPOINT_INTERVAL = 300  # 5 minutes


class PersistenceManager:
    """SQLite-backed persistence for request logs and metrics.

    The request log retains successful entries up to ``success_max``.
    Error entries (status_code >= 400) are retained without a separate
    count cap — they are bounded by age-based cleanup only.

    Error *dumps* (the ``error_dumps`` table) have their own retention
    cap controlled by ``dump_max``.

    Use the async :meth:`create` classmethod to construct instances::

        pm = await PersistenceManager.create("/var/data/myproxy")

    The synchronous ``__init__`` is retained for backward compatibility
    but the instance will not have an open database connection — call
    :meth:`_open` before using any database methods.

    Args:
        data_dir: Directory for the database file (created if missing).
        success_max: Maximum number of successful request log entries to
            retain.  Defaults to :data:`DEFAULT_SUCCESS_MAX`.
        dump_max: Maximum number of error dump entries to retain.
            Defaults to :data:`DEFAULT_DUMP_MAX` (10 000).
        max_entries: Deprecated.  When provided and ``success_max`` is
            not, used as the success cap for backward compatibility.
            Emits a :class:`DeprecationWarning`.
    """

    def __init__(
        self,
        data_dir: str,
        success_max: int | None = None,
        dump_max: int | None = None,
        *,
        max_entries: int | None = None,
        ops_info_max: int | None = None,
        ops_warn_max: int | None = None,
        ops_log_max: int | None = None,
    ) -> None:
        if max_entries is not None:
            warnings.warn(
                "PersistenceManager(max_entries=...) is deprecated; "
                "use success_max= instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            if success_max is None:
                success_max = max_entries

        self._data_dir = Path(data_dir)
        self._retention = RetentionPolicy(
            success_max=(
                success_max if success_max is not None else DEFAULT_SUCCESS_MAX
            ),
            dump_max=dump_max if dump_max is not None else self.DEFAULT_DUMP_MAX,
            ops_info_max=(
                ops_info_max
                if ops_info_max is not None
                else (ops_log_max if ops_log_max is not None else DEFAULT_OPS_INFO_MAX)
            ),
            ops_warn_max=(
                ops_warn_max
                if ops_warn_max is not None
                else (ops_log_max if ops_log_max is not None else DEFAULT_OPS_WARN_MAX)
            ),
        )
        self._tracker = RetentionTracker()
        self._wal_max_bytes = DEFAULT_WAL_MAX_BYTES
        self._wal_task: asyncio.Task[None] | None = None
        self.on_prune: Callable[[str, int], Awaitable[None]] | None = None
        self._data_dir.mkdir(parents=True, exist_ok=True)

        # Connection is opened by create() or _open(); asserted non-None
        # in all methods (callers must use create() or _open() first).
        self._conn: aiosqlite.Connection = None  # type: ignore[assignment]  # ty: ignore[invalid-assignment]

    async def _open(self) -> None:
        """Open the database connection, create tables, and run migrations."""
        self._conn = await aiosqlite.connect(str(self.db_path))
        await self._conn.execute("PRAGMA journal_mode=WAL")
        await self._conn.execute("PRAGMA synchronous=NORMAL")
        await self._conn.execute("PRAGMA foreign_keys=ON")
        await self._init_tables()
        await self._run_migrations()
        await self._migrate_legacy()

    @classmethod
    async def create(
        cls,
        data_dir: str,
        success_max: int | None = None,
        dump_max: int | None = None,
        *,
        max_entries: int | None = None,
        ops_info_max: int | None = None,
        ops_warn_max: int | None = None,
        ops_log_max: int | None = None,
    ) -> PersistenceManager:
        """Async factory: create a PersistenceManager with an open connection.

        This is the preferred way to construct instances in async code.
        """
        instance = cls(
            data_dir,
            success_max=success_max,
            dump_max=dump_max,
            max_entries=max_entries,
            ops_info_max=ops_info_max,
            ops_warn_max=ops_warn_max,
            ops_log_max=ops_log_max,
        )
        await instance._open()
        return instance

    @property
    def retention(self) -> RetentionPolicy:
        """The current retention policy."""
        return self._retention

    @property
    def success_max(self) -> int:
        """Cap on retained successful request log entries."""
        return self._retention.success_max

    @success_max.setter
    def success_max(self, value: int) -> None:
        self._retention.success_max = value

    @property
    def dump_max(self) -> int:
        """Cap on retained error dump entries."""
        return self._retention.dump_max

    @dump_max.setter
    def dump_max(self, value: int) -> None:
        self._retention.dump_max = value

    @property
    def db_path(self) -> Path:
        return self._data_dir / _DB_FILENAME

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    async def _init_tables(self) -> None:
        await self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS request_log (
                id              TEXT PRIMARY KEY,
                timestamp       TEXT NOT NULL,
                model           TEXT NOT NULL,
                source_provider TEXT NOT NULL,
                target_provider TEXT NOT NULL,
                is_stream       INTEGER NOT NULL,
                status_code     INTEGER NOT NULL,
                duration_ms     REAL NOT NULL,
                error_detail    TEXT,
                api_key_label   TEXT,
                target_provider_name TEXT,
                client_ip       TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_rl_timestamp
                ON request_log(timestamp DESC);
            CREATE INDEX IF NOT EXISTS idx_rl_status
                ON request_log(status_code);
            CREATE INDEX IF NOT EXISTS idx_rl_success_ts
                ON request_log(timestamp ASC) WHERE status_code < 400;
            CREATE TABLE IF NOT EXISTS metrics (
                key   TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS dump_bodies (
                hash        TEXT PRIMARY KEY,
                data        BLOB NOT NULL,
                orig_bytes  INTEGER NOT NULL,
                created     TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS error_dumps (
                id                  TEXT PRIMARY KEY,
                request_log_id      TEXT REFERENCES request_log(id) ON DELETE SET NULL,
                timestamp           TEXT NOT NULL,
                model               TEXT,
                source_provider     TEXT,
                target_provider     TEXT,
                provider_name       TEXT,
                status_code         INTEGER,
                error_phase         TEXT,
                body_hash           TEXT,
                response_text       TEXT,
                upstream_url        TEXT,
                converted_body_hash TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_ed_timestamp
                ON error_dumps(timestamp DESC);
            CREATE INDEX IF NOT EXISTS idx_ed_request_log
                ON error_dumps(request_log_id);

            CREATE TABLE IF NOT EXISTS ops_log (
                id          TEXT PRIMARY KEY,
                timestamp   TEXT NOT NULL,
                event_type  TEXT NOT NULL,
                severity    TEXT NOT NULL,
                message     TEXT NOT NULL,
                details     TEXT,
                source      TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_ol_timestamp
                ON ops_log(timestamp DESC);
            CREATE INDEX IF NOT EXISTS idx_ol_event_type
                ON ops_log(event_type);
            CREATE INDEX IF NOT EXISTS idx_ol_severity_ts
                ON ops_log(severity, timestamp DESC);

            CREATE TABLE IF NOT EXISTS schema_version (
                version     INTEGER PRIMARY KEY,
                applied_at  TEXT NOT NULL
            );
        """)

    # ------------------------------------------------------------------
    # Versioned migrations
    # ------------------------------------------------------------------

    # Each entry: (version, description, sql_statements)
    # Migrations are applied in order; each bumps the schema version.
    _MIGRATIONS: list[tuple[int, str, list[str]]] = [
        (
            1,
            "add request_log token/profile columns",
            [
                "ALTER TABLE request_log ADD COLUMN profile TEXT",
                "ALTER TABLE request_log ADD COLUMN input_tokens INTEGER",
                "ALTER TABLE request_log ADD COLUMN output_tokens INTEGER",
                "ALTER TABLE request_log ADD COLUMN total_tokens INTEGER",
                "ALTER TABLE request_log ADD COLUMN cache_read_tokens INTEGER",
                "ALTER TABLE request_log ADD COLUMN cache_creation_tokens INTEGER",
                "ALTER TABLE request_log ADD COLUMN reasoning_tokens INTEGER",
            ],
        ),
        (
            2,
            "add error_dumps indexes on model and error_phase",
            [
                "CREATE INDEX IF NOT EXISTS idx_ed_model ON error_dumps(model)",
                "CREATE INDEX IF NOT EXISTS idx_ed_error_phase ON error_dumps(error_phase)",
            ],
        ),
    ]

    async def _run_migrations(self) -> None:
        """Apply pending versioned migrations."""
        row = await self._conn.execute_fetchone(
            "SELECT MAX(version) FROM schema_version"
        )
        current = row[0] if row and row[0] is not None else 0

        # Bootstrap: detect columns already added by old _migrate_add_columns.
        # Check for the last column added by v1 (reasoning_tokens) to confirm
        # the old migration fully completed.
        if current == 0:
            cursor = await self._conn.execute("PRAGMA table_info(request_log)")
            columns = {r[1] for r in await cursor.fetchall()}
            if "reasoning_tokens" in columns:
                from datetime import datetime, timezone

                await self._conn.execute(
                    "INSERT OR IGNORE INTO schema_version (version, applied_at) "
                    "VALUES (?, ?)",
                    (1, datetime.now(timezone.utc).isoformat()),
                )
                await self._conn.commit()
                current = 1

        for version, description, statements in self._MIGRATIONS:
            if version <= current:
                continue
            logger.info("Applying migration v%d: %s", version, description)
            for sql in statements:
                try:
                    await self._conn.execute(sql)
                except Exception as exc:
                    if "duplicate column" in str(exc).lower():
                        continue
                    raise
            from datetime import datetime, timezone

            await self._conn.execute(
                "INSERT INTO schema_version (version, applied_at) VALUES (?, ?)",
                (version, datetime.now(timezone.utc).isoformat()),
            )
            await self._conn.commit()
            logger.info("Migration v%d applied", version)

    async def schema_version(self) -> int:
        """Return the current schema version number."""
        try:
            row = await self._conn.execute_fetchone(
                "SELECT MAX(version) FROM schema_version"
            )
            return row[0] if row and row[0] is not None else 0
        except Exception:
            return 0

    async def backfill_provider_names(
        self, model_to_provider: Mapping[str, str]
    ) -> int:
        """Backfill target_provider_name for old entries using the model->provider mapping.

        Only updates rows where target_provider_name is NULL and the
        model exists in the current config.

        Args:
            model_to_provider: Mapping from model name to provider name
                (e.g. ``{"argo:claude-opus-4.6": "Argo Claude"}``).

        Returns:
            Number of rows updated.
        """
        if not model_to_provider:
            return 0
        total = 0
        for model_name, provider_name in model_to_provider.items():
            cursor = await self._conn.execute(
                "UPDATE request_log SET target_provider_name = ? "
                "WHERE model = ? AND target_provider_name IS NULL",
                (provider_name, model_name),
            )
            total += cursor.rowcount
        if total:
            await self._conn.commit()
        return total

    async def backfill_total_tokens(self) -> int:
        """Recalculate total_tokens for rows where cache tokens were excluded.

        Anthropic's cache_read/cache_creation tokens are additive to
        input_tokens, but earlier code computed total = input + output
        only.  This backfill corrects historical rows.

        Returns:
            Number of rows updated.
        """
        cursor = await self._conn.execute(
            "UPDATE request_log "
            "SET total_tokens = COALESCE(input_tokens, 0) "
            "    + COALESCE(output_tokens, 0) "
            "    + COALESCE(cache_read_tokens, 0) "
            "    + COALESCE(cache_creation_tokens, 0) "
            "WHERE (cache_read_tokens > 0 OR cache_creation_tokens > 0) "
            "  AND total_tokens < COALESCE(input_tokens, 0) "
            "    + COALESCE(output_tokens, 0) "
            "    + COALESCE(cache_read_tokens, 0) "
            "    + COALESCE(cache_creation_tokens, 0)"
        )
        updated = cursor.rowcount
        if updated:
            await self._conn.commit()
            logger.info("Backfilled total_tokens for %d rows", updated)
        return updated

    # ------------------------------------------------------------------
    # Request log
    # ------------------------------------------------------------------

    _LOG_COLUMNS = [
        "id",
        "timestamp",
        "model",
        "source_provider",
        "target_provider",
        "is_stream",
        "status_code",
        "duration_ms",
        "error_detail",
        "api_key_label",
        "target_provider_name",
        "client_ip",
        "profile",
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "cache_read_tokens",
        "cache_creation_tokens",
        "reasoning_tokens",
    ]

    async def insert_log_entries(self, entries: list[dict[str, Any]]) -> None:
        """Insert request log entries, pruning oldest if over capacity."""
        if not entries:
            return
        await self._conn.executemany(
            "INSERT OR IGNORE INTO request_log "
            "(id, timestamp, model, source_provider, target_provider, "
            "is_stream, status_code, duration_ms, error_detail, api_key_label, "
            "target_provider_name, client_ip, profile, "
            "input_tokens, output_tokens, total_tokens, "
            "cache_read_tokens, cache_creation_tokens, reasoning_tokens) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    e["id"],
                    e["timestamp"],
                    e["model"],
                    e["source_provider"],
                    e["target_provider"],
                    int(e["is_stream"]),
                    e["status_code"],
                    e["duration_ms"],
                    e.get("error_detail"),
                    e.get("api_key_label"),
                    e.get("target_provider_name"),
                    e.get("client_ip"),
                    json.dumps(e["profile"]) if e.get("profile") else None,
                    e.get("input_tokens"),
                    e.get("output_tokens"),
                    e.get("total_tokens"),
                    e.get("cache_read_tokens"),
                    e.get("cache_creation_tokens"),
                    e.get("reasoning_tokens"),
                )
                for e in entries
            ],
        )
        await self._conn.commit()
        if self._tracker.note_insert("request_log", len(entries)):
            await self._prune()
        elif await self.count_success_entries() > self._retention.success_max:
            await self._prune()

    async def query_log_entries(
        self,
        *,
        limit: int = 50,
        offset: int = 0,
        model: str | None = None,
        provider: str | None = None,
        provider_type: str | None = None,
        status: str | None = None,
        api_key_label: str | None = None,
    ) -> tuple[list[dict[str, Any]], int]:
        """Query request log with optional filters, newest first.

        Args:
            provider: Provider display name (e.g. ``"Gemini"``).
            provider_type: Resolved API type for *provider* (e.g.
                ``"google"``).  Enables matching legacy rows that only
                have ``target_provider`` (the API type) without a
                ``target_provider_name`` backfill.

        Returns:
            A ``(entries, total)`` tuple.
        """
        where_clauses: list[str] = []
        params: list[Any] = []

        if model:
            where_clauses.append("model = ?")
            params.append(model)
        if provider:
            if provider_type and provider_type != provider:
                # Match by name, OR fall back to API type only for legacy
                # rows that have no target_provider_name (avoids cross-
                # contamination when multiple providers share a base type).
                where_clauses.append(
                    "(target_provider_name = ? OR target_provider = ? "
                    "OR (target_provider_name IS NULL AND target_provider = ?))"
                )
                params.extend([provider, provider, provider_type])
            else:
                where_clauses.append(
                    "(target_provider_name = ? OR target_provider = ?)"
                )
                params.extend([provider, provider])
        if status == "ok":
            where_clauses.append("status_code < 400")
        elif status == "error":
            where_clauses.append("status_code >= 400")
        elif status == "4xx":
            where_clauses.append("status_code >= 400 AND status_code < 500")
        elif status == "5xx":
            where_clauses.append("status_code >= 500 AND status_code < 600")
        elif status and status.isdigit():
            where_clauses.append("status_code = ?")
            params.append(int(status))
        if api_key_label:
            where_clauses.append("api_key_label = ?")
            params.append(api_key_label)

        where_sql = ""
        if where_clauses:
            where_sql = "WHERE " + " AND ".join(where_clauses)

        count_row = await self._conn.execute_fetchone(
            f"SELECT COUNT(*) FROM request_log {where_sql}", params
        )
        total = count_row[0] if count_row else 0

        rows = await self._conn.execute_fetchall(
            f"SELECT * FROM request_log {where_sql} "
            f"ORDER BY timestamp DESC LIMIT ? OFFSET ?",
            [*params, limit, offset],
        )

        entries = [self._row_to_dict(row) for row in rows]
        return entries, total

    async def get_log_entry(self, entry_id: str) -> dict[str, Any] | None:
        """Return a single log entry by id, or ``None``.

        Includes ``_offset`` — the entry's position in the newest-first
        list — so the admin UI can jump directly to the correct page.
        Uses an indexed timestamp comparison; acceptable for single-entry
        lookups (not called in hot paths).
        """
        row = await self._conn.execute_fetchone(
            "SELECT * FROM request_log WHERE id = ?", (entry_id,)
        )
        if row is None:
            return None
        d = self._row_to_dict(row)
        offset_row = await self._conn.execute_fetchone(
            "SELECT COUNT(*) FROM request_log WHERE timestamp > ?",
            (d["timestamp"],),
        )
        d["_offset"] = offset_row[0] if offset_row else 0
        return d

    async def backfill_error_dump_log_ids(
        self,
        window_seconds: float = 0.1,
        model_aliases: dict[str, str] | None = None,
    ) -> int:
        """Match error dumps with NULL request_log_id to request_log entries.

        Uses timestamp proximity, source/target provider, status code, and
        model name to find the best match.  *model_aliases* maps
        ``request_model -> upstream_model`` (from gateway config) so that
        dump model names (which use upstream names) can be matched against
        request log entries (which use request names).

        Returns the number of rows updated.
        """
        reverse_aliases: dict[str, list[str]] = {}
        for req_name, up_name in (model_aliases or {}).items():
            reverse_aliases.setdefault(up_name, []).append(req_name)

        unmatched = await self._conn.execute_fetchall(
            "SELECT id, timestamp, model, status_code, source_provider, target_provider "
            "FROM error_dumps WHERE request_log_id IS NULL OR request_log_id = ''"
        )
        if not unmatched:
            return 0
        updated = 0
        for dump_id, ts, dump_model, status, source, target in unmatched:
            if not ts:
                continue
            candidate_models = [dump_model] if dump_model else []
            for alias in reverse_aliases.get(dump_model or "", []):
                if alias not in candidate_models:
                    candidate_models.append(alias)
            row = None
            for m in candidate_models:
                row = await self._conn.execute_fetchone(
                    "SELECT id FROM request_log "
                    "WHERE source_provider = ? AND target_provider = ? "
                    "AND status_code = ? AND model = ? "
                    "AND abs(julianday(timestamp) - julianday(?)) * 86400 < ? "
                    "ORDER BY abs(julianday(timestamp) - julianday(?)) "
                    "LIMIT 1",
                    (source, target, status, m, ts, window_seconds, ts),
                )
                if row:
                    break
            if not row and dump_model:
                # Fallback: match without model constraint. May mis-link if
                # two different models error with the same status in the window.
                row = await self._conn.execute_fetchone(
                    "SELECT id FROM request_log "
                    "WHERE source_provider = ? AND target_provider = ? "
                    "AND status_code = ? "
                    "AND abs(julianday(timestamp) - julianday(?)) * 86400 < ? "
                    "ORDER BY abs(julianday(timestamp) - julianday(?)) "
                    "LIMIT 1",
                    (source, target, status, ts, window_seconds, ts),
                )
            if row:
                await self._conn.execute(
                    "UPDATE error_dumps SET request_log_id = ? WHERE id = ?",
                    (row[0], dump_id),
                )
                updated += 1
        if updated:
            await self._conn.commit()
        return updated

    async def get_api_key_labels(self) -> list[str]:
        """Return distinct API key labels seen in request logs."""
        rows = await self._conn.execute_fetchall(
            "SELECT DISTINCT api_key_label FROM request_log "
            "WHERE api_key_label IS NOT NULL AND api_key_label != '' "
            "ORDER BY api_key_label"
        )
        return [row[0] for row in rows]

    async def iter_log_rows_for_rebuild(
        self, batch_size: int = 5000
    ) -> AsyncIterator[dict[str, Any]]:
        """Yield lightweight dicts for every log entry (for counter rebuild).

        Only fetches the columns needed by
        :meth:`~MetricsCollector.rebuild_counters`.  Rows are fetched
        in batches of *batch_size* to bound memory usage regardless of
        table size.
        """
        cursor = await self._conn.execute(
            "SELECT model, source_provider, target_provider, "
            "target_provider_name, is_stream, status_code, "
            "input_tokens, output_tokens, total_tokens, "
            "cache_read_tokens, cache_creation_tokens, reasoning_tokens "
            "FROM request_log"
        )
        while True:
            batch = await cursor.fetchmany(batch_size)
            if not batch:
                break
            for r in batch:
                yield {
                    "model": r[0],
                    "source_provider": r[1],
                    "target_provider": r[2],
                    "target_provider_name": r[3],
                    "is_stream": bool(r[4]),
                    "status_code": r[5],
                    "input_tokens": r[6],
                    "output_tokens": r[7],
                    "total_tokens": r[8],
                    "cache_read_tokens": r[9],
                    "cache_creation_tokens": r[10],
                    "reasoning_tokens": r[11],
                }

    async def count_log_entries(self) -> int:
        """Return the total number of log entries."""
        row = await self._conn.execute_fetchone("SELECT COUNT(*) FROM request_log")
        return row[0] if row else 0

    async def count_success_entries(self) -> int:
        """Return the number of successful log entries (status_code < 400)."""
        row = await self._conn.execute_fetchone(
            "SELECT COUNT(*) FROM request_log WHERE status_code < 400"
        )
        return row[0] if row else 0

    async def count_error_entries(self) -> int:
        """Return the number of error log entries (status_code >= 400)."""
        row = await self._conn.execute_fetchone(
            "SELECT COUNT(*) FROM request_log WHERE status_code >= 400"
        )
        return row[0] if row else 0

    def db_file_sizes(self) -> dict[str, int]:
        """Return on-disk byte sizes of the SQLite database files.

        Returns:
            Dict with keys ``db_bytes`` (main file), ``wal_bytes`` (WAL),
            and ``shm_bytes`` (shared memory).  Missing files report 0.
        """
        db = self.db_path
        sizes = {"db_bytes": 0, "wal_bytes": 0, "shm_bytes": 0}
        for key, suffix in (
            ("db_bytes", ""),
            ("wal_bytes", "-wal"),
            ("shm_bytes", "-shm"),
        ):
            p = db.with_name(db.name + suffix)
            try:
                sizes[key] = p.stat().st_size
            except OSError:
                sizes[key] = 0
        return sizes

    async def clear_log(self) -> None:
        """Delete all request log entries."""
        await self._conn.execute("DELETE FROM request_log")
        await self._conn.commit()

    # ------------------------------------------------------------------
    # Ops log
    # ------------------------------------------------------------------

    _OPS_LOG_COLUMNS = [
        "id",
        "timestamp",
        "event_type",
        "severity",
        "message",
        "details",
        "source",
    ]

    async def insert_ops_log_entries(
        self,
        entries: list[dict[str, Any]],
        *,
        _skip_prune: bool = False,
    ) -> None:
        """Insert ops log entries, pruning oldest if over capacity."""
        if not entries:
            return
        await self._conn.executemany(
            "INSERT OR IGNORE INTO ops_log "
            "(id, timestamp, event_type, severity, message, details, source) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    e["id"],
                    e["timestamp"],
                    e["event_type"],
                    e["severity"],
                    e["message"],
                    json.dumps(e["details"]) if e.get("details") else None,
                    e.get("source"),
                )
                for e in entries
            ],
        )
        await self._conn.commit()
        if _skip_prune:
            return
        if self._tracker.note_insert("ops_log", len(entries)):
            await self._prune_ops_log()

    async def query_ops_log_entries(
        self,
        *,
        limit: int = 50,
        offset: int = 0,
        event_type: str | None = None,
        severity: str | None = None,
        source: str | None = None,
    ) -> tuple[list[dict[str, Any]], int]:
        """Return filtered ops log entries (newest-first) and total count."""
        where: list[str] = []
        params: list[Any] = []
        if event_type:
            where.append("event_type = ?")
            params.append(event_type)
        if severity:
            where.append("severity = ?")
            params.append(severity)
        if source:
            where.append("source = ?")
            params.append(source)

        clause = (" WHERE " + " AND ".join(where)) if where else ""

        row = await self._conn.execute_fetchone(
            f"SELECT COUNT(*) FROM ops_log{clause}", params
        )
        total = row[0] if row else 0

        rows = await self._conn.execute_fetchall(
            f"SELECT {', '.join(self._OPS_LOG_COLUMNS)} FROM ops_log"
            f"{clause} ORDER BY timestamp DESC LIMIT ? OFFSET ?",
            [*params, limit, offset],
        )
        entries = [self._ops_row_to_dict(r) for r in rows]
        return entries, total

    async def count_ops_log_entries(self) -> int:
        """Return the total number of ops log entries."""
        row = await self._conn.execute_fetchone("SELECT COUNT(*) FROM ops_log")
        return row[0] if row else 0

    async def count_ops_info_entries(self) -> int:
        """Return ops log entries with severity='info'."""
        row = await self._conn.execute_fetchone(
            "SELECT COUNT(*) FROM ops_log WHERE severity = 'info'"
        )
        return row[0] if row else 0

    async def count_ops_warn_entries(self) -> int:
        """Return ops log entries with severity in ('warning', 'error')."""
        row = await self._conn.execute_fetchone(
            "SELECT COUNT(*) FROM ops_log WHERE severity != 'info'"
        )
        return row[0] if row else 0

    async def clear_ops_log(self) -> None:
        """Delete all ops log entries."""
        await self._conn.execute("DELETE FROM ops_log")
        await self._conn.commit()

    async def cleanup_ops_log_by_age(self, max_age_days: int) -> dict[str, Any]:
        """Delete ops_log rows older than *max_age_days*."""
        from datetime import datetime, timedelta, timezone

        cutoff = (datetime.now(timezone.utc) - timedelta(days=max_age_days)).isoformat()
        size_before = self.db_path.stat().st_size
        deleted = await self._batched_delete("ops_log", "timestamp < ?", (cutoff,))
        vacuum_ok = await self._conditional_vacuum(deleted)
        size_after = self.db_path.stat().st_size

        return {
            "deleted": deleted,
            "freed_bytes": max(0, size_before - size_after),
            "size_before": size_before,
            "size_after": size_after,
            "max_age_days": max_age_days,
            "vacuum": vacuum_ok,
        }

    async def _prune_ops_log(self) -> None:
        """Remove oldest ops log entries beyond per-severity retention limits."""
        committed = False

        info_excess = await self.count_ops_info_entries() - self._retention.ops_info_max
        if info_excess > 0:
            await self._conn.execute(
                "DELETE FROM ops_log "
                "WHERE rowid IN ("
                "    SELECT rowid FROM ops_log "
                "    WHERE severity = 'info' "
                "    ORDER BY timestamp ASC "
                "    LIMIT ?"
                ")",
                (info_excess,),
            )
            committed = True

        warn_excess = await self.count_ops_warn_entries() - self._retention.ops_warn_max
        if warn_excess > 0:
            await self._conn.execute(
                "DELETE FROM ops_log "
                "WHERE rowid IN ("
                "    SELECT rowid FROM ops_log "
                "    WHERE severity != 'info' "
                "    ORDER BY timestamp ASC "
                "    LIMIT ?"
                ")",
                (warn_excess,),
            )
            committed = True

        if committed:
            await self._conn.commit()
            total_pruned = max(0, info_excess) + max(0, warn_excess)
            if total_pruned > 0 and self.on_prune is not None:
                await self.on_prune("ops_log", total_pruned)

    @classmethod
    def _ops_row_to_dict(cls, row: tuple[Any, ...]) -> dict[str, Any]:
        """Convert an ops_log row tuple to a dict, omitting None fields."""
        d: dict[str, Any] = {}
        for col, val in zip(cls._OPS_LOG_COLUMNS, row):
            if col == "details" and val is not None:
                try:
                    val = json.loads(val)
                except (json.JSONDecodeError, TypeError):
                    pass
            if val is not None:
                d[col] = val
        # Always include required fields even if somehow None
        for key in ("id", "timestamp", "event_type", "severity", "message"):
            if key not in d:
                d[key] = ""
        return d

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    async def save_metrics(self, data: dict[str, Any]) -> None:
        """Persist metrics counters."""
        await self._conn.execute(
            "INSERT OR REPLACE INTO metrics (key, value) VALUES (?, ?)",
            ("counters", json.dumps(data, ensure_ascii=False)),
        )
        await self._conn.commit()

    async def load_metrics(self) -> dict[str, Any] | None:
        """Load metrics counters, or ``None`` if not yet saved."""
        row = await self._conn.execute_fetchone(
            "SELECT value FROM metrics WHERE key = ?", ("counters",)
        )
        if row is None:
            return None
        try:
            return json.loads(row[0])
        except (json.JSONDecodeError, TypeError) as exc:
            logger.warning("Failed to load metrics: %s", exc)
            return None

    async def set_rebuild_flag(self) -> None:
        """Signal that counters need rebuilding (used by CLI cleanup)."""
        await self._conn.execute(
            "INSERT OR REPLACE INTO metrics (key, value) VALUES (?, ?)",
            ("rebuild_needed", "1"),
        )
        await self._conn.commit()

    async def check_and_clear_rebuild_flag(self) -> bool:
        """Check if a rebuild was requested; atomically clear the flag if set."""
        row = await self._conn.execute_fetchone(
            "DELETE FROM metrics WHERE key = ? AND value = ? RETURNING value",
            ("rebuild_needed", "1"),
        )
        if row is not None:
            await self._conn.commit()
            return True
        return False

    # ------------------------------------------------------------------
    # Error dumps
    # ------------------------------------------------------------------

    # Default retention cap for error_dumps rows.
    DEFAULT_DUMP_MAX = _RP_DEFAULTS.dump_max

    async def insert_dump_body(
        self, body_hash: str, data: bytes, orig_bytes: int
    ) -> None:
        """Insert a compressed body blob, deduplicating by hash.

        If the hash already exists the row is silently skipped.
        """
        from datetime import datetime, timezone

        await self._conn.execute(
            "INSERT OR IGNORE INTO dump_bodies (hash, data, orig_bytes, created) "
            "VALUES (?, ?, ?, ?)",
            (body_hash, data, orig_bytes, datetime.now(timezone.utc).isoformat()),
        )
        await self._conn.commit()

    async def insert_error_dump(
        self,
        *,
        dump_id: str,
        request_log_id: str | None,
        timestamp: str,
        model: str | None,
        source_provider: str | None,
        target_provider: str | None,
        provider_name: str | None,
        status_code: int | None,
        error_phase: str | None,
        body_hash: str | None,
        response_text: str | None,
        upstream_url: str | None,
        converted_body_hash: str | None = None,
    ) -> None:
        """Insert an error dump record and prune if over capacity."""
        await self._conn.execute(
            "INSERT OR IGNORE INTO error_dumps "
            "(id, request_log_id, timestamp, model, source_provider, "
            "target_provider, provider_name, status_code, error_phase, "
            "body_hash, response_text, upstream_url, converted_body_hash) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                dump_id,
                request_log_id,
                timestamp,
                model,
                source_provider,
                target_provider,
                provider_name,
                status_code,
                error_phase,
                body_hash,
                response_text,
                upstream_url,
                converted_body_hash,
            ),
        )
        await self._conn.commit()
        await self._prune_error_dumps()

    async def query_error_dumps(
        self,
        *,
        limit: int = 50,
        offset: int = 0,
        model: str | None = None,
        error_phase: str | None = None,
        provider: str | None = None,
    ) -> tuple[list[dict[str, Any]], int]:
        """Query error dumps with optional filters, newest first.

        Returns:
            A ``(entries, total)`` tuple.
        """
        where_clauses: list[str] = []
        params: list[Any] = []

        if model:
            where_clauses.append("model = ?")
            params.append(model)
        if error_phase:
            where_clauses.append("error_phase = ?")
            params.append(error_phase)
        if provider:
            where_clauses.append("(provider_name = ? OR target_provider = ?)")
            params.extend([provider, provider])

        where_sql = ""
        if where_clauses:
            where_sql = "WHERE " + " AND ".join(where_clauses)

        count_row = await self._conn.execute_fetchone(
            f"SELECT COUNT(*) FROM error_dumps {where_sql}", params
        )
        total = count_row[0] if count_row else 0

        cols = (
            "id, request_log_id, timestamp, model, source_provider, "
            "target_provider, provider_name, status_code, error_phase, "
            "body_hash, response_text, upstream_url, converted_body_hash"
        )
        rows = await self._conn.execute_fetchall(
            f"SELECT {cols} FROM error_dumps {where_sql} "
            f"ORDER BY timestamp DESC LIMIT ? OFFSET ?",
            [*params, limit, offset],
        )

        col_names = [
            "id",
            "request_log_id",
            "timestamp",
            "model",
            "source_provider",
            "target_provider",
            "provider_name",
            "status_code",
            "error_phase",
            "body_hash",
            "response_text",
            "upstream_url",
            "converted_body_hash",
        ]
        entries = [
            {k: v for k, v in zip(col_names, row) if v is not None} for row in rows
        ]
        return entries, total

    async def get_error_dump(self, dump_id: str) -> dict[str, Any] | None:
        """Return a single error dump by ID, or ``None``."""
        cols = (
            "id, request_log_id, timestamp, model, source_provider, "
            "target_provider, provider_name, status_code, error_phase, "
            "body_hash, response_text, upstream_url, converted_body_hash"
        )
        row = await self._conn.execute_fetchone(
            f"SELECT {cols} FROM error_dumps WHERE id = ?", (dump_id,)
        )
        if row is None:
            return None
        col_names = [
            "id",
            "request_log_id",
            "timestamp",
            "model",
            "source_provider",
            "target_provider",
            "provider_name",
            "status_code",
            "error_phase",
            "body_hash",
            "response_text",
            "upstream_url",
            "converted_body_hash",
        ]
        return {k: v for k, v in zip(col_names, row) if v is not None}

    async def get_dump_body(self, body_hash: str) -> bytes | None:
        """Return the compressed body blob for a hash, or ``None``."""
        row = await self._conn.execute_fetchone(
            "SELECT data FROM dump_bodies WHERE hash = ?", (body_hash,)
        )
        return row[0] if row else None

    async def count_error_dumps(self) -> int:
        """Return the total number of error dump entries."""
        row = await self._conn.execute_fetchone("SELECT COUNT(*) FROM error_dumps")
        return row[0] if row else 0

    async def delete_error_dump(self, dump_id: str) -> bool:
        """Delete a single error dump by ID and clean up orphaned bodies."""
        cur = await self._conn.execute(
            "DELETE FROM error_dumps WHERE id = ?", (dump_id,)
        )
        if cur.rowcount == 0:
            return False
        await self._delete_orphan_bodies()
        await self._conn.commit()
        return True

    async def clear_error_dumps(self) -> None:
        """Delete all error dumps and orphaned bodies."""
        await self._conn.execute("DELETE FROM error_dumps")
        await self._conn.execute(
            "DELETE FROM dump_bodies WHERE hash NOT IN "
            "(SELECT body_hash FROM error_dumps WHERE body_hash IS NOT NULL "
            " UNION SELECT converted_body_hash FROM error_dumps "
            " WHERE converted_body_hash IS NOT NULL)"
        )
        await self._conn.commit()

    async def _delete_orphan_bodies(self) -> int:
        """Delete dump_bodies not referenced by any error_dumps."""
        cur = await self._conn.execute(
            "DELETE FROM dump_bodies WHERE hash NOT IN ("
            "    SELECT body_hash FROM error_dumps "
            "    WHERE body_hash IS NOT NULL"
            "    UNION "
            "    SELECT converted_body_hash FROM error_dumps "
            "    WHERE converted_body_hash IS NOT NULL"
            ")"
        )
        return cur.rowcount

    async def _vacuum(self) -> bool:
        """Run VACUUM; return True on success, False if locked."""
        try:
            await self._conn.execute("VACUUM")
        except Exception:
            logger.warning("VACUUM skipped — database is locked by another connection")
            return False
        return True

    async def _conditional_vacuum(self, total_deleted: int) -> bool:
        """VACUUM only when enough rows were deleted to justify the cost."""
        if total_deleted >= _VACUUM_THRESHOLD:
            return await self._vacuum()
        if total_deleted > 0:
            try:
                await self._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            except Exception as exc:
                logger.warning("WAL checkpoint after deletion failed: %s", exc)
        return False

    async def _batched_delete(
        self, table: str, where: str, params: tuple[Any, ...]
    ) -> int:
        """Delete rows matching *where* in batches, return total deleted."""
        total = 0
        while True:
            cur = await self._conn.execute(
                f"DELETE FROM {table} WHERE rowid IN ("  # noqa: S608
                f"  SELECT rowid FROM {table} WHERE {where} LIMIT ?"
                f")",
                (*params, _PRUNE_BATCH_SIZE),
            )
            await self._conn.commit()
            if cur.rowcount == 0:
                break
            total += cur.rowcount
        return total

    async def vacuum(self) -> dict[str, Any]:
        """Run VACUUM and return freed bytes."""
        size_before = self.db_path.stat().st_size
        ok = await self._vacuum()
        size_after = self.db_path.stat().st_size
        return {
            "freed_bytes": max(0, size_before - size_after),
            "vacuumed": ok,
        }

    async def cleanup_logs_by_age(self, max_age_days: int) -> dict[str, Any]:
        """Delete request_log rows older than *max_age_days*."""
        from datetime import datetime, timedelta, timezone

        cutoff = (datetime.now(timezone.utc) - timedelta(days=max_age_days)).isoformat()
        size_before = self.db_path.stat().st_size
        deleted = await self._batched_delete("request_log", "timestamp < ?", (cutoff,))
        vacuum_ok = await self._conditional_vacuum(deleted)
        size_after = self.db_path.stat().st_size

        return {
            "deleted": deleted,
            "freed_bytes": max(0, size_before - size_after),
            "size_before": size_before,
            "size_after": size_after,
            "max_age_days": max_age_days,
            "vacuum": vacuum_ok,
        }

    async def cleanup_error_dumps_by_age(self, max_age_days: int) -> dict[str, Any]:
        """Delete error_dumps and orphaned dump_bodies older than *max_age_days*."""
        from datetime import datetime, timedelta, timezone

        cutoff = (datetime.now(timezone.utc) - timedelta(days=max_age_days)).isoformat()
        size_before = self.db_path.stat().st_size
        error_dumps_deleted = await self._batched_delete(
            "error_dumps", "timestamp < ?", (cutoff,)
        )
        dump_bodies_deleted = await self._delete_orphan_bodies()
        await self._conn.commit()
        vacuum_ok = await self._conditional_vacuum(error_dumps_deleted)
        size_after = self.db_path.stat().st_size

        return {
            "error_dumps_deleted": error_dumps_deleted,
            "dump_bodies_deleted": dump_bodies_deleted,
            "freed_bytes": max(0, size_before - size_after),
            "size_before": size_before,
            "size_after": size_after,
            "max_age_days": max_age_days,
            "vacuum": vacuum_ok,
        }

    async def cleanup_by_age(self, max_age_days: int = 90) -> dict[str, Any]:
        """Delete all records older than *max_age_days*.

        Convenience method that cleans both request logs and error dumps.
        """
        from datetime import datetime, timedelta, timezone

        cutoff = (datetime.now(timezone.utc) - timedelta(days=max_age_days)).isoformat()
        size_before = self.db_path.stat().st_size
        request_log_deleted = await self._batched_delete(
            "request_log", "timestamp < ?", (cutoff,)
        )
        error_dumps_deleted = await self._batched_delete(
            "error_dumps", "timestamp < ?", (cutoff,)
        )
        dump_bodies_deleted = await self._delete_orphan_bodies()
        await self._conn.commit()
        total = request_log_deleted + error_dumps_deleted
        vacuum_ok = await self._conditional_vacuum(total)
        size_after = self.db_path.stat().st_size

        return {
            "request_log_deleted": request_log_deleted,
            "error_dumps_deleted": error_dumps_deleted,
            "dump_bodies_deleted": dump_bodies_deleted,
            "freed_bytes": max(0, size_before - size_after),
            "size_before": size_before,
            "size_after": size_after,
            "max_age_days": max_age_days,
            "vacuum": vacuum_ok,
        }

    # ------------------------------------------------------------------
    # Targeted cleanup: by date / date range / trim to cap
    # ------------------------------------------------------------------

    async def cleanup_before(
        self,
        before_iso: str,
        *,
        tables: tuple[str, ...] = ("request_log", "error_dumps", "ops_log"),
    ) -> dict[str, Any]:
        """Delete rows with ``timestamp < before_iso`` from *tables*."""
        size_before = self.db_path.stat().st_size
        result: dict[str, Any] = {}
        total = 0
        for table in tables:
            n = await self._batched_delete(table, "timestamp < ?", (before_iso,))
            result[f"{table}_deleted"] = n
            total += n
        if "error_dumps" in tables:
            result["dump_bodies_deleted"] = await self._delete_orphan_bodies()
            await self._conn.commit()
        vacuum_ok = await self._conditional_vacuum(total)
        size_after = self.db_path.stat().st_size
        result.update(
            freed_bytes=max(0, size_before - size_after),
            vacuum=vacuum_ok,
        )
        return result

    async def cleanup_range(
        self,
        start_iso: str,
        end_iso: str,
        *,
        tables: tuple[str, ...] = ("request_log", "error_dumps", "ops_log"),
    ) -> dict[str, Any]:
        """Delete rows with ``start_iso <= timestamp < end_iso`` from *tables*."""
        size_before = self.db_path.stat().st_size
        result: dict[str, Any] = {}
        total = 0
        for table in tables:
            n = await self._batched_delete(
                table, "timestamp >= ? AND timestamp < ?", (start_iso, end_iso)
            )
            result[f"{table}_deleted"] = n
            total += n
        if "error_dumps" in tables:
            result["dump_bodies_deleted"] = await self._delete_orphan_bodies()
            await self._conn.commit()
        vacuum_ok = await self._conditional_vacuum(total)
        size_after = self.db_path.stat().st_size
        result.update(
            freed_bytes=max(0, size_before - size_after),
            vacuum=vacuum_ok,
        )
        return result

    async def trim_to_cap(self) -> dict[str, Any]:
        """Trim all tables to their configured retention caps."""
        size_before = self.db_path.stat().st_size

        success_count = await self.count_success_entries()
        rl_excess = max(0, success_count - self._retention.success_max)
        rl_trimmed = 0
        if rl_excess > 0:
            while rl_excess > 0:
                batch = min(rl_excess, _PRUNE_BATCH_SIZE)
                await self._conn.execute(
                    "DELETE FROM request_log "
                    "WHERE rowid IN ("
                    "    SELECT rowid FROM request_log "
                    "    WHERE status_code < 400 "
                    "    ORDER BY timestamp ASC "
                    "    LIMIT ?"
                    ")",
                    (batch,),
                )
                await self._conn.commit()
                rl_trimmed += batch
                rl_excess -= batch

        dump_count = await self.count_error_dumps()
        dump_excess = max(0, dump_count - self._retention.dump_max)
        dumps_trimmed = 0
        if dump_excess > 0:
            await self._conn.execute(
                "DELETE FROM error_dumps "
                "WHERE rowid IN ("
                "    SELECT rowid FROM error_dumps "
                "    ORDER BY timestamp ASC "
                "    LIMIT ?"
                ")",
                (dump_excess,),
            )
            dumps_trimmed = dump_excess
            await self._delete_orphan_bodies()
            await self._conn.commit()

        ops_trimmed = 0
        for severity, cap in (
            ("info", self._retention.ops_info_max),
            ("warn", self._retention.ops_warn_max),
        ):
            where = "severity = 'info'" if severity == "info" else "severity != 'info'"
            row = await self._conn.execute_fetchone(
                f"SELECT COUNT(*) FROM ops_log WHERE {where}"  # noqa: S608
            )
            count = row[0] if row else 0
            excess = max(0, count - cap)
            if excess > 0:
                await self._conn.execute(
                    f"DELETE FROM ops_log WHERE rowid IN ("  # noqa: S608
                    f"  SELECT rowid FROM ops_log WHERE {where} "
                    f"  ORDER BY timestamp ASC LIMIT ?"
                    f")",
                    (excess,),
                )
                ops_trimmed += excess
        if ops_trimmed > 0:
            await self._conn.commit()

        total = rl_trimmed + dumps_trimmed + ops_trimmed
        vacuum_ok = await self._conditional_vacuum(total)
        size_after = self.db_path.stat().st_size

        return {
            "request_log_trimmed": rl_trimmed,
            "error_dumps_trimmed": dumps_trimmed,
            "ops_log_trimmed": ops_trimmed,
            "freed_bytes": max(0, size_before - size_after),
            "vacuum": vacuum_ok,
        }

    async def export_error_dumps(
        self, *, start: str | None = None, end: str | None = None
    ) -> bytes:
        """Export error dumps in a date range as tar.gz bytes.

        Args:
            start: ISO timestamp lower bound (inclusive). None = no lower bound.
            end: ISO timestamp upper bound (inclusive). None = no upper bound.

        Returns:
            In-memory tar.gz archive with metadata.json and bodies/<hash>.bin.
        """
        import io
        import tarfile

        where_clauses: list[str] = []
        params: list[str] = []
        if start:
            where_clauses.append("timestamp >= ?")
            params.append(start)
        if end:
            where_clauses.append("timestamp <= ?")
            params.append(end)

        where_sql = ""
        if where_clauses:
            where_sql = "WHERE " + " AND ".join(where_clauses)

        cols = (
            "id, request_log_id, timestamp, model, source_provider, "
            "target_provider, provider_name, status_code, error_phase, "
            "body_hash, response_text, upstream_url, converted_body_hash"
        )
        col_names = [
            "id",
            "request_log_id",
            "timestamp",
            "model",
            "source_provider",
            "target_provider",
            "provider_name",
            "status_code",
            "error_phase",
            "body_hash",
            "response_text",
            "upstream_url",
            "converted_body_hash",
        ]
        rows = await self._conn.execute_fetchall(
            f"SELECT {cols} FROM error_dumps {where_sql} ORDER BY timestamp DESC",
            params,
        )

        entries = [
            {k: v for k, v in zip(col_names, row) if v is not None} for row in rows
        ]

        # Collect unique body hashes
        hashes: set[str] = set()
        for e in entries:
            if e.get("body_hash"):
                hashes.add(e["body_hash"])
            if e.get("converted_body_hash"):
                hashes.add(e["converted_body_hash"])

        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tar:
            # metadata.json
            meta_bytes = json.dumps(entries, indent=2, ensure_ascii=False).encode()
            info = tarfile.TarInfo(name="metadata.json")
            info.size = len(meta_bytes)
            tar.addfile(info, io.BytesIO(meta_bytes))

            # bodies/<hash>.bin
            for h in sorted(hashes):
                row = await self._conn.execute_fetchone(
                    "SELECT data FROM dump_bodies WHERE hash = ?", (h,)
                )
                if row and row[0]:
                    data = row[0]
                    info = tarfile.TarInfo(name=f"bodies/{h}.bin")
                    info.size = len(data)
                    tar.addfile(info, io.BytesIO(data))

        return buf.getvalue()

    async def _prune_error_dumps(self) -> None:
        """Remove oldest error dumps beyond the retention cap.

        Also cleans up orphaned dump_bodies entries.
        """
        count = await self.count_error_dumps()
        excess = count - self._retention.dump_max
        if excess <= 0:
            return

        await self._conn.execute(
            "DELETE FROM error_dumps "
            "WHERE rowid IN ("
            "    SELECT rowid FROM error_dumps "
            "    ORDER BY timestamp ASC "
            "    LIMIT ?"
            ")",
            (excess,),
        )
        # Clean up orphaned bodies
        await self._conn.execute(
            "DELETE FROM dump_bodies WHERE hash NOT IN ("
            "    SELECT body_hash FROM error_dumps "
            "    WHERE body_hash IS NOT NULL"
            "    UNION "
            "    SELECT converted_body_hash FROM error_dumps "
            "    WHERE converted_body_hash IS NOT NULL"
            ")"
        )
        await self._conn.commit()

        if excess > 0 and self.on_prune is not None:
            await self.on_prune("error_dumps", excess)

    # ------------------------------------------------------------------
    # WAL checkpoint management
    # ------------------------------------------------------------------

    @property
    def _wal_path(self) -> Path:
        return self.db_path.with_suffix(".db-wal")

    def wal_size(self) -> int:
        """Return current WAL file size in bytes, 0 if absent."""
        try:
            return self._wal_path.stat().st_size
        except FileNotFoundError:
            return 0

    async def wal_checkpoint(self) -> dict[str, Any]:
        """Run a WAL checkpoint and return status."""
        wal_before = self.wal_size()
        try:
            row = await self._conn.execute_fetchone("PRAGMA wal_checkpoint(TRUNCATE)")
            busy, log_frames, checkpointed = row if row else (0, 0, 0)
        except Exception as exc:
            logger.warning("WAL checkpoint failed: %s", exc)
            return {"ok": False, "error": str(exc), "wal_before": wal_before}

        wal_after = self.wal_size()
        if busy:
            logger.warning(
                "WAL checkpoint incomplete — database busy (log=%d, checkpointed=%d)",
                log_frames,
                checkpointed,
            )
        return {
            "ok": not busy,
            "wal_before": wal_before,
            "wal_after": wal_after,
            "log_frames": log_frames,
            "checkpointed": checkpointed,
        }

    async def _periodic_wal_checkpoint(self) -> None:
        """Background task: checkpoint WAL every interval or when oversized."""
        while True:
            await asyncio.sleep(_WAL_CHECKPOINT_INTERVAL)
            try:
                wal_bytes = self.wal_size()
                if wal_bytes > self._wal_max_bytes:
                    logger.info(
                        "WAL size %d bytes exceeds cap %d, checkpointing",
                        wal_bytes,
                        self._wal_max_bytes,
                    )
                result = await self.wal_checkpoint()
                if result.get("ok") and result.get("wal_before", 0) > 0:
                    logger.debug(
                        "WAL checkpoint: %d → %d bytes",
                        result["wal_before"],
                        result.get("wal_after", 0),
                    )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning("Periodic WAL checkpoint error: %s", exc)

    def start_wal_task(self) -> None:
        """Start the periodic WAL checkpoint background task."""
        if self._wal_task is None or self._wal_task.done():
            self._wal_task = asyncio.create_task(
                self._periodic_wal_checkpoint(),
                name="persistence-wal-checkpoint",
            )

    async def stop_wal_task(self) -> None:
        """Cancel the WAL checkpoint background task."""
        if self._wal_task is not None and not self._wal_task.done():
            self._wal_task.cancel()
            try:
                await self._wal_task
            except asyncio.CancelledError:
                pass
            self._wal_task = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def close(self) -> None:
        """Commit and close the database connection."""
        await self.stop_wal_task()
        if self._conn is None:
            return
        try:
            await self._conn.commit()
        except Exception:
            pass
        try:
            await self._conn.close()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _prune(self) -> None:
        """Remove oldest successful entries beyond the retention cap.

        Error rows (status_code >= 400) are not pruned by count — they
        are bounded by age-based cleanup only.

        Deletes the oldest *excess* rows by rowid.  When the excess is
        large (e.g. first run against a bloated table), deletion is
        batched to avoid holding a long write-lock.
        """
        count = await self.count_success_entries()
        excess = count - self._retention.success_max
        if excess <= 0:
            return

        was_large = excess > _PRUNE_BATCH_SIZE
        while excess > 0:
            batch = min(excess, _PRUNE_BATCH_SIZE)
            await self._conn.execute(
                "DELETE FROM request_log "
                "WHERE rowid IN ("
                "    SELECT rowid FROM request_log "
                "    WHERE status_code < 400 "
                "    ORDER BY timestamp ASC "
                "    LIMIT ?"
                ")",
                (batch,),
            )
            await self._conn.commit()
            excess -= batch

        if was_large:
            try:
                await self._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            except Exception as exc:
                logger.warning("WAL checkpoint after prune failed: %s", exc)

        pruned = count - self._retention.success_max  # original excess
        if pruned > 0 and self.on_prune is not None:
            await self.on_prune("request_log", pruned)

    async def update_entry_profile(
        self, entry_id: str, profile_update: dict[str, Any]
    ) -> None:
        """Merge additional profile data into an existing log entry.

        Reads the current profile JSON, merges *profile_update* on top,
        and writes it back.  Used by the streaming path to write back
        stream metrics after the stream completes.

        Args:
            entry_id: The log entry ID to update.
            profile_update: Profile keys to merge.
        """
        row = await self._conn.execute_fetchone(
            "SELECT profile FROM request_log WHERE id = ?", (entry_id,)
        )
        if row is None:
            return
        existing: dict[str, Any] = {}
        if row[0]:
            try:
                existing = json.loads(row[0])
            except (json.JSONDecodeError, TypeError):
                pass
        existing.update(profile_update)
        await self._conn.execute(
            "UPDATE request_log SET profile = ? WHERE id = ?",
            (json.dumps(existing, ensure_ascii=False), entry_id),
        )
        await self._conn.commit()

    async def query_token_usage_by_day(
        self,
        *,
        days: int = 7,
        api_key_label: str | None = None,
    ) -> dict[str, Any]:
        """Aggregate token usage grouped by day.

        Args:
            days: Number of days to look back.
            api_key_label: Optional filter by API key label.

        Returns:
            Dict with ``days`` (list of per-day dicts) and ``totals``.
        """
        from datetime import datetime, timedelta, timezone

        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        sql = (
            "SELECT date(timestamp) as day, "
            "COUNT(*) as request_count, "
            "COALESCE(SUM(input_tokens), 0), "
            "COALESCE(SUM(output_tokens), 0), "
            "COALESCE(SUM(total_tokens), 0), "
            "COALESCE(SUM(cache_read_tokens), 0), "
            "COALESCE(SUM(cache_creation_tokens), 0), "
            "COALESCE(SUM(reasoning_tokens), 0) "
            "FROM request_log WHERE timestamp >= ?"
        )
        params: list[Any] = [cutoff]
        if api_key_label:
            sql += " AND api_key_label = ?"
            params.append(api_key_label)
        sql += " GROUP BY date(timestamp) ORDER BY day DESC"

        rows = await self._conn.execute_fetchall(sql, params)
        day_list = []
        totals = {
            "request_count": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "cache_read_tokens": 0,
            "cache_creation_tokens": 0,
            "reasoning_tokens": 0,
        }
        for r in rows:
            entry = {
                "day": r[0],
                "request_count": r[1],
                "input_tokens": r[2],
                "output_tokens": r[3],
                "total_tokens": r[4],
                "cache_read_tokens": r[5],
                "cache_creation_tokens": r[6],
                "reasoning_tokens": r[7],
            }
            day_list.append(entry)
            for k in totals:
                totals[k] += entry[k]
        return {"days": day_list, "totals": totals}

    async def query_rolling_24h_tokens(self) -> dict[str, int]:
        """Sum token usage over the last 24 hours."""
        from datetime import datetime, timedelta, timezone

        cutoff = (datetime.now(timezone.utc) - timedelta(hours=24)).isoformat()
        row = await self._conn.execute_fetchone(
            "SELECT COALESCE(SUM(input_tokens), 0), "
            "COALESCE(SUM(output_tokens), 0), "
            "COALESCE(SUM(cache_read_tokens), 0), "
            "COALESCE(SUM(cache_creation_tokens), 0), "
            "COALESCE(SUM(reasoning_tokens), 0) "
            "FROM request_log WHERE timestamp >= ?",
            (cutoff,),
        )
        return {
            "input_tokens_24h": row[0],
            "output_tokens_24h": row[1],
            "cache_read_tokens_24h": row[2],
            "cache_creation_tokens_24h": row[3],
            "reasoning_tokens_24h": row[4],
        }

    async def update_entry_usage(
        self,
        entry_id: str,
        input_tokens: int | None,
        output_tokens: int | None,
        total_tokens: int | None,
        *,
        cache_read_tokens: int | None = None,
        cache_creation_tokens: int | None = None,
        reasoning_tokens: int | None = None,
    ) -> None:
        """Write back token usage for an existing log entry.

        Used by the streaming path to record usage extracted from the
        final stream event.
        """
        await self._conn.execute(
            "UPDATE request_log SET input_tokens = ?, output_tokens = ?, "
            "total_tokens = ?, cache_read_tokens = ?, "
            "cache_creation_tokens = ?, reasoning_tokens = ? WHERE id = ?",
            (
                input_tokens,
                output_tokens,
                total_tokens,
                cache_read_tokens,
                cache_creation_tokens,
                reasoning_tokens,
                entry_id,
            ),
        )
        await self._conn.commit()

    @classmethod
    def _row_to_dict(cls, row: tuple[Any, ...]) -> dict[str, Any]:
        d: dict[str, Any] = {}
        for col, val in zip(cls._LOG_COLUMNS, row):
            if col == "is_stream":
                d[col] = bool(val)
            elif col == "profile":
                if val is not None:
                    try:
                        d[col] = json.loads(val)
                    except (json.JSONDecodeError, TypeError):
                        d[col] = None
                # omit if None (match old behavior for optional fields)
            elif (
                col
                in (
                    "error_detail",
                    "api_key_label",
                    "client_ip",
                    "input_tokens",
                    "output_tokens",
                    "total_tokens",
                    "cache_read_tokens",
                    "cache_creation_tokens",
                    "reasoning_tokens",
                )
                and val is None
            ):
                continue  # omit None optional fields (match old behavior)
            else:
                d[col] = val
        return d

    # ------------------------------------------------------------------
    # Legacy migration
    # ------------------------------------------------------------------

    async def _migrate_legacy(self) -> None:
        """Import data from legacy JSONL/JSON files if present."""
        migrated_anything = False

        # Migrate request log
        log_path = self._data_dir / _LEGACY_LOG
        if log_path.exists():
            entries: list[dict[str, Any]] = []
            # Read compressed backups first (oldest)
            for i in range(3, 0, -1):
                gz_path = self._data_dir / f"request_log.{i}.jsonl.gz"
                if gz_path.exists():
                    entries.extend(_read_jsonl_gz(gz_path))
                    gz_path.rename(gz_path.parent / (gz_path.name + ".migrated"))
            # Then current log
            entries.extend(_read_jsonl(log_path))
            if entries:
                await self.insert_log_entries(entries)
                logger.info(
                    "Migrated %d request log entries from legacy files",
                    len(entries),
                )
            log_path.rename(log_path.with_suffix(".migrated"))
            migrated_anything = True

        # Migrate metrics
        metrics_path = self._data_dir / _LEGACY_METRICS
        if metrics_path.exists():
            try:
                data = json.loads(metrics_path.read_text(encoding="utf-8"))
                await self.save_metrics(data)
                logger.info("Migrated metrics from legacy JSON file")
            except Exception as exc:
                logger.warning("Failed to migrate metrics: %s", exc)
            metrics_path.rename(metrics_path.with_suffix(".migrated"))
            migrated_anything = True

        if migrated_anything:
            logger.info("Legacy file migration complete")


# ------------------------------------------------------------------
# JSONL readers (used for legacy migration only)
# ------------------------------------------------------------------


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file, skipping malformed lines."""
    if not path.exists():
        return []
    entries: list[dict[str, Any]] = []
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except OSError as exc:
        logger.warning("Failed to read %s: %s", path, exc)
    return entries


def _read_jsonl_gz(path: Path) -> list[dict[str, Any]]:
    """Read a gzipped JSONL file, skipping malformed lines."""
    entries: list[dict[str, Any]] = []
    try:
        with gzip.open(path, "rt", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except (OSError, gzip.BadGzipFile) as exc:
        logger.warning("Failed to read %s: %s", path, exc)
    return entries
