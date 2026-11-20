"""Request-log rows: insert, query, count and in-place updates."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

from llm_rosetta.formats import canonical_model_type
from llm_rosetta.observability.usage import DEFAULT_MODEL_TYPE

from ._common import (
    _and,
    _where,
    logger,
)
from ._state import _StoreState


class _RequestLogMixin(_StoreState):
    """Request-log rows: insert, query, count and in-place updates."""

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
        "api_key_bucket",
        "model_type",
        "request_id",
        "ttft_ms",
        "user_agent",
        "conversation_id",
        "upstream_model",
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
            "cache_read_tokens, cache_creation_tokens, reasoning_tokens, "
            "api_key_bucket, model_type, request_id, ttft_ms, user_agent, "
            "conversation_id, upstream_model) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
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
                    e.get("api_key_bucket"),
                    # Coerced rather than left to the column default: a NULL
                    # would fall out of every model-type breakdown, and a row
                    # that no breakdown counts is a row the report cannot
                    # reconcile.
                    canonical_model_type(e.get("model_type") or DEFAULT_MODEL_TYPE),
                    e.get("request_id"),
                    # Absent on a non-stream row and on every row written
                    # before the column existed.
                    e.get("ttft_ms"),
                    e.get("user_agent"),
                    e.get("conversation_id"),
                    e.get("upstream_model"),
                )
                for e in entries
            ],
        )
        await self._conn.commit()
        if self._tracker.note_insert("request_log", len(entries)):
            await self._prune()
        elif (
            await self.count_success_entries(include_hidden=True)
            > self._retention.success_max
        ):
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
        since: str | None = None,
        until: str | None = None,
    ) -> tuple[list[dict[str, Any]], int]:
        """Query request log with optional filters, newest first.

        Args:
            provider: Provider display name (e.g. ``"Gemini"``).
            provider_type: Resolved API type for *provider* (e.g.
                ``"google"``).  Enables matching legacy rows that only
                have ``target_provider`` (the API type) without a
                ``target_provider_name`` backfill.
            since: ISO 8601 lower bound (inclusive), compared against the
                stored UTC timestamp.
            until: ISO 8601 upper bound (exclusive), so consecutive
                windows do not double-count an event on the boundary.

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
        if since:
            where_clauses.append("timestamp >= ?")
            params.append(since)
        if until:
            where_clauses.append("timestamp < ?")
            params.append(until)

        vis_sql, vis_params = self._visible("request_log", since=since, until=until)
        if vis_sql:
            where_clauses.append(vis_sql)
            params.extend(vis_params)

        where_sql = ""
        if where_clauses:
            where_sql = "WHERE " + " AND ".join(where_clauses)

        count_row = await self._conn.execute_fetchone(
            f"SELECT COUNT(*) FROM request_log {where_sql}", params
        )
        total = count_row[0] if count_row else 0

        rows = await self._conn.execute_fetchall(
            f"SELECT {', '.join(self._LOG_COLUMNS)} FROM request_log {where_sql} "
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
        vis_sql, vis_params = self._visible("request_log")
        row = await self._conn.execute_fetchone(
            f"SELECT {', '.join(self._LOG_COLUMNS)} FROM request_log "
            f"WHERE id = ?{_and(vis_sql)}",
            (entry_id, *vis_params),
        )
        if row is None:
            return None
        d = self._row_to_dict(row)
        offset_vis, offset_params = self._visible("request_log", since=d["timestamp"])
        offset_row = await self._conn.execute_fetchone(
            f"SELECT COUNT(*) FROM request_log WHERE timestamp > ?{_and(offset_vis)}",
            (d["timestamp"], *offset_params),
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

        # Both sides are reads the operator can see: linking a dump that is
        # already queued for deletion, or pointing a visible dump at a row
        # inside a cleared range, only produces a link that 404s.
        dump_sql, dump_params = self._visible("error_dumps")
        log_sql, log_params = self._visible("request_log")
        unmatched = await self._conn.execute_fetchall(
            "SELECT id, timestamp, model, status_code, source_provider, target_provider "
            "FROM error_dumps "
            "WHERE (request_log_id IS NULL OR request_log_id = '')" + _and(dump_sql),
            dump_params,
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
                    "AND abs(julianday(timestamp) - julianday(?)) * 86400 < ?"
                    + _and(log_sql)
                    + " ORDER BY abs(julianday(timestamp) - julianday(?)) "
                    "LIMIT 1",
                    (source, target, status, m, ts, window_seconds, *log_params, ts),
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
                    "AND abs(julianday(timestamp) - julianday(?)) * 86400 < ?"
                    + _and(log_sql)
                    + " ORDER BY abs(julianday(timestamp) - julianday(?)) "
                    "LIMIT 1",
                    (source, target, status, ts, window_seconds, *log_params, ts),
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
        vis_sql, vis_params = self._visible("request_log")
        rows = await self._conn.execute_fetchall(
            "SELECT DISTINCT api_key_label FROM request_log "
            "WHERE api_key_label IS NOT NULL AND api_key_label != ''"
            f"{_and(vis_sql)} ORDER BY api_key_label",
            vis_params,
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
        vis_sql, vis_params = self._visible("request_log")
        cursor = await self._conn.execute(
            "SELECT model, source_provider, target_provider, "
            "target_provider_name, is_stream, status_code, "
            "input_tokens, output_tokens, total_tokens, "
            "cache_read_tokens, cache_creation_tokens, reasoning_tokens "
            f"FROM request_log{_where(vis_sql)}",
            vis_params,
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

    async def count_log_entries(self, *, include_hidden: bool = False) -> int:
        """Return the number of log entries.

        Counts what the UI shows by default: a cleared row is gone from the
        operator's point of view the moment the range is recorded.
        ``include_hidden`` is for the physical layer — retention, trimming and
        the startup drift check, which care about the disk, not the view.
        """
        vis_sql, vis_params = (
            ("", []) if include_hidden else self._visible("request_log")
        )
        row = await self._conn.execute_fetchone(
            f"SELECT COUNT(*) FROM request_log{_where(vis_sql)}", vis_params
        )
        return row[0] if row else 0

    async def count_success_entries(self, *, include_hidden: bool = False) -> int:
        """Return the number of successful log entries (status_code < 400)."""
        vis_sql, vis_params = (
            ("", []) if include_hidden else self._visible("request_log")
        )
        row = await self._conn.execute_fetchone(
            f"SELECT COUNT(*) FROM request_log WHERE status_code < 400{_and(vis_sql)}",
            vis_params,
        )
        return row[0] if row else 0

    async def count_error_entries(self, *, include_hidden: bool = False) -> int:
        """Return the number of error log entries (status_code >= 400)."""
        vis_sql, vis_params = (
            ("", []) if include_hidden else self._visible("request_log")
        )
        row = await self._conn.execute_fetchone(
            f"SELECT COUNT(*) FROM request_log WHERE status_code >= 400{_and(vis_sql)}",
            vis_params,
        )
        return row[0] if row else 0

    async def log_activity_by_model(self, since: str) -> dict[str, dict[str, Any]]:
        """Count logged requests per model since *since*, and when it last ran.

        One grouped read rather than a query per model: the Routing panel's
        model list asks for every model at once, and a hundred small queries
        would make a tab switch feel like a scan.  The last request time is
        what orders the list — the model an operator just used belongs at the
        top — and the count is what the row prints.

        Args:
            since: ISO 8601 lower bound (inclusive).

        Returns:
            ``{model: {"requests": int, "last_seen": str}}`` for the models
            with at least one row in the window.  A model with none is
            absent, not zero.
        """
        vis_sql, vis_params = self._visible("request_log", since=since)
        rows = await self._conn.execute_fetchall(
            "SELECT model, COUNT(*), MAX(timestamp) FROM request_log "
            f"WHERE timestamp >= ?{_and(vis_sql)} GROUP BY model",
            [since, *vis_params],
        )
        return {row[0]: {"requests": int(row[1]), "last_seen": row[2]} for row in rows}

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

    async def update_entry_ttft(self, entry_id: str, ttft_ms: float) -> None:
        """Record time-to-first-token on an existing log entry.

        The row is written when the handler returns, before the response body
        runs, so a stream's first chunk arrives afterwards.

        Args:
            entry_id: The log entry ID to update.
            ttft_ms: Milliseconds from the upstream request to its first
                stream chunk.
        """
        await self._conn.execute(
            "UPDATE request_log SET ttft_ms = ? WHERE id = ?",
            (float(ttft_ms), entry_id),
        )
        await self._conn.commit()

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
            logger.debug("update_entry_profile: entry %s not found", entry_id)
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
                    "api_key_bucket",
                    "request_id",
                    "ttft_ms",
                    "conversation_id",
                    "upstream_model",
                )
                and val is None
            ):
                continue  # omit None optional fields (match old behavior)
            else:
                d[col] = val
        return d
