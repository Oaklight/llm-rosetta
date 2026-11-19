"""Ops-log rows: insert, query, count and pruning."""

from __future__ import annotations

import json
from typing import Any

from ._common import (
    _and,
    _where,
)
from ._state import _StoreState


class _OpsLogMixin(_StoreState):
    """Ops-log rows: insert, query, count and pruning."""

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

        vis_sql, vis_params = self._visible("ops_log")
        if vis_sql:
            where.append(vis_sql)
            params.extend(vis_params)

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

    async def count_ops_log_entries(self, *, include_hidden: bool = False) -> int:
        """Return the total number of ops log entries."""
        vis_sql, vis_params = ("", []) if include_hidden else self._visible("ops_log")
        row = await self._conn.execute_fetchone(
            f"SELECT COUNT(*) FROM ops_log{_where(vis_sql)}", vis_params
        )
        return row[0] if row else 0

    async def count_ops_info_entries(self, *, include_hidden: bool = False) -> int:
        """Return ops log entries with severity='info'."""
        vis_sql, vis_params = ("", []) if include_hidden else self._visible("ops_log")
        row = await self._conn.execute_fetchone(
            f"SELECT COUNT(*) FROM ops_log WHERE severity = 'info'{_and(vis_sql)}",
            vis_params,
        )
        return row[0] if row else 0

    async def count_ops_warn_entries(self, *, include_hidden: bool = False) -> int:
        """Return ops log entries with severity in ('warning', 'error')."""
        vis_sql, vis_params = ("", []) if include_hidden else self._visible("ops_log")
        row = await self._conn.execute_fetchone(
            f"SELECT COUNT(*) FROM ops_log WHERE severity != 'info'{_and(vis_sql)}",
            vis_params,
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

        info_excess = (
            await self.count_ops_info_entries(include_hidden=True)
            - self._retention.ops_info_max
        )
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

        warn_excess = (
            await self.count_ops_warn_entries(include_hidden=True)
            - self._retention.ops_warn_max
        )
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
