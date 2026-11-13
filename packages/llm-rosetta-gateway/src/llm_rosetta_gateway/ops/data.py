"""Data mutation operations for the gateway ops layer.

Covers all admin-initiated and background data cleanup, trim, vacuum,
and metrics rebuild operations.  Each writes an audit record to ops_log
via the base :meth:`OpsBase._record`.

An operation accepts a ``job`` context.  With one, the work runs in batches
between checkpoints so a background job can report progress and stop when
asked; without one, it keeps the original single-shot behaviour for the
command line and the tests.
"""

from __future__ import annotations

from typing import Any, ClassVar

from llm_rosetta_gateway.observability.ops_log import (
    EVENT_DATA_CLEARED,
    EVENT_DATA_CLEANUP,
    EVENT_DATA_REBUILT,
    EVENT_DATA_TRIMMED,
    EVENT_DATA_VACUUMED,
    EVENT_OPS_LOG_CLEARED,
    SEVERITY_INFO,
    SEVERITY_WARNING,
)

from .base import OpsBase, OpsContext


class _DataMutationMixin:
    """Shared counter-rebuild and batched-delete logic for data operations."""

    _job: Any = None

    async def _delete_where(
        self: Any, table: str, where: str, params: tuple[Any, ...] = ()
    ) -> int:
        """Delete matching rows in batches, checkpointing between them.

        The batch size is the store's; the checkpoint is what makes a job
        cancellable and what keeps its lease alive across a long delete.
        """
        p = self._ctx.persistence
        total = 0
        while True:
            if self._job is not None:
                await self._job.checkpoint()
            deleted = await p.delete_batch(table, where, params)
            if deleted == 0:
                return total
            total += deleted
            if self._job is not None:
                self._job.note_deleted(deleted)
                await self._job.progress(total, unit="rows")

    async def _rebuild_counters(self: Any) -> None:
        p = self._ctx.persistence
        m = self._ctx.metrics
        if p is None or m is None:
            return
        if self._job is not None:
            # Never scan the whole log inside the job that deleted rows: the
            # periodic flush sees the flag and queues a rebuild job, which
            # streams the scan off the event loop.
            await p.set_rebuild_flag()
            return
        await m.rebuild_from_stream(p.iter_log_rows_for_rebuild())
        await p.save_metrics(m.export_counters())


class OpsClearData(_DataMutationMixin, OpsBase):
    """Clear all entries from a single table."""

    event_type: ClassVar[str] = EVENT_DATA_CLEARED
    severity: ClassVar[str] = SEVERITY_WARNING

    __slots__ = ("_table", "_job")

    def __init__(self, ctx: OpsContext, *, table: str, job: Any = None) -> None:
        super().__init__(ctx)
        self._table = table
        self._job = job

    async def _run(self) -> dict[str, Any]:
        p = self._ctx.persistence
        if self._table == "request_log":
            rl = self._ctx.request_log
            if rl is None:
                return {}
            if self._job is None:
                count = await p.count_log_entries(include_hidden=True) if p else 0
                await rl.clear()
                return {"deleted": count}
            total = await self._delete_where("request_log", "1 = 1")
            await p.settle_deletions(total)
            return {"deleted": total}
        if p is None:
            return {}
        if self._table == "error_dumps":
            if self._job is None:
                await p.clear_error_dumps()
                return {}
            total = await self._delete_where("error_dumps", "1 = 1")
            bodies = await p.sweep_orphan_bodies()
            await p.settle_deletions(total)
            return {"error_dumps_deleted": total, "dump_bodies_deleted": bodies}
        return {}

    async def _record(self, result: Any, *, error: BaseException | None = None) -> None:
        await super()._record(result, error=error)
        if error is None and self._table == "request_log":
            await self._rebuild_counters()

    def _message(self, result: Any) -> str:
        return f"{self._table} cleared"

    def _details(self, result: Any) -> dict[str, Any]:
        return {"table": self._table, **(result or {})}


class OpsDeleteErrorDump(OpsBase):
    """Delete a single error dump by ID."""

    event_type: ClassVar[str] = EVENT_DATA_CLEARED
    severity: ClassVar[str] = SEVERITY_INFO

    __slots__ = ("_dump_id",)

    def __init__(self, ctx: OpsContext, *, dump_id: str) -> None:
        super().__init__(ctx)
        self._dump_id = dump_id

    async def _run(self) -> bool:
        p = self._ctx.persistence
        if p is None:
            return False
        return await p.delete_error_dump(self._dump_id)

    def _message(self, result: Any) -> str:
        return f"Error dump deleted: {self._dump_id}"

    def _details(self, result: Any) -> dict[str, Any]:
        return {"table": "error_dumps", "dump_id": self._dump_id, "found": bool(result)}


class OpsCleanupData(_DataMutationMixin, OpsBase):
    """Cleanup data by age, before a date, or within a date range."""

    event_type: ClassVar[str] = EVENT_DATA_CLEANUP
    severity: ClassVar[str] = SEVERITY_WARNING

    __slots__ = (
        "_mode",
        "_tables",
        "_max_age_days",
        "_before",
        "_start",
        "_end",
        "_cutoff",
        "_job",
    )

    def __init__(
        self,
        ctx: OpsContext,
        *,
        mode: str,
        tables: tuple[str, ...] = ("request_log", "error_dumps", "ops_log"),
        max_age_days: int | None = None,
        before: str | None = None,
        start: str | None = None,
        end: str | None = None,
        cutoff: str | None = None,
        job: Any = None,
    ) -> None:
        super().__init__(ctx)
        self._mode = mode
        self._tables = tables
        self._max_age_days = max_age_days
        self._before = before
        self._start = start
        self._end = end
        self._cutoff = cutoff
        self._job = job

    async def _run(self) -> dict[str, Any]:
        p = self._ctx.persistence
        if p is None:
            return {}
        if self._job is not None:
            return await self._run_batched(p)
        if self._mode == "age" and self._tables == ("request_log",):
            return await p.cleanup_logs_by_age(self._max_age_days or 90)
        elif self._mode == "age" and self._tables == ("error_dumps",):
            return await p.cleanup_error_dumps_by_age(self._max_age_days or 90)
        elif self._mode == "age" and self._tables == ("ops_log",):
            return await p.cleanup_ops_log_by_age(self._max_age_days or 90)
        elif self._mode == "age":
            # Catch-all: cleanup_by_age cleans request_log + error_dumps
            return await p.cleanup_by_age(self._max_age_days or 90)
        elif self._mode == "before":
            return await p.cleanup_before(self._before or "", tables=self._tables)
        elif self._mode == "range":
            return await p.cleanup_range(
                self._start or "", self._end or "", tables=self._tables
            )
        return {}

    async def _run_batched(self, p: Any) -> dict[str, Any]:
        """The same deletion, in batches with progress, for a background job."""
        from datetime import datetime, timedelta, timezone

        if self._mode == "age":
            days = self._max_age_days or 90
            # The enqueue computed this cutoff and hid up to it; using its
            # value keeps the hidden range and the deletion identical.
            cutoff = (
                self._cutoff
                or (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
            )
            where, params = "timestamp < ?", (cutoff,)
        elif self._mode == "before":
            where, params = "timestamp < ?", (self._before or "",)
        elif self._mode == "range":
            where, params = (
                "timestamp >= ? AND timestamp < ?",
                (self._start or "", self._end or ""),
            )
        else:
            return {}

        result: dict[str, Any] = {}
        total = 0
        for table in self._tables:
            deleted = await self._delete_where(table, where, params)
            result[f"{table}_deleted"] = deleted
            total += deleted
        if "error_dumps" in self._tables:
            result["dump_bodies_deleted"] = await p.sweep_orphan_bodies()
        result["deleted"] = total
        result["vacuum"] = await p.settle_deletions(total)
        return result

    async def _record(self, result: Any, *, error: BaseException | None = None) -> None:
        await super()._record(result, error=error)
        if error is None and "request_log" in self._tables:
            await self._rebuild_counters()

    def _message(self, result: Any) -> str:
        if self._mode == "age":
            return f"Cleanup (>{self._max_age_days}d) on {', '.join(self._tables)}"
        elif self._mode == "before":
            return f"Cleanup (before {self._before}) on {', '.join(self._tables)}"
        return f"Cleanup ({self._start}..{self._end}) on {', '.join(self._tables)}"

    def _details(self, result: Any) -> dict[str, Any]:
        d: dict[str, Any] = {"tables": list(self._tables), "mode": self._mode}
        if self._max_age_days is not None:
            d["max_age_days"] = self._max_age_days
        if self._before is not None:
            d["before"] = self._before
        if self._start is not None:
            d["start"] = self._start
        if self._end is not None:
            d["end"] = self._end
        d.update(result or {})
        return d


class OpsTrimData(_DataMutationMixin, OpsBase):
    """Trim all tables to their configured retention caps."""

    event_type: ClassVar[str] = EVENT_DATA_TRIMMED
    severity: ClassVar[str] = SEVERITY_WARNING

    __slots__ = ("_job",)

    def __init__(self, ctx: OpsContext, *, job: Any = None) -> None:
        super().__init__(ctx)
        self._job = job

    async def _run(self) -> dict[str, Any]:
        p = self._ctx.persistence
        if p is None:
            return {}
        result = await p.trim_to_cap()
        if self._job is not None:
            self._job.note_deleted(
                int(result.get("request_log_trimmed", 0))
                + int(result.get("error_dumps_trimmed", 0))
                + int(result.get("ops_log_trimmed", 0))
            )
            await self._job.progress(1, total=1, unit="pass")
        return result

    async def _record(self, result: Any, *, error: BaseException | None = None) -> None:
        await super()._record(result, error=error)
        if error is None:
            await self._rebuild_counters()

    def _message(self, result: Any) -> str:
        return "Database trimmed to retention caps"

    def _details(self, result: Any) -> dict[str, Any]:
        return dict(result or {})


class OpsVacuum(OpsBase):
    """Run VACUUM on the database to reclaim disk space."""

    event_type: ClassVar[str] = EVENT_DATA_VACUUMED

    __slots__ = ("_job",)

    def __init__(self, ctx: OpsContext, *, job: Any = None) -> None:
        super().__init__(ctx)
        self._job = job

    async def _run(self) -> dict[str, Any]:
        p = self._ctx.persistence
        if p is None:
            return {}
        result = await p.vacuum()
        if self._job is not None:
            await self._job.progress(1, total=1, unit="pass")
        return result

    def _message(self, result: Any) -> str:
        return "Database vacuumed"

    def _details(self, result: Any) -> dict[str, Any]:
        return dict(result or {})


class OpsRebuildMetrics(_DataMutationMixin, OpsBase):
    """Rebuild metrics counters from request log entries."""

    event_type: ClassVar[str] = EVENT_DATA_REBUILT

    __slots__ = ("_before", "_after", "_count", "_job")

    def __init__(self, ctx: OpsContext, *, job: Any = None) -> None:
        super().__init__(ctx)
        self._before: dict[str, Any] = {}
        self._after: dict[str, Any] = {}
        self._count: int = 0
        self._job = job

    async def _run(self) -> dict[str, Any]:
        p = self._ctx.persistence
        m = self._ctx.metrics
        if p is None or m is None:
            return {}
        self._before = m.export_counters()
        self._count = await m.rebuild_from_stream(p.iter_log_rows_for_rebuild())
        self._after = m.export_counters()
        await p.save_metrics(self._after)
        if self._job is not None:
            await self._job.progress(self._count, total=self._count, unit="rows")
        return {
            "rebuilt_from": self._count,
            "before": self._before,
            "counters": self._after,
        }

    def _message(self, result: Any) -> str:
        return f"Metrics rebuilt from {self._count} log entries"

    def _details(self, result: Any) -> dict[str, Any]:
        return {"rebuilt_from": self._count}


class OpsClearOpsLog(_DataMutationMixin, OpsBase):
    """Clear the ops log, recording the clear action itself."""

    event_type: ClassVar[str] = EVENT_OPS_LOG_CLEARED

    __slots__ = ("_job",)

    def __init__(self, ctx: OpsContext, *, job: Any = None) -> None:
        super().__init__(ctx)
        self._job = job

    async def _run(self) -> dict[str, Any]:
        if self._job is None:
            ops_log = self._ctx.ops_log
            if ops_log is None:
                return {"cleared": 0}
            count = await ops_log.clear()
            return {"cleared": count}
        p = self._ctx.persistence
        if p is None:
            return {"cleared": 0}
        total = await self._delete_where("ops_log", "1 = 1")
        await p.settle_deletions(total)
        return {"cleared": total}

    def _message(self, result: Any) -> str:
        count = (result or {}).get("cleared", 0)
        return f"Ops log cleared ({count} entries removed)"

    def _details(self, result: Any) -> dict[str, Any]:
        return {"cleared_count": (result or {}).get("cleared", 0)}


class OpsPrune(OpsBase):
    """Record an amortized prune event (prune already happened in persistence)."""

    event_type: ClassVar[str] = EVENT_DATA_TRIMMED
    source: ClassVar[str] = "persistence"

    __slots__ = ("_table", "_count")

    def __init__(self, ctx: OpsContext, *, table: str, count: int) -> None:
        super().__init__(ctx)
        self._table = table
        self._count = count

    async def _run(self) -> None:
        return None

    def _message(self, result: Any) -> str:
        return f"Auto-prune: {self._count} rows removed from {self._table}"

    def _details(self, result: Any) -> dict[str, Any]:
        return {"table": self._table, "pruned": self._count}


class OpsPeriodicCleanup(_DataMutationMixin, OpsBase):
    """Record a periodic background cleanup."""

    event_type: ClassVar[str] = EVENT_DATA_CLEANUP
    source: ClassVar[str] = "persistence"

    __slots__ = (
        "_rl_age",
        "_ol_age",
        "_rl_result",
        "_ed_result",
        "_ol_result",
        "_job",
    )

    def __init__(
        self,
        ctx: OpsContext,
        *,
        rl_age: int = 90,
        ol_age: int = 90,
        job: Any = None,
    ) -> None:
        super().__init__(ctx)
        self._rl_age = rl_age
        self._ol_age = ol_age
        self._job = job
        self._rl_result: dict[str, Any] = {}
        self._ed_result: dict[str, Any] = {}
        self._ol_result: dict[str, Any] = {}

    async def _run(self) -> dict[str, Any]:
        p = self._ctx.persistence
        if p is None:
            return {}
        self._rl_result = await p.cleanup_logs_by_age(self._rl_age)
        self._ed_result = await p.cleanup_error_dumps_by_age(self._rl_age)
        self._ol_result = await p.cleanup_ops_log_by_age(self._ol_age)
        total = (
            self._rl_result.get("deleted", 0)
            + self._ed_result.get("error_dumps_deleted", 0)
            + self._ol_result.get("deleted", 0)
        )
        return {"total": total, **self._rl_result, **self._ed_result, **self._ol_result}

    async def _record(self, result: Any, *, error: BaseException | None = None) -> None:
        total = (result or {}).get("total", 0)
        if total > 0 or error is not None:
            await super()._record(result, error=error)
            await self._rebuild_counters()

    def _message(self, result: Any) -> str:
        total = (result or {}).get("total", 0)
        return f"Periodic cleanup: {total} entries removed"

    def _details(self, result: Any) -> dict[str, Any]:
        return {
            "rl_age": self._rl_age,
            "ol_age": self._ol_age,
            "request_log_deleted": self._rl_result.get("deleted", 0),
            "error_dumps_deleted": self._ed_result.get("error_dumps_deleted", 0),
            "ops_log_deleted": self._ol_result.get("deleted", 0),
        }
