"""The job store: one SQLite row per job, claimed under a lease.

Every write goes through :meth:`PersistenceManager.transaction`, so the
"record the hidden range and the job that will delete it" pair cannot be torn:
either both rows exist or neither does.  The claim is a select-then-update
inside that same lock; the update repeats its ``status = 'queued'`` predicate,
so a second process that read the same candidate loses the race instead of
running the job twice.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Sequence
from typing import Any

from ..observability.visibility import HIDDEN_TABLES, MAX_TS, MIN_TS
from .model import (
    DEFAULT_LEASE_SECONDS,
    PRIORITY_DEFAULT,
    JOB_CANCELLED,
    JOB_FAILED,
    JOB_QUEUED,
    JOB_RUNNING,
    JOB_SUCCEEDED,
    PAYLOAD_VERSION,
    Job,
    iso_after,
    now_iso,
)

_JOB_COLUMNS = (
    "id, kind, payload, lane, status, priority, created_at, started_at, "
    "finished_at, attempts, max_attempts, progress_done, progress_total, "
    "progress_unit, result, error, hidden_range_ids, dedupe_key, "
    "cancel_requested"
)


class JobStore:
    """Reads and writes the ``jobs`` table for one persistence manager."""

    def __init__(self, persistence: Any) -> None:
        self._p = persistence

    # ------------------------------------------------------------------
    # Enqueue
    # ------------------------------------------------------------------

    async def enqueue(
        self,
        kind: str,
        payload: dict[str, Any] | None = None,
        *,
        lane: str = "sqlite",
        priority: int = PRIORITY_DEFAULT,
        dedupe_key: str | None = None,
        hidden_ranges: Sequence[tuple[str, str, str]] = (),
        max_attempts: int = 3,
    ) -> Job:
        """Record a job, plus the ranges it will delete, and return it.

        *hidden_ranges* is a sequence of ``(table, start, end)``; the rows in
        those ranges stop being returned by every read as soon as this
        commits.  An empty bound means "open at that end".

        A *dedupe_key* makes the call idempotent while a job with the same key
        is queued or running: the existing job is returned and nothing new is
        inserted.
        """
        if dedupe_key:
            existing = await self._active_for_key(dedupe_key)
            if existing is not None:
                return existing

        job_id = uuid.uuid4().hex
        created = now_iso()
        normalized = [
            (table, start or MIN_TS, end or MAX_TS)
            for table, start, end in hidden_ranges
        ]
        range_ids: list[str] = []
        async with self._p.transaction() as conn:
            for table, start, end in normalized:
                range_id = uuid.uuid4().hex
                range_ids.append(range_id)
                await conn.execute(
                    "INSERT INTO hidden_ranges "
                    "(id, table_name, start_ts, end_ts, job_id, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (range_id, table, start, end, job_id, created),
                )
            stored = {"v": PAYLOAD_VERSION, **(payload or {})}
            if normalized:
                # Keep the resolved window beside the job: a clean-undo cancel
                # deletes the range rows, and resume has to hide the same rows
                # again before it deletes them.
                stored["_hide"] = [list(item) for item in normalized]
            await conn.execute(
                "INSERT INTO jobs (id, kind, payload, lane, status, priority, "
                "created_at, attempts, max_attempts, progress_done, "
                "hidden_range_ids, dedupe_key) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, 0, ?, 0, ?, ?)",
                (
                    job_id,
                    kind,
                    json.dumps(stored),
                    lane,
                    JOB_QUEUED,
                    priority,
                    created,
                    max_attempts,
                    json.dumps(range_ids),
                    dedupe_key,
                ),
            )
        for table, start, end in normalized:
            self._p.visibility.add(table, start, end)
        job = await self.get(job_id)
        assert job is not None  # just inserted
        return job

    async def _active_for_key(self, dedupe_key: str) -> Job | None:
        row = await self._p.connection.execute_fetchone(
            "SELECT id FROM jobs WHERE dedupe_key = ? "
            "AND status IN ('queued', 'running') LIMIT 1",
            (dedupe_key,),
        )
        return await self.get(row[0]) if row else None

    # ------------------------------------------------------------------
    # Claim and run
    # ------------------------------------------------------------------

    async def claim(
        self,
        lane: str,
        owner: str,
        *,
        lease_seconds: float = DEFAULT_LEASE_SECONDS,
    ) -> Job | None:
        """Take the next queued job in *lane*, or return ``None``."""
        now = now_iso()
        expires = iso_after(lease_seconds)
        async with self._p.transaction() as conn:
            row = await conn.execute_fetchone(
                "SELECT id FROM jobs WHERE status = ? AND lane = ? "
                "AND cancel_requested = 0 "
                "ORDER BY priority ASC, created_at ASC LIMIT 1",
                (JOB_QUEUED, lane),
            )
            if row is None:
                return None
            cur = await conn.execute(
                "UPDATE jobs SET status = ?, lease_owner = ?, "
                "lease_expires_at = ?, heartbeat_at = ?, "
                "started_at = COALESCE(started_at, ?), attempts = attempts + 1 "
                "WHERE id = ? AND status = ?",
                (JOB_RUNNING, owner, expires, now, now, row[0], JOB_QUEUED),
            )
            if cur.rowcount == 0:
                return None
        return await self.get(row[0])

    async def heartbeat(
        self,
        job_id: str,
        owner: str,
        *,
        lease_seconds: float = DEFAULT_LEASE_SECONDS,
    ) -> bool:
        """Extend a running job's lease; False if it is no longer ours."""
        async with self._p.transaction() as conn:
            cur = await conn.execute(
                "UPDATE jobs SET heartbeat_at = ?, lease_expires_at = ? "
                "WHERE id = ? AND lease_owner = ? AND status = ?",
                (now_iso(), iso_after(lease_seconds), job_id, owner, JOB_RUNNING),
            )
        return cur.rowcount > 0

    async def set_progress(
        self,
        job_id: str,
        done: int,
        *,
        total: int | None = None,
        unit: str | None = None,
    ) -> None:
        """Record how far a running job has got."""
        async with self._p.transaction() as conn:
            await conn.execute(
                "UPDATE jobs SET progress_done = ?, progress_total = ?, "
                "progress_unit = ? WHERE id = ?",
                (done, total, unit, job_id),
            )

    async def release(self, job_id: str, owner: str) -> bool:
        """Put a running job back in the queue (a graceful shutdown).

        The attempt the claim consumed is given back: an orderly restart is not
        a failure, and burning the retry budget on it would eventually fail a
        job that never ran.
        """
        async with self._p.transaction() as conn:
            cur = await conn.execute(
                "UPDATE jobs SET status = ?, lease_owner = NULL, "
                "lease_expires_at = NULL, attempts = MAX(attempts - 1, 0) "
                "WHERE id = ? AND lease_owner = ? AND status = ?",
                (JOB_QUEUED, job_id, owner, JOB_RUNNING),
            )
        return cur.rowcount > 0

    async def finish(
        self,
        job_id: str,
        owner: str,
        status: str,
        *,
        result: dict[str, Any] | None = None,
        error: str | None = None,
        deleted: int = 0,
    ) -> bool:
        """Close a running job.

        A job that succeeded has deleted its rows, so its ranges go with them.
        A job that was cancelled before deleting anything is a clean undo, so
        its ranges go too.  A failure keeps them: the intent was deletion, the
        rows are already invisible, and retrying finishes the job — unhiding
        would resurrect data that may be half gone.
        """
        if status not in (JOB_SUCCEEDED, JOB_FAILED, JOB_CANCELLED):
            raise ValueError(f"{status!r} is not a terminal job state")
        dropped: list[tuple[str, str, str]] = []
        async with self._p.transaction() as conn:
            # Success is a claim about the rows, so check the rows.  A handler
            # that returns normally while its own range still covers data (a
            # predicate that matched nothing, a batch that silently skipped)
            # would otherwise end with the rows visible again and the job
            # reporting success — the one promise this subsystem makes.
            if status == JOB_SUCCEEDED:
                ranges = await conn.execute_fetchall(
                    "SELECT table_name, start_ts, end_ts FROM hidden_ranges "
                    "WHERE job_id = ?",
                    (job_id,),
                )
                remaining = await self._rows_in_ranges(conn, ranges)
                if remaining:
                    status = JOB_FAILED
                    error = error or (
                        f"{remaining} hidden rows remain after the handler "
                        "reported success"
                    )
            drop_ranges = status == JOB_SUCCEEDED or (
                status == JOB_CANCELLED and deleted == 0
            )
            cur = await conn.execute(
                "UPDATE jobs SET status = ?, finished_at = ?, result = ?, "
                "error = ?, lease_owner = NULL, lease_expires_at = NULL"
                + (", hidden_range_ids = '[]'" if drop_ranges else "")
                + " WHERE id = ? AND lease_owner = ? AND status = ?",
                (
                    status,
                    now_iso(),
                    json.dumps(result) if result else None,
                    error,
                    job_id,
                    owner,
                    JOB_RUNNING,
                ),
            )
            if cur.rowcount == 0:
                return False
            if drop_ranges:
                dropped = await self._drop_ranges(conn, job_id)
        for table, start, end in dropped:
            self._p.visibility.remove(table, start, end)
        return True

    async def _rows_in_ranges(self, conn: Any, ranges: Sequence[Sequence[str]]) -> int:
        """How many rows the job's own ranges still cover.

        The table name is the one value that reaches the query text, so it is
        checked against the tables this subsystem hides rather than trusted.
        """
        total = 0
        for table, start, end in ranges:
            if table not in HIDDEN_TABLES:
                continue
            row = await conn.execute_fetchone(
                f"SELECT COUNT(*) FROM {table} "  # noqa: S608
                "WHERE timestamp >= ? AND timestamp < ?",
                (start, end),
            )
            total += row[0] if row else 0
        return total

    async def _drop_ranges(self, conn: Any, job_id: str) -> list[tuple[str, str, str]]:
        rows = await conn.execute_fetchall(
            "SELECT table_name, start_ts, end_ts FROM hidden_ranges WHERE job_id = ?",
            (job_id,),
        )
        await conn.execute("DELETE FROM hidden_ranges WHERE job_id = ?", (job_id,))
        return [(row[0], row[1], row[2]) for row in rows]

    # ------------------------------------------------------------------
    # Cancel, retry, resume, unhide
    # ------------------------------------------------------------------

    async def request_cancel(self, job_id: str) -> tuple[bool, str | None]:
        """Ask a job to stop; returns ``(found, status)``.

        A queued job is cancelled outright.  A running one is flagged, and its
        worker stops at the next checkpoint, because a delete already handed to
        SQLite cannot be recalled.
        """
        job = await self.get(job_id)
        if job is None:
            return False, None
        if job.status == JOB_QUEUED:
            dropped: list[tuple[str, str, str]] = []
            async with self._p.transaction() as conn:
                await conn.execute(
                    "UPDATE jobs SET status = ?, finished_at = ?, "
                    "cancel_requested = 1, hidden_range_ids = '[]' "
                    "WHERE id = ? AND status = ?",
                    (JOB_CANCELLED, now_iso(), job_id, JOB_QUEUED),
                )
                dropped = await self._drop_ranges(conn, job_id)
            for table, start, end in dropped:
                self._p.visibility.remove(table, start, end)
            return True, JOB_CANCELLED
        if job.status == JOB_RUNNING:
            async with self._p.transaction() as conn:
                await conn.execute(
                    "UPDATE jobs SET cancel_requested = 1 WHERE id = ?", (job_id,)
                )
            return True, JOB_RUNNING
        return True, job.status

    async def retry(self, job_id: str) -> bool:
        """Put a failed or cancelled job back in the queue, fresh."""
        return await self._requeue(job_id, reset_attempts=True)

    async def resume(self, job_id: str) -> bool:
        """Continue a cancelled job, keeping the rows it has not deleted yet."""
        return await self._requeue(job_id, reset_attempts=False)

    async def _requeue(self, job_id: str, *, reset_attempts: bool) -> bool:
        """Queue a terminal job again, re-hiding its rows if it lost them.

        A cancel before anything was deleted is a clean undo and drops the
        range rows, so resuming that job would otherwise delete rows that were
        never hidden: they would vanish on the first batch instead of leaving
        the listings first.  The window is the one recorded at enqueue, not a
        recomputed one — for a clear, recomputing would sweep in traffic that
        arrived after the operator asked for it.
        """
        restored: list[tuple[str, str, str]] = []
        async with self._p.transaction() as conn:
            # The UPDATE is the fence: a job that changed state under us must
            # not get its ranges back on the way out.
            cur = await conn.execute(
                "UPDATE jobs SET status = ?, finished_at = NULL, error = NULL, "
                "cancel_requested = 0, lease_owner = NULL, lease_expires_at = NULL"
                + (", attempts = 0" if reset_attempts else "")
                + " WHERE id = ? AND status IN ('failed', 'cancelled')",
                (JOB_QUEUED, job_id),
            )
            if cur.rowcount == 0:
                return False
            row = await conn.execute_fetchone(
                "SELECT payload FROM jobs WHERE id = ?", (job_id,)
            )
            owned = await conn.execute_fetchone(
                "SELECT COUNT(*) FROM hidden_ranges WHERE job_id = ?", (job_id,)
            )
            if not (owned and owned[0]):
                spec = (json.loads(row[0]) or {}).get("_hide") or []
                range_ids: list[str] = []
                created = now_iso()
                for table, start, end in spec:
                    range_id = uuid.uuid4().hex
                    range_ids.append(range_id)
                    await conn.execute(
                        "INSERT INTO hidden_ranges "
                        "(id, table_name, start_ts, end_ts, job_id, created_at) "
                        "VALUES (?, ?, ?, ?, ?, ?)",
                        (range_id, table, start, end, job_id, created),
                    )
                    restored.append((table, start, end))
                if range_ids:
                    await conn.execute(
                        "UPDATE jobs SET hidden_range_ids = ? WHERE id = ?",
                        (json.dumps(range_ids), job_id),
                    )
        for table, start, end in restored:
            self._p.visibility.add(table, start, end)
        return True

    async def _cancel_queued(self, job_id: str) -> int:
        """Cancel a queued job and drop its ranges; returns ranges removed.

        The same transaction as `request_cancel`'s queued branch, so unhiding
        and cancelling cannot disagree about what the job owns.
        """
        dropped: list[tuple[str, str, str]] = []
        async with self._p.transaction() as conn:
            cur = await conn.execute(
                "UPDATE jobs SET status = ?, finished_at = ?, "
                "cancel_requested = 1, hidden_range_ids = '[]' "
                "WHERE id = ? AND status = ?",
                (JOB_CANCELLED, now_iso(), job_id, JOB_QUEUED),
            )
            if cur.rowcount == 0:
                raise ValueError("cannot unhide a job that left the queue")
            dropped = await self._drop_ranges(conn, job_id)
        for table, start, end in dropped:
            self._p.visibility.remove(table, start, end)
        return len(dropped)

    async def unhide(self, job_id: str) -> int:
        """Make a job's hidden rows visible again; returns ranges removed.

        A queued job is cancelled instead: it has deleted nothing yet, and
        leaving it runnable would bring the rows back only for its first batch
        to delete them again, unannounced.  A running job is refused — the
        worker deletes by its own predicate, so the rows would come back and
        disappear again; cancel it first, then unhide the cancelled job.

        Raises:
            KeyError: no such job.
            ValueError: the job is running, already succeeded, or left the
                queue between the read and the cancel.
        """
        job = await self.get(job_id)
        if job is None:
            raise KeyError(job_id)
        if job.status == JOB_QUEUED:
            return await self._cancel_queued(job_id)
        if job.status not in (JOB_FAILED, JOB_CANCELLED):
            raise ValueError(f"cannot unhide a {job.status} job")
        dropped: list[tuple[str, str, str]] = []
        async with self._p.transaction() as conn:
            dropped = await self._drop_ranges(conn, job_id)
            await conn.execute(
                "UPDATE jobs SET hidden_range_ids = '[]' WHERE id = ?", (job_id,)
            )
        for table, start, end in dropped:
            self._p.visibility.remove(table, start, end)
        return len(dropped)

    # ------------------------------------------------------------------
    # Recovery and reads
    # ------------------------------------------------------------------

    async def sweep_orphan_ranges(self) -> int:
        """Drop ranges whose job row no longer exists, and reload the cache.

        A range outlives its job only if the job was deleted by something other
        than its own terminal transition — a hand-edited database, or a job row
        pruned while it still owned the range.  Left alone it would hide rows
        that nothing will ever delete, so startup repairs it.
        """
        async with self._p.transaction() as conn:
            rows = await conn.execute_fetchall(
                "SELECT id FROM hidden_ranges WHERE job_id NOT IN (SELECT id FROM jobs)"
            )
            if rows:
                await conn.execute(
                    "DELETE FROM hidden_ranges "
                    "WHERE job_id NOT IN (SELECT id FROM jobs)"
                )
        if rows:
            await self._p.reload_visibility()
        return len(rows)

    async def recover(
        self, *, owner: str | None = None, skip: Sequence[str] = ()
    ) -> int:
        """Return interrupted jobs to the queue.

        A lease that expired is an interrupted run.  Passing *owner* widens
        the sweep to rows leased by anyone else: a process that has just
        started cannot own a row it never claimed, so at startup those are
        interrupted too.  Without that, a crash-restart inside the 60-second
        lease leaves the job `running` — rows hidden, no worker deleting them,
        and nothing but a second restart to free it.

        *skip* leaves rows alone: the caller passes the jobs it is running
        right now, which are not interrupted even if their lease lapsed.

        A job that has already used its attempts becomes failed instead: it
        would otherwise be claimed, killed and requeued forever.
        """
        now = now_iso()
        foreign = " OR lease_owner IS NOT ?" if owner is not None else ""
        untouched = ""
        if skip:
            untouched = " AND id NOT IN (" + ", ".join("?" for _ in skip) + ")"
        params: list[Any] = [
            JOB_CANCELLED,
            JOB_FAILED,
            JOB_QUEUED,
            now,
            now,
            JOB_RUNNING,
            now,
        ]
        if owner is not None:
            params.append(owner)
        params.extend(skip)
        async with self._p.transaction() as conn:
            # A job the operator cancelled while it ran has its terminal state
            # decided already.  Requeueing it would run to completion work that
            # was stopped, and `claim` skips a flagged row, so it would sit
            # queued and unclaimable forever.  Settle it as cancelled and keep
            # its ranges: deletion may have run partially.
            cur = await conn.execute(
                "UPDATE jobs SET "
                "status = CASE "
                "WHEN cancel_requested = 1 THEN ? "
                "WHEN attempts >= max_attempts THEN ? ELSE ? END, "
                "lease_owner = NULL, lease_expires_at = NULL, "
                "error = CASE "
                "WHEN cancel_requested = 1 "
                "THEN COALESCE(error, 'cancelled while running') "
                "WHEN attempts >= max_attempts "
                "THEN COALESCE(error, 'interrupted by a restart') "
                "ELSE error END, "
                "finished_at = CASE "
                "WHEN cancel_requested = 1 THEN ? "
                "WHEN attempts >= max_attempts THEN ? ELSE NULL END "
                "WHERE status = ? "
                "AND (lease_expires_at IS NULL OR lease_expires_at < ?"
                + foreign
                + ")"
                + untouched,
                params,
            )
        return cur.rowcount

    async def get(self, job_id: str) -> Job | None:
        """Return one job, or ``None``."""
        row = await self._p.connection.execute_fetchone(
            f"SELECT {_JOB_COLUMNS} FROM jobs WHERE id = ?", (job_id,)
        )
        return _row_to_job(row) if row else None

    async def list_jobs(
        self,
        *,
        status: str | None = None,
        kind: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[Job], int]:
        """Return jobs newest first, and the total matching the filters."""
        where: list[str] = []
        params: list[Any] = []
        if status:
            where.append("status = ?")
            params.append(status)
        if kind:
            where.append("kind = ?")
            params.append(kind)
        clause = (" WHERE " + " AND ".join(where)) if where else ""
        conn = self._p.connection
        row = await conn.execute_fetchone(f"SELECT COUNT(*) FROM jobs{clause}", params)
        total = row[0] if row else 0
        rows = await conn.execute_fetchall(
            f"SELECT {_JOB_COLUMNS} FROM jobs{clause} "
            "ORDER BY created_at DESC LIMIT ? OFFSET ?",
            [*params, limit, offset],
        )
        return [_row_to_job(r) for r in rows], total

    async def active_count(self) -> int:
        """How many jobs are queued or running."""
        row = await self._p.connection.execute_fetchone(
            "SELECT COUNT(*) FROM jobs WHERE status IN ('queued', 'running')"
        )
        return row[0] if row else 0

    async def prune(self, keep: int = 200) -> int:
        """Delete old terminal jobs, never one that still owns a range."""
        async with self._p.transaction() as conn:
            cur = await conn.execute(
                "DELETE FROM jobs WHERE status IN "
                "('succeeded', 'failed', 'cancelled') "
                "AND id NOT IN (SELECT job_id FROM hidden_ranges) "
                "AND id NOT IN ("
                "  SELECT id FROM jobs WHERE status IN "
                "  ('succeeded', 'failed', 'cancelled') "
                "  ORDER BY created_at DESC LIMIT ?"
                ")",
                (keep,),
            )
        return cur.rowcount


def _row_to_job(row: Sequence[Any]) -> Job:
    payload = json.loads(row[2]) if row[2] else {}
    payload.pop("v", None)
    # The hide window is the store's, not the handler's: only `_requeue` reads
    # it, and it must not reach a handler or the admin API as a payload field.
    payload.pop("_hide", None)
    return Job(
        id=row[0],
        kind=row[1],
        payload=payload,
        lane=row[3],
        status=row[4],
        priority=row[5],
        created_at=row[6],
        started_at=row[7],
        finished_at=row[8],
        attempts=row[9],
        max_attempts=row[10],
        progress_done=row[11],
        progress_total=row[12],
        progress_unit=row[13],
        result=json.loads(row[14]) if row[14] else None,
        error=row[15],
        hidden_range_ids=json.loads(row[16]) if row[16] else [],
        dedupe_key=row[17],
        cancel_requested=bool(row[18]),
    )
