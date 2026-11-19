"""Retention: deletion batches, vacuum, age-based cleanup, caps."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ._common import (
    _PRUNE_BATCH_SIZE,
    _VACUUM_THRESHOLD,
    logger,
)
from ._state import _StoreState


class _RetentionMixin(_StoreState):
    """Retention: deletion batches, vacuum, age-based cleanup, caps."""

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

    async def delete_batch(
        self,
        table: str,
        where: str,
        params: Sequence[Any] = (),
        *,
        limit: int = _PRUNE_BATCH_SIZE,
    ) -> int:
        """Delete up to *limit* rows matching *where*; return how many.

        One statement, one commit.  A caller that must stay cancellable and
        report progress deletes in batches rather than holding the writer for
        the whole table, which is what makes the job envelope's checkpoints
        possible.
        """
        cur = await self._conn.execute(
            f"DELETE FROM {table} WHERE rowid IN ("  # noqa: S608
            f"  SELECT rowid FROM {table} WHERE {where} LIMIT ?"
            f")",
            (*params, limit),
        )
        await self._conn.commit()
        return cur.rowcount

    async def _batched_delete(
        self, table: str, where: str, params: tuple[Any, ...]
    ) -> int:
        """Delete rows matching *where* in batches, return total deleted."""
        total = 0
        while True:
            deleted = await self.delete_batch(table, where, params)
            if deleted == 0:
                break
            total += deleted
        return total

    async def settle_deletions(self, total_deleted: int) -> bool:
        """Vacuum or checkpoint after a batched delete.

        The same rule the single-shot cleanup paths use: a big deletion earns
        a VACUUM, a small one only a WAL truncation.  Public because a job
        handler deletes in its own loop and still owes the store this step.
        """
        return await self._conditional_vacuum(total_deleted)

    async def sweep_orphan_bodies(self) -> int:
        """Delete dump bodies no error dump references; returns how many."""
        removed = await self._delete_orphan_bodies()
        await self._conn.commit()
        return removed

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

        success_count = await self.count_success_entries(include_hidden=True)
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

        dump_count = await self.count_error_dumps(include_hidden=True)
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

    async def _prune_error_dumps(self) -> None:
        """Remove oldest error dumps beyond the retention cap.

        Also cleans up orphaned dump_bodies entries.
        """
        count = await self.count_error_dumps(include_hidden=True)
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

    async def _prune(self) -> None:
        """Remove oldest successful entries beyond the retention cap.

        Error rows (status_code >= 400) are not pruned by count — they
        are bounded by age-based cleanup only.

        Deletes the oldest *excess* rows by rowid.  When the excess is
        large (e.g. first run against a bloated table), deletion is
        batched to avoid holding a long write-lock.
        """
        count = await self.count_success_entries(include_hidden=True)
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
