"""Backup, integrity check and WAL checkpoint management."""

from __future__ import annotations

import asyncio
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ._common import (
    _BACKUP_DIR,
    _BACKUP_MAX_KEEP,
    _WAL_CHECKPOINT_INTERVAL,
    logger,
)
from ._state import _StoreState


class _BackupMixin(_StoreState):
    """Backup, integrity check and WAL checkpoint management."""

    # ------------------------------------------------------------------
    # Backup and integrity
    # ------------------------------------------------------------------

    async def _create_backup(self) -> None:
        """Create a timestamped backup of the database, keeping last N copies."""

        backup_dir = self._data_dir / _BACKUP_DIR
        backup_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        backup_path = backup_dir / f"gateway-{ts}.db"

        loop = asyncio.get_running_loop()
        try:
            await loop.run_in_executor(None, self._sync_backup, backup_path)
            size_mb = backup_path.stat().st_size / (1024 * 1024)
            logger.info("Database backup created: %s (%.1f MB)", backup_path, size_mb)
        except Exception as exc:
            logger.warning("Database backup failed: %s", exc)
            return

        existing = sorted(backup_dir.glob("gateway-*.db"))
        if len(existing) > _BACKUP_MAX_KEEP:
            for old in existing[: len(existing) - _BACKUP_MAX_KEEP]:
                try:
                    old.unlink()
                except OSError:
                    pass

    def _sync_backup(self, dest: Path) -> None:
        """Run sqlite3.backup() synchronously (called from executor)."""
        src_conn = sqlite3.connect(str(self.db_path))
        dst_conn = sqlite3.connect(str(dest))
        try:
            src_conn.backup(dst_conn)
        finally:
            dst_conn.close()
            src_conn.close()

    def _integrity_check(self) -> None:
        """Run a quick integrity check on startup."""
        try:
            conn = sqlite3.connect(str(self.db_path))
            result = conn.execute("PRAGMA quick_check").fetchone()
            conn.close()
            if result and result[0] != "ok":
                logger.warning("Database integrity issue: %s", result[0])
            else:
                logger.debug("Database integrity check: ok")
        except Exception as exc:
            logger.warning("Database integrity check failed: %s", exc)

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
        # Check immediately on startup — don't wait a full interval if WAL
        # is already huge (e.g. after ungraceful shutdown).
        wal_bytes = self.wal_size()
        if wal_bytes > self._wal_max_bytes:
            logger.info(
                "WAL size %d bytes exceeds cap %d on startup, checkpointing",
                wal_bytes,
                self._wal_max_bytes,
            )
            await self.wal_checkpoint()
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
