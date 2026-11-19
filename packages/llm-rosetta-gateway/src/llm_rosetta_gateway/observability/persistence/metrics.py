"""Metrics snapshot storage and the rebuild flag."""

from __future__ import annotations

import json
from typing import Any

from ._common import (
    logger,
)
from ._state import _StoreState


class _MetricsMixin(_StoreState):
    """Metrics snapshot storage and the rebuild flag."""

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
