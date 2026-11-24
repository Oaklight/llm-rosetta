"""Metrics snapshot storage and the rebuild flag."""

from __future__ import annotations

import json
import uuid
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
        """Stamp a fresh rebuild request.

        Each call writes a new token, not a constant.  A rebuild that started
        earlier clears only the token it captured, so a request made while it
        scans stays behind for the next pass instead of being lost.
        """
        await self._conn.execute(
            "INSERT OR REPLACE INTO metrics (key, value) VALUES (?, ?)",
            ("rebuild_needed", uuid.uuid4().hex),
        )
        await self._conn.commit()

    async def rebuild_flag_token(self) -> str | None:
        """The pending rebuild request's token, or ``None``."""
        row = await self._conn.execute_fetchone(
            "SELECT value FROM metrics WHERE key = ?", ("rebuild_needed",)
        )
        return row[0] if row is not None else None

    async def clear_rebuild_flag(self, token: str) -> bool:
        """Clear the request only while it still holds *token*.

        A newer request overwrote the token, so this returns ``False`` and
        leaves the flag for the next rebuild.  The caller has already scanned
        the log for the older request.
        """
        row = await self._conn.execute_fetchone(
            "DELETE FROM metrics WHERE key = ? AND value = ? RETURNING value",
            ("rebuild_needed", token),
        )
        if row is not None:
            await self._conn.commit()
            return True
        return False

    async def check_and_clear_rebuild_flag(self) -> bool:
        """Check if a rebuild was requested; atomically clear the flag if set.

        Startup uses this before a worker exists, so there is no concurrent
        rebuild whose token has to survive.  The periodic flush peeks instead
        and lets the rebuild job clear its own token.
        """
        row = await self._conn.execute_fetchone(
            "DELETE FROM metrics WHERE key = ? RETURNING value",
            ("rebuild_needed",),
        )
        if row is not None:
            await self._conn.commit()
            return True
        return False
