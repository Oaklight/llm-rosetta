"""The shape a persistence mixin expects of the final ``PersistenceManager``.

The mixins are split by responsibility but share one object: each reads state
that ``PersistenceManager.__init__`` sets and calls methods another mixin
defines.  Those names exist only for the type checker (the block below never
runs), so the runtime class layout is unchanged.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..._vendor import aiosqlite
    from ..retention import RetentionPolicy, RetentionTracker
    from ..visibility import VisibilityFilter


class _StoreState:
    """Base of every persistence mixin; declares the shared surface."""

    if TYPE_CHECKING:
        # State set by ``PersistenceManager.__init__``.
        _data_dir: Path
        _retention: RetentionPolicy
        _tracker: RetentionTracker
        _wal_max_bytes: int
        _conn: aiosqlite.Connection
        _visibility: VisibilityFilter
        _txn_lock: asyncio.Lock
        on_prune: Callable[[str, int], Awaitable[None]] | None

        @property
        def db_path(self) -> Path: ...

        # Methods one mixin defines and another calls.
        def _visible(
            self,
            table: str,
            *,
            alias: str = "",
            since: str | None = None,
            until: str | None = None,
        ) -> tuple[str, list[str]]: ...

        def _dump_filter_sql(
            self,
            *,
            model: str | None = None,
            model_search: str | None = None,
            error_phase: str | None = None,
            provider: str | None = None,
            status_min: int | None = None,
            status_max: int | None = None,
            start: str | None = None,
            end: str | None = None,
            q: str | None = None,
            ids: Sequence[str] | None = None,
            before: tuple[str, str] | None = None,
        ) -> tuple[str, list[Any]]: ...

        async def save_metrics(self, data: dict[str, Any]) -> None: ...
        async def insert_log_entries(self, entries: list[dict[str, Any]]) -> None: ...
        async def count_error_dumps(self, *, include_hidden: bool = False) -> int: ...
        async def count_success_entries(
            self, *, include_hidden: bool = False
        ) -> int: ...
        async def _delete_orphan_bodies(self) -> int: ...
        async def _prune_error_dumps(self) -> None: ...
        async def _prune(self) -> None: ...
        async def _conditional_vacuum(self, total_deleted: int) -> bool: ...
        async def _batched_delete(
            self, table: str, where: str, params: tuple[Any, ...]
        ) -> int: ...
