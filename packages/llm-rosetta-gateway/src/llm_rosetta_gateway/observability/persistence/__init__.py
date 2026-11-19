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

from ._common import (  # noqa: F401  (private names stay importable from here)
    _BACKUP_DIR,
    _BACKUP_MAX_KEEP,
    _CACHE_ADDITIVE_TARGETS,
    _DB_FILENAME,
    _DUMP_COLUMN_NAMES,
    _DUMP_COLUMNS,
    _DUMP_EXPORT_PAGE,
    _DUMP_ORDER,
    _LEGACY_LOG,
    _LEGACY_METRICS,
    _PRUNE_BATCH_SIZE,
    _RP_DEFAULTS,
    _VACUUM_THRESHOLD,
    _WAL_CHECKPOINT_INTERVAL,
    DEFAULT_MAX_AGE_DAYS,
    DEFAULT_OPS_INFO_MAX,
    DEFAULT_OPS_WARN_MAX,
    DEFAULT_SUCCESS_MAX,
    DEFAULT_WAL_MAX_BYTES,
    _and,
    _ChunkSink,
    _dump_metadata_member,
    _like_needle,
    _where,
    logger,
)
from .manager import PersistenceManager

__all__ = [
    "PersistenceManager",
    "DEFAULT_SUCCESS_MAX",
    "DEFAULT_MAX_AGE_DAYS",
    "DEFAULT_OPS_INFO_MAX",
    "DEFAULT_OPS_WARN_MAX",
    "DEFAULT_WAL_MAX_BYTES",
]
