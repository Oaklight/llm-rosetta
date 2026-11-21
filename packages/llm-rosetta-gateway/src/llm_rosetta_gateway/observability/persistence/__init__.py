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
from ..._vendor import aiosqlite  # noqa: F401
from ..maintenance import Policy, Registry, Step, StepContext  # noqa: F401
from ..retention import RetentionPolicy, RetentionTracker  # noqa: F401
from ..visibility import MAX_TS, MIN_TS, VisibilityFilter  # noqa: F401
from .legacy import _read_jsonl, _read_jsonl_gz  # noqa: F401
from .manager import PersistenceManager
from .schema import (  # noqa: F401
    _adopt_pre_registry_schema,
    _apply_migration,
    _migration_steps,
    _recorded_version,
)
from llm_rosetta.contracts.usage import (  # noqa: F401, E402
    ADDITIVE_CACHE_STANDARDS,
    DEFAULT_MODEL_TYPE,
    LEDGER_COLUMNS,
    USAGE_WINDOWS,
)

# The module this package replaced had no ``__all__``; the names a consumer
# could import from it (and a star-import used to yield) stay importable.
__all__ = [
    "ADDITIVE_CACHE_STANDARDS",
    "DEFAULT_MODEL_TYPE",
    "LEDGER_COLUMNS",
    "MAX_TS",
    "MIN_TS",
    "USAGE_WINDOWS",
    "Policy",
    "Registry",
    "RetentionPolicy",
    "RetentionTracker",
    "Step",
    "StepContext",
    "VisibilityFilter",
    "PersistenceManager",
    "DEFAULT_SUCCESS_MAX",
    "DEFAULT_MAX_AGE_DAYS",
    "DEFAULT_OPS_INFO_MAX",
    "DEFAULT_OPS_WARN_MAX",
    "DEFAULT_WAL_MAX_BYTES",
]
