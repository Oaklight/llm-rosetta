"""The ``PersistenceManager`` facade."""

from __future__ import annotations

import asyncio
import warnings
from collections.abc import Awaitable, Callable
from pathlib import Path

from ..._vendor import aiosqlite
from ..maintenance import Registry
from ..retention import RetentionPolicy, RetentionTracker
from ..visibility import VisibilityFilter
from ._common import (
    _DB_FILENAME,
    DEFAULT_OPS_INFO_MAX,
    DEFAULT_OPS_WARN_MAX,
    DEFAULT_SUCCESS_MAX,
    DEFAULT_WAL_MAX_BYTES,
)
from .backup import _BackupMixin
from .connection import _ConnectionMixin
from .dump_export import _DumpExportMixin
from .dumps import _DumpsMixin
from .legacy import _LegacyMixin
from .metrics import _MetricsMixin
from .ops_log import _OpsLogMixin
from .request_log import _RequestLogMixin
from .retention import _RetentionMixin
from .schema import _SchemaMixin
from .usage import _UsageMixin


class PersistenceManager(
    _SchemaMixin,
    _ConnectionMixin,
    _MetricsMixin,
    _RequestLogMixin,
    _OpsLogMixin,
    _DumpsMixin,
    _DumpExportMixin,
    _RetentionMixin,
    _BackupMixin,
    _UsageMixin,
    _LegacyMixin,
):
    """SQLite-backed persistence for request logs and metrics.

    The request log retains successful entries up to ``success_max``.
    Error entries (status_code >= 400) are retained without a separate
    count cap — they are bounded by age-based cleanup only.

    Error *dumps* (the ``error_dumps`` table) have their own retention
    cap controlled by ``dump_max``.

    Use the async :meth:`create` classmethod to construct instances::

        pm = await PersistenceManager.create("/var/data/myproxy")

    The synchronous ``__init__`` is retained for backward compatibility
    but the instance will not have an open database connection — call
    :meth:`_open` before using any database methods.

    Args:
        data_dir: Directory for the database file (created if missing).
        success_max: Maximum number of successful request log entries to
            retain.  Defaults to :data:`DEFAULT_SUCCESS_MAX`.
        dump_max: Maximum number of error dump entries to retain.
            Defaults to :data:`DEFAULT_DUMP_MAX` (10 000).
        max_entries: Deprecated.  When provided and ``success_max`` is
            not, used as the success cap for backward compatibility.
            Emits a :class:`DeprecationWarning`.
    """

    def __init__(
        self,
        data_dir: str,
        success_max: int | None = None,
        dump_max: int | None = None,
        *,
        max_entries: int | None = None,
        ops_info_max: int | None = None,
        ops_warn_max: int | None = None,
        ops_log_max: int | None = None,
        wal_max_bytes: int | None = None,
    ) -> None:
        if max_entries is not None:
            warnings.warn(
                "PersistenceManager(max_entries=...) is deprecated; "
                "use success_max= instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            if success_max is None:
                success_max = max_entries

        self._data_dir = Path(data_dir)
        self._retention = RetentionPolicy(
            success_max=(
                success_max if success_max is not None else DEFAULT_SUCCESS_MAX
            ),
            dump_max=dump_max if dump_max is not None else self.DEFAULT_DUMP_MAX,
            ops_info_max=(
                ops_info_max
                if ops_info_max is not None
                else (ops_log_max if ops_log_max is not None else DEFAULT_OPS_INFO_MAX)
            ),
            ops_warn_max=(
                ops_warn_max
                if ops_warn_max is not None
                else (ops_log_max if ops_log_max is not None else DEFAULT_OPS_WARN_MAX)
            ),
        )
        self._tracker = RetentionTracker()
        self._wal_max_bytes = (
            wal_max_bytes if wal_max_bytes is not None else DEFAULT_WAL_MAX_BYTES
        )
        self._wal_task: asyncio.Task[None] | None = None
        self.on_prune: Callable[[str, int], Awaitable[None]] | None = None
        #: The migration registry that ran when this store was opened, and its
        #: results — ``None`` until then, so a caller can tell "not opened yet"
        #: from "opened and nothing needed doing".
        self.maintenance: Registry | None = None
        self._data_dir.mkdir(parents=True, exist_ok=True)

        # Connection is opened by create() or _open(); asserted non-None
        # in all methods (callers must use create() or _open() first).
        self._conn: aiosqlite.Connection = None  # type: ignore[assignment]  # ty: ignore[invalid-assignment]
        #: Ranges the operator cleared.  Every read consults this, and a read
        #: pays nothing while it is empty, which is the normal state.
        self._visibility = VisibilityFilter()
        #: Serialises a multi-statement write (hide + enqueue) so no other
        #: handler's statement can land in the middle of the unit.
        self._txn_lock = asyncio.Lock()

    async def _open(self, *, backup: bool = True, migrate: bool = True) -> None:
        """Open the database connection, create tables, and run migrations.

        ``backup=False`` skips the pre-open backup.  A long-running gateway
        wants that copy of the file before the code touches it; a read-mostly
        caller (the usage CLI) would otherwise stamp a new backup into the
        data directory on every invocation.

        ``migrate=False`` opens the store **read-only**: SQLite's own read-only
        mode, no pre-open backup, no integrity check, no table creation, no
        versioned migrations, no legacy-file import, and none of the write-mode
        pragmas.  It is for a caller reading a store it does not own — SQLite
        then refuses the write rather than the code having to remember not to
        make one — and it assumes the caller has checked
        :meth:`expected_schema_version` against :meth:`read_schema_version`
        first: a store that is behind would be read with the wrong shape rather
        than migrated, and a fresh directory would have no tables at all.

        What a read-only open needs from the filesystem is SQLite's rule, not
        this code's, and it is narrower than it sounds: with the ``-wal`` and
        ``-shm`` sidecars already present, a read-only connection attaches to
        them and the **directory** may be read-only too; it needs write access
        to those sidecar *files*, and write access to the directory only when it
        has to create them.  A store with a pending ``-wal`` and no ``-shm``
        therefore cannot be read from a read-only directory at all — SQLite sees
        an empty database, so the schema reads as nothing
        (:meth:`read_schema_version` returns ``None`` rather than 0 for exactly
        that reason).
        """
        if migrate:
            if self.db_path.exists():
                if backup:
                    await self._create_backup()
                self._integrity_check()
            self._conn = await aiosqlite.connect(str(self.db_path))
            await self._conn.execute("PRAGMA journal_mode=WAL")
            await self._conn.execute("PRAGMA synchronous=NORMAL")
            await self._conn.execute("PRAGMA foreign_keys=ON")
            # Stated rather than inherited: the stock sqlite3 timeout is 5 s,
            # and a CLI cleanup or a backup writing to the same file must not
            # surface as "database is locked" because another Python build
            # picked a different default.
            await self._conn.execute("PRAGMA busy_timeout=5000")
            await self._init_tables()
            await self._run_migrations()
            await self._migrate_legacy()
            await self._load_visibility()
            return
        self._conn = await aiosqlite.connect(
            f"{self.db_path.resolve().as_uri()}?mode=ro", uri=True
        )
        await self._load_visibility()

    @classmethod
    def expected_schema_version(cls) -> int:
        """The version this build's migrations bring a store to.

        Read by a caller that must decide whether it may open a store without
        migrating it: see ``_open(..., migrate=False)``.
        """
        return max(version for version, _description, _sql in cls._MIGRATIONS)

    @classmethod
    async def create(
        cls,
        data_dir: str,
        success_max: int | None = None,
        dump_max: int | None = None,
        *,
        max_entries: int | None = None,
        ops_info_max: int | None = None,
        ops_warn_max: int | None = None,
        ops_log_max: int | None = None,
        wal_max_bytes: int | None = None,
        backup: bool = True,
        migrate: bool = True,
    ) -> PersistenceManager:
        """Async factory: create a PersistenceManager with an open connection.

        This is the preferred way to construct instances in async code.
        ``backup`` and ``migrate`` are passed to :meth:`_open`; both default to
        the gateway's behaviour, and a caller that must not write to the store
        turns both off after checking the schema itself.
        """
        instance = cls(
            data_dir,
            success_max=success_max,
            dump_max=dump_max,
            max_entries=max_entries,
            ops_info_max=ops_info_max,
            ops_warn_max=ops_warn_max,
            ops_log_max=ops_log_max,
            wal_max_bytes=wal_max_bytes,
        )
        await instance._open(backup=backup, migrate=migrate)
        return instance

    @property
    def retention(self) -> RetentionPolicy:
        """The current retention policy."""
        return self._retention

    @property
    def success_max(self) -> int:
        """Cap on retained successful request log entries."""
        return self._retention.success_max

    @success_max.setter
    def success_max(self, value: int) -> None:
        self._retention.success_max = value

    @property
    def dump_max(self) -> int:
        """Cap on retained error dump entries."""
        return self._retention.dump_max

    @dump_max.setter
    def dump_max(self, value: int) -> None:
        self._retention.dump_max = value

    @property
    def db_path(self) -> Path:
        return self._data_dir / _DB_FILENAME

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def close(self) -> None:
        """Commit and close the database connection."""
        await self.stop_wal_task()
        if self._conn is None:
            return
        try:
            await self._conn.commit()
        except Exception:
            pass
        try:
            await self._conn.close()
        except Exception:
            pass
