"""SQLite-based persistence for gateway admin data.

Stores request log entries and metrics counters in a single SQLite
database (``gateway.db``) using WAL journal mode.  Automatically
migrates legacy JSONL/JSON files on first startup.
"""

from __future__ import annotations

import gzip
import json
import logging
import sqlite3
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import Any

logger = logging.getLogger("llm-rosetta-gateway")

_DB_FILENAME = "gateway.db"

# Legacy filenames for migration
_LEGACY_LOG = "request_log.jsonl"
_LEGACY_METRICS = "metrics.json"

# Retention defaults: keep many successes for capacity planning, keep
# errors longer because they are rare and operationally valuable.
DEFAULT_SUCCESS_MAX = 50000
DEFAULT_ERROR_MAX = 10000


class PersistenceManager:
    """SQLite-backed persistence for request logs and metrics.

    The request log uses a dual-threshold retention policy: successful
    requests (status_code < 400) and error requests (status_code >= 400)
    are pruned independently.  Errors typically make up a tiny fraction
    of traffic but are the most valuable rows to keep around for
    debugging, so they get their own cap that the success rotation
    cannot evict.

    Args:
        data_dir: Directory for the database file (created if missing).
        success_max: Maximum number of successful request log entries to
            retain.  Defaults to :data:`DEFAULT_SUCCESS_MAX`.
        error_max: Maximum number of error request log entries to retain
            (status_code >= 400).  Defaults to :data:`DEFAULT_ERROR_MAX`.
        max_entries: Deprecated.  When provided and ``success_max`` is
            not, used as the success cap for backward compatibility.
            Emits a :class:`DeprecationWarning`.
    """

    def __init__(
        self,
        data_dir: str,
        success_max: int | None = None,
        error_max: int | None = None,
        *,
        max_entries: int | None = None,
    ) -> None:
        if max_entries is not None:
            warnings.warn(
                "PersistenceManager(max_entries=...) is deprecated; "
                "use success_max= (and optionally error_max=) instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            if success_max is None:
                success_max = max_entries

        self._data_dir = Path(data_dir)
        self._success_max = (
            success_max if success_max is not None else DEFAULT_SUCCESS_MAX
        )
        self._error_max = error_max if error_max is not None else DEFAULT_ERROR_MAX
        self._insert_count = 0
        self._data_dir.mkdir(parents=True, exist_ok=True)

        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._init_tables()
        self._migrate_legacy()

    @property
    def success_max(self) -> int:
        """Cap on retained successful request log entries."""
        return self._success_max

    @property
    def error_max(self) -> int:
        """Cap on retained error request log entries (status_code >= 400)."""
        return self._error_max

    @property
    def db_path(self) -> Path:
        return self._data_dir / _DB_FILENAME

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def _init_tables(self) -> None:
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS request_log (
                id              TEXT PRIMARY KEY,
                timestamp       TEXT NOT NULL,
                model           TEXT NOT NULL,
                source_provider TEXT NOT NULL,
                target_provider TEXT NOT NULL,
                is_stream       INTEGER NOT NULL,
                status_code     INTEGER NOT NULL,
                duration_ms     REAL NOT NULL,
                error_detail    TEXT,
                api_key_label   TEXT,
                target_provider_name TEXT,
                client_ip       TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_rl_timestamp
                ON request_log(timestamp DESC);
            CREATE INDEX IF NOT EXISTS idx_rl_status
                ON request_log(status_code);
            CREATE TABLE IF NOT EXISTS metrics (
                key   TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
        """)
        self._migrate_add_columns()

    def _migrate_add_columns(self) -> None:
        """Add nullable columns missing from older schema versions."""
        cursor = self._conn.execute("PRAGMA table_info(request_log)")
        columns = {row[1] for row in cursor.fetchall()}
        added = False
        for col in ("target_provider_name", "client_ip"):
            if col not in columns:
                self._conn.execute(f"ALTER TABLE request_log ADD COLUMN {col} TEXT")
                added = True
        if added:
            self._conn.commit()

    def backfill_provider_names(self, model_to_provider: Mapping[str, str]) -> int:
        """Backfill target_provider_name for old entries using the model→provider mapping.

        Only updates rows where target_provider_name is NULL and the
        model exists in the current config.

        Args:
            model_to_provider: Mapping from model name to provider name
                (e.g. ``{"argo:claude-opus-4.6": "Argo Claude"}``).

        Returns:
            Number of rows updated.
        """
        if not model_to_provider:
            return 0
        total = 0
        for model_name, provider_name in model_to_provider.items():
            cursor = self._conn.execute(
                "UPDATE request_log SET target_provider_name = ? "
                "WHERE model = ? AND target_provider_name IS NULL",
                (provider_name, model_name),
            )
            total += cursor.rowcount
        if total:
            self._conn.commit()
        return total

    # ------------------------------------------------------------------
    # Request log
    # ------------------------------------------------------------------

    _LOG_COLUMNS = [
        "id",
        "timestamp",
        "model",
        "source_provider",
        "target_provider",
        "is_stream",
        "status_code",
        "duration_ms",
        "error_detail",
        "api_key_label",
        "target_provider_name",
        "client_ip",
    ]

    def insert_log_entries(self, entries: list[dict[str, Any]]) -> None:
        """Insert request log entries, pruning oldest if over capacity."""
        if not entries:
            return
        self._conn.executemany(
            "INSERT OR IGNORE INTO request_log "
            "(id, timestamp, model, source_provider, target_provider, "
            "is_stream, status_code, duration_ms, error_detail, api_key_label, "
            "target_provider_name, client_ip) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    e["id"],
                    e["timestamp"],
                    e["model"],
                    e["source_provider"],
                    e["target_provider"],
                    int(e["is_stream"]),
                    e["status_code"],
                    e["duration_ms"],
                    e.get("error_detail"),
                    e.get("api_key_label"),
                    e.get("target_provider_name"),
                    e.get("client_ip"),
                )
                for e in entries
            ],
        )
        self._conn.commit()
        self._insert_count += len(entries)
        # Periodic prune amortizes the DELETE cost; opportunistic prune
        # bounds memory when the success cap is small.
        if self._insert_count >= 100:
            self._prune()
            self._insert_count = 0
        elif self.count_success_entries() > self._success_max or (
            self.count_error_entries() > self._error_max
        ):
            self._prune()

    def query_log_entries(
        self,
        *,
        limit: int = 50,
        offset: int = 0,
        model: str | None = None,
        provider: str | None = None,
        provider_type: str | None = None,
        status: str | None = None,
        api_key_label: str | None = None,
    ) -> tuple[list[dict[str, Any]], int]:
        """Query request log with optional filters, newest first.

        Args:
            provider: Provider display name (e.g. ``"Gemini"``).
            provider_type: Resolved API type for *provider* (e.g.
                ``"google"``).  Enables matching legacy rows that only
                have ``target_provider`` (the API type) without a
                ``target_provider_name`` backfill.

        Returns:
            A ``(entries, total)`` tuple.
        """
        where_clauses: list[str] = []
        params: list[Any] = []

        if model:
            where_clauses.append("model = ?")
            params.append(model)
        if provider:
            if provider_type and provider_type != provider:
                # Match by name, OR fall back to API type only for legacy
                # rows that have no target_provider_name (avoids cross-
                # contamination when multiple providers share a base type).
                where_clauses.append(
                    "(target_provider_name = ? OR target_provider = ? "
                    "OR (target_provider_name IS NULL AND target_provider = ?))"
                )
                params.extend([provider, provider, provider_type])
            else:
                where_clauses.append(
                    "(target_provider_name = ? OR target_provider = ?)"
                )
                params.extend([provider, provider])
        if status == "ok":
            where_clauses.append("status_code < 400")
        elif status == "error":
            where_clauses.append("status_code >= 400")
        if api_key_label:
            where_clauses.append("api_key_label = ?")
            params.append(api_key_label)

        where_sql = ""
        if where_clauses:
            where_sql = "WHERE " + " AND ".join(where_clauses)

        count_row = self._conn.execute(
            f"SELECT COUNT(*) FROM request_log {where_sql}", params
        ).fetchone()
        total = count_row[0] if count_row else 0

        rows = self._conn.execute(
            f"SELECT * FROM request_log {where_sql} "
            f"ORDER BY timestamp DESC LIMIT ? OFFSET ?",
            [*params, limit, offset],
        ).fetchall()

        entries = [self._row_to_dict(row) for row in rows]
        return entries, total

    def get_log_entry(self, entry_id: str) -> dict[str, Any] | None:
        """Return a single log entry by id, or ``None``."""
        row = self._conn.execute(
            "SELECT * FROM request_log WHERE id = ?", (entry_id,)
        ).fetchone()
        if row is None:
            return None
        return self._row_to_dict(row)

    def get_api_key_labels(self) -> list[str]:
        """Return distinct API key labels seen in request logs."""
        rows = self._conn.execute(
            "SELECT DISTINCT api_key_label FROM request_log "
            "WHERE api_key_label IS NOT NULL AND api_key_label != '' "
            "ORDER BY api_key_label"
        ).fetchall()
        return [row[0] for row in rows]

    def count_log_entries(self) -> int:
        """Return the total number of log entries."""
        row = self._conn.execute("SELECT COUNT(*) FROM request_log").fetchone()
        return row[0] if row else 0

    def count_success_entries(self) -> int:
        """Return the number of successful log entries (status_code < 400)."""
        row = self._conn.execute(
            "SELECT COUNT(*) FROM request_log WHERE status_code < 400"
        ).fetchone()
        return row[0] if row else 0

    def count_error_entries(self) -> int:
        """Return the number of error log entries (status_code >= 400)."""
        row = self._conn.execute(
            "SELECT COUNT(*) FROM request_log WHERE status_code >= 400"
        ).fetchone()
        return row[0] if row else 0

    def db_file_sizes(self) -> dict[str, int]:
        """Return on-disk byte sizes of the SQLite database files.

        Returns:
            Dict with keys ``db_bytes`` (main file), ``wal_bytes`` (WAL),
            and ``shm_bytes`` (shared memory).  Missing files report 0.
        """
        db = self.db_path
        sizes = {"db_bytes": 0, "wal_bytes": 0, "shm_bytes": 0}
        for key, suffix in (
            ("db_bytes", ""),
            ("wal_bytes", "-wal"),
            ("shm_bytes", "-shm"),
        ):
            p = db.with_name(db.name + suffix)
            try:
                sizes[key] = p.stat().st_size
            except OSError:
                sizes[key] = 0
        return sizes

    def clear_log(self) -> None:
        """Delete all request log entries."""
        self._conn.execute("DELETE FROM request_log")
        self._conn.commit()

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def save_metrics(self, data: dict[str, Any]) -> None:
        """Persist metrics counters."""
        self._conn.execute(
            "INSERT OR REPLACE INTO metrics (key, value) VALUES (?, ?)",
            ("counters", json.dumps(data, ensure_ascii=False)),
        )
        self._conn.commit()

    def load_metrics(self) -> dict[str, Any] | None:
        """Load metrics counters, or ``None`` if not yet saved."""
        row = self._conn.execute(
            "SELECT value FROM metrics WHERE key = ?", ("counters",)
        ).fetchone()
        if row is None:
            return None
        try:
            return json.loads(row[0])
        except (json.JSONDecodeError, TypeError) as exc:
            logger.warning("Failed to load metrics: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Commit and close the database connection."""
        try:
            self._conn.commit()
        except Exception:
            pass
        try:
            self._conn.close()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _prune(self) -> None:
        """Remove oldest entries beyond the per-class retention limits.

        Success and error rows are pruned independently so that rare
        error rows are not evicted by a flood of successful traffic.
        """
        self._conn.execute(
            "DELETE FROM request_log "
            "WHERE status_code < 400 AND id NOT IN ("
            "    SELECT id FROM request_log WHERE status_code < 400 "
            "    ORDER BY timestamp DESC LIMIT ?"
            ")",
            (self._success_max,),
        )
        self._conn.execute(
            "DELETE FROM request_log "
            "WHERE status_code >= 400 AND id NOT IN ("
            "    SELECT id FROM request_log WHERE status_code >= 400 "
            "    ORDER BY timestamp DESC LIMIT ?"
            ")",
            (self._error_max,),
        )
        self._conn.commit()

    @classmethod
    def _row_to_dict(cls, row: tuple[Any, ...]) -> dict[str, Any]:
        d: dict[str, Any] = {}
        for col, val in zip(cls._LOG_COLUMNS, row):
            if col == "is_stream":
                d[col] = bool(val)
            elif col in ("error_detail", "api_key_label", "client_ip") and val is None:
                continue  # omit None optional fields (match old behavior)
            else:
                d[col] = val
        return d

    # ------------------------------------------------------------------
    # Legacy migration
    # ------------------------------------------------------------------

    def _migrate_legacy(self) -> None:
        """Import data from legacy JSONL/JSON files if present."""
        migrated_anything = False

        # Migrate request log
        log_path = self._data_dir / _LEGACY_LOG
        if log_path.exists():
            entries: list[dict[str, Any]] = []
            # Read compressed backups first (oldest)
            for i in range(3, 0, -1):
                gz_path = self._data_dir / f"request_log.{i}.jsonl.gz"
                if gz_path.exists():
                    entries.extend(_read_jsonl_gz(gz_path))
                    gz_path.rename(gz_path.parent / (gz_path.name + ".migrated"))
            # Then current log
            entries.extend(_read_jsonl(log_path))
            if entries:
                self.insert_log_entries(entries)
                logger.info(
                    "Migrated %d request log entries from legacy files",
                    len(entries),
                )
            log_path.rename(log_path.with_suffix(".migrated"))
            migrated_anything = True

        # Migrate metrics
        metrics_path = self._data_dir / _LEGACY_METRICS
        if metrics_path.exists():
            try:
                data = json.loads(metrics_path.read_text(encoding="utf-8"))
                self.save_metrics(data)
                logger.info("Migrated metrics from legacy JSON file")
            except Exception as exc:
                logger.warning("Failed to migrate metrics: %s", exc)
            metrics_path.rename(metrics_path.with_suffix(".migrated"))
            migrated_anything = True

        if migrated_anything:
            logger.info("Legacy file migration complete")


# ------------------------------------------------------------------
# JSONL readers (used for legacy migration only)
# ------------------------------------------------------------------


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file, skipping malformed lines."""
    if not path.exists():
        return []
    entries: list[dict[str, Any]] = []
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except OSError as exc:
        logger.warning("Failed to read %s: %s", path, exc)
    return entries


def _read_jsonl_gz(path: Path) -> list[dict[str, Any]]:
    """Read a gzipped JSONL file, skipping malformed lines."""
    entries: list[dict[str, Any]] = []
    try:
        with gzip.open(path, "rt", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except (OSError, gzip.BadGzipFile) as exc:
        logger.warning("Failed to read %s: %s", path, exc)
    return entries
