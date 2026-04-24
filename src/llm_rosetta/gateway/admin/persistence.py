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
from pathlib import Path
from typing import Any

logger = logging.getLogger("llm-rosetta-gateway")

_DB_FILENAME = "gateway.db"

# Legacy filenames for migration
_LEGACY_LOG = "request_log.jsonl"
_LEGACY_METRICS = "metrics.json"


class PersistenceManager:
    """SQLite-backed persistence for request logs and metrics.

    Args:
        data_dir: Directory for the database file (created if missing).
        max_entries: Maximum request log entries to retain.
    """

    def __init__(
        self,
        data_dir: str,
        max_entries: int = 5000,
    ) -> None:
        self._data_dir = Path(data_dir)
        self._max_entries = max_entries
        self._insert_count = 0
        self._data_dir.mkdir(parents=True, exist_ok=True)

        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._init_tables()
        self._migrate_legacy()

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
                api_key_label   TEXT
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
    ]

    def insert_log_entries(self, entries: list[dict[str, Any]]) -> None:
        """Insert request log entries, pruning oldest if over capacity."""
        if not entries:
            return
        self._conn.executemany(
            "INSERT OR IGNORE INTO request_log "
            "(id, timestamp, model, source_provider, target_provider, "
            "is_stream, status_code, duration_ms, error_detail, api_key_label) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
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
                )
                for e in entries
            ],
        )
        self._conn.commit()
        self._insert_count += len(entries)
        if self._insert_count >= 100:
            self._prune()
            self._insert_count = 0
        elif self.count_log_entries() > self._max_entries:
            self._prune()

    def query_log_entries(
        self,
        *,
        limit: int = 50,
        offset: int = 0,
        model: str | None = None,
        provider: str | None = None,
        status: str | None = None,
    ) -> tuple[list[dict[str, Any]], int]:
        """Query request log with optional filters, newest first.

        Returns:
            A ``(entries, total)`` tuple.
        """
        where_clauses: list[str] = []
        params: list[Any] = []

        if model:
            where_clauses.append("model = ?")
            params.append(model)
        if provider:
            where_clauses.append("target_provider = ?")
            params.append(provider)
        if status == "ok":
            where_clauses.append("status_code < 400")
        elif status == "error":
            where_clauses.append("status_code >= 400")

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

    def count_log_entries(self) -> int:
        """Return the total number of log entries."""
        row = self._conn.execute("SELECT COUNT(*) FROM request_log").fetchone()
        return row[0] if row else 0

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
        """Remove oldest entries beyond the retention limit."""
        self._conn.execute(
            "DELETE FROM request_log WHERE id NOT IN "
            "(SELECT id FROM request_log ORDER BY timestamp DESC LIMIT ?)",
            (self._max_entries,),
        )
        self._conn.commit()

    @classmethod
    def _row_to_dict(cls, row: tuple[Any, ...]) -> dict[str, Any]:
        d: dict[str, Any] = {}
        for col, val in zip(cls._LOG_COLUMNS, row):
            if col == "is_stream":
                d[col] = bool(val)
            elif col in ("error_detail", "api_key_label") and val is None:
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
