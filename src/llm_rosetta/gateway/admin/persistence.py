"""File-based persistence for metrics and request logs.

Stores request log entries as JSONL and metrics counters as JSON,
with size-based rotation and gzip compression.  All I/O uses the
Python standard library for cross-platform compatibility.
"""

from __future__ import annotations

import gzip
import json
import logging
import os
import tempfile
from pathlib import Path

logger = logging.getLogger("llm-rosetta-gateway")

_LOG_FILENAME = "request_log.jsonl"
_METRICS_FILENAME = "metrics.json"


class PersistenceManager:
    """Manages on-disk persistence for gateway admin data.

    Args:
        data_dir: Directory for data files (created if missing).
        max_file_size: Max size in bytes for ``request_log.jsonl``
            before rotation (default 2 MB).
        max_backups: Number of rotated ``.jsonl.gz`` files to keep.
    """

    def __init__(
        self,
        data_dir: str,
        max_file_size: int = 2 * 1024 * 1024,
        max_backups: int = 3,
    ) -> None:
        self._data_dir = Path(data_dir)
        self._max_file_size = max_file_size
        self._max_backups = max_backups
        self._data_dir.mkdir(parents=True, exist_ok=True)

    @property
    def log_path(self) -> Path:
        return self._data_dir / _LOG_FILENAME

    @property
    def metrics_path(self) -> Path:
        return self._data_dir / _METRICS_FILENAME

    # ------------------------------------------------------------------
    # Request log
    # ------------------------------------------------------------------

    def append_log_entries(self, entries: list[dict]) -> None:
        """Append *entries* to the JSONL log file, rotating if needed."""
        if not entries:
            return
        self._rotate_if_needed()
        with open(self.log_path, "a", encoding="utf-8") as f:
            for entry in entries:
                f.write(json.dumps(entry, ensure_ascii=False))
                f.write("\n")

    def load_log_entries(self, max_entries: int = 500) -> list[dict]:
        """Load the most recent *max_entries* from the log file.

        Reads the current JSONL first, then compressed backups if more
        entries are needed.  Returns entries in chronological order
        (oldest first).
        """
        entries: list[dict] = []

        # Read current log file
        entries.extend(self._read_jsonl(self.log_path))

        # Read compressed backups (newest backup = .1) if we need more
        if len(entries) < max_entries:
            for i in range(1, self._max_backups + 1):
                gz_path = self._data_dir / f"request_log.{i}.jsonl.gz"
                if not gz_path.exists():
                    break
                backup_entries = self._read_jsonl_gz(gz_path)
                entries = backup_entries + entries
                if len(entries) >= max_entries:
                    break

        # Keep only the newest max_entries
        if len(entries) > max_entries:
            entries = entries[-max_entries:]
        return entries

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def save_metrics(self, data: dict) -> None:
        """Atomically write metrics counters to ``metrics.json``."""
        # Write to temp file then rename for crash safety
        fd, tmp_path = tempfile.mkstemp(
            dir=str(self._data_dir), suffix=".tmp", prefix="metrics_"
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False)
                f.write("\n")
            # os.replace is atomic on all platforms
            os.replace(tmp_path, str(self.metrics_path))
        except Exception:
            # Clean up temp file on failure
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    def load_metrics(self) -> dict | None:
        """Load metrics counters from ``metrics.json``.

        Returns ``None`` if the file is missing or corrupt.
        """
        if not self.metrics_path.exists():
            return None
        try:
            with open(self.metrics_path, encoding="utf-8") as f:
                return json.load(f)
        except Exception as exc:
            logger.warning("Failed to load metrics from %s: %s", self.metrics_path, exc)
            return None

    # ------------------------------------------------------------------
    # Rotation
    # ------------------------------------------------------------------

    def _rotate_if_needed(self) -> None:
        """Rotate ``request_log.jsonl`` if it exceeds the size limit."""
        if not self.log_path.exists():
            return
        try:
            size = self.log_path.stat().st_size
        except OSError:
            return
        if size < self._max_file_size:
            return

        # Shift existing backups: .3 -> delete, .2 -> .3, .1 -> .2
        for i in range(self._max_backups, 0, -1):
            src = self._data_dir / f"request_log.{i}.jsonl.gz"
            if i == self._max_backups:
                # Delete the oldest backup
                if src.exists():
                    src.unlink()
            else:
                dst = self._data_dir / f"request_log.{i + 1}.jsonl.gz"
                if src.exists():
                    src.rename(dst)

        # Compress current log -> .1.jsonl.gz
        gz_path = self._data_dir / "request_log.1.jsonl.gz"
        try:
            with open(self.log_path, "rb") as f_in:
                with gzip.open(gz_path, "wb") as f_out:
                    while True:
                        chunk = f_in.read(65536)
                        if not chunk:
                            break
                        f_out.write(chunk)
            # Truncate the current log
            with open(self.log_path, "w", encoding="utf-8"):
                pass
            logger.info("Rotated request log (%d bytes) -> %s", size, gz_path)
        except Exception as exc:
            logger.warning("Log rotation failed: %s", exc)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _read_jsonl(path: Path) -> list[dict]:
        """Read a JSONL file, skipping malformed lines."""
        if not path.exists():
            return []
        entries: list[dict] = []
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

    @staticmethod
    def _read_jsonl_gz(path: Path) -> list[dict]:
        """Read a gzipped JSONL file, skipping malformed lines."""
        entries: list[dict] = []
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
