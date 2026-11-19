"""Import of the legacy JSONL/JSON stores."""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any

from ._common import (
    _LEGACY_LOG,
    _LEGACY_METRICS,
    logger,
)
from ._state import _StoreState


class _LegacyMixin(_StoreState):
    """Import of the legacy JSONL/JSON stores."""

    # ------------------------------------------------------------------
    # Legacy migration
    # ------------------------------------------------------------------

    async def _migrate_legacy(self) -> None:
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
                await self.insert_log_entries(entries)
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
                await self.save_metrics(data)
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


# ----------------------------------------------------------------------
# The request-log migrations, as maintenance steps
# ----------------------------------------------------------------------


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
