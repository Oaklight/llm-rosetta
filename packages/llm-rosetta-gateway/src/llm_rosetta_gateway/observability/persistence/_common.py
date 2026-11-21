"""Shared constants and helpers for the persistence package."""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from typing import Any
from urllib.parse import quote

from llm_rosetta.contracts.usage import ADDITIVE_CACHE_STANDARDS

from ..retention import RetentionPolicy

logger = logging.getLogger("llm-rosetta.observability")

_DB_FILENAME = "gateway.db"

# Legacy filenames for migration
_LEGACY_LOG = "request_log.jsonl"
_LEGACY_METRICS = "metrics.json"

# Backward-compatible aliases — values from RetentionPolicy defaults
_RP_DEFAULTS = RetentionPolicy()
DEFAULT_SUCCESS_MAX = _RP_DEFAULTS.success_max
DEFAULT_MAX_AGE_DAYS = _RP_DEFAULTS.max_age_days
DEFAULT_OPS_INFO_MAX = _RP_DEFAULTS.ops_info_max
DEFAULT_OPS_WARN_MAX = _RP_DEFAULTS.ops_warn_max

_PRUNE_BATCH_SIZE = 5000
_VACUUM_THRESHOLD = 1000

_BACKUP_DIR = "backups"
_BACKUP_MAX_KEEP = 3

# WAL checkpoint defaults
DEFAULT_WAL_MAX_BYTES = 64 * 1024 * 1024  # 64 MB
_WAL_CHECKPOINT_INTERVAL = 300  # 5 minutes

# Standards whose cache counters add to the input, derived from the declared
# semantics in observability.usage rather than restated — sorted so the SQL is
# deterministic.
_CACHE_ADDITIVE_TARGETS: tuple[str, ...] = tuple(sorted(ADDITIVE_CACHE_STANDARDS))


def _like_needle(text: str) -> str:
    """Wrap *text* as a ``LIKE`` pattern with its own wildcards escaped.

    SQLite treats ``%`` and ``_`` as wildcards inside a pattern, so a search
    for ``100%`` would otherwise match every row.  The backslash escape is
    declared by the caller's ``ESCAPE '\\'`` clause.
    """
    escaped = text.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return f"%{escaped}%"


#: The order every error-dump read uses: newest first, with the id breaking a
#: tie.  A tie-break is what makes the ``(timestamp, id)`` keyset cursor a
#: page boundary that is still the same position after a row is inserted — an
#: ``OFFSET`` on this table shifts underneath a walk that is already running.
_DUMP_ORDER = "ORDER BY timestamp DESC, id DESC"

#: The columns every error-dump read returns, and the names they unpack to.
#: One spelling, so a row and a dict cannot drift apart.
_DUMP_COLUMNS = (
    "id, request_log_id, timestamp, model, source_provider, "
    "target_provider, provider_name, status_code, error_phase, "
    "body_hash, response_text, upstream_url, converted_body_hash, "
    "tool_name_collisions"
)
_DUMP_COLUMN_NAMES = tuple(name.strip() for name in _DUMP_COLUMNS.split(","))

#: Rows per export read: the largest entry page an export holds at once.
_DUMP_EXPORT_PAGE = 500


def _dump_metadata_member(entry: Mapping[str, Any]) -> tuple[str, bytes]:
    """Return one dump's metadata member name and payload.

    A single array would have to be serialized whole before tar could name
    its size, and the store can hold tens of thousands of dumps; one entry's
    size is known as soon as the entry is.  So each dump is its own member.

    A member name is a path, and the id is text from the store: the gateway
    writes ``uuid4().hex``, but :meth:`PersistenceManager.insert_error_dump`
    is public and takes any string, and an id of ``../../outside`` would be
    extracted above the archive's root.  Percent-encoding closes that and is
    injective, so two ids cannot collapse into one member name.
    """
    name = quote(str(entry["id"]), safe="")
    payload = json.dumps(entry, indent=2, ensure_ascii=False).encode()
    return f"metadata/{name}.json", payload


class _ChunkSink:
    """A writable file object that hands back the bytes written to it.

    ``tarfile`` writes synchronously, so a member cannot be yielded to the
    socket while it is being written.  The sink collects what the archive
    wrote and the caller drains it after each member, which keeps at most
    one member's compressed output resident instead of the whole archive.
    """

    __slots__ = ("_parts", "_pos")

    def __init__(self) -> None:
        self._parts: list[bytes] = []
        self._pos = 0

    def write(self, data: bytes) -> int:
        """Append *data* and return its length."""
        self._parts.append(bytes(data))
        self._pos += len(data)
        return len(data)

    def tell(self) -> int:
        """Return the bytes written so far; ``tarfile`` asks on open."""
        return self._pos

    def flush(self) -> None:
        """Nothing is buffered outside the parts list."""

    def drain(self) -> bytes:
        """Return what was written since the last drain, and forget it."""
        chunk = b"".join(self._parts)
        self._parts.clear()
        return chunk


def _and(clause: str) -> str:
    """Prefix *clause* with ``AND``, or produce nothing when it is empty."""
    return f" AND {clause}" if clause else ""


def _where(clause: str) -> str:
    """Prefix *clause* with a space and ``WHERE``, or produce nothing."""
    return f" WHERE {clause}" if clause else ""
