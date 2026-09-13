"""Server operations log with optional SQLite persistence.

Captures operational events (startup, shutdown, config changes, key
management) with structured metadata.  Delegates to SQLite persistence
when available, falls back to an in-memory ring buffer otherwise.

This module is framework-agnostic and can be used by any consumer.

Details schema per event type
-----------------------------
All ``details`` dicts must contain only non-sensitive metadata.
Never store raw API keys, tokens, or secrets.

- ``startup``: ``{"host", "port", "provider_count", "model_count"}``
- ``shutdown``: (no details)
- ``config_reload``: ``{"provider_count", "model_count"}``
- ``key_create``: ``{"key_id", "label"}``
- ``key_update``: ``{"key_id", "changed_fields"}``
- ``key_delete``: ``{"key_id", "label"}``
- ``key_rotate``: ``{"key_id", "label"}``
- ``health_status_change``: ``{"provider", "old_status", "new_status"}``
- ``admin_setup``: (no details)
- ``ops_log_cleared``: ``{"cleared_count"}``
"""

from __future__ import annotations

import uuid
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from llm_rosetta.observability.persistence import PersistenceManager

# -- Event type constants --------------------------------------------------

EVENT_STARTUP = "startup"
EVENT_SHUTDOWN = "shutdown"
EVENT_CONFIG_RELOAD = "config_reload"
EVENT_KEY_CREATE = "key_create"
EVENT_KEY_UPDATE = "key_update"
EVENT_KEY_DELETE = "key_delete"
EVENT_KEY_ROTATE = "key_rotate"
EVENT_HEALTH_CHANGE = "health_status_change"
EVENT_ADMIN_SETUP = "admin_setup"
EVENT_OPS_LOG_CLEARED = "ops_log_cleared"

ALL_EVENT_TYPES = [
    EVENT_STARTUP,
    EVENT_SHUTDOWN,
    EVENT_CONFIG_RELOAD,
    EVENT_KEY_CREATE,
    EVENT_KEY_UPDATE,
    EVENT_KEY_DELETE,
    EVENT_KEY_ROTATE,
    EVENT_HEALTH_CHANGE,
    EVENT_ADMIN_SETUP,
    EVENT_OPS_LOG_CLEARED,
]

# -- Severity constants ----------------------------------------------------

SEVERITY_INFO = "info"
SEVERITY_WARNING = "warning"
SEVERITY_ERROR = "error"

ALL_SEVERITIES = [SEVERITY_INFO, SEVERITY_WARNING, SEVERITY_ERROR]

# -- Source subsystem constants --------------------------------------------

SOURCE_GATEWAY = "gateway"
SOURCE_ADMIN = "admin"
SOURCE_KEYS = "keys"
SOURCE_CONFIG = "config"
SOURCE_PERSISTENCE = "persistence"

ALL_SOURCES = [
    SOURCE_GATEWAY,
    SOURCE_ADMIN,
    SOURCE_KEYS,
    SOURCE_CONFIG,
    SOURCE_PERSISTENCE,
]


@dataclass(frozen=True)
class OpsLogEntry:
    """A single server operations log entry."""

    id: str
    timestamp: str  # ISO 8601
    event_type: str
    severity: str
    message: str
    details: dict[str, Any] | None = None
    source: str | None = None

    @classmethod
    def create(
        cls,
        *,
        event_type: str,
        severity: str,
        message: str,
        details: dict[str, Any] | None = None,
        source: str | None = None,
    ) -> OpsLogEntry:
        """Factory with auto-generated id and timestamp."""
        return cls(
            id=uuid.uuid4().hex,
            timestamp=datetime.now(timezone.utc).isoformat(),
            event_type=event_type,
            severity=severity,
            message=message,
            details=details,
            source=source,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dict, omitting None-valued fields."""
        d: dict[str, Any] = {
            "id": self.id,
            "timestamp": self.timestamp,
            "event_type": self.event_type,
            "severity": self.severity,
            "message": self.message,
        }
        if self.details is not None:
            d["details"] = self.details
        if self.source is not None:
            d["source"] = self.source
        return d


class OpsLog:
    """Server operations log with optional SQLite persistence.

    When *persistence* is provided, all operations delegate to SQLite.
    Otherwise falls back to an in-memory :class:`collections.deque`
    ring buffer.
    """

    def __init__(
        self,
        persistence: PersistenceManager | None = None,
        max_entries: int = 500,
    ) -> None:
        self._persistence = persistence
        self._entries: deque[OpsLogEntry] = deque(maxlen=max_entries)

    def add(self, entry: OpsLogEntry, *, _skip_prune: bool = False) -> None:
        """Record an operational event.

        Args:
            _skip_prune: Bypass amortized pruning (used for shutdown
                events to avoid unnecessary work before close).
        """
        if self._persistence is not None:
            self._persistence.insert_ops_log_entries(
                [entry.to_dict()], _skip_prune=_skip_prune
            )
        else:
            self._entries.append(entry)

    def get_entries(
        self,
        *,
        limit: int = 50,
        offset: int = 0,
        event_type: str | None = None,
        severity: str | None = None,
        source: str | None = None,
    ) -> tuple[list[dict[str, Any]], int]:
        """Return filtered entries (newest-first) and total count."""
        if self._persistence is not None:
            return self._persistence.query_ops_log_entries(
                limit=limit,
                offset=offset,
                event_type=event_type,
                severity=severity,
                source=source,
            )

        filtered: list[OpsLogEntry] = list(reversed(self._entries))
        if event_type:
            filtered = [e for e in filtered if e.event_type == event_type]
        if severity:
            filtered = [e for e in filtered if e.severity == severity]
        if source:
            filtered = [e for e in filtered if e.source == source]
        total = len(filtered)
        page = filtered[offset : offset + limit]
        return [e.to_dict() for e in page], total

    def clear(self) -> int:
        """Remove all entries. Returns the count of cleared entries."""
        if self._persistence is not None:
            count = self._persistence.count_ops_log_entries()
            self._persistence.clear_ops_log()
        else:
            count = len(self._entries)
            self._entries.clear()
        return count

    def __len__(self) -> int:
        if self._persistence is not None:
            return self._persistence.count_ops_log_entries()
        return len(self._entries)
