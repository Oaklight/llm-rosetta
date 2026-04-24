"""Request log for the gateway admin panel.

Delegates to SQLite persistence when available, falls back to an
in-memory ring buffer otherwise.
"""

from __future__ import annotations

import uuid
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .persistence import PersistenceManager


@dataclass(frozen=True)
class RequestLogEntry:
    """A single logged proxy request."""

    id: str
    timestamp: str  # ISO 8601
    model: str
    source_provider: str
    target_provider: str
    is_stream: bool
    status_code: int
    duration_ms: float
    error_detail: str | None = None
    api_key_label: str | None = None

    @classmethod
    def create(
        cls,
        *,
        model: str,
        source_provider: str,
        target_provider: str,
        is_stream: bool,
        status_code: int,
        duration_ms: float,
        error_detail: str | None = None,
        api_key_label: str | None = None,
    ) -> RequestLogEntry:
        """Factory with auto-generated id and timestamp."""
        return cls(
            id=uuid.uuid4().hex,
            timestamp=datetime.now(timezone.utc).isoformat(),
            model=model,
            source_provider=source_provider,
            target_provider=target_provider,
            is_stream=is_stream,
            status_code=status_code,
            duration_ms=round(duration_ms, 2),
            error_detail=error_detail,
            api_key_label=api_key_label,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dict."""
        d: dict[str, Any] = {
            "id": self.id,
            "timestamp": self.timestamp,
            "model": self.model,
            "source_provider": self.source_provider,
            "target_provider": self.target_provider,
            "is_stream": self.is_stream,
            "status_code": self.status_code,
            "duration_ms": self.duration_ms,
        }
        if self.error_detail is not None:
            d["error_detail"] = self.error_detail
        if self.api_key_label is not None:
            d["api_key_label"] = self.api_key_label
        return d


class RequestLog:
    """Proxy request log with optional SQLite persistence.

    When *persistence* is provided, all operations delegate to SQLite.
    Otherwise falls back to an in-memory :class:`collections.deque`
    ring buffer (used when no config path is available).
    """

    def __init__(
        self,
        persistence: PersistenceManager | None = None,
        max_entries: int = 500,
    ) -> None:
        self._persistence = persistence
        # Fallback in-memory storage (only used when persistence is None)
        self._entries: deque[RequestLogEntry] = deque(maxlen=max_entries)
        self._pending: list[RequestLogEntry] = []

    def add(self, entry: RequestLogEntry) -> None:
        """Record a proxy request."""
        if self._persistence is not None:
            self._persistence.insert_log_entries([entry.to_dict()])
        else:
            self._entries.append(entry)
            self._pending.append(entry)

    def get_entries(
        self,
        *,
        limit: int = 50,
        offset: int = 0,
        model: str | None = None,
        provider: str | None = None,
        status: str | None = None,
    ) -> tuple[list[dict[str, Any]], int]:
        """Return filtered entries (newest-first) and total count."""
        if self._persistence is not None:
            return self._persistence.query_log_entries(
                limit=limit,
                offset=offset,
                model=model,
                provider=provider,
                status=status,
            )

        # Fallback: in-memory filtering
        filtered: list[RequestLogEntry] = list(reversed(self._entries))
        if model:
            filtered = [e for e in filtered if e.model == model]
        if provider:
            filtered = [e for e in filtered if e.target_provider == provider]
        if status == "ok":
            filtered = [e for e in filtered if e.status_code < 400]
        elif status == "error":
            filtered = [e for e in filtered if e.status_code >= 400]
        total = len(filtered)
        page = filtered[offset : offset + limit]
        return [e.to_dict() for e in page], total

    def get_entry(self, entry_id: str) -> dict[str, Any] | None:
        """Return a single entry by id, or ``None``."""
        if self._persistence is not None:
            return self._persistence.get_log_entry(entry_id)
        for e in self._entries:
            if e.id == entry_id:
                return e.to_dict()
        return None

    def load_entries(self, entries: list[dict[str, Any]]) -> None:
        """Bulk-load entries (in-memory fallback only)."""
        for d in entries:
            try:
                entry = RequestLogEntry(**d)
                self._entries.append(entry)
            except (TypeError, KeyError):
                continue

    def pending_entries(self) -> list[dict[str, Any]]:
        """Return and clear entries added since last call.

        Only meaningful in fallback mode; returns ``[]`` when using
        SQLite persistence (entries are written immediately).
        """
        if self._persistence is not None:
            return []
        entries = [e.to_dict() for e in self._pending]
        self._pending.clear()
        return entries

    def clear(self) -> None:
        """Remove all entries."""
        if self._persistence is not None:
            self._persistence.clear_log()
        else:
            self._entries.clear()

    def __len__(self) -> int:
        if self._persistence is not None:
            return self._persistence.count_log_entries()
        return len(self._entries)
