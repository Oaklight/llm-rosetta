"""Ring-buffer request log for the gateway admin panel."""

from __future__ import annotations

import uuid
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone


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
        )

    def to_dict(self) -> dict:
        """Return a JSON-serializable dict."""
        d: dict = {
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
        return d


class RequestLog:
    """Fixed-size ring buffer of recent proxy requests.

    Uses :class:`collections.deque` with *maxlen* to automatically
    evict the oldest entries when capacity is exceeded.
    """

    def __init__(self, max_entries: int = 500) -> None:
        self._entries: deque[RequestLogEntry] = deque(maxlen=max_entries)
        self._max_entries = max_entries
        self._pending: list[RequestLogEntry] = []

    def add(self, entry: RequestLogEntry) -> None:
        """Append *entry* to the log (oldest entry evicted if full)."""
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
    ) -> tuple[list[dict], int]:
        """Return filtered entries (newest-first) and total count.

        Args:
            limit: Max entries to return.
            offset: Number of entries to skip.
            model: Filter by model name.
            provider: Filter by target provider.
            status: Filter by status category: ``"ok"`` (2xx/3xx) or
                ``"error"`` (4xx/5xx).

        Returns:
            A ``(entries, total)`` tuple where *entries* is a list of
            dicts and *total* is the filtered count before pagination.
        """
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

    def get_entry(self, entry_id: str) -> dict | None:
        """Return a single entry by id, or ``None``."""
        for e in self._entries:
            if e.id == entry_id:
                return e.to_dict()
        return None

    def load_entries(self, entries: list[dict]) -> None:
        """Bulk-load entries from persistence (oldest first)."""
        for d in entries:
            try:
                entry = RequestLogEntry(**d)
                self._entries.append(entry)
            except (TypeError, KeyError):
                continue  # skip malformed entries

    def pending_entries(self) -> list[dict]:
        """Return and clear entries added since last call.

        Used by the persistence flush loop to get new entries
        without re-writing the entire log.
        """
        entries = [e.to_dict() for e in self._pending]
        self._pending.clear()
        return entries

    def clear(self) -> None:
        """Remove all entries."""
        self._entries.clear()

    def __len__(self) -> int:
        return len(self._entries)
