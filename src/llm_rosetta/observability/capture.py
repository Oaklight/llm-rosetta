"""In-memory content capture for request/response inspection.

Provides :class:`CaptureState` — a lightweight ephemeral store that
captures the full three-stage request flow (original request →
converted body → upstream response) so operators can inspect exactly
what the gateway sends and receives.

Design constraints:
- No SQLite / file I/O — purely in-memory.
- Results are lost on restart by design.
- Follows the same enable-for-N-requests pattern as :class:`ProfilerState`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any

import threading


@dataclass
class CapturedRequest:
    """One captured request/response cycle."""

    original_request: dict[str, Any] | None = None
    converted_body: dict[str, Any] | None = None
    upstream_response: dict[str, Any] | list[dict[str, Any]] | None = None
    request_id: str = ""
    model: str = ""
    source_provider: str = ""
    target_provider: str = ""
    is_stream: bool = False
    status_code: int | None = None
    timestamp: str = ""

    def __post_init__(self) -> None:
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def summary(self) -> dict[str, Any]:
        """Return a lightweight summary without bodies."""
        return {
            "request_id": self.request_id,
            "model": self.model,
            "source_provider": self.source_provider,
            "target_provider": self.target_provider,
            "is_stream": self.is_stream,
            "status_code": self.status_code,
            "timestamp": self.timestamp,
        }


class CaptureState:
    """In-memory content capture controller.

    Thread-safe.  Call :meth:`should_capture` from the request path to
    atomically claim a capture slot; if it returns ``True``, the caller
    must later call :meth:`record` with the populated
    :class:`CapturedRequest`.
    """

    MAX_RESULTS = 20

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._enabled = False
        self._remaining = 0
        self._results: list[CapturedRequest] = []

    # -- Control ------------------------------------------------------------

    def enable(self, requests: int = 5) -> dict[str, Any]:
        with self._lock:
            self._enabled = True
            self._remaining = max(1, requests)
            return self._status_unlocked()

    def disable(self) -> dict[str, Any]:
        with self._lock:
            self._enabled = False
            self._remaining = 0
            return self._status_unlocked()

    # -- Request path -------------------------------------------------------

    def should_capture(self) -> bool:
        """Atomically claim a capture slot.  Returns True if this
        request should be captured."""
        with self._lock:
            if not self._enabled or self._remaining <= 0:
                return False
            self._remaining -= 1
            if self._remaining <= 0:
                self._enabled = False
            return True

    def record(self, captured: CapturedRequest) -> None:
        """Store a completed capture."""
        with self._lock:
            self._results.append(captured)
            if len(self._results) > self.MAX_RESULTS:
                self._results = self._results[-self.MAX_RESULTS :]

    # -- Query --------------------------------------------------------------

    def _status_unlocked(self) -> dict[str, Any]:
        """Return status dict. Caller must already hold ``_lock``."""
        return {
            "enabled": self._enabled,
            "remaining": self._remaining,
            "captured": len(self._results),
        }

    def status(self) -> dict[str, Any]:
        with self._lock:
            return self._status_unlocked()

    @property
    def results(self) -> list[CapturedRequest]:
        return list(self._results)

    def clear_results(self) -> None:
        with self._lock:
            self._results.clear()
