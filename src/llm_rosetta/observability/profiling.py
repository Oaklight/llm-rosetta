"""On-demand deep profiling state management.

Provides :class:`ProfilerState` for managing pyinstrument profiling
sessions.  The data layer is framework-agnostic — route handlers that
wire ``ProfilerState`` into a web framework live in the consumer
(e.g. ``gateway/admin/routes/profiling.py``).

This module is framework-agnostic and can be used by any consumer
(the llm-rosetta gateway, argo-proxy, or standalone scripts).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


class ProfilerState:
    """Manages on-demand per-request pyinstrument profiling sessions.

    Each profiled request gets its own
    :class:`~llm_rosetta.profiling.DeepProfiler` instance to avoid
    cross-request contamination on the async event loop.

    Attributes:
        enabled: Whether profiling is currently active.
        remaining: Number of requests left to profile.
        results: Collected profiling results (capped at *max_results*).
    """

    def __init__(self, *, max_results: int = 20) -> None:
        self.enabled: bool = False
        self.remaining: int = 0
        self.results: list[dict[str, Any]] = []
        self._max_results = max_results

    def enable(self, requests: int = 5) -> dict[str, Any]:
        """Enable profiling for the next *requests* requests.

        Args:
            requests: Number of requests to profile.

        Returns:
            Current status dict.
        """
        self.enabled = True
        self.remaining = max(1, requests)
        return self.status()

    def disable(self) -> dict[str, Any]:
        """Manually disable profiling.

        Returns:
            Current status dict.
        """
        self.enabled = False
        self.remaining = 0
        return self.status()

    def should_profile(self) -> bool:
        """Check and consume one profiling slot.

        Returns ``True`` if the current request should be profiled
        (and decrements the remaining counter).  Auto-disables when
        the counter reaches zero.
        """
        if not self.enabled or self.remaining <= 0:
            return False
        self.remaining -= 1
        if self.remaining <= 0:
            self.enabled = False
        return True

    def create_profiler(self) -> Any:
        """Create a new per-request DeepProfiler instance.

        Returns:
            A :class:`~llm_rosetta.profiling.DeepProfiler` instance.

        Raises:
            RuntimeError: If pyinstrument is not installed.
        """
        from llm_rosetta.profiling import DeepProfiler

        return DeepProfiler(async_mode=True)

    def store_result(
        self,
        profiler: Any,
        *,
        request_id: str = "",
        model: str = "",
        source: str = "",
        target: str = "",
        is_stream: bool = False,
        duration_ms: float = 0.0,
    ) -> None:
        """Store profiling result from a completed request.

        Args:
            profiler: A stopped DeepProfiler instance.
            request_id: The request's trace ID.
            model: Model name.
            source: Source provider.
            target: Target provider.
            is_stream: Whether the request was streaming.
            duration_ms: End-to-end request duration.
        """
        result: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "request_id": request_id,
            "model": model,
            "source": source,
            "target": target,
            "is_stream": is_stream,
            "duration_ms": round(duration_ms, 2),
            "html": profiler.output_html(),
            "text": profiler.output_text(),
        }
        self.results.append(result)
        # Trim to max
        if len(self.results) > self._max_results:
            self.results = self.results[-self._max_results :]

    def status(self) -> dict[str, Any]:
        """Return current profiling status."""
        return {
            "enabled": self.enabled,
            "remaining": self.remaining,
            "results_count": len(self.results),
            "max_results": self._max_results,
        }

    def clear_results(self) -> None:
        """Remove all stored profiling results."""
        self.results.clear()
