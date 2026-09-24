"""On-demand deep profiling for llm-rosetta conversions.

Provides :class:`DeepProfiler`, a thin wrapper around the vendored
zerodep ``Profiler`` (cProfile-based) that supports both sync and
async context managers.  Always available — no optional dependencies.

Usage (sync)::

    from llm_rosetta.profiling import DeepProfiler

    with DeepProfiler() as dp:
        result = convert(body, "anthropic")
    print(dp.output_text())

Usage (async)::

    async with DeepProfiler() as dp:
        target = pipeline.convert_request(body)
        resp = await transport.send(target)
    dp.save_html("profile.html")

Per-call tracing (higher overhead, finer granularity)::

    with DeepProfiler(tracing=True) as dp:
        result = convert(body, "anthropic")
    records = dp.traces()  # list of TraceRecord
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from llm_rosetta._vendor.profiler import Profiler, TracingProfiler

__all__ = ["DeepProfiler"]


class DeepProfiler:
    """Thin wrapper around the vendored zerodep Profiler.

    Args:
        async_mode: If ``True`` (default), create the underlying
            profiler with ``async_mode=True`` so it can be used
            as an async context manager.  Set to ``False`` for purely
            synchronous workloads.
        tracing: If ``True``, use ``TracingProfiler`` (per-call tracing
            via ``sys.monitoring`` / ``sys.settrace``) instead of the
            default cProfile-based ``Profiler``.  Higher overhead but
            records every function call with nanosecond timing.
    """

    def __init__(self, *, async_mode: bool = True, tracing: bool = False) -> None:
        self._async_mode = async_mode
        self._tracing = tracing
        self._started = False
        self._tracing_profiler: TracingProfiler | None = None
        if tracing:
            self._tracing_profiler = TracingProfiler(async_mode=async_mode)
            self._profiler: Profiler | TracingProfiler = self._tracing_profiler
        else:
            self._profiler = Profiler(async_mode=async_mode)

    # ------------------------------------------------------------------
    # Explicit start / stop API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start profiling.

        Raises:
            RuntimeError: If the profiler is already running.
        """
        if self._profiler.is_running:
            raise RuntimeError("Profiler is already running")
        self._profiler.start()
        self._started = True

    def stop(self) -> None:
        """Stop profiling.

        Raises:
            RuntimeError: If the profiler is not running.
        """
        if not self._profiler.is_running:
            raise RuntimeError("Profiler is not running")
        self._profiler.stop()

    @property
    def is_running(self) -> bool:
        """Whether the profiler is currently active."""
        return self._profiler.is_running

    @property
    def is_tracing(self) -> bool:
        """Whether this instance uses per-call tracing."""
        return self._tracing

    # ------------------------------------------------------------------
    # Sync context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> DeepProfiler:
        self.start()
        return self

    def __exit__(self, *exc_info: Any) -> None:
        if self.is_running:
            self.stop()

    # ------------------------------------------------------------------
    # Async context manager
    # ------------------------------------------------------------------

    async def __aenter__(self) -> DeepProfiler:
        self.start()
        return self

    async def __aexit__(self, *exc_info: Any) -> None:
        if self.is_running:
            self.stop()

    # ------------------------------------------------------------------
    # Output methods
    # ------------------------------------------------------------------

    def output_text(self, **kwargs: Any) -> str:
        """Return profiling results as formatted text.

        Raises:
            RuntimeError: If the profiler was never started or is
                still running.
        """
        self._check_stopped()
        return self._profiler.output_text(**kwargs)

    def output_html(self, **kwargs: Any) -> str:
        """Return profiling results as a self-contained HTML report.

        Supports ``style`` kwarg (``"table"``, ``"flamegraph"``,
        ``"icicle"``).

        Raises:
            RuntimeError: If the profiler was never started or is
                still running.
        """
        self._check_stopped()
        return self._profiler.output_html(**kwargs)

    def save_html(self, path: str | Path, **kwargs: Any) -> None:
        """Write the HTML report to a file.

        Args:
            path: Destination file path.

        Raises:
            RuntimeError: If the profiler was never started or is
                still running.
        """
        html = self.output_html(**kwargs)
        Path(path).write_text(html, encoding="utf-8")

    def traces(self) -> list:
        """Return raw per-call trace records (tracing mode only).

        Returns:
            List of ``TraceRecord`` named tuples.

        Raises:
            RuntimeError: If not in tracing mode, or if the profiler
                was never started or is still running.
        """
        if not self._tracing or self._tracing_profiler is None:
            raise RuntimeError("traces() requires tracing=True")
        self._check_stopped()
        return self._tracing_profiler.traces()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _check_stopped(self) -> None:
        """Ensure the profiler has been started and stopped."""
        if self._profiler.is_running:
            raise RuntimeError("Profiler is still running — call stop() first")
        if not self._started:
            raise RuntimeError("Profiler was never started")
