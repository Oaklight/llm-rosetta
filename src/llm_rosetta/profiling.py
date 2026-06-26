"""On-demand deep profiling for llm-rosetta conversions.

Provides :class:`DeepProfiler`, a thin wrapper around **pyinstrument**
that supports both sync and async context managers.  ``pyinstrument``
is lazy-imported at :meth:`start` / ``__enter__`` time — the class
itself is always importable even without ``pyinstrument`` installed.

Install the optional dependency::

    pip install llm-rosetta[profiling]

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

Note:
    Under concurrent async workloads, flamegraphs may include minor
    noise from other coroutines on the same event loop.  Results are
    best-effort; per-instance isolation minimizes contamination.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


class DeepProfiler:
    """Thin wrapper around pyinstrument for on-demand profiling.

    Args:
        async_mode: If ``True`` (default), create the underlying
            ``Profiler`` with ``async_mode="enabled"`` so that
            ``await`` boundaries are correctly attributed.  Set to
            ``False`` for purely synchronous workloads.
    """

    def __init__(self, *, async_mode: bool = True) -> None:
        self._async_mode = async_mode
        self._profiler: Any = None  # lazy — set in start()

    # ------------------------------------------------------------------
    # Lazy import helper
    # ------------------------------------------------------------------

    @staticmethod
    def _import_pyinstrument() -> Any:
        """Import and return the ``pyinstrument`` module.

        Raises:
            RuntimeError: If ``pyinstrument`` is not installed.
        """
        try:
            import pyinstrument

            return pyinstrument
        except ImportError:
            raise RuntimeError(
                "pyinstrument is not installed. "
                "Install with: pip install llm-rosetta[profiling]"
            ) from None

    # ------------------------------------------------------------------
    # Explicit start / stop API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start profiling.

        Creates a fresh ``pyinstrument.Profiler`` instance (one per
        call, never shared across requests).

        Raises:
            RuntimeError: If ``pyinstrument`` is not installed, or if
                the profiler is already running.
        """
        if self._profiler is not None:
            raise RuntimeError("Profiler is already running")
        pyinstrument = self._import_pyinstrument()
        kwargs: dict[str, Any] = {}
        if self._async_mode:
            kwargs["async_mode"] = "enabled"
        self._profiler = pyinstrument.Profiler(**kwargs)
        self._profiler.start()

    def stop(self) -> None:
        """Stop profiling.

        Raises:
            RuntimeError: If the profiler is not running.
        """
        if self._profiler is None:
            raise RuntimeError("Profiler is not running")
        self._profiler.stop()

    @property
    def is_running(self) -> bool:
        """Whether the profiler is currently active."""
        return self._profiler is not None and self._profiler.is_running

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
        """Return profiling results as a human-readable text tree.

        Raises:
            RuntimeError: If the profiler was never started or is
                still running.
        """
        self._check_stopped()
        return self._profiler.output_text(**kwargs)

    def output_html(self, **kwargs: Any) -> str:
        """Return profiling results as an interactive HTML flamegraph.

        Raises:
            RuntimeError: If the profiler was never started or is
                still running.
        """
        self._check_stopped()
        return self._profiler.output_html(**kwargs)

    def save_html(self, path: str | Path, **kwargs: Any) -> None:
        """Write the HTML flamegraph to a file.

        Args:
            path: Destination file path.

        Raises:
            RuntimeError: If the profiler was never started or is
                still running.
        """
        html = self.output_html(**kwargs)
        Path(path).write_text(html, encoding="utf-8")

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _check_stopped(self) -> None:
        """Ensure the profiler has been started and stopped."""
        if self._profiler is None:
            raise RuntimeError("Profiler was never started")
        if self._profiler.is_running:
            raise RuntimeError("Profiler is still running — call stop() first")
