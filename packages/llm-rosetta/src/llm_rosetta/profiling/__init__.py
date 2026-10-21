"""On-demand deep profiling for llm-rosetta conversions.

:class:`DeepProfiler` is the cProfile-backed wrapper.  The gateway's
on/off profiler state is a separate concern and lives in
:mod:`llm_rosetta.observability.profiling`.

``Profiler`` and ``TracingProfiler`` stay importable from here because the old
flat module bound them at module scope; they are not part of the declared
surface (``__all__``), so ``import *`` still yields only ``DeepProfiler``.
"""

from __future__ import annotations

# Re-exported from the old flat module's module scope; not declared API.
from llm_rosetta._vendor.profiler import (  # noqa: F401
    Profiler,
    TracingProfiler,
)

from .deep import DeepProfiler

__all__ = ["DeepProfiler"]
